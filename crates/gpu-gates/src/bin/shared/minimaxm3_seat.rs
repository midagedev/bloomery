//! The seat the MiniMax-M3 end-to-end gate drives the body through (`gate_minimaxm3_e2e`): the load of the body on the gate card
//! and the [`Seat`] calls over it ([`crate::lcpp_e2e`] holds the trait and its contract), the tap ceilings of the model's layers
//! and the gate plan's printed terms (`--plan`). A gate includes it by `#[path]` beside `shared/lcpp_e2e.rs` and
//! `shared/gate_card.rs`.
//!
//! The prompt call and the stores are the body's own ([`bloomery_gpu_minimaxm3`]): [`Seat::prompt`] runs `ids` from the seat's
//! position by the prompt batch (up to `bloomery_gpu_minimaxm3::prefill::T_MAX` rows a unit); [`Seat::stores`] names each
//! layer's K and V planes and each sparse layer's index-key plane over the written positions.
//!
//! The model is a GGUF split or, when the path is a directory, a checkpoint folder read as stored; the open is the serve seat's
//! own (`serve_seats/minimaxm3_folder.rs`), and a build without the `checkpoint` feature refuses a folder by name.

use crate::lcpp_e2e::{Arm, Dims, ForcedRoute, ForcedRow, Seat, fnv1a};
use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
use bloomery_gpu::host::served::TierBody;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::xstream::XMode;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::prompts::SIGMA_LOGIT_V41;
use bloomery_gpu_gates::tier::Tier;
use bloomery_gpu_minimaxm3::prefill::{prefill, set_prefill_group};
use bloomery_gpu_minimaxm3::{Body, MiniMaxM3Model, set_taps};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::minimaxm3::hparams::FfnKind;
use model::arch::minimaxm3::place::PlanInputs;
use model::placement::workstation::{self, HostRead, PLAN_ROOM};
use model::placement::{Machine, PlanLevers};
use refset::arch::minimaxm3;
use std::num::NonZeroUsize;
use std::path::Path;

use super::xstream_rule::{Stage, plans_ring};

/// The seats' one owner of a checkpoint folder's open, which the MiniMax folder's open sits on.
#[path = "serve_seats/folder.rs"]
mod folder;

/// The model at a path: the one owner the serve seat opens a MiniMax folder through.
#[path = "serve_seats/minimaxm3_folder.rs"]
mod minimaxm3_folder;

/// The dense lead (`leading_dense_block_count`): the layers before the first sparse one.
pub const TAP_EARLY: usize = 3;

/// The fixture cut's dense lead: the cut keeps the folder's layers 0, 3 and 59, so its layer 0 is the one dense layer.
pub const FIXTURE_EARLY: usize = 1;

/// The two ceilings of a layer output's relative distance between two implementations of the model: `TAP_EARLY_CEIL` over the
/// first [`TAP_EARLY`] layers, `TAP_CEIL` after.
// PIN(2026-10-21): provisional, the ceilings of the other llama.cpp-oracle gate (Solar-Open2's, two implementations of the
// quantized model); re-pinned from the first run's printed distances, never tuned to pass.
pub const TAP_EARLY_CEIL: f64 = 0.05;
pub const TAP_CEIL: f64 = 0.3;

/// The standard deviation, in logit units, of one id's logit between two implementations of the model: V4.1's, until
/// MiniMax-M3.1's own is measured. The gate's description reads it (`Gate::sigma_logit`); the clauses that use it are deferred
/// by name for this family.
pub const SIGMA_LOGIT: f64 = SIGMA_LOGIT_V41;

/// The teacher-forced arm's own per-layer error: the free chain's layers before the prefix its
/// routing holds are held to `√prefix` times it, and the layers from the prefix on to nothing (a
/// flip is a discrete exchange of experts, not a rounding of the activations).
///
/// PIN(2026-10-21): the forced arm's own worst per-layer error over the layers the prefix covers
/// (the prefix is layer 3 in every measured set: the dense lead and the first sparse layer), as
/// the arm measures it: 2.496e-3 at layer 1 of set a, 8.687e-4 at layer 2.
pub const FORCED_ERR: f64 = 2.5e-3;

/// The free chain's ceiling of a layer's output distance, over the layers before the prefix the
/// run's routing holds: `√prefix` times the forced arm's own per-layer error.
pub fn free_ceil(prefix: usize, _layer: usize) -> f64 {
    (prefix as f64).sqrt() * FORCED_ERR
}

/// The layers the forced arm's prefix covers: the dense lead and the first sparse layer. The
/// prefix is layer 3 in every measured set, so the layers from it on may carry an exchange of
/// experts (a flip of the top 4 at a near tie) and are held to nothing.
pub const FORCED_PREFIX: usize = 3;

/// The forced arm's prefix in `tier`'s model: [`FORCED_PREFIX`] in the real folder, the fixture cut's dense lead in the cut.
// PIN(2026-10-23): the fixture cut's prefix is its dense lead (layer 0), the layers before its first sparse layer, where no
// exchange of experts can happen.
fn prefix(tier: Tier) -> usize {
    match tier {
        Tier::Real => FORCED_PREFIX,
        Tier::Fixture => FIXTURE_EARLY,
    }
}

/// The prefill gate's ceiling of a layer's output distance: the forced arm's own per-layer error
/// over the layers its prefix covers, and none over the layers from it on.
pub fn prefill_ceil(tier: Tier, layer: usize) -> f64 {
    let prefix = prefix(tier);
    if layer < prefix {
        free_ceil(prefix, layer)
    } else {
        f64::INFINITY
    }
}

/// The ceilings of [`prefill_ceil`] as a line prints them.
pub fn prefill_ceils(tier: Tier) -> String {
    let prefix = prefix(tier);
    format!(
        "the forced arm's {:.2e} over the layers before {prefix} and none over the layers from it on",
        free_ceil(prefix, 0)
    )
}

/// The dense lead of `tier`'s model: [`TAP_EARLY`] layers in the real folder, [`FIXTURE_EARLY`] in the fixture cut.
fn early(tier: Tier) -> usize {
    match tier {
        Tier::Real => TAP_EARLY,
        Tier::Fixture => FIXTURE_EARLY,
    }
}

/// The ceiling of layer `layer`'s output distance: [`TAP_EARLY_CEIL`] over the dense lead of `tier`'s model, [`TAP_CEIL`]
/// after.
// PIN(2026-10-23): the fixture cut holds its layers to the real tier's two ceilings over its own dense lead (layer 0): the
// same layer kinds, so the same provisional bands until a fixture run's printed distances re-pin them.
pub fn tap_ceil(tier: Tier, layer: usize) -> f64 {
    if layer < early(tier) {
        TAP_EARLY_CEIL
    } else {
        TAP_CEIL
    }
}

/// The ceilings of [`tap_ceil`] as a gate's line prints them.
pub fn tap_ceils(tier: Tier) -> String {
    format!(
        "ceilings {TAP_EARLY_CEIL} (layers < {}) and {TAP_CEIL}",
        early(tier)
    )
}

/// The plan's inputs of `split` as the gate holds them: the file as it is, the host room the workstation's plan room.
///
/// The gates feed ids, so the features of the file that only the tokenizer and the tool-call parser carry
/// (`PlanInputs::read` lists them as unimplemented) are not the gate's. The plan this gate holds its pins to is the unpaged one
/// at the workstation's plan room: no reading of the box's available bytes enters it.
fn described(split: &Split) -> Result<PlanInputs, GateError> {
    let mut inputs = PlanInputs::describe(split)?;
    inputs.room = (PLAN_ROOM, HostRead::Given);
    Ok(inputs)
}

/// The text of `ids` by the tokenizer of the checkpoint folder `dir` (`model_hf::folder::open`, checked against Hugging Face's).
pub fn decode(dir: &Path, ids: &[u32]) -> Result<String, GateError> {
    Ok(minimaxm3_folder::open(dir)?.tokenizer.decode(ids))
}

/// Open `file` at `ctx` positions by the placement of `arm`: the gate placement (`gate_card::plan_gate`) on the gate card for
/// `a`, plan (b′) (`workstation::plan_bp`) for `bp`, the next-largest card an expert tier under the host tier holding the
/// tier's prompt-batch bytes (`bloomery_gpu_minimaxm3::tier::tier_batch`). The body is opened as the serve seat opens
/// it (`Body::open_placed`, the plan `PlanInputs::plan` makes), the taps armed before the step is first captured, so a
/// replay writes them.
///
/// The including gate declares `shared/gate_card.rs` as `gate_card` beside this module.
pub fn open(file: &Path, ctx: usize, arm: Arm) -> Result<Box<dyn Seat>, GateError> {
    Ok(Box::new(open_loaded(file, ctx, arm)?))
}

/// [`open`]'s seat as the body behind it, for a gate that reads the body between the seat's calls.
pub fn open_loaded(file: &Path, ctx: usize, arm: Arm) -> Result<Loaded, GateError> {
    load(file, ctx, false, arm)
}

/// The body loaded by [`open`]'s plan of `arm`; `route_trace` makes the load one the host tier can write a route trace for
/// ([`route_trace`]). `BLOOMERY_XSTREAM`, when set, is the body's host streaming from the load on (`Body::set_xstream`);
/// where it starts the stream (`split`, [`plans_ring`] at a gate's stage) the plan keeps the stream's room
/// (`PlanInputs::with_stream`) and its lane staging on the host (`workstation::reserve_stream_staging`).
fn load(file: &Path, ctx: usize, route_trace: bool, arm: Arm) -> Result<Loaded, GateError> {
    let levers =
        bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET, bloomery_levers::XSTREAM])?;
    let mode = levers.xstream().map(XMode::parse).transpose()?;
    let model = minimaxm3_folder::model_at(file)?;
    let inputs = described(model.header())?;
    let inputs = if plans_ring(levers.xstream(), Stage::Other, false) {
        inputs.with_stream()?
    } else {
        inputs
    };
    let mut machine = arm.machine(
        inputs.model.layers,
        bloomery_gpu_minimaxm3::tier::tier_batch(&inputs)?,
    )?;
    workstation::reserve_stream_staging(&mut machine, inputs.stream_slot);
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let host = HostCfg {
        route_trace,
        ..levers.host()
    };
    let mut loaded = seated(Body::open_placed(model, &plan, &inputs, 0, host)?)?;
    if let Some(mode) = mode {
        let (gpu, _, body) = loaded.m.body_parts("minimaxm3 seat")?;
        body.set_xstream(gpu, mode)?;
    }
    Ok(loaded)
}

/// `m` as the seat drives it: the taps armed under eager steps, then captured with them.
fn seated(mut m: MiniMaxM3Model) -> Result<Loaded, GateError> {
    set_taps(&mut m, true)?;
    m.set_mode(StepMode::Graph);
    let (layers, vocab, embd) = {
        let (_, _, b) = m.body_parts("minimaxm3 seat")?;
        let d = b.dims();
        (b.kinds().len(), d.vocab, d.embd)
    };
    Ok(Loaded {
        m,
        dims: Dims {
            layers,
            embd,
            vocab,
        },
    })
}

/// The card budget the two-card arm plans both its loads under in the real tier: the stage card's dense terms (the MXFP8
/// attention, the shared experts, the dense lead, the BF16 head) and the experts beside them, and at most this much of the
/// tier's experts, so the reference's union holds at most twice it of the A6000's bytes [derived from the placement's rule].
pub const TWO_BUDGET: u64 = 20 << 30;

/// What the two-card arm (`gate_minimaxm3_prefill --arm bp`) loaded: the reference's walk and the plan (b′) load beside the
/// plan's tier experts by layer.
pub struct TwoCard {
    /// The ids the walk fed, one decode step each.
    pub walk: Vec<u32>,
    /// The plan (b′) load, as [`open`] leaves one.
    pub loaded: Loaded,
    /// The plan's experts on the tier card by layer.
    pub planned: Vec<u64>,
    /// The reference's ids and logits rows over the walk, one decode step an id in graph mode.
    pub reference: (Vec<u32>, Vec<Vec<f32>>),
    /// When the plan (b′) load began.
    pub opened: std::time::Instant,
}

/// The steps' token walk: `walk(steps, vocab)` gives the ids the reference decodes.
pub type Walk = fn(usize, usize) -> Vec<u32>;

/// The two-card arm's loads at `ctx` positions, one after the other: plan (b′) (`workstation::plan_bp`, the stage the A6000,
/// the tier the 3090) under one card budget no environment moves ([`TWO_BUDGET`]); the reference, that plan's union on the A6000 alone, its ids and logits rows read over the walk
/// `walk(steps, vocab)` and released; then plan (b′) itself. A budget the environment sets, a union past the A6000's usable
/// bytes, a reference opened elsewhere or holding a tier, and the fixture tier (its cut's routed layers fit the stage card,
/// so the tier would hold nothing) are refused by name.
pub fn two_card(
    file: &Path,
    (ctx, tier): (usize, Tier),
    (steps, walk): (usize, Walk),
) -> Result<TwoCard, GateError> {
    if tier == Tier::Fixture {
        return Err("the two-card arm runs in the real tier: the fixture cut's routed layers fit the stage card".into());
    }
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    if PlanLevers::from_levers(&levers)?.card_budget_bytes.is_set() {
        return Err(
            "the two-card arm plans under its own card budget: unset BLOOMERY_CARD_BUDGET".into(),
        );
    }
    let inputs = described(minimaxm3_folder::model_at(file)?.header())?;
    let layers = inputs.model.layers;
    let walk = walk(steps, inputs.hp.n_vocab);
    let two = Arm::Bp.machine(layers, bloomery_gpu_minimaxm3::tier::tier_batch(&inputs)?)?;
    let flat = Machine {
        tiers: Vec::new(),
        ..two.clone()
    };
    let ctx = u64::try_from(ctx)?;
    let budget = PlanLevers {
        card_budget_bytes: Some(TWO_BUDGET).into(),
    };
    let plan = inputs.plan(&two, ctx, &budget)?;
    let on_tier = plan.tier_n_l.first().cloned().unwrap_or_default();
    let planned: Vec<u64> = (0..layers)
        .map(|l| on_tier.get(l).copied().unwrap_or(0))
        .collect();
    let stage = &plan.cards[0];
    let tier_bytes = plan.cards.get(1).map_or(0, |c| c.expert_bytes);
    let union_bytes = stage.dense_bytes + stage.expert_bytes + stage.kv_bytes + tier_bytes;
    let usable = workstation::A6000.usable_bytes();
    println!(
        "arm bp plan (b′) under a card budget of {} at ctx {ctx}: stage {} experts ({} B), tier {} experts ({tier_bytes} \
         B) on {} layers, host {} experts ({} B); the reference's union {union_bytes} B of the A6000's {usable} (plan values)",
        model::placement::card_budget::in_bytes(&budget.card_budget_bytes),
        stage.experts,
        stage.expert_bytes,
        plan.cards.get(1).map_or(0, |c| c.experts),
        planned.iter().filter(|&&n| n > 0).count(),
        plan.host.experts,
        plan.host.expert_bytes,
    );
    if union_bytes > usable {
        return Err(format!(
            "the reference's union of {union_bytes} B passes the A6000's {usable} B: lower the two-card budget"
        )
        .into());
    }
    let host = levers.host();
    let reference = {
        let uplan = model::placement::union_plan(&plan, &flat)?;
        let mut r = Body::open_placed(minimaxm3_folder::model_at(file)?, &uplan, &inputs, 0, host)?;
        r.set_mode(StepMode::Graph);
        if !r.body("minimaxm3 seat")?.hybrid().tiers().is_empty() {
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
        let (mut ids, mut rows) = (
            Vec::with_capacity(walk.len()),
            Vec::with_capacity(walk.len()),
        );
        for &id in &walk {
            ids.push(r.step(&[id])?);
            rows.push(r.logits()?);
        }
        (ids, rows)
    };
    let opened = std::time::Instant::now();
    let loaded = seated(Body::open_placed(
        minimaxm3_folder::model_at(file)?,
        &plan,
        &inputs,
        0,
        host,
    )?)?;
    Ok(TwoCard {
        walk,
        loaded,
        planned,
        reference,
        opened,
    })
}

/// Load the file as [`open`] does with the host tier writing a route trace into the new directory `dir`, feed `ids` one step each
/// from the first position, and finish the trace: every sparse layer's routed ids a position, in router rank order, and the slot
/// kind each ran in, as a router set (`crates/gpu/src/host/route_trace.rs`) beside the reference set's `ffn_moe_topk-*` for the
/// first layer and position where the two disagree. Returns the positions on disk.
pub fn route_trace(file: &Path, ctx: usize, dir: &Path, ids: &[u32]) -> Result<u64, GateError> {
    let mut seat = load(file, ctx, true, Arm::A)?;
    let inputs = described(minimaxm3_folder::model_at(file)?.header())?;
    let hp = &inputs.hp;
    let (n_expert, n_used, layers) = (hp.experts.n_expert, hp.experts.n_used, hp.n_layer);
    let first = (0..layers)
        .find(|&l| hp.kind(l).ffn == FfnKind::Routed)
        .ok_or("the file holds no sparse layer")?;
    let header = TraceHeader {
        model: file.to_path_buf(),
        arch: minimaxm3::ARCH.to_owned(),
        build: format!("bloomery-gpu-gates {}", env!("CARGO_PKG_VERSION")),
        n_expert,
        n_used,
        first_layer: first,
        n_layer: layers - first,
        extra: vec![("placement".to_owned(), "gate".to_owned())],
    };
    let mut trace = RouteTrace::create(dir, header)?;
    trace.prompt(0, ids.len())?;
    seat.m
        .body_parts("minimaxm3 route trace")?
        .2
        .hybrid_mut()
        .attach_route_trace(trace)?;
    for &id in ids {
        seat.step(id)?;
    }
    let trace = seat
        .m
        .body_parts("minimaxm3 route trace")?
        .2
        .hybrid_mut()
        .take_route_trace()
        .ok_or("the route trace was detached under the run")?;
    Ok(trace.finish()?)
}

/// The body on the gate card behind the seat's calls.
pub struct Loaded {
    m: MiniMaxM3Model,
    dims: Dims,
}

impl Loaded {
    /// The body behind the seat's calls.
    pub fn model(&self) -> &MiniMaxM3Model {
        &self.m
    }

    /// The body behind the seat's calls, to step it outside them.
    pub fn model_mut(&mut self) -> &mut MiniMaxM3Model {
        &mut self.m
    }

    /// The last call's logits row.
    fn row(&self) -> Result<Vec<f32>, GateError> {
        let mut row = vec![0.0; self.dims.vocab];
        self.m.logits_into(&mut row)?;
        Ok(row)
    }
}

impl Seat for Loaded {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    fn pos(&self) -> u32 {
        self.m.pos()
    }

    fn step(&mut self, id: u32) -> Result<Vec<f32>, GateError> {
        self.m.step(&[id])?;
        self.row()
    }

    fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError> {
        prompt_pass(&mut self.m, |m| {
            m.run_rows(ids.len(), "minimaxm3 prompt", |gpu, w, body, head, pos| {
                prefill(gpu, w, body, head, ids, pos)
            })
        })
        .map_err(|e| e.to_string())?;
        self.row()
    }

    fn set_group(&mut self, group: usize) -> Result<(), GateError> {
        set_prefill_group(&mut self.m, group).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn taps(&mut self) -> Result<Vec<f32>, GateError> {
        let (gpu, _, body) = self.m.body_parts("minimaxm3 seat")?;
        Ok(body.taps(gpu)?)
    }

    fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError> {
        let pos = self.m.pos();
        let (gpu, _, body) = self.m.body_parts("minimaxm3 seat")?;
        Ok(body
            .store_bytes_at(gpu, pos)?
            .into_iter()
            .map(|(name, bytes)| (name, fnv1a(&bytes)))
            .collect())
    }

    fn fault(&mut self) -> Result<(), GateError> {
        if let Some(fault) = self.m.poisoned() {
            return Err(format!("the model is poisoned by {fault:?}").into());
        }
        let (gpu, _, _) = self.m.body_parts("minimaxm3 seat")?;
        match gpu.fault()? {
            None => Ok(()),
            Some(fault) => Err(format!("the card's fault word holds {fault:?}").into()),
        }
    }

    fn forced_row(
        &mut self,
        layer: usize,
        pos: u32,
        x: &[f32],
        ffn_x: Option<&[f32]>,
    ) -> Result<ForcedRow, GateError> {
        let (gpu, w, body) = self.m.body_parts("minimaxm3 seat")?;
        let row = body.forced_row(gpu, w, layer, pos, x, ffn_x)?;
        Ok(ForcedRow {
            ffn_in: row.ffn_in,
            ffn_out: row.ffn_out,
            route: row.route.map(|r| ForcedRoute {
                ids: r.ids,
                bias: r.bias,
            }),
            out: row.out,
        })
    }

    fn route_ids(&mut self) -> Result<Vec<Vec<u32>>, GateError> {
        let (_, _, body) = self.m.body_parts("minimaxm3 seat")?;
        Ok(body.hybrid().step_ids().to_vec())
    }

    fn head_row(&mut self, x: &[f32]) -> Result<Vec<f32>, GateError> {
        self.m.reset()?;
        self.m
            .run_rows(1, "minimaxm3 head probe", |gpu, w, _body, head, _pos| {
                head.set_input(gpu, x)?;
                head.enqueue(gpu, w)?;
                Ok(true)
            })?;
        let row = self.row()?;
        self.m.reset()?;
        Ok(row)
    }
}

// ------------------------------------------------------------ the plan, host only

/// `bytes` with its GiB beside it.
fn gib(bytes: u64) -> String {
    format!("{bytes} B ({:.2} GiB)", bytes as f64 / (1u64 << 30) as f64)
}

/// The gate plan of the file at `ctx` positions, planned exactly as [`open`] plans it on the 3090's bytes
/// (`workstation::plan_gate`, which `gate_card::plan_gate` fixes the census name onto), and printed as lines: no card is opened,
/// no census read, no weight loaded. One line a run of sparse layers that place the same experts on the card, the dense lead's
/// line, the card's bytes by term, the host's by term. The numbers are the plan's own ([derived]), not a measurement.
pub fn plan_lines(file: &Path, ctx: usize) -> Result<Vec<String>, GateError> {
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    let inputs = described(minimaxm3_folder::model_at(file)?.header())?;
    let layers = inputs.model.layers;
    let machine = workstation::plan_gate(layers);
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let card = machine
        .cards
        .first()
        .ok_or("the gate machine holds no card")?;
    let totals = plan.cards.first().ok_or("the plan holds no card")?;
    let mut lines = vec![format!(
        "plan: {layers} layers, ctx {ctx}, card {} (usable {}), host room {}",
        card.name,
        gib(card.usable_bytes),
        gib(PLAN_ROOM)
    )];
    let (mut card_experts, mut host_experts) = (0u64, 0u64);
    let mut l = 0;
    while l < layers {
        let sparse = inputs.hp.kind(l).ffn == FfnKind::Routed;
        let on_card = plan.n_l.get(l).copied().unwrap_or(0);
        let total = inputs.model.layer_experts(l);
        let mut end = l + 1;
        while end < layers
            && (inputs.hp.kind(end).ffn == FfnKind::Routed) == sparse
            && plan.n_l.get(end).copied().unwrap_or(0) == on_card
            && inputs.model.layer_experts(end) == total
        {
            end += 1;
        }
        if sparse {
            let run = (end - l) as u64;
            card_experts += run * on_card;
            host_experts += run * (total - on_card);
            lines.push(format!(
                "plan: sparse layers {l}..={} ({} layers): {on_card} card experts and {} host experts of {total} a layer",
                end - 1,
                end - l,
                total - on_card
            ));
        } else {
            lines.push(format!(
                "plan: dense layers {l}..={} ({} layers): no routed experts",
                end - 1,
                end - l
            ));
        }
        l = end;
    }
    lines.push(format!(
        "plan: routed experts in all: {card_experts} on the card, {host_experts} on the host"
    ));
    // The loader's quantization of a tensor whose file type the card holds otherwise (a BF16 head in MXFP8 planes).
    for t in inputs.model.requantized() {
        lines.push(format!(
            "plan: requantized on the card (tensor:file type:card format): {t}"
        ));
    }
    lines.push(format!(
        "plan: card bytes: dense {}, experts {}, granules {}, kv {}, stage rows {}, scratch {}, context {}, reserves {}, headroom {} B; \
         planned {} of usable {}",
        gib(totals.dense_bytes),
        gib(totals.expert_bytes),
        gib(totals.rounding_bytes),
        gib(totals.kv_bytes),
        gib(totals.stage_rows_bytes),
        gib(totals.scratch_bytes),
        gib(totals.context_bytes),
        gib(totals.reserve_bytes),
        totals.headroom_bytes,
        gib(totals.bytes()),
        gib(card.usable_bytes)
    ));
    let host = &plan.host;
    lines.push(format!(
        "plan: host bytes: experts {} ({} experts), nvme experts {}, tables {}, shadows {}, reserves {}",
        gib(host.expert_bytes),
        host.experts,
        gib(host.nvme_expert_bytes),
        gib(host.table_bytes),
        gib(host.shadow_bytes),
        gib(host.reserve_bytes)
    ));
    Ok(lines)
}

/// The body asked for two resident sequences at the gate plan of `file`: refused by name before any card is opened, and
/// the refusal's text answered. An open that is not refused, or refused by another message, is an error.
pub fn refuse_second_slot(file: &Path, ctx: usize) -> Result<String, GateError> {
    const WANT: &str = "2 resident sequences";
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    let model = minimaxm3_folder::model_at(file)?;
    let inputs = described(model.header())?;
    let machine = workstation::plan_gate(inputs.model.layers);
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let two = NonZeroUsize::new(2).ok_or("two is not zero")?;
    match Body::open_placed_slots(model, &plan, &inputs, 0, levers.host(), Residency::Off, two) {
        Ok(_) => Err("the body opened with two resident sequences".into()),
        Err(e) if e.to_string().contains(WANT) => Ok(e.to_string()),
        Err(e) => Err(format!("the refusal does not name {WANT:?}: {e}").into()),
    }
}
