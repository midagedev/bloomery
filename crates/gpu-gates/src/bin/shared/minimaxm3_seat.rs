//! The seat the MiniMax-M3 end-to-end gate drives the body through (`gate_minimaxm3_e2e`): the load of the body on the gate card
//! and the [`Seat`] calls over it ([`crate::lcpp_e2e`] holds the trait and its contract), the tap ceilings of the model's layers
//! and the gate plan's printed terms (`--plan`). A gate includes it by `#[path]` beside `shared/lcpp_e2e.rs` and
//! `shared/gate_card.rs`.
//!
//! The prompt call and the stores are the body's own ([`bloomery_gpu_minimaxm3`]): [`Seat::prompt`] runs `ids` from the seat's
//! position by the prompt batch (up to `bloomery_gpu_minimaxm3::prefill::T_MAX` rows a unit); [`Seat::stores`] names each
//! layer's K and V planes and each sparse layer's index-key plane over the written positions.

use crate::lcpp_e2e::{Arm, Dims, ForcedRoute, ForcedRow, Seat, fnv1a};
use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
use bloomery_gpu::host::served::TierBody;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::tier::Tier;
use bloomery_gpu_minimaxm3::prefill::{prefill, set_prefill_group};
use bloomery_gpu_minimaxm3::{Body, MiniMaxM3Model, set_taps};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::minimaxm3::hparams::Kind;
use model::arch::minimaxm3::place::PlanInputs;
use model::placement::PlanLevers;
use model::placement::workstation::{self, HostRead, PLAN_ROOM};
use refset::arch::minimaxm3;
use std::num::NonZeroUsize;
use std::path::Path;

/// The dense lead (`leading_dense_block_count`): the layers before the first sparse one.
pub const TAP_EARLY: usize = 3;

/// The two ceilings of a layer output's relative distance between two implementations of the model: `TAP_EARLY_CEIL` over the
/// first [`TAP_EARLY`] layers, `TAP_CEIL` after.
// PIN(2026-10-21): provisional, the ceilings of the other llama.cpp-oracle gate (Solar-Open2's, two implementations of the
// quantized model); re-pinned from the first run's printed distances, never tuned to pass.
pub const TAP_EARLY_CEIL: f64 = 0.05;
pub const TAP_CEIL: f64 = 0.3;

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

/// The prefill gate's ceiling of a layer's output distance: the forced arm's own per-layer error
/// over the layers its prefix covers, and none over the layers from it on.
pub fn prefill_ceil(_tier: Tier, layer: usize) -> f64 {
    if layer < FORCED_PREFIX {
        free_ceil(FORCED_PREFIX, layer)
    } else {
        f64::INFINITY
    }
}

/// The ceilings of [`prefill_ceil`] as a line prints them.
pub fn prefill_ceils(_tier: Tier) -> String {
    format!(
        "the forced arm's {:.2e} over the layers before {FORCED_PREFIX} and none over the layers from it on",
        free_ceil(FORCED_PREFIX, 0)
    )
}

/// The ceiling of layer `layer`'s output distance. The model has no fixture file, so only the real tier reaches it.
///
/// # Panics
/// On the fixture tier, which the gate refuses by name before any load.
pub fn tap_ceil(tier: Tier, layer: usize) -> f64 {
    match tier {
        Tier::Real if layer < TAP_EARLY => TAP_EARLY_CEIL,
        Tier::Real => TAP_CEIL,
        Tier::Fixture => unreachable!("the minimaxm3 gate has no fixture tier"),
    }
}

/// The ceilings of [`tap_ceil`] as a gate's line prints them.
///
/// # Panics
/// On the fixture tier, as [`tap_ceil`].
pub fn tap_ceils(tier: Tier) -> String {
    match tier {
        Tier::Real => format!("ceilings {TAP_EARLY_CEIL} (layers < {TAP_EARLY}) and {TAP_CEIL}"),
        Tier::Fixture => unreachable!("the minimaxm3 gate has no fixture tier"),
    }
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

/// Open `file` on the gate card at `ctx` positions by the gate placement (`gate_card::plan_gate`) of `arm`: the body opened as
/// the serve seat opens it (`Body::open_placed`, the plan `PlanInputs::plan` makes), the taps armed before the step is first
/// captured, so a replay writes them.
///
/// The including gate declares `shared/gate_card.rs` as `gate_card` beside this module.
pub fn open(file: &Path, ctx: usize, arm: Arm) -> Result<Box<dyn Seat>, GateError> {
    if arm == Arm::Bp {
        return Err(
            "minimaxm3 seat: arm bp (plan (b′): the next-largest card an expert tier under the host tier) is refused by name: the \
             MiniMax-M3 body serves no expert tier card yet (`PlaceError::Tier`)"
                .into(),
        );
    }
    Ok(Box::new(load(file, ctx, false)?))
}

/// The body on the gate card, loaded by [`open`]'s plan; `route_trace` makes the load one the host tier can write a route trace
/// for ([`route_trace`]).
fn load(file: &Path, ctx: usize, route_trace: bool) -> Result<Loaded, GateError> {
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    crate::gate_card::init()?;
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    let inputs = described(&split)?;
    let machine = crate::gate_card::plan_gate(inputs.model.layers);
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let host = HostCfg {
        route_trace,
        ..levers.host()
    };
    let mut m = Body::open_placed(split, &plan, &inputs, 0, host)?;
    // Armed under eager steps, then captured with the taps.
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

/// Load the file as [`open`] does with the host tier writing a route trace into the new directory `dir`, feed `ids` one step each
/// from the first position, and finish the trace: every sparse layer's routed ids a position, in router rank order, and the slot
/// kind each ran in, as a router set (`crates/gpu/src/host/route_trace.rs`) beside the reference set's `ffn_moe_topk-*` for the
/// first layer and position where the two disagree. Returns the positions on disk.
pub fn route_trace(file: &Path, ctx: usize, dir: &Path, ids: &[u32]) -> Result<u64, GateError> {
    let mut seat = load(file, ctx, true)?;
    let inputs =
        described(&Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?)?;
    let hp = &inputs.hp;
    let (n_expert, n_used, layers) = (hp.experts.n_expert, hp.experts.n_used, hp.n_layer);
    let first = (0..layers)
        .find(|&l| hp.kind(l) == Kind::Sparse)
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
struct Loaded {
    m: MiniMaxM3Model,
    dims: Dims,
}

impl Loaded {
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
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    let inputs = described(&split)?;
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
        let sparse = inputs.hp.kind(l) == Kind::Sparse;
        let on_card = plan.n_l.get(l).copied().unwrap_or(0);
        let total = inputs.model.layer_experts(l);
        let mut end = l + 1;
        while end < layers
            && (inputs.hp.kind(end) == Kind::Sparse) == sparse
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
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    let inputs = described(&split)?;
    let machine = workstation::plan_gate(inputs.model.layers);
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let two = NonZeroUsize::new(2).ok_or("two is not zero")?;
    match Body::open_placed_slots(split, &plan, &inputs, 0, levers.host(), Residency::Off, two) {
        Ok(_) => Err("the body opened with two resident sequences".into()),
        Err(e) if e.to_string().contains(WANT) => Ok(e.to_string()),
        Err(e) => Err(format!("the refusal does not name {WANT:?}: {e}").into()),
    }
}
