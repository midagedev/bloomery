//! The placed-load gate's shared body. A hybrid family's gate opens the body's placed load of the gate plan and
//! proves it against the plan; what is the same for every such family is here: the plan's context and the host tier's
//! configuration, the plan's levers, the clauses that read only the plan, the loaded weights and the slot map, the F32
//! readback, the report of a clause and the run's tail. A family's gate keeps what its file and body force: how the
//! inputs are read, how the body is opened, the type its routed stacks sit in on the card and the lines it prints of
//! them.

use std::collections::BTreeSet;
use std::time::Duration;

use bloomery_gpu::hybrid::{HOST, SlotMap};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu_gates::tier::{self, Tag};
use bloomery_gpu_gates::{GateError, checks_failed, split_f32, verdict};
use cuda_core::CudaStream;
use model::arch::place_hybrid::PlanInputs;
use model::placement::{Device, Machine, Plan, PlanLevers, Role, accepts};

/// The plan's and the load's context: the fixture's and the real file's prompts all fit, and the K/V planes stay small
/// beside the weights.
pub const CTX: u64 = 4_096;

/// The host tier's configuration the load opens with: no lever is read.
pub const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
    populate: false,
    lock: false,
    card_dontneed: false,
    r8: false,
    route_trace: false,
};

/// The routed stacks, in the order the host leg reads them.
pub const STACKS: [&str; 3] = ["gate", "up", "down"];

/// A clause that runs after the load.
pub type Clause<'a> = Box<dyn Fn() -> Result<bool, GateError> + 'a>;

/// The layers of a file the clauses count: every layer holds a K/V plane, the layers after the dense lead hold the
/// routed experts.
#[derive(Clone, Copy, Debug)]
pub struct Routed {
    /// The file's layers.
    pub n_layer: usize,
    /// The leading layers that hold no routed expert and have no row in the slot map.
    pub dense_lead: usize,
    /// The routed experts of each routed layer.
    pub n_expert: usize,
}

impl Routed {
    fn routed_layers(self) -> usize {
        self.n_layer - self.dense_lead
    }

    /// The routed layers a sample reads: the first, the middle and the last.
    pub fn sample(self) -> Vec<usize> {
        let first = self.dense_lead;
        let mut layers = vec![first, first + (self.n_layer - first) / 2, self.n_layer - 1];
        layers.dedup();
        layers
    }
}

/// The plan's levers: the fixture's card keeps half its experts (so the slot map has host marks and the prefix is a
/// proper one), the real tier's the card's own bytes, which already leave most to the host.
pub fn levers<H>(
    inputs: &PlanInputs<H>,
    machine: &Machine,
    tier_now: tier::Tier,
) -> Result<PlanLevers, GateError> {
    if tier_now == tier::Tier::Real {
        return Ok(PlanLevers::default());
    }
    let all = inputs.plan(machine, CTX, &PlanLevers::default())?;
    let c = &all.cards[0];
    let card = &machine.cards[0];
    let kept = c.dense_bytes
        + c.rounding_bytes
        + c.kv_bytes
        + c.stage_rows_bytes
        + card
            .set_aside_bytes()
            .ok_or("the card's set-aside is unbounded")?
        + card.margin_bytes
        + inputs.prompt_reserve
        + c.expert_bytes / 2;
    Ok(PlanLevers {
        card_budget_bytes: Some(kept).into(),
        ..PlanLevers::default()
    })
}

/// A clause's line and, for each problem it found, a `FAIL` line; whether it found none.
pub fn report(clause: &str, text: &str, bad: &[String]) -> bool {
    println!("{clause}: {text} {}", verdict(bad.is_empty()));
    for b in bad {
        println!("FAIL: {clause}: {b}");
    }
    bad.is_empty()
}

/// The routed stacks' bytes where `plan` puts each layer's experts: its card prefix in the card format the plan loads
/// it in (`accepts::leg_card_format`, measured by `CardFormat::buffer_bytes`: a stack of a whole number of words, so a
/// stack whose rows end inside a word takes more on the card than in the file), and the rest in the file's bytes.
fn placed_stack_bytes<H>(plan: &Plan<'_>, inputs: &PlanInputs<H>) -> Result<(u64, u64), GateError> {
    let (mut card, mut host) = (0u64, 0u64);
    for t in inputs
        .model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        let (Some(l), &[k, rows, experts]) = (t.layer, t.dims.as_slice()) else {
            return Err(format!("{}: a routed stack of dims {:?}", t.name, t.dims).into());
        };
        let on = plan.n_l[l];
        let per_card: u64 = accepts::leg_card_format(t.ty)
            .buffer_bytes(t.ty, k, rows)
            .ok_or_else(|| format!("{}: no card buffers of {rows} rows of {k}", t.name))?
            .iter()
            .sum();
        card += on * per_card;
        host += (experts - on) * (t.file_bytes / experts);
    }
    Ok((card, host))
}

/// The plan puts each routed expert in exactly one place — the card's prefix and the host's rest, no tier and no paged
/// expert, the dense lead keeping none — its expert bytes are the stacks' in the card format and in the file, and its
/// KV term is the typed description's planes.
pub fn plan_places_every_expert<H>(
    plan: &Plan<'_>,
    inputs: &PlanInputs<H>,
    shape: Routed,
) -> Result<bool, GateError> {
    const CLAUSE: &str = "plan_places_every_expert";
    let card = &plan.cards[0];
    let on_card: u64 = plan.n_l.iter().sum();
    let routed = shape.routed_layers();
    let total = (routed * shape.n_expert) as u64;
    let over = if shape.dense_lead == 0 {
        String::new()
    } else {
        format!(" over {routed} routed layers")
    };
    let (card_stacks, host_stacks) = placed_stack_bytes(plan, inputs)?;
    let planes = inputs.kv.bytes(0..shape.n_layer, plan.ctx_max);
    let mut bad = Vec::new();
    if on_card + plan.host.experts != total || card.experts != on_card {
        bad.push(format!(
            "{on_card} experts on the card (totals {}) + {} on the host, the file's {total}{over}",
            card.experts, plan.host.experts
        ));
    }
    if let Some(l) = (0..shape.dense_lead).find(|&l| plan.n_l[l] != 0) {
        bad.push(format!(
            "the dense lead layer {l} keeps {} card experts",
            plan.n_l[l]
        ));
    }
    if (card.expert_bytes, plan.host.expert_bytes) != (card_stacks, host_stacks) {
        bad.push(format!(
            "expert bytes {} on the card + {} on the host, the stacks' {card_stacks} in the \
             card format + {host_stacks} in the file",
            card.expert_bytes, plan.host.expert_bytes
        ));
    }
    if plan.host.nvme_expert_bytes != 0 || plan.tier_n_l.iter().flatten().any(|&n| n != 0) {
        bad.push(format!(
            "paged or tiered experts: nvme {} B, tier counts {:?}",
            plan.host.nvme_expert_bytes, plan.tier_n_l
        ));
    }
    if card.kv_bytes != planes {
        bad.push(format!(
            "kv bytes {} B, the planes' {planes} B",
            card.kv_bytes
        ));
    }
    let text = format!(
        "ctx {} card experts {on_card} ({} B), host experts {} ({} B), kv {} B (plan values)",
        plan.ctx_max, card.expert_bytes, plan.host.experts, plan.host.expert_bytes, card.kv_bytes
    );
    Ok(report(CLAUSE, &text, &bad))
}

/// A resident weight's buffers, measured from the device.
fn buffers(dw: &DevWeight) -> Vec<u64> {
    dw.buffer_bytes().into_iter().map(|b| b as u64).collect()
}

/// Every segment the plan puts on the card is resident at the plan's buffers, nothing else is, and the total is the
/// plan's dense plus expert bytes.
pub fn segments_are_the_plans(plan: &Plan<'_>, w: &Weights) -> Result<bool, GateError> {
    const CLAUSE: &str = "segments_are_the_plans";
    let (mut segments, mut planned, mut bad) = (0usize, BTreeSet::new(), Vec::new());
    for row in &plan.rows {
        let t = &plan.model.tensors[row.tensor];
        for seg in row.segments.iter().filter(|s| s.device == Device::Card(0)) {
            segments += 1;
            planned.insert(t.name.as_str());
            let want = seg.buffer_bytes(t, plan.model.experts);
            match (w.get(&t.name).map(buffers), want) {
                (Some(got), Ok(want)) if got == want => {}
                (Some(got), want) => bad.push(format!(
                    "{}: buffers {got:?} B resident, the plan's segment {want:?}",
                    t.name
                )),
                (None, _) => bad.push(format!("{}: not resident", t.name)),
            }
        }
    }
    for n in w.names().filter(|n| !planned.contains(n)) {
        bad.push(format!("{n}: resident, and the plan has no segment of it"));
    }
    let t = &plan.cards[0];
    let (got, want) = (w.resident_bytes() as u64, t.dense_bytes + t.expert_bytes);
    if got != want {
        bad.push(format!(
            "total {got} B, the plan's dense + experts {want} B"
        ));
    }
    let text = format!("{segments} segments, {got} B resident, plan dense + experts {want} B");
    Ok(report(CLAUSE, &text, &bad))
}

/// The F32 tensor `name` on the card against the file's values, bit for bit; a difference is pushed to `bad`. Whether
/// the tensor was resident and read back.
pub fn check_f32(
    file: &gguf::Split,
    w: &Weights,
    stream: &CudaStream,
    name: &str,
    bad: &mut Vec<String>,
) -> Result<bool, GateError> {
    let (_, info) = file
        .find(name)
        .ok_or_else(|| format!("{name} is not in the file"))?;
    let n = usize::try_from(info.dims.iter().product::<u64>())?;
    let want = split_f32(file, name, n)?;
    let Some(DevWeight::F32 { w: tensor, .. }) = w.get(name) else {
        bad.push(format!("{name}: not resident as an F32 tensor"));
        return Ok(false);
    };
    let got = tensor.buf().to_host_vec(stream)?;
    if got.len() != want.len()
        || got
            .iter()
            .zip(&want)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bad.push(format!("{name}: values differ from the file's"));
    }
    Ok(true)
}

/// Each routed layer's row of the tier's slot map holds the card's experts at slots `0..n_l` in id order and the host
/// mark on the rest, the layer's card count is the plan's, and the dense lead has no row.
pub fn slot_map_is_the_prefix(
    map: &SlotMap,
    plan: &Plan<'_>,
    shape: Routed,
) -> Result<bool, GateError> {
    const CLAUSE: &str = "slot_map_is_the_prefix";
    let mut bad = Vec::new();
    for l in 0..shape.dense_lead {
        if map.row(l).is_some() {
            bad.push(format!("the dense lead layer {l} has a row"));
        }
    }
    for (l, &planned) in plan
        .n_l
        .iter()
        .enumerate()
        .take(shape.n_layer)
        .skip(shape.dense_lead)
    {
        let n_l = usize::try_from(planned)?;
        let Some(row) = map.row(l) else {
            bad.push(format!("layer {l}: the map has no row"));
            continue;
        };
        let prefix = row
            .iter()
            .enumerate()
            .all(|(e, &s)| if e < n_l { s == e as u32 } else { s == HOST });
        if row.len() != shape.n_expert || !prefix || map.on_card(l)? != n_l {
            bad.push(format!(
                "layer {l}: row of {} entries, card count {} (plan {n_l}), prefix {prefix}",
                row.len(),
                map.on_card(l)?
            ));
        }
    }
    let text = if shape.dense_lead == 0 {
        format!(
            "{} layers, the plan's card counts are the ids' prefix",
            shape.n_layer
        )
    } else {
        format!(
            "{} routed layers, the plan's card counts are the ids' prefix, the dense lead has no \
             row",
            shape.routed_layers()
        )
    };
    Ok(report(CLAUSE, &text, &bad))
}

/// The load's resident bytes stay inside the plan's terms for them, the prompt reserve among them.
pub fn resident_inside_the_plan(
    resident: u64,
    plan: &Plan<'_>,
    prompt_reserve: u64,
) -> Result<bool, GateError> {
    const CLAUSE: &str = "resident_inside_the_plan";
    let c = &plan.cards[0];
    let planned = c.dense_bytes
        + c.expert_bytes
        + c.kv_bytes
        + c.stage_rows_bytes
        + c.scratch_bytes
        + prompt_reserve;
    let pass = resident <= planned;
    println!(
        "{CLAUSE}: resident {resident} B (runtime value) against the plan's dense + experts + \
         kv + stage rows + scratch + prompt reserve {planned} B {}",
        verdict(pass)
    );
    Ok(pass)
}

/// The load's line: runtime values.
pub fn load_line(resident_bytes: usize, layers: usize, took: Duration) {
    println!(
        "load resident_bytes={resident_bytes} ctx={CTX} layers={layers} in {:.1} s (runtime value)",
        took.as_secs_f64()
    );
}

/// `clause` when the tier runs `name` (every clause is self-consistency, run in both tiers); whether it passed, and
/// true for a clause the tier left out.
pub fn run_clause(
    name: &str,
    clause: impl FnOnce() -> Result<bool, GateError>,
) -> Result<bool, GateError> {
    if !tier::run_clause(name, Tag::SelfConsistency)? {
        return Ok(true);
    }
    let pass = clause()?;
    if !pass {
        println!("clause {name} FAIL");
    }
    Ok(pass)
}

/// [`run_clause`] over each of `clauses` in order; whether all passed.
pub fn run_clauses(clauses: &[(&str, Clause<'_>)]) -> Result<bool, GateError> {
    let mut ok = true;
    for (name, clause) in clauses {
        ok &= run_clause(name, clause)?;
    }
    Ok(ok)
}

/// The tally of the clauses that ran, and the run's failure when one did not pass.
pub fn conclude(ok: bool) -> Result<(), GateError> {
    println!("{}", tier::tally_line());
    if ok { Ok(()) } else { Err(checks_failed()) }
}
