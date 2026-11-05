//! The real six-slot tier protocol against the same plan's one-card expert union.

use std::path::Path;

use bloomery_gpu::fault::{Fault, FaultSite};
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::tier::TierStats;
use bloomery_gpu::model::StepMode;
use bloomery_gpu::{GpuError, hybrid::HOST};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::flint::place::{PlanInputs, tier_batch};
use model::placement::{Plan, PlanLevers, union_plan, workstation};
use models::Ffn;

use crate::{Body, FlintModel, set_taps};

const WHAT: &str = "flint two-card full-file acceptance";
const CTX: usize = 1024;
const PROMPT: usize = 22;
const IDS: usize = 96;
const MODEL_BYTES: u64 = 25_054_688_704;
// PIN(2026-10-25): full-file SHA from the pinned Flint diagnostic model receipt.
const MODEL_SHA256: &str = "63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f";

type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    position: u32,
    argmax: u32,
    logits: Vec<u32>,
    layers: Vec<u32>,
}

fn observe(m: &FlintModel, argmax: u32) -> Result<Observation, GpuError> {
    let logits = m.logits()?;
    let layers = m.body(WHAT)?.taps(m.gpu())?;
    assert_eq!(logits.len(), 150_144, "{WHAT}: full vocabulary");
    assert_eq!(
        layers.len(),
        49 * 2048,
        "{WHAT}: every layer's merge output"
    );
    assert!(
        logits.iter().chain(&layers).all(|v| v.is_finite()),
        "{WHAT}: completed finite outputs, not an unfilled tier row"
    );
    Ok(Observation {
        position: m.pos(),
        argmax,
        logits: logits.into_iter().map(f32::to_bits).collect(),
        layers: layers.into_iter().map(f32::to_bits).collect(),
    })
}

fn same(label: &str, want: &[Observation], got: &[Observation]) {
    assert_eq!(got.len(), want.len(), "{label}: completed calls");
    for (call, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(
            (g.position, g.argmax),
            (w.position, w.argmax),
            "{label}: call {call} position and token"
        );
        for (name, a, b) in [
            ("logits", &g.logits, &w.logits),
            ("layer outputs", &g.layers, &w.layers),
        ] {
            let first = a.iter().zip(b).position(|(a, b)| a != b);
            assert!(
                a.len() == b.len() && first.is_none(),
                "{label}: call {call} {name} first differing bits {first:?}"
            );
        }
    }
}

fn prompt(m: &mut FlintModel, ids: &[u32]) -> Result<u32, GpuError> {
    prompt_pass(m, |m| {
        m.run_rows(ids.len(), WHAT, |gpu, w, body, head, at| {
            crate::prefill(gpu, w, body, head, ids, at)
        })
    })?
    .ok_or(GpuError::State {
        what: WHAT,
        missing: "the batched prompt's last token",
    })
}

fn tier_stats(m: &FlintModel) -> Result<TierStats, GpuError> {
    let tiers = m.body(WHAT)?.hybrid().tiers();
    assert_eq!(tiers.len(), 1, "{WHAT}: one actual expert tier");
    Ok(tiers[0].stats())
}

fn sent_every(before: &TierStats, after: &TierStats, layers: &[usize], first: usize) {
    for &layer in layers {
        let i = layer - first;
        assert!(
            after.layer_hits[i] > before.layer_hits[i],
            "{WHAT}: real routing sent no tier slot at layer {layer}"
        );
    }
}

/// Read the actual router's six ranked picks and the handoff's published tier places.
fn published(m: &FlintModel, layer: usize) -> Result<Vec<u32>, GpuError> {
    let gpu = m.gpu();
    gpu.context().bind_to_thread()?;
    let body = m.body(WHAT)?;
    let picks = body.ffn.route_ids().to_host_vec(gpu.stream())?;
    assert_eq!(picks.len(), 6, "{WHAT}: six real routed slots");
    assert!(picks.iter().all(|&id| id < 256), "{WHAT}: valid expert ids");
    for (i, id) in picks.iter().enumerate() {
        assert!(!picks[..i].contains(id), "{WHAT}: distinct top-six picks");
    }
    let side = body.ffn.tier().expect("the stage has a tier join");
    let places = side.tsel(0)?.to_host_vec(gpu.stream())?;
    let tier = &body.hybrid().tiers()[0];
    let ids = tier
        .set()
        .ids(layer)
        .expect("last routed layer in tier set");
    let expected: Vec<u32> = picks
        .iter()
        .map(|id| ids.iter().position(|i| i == id).map_or(HOST, |i| i as u32))
        .collect();
    assert_eq!(
        places, expected,
        "{WHAT}: rank-preserving six-slot publication"
    );
    let at = body.s.pos.to_host_vec(gpu.stream())?;
    assert_eq!(
        at,
        [m.pos() - 1],
        "{WHAT}: replay refreshed its position input"
    );
    gpu.stream().synchronize()?;
    Ok(picks)
}

fn history(
    m: &mut FlintModel,
    ids: &[u32],
    mode: StepMode,
    tier_layers: &[usize],
) -> Result<Vec<Observation>, GpuError> {
    m.reset()?;
    m.set_mode(mode);
    let tiered = !tier_layers.is_empty();
    let first = if tiered {
        m.body(WHAT)?.hybrid().tiers()[0].set().layers().start
    } else {
        0
    };
    let before = tiered.then(|| tier_stats(m)).transpose()?;
    let token = prompt(m, &ids[..PROMPT])?;
    let mut out = vec![observe(m, token)?];
    let after_prompt = tiered.then(|| tier_stats(m)).transpose()?;
    if let (Some(b), Some(a)) = (&before, &after_prompt) {
        sent_every(b, a, tier_layers, first);
        let batch = m.body(WHAT)?.hybrid().tier_batch_stats();
        assert_eq!(batch.len(), 1, "{WHAT}: actual tier batch port");
        assert!(
            batch[0].served > 0 && batch[0].settles > 0,
            "{WHAT}: batched prompt published and completed tier work"
        );
    }
    let mut previous_picks = None;
    let mut changed_picks = false;
    let mut graph_nodes = None;
    for &id in &ids[PROMPT..] {
        let token = m.step(&[id])?;
        out.push(observe(m, token)?);
        if mode == StepMode::Graph {
            assert!(m.has_capture(), "{WHAT}: graph mode really captured");
            let nodes = m.step_graph_nodes()?.len();
            assert!(nodes > 0, "{WHAT}: nonempty captured decode");
            if let Some(previous) = graph_nodes {
                assert_eq!(
                    nodes, previous,
                    "{WHAT}: changed inputs reuse the captured chain"
                );
            }
            graph_nodes = Some(nodes);
        }
        if let Some(&last) = tier_layers.last() {
            let picks = published(m, last)?;
            changed_picks |= previous_picks.as_ref().is_some_and(|p| *p != picks);
            previous_picks = Some(picks);
        }
    }
    assert_eq!(
        m.pos(),
        IDS as u32,
        "{WHAT}: consumed the whole teacher-forced history"
    );
    if let Some(before) = after_prompt {
        let after = tier_stats(m)?;
        sent_every(&before, &after, tier_layers, first);
        assert!(
            after.issued > before.issued && after.settles > before.settles,
            "{WHAT}: decode published and completed actual tier work"
        );
        assert!(
            changed_picks,
            "{WHAT}: replay refreshed actual routing, not just tensor shapes"
        );
        if mode == StepMode::Graph {
            assert!(
                after.captures > before.captures && after.replays > before.replays,
                "{WHAT}: the second physical card captured and replayed its leg"
            );
        }
    }
    Ok(out)
}

fn open(path: &Path, plan: &Plan<'_>, inputs: &PlanInputs) -> Result<FlintModel, Error> {
    let host = HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };
    let mut m = Body::open_placed(Split::open(path)?, plan, inputs, 0, host)?;
    set_taps(&mut m, true)?;
    Ok(m)
}

#[test]
#[ignore = "needs exactly two visible CUDA devices, the complete pinned Flint GGUF in BLOOMERY_MODEL, and host RAM for its host experts"]
fn hw_two_card_full_file_union_capture_publication_fault_reset() -> Result<(), Error> {
    let path = std::env::var_os("BLOOMERY_MODEL")
        .ok_or("flint two-card acceptance: BLOOMERY_MODEL must name the complete pinned GGUF")?;
    let path = Path::new(&path);
    assert_eq!(
        std::fs::metadata(path)?.len(),
        MODEL_BYTES,
        "{WHAT}: complete file"
    );
    assert_eq!(
        gguf::digest::sha256_file(path)?,
        MODEL_SHA256,
        "{WHAT}: pinned model bytes"
    );
    let file = Split::open(path)?;
    let inputs = PlanInputs::read(&file)?;
    assert_eq!(
        (
            inputs.spec.layers.len(),
            inputs.spec.hidden,
            inputs.spec.vocab
        ),
        (49, 2048, 150_144),
        "{WHAT}: released geometry"
    );
    assert_eq!(
        inputs.model.experts_used, 6,
        "{WHAT}: production top-six program"
    );
    let census = bloomery_gpu::census()?;
    let [stage, tier] = census.as_slice() else {
        return Err(format!(
            "{WHAT}: exactly two CUDA devices must be visible, found {}",
            census.len()
        )
        .into());
    };
    let stage = workstation::spec_of_device(stage);
    let tier = workstation::spec_of_device(tier);
    assert!(
        !stage.same_device(&tier),
        "{WHAT}: two distinct physical devices"
    );
    // A reduced budget makes real tier assignments while leaving room for the union reference.
    // It is bounded by each actual free reading, not an assumed card model or the file's length.
    let budget = (8u64 << 30)
        .min(stage.free_bytes.unwrap_or(stage.usable_bytes()) / 3)
        .min(tier.free_bytes.unwrap_or(tier.usable_bytes()) / 2);
    let machine = workstation::plan_tiers(49, stage, &[tier], None, tier_batch(&inputs)?)?;
    let mut flat = machine.clone();
    flat.tiers.clear();
    let levers = PlanLevers {
        card_budget_bytes: Some(budget).into(),
        ..PlanLevers::default()
    };
    let plan = inputs.plan(&machine, CTX as u64, &levers)?;
    let routed: Vec<usize> = inputs
        .spec
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, l)| matches!(l.ffn, Ffn::Moe(_)).then_some(i))
        .collect();
    assert_eq!(plan.tier_n_l.len(), 1, "{WHAT}: one planned tier");
    for &l in &routed {
        assert!(
            plan.n_l[l] > 0 && plan.tier_n_l[0][l] > 0,
            "{WHAT}: layer {l} must assign actual experts to both physical cards, budget {budget}"
        );
    }
    let union = union_plan(&plan, &flat)?;
    assert!(
        union.cards[0].headroom_bytes >= i128::from(flat.cards[0].margin_bytes),
        "{WHAT}: the actual stage must fit the union and its reserves"
    );
    let mut state = 0x4f1a_29c3u32;
    let ids: Vec<u32> = (0..IDS)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state % inputs.spec.vocab
        })
        .collect();
    assert!(
        ids.windows(2).all(|w| w[0] != w[1]),
        "{WHAT}: changed decode inputs"
    );
    let reference = {
        let mut m = open(path, &union, &inputs)?;
        assert!(
            m.body(WHAT)?.hybrid().tiers().is_empty(),
            "{WHAT}: reference computes the union on the stage"
        );
        history(&mut m, &ids, StepMode::Eager, &[])?
    };
    let mut m = open(path, &plan, &inputs)?;
    let actual_stage = m.gpu().device_id()?;
    let actual_tier = m.body(WHAT)?.hybrid().tiers()[0].gpu()?.device_id()?;
    assert_ne!(
        actual_stage.uuid, actual_tier.uuid,
        "{WHAT}: not same-GPU wrappers"
    );
    assert_eq!(actual_stage, stage.device.expect("census stage identity"));
    assert_eq!(actual_tier, tier.device.expect("census tier identity"));
    let eager = history(&mut m, &ids, StepMode::Eager, &routed)?;
    same("two-card eager versus same-plan union", &reference, &eager);
    let graph = history(&mut m, &ids, StepMode::Graph, &routed)?;
    same(
        "two-card capture versus same-plan union",
        &reference,
        &graph,
    );

    let fault = Fault::at(routed[0] as u32, FaultSite::QuantColumn);
    m.gpu().context().bind_to_thread()?;
    assert_eq!(m.gpu().fault()?, None, "{WHAT}: stage starts clean");
    {
        let gpu = m.body(WHAT)?.hybrid().tiers()[0].gpu()?;
        gpu.context().bind_to_thread()?;
        let x = cuda_core::DeviceBuffer::from_host(gpu.stream(), &[f32::NAN; 256])?;
        let mut act = bloomery_gpu::Q8Act::with_slots(gpu.stream(), 1, 256)?;
        gpu.enqueue_quantize_q8_1_layer(&x, &mut act, routed[0])?;
        gpu.stream().synchronize()?;
        assert_eq!(gpu.fault()?, Some(fault), "{WHAT}: real tier kernel fault");
    }
    m.gpu().context().bind_to_thread()?;
    assert_eq!(
        m.gpu().fault()?,
        None,
        "{WHAT}: tier fault is not a stage injection"
    );
    let failed = m.step(&[ids[0]]);
    assert!(
        matches!(failed, Err(GpuError::Fault { fault: got, .. }) if got == fault),
        "{WHAT}: actual readback must carry the tier kernel fault, got {failed:?}"
    );
    assert_eq!(
        m.poisoned(),
        Some(fault),
        "{WHAT}: fault poisoned the model"
    );
    let position = m.pos();
    assert!(
        matches!(m.step(&[ids[1]]), Err(GpuError::Poisoned { fault: got, .. }) if got == fault),
        "{WHAT}: poisoned model must refuse further work"
    );
    assert_eq!(m.pos(), position, "{WHAT}: refusal advanced no sequence");
    m.reset()?;
    assert_eq!(m.poisoned(), None, "{WHAT}: reset lifted fault poison");
    assert_eq!(m.pos(), 0, "{WHAT}: reset rewound sequence");
    m.set_mode(StepMode::Graph);
    let token = prompt(&mut m, &ids[..PROMPT])?;
    let mut recovered = vec![observe(&m, token)?];
    let token = m.step(&ids[PROMPT..PROMPT + 1])?;
    recovered.push(observe(&m, token)?);
    same(
        "fault/reset/refeed versus clean union",
        &reference[..2],
        &recovered,
    );
    assert!(
        m.has_capture(),
        "{WHAT}: recovered model has a real decode capture"
    );
    Ok(())
}
