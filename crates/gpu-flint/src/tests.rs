//! Full-file consumer transitions, compared against an untouched resident sequence.

use std::num::NonZeroUsize;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::flint::fixture as flint_fixture;
use model::arch::flint::place::PlanInputs;
use model::placement::{PlanLevers, workstation};

use crate::{Body, FlintModel};

const CTX: usize = 1536;
const PREFIX: usize = 512;
const WRITTEN: usize = 1025;
const SLOTS: usize = 3;
const WHAT: &str = "flint full-file transitions";

type Error = Box<dyn std::error::Error + Send + Sync>;

struct Observation {
    token: u32,
    pos: u32,
    logits: Vec<u32>,
}

fn observe(m: &FlintModel, token: u32) -> Result<Observation, GpuError> {
    let logits = m.logits()?;
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "{WHAT}: finite final logits"
    );
    Ok(Observation {
        token,
        pos: m.pos(),
        logits: logits.into_iter().map(f32::to_bits).collect(),
    })
}

fn equal(label: &str, actual: &Observation, expected: &Observation) {
    assert_eq!(actual.pos, expected.pos, "{label}: sequence position");
    assert_eq!(actual.token, expected.token, "{label}: next-token argmax");
    assert_eq!(
        actual.logits.len(),
        expected.logits.len(),
        "{label}: vocabulary"
    );
    let mismatch = actual
        .logits
        .iter()
        .zip(&expected.logits)
        .position(|(a, b)| a != b);
    assert!(
        mismatch.is_none(),
        "{label}: final logit bits first differ at {mismatch:?}"
    );
}

fn prompt(m: &mut FlintModel, ids: &[u32]) -> Result<u32, GpuError> {
    prompt_pass(m, |m| {
        m.run_rows(ids.len(), WHAT, |gpu, w, body, head, at| {
            crate::prefill::prefill(gpu, w, body, head, ids, at)
        })
    })?
    .ok_or(GpuError::State {
        what: WHAT,
        missing: "the prompt's final token",
    })
}

fn stream(vocab: u32, seed: u32) -> Vec<u32> {
    let mut state = seed;
    (0..=WRITTEN)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state % vocab
        })
        .collect()
}

/// Two decode observations straddle the same warmed context in every arm. The remainder wraps
/// the 1024-row window ring, while leaving a saved decode result exactly at its rollback boundary.
fn feed(m: &mut FlintModel, ids: &[u32]) -> Result<([Observation; 2], Observation), GpuError> {
    prompt(m, &ids[..PREFIX])?;
    let first = m.step(&ids[PREFIX..PREFIX + 1])?;
    let first = observe(m, first)?;
    let second = m.step(&ids[PREFIX + 1..PREFIX + 2])?;
    let second = observe(m, second)?;
    let last = prompt(m, &ids[PREFIX + 2..WRITTEN])?;
    Ok(([first, second], observe(m, last)?))
}

#[test]
#[ignore = "needs a CUDA card and the complete Flint GGUF in BLOOMERY_MODEL"]
fn hw_full_file_capture_slots_reset_and_ring_boundary() -> Result<(), Error> {
    let path = std::env::var_os("BLOOMERY_MODEL")
        .ok_or("flint full-file transitions: BLOOMERY_MODEL must name the complete Flint GGUF")?;
    let file = Split::open(path)?;
    let inputs = PlanInputs::read(&file)?;
    // PIN(2026-11-18): the released file's 49 layers or the flint fixture's (only the headline models keep real files).
    // The fixture holds every kernel class and both attention kinds (`model::arch::flint::fixture`), at the released widths.
    assert!(
        [49, flint_fixture::LAYER_MAP.len()].contains(&inputs.spec.layers.len()),
        "the released 49 layers or the flint fixture's {}, not another cut: {}",
        flint_fixture::LAYER_MAP.len(),
        inputs.spec.layers.len()
    );
    assert_eq!(
        inputs.spec.hidden, 2048,
        "the released full model residual width"
    );
    let ids = stream(inputs.spec.vocab, 0x4f1a_29c3);
    let foreign = stream(inputs.spec.vocab, 0xe72d_810b);
    assert_ne!(
        &ids[..PREFIX],
        &foreign[..PREFIX],
        "the interleaved slot has a distinct history"
    );
    let census = bloomery_gpu::census()?;
    let [device] = census.as_slice() else {
        return Err(format!(
            "{WHAT}: exactly one CUDA device must be visible, found {}",
            census.len()
        )
        .into());
    };
    let machine = workstation::plan_on(
        workstation::spec_of_device(device),
        inputs.spec.layers.len(),
    );
    let slots = NonZeroUsize::new(SLOTS).expect("three resident sequences");
    let plan = inputs.plan_with_slots(&machine, CTX as u64, &PlanLevers::default(), slots)?;
    let host = HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };
    let mut m = Body::open_placed_slots(file, &plan, &inputs, 0, host, Residency::Off, slots)?;
    m.add_slots(SLOTS)?;

    // Slot 2 is a fresh sequence run without interleaving: its outputs are the control, not a guessed numerical band.
    m.select_slot(2)?;
    m.set_mode(StepMode::Eager);
    let (control_decode, control_end) = feed(&mut m, &ids)?;
    let control_next = m.step(&ids[WRITTEN..WRITTEN + 1])?;
    let control_next = observe(&m, control_next)?;

    m.select_slot(0)?;
    m.set_mode(StepMode::Graph);
    let (captured_decode, captured_end) = feed(&mut m, &ids)?;
    for (row, (actual, expected)) in captured_decode.iter().zip(&control_decode).enumerate() {
        equal(
            &format!("eager versus captured decode row {row}"),
            actual,
            expected,
        );
    }
    equal(
        "eager versus captured after ring wrap",
        &captured_end,
        &control_end,
    );
    assert!(m.has_capture(), "slot 0 actually captured a decode chain");
    let capture_nodes = m.step_graph_nodes()?.len();

    // A second captured slot must not redirect the first slot's graph or its cache handles.
    m.select_slot(1)?;
    feed(&mut m, &foreign)?;
    assert!(m.has_capture(), "slot 1 captured its own decode chain");
    m.select_slot(0)?;
    assert_eq!(
        m.pos(),
        WRITTEN as u32,
        "switching preserved slot 0's position"
    );
    assert!(
        m.has_capture(),
        "slot 0's capture remained parked with its stores"
    );
    assert_eq!(
        m.step_graph_nodes()?.len(),
        capture_nodes,
        "slot 0's captured chain survived the switch"
    );
    let resumed = m.step(&ids[WRITTEN..WRITTEN + 1])?;
    equal(
        "resumed slot versus untouched control",
        &observe(&m, resumed)?,
        &control_next,
    );

    // Reset after overwrite, then refeed through the retained capture, must reproduce a fresh sequence.
    m.reset()?;
    assert_eq!(m.pos(), 0, "reset rewound the selected sequence");
    assert!(m.has_capture(), "reset retains the reusable decode capture");
    let (refed_decode, refed_end) = feed(&mut m, &ids)?;
    for (actual, expected) in refed_decode.iter().zip(&control_decode) {
        equal(
            "reset/refeed decode versus fresh sequence",
            actual,
            expected,
        );
    }
    equal(
        "reset/refeed wrapped state versus fresh sequence",
        &refed_end,
        &control_end,
    );

    // At written end 1025, the ring holds positions 1..1025. Replaying position 513
    // needs 1..513, but position 512 needs overwritten position 0: depth 512, not 513.
    let safe = WRITTEN as u32 - 512;
    let lost = safe - 1;
    assert!(
        m.body(WHAT)?.cut_keeps(m.pos(), safe),
        "depth 512 is retained"
    );
    assert!(
        !m.body(WHAT)?.cut_keeps(m.pos(), lost),
        "depth 513 requires reset/refeed"
    );
    m.rollback(safe)?;
    assert_eq!(
        m.pos(),
        safe,
        "retained rollback moved the logical position"
    );
    let refusal = m.rollback(lost);
    assert!(
        matches!(&refusal, Err(GpuError::Shape { what, detail })
        if *what == "flint Body" && detail.contains("ring keys") && detail.contains("reset and refeed")),
        "lost-window rollback must name the ring and recovery, got {refusal:?}"
    );
    assert_eq!(m.pos(), safe, "refused rollback moved no sequence position");
    let replayed = m.step(&ids[safe as usize..safe as usize + 1])?;
    equal(
        "retained boundary replay",
        &observe(&m, replayed)?,
        &control_decode[1],
    );

    // Reset/refeed of slot 0 did not erase the parked control's position or continuation.
    m.select_slot(2)?;
    assert_eq!(
        m.pos(),
        (WRITTEN + 1) as u32,
        "reset did not rewind a parked sequence"
    );
    Ok(())
}
