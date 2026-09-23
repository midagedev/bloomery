//! GPU gate for the output head (docs/gpu-design.md P8's head piece):
//! `result_norm → lm_head (Q6_K) → argmax` as one captured graph over
//! resident scratch, m = 1, against the ik CUDA oracle (`ref_cuda_v2`). The
//! What is asserted:
//! - (a) every tap finite, and inside the pinned head bands (`BANDS`);
//! - a structural fence per tap: `rel <= 0.25`, an order of magnitude above
//!   the head's expected rels and below the O(0.5..1) rels a wrong
//!   gain/geometry produces (measured with the FAIL-first mutation below). A
//!   violation here is a layout defect, not noise;
//! - (b) an eager rerun bit-identical;
//! - (c) the captured graph replays bit-identical to eager (node count
//!   printed);
//! - (d) the argmax: the device token equals the oracle's argmax over
//!   `result_output`'s last row (both proven against the pinned ik token) and
//!   equals the host argmax of our own logits under the kernel's tie rule —
//!   strictly greater replaces, so equal values keep the LOWER index;
//! - (e) a second input replayed through the SAME graph equals its eager run
//!   bit for bit — what the mutable input buffer buys. The dump holds the
//!   head input for the last position only (`l_out-26` is `{2048, 1}`), so
//!   the second input is a rotation of the first, deterministic.
//!
//! The head's input is the dump's `l_out-26` (occurrence 0, the residual ADD,
//! `{2048, 1}`) — the same tensor the CPU head gate feeds
//! (`crates/model/tests/head.rs`).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_head_gpu: built without the `gpu` feature; see `just gate-gpu-head`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::Gpu;
#[cfg(feature = "gpu")]
use bloomery_gpu::head::Head;
#[cfg(feature = "gpu")]
use bloomery_gpu::weights::Weights;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::block::{self, Bands, BlockKind, M_TOKENS, TapKind, TapResult};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::oracle::deepseek2::L_OUT_26;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, bits_equal, find_ref_row, ref_dir, ref_manifest, ref_model_path,
    ref_tensor_logical_in, verdict,
};
#[cfg(feature = "gpu")]
use gguf::Split;

/// The oracle's own greedy token for the prompt the dump sets were made with
/// (`gate_block`'s pin: both of ik's backends argmax here). Re-proved here so
/// a wrong dump set fails before any engine number is read.
#[cfg(feature = "gpu")]
const IK_ARGMAX: usize = 8913;

/// Structural fence — see the module doc. Not a band.
#[cfg(feature = "gpu")]
const FENCE: f32 = 0.25;

/// Head bands, PIN(2026-09-21) from this gate's first table (engine vs
/// `ref_cuda_v2`, fed the oracle's own `l_out-26`: result_norm 8.4e-8,
/// result_output 1.21e-2) and the two oracles' own distance on the logits
/// (ik CPU vs CUDA, `gate_block`: result_output 2.9e-2). Derivation as in
/// `gate_p8`: the larger of 2 x the measured rel and the oracle-pair
/// distance, rounded up to one digit. `result_norm` is exact arithmetic on an
/// identical input, so its band is 1e-6. The logits' 1.2e-2 is the q8_1
/// activation noise of one Q6_K gemv over 102400 rows (the lm_head's raw-x
/// rel measured 1.27e-2 in `probe-gpu-real-x`).
#[cfg(feature = "gpu")]
const BANDS: [(TapKind, usize, f32); 2] = [
    (TapKind::ResultNorm, 27, 1e-6),
    (TapKind::ResultOutput, 27, 3e-2),
];

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_head_gpu", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let mut ok = true;
    let file = Split::open(ref_model_path()?)?;
    let man = ref_manifest()?;
    let dir = ref_dir();

    // The head's input: the dump's l_out-26 — the last block's residual ADD,
    // last position only (the reference runs the head there alone). The same
    // contract the CPU head gate pins.
    let in_row = find_ref_row(&man, L_OUT_26, 0)?;
    if in_row.ty != "f32" || in_row.op != "ADD" || in_row.ne != [2048, 1, 1, 1] {
        return Err(format!(
            "gate_head_gpu: l_out-26/0 is {} {} {:?}, want f32 ADD [2048, 1]",
            in_row.ty, in_row.op, in_row.ne
        )
        .into());
    }
    let x_in = ref_tensor_logical_in(&dir, in_row)?;

    // The architecture-wide rms eps, read here independently of the caller
    // that hands it to Head (the CPU head gate's second-reader pattern).
    let eps = file
        .arch_get_f32("attention.layer_norm_rms_epsilon")
        .ok_or("gate_head_gpu: rms eps key missing or not f32")?;

    let gpu = Gpu::new()?;
    // Globals only: the head reads no block tensors and no derived weights,
    // so nothing is derived after the load.
    let w = Weights::load(gpu.stream(), &file, 0..0, true)?;
    let mut head = Head::new(&gpu, &w, eps)?;
    println!(
        "resident head_scratch_bytes={} hidden={} n_vocab={} m=1 input=l_out-26/0",
        head.resident_bytes(),
        head.hidden(),
        head.n_vocab()
    );

    // ---- eager run on the dump's last-position head input
    head.set_input(&gpu, &x_in)?;
    head.enqueue(&gpu, &w)?;
    let normed1 = head.normed_to_host(&gpu)?;
    let logits1 = head.logits_to_host(&gpu)?;
    let token1 = head.token(&gpu)? as usize;
    for (name, v) in [("result_norm", &normed1), ("result_output", &logits1)] {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            eprintln!("FAIL: {name} holds a non-finite value at {i}");
            ok = false;
        }
    }

    // ---- tap table against the oracle, inside the pinned bands
    let out_row = find_ref_row(&man, "result_output", 0)?;
    let oracle = ref_tensor_logical_in(&dir, out_row)?;
    if oracle.len() != head.n_vocab() {
        return Err(format!(
            "gate_head_gpu: result_output holds {} values, output.weight has {} rows",
            oracle.len(),
            head.n_vocab()
        )
        .into());
    }
    let mut results: Vec<TapResult> = Vec::new();
    for tap in block::taps(BlockKind::Head, 27) {
        let got: &[f32] = match tap.kind {
            TapKind::ResultNorm => &normed1,
            TapKind::ResultOutput => &logits1,
            _ => unreachable!("the head tap list holds only the two kinds above"),
        };
        results.push(block::compare_in(&dir, &man, &tap, got, M_TOKENS)?);
    }
    let bands = Bands::pinned(&BANDS);
    block::print_table(
        "head",
        "gpu_head",
        "ref_cuda[last_token]",
        &results,
        Some(&bands),
    );
    if let Err(v) = bands.assert_within(&results) {
        eprintln!("FAIL: {v}");
        ok = false;
    }
    for r in &results {
        if r.rel > FENCE {
            eprintln!(
                "FAIL: tap {} rel={:.3e} leaves the structural fence {FENCE}",
                r.tap.name(),
                r.rel
            );
            ok = false;
        }
    }

    // ---- (d) argmax: device token vs the oracle row vs our own logits,
    // one tie rule everywhere (equal values -> lower index).
    let oracle_am = argmax_low(&oracle);
    let host_am = argmax_low(&logits1);
    let am_ok = token1 == oracle_am && token1 == host_am && oracle_am == IK_ARGMAX;
    println!(
        "argmax device={token1} host={host_am} oracle={oracle_am} ik_pin={IK_ARGMAX} {}",
        verdict(am_ok)
    );
    if !am_ok {
        ok = false;
    }

    // ---- (b) eager rerun bit-identical
    head.set_input(&gpu, &x_in)?;
    head.enqueue(&gpu, &w)?;
    let rerun_same = bits_equal(&normed1, &head.normed_to_host(&gpu)?)
        && bits_equal(&logits1, &head.logits_to_host(&gpu)?)
        && token1 == head.token(&gpu)? as usize;
    println!(
        "rerun eager_bit_identical={rerun_same} {}",
        verdict(rerun_same)
    );
    if !rerun_same {
        ok = false;
    }

    // ---- (c) captured graph: replay bit-identical to eager
    let nodes = head.capture(&gpu, &w)?;
    head.set_input(&gpu, &x_in)?;
    head.launch(&gpu)?;
    let replay_same = bits_equal(&normed1, &head.normed_to_host(&gpu)?)
        && bits_equal(&logits1, &head.logits_to_host(&gpu)?)
        && token1 == head.token(&gpu)? as usize;
    println!(
        "graph graph_nodes={nodes} eager_vs_replay_bit_identical={replay_same} {}",
        verdict(replay_same)
    );
    if !replay_same {
        ok = false;
    }

    // ---- (e) second input through the SAME graph. The replay runs first:
    // the input buffer still holds the first vector at that point, so a
    // replay that skipped its set_input reads the STALE input and must break
    // the equality below (the check's FAIL-first). The eager run then proves
    // the replay's numbers are the permuted input's own.
    let n = x_in.len();
    let second: Vec<f32> = (0..n).map(|i| x_in[(i + 1) % n]).collect();
    head.set_input(&gpu, &second)?;
    head.launch(&gpu)?;
    let normed_r2 = head.normed_to_host(&gpu)?;
    let logits_r2 = head.logits_to_host(&gpu)?;
    let token_r2 = head.token(&gpu)? as usize;
    head.set_input(&gpu, &second)?;
    head.enqueue(&gpu, &w)?;
    let normed_e2 = head.normed_to_host(&gpu)?;
    let logits_e2 = head.logits_to_host(&gpu)?;
    let token_e2 = head.token(&gpu)? as usize;
    let second_same = bits_equal(&normed_r2, &normed_e2)
        && bits_equal(&logits_r2, &logits_e2)
        && token_r2 == token_e2;
    // The check is only meaningful if the second input moves the logits; a
    // rotation of a real hidden vector cannot leave 102400 logits untouched.
    let second_differs = !bits_equal(&logits1, &logits_e2);
    println!(
        "second_input rotation replay_vs_eager_bit_identical={second_same} \
         changes_logits={second_differs} {}",
        verdict(second_same && second_differs)
    );
    if !(second_same && second_differs) {
        ok = false;
    }

    // ---- for the record: top-5 of both sides
    println!("top-5 ours: {:?}", top5(&logits1));
    println!("top-5 oracle: {:?}", top5(&oracle));

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_head_gpu — head chain taps printed against the dump's last \
         token inside the pinned bands, rerun/replay/second-input bit-identical, \
         argmax equal to the oracle's under one tie rule"
    );
    Ok(())
}

/// Index of the maximum, ties to the LOWER index — `elem::argmax`'s rule
/// (`argmax_take`: strictly greater replaces, equal never does under an
/// ascending scan). Every argmax in this gate runs through it.
#[cfg(feature = "gpu")]
fn argmax_low(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Top-5 as (id, logit), sorted by (logit desc, id asc) — the same total
/// order the argmax walks, so the printed ranking cannot disagree with the
/// asserted token.
#[cfg(feature = "gpu")]
fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(5);
    idx.into_iter().map(|i| (i, v[i])).collect()
}
