//! Block / end-to-end harness self-check (package P7b) — host-only, no
//! device code. Before any engine output is judged against the oracle,
//! the instrument itself must pass:
//!
//! (a) a dump compared against itself through the whole harness: every
//!     tap's rel exactly 0, every tap's manifest contract verified, and
//!     the logical VIEW twins proven against their base tensors (the
//!     reconstruction the kernel gates do by hand);
//! (b) ik's CPU dump against its CUDA dump through the same path
//!     reproducing the measured backend-to-backend distances — the
//!     harness's calibration pair — with the last-row argmax equal;
//! (c) a corrupted mid-chain tensor named as the FIRST divergence under
//!     bands built from (b) x 2: the debugging property the engine gate
//!     will rely on, "which op broke first".
//!
//! The full per-tap CPU-vs-CUDA table for layers 0/1/13/26 and the head
//! is the table the lead pins the engine block bands from. The CUDA set
//! defaults to `ref_cuda_v2` (`BLOOMERY_REF_SET` overrides, the same
//! variable `tools/ref/dump.sh` names its output by), the CPU set to
//! `ref` (`BLOOMERY_REF_CPU_SET`).

use bloomery_gpu_gates::block::{
    Bands, BlockKind, M_TOKENS, TapKind, check_logical_views, check_row, compare_in, print_table,
    taps,
};
use bloomery_gpu_gates::{
    GateError, find_ref_row_in, ref_dir, ref_dir_named, ref_manifest_in, ref_tensor_logical_in,
    verdict,
};
use std::path::PathBuf;

/// The distance between ik's own CPU and CUDA backends, measured when the
/// two oracle sets were made — the harness's instrument check reproduces
/// these within 10 % relative. Anything else means the comparison path
/// (not the engine) changed.
/// PIN(2026-09-21): l_out-0 5.0e-3, l_out-1 1.4e-2, l_out-13 6.4e-3,
/// l_out-26 1.2e-2, result_output 2.9e-2, last-row argmax 8913.
const KNOWN_DISTANCES: [(TapKind, usize, f32); 5] = [
    (TapKind::LOut, 0, 5.0e-3),
    (TapKind::LOut, 1, 1.4e-2),
    (TapKind::LOut, 13, 6.4e-3),
    (TapKind::LOut, 26, 1.2e-2),
    (TapKind::ResultOutput, 27, 2.9e-2),
];

/// The oracle's own greedy token for the prompt both sets were dumped
/// with; both backends' last row must argmax here.
const IK_ARGMAX: usize = 8913;

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_block", run())
}

fn run() -> Result<(), GateError> {
    let mut ok = true;

    let cpu_set = std::env::var("BLOOMERY_REF_CPU_SET").unwrap_or_else(|_| "ref".to_string());
    // The CUDA set is `ref_dir()`'s (default `ref_cuda_v2`, overridable by
    // `BLOOMERY_REF_SET` / `BLOOMERY_REF_CUDA`) — one owner of the default.
    let cuda_dir: PathBuf = ref_dir();
    let cuda_set = cuda_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| cuda_dir.display().to_string());
    let cpu_dir: PathBuf = ref_dir_named(&cpu_set);
    let man_c = ref_manifest_in(&cuda_dir)?;
    let man_cpu = ref_manifest_in(&cpu_dir)?;
    println!(
        "gate_block: cuda set {} ({} rows), cpu set {} ({} rows)",
        cuda_dir.display(),
        man_c.len(),
        cpu_dir.display(),
        man_cpu.len()
    );

    // The blocks the tap tables cover: the dense block, two MoE blocks,
    // the last block, the head. Layer 27 is the head's ordering slot.
    let blocks: [(BlockKind, usize); 5] = [
        (BlockKind::Dense0, 0),
        (BlockKind::Moe, 1),
        (BlockKind::Moe, 13),
        (BlockKind::Last, 26),
        (BlockKind::Head, 27),
    ];

    // ---- (a) self-vs-self: read every tap twice through the full path
    // (manifest lookup, contract check, logical load, compare); the two
    // reads must be exactly equal. A nonzero rel here is a loader defect,
    // not a model fact.
    println!("self-check a: cuda set against itself");
    for &(kind, layer) in &blocks {
        for tap in taps(kind, layer) {
            let name = tap.name();
            let row = find_ref_row_in(&cuda_dir, &man_c, &name, tap.occurrence)?;
            check_row(row, &tap, M_TOKENS)?;
            let first = ref_tensor_logical_in(&cuda_dir, row)?;
            let r = compare_in(&cuda_dir, &man_c, &tap, &first, M_TOKENS)?;
            let zero = r.rel == 0.0;
            println!(
                "self L{:>2} {:<16} occ={} n={} rel={} {}",
                layer,
                name,
                tap.occurrence,
                r.n,
                r.rel,
                verdict(zero)
            );
            if !zero {
                ok = false;
            }
        }
    }
    for &layer in &[0usize, 1, 13, 26] {
        check_logical_views(&cuda_dir, &man_c, layer, M_TOKENS)?;
        println!("self L{layer} view-logical twins gather-check PASS");
    }

    // ---- (b) CPU vs CUDA: the calibration pair. The printed table is
    // what the engine block bands get pinned from.
    let mut results = Vec::new();
    for &(kind, layer) in &blocks {
        for tap in taps(kind, layer) {
            let name = tap.name();
            let row_cpu = find_ref_row_in(&cpu_dir, &man_cpu, &name, tap.occurrence)?;
            check_row(row_cpu, &tap, M_TOKENS)?;
            let cpu_vals = ref_tensor_logical_in(&cpu_dir, row_cpu)?;
            results.push(compare_in(&cuda_dir, &man_c, &tap, &cpu_vals, M_TOKENS)?);
        }
    }
    print_table("cpu-vs-cuda", &cpu_set, &cuda_set, &results, None);
    for &(kind, layer, want) in &KNOWN_DISTANCES {
        let r = results
            .iter()
            .find(|r| r.tap.kind == kind && r.tap.layer == layer)
            .ok_or_else(|| format!("gate_block: no result for {kind:?} L{layer}"))?;
        let diff = (r.rel - want).abs();
        let pass = diff <= 0.10 * want;
        println!(
            "calib L{layer} {:<14} rel={:.3e} known={:.1e} within10%={} {}",
            kind.tensor_name(layer),
            r.rel,
            want,
            pass,
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }
    let argmax = |vals: &[f32]| -> usize {
        vals.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            })
            .0
    };
    let out_row_c = find_ref_row_in(&cuda_dir, &man_c, "result_output", 0)?;
    let out_row_cpu = find_ref_row_in(&cpu_dir, &man_cpu, "result_output", 0)?;
    let am_c = argmax(&ref_tensor_logical_in(&cuda_dir, out_row_c)?);
    let am_cpu = argmax(&ref_tensor_logical_in(&cpu_dir, out_row_cpu)?);
    let am_ok = am_c == IK_ARGMAX && am_cpu == IK_ARGMAX;
    println!(
        "calib argmax cuda={am_c} cpu={am_cpu} both={IK_ARGMAX} {}",
        verdict(am_ok)
    );
    if !am_ok {
        ok = false;
    }

    // ---- (c) first divergence: corrupt the CPU ffn_norm-13 in memory by
    // 1e-1 of its amax — a mid-chain defect downstream taps cannot hide,
    // upstream taps cannot show. Bands from (b) x 2 hold every untouched
    // tap (its own rel is half its band), so the first violation in
    // forward order must be exactly the corrupted tap and nothing else.
    let bands = Bands::from_results(&results, 2.0);
    let idx = results
        .iter()
        .position(|r| r.tap.kind == TapKind::FfnNorm && r.tap.layer == 13)
        .ok_or("gate_block: no ffn_norm-13 result")?;
    let tap = results[idx].tap.clone();
    let row_cpu = find_ref_row_in(&cpu_dir, &man_cpu, &tap.name(), tap.occurrence)?;
    let mut corrupted = ref_tensor_logical_in(&cpu_dir, row_cpu)?;
    let amax = corrupted.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    for v in &mut corrupted {
        *v += 1.0e-1 * amax;
    }
    let r2 = compare_in(&cuda_dir, &man_c, &tap, &corrupted, M_TOKENS)?;
    let mut results2 = results.clone();
    results2[idx] = r2.clone();
    match bands.assert_within(&results2) {
        Err(v) => {
            let named = v.first == Some((TapKind::FfnNorm, 13));
            let only = v.all.len() == 1;
            let pass = named && only;
            println!(
                "self-check c: corrupt ffn_norm-13 (+1e-1*amax={amax:.3e}) rel={:.3e} band={:.3e} named_first={} sole_violation={} {}",
                r2.rel,
                bands.band(TapKind::FfnNorm, 13).unwrap_or(f32::NAN),
                named,
                only,
                verdict(pass)
            );
            println!("self-check c report: {v}");
            if !pass {
                ok = false;
            }
        }
        Ok(()) => {
            println!("self-check c: FAIL — the corrupted ffn_norm-13 stayed inside its band");
            ok = false;
        }
    }

    if !ok {
        eprintln!("FAILED: gate_block");
        std::process::exit(1);
    }
    println!(
        "PASSED: gate_block self-check — self-vs-self exact; view logical twins proven; \
         the five CPU-vs-CUDA calibration distances reproduced within 10 %; argmax equal; \
         a corrupted mid-chain tensor is named as the first divergence"
    );
    Ok(())
}
