//! GPU gate for package P8 (docs/gpu-design.md 작업 꾸러미): the assembled
//! block-0 decode step (`GpuModel::step_block0_taps`), m = 1, against the ik
//! CUDA oracle dump. The tap rels are PRINTED, never banded — the lead pins
//! the block bands from this table. What is asserted:
//! - (a) every tap finite;
//! - a structural fence per tap: `rel <= 0.25`. This is NOT a pinned band —
//!   it is an order of magnitude above the quantization noise the design
//!   predicts for `l_out` (1e-2 order) and below the O(0.5..1) rels a wrong
//!   concat/permutation produces (measured with the FAIL-first mutation
//!   below). A violation here is a layout defect, not noise;
//! - (b) an eager rerun bit-identical;
//! - (c) the captured graph replays bit-identical to eager (node count
//!   printed);
//! - (d) the SAME graph, reseeded to a second position (pos 4, n_keys 5),
//!   reproduces an eager pos-4 run's `l_out-0` bit for bit — what the
//!   device-side `pos_buf`/`n_keys_buf` buy.
//!
//! `--time` (lead-only, under the machine lease) replays the graph 2000x
//! and prints us/replay plus an empty-graph reference; correctness runs
//! never reach it.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p8: built without the `gpu` feature; see `just gate-gpu-p8`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::GpuModel;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::block::{self, BlockKind, M_TOKENS, TapKind, TapResult};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    find_ref_row, find_ref_row_in, max_rel_err, open_model, ref_dir, ref_manifest,
    ref_tensor_logical_in, widened_f16_bits,
};

/// The six-token prompt the CUDA dump was made for (gate_p4's constant —
/// the same dump set), positions 0..5.
#[cfg(feature = "gpu")]
const PROMPT: [u32; 6] = [100000, 549, 6077, 280, 7239, 317];
/// Cache rows allocated for the gate's runs (the dump uses 6).
#[cfg(feature = "gpu")]
const CTX_MAX: usize = 64;
/// Structural fence — see the module doc. Not a band; the pinned bands come
/// from this gate's printed table.
#[cfg(feature = "gpu")]
const FENCE: f32 = 0.25;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ok = true;
    let gguf = open_model()?;
    let man = ref_manifest()?;
    let mut model = GpuModel::load_blocks(&gguf, CTX_MAX, 0..1)?;
    println!(
        "resident stage_bytes={} ctx_max={CTX_MAX} m=1",
        model.stages()[0].resident_bytes()
    );

    // The dump's own cache rows for tokens 0..4: seeding them makes the last
    // token's attention read exactly the keys ik's did (gate_p5 proved our
    // appended bits equal ik's on the same rows; here the prefix comes from
    // the oracle side so the tap table isolates this step's own ops).
    let ik_cache = find_ref_row(&man, "kv_cache-0", 0)?;
    if ik_cache.ty != "f16" || ik_cache.op != "VIEW" || ik_cache.ne != [576, 256, 1, 1] {
        return Err(format!(
            "gate_p8: kv_cache-0 is {} {} {:?}, want VIEW f16 [576, 256]",
            ik_cache.op, ik_cache.ty, ik_cache.ne
        )
        .into());
    }

    // ---- eager run at the last position, tap table, finiteness
    let seed5 = widened_f16_bits(ik_cache, M_TOKENS - 1)?;
    model.seed_block0_cache(&seed5)?;
    let taps1 = model.step_block0_taps(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32)?;
    if let Some(name) = taps1.non_finite() {
        eprintln!("FAIL: tap {name} holds a non-finite value");
        ok = false;
    }

    let dir = ref_dir();
    let mut results: Vec<TapResult> = Vec::new();
    for tap in block::taps(BlockKind::Dense0, 0) {
        // The fused FFN exposes neither the normed vector nor the bare down
        // projection; l_out-0 carries the span.
        if matches!(tap.kind, TapKind::FfnNorm | TapKind::FfnOut) {
            println!(
                "tap {} op={} unexposed=fused_ffn covered_by=l_out-0",
                tap.name(),
                tap.op
            );
            continue;
        }
        let got: &[f32] = match tap.kind {
            TapKind::AttnNorm => &taps1.attn_norm,
            TapKind::Q => &taps1.q,
            TapKind::KvRopeCompressed => &taps1.kv_rope_compressed,
            TapKind::QRope => &taps1.q_rope,
            TapKind::KRope => &taps1.k_rope,
            TapKind::KvCompressed => &taps1.kv_compressed,
            TapKind::KqvCompressed => &taps1.kqv_compressed,
            TapKind::KqvOut => &taps1.kqv_out,
            TapKind::FfnInp => &taps1.ffn_inp,
            TapKind::LOut => &taps1.l_out,
            _ => unreachable!("the dense-0 tap list holds only the kinds above"),
        };
        let row = find_ref_row_in(&dir, &man, &tap.name(), tap.occurrence)?;
        block::check_row(row, &tap, M_TOKENS)?;
        let ref_all = ref_tensor_logical_in(&dir, row)?;
        let per = tap.kind.per_token();
        let ref_last = &ref_all[(M_TOKENS - 1) * per..M_TOKENS * per];
        if got.len() != per {
            return Err(format!(
                "gate_p8: {} got {} values, the tap's token column is {per}",
                tap.name(),
                got.len()
            )
            .into());
        }
        let rel = max_rel_err(got, ref_last)?;
        let mut worst_index = 0usize;
        let mut worst_d = -1.0f32;
        for (i, (&g, &r)) in got.iter().zip(ref_last).enumerate() {
            let d = (g - r).abs();
            if d > worst_d {
                worst_d = d;
                worst_index = i;
            }
        }
        results.push(TapResult {
            tap,
            rel,
            worst_index,
            n: per,
        });
    }
    block::print_table(
        "block0_step",
        "gpu_step",
        "ref_cuda[last_token]",
        &results,
        None,
    );
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

    // ---- (b) eager rerun bit-identical
    let taps2 = model.step_block0_taps(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32)?;
    let rerun_same = taps1.bits_equal(&taps2);
    println!(
        "rerun eager_bit_identical={rerun_same} {}",
        verdict(rerun_same)
    );
    if !rerun_same {
        ok = false;
    }

    // ---- (c) captured graph: replay bit-identical to eager
    let nodes = model.capture_block0()?;
    model.replay_block0(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32)?;
    let taps3 = model.block0_taps()?;
    let replay_same = taps3.bits_equal(&taps1);
    println!(
        "graph graph_nodes={nodes} eager_vs_replay_bit_identical={replay_same} {}",
        verdict(replay_same)
    );
    if !replay_same {
        ok = false;
    }

    // ---- (d) the same graph at a second position: reseed tokens 0..3,
    // pos 4 / n_keys 5; the replay must equal a fresh eager run there.
    let seed4 = widened_f16_bits(ik_cache, M_TOKENS - 2)?;
    model.seed_block0_cache(&seed4)?;
    let taps4 = model.step_block0_taps(PROMPT[M_TOKENS - 2], (M_TOKENS - 2) as u32)?;
    model.replay_block0(PROMPT[M_TOKENS - 2], (M_TOKENS - 2) as u32)?;
    let taps5 = model.block0_taps()?;
    let second_pos_same = bits_equal(&taps4.l_out, &taps5.l_out);
    println!(
        "second_pos pos={} n_keys={} replay_l_out_bit_identical_to_eager={second_pos_same} {}",
        M_TOKENS - 2,
        M_TOKENS - 1,
        verdict(second_pos_same)
    );
    if !second_pos_same {
        ok = false;
    }

    // ---- lead-only timing under the machine lease; correctness runs never
    // reach this.
    if std::env::args().any(|a| a == "--time") {
        const N: u32 = 2000;
        let probe = bloomery_gpu::probe::Probe::load(model.stages()[0].gpu().context())?;
        let stream = model.stages()[0].gpu().stream();
        let mut tbuf = cuda_core::DeviceBuffer::<f32>::zeroed(stream, 32)?;
        let empty = model.stages()[0]
            .gpu()
            .capture(|_| (0..4).try_for_each(|_| probe.enqueue_touch(stream, &mut tbuf)))?;
        let time_replays = |launch: &dyn Fn() -> Result<(), Box<dyn std::error::Error>>| -> Result<
            f64,
            Box<dyn std::error::Error>,
        > {
            for _ in 0..2 {
                launch()?;
            }
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..N {
                launch()?;
            }
            stream.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(N))
        };
        let step_us = time_replays(&|| {
            model
                .launch_block0_graph()
                .map_err(|e| -> Box<dyn std::error::Error> { e })
        })?;
        let touch_us = time_replays(&|| {
            empty
                .launch(stream)
                .map_err(|e| -> Box<dyn std::error::Error> { e })
        })?;
        println!("time n={N} step_us_per_replay={step_us:.3} touch4_us_per_replay={touch_us:.3}");
    }

    if !ok {
        eprintln!("FAILED: gate_p8");
        std::process::exit(1);
    }
    println!(
        "PASSED: gate_p8 — block-0 step assembled: taps printed against the dump's last \
         token, rerun/replay/second-position bit-identical, inside the structural fence"
    );
    Ok(())
}

#[cfg(feature = "gpu")]
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(feature = "gpu")]
fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}
