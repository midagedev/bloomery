//! GPU gate for package P8 (docs/gpu-design.md 작업 꾸러미): the assembled
//! block-0 decode step (`GpuModel::step_block0_taps`), m = 1, against the ik
//! CUDA oracle dump. The tap rels are PRINTED, never banded — the lead pins
//! the block bands from this table. What is asserted:
//! - (a) every tap finite;
//! - the pinned block bands (`BANDS`, one per tap) — engine-vs-oracle rels
//!   must stay inside them; the first violation in forward order names the
//!   op that broke;
//! - a structural fence per tap: `rel <= 0.25`, an order of magnitude above
//!   the bands and below the O(0.5..1) rels a wrong concat/permutation
//!   produces (measured with the FAIL-first mutation below). A violation
//!   here is a layout defect, not noise;
//! - (b) an eager rerun bit-identical;
//! - (c) the captured graph replays bit-identical to eager (node count
//!   printed);
//! - (d) the SAME graph, reseeded to a second position (pos 4, n_keys 5),
//!   reproduces an eager pos-4 run's `l_out-0` bit for bit — what the
//!   device-side `pos_buf`/`n_keys_buf` buy.
//!
//! `--time` (lead-only, under the machine lease) replays the graph 2000x
//! and prints us/replay plus an empty-graph reference; correctness runs
//! never reach it. `--profile` (lead-only, the same lease — the profiling
//! numbers are measurements too) additionally runs `GpuModel::profile_block0`:
//! one line per op — eager launch + body + one synchronize, so the printed
//! `sync_floor_us` (a bare `touch` launch + sync, sampled the same way) is
//! the per-op overhead to subtract via `net_us` — then the sums, the
//! attention/FFN split (ops up to and including the attention residual add
//! vs the four fused FFN ops), `refresh_params`' host time, and the
//! graph-replay number of `--time` measured in the same process.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p8: built without the `gpu` feature; see `just gate-gpu-p8`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::GpuModel;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::block::{self, Bands, BlockKind, M_TOKENS, TapKind, TapResult};
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
/// Structural fence — see the module doc. Not a band.
#[cfg(feature = "gpu")]
const FENCE: f32 = 0.25;

/// Block-0 bands, PIN(2026-09-21) from this gate's first table (engine vs
/// `ref_cuda_v2`, last token: attn_norm 1.1e-7, q 4.2e-3,
/// kv_rope_compressed 2.4e-3, q_rope 4.2e-3, k_rope 3.3e-3, kv_compressed
/// 5.6e-3, kqv_compressed 3.4e-3, kqv_out 4.8e-3, ffn_inp 2.5e-3, l_out
/// 1.65e-3) and the ruler of the two oracles' own distance (ik CPU vs CUDA,
/// `gate_block`: q 6.3e-3, kv_rope_compressed 3.6e-3, q_rope 6.1e-3, k_rope
/// 5.1e-3, kv_compressed 6.7e-3, kqv_compressed 2.6e-2, kqv_out 2.3e-2,
/// ffn_inp 1.1e-2, l_out 5.0e-3). Derivation: each band is the larger of
/// 2 x the measured rel and the oracle-pair distance, rounded up to one
/// digit — a band cannot be narrower than the distance between the two
/// oracles themselves, and 2 x is the rerun margin `gate_block` uses.
/// `attn_norm` is exact arithmetic on both sides (1.1e-7 on every table),
/// so its band is 1e-6.
#[cfg(feature = "gpu")]
const BANDS: [(TapKind, usize, f32); 10] = [
    (TapKind::AttnNorm, 0, 1e-6),
    (TapKind::Q, 0, 1e-2),
    (TapKind::KvRopeCompressed, 0, 5e-3),
    (TapKind::QRope, 0, 1e-2),
    (TapKind::KRope, 0, 7e-3),
    (TapKind::KvCompressed, 0, 1.2e-2),
    (TapKind::KqvCompressed, 0, 3e-2),
    (TapKind::KqvOut, 0, 2.5e-2),
    (TapKind::FfnInp, 0, 1.2e-2),
    (TapKind::LOut, 0, 6e-3),
];

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
    let bands = Bands::pinned(&BANDS);
    block::print_table(
        "block0_step",
        "gpu_step",
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

    // ---- (e) the profile mode leaves no state behind: a small
    // `profile_block0` run must observe exactly the captured graph's node
    // count (every instrumented op is a node), and a fresh eager step after
    // it must reproduce the pre-profile eager taps bit for bit.
    {
        let ops = model.profile_block0(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32, 3)?;
        model.seed_block0_cache(&seed5)?;
        let taps6 = model.step_block0_taps(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32)?;
        let ops_match_graph = ops.len() == nodes;
        let post_profile_same = taps6.bits_equal(&taps1);
        println!(
            "profile ops={} graph_nodes={nodes} ops_match_graph={ops_match_graph} \
             post_profile_eager_bit_identical={post_profile_same} {}",
            ops.len(),
            verdict(ops_match_graph && post_profile_same)
        );
        if !(ops_match_graph && post_profile_same) {
            ok = false;
        }
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

    // ---- lead-only per-op profile under the machine lease; correctness
    // runs never reach this. Each op's sample is eager launch + body + one
    // synchronize; `sync_floor_us` (a `touch` launch + sync, same sample
    // count) is that per-op overhead, subtracted in `net_us`.
    if std::env::args().any(|a| a == "--profile") {
        const PROF_REPS: u32 = 200;
        const N: u32 = 2000;
        // The &mut model calls come first; the shared `stream` borrow taken
        // below must not overlap them.
        let ops = model.profile_block0(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32, PROF_REPS)?;
        let refresh_us =
            model.refresh_params_us(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32, PROF_REPS)?;
        let probe = bloomery_gpu::probe::Probe::load(model.stages()[0].gpu().context())?;
        let stream = model.stages()[0].gpu().stream();
        let mut tbuf = cuda_core::DeviceBuffer::<f32>::zeroed(stream, 32)?;
        for _ in 0..20 {
            probe.enqueue_touch(stream, &mut tbuf)?;
            stream.synchronize()?;
        }
        let mut floor_min = f64::INFINITY;
        let mut floor_sum = 0.0f64;
        for _ in 0..PROF_REPS {
            let t0 = std::time::Instant::now();
            probe.enqueue_touch(stream, &mut tbuf)?;
            stream.synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6;
            floor_min = floor_min.min(us);
            floor_sum += us;
        }
        // The --time number, measured in this same process.
        for _ in 0..2 {
            model.launch_block0_graph()?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            model.launch_block0_graph()?;
        }
        stream.synchronize()?;
        let graph_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(N);

        let ffn_start = ops
            .iter()
            .position(|o| o.name == "ffn_norm_quant")
            .ok_or("gate_p8: the profile holds no ffn_norm_quant op")?;
        if ops.len() - ffn_start != 4 {
            return Err(format!(
                "gate_p8: the fused FFN is {} ops after {ffn_start}, want the four of the \
                 P0b shape",
                ops.len() - ffn_start
            )
            .into());
        }
        println!(
            "prof reps={PROF_REPS} warmup=20 ops={} sync_floor_samples={PROF_REPS} \
             sample=eager_launch+body+one_sync",
            ops.len()
        );
        for op in &ops {
            println!(
                "prof i={:02} op={} us_mean={:.3} us_min={:.3} net_us={:.3}",
                op.index,
                op.name,
                op.us_mean,
                op.us_min,
                op.us_min - floor_min
            );
        }
        let sum_us: f64 = ops.iter().map(|o| o.us_mean).sum();
        let sum_net_us: f64 = ops.iter().map(|o| o.us_min - floor_min).sum();
        let attn_us: f64 = ops[..ffn_start].iter().map(|o| o.us_mean).sum();
        let ffn_us: f64 = ops[ffn_start..].iter().map(|o| o.us_mean).sum();
        let attn_net_us: f64 = ops[..ffn_start].iter().map(|o| o.us_min - floor_min).sum();
        let ffn_net_us: f64 = ops[ffn_start..].iter().map(|o| o.us_min - floor_min).sum();
        println!(
            "prof sum_us={sum_us:.3} sum_net_us={sum_net_us:.3} sync_floor_us={floor_min:.3} \
             sync_floor_mean_us={:.3} refresh_params_us={refresh_us:.3} \
             graph_replay_us={graph_us:.3}",
            floor_sum / f64::from(PROF_REPS)
        );
        println!(
            "prof split attn_ops={ffn_start} ffn_ops={} attn_us={attn_us:.3} ffn_us={ffn_us:.3} \
             attn_net_us={attn_net_us:.3} ffn_net_us={ffn_net_us:.3}",
            ops.len() - ffn_start
        );
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
