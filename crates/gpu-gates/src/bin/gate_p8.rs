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
//! - (c) the captured graph replays bit-identical to eager, and the node
//!   count equals its pin (`NODES_BLOCK0`, plus the flash merge when the
//!   cache is cut into segments);
//! - (d) the SAME graph, reseeded to a second position (pos 4, n_keys 5),
//!   reproduces an eager pos-4 run's `l_out-0` bit for bit — what the
//!   device-side `pos_buf`/`n_keys_buf` buy.
//!
//! `--profile-pos <P>` moves the profiled step to position `P` (default: the
//! dump's last position) so the per-op table can be read at depth; it raises
//! the resident cache to `P + 1` rows for that run only. `--profile-ctx <N>`
//! raises that cache height on its own, without moving the position — the
//! shallow step on a tall cache, which is what a real decode run looks like
//! at its start and the case a key-split attention has to stay honest in.
//! The correctness arms above still run at the dump's positions, and the
//! extra rows stay past every kernel's live extent (`n_keys` clamps the
//! flash, `pos` the append) — but the cache height also picks the flash
//! path, so a run with either flag is the split launch's block-level
//! correctness run and one without it the single-block kernel's. The two
//! differ in summation order, so their taps are the same to the band and not
//! to the bit. Rows the dump did not seed are zeros — a real key row for
//! timing, not a skipped one.
//!
//! `--profile-layer <L>` profiles layer `L` instead of block 0 — blocks
//! `0..=L` then load, and the profiled layer's input residual is block 0's
//! own output at the profiled position (a lone layer has none of its own).
//! The correctness arms stay on block 0 whatever the flag says.
//!
//! `--time` (lead-only, under the machine lease) replays the graph 2000x
//! and prints us/replay plus an empty-graph reference; correctness runs
//! never reach it. `--profile` (lead-only, the same lease — the profiling
//! numbers are measurements too) additionally runs `GpuModel::profile_block0`:
//! one line per op — eager launch + body + one synchronize, so the printed
//! `sync_floor_us` (a bare `touch` launch + sync, sampled the same way) is
//! the per-op overhead to subtract via `net_us` — with the bytes that op
//! touches and `bytes / net_us` as GB/s beside it, then the sums, the
//! attention/FFN split (ops up to and including the attention residual add
//! vs the layer's FFN ops), `refresh_params`' host time, and the
//! graph-replay number of `--time` measured in the same process. The bytes
//! are exact counts of what the launch addresses, carried out of the engine
//! by the same tick that names the op; the footer prints their sum, the
//! effective GB/s of the whole chain and the two references of
//! docs/roofline.md so a reader sees where each op sits between them.

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
/// PIN(2026-09-21): block 0's chain captures exactly this many graph nodes
/// on a cache that fits in one flash segment, and one more when the cache is
/// cut into segments and the merge pass joins the chain (`NODES_BLOCK0 +
/// (segments > 1)`). Measured by this gate's own `graph graph_nodes=21` line
/// at ctx_max 64 (prof2 round, 2026-09-21 — HANDOFF §7 "prof2 머지"; the lead's
/// own rerun agreed). This is not a band: the node
/// count is a deterministic property of the chain, so the pin is exact and
/// its margin is zero — the derivation is "what the chain enqueues today,
/// after the fuse1 round's two fusions". A silently added launch — a fused
/// pair that stopped fusing, a debug copy left in the chain — keeps every
/// other arm of this gate green (eager still equals replay, ops still equal
/// nodes) and fails only here.
#[cfg(feature = "gpu")]
const NODES_BLOCK0: usize = 21;
/// Print-only markers for the profile table, not gates: an op touching at
/// least a mebibyte should be paying for bytes, not for its launch, so one
/// that stays under this effective bandwidth is a shape-defect candidate the
/// reader should look at.
#[cfg(feature = "gpu")]
const MARK_BYTES: u64 = 1 << 20;
#[cfg(feature = "gpu")]
const MARK_GBPS: f64 = 100.0;
/// The two references the per-op GB/s is read against, both derived in
/// docs/roofline.md: the kernel-level floor this card's gemvs reach and the
/// whole-pass average the reference engine shows.
#[cfg(feature = "gpu")]
const REF_KERNEL_FLOOR_GBPS: f64 = 700.0;
#[cfg(feature = "gpu")]
const REF_WHOLE_PASS_GBPS: f64 = 247.0;

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
    // `--profile-pos` is read before the load: the cache height is a load-time
    // decision, and the profile arm needs `pos + 1` rows.
    let profile_pos = parse_u32_flag(std::env::args(), "--profile-pos")?;
    let profile_ctx = parse_u32_flag(std::env::args(), "--profile-ctx")?;
    // The profiled layer is a load-time decision too: layer L needs blocks
    // 0..=L resident, block 0 because a lone layer has no embedding to make
    // its input residual and this gate takes that residual from block 0's
    // own output. The correctness arms below stay on block 0 either way.
    let profile_layer = parse_u32_flag(std::env::args(), "--profile-layer")?.unwrap_or(0) as usize;
    let ctx_max = match profile_pos {
        Some(p) => CTX_MAX.max(p as usize + 1),
        None => CTX_MAX,
    }
    .max(profile_ctx.unwrap_or(0) as usize);
    let gguf = open_model()?;
    let man = ref_manifest()?;
    let mut model = GpuModel::load_blocks(&gguf, ctx_max, 0..profile_layer + 1)?;
    println!(
        "resident stage_bytes={} ctx_max={ctx_max} m=1 layers=0..{}",
        model.stages()[0].resident_bytes(),
        profile_layer + 1
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
    // The merge pass exists only when the cache is cut into segments, so the
    // pinned count carries that one term and nothing else.
    let want_nodes = NODES_BLOCK0 + usize::from(bloomery_gpu::flash::segments_for(ctx_max) > 1);
    let nodes_pinned = nodes == want_nodes;
    println!(
        "graph graph_nodes={nodes} pinned_nodes={want_nodes} nodes_equal_pin={nodes_pinned} \
         eager_vs_replay_bit_identical={replay_same} {}",
        verdict(replay_same && nodes_pinned)
    );
    if !replay_same {
        ok = false;
    }
    if !nodes_pinned {
        eprintln!(
            "FAIL: block 0 captures {nodes} nodes, the pin is {want_nodes} \
             (NODES_BLOCK0 {NODES_BLOCK0} + {} for the flash merge) — a launch was added or \
             removed",
            usize::from(bloomery_gpu::flash::segments_for(ctx_max) > 1)
        );
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
        // The profiled position: the dump's last by default, `--profile-pos`
        // otherwise. The token id is the dump's last either way — at depth the
        // keys past the seeded prefix are the cache's zero rows, which is a
        // key count, not a correctness claim.
        let prof_pos = profile_pos.unwrap_or((M_TOKENS - 1) as u32);
        let prof_token = PROMPT[M_TOKENS - 1];
        // The &mut model calls come first; the shared `stream` borrow taken
        // below must not overlap them.
        // A layer above 0 reads its input residual from the resident input
        // buffer: block 0's own output at this position fills it, which is
        // what the layer would receive in a stitched step.
        if profile_layer > 0 {
            let taps = model.step_block0_taps(prof_token, prof_pos)?;
            model.set_layer_input(&taps.l_out)?;
        }
        let ops = model.profile_layer(profile_layer, prof_token, prof_pos, PROF_REPS)?;
        let refresh_us = model.refresh_params_us(prof_token, prof_pos, PROF_REPS)?;
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

        // The FFN half starts at the layer's own norm: the fused dense four
        // for a layer without a router, the routed ten for a layer with one.
        let (anchor, want_ffn_ops) = match ops.iter().any(|o| o.name == "moe_ffn_norm_quant") {
            true => ("moe_ffn_norm_quant", 10),
            false => ("ffn_norm_quant", 4),
        };
        let ffn_start = ops
            .iter()
            .position(|o| o.name == anchor)
            .ok_or("gate_p8: the profile holds no FFN norm op")?;
        if ops.len() - ffn_start != want_ffn_ops {
            return Err(format!(
                "gate_p8: the FFN half is {} ops after {ffn_start}, want the {want_ffn_ops} of \
                 the {anchor} shape",
                ops.len() - ffn_start
            )
            .into());
        }
        println!(
            "prof reps={PROF_REPS} warmup=20 layer={profile_layer} ops={} \
             sync_floor_samples={PROF_REPS} sample=eager_launch+body+one_sync pos={prof_pos} \
             n_keys={} ctx_max={ctx_max} seg_keys={} flash_segs={}",
            ops.len(),
            prof_pos + 1,
            bloomery_gpu::flash::seg_keys(),
            bloomery_gpu::flash::segments_for(ctx_max)
        );
        // Bytes are exact counts of what the launch addresses (weight rows,
        // activation planes, outputs), each distinct byte once; `?` is an op
        // whose count the engine cannot derive. GB/s is that count over the
        // op's net time, so it is only as honest as `net_us` — an op whose
        // net time is not positive prints `?` rather than a clamped number.
        let gbps = |op: &bloomery_gpu::model::OpTime| -> Option<f64> {
            let net = op.us_min - floor_min;
            match (op.bytes, net > 0.0) {
                (Some(b), true) => Some(b as f64 / net / 1e3),
                _ => None,
            }
        };
        let show = |v: Option<f64>| match v {
            Some(x) => format!("{x:.1}"),
            None => "?".to_string(),
        };
        for op in &ops {
            let mark = match (op.bytes, gbps(op)) {
                (Some(b), Some(g)) if b >= MARK_BYTES && g < MARK_GBPS => " shape_defect_candidate",
                _ => "",
            };
            println!(
                "prof i={:02} op={} us_mean={:.3} us_min={:.3} net_us={:.3} bytes={} gbps={}{mark}",
                op.index,
                op.name,
                op.us_mean,
                op.us_min,
                op.us_min - floor_min,
                match op.bytes {
                    Some(b) => b.to_string(),
                    None => "?".to_string(),
                },
                show(gbps(op))
            );
        }
        let sum_us: f64 = ops.iter().map(|o| o.us_mean).sum();
        let sum_net_us: f64 = ops.iter().map(|o| o.us_min - floor_min).sum();
        let attn_us: f64 = ops[..ffn_start].iter().map(|o| o.us_mean).sum();
        let ffn_us: f64 = ops[ffn_start..].iter().map(|o| o.us_mean).sum();
        let attn_net_us: f64 = ops[..ffn_start].iter().map(|o| o.us_min - floor_min).sum();
        let ffn_net_us: f64 = ops[ffn_start..].iter().map(|o| o.us_min - floor_min).sum();
        let unknown = ops.iter().filter(|o| o.bytes.is_none()).count();
        let sum_bytes: u64 = ops.iter().filter_map(|o| o.bytes).sum();
        println!(
            "prof sum_us={sum_us:.3} sum_net_us={sum_net_us:.3} sync_floor_us={floor_min:.3} \
             sync_floor_mean_us={:.3} refresh_params_us={refresh_us:.3} \
             graph_replay_us={graph_us:.3} graph_of=block0",
            floor_sum / f64::from(PROF_REPS)
        );
        println!(
            "prof bytes sum_bytes={sum_bytes} bytes_unknown_ops={unknown} \
             sum_net_us={sum_net_us:.3} effective_gbps={} \
             ref_kernel_floor_gbps={REF_KERNEL_FLOOR_GBPS:.0} \
             ref_whole_pass_gbps={REF_WHOLE_PASS_GBPS:.0} refs=docs/roofline.md \
             marker=bytes>={MARK_BYTES}_and_gbps<{MARK_GBPS:.0}",
            show((sum_net_us > 0.0).then(|| sum_bytes as f64 / sum_net_us / 1e3))
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

/// One `<flag> <u32>` pair off the command line: `--profile-pos` names the
/// position the `--profile` table is taken at, `--profile-ctx` the cache
/// height to allocate. Absent, the profile stays at the dump's last position
/// on the gate's own cache height.
#[cfg(feature = "gpu")]
fn parse_u32_flag(
    args: impl Iterator<Item = String>,
    flag: &str,
) -> Result<Option<u32>, Box<dyn std::error::Error>> {
    let mut args = args.skip_while(|a| a != flag);
    match args.next() {
        None => Ok(None),
        Some(_) => {
            let v = args
                .next()
                .ok_or_else(|| format!("gate_p8: {flag} wants a number argument"))?;
            let p: u32 = v
                .parse()
                .map_err(|_| format!("gate_p8: {flag} {v} is not a number"))?;
            Ok(Some(p))
        }
    }
}

#[cfg(feature = "gpu")]
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(feature = "gpu")]
fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}
