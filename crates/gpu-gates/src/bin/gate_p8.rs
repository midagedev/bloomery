//! GPU gate for package P8 (docs/gpu-design.md work package): the assembled
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
//!
//! `--bench-arm <op>` with `--bench-kernels` runs only the grouped-GEMM arm
//! whose row is `bench op=<op>` (`gemm_q4k_moe_t4096`, say) and none of the
//! other arms, so every `gemm_q4k` launch of the process is that arm's — the
//! shape a counter run (`just ncu-gpu-gemm`) filters by kernel name. A name
//! no arm carries is a named error.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p8: built without the `gpu` feature; see `just gate-gpu-p8`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::Deepseek2Model;
#[cfg(feature = "gpu")]
use bloomery_gpu::model::StepProbe;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::block::{self, Bands, BlockKind, M_TOKENS, TapKind, TapResult};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, bits_equal, find_ref_row, find_ref_row_in, max_rel_err, ref_dir, ref_manifest,
    ref_model_path, ref_tensor_logical_in, verdict, widened_f16_bits,
};
#[cfg(feature = "gpu")]
use gguf::Split;

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
/// at ctx_max 64. This is not a band: the node
/// count is a deterministic property of the chain, so the pin is exact and
/// its margin is zero — the derivation is "what the chain enqueues today,
/// after the fuse1 round's two fusions". A silently added launch — a fused
/// pair that stopped fusing, a debug copy left in the chain — keeps every
/// other arm of this gate green (eager still equals replay, ops still equal
/// nodes) and fails only here.
///
/// PIN(2026-09-21, lfold round): 21 → 20. The attention half's two
/// `gemv_q3k_heads` launches (heads 0..7 and 8..15) became one
/// `gemv_q3k_heads_pair` launch over all sixteen, each head still dotting
/// its own activation column — one launch fewer, the same rows, the same
/// arithmetic. Derivation: 21 − 1. FAIL-first held: the same source with
/// this constant still 21 printed `block 0 captures 20 nodes, the pin is
/// 21` while every bit-identity arm of this gate stayed green.
///
/// PIN(2026-09-21, lfold round): 20 → 19. The attention half's two
/// `quantize_q8_1(kqvc_*)` launches became one `quantize_q8_1_pair` whose
/// grid covers both halves of the same buffer, each block running the same
/// per-block body. Derivation: 20 − 1. FAIL-first held the same way
/// (`block 0 captures 19 nodes, the pin is 20`, every other arm green).
/// Both merges carry a value-neutral rollback lever
/// (`StepProbe::split_heads`, `split_kqvc`), so this count is the default
/// path's, not the only one the binary can capture.
///
/// PIN(2026-09-22, fmerge round): 19 → 18. The `kqvc` q8_1 quantization
/// stopped being a launch: the attention launch's last kernel already holds
/// one head's whole latent row per block, so it emits the quantized form as
/// a side output (`flash_latent_q8` on a one-segment cache,
/// `flash_merge_q8` on a split one). Derivation: 19 − 1, the whole
/// `quantize_q8_1(kqvc)` node. FAIL-first held: the same source with this
/// constant still 19 printed `FAIL: block 0 captures 18 nodes, the pin is
/// 19 (NODES_BLOCK0 19 + 0 for the flash merge)` while every bit-identity
/// arm stayed green, the new `fold` arm included. The rollback lever is
/// `StepProbe::split_flash_quant`, which puts the quantize back on its own
/// launch (and is what `split_kqvc` now needs to be set with to have any
/// effect at all).
#[cfg(feature = "gpu")]
const NODES_BLOCK0: usize = 18;
/// Print-only markers for the profile table, not gates: an op touching at
/// least a mebibyte should be paying for bytes, not for its launch, so one
/// that stays under this effective bandwidth is a shape-defect candidate the
/// reader should look at.
#[cfg(feature = "gpu")]
const MARK_BYTES: u64 = 1 << 20;
#[cfg(feature = "gpu")]
const MARK_GBPS: f64 = 100.0;
/// The second axis, for the ops the bandwidth marker above cannot see: an op
/// under `MARK_BYTES` has no bytes to be slow for, so its cost is latency —
/// a lane walking a row with one load in flight, or one thread walking a
/// table. `sync_floor_us` is this harness's own per-op cost, so an op below
/// it is mostly instrument and an op at several times it is spending real
/// device time on almost no traffic. Two floors is where that separation
/// falls on this chain: it names the two ops of the router pair and no
/// other row of the layer-1 table, on the table before this round's kernels
/// and on the table after.
#[cfg(feature = "gpu")]
const MARK_FLOOR_MULT: f64 = 2.0;
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
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p8", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
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
    let file = Split::open(ref_model_path()?)?;
    let man = ref_manifest()?;
    let mut model = Deepseek2Model::load_blocks(&file, ctx_max, 0..profile_layer + 1)?;
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

    // ---- (b2) the folded q8_1 side output is bit-identical to the
    // standalone quantize launch. `kqvc`'s quantization moved inside the
    // attention launch, where a block holds one head's whole latent row;
    // this asserts what that move is only allowed to be — the same bytes,
    // reached by a different launch shape. It is the arm that would catch a
    // scale computed over a different set of values, which changes every
    // byte quantized with it and which no band on `l_out` is sharp enough to
    // see reliably.
    model.set_probe(StepProbe {
        split_flash_quant: true,
        ..StepProbe::default()
    })?;
    let taps_split = model.step_block0_taps(PROMPT[M_TOKENS - 1], (M_TOKENS - 1) as u32)?;
    model.set_probe(StepProbe::default())?;
    let fold_same = taps1.bits_equal(&taps_split);
    println!(
        "fold folded_quant_bit_identical_to_split_launch={fold_same} {}",
        verdict(fold_same)
    );
    if !fold_same {
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
        let time_replays = |launch: &dyn Fn() -> Result<(), GateError>| -> Result<f64, GateError> {
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
                .map_err(|e| -> GateError { Box::new(e) })
        })?;
        let touch_us = time_replays(&|| {
            empty
                .launch(stream)
                .map_err(|e| -> GateError { Box::new(e) })
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
        // Scoped: the stream borrow taken here must end before the layer
        // capture below, which needs `&mut model`.
        let (floor_min, floor_sum, graph_us) = {
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
            (
                floor_min,
                floor_sum,
                t0.elapsed().as_secs_f64() * 1e6 / f64::from(N),
            )
        };

        // The FFN half starts at the layer's own norm: the fused dense four
        // for a layer without a router, the routed ten for a layer with one.
        // The FFN half runs from the layer's own norm to the op that writes
        // the block output. Both ends are named, and the count between them
        // is derived — it used to be a literal (10 routed, 4 dense), which
        // is a restatement of the chain that goes stale the first time two
        // launches inside the half merge, and does so inside `--profile`,
        // the measurement path, rather than in a gate. What is worth
        // asserting is the boundary, not the launch count: the anchor must
        // exist exactly once, and the last op must be the terminal.
        let (anchor, terminal) = match ops.iter().any(|o| o.name == "moe_ffn_norm_quant") {
            true => ("moe_ffn_norm_quant", "moe_combine"),
            false => ("ffn_norm_quant", "ffn_down_add"),
        };
        let anchors = ops.iter().filter(|o| o.name == anchor).count();
        if anchors != 1 {
            return Err(format!(
                "gate_p8: the profile holds {anchors} ops named {anchor}, want exactly one — \
                 the FFN split point is ambiguous"
            )
            .into());
        }
        let ffn_start = ops
            .iter()
            .position(|o| o.name == anchor)
            .ok_or("gate_p8: the profile holds no FFN norm op")?;
        match ops.last() {
            Some(o) if o.name == terminal => {}
            Some(o) => {
                return Err(format!(
                    "gate_p8: the profile's last op is {}, want {terminal} — the {anchor} \
                     shape does not end where the block output is written",
                    o.name
                )
                .into());
            }
            None => return Err("gate_p8: the profile holds no ops".into()),
        }
        let want_ffn_ops = ops.len() - ffn_start;
        println!(
            "prof reps={PROF_REPS} warmup=20 layer={profile_layer} ops={} \
             ffn_shape={anchor}..{terminal} ffn_ops_derived={want_ffn_ops} \
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
            let net = op.us_min - floor_min;
            let mark = match (op.bytes, gbps(op)) {
                (Some(b), Some(g)) if b >= MARK_BYTES && g < MARK_GBPS => " shape_defect_candidate",
                (Some(b), _) if b < MARK_BYTES && net >= MARK_FLOOR_MULT * floor_min => {
                    " latency_defect_candidate"
                }
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
             marker=bytes>={MARK_BYTES}_and_gbps<{MARK_GBPS:.0} \
             marker2=bytes<{MARK_BYTES}_and_net_us>={mark_floor:.3}(={MARK_FLOOR_MULT:.0}x_sync_floor)",
            show((sum_net_us > 0.0).then(|| sum_bytes as f64 / sum_net_us / 1e3)),
            mark_floor = MARK_FLOOR_MULT * floor_min,
        );
        println!(
            "prof split attn_ops={ffn_start} ffn_ops={} attn_us={attn_us:.3} ffn_us={ffn_us:.3} \
             attn_net_us={attn_net_us:.3} ffn_net_us={ffn_net_us:.3}",
            ops.len() - ffn_start
        );

        // The profiled layer, captured and replayed. Every row above is an
        // eager launch + body + one synchronize; the replay is what the same
        // chain costs as graph nodes, which is the form the decode step runs
        // in. The two together say whether a row's net time is device work or
        // the profile's own submit path. Only for a layer above 0: block 0's
        // replay is `graph_replay_us` above, and `capture_layer` captures
        // without the embedding in front, so at layer 0 it would time a
        // shorter chain than the table it is printed beside.
        if profile_layer > 0 {
            let layer_nodes = model.capture_layer(profile_layer)?;
            let stream = model.stages()[0].gpu().stream();
            for _ in 0..2 {
                model.launch_layer_graph(profile_layer)?;
            }
            stream.synchronize()?;
            let mut best = f64::INFINITY;
            let mut worst = 0.0f64;
            let mut sum = 0.0f64;
            const ROUNDS: u32 = 7;
            for _ in 0..ROUNDS {
                let t0 = std::time::Instant::now();
                for _ in 0..N {
                    model.launch_layer_graph(profile_layer)?;
                }
                stream.synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(N);
                best = best.min(us);
                worst = worst.max(us);
                sum += us;
            }
            println!(
                "prof layer_replay layer={profile_layer} nodes={layer_nodes} rounds={ROUNDS} \
                 n_per_round={N} replay_us_min={best:.3} replay_us_mean={:.3} \
                 replay_us_max={worst:.3} eager_sum_net_us={sum_net_us:.3} \
                 eager_sum_us={sum_us:.3} replay_over_eager_net={:.3}",
                sum / f64::from(ROUNDS),
                best / sum_net_us
            );
        }
    }

    // ---- R-1(b/c): the honest per-launch cost of the kernels the profile
    // table flags, free of its per-op synchronize. Lead-only, same lease.
    if std::env::args().any(|a| a == "--bench-kernels") {
        bench_kernels(&model)?;
    }

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_p8 — block-0 step assembled: taps printed against the dump's last \
         token, rerun/replay/second-position bit-identical, inside the structural fence"
    );
    Ok(())
}

/// R-1: per-launch cost of the kernels the profile table flags, measured the
/// two ways that bracket what a per-op row cannot say. `eager` is N launches
/// issued back to back with one synchronize at the end — the larger of host
/// submit rate and device time, with no per-op sync in it. `graph` is the
/// same N launches captured into one graph and replayed — what a node of the
/// step's own graph costs. `touch` (an empty kernel) is the floor of both.
/// The f32 gemv runs at three row counts in one process, so the router's
/// 64-row shape is read against the big shapes on the same instrument.
///
/// The K-quant arms run at the layer's own shapes, and each Q3_K shape runs
/// twice: `q3k_gemv` takes its column count at launch, `q3k_gemv_sel` on a
/// one-expert stack of one slot is the same core over the same rows with
/// that count a compile-time 1. Same body, same accumulation order, same
/// traffic — so the pair prices the runtime column guards on their own,
/// with no kernel change. `_sel` also reads one `sel` word before its first
/// weight load, a round trip `q3k_gemv` does not pay, so the pair's gap is
/// a lower bound on the guards' cost.
///
/// Weight and activation bytes come from [`fill_pattern`], a fixed
/// non-trivial bit pattern. A K-quant core's control flow does not branch on
/// a value, but its f16 scale decode does: `cores::half_to_f32` takes its
/// cheapest arm on a zero, so a zeroed weight buffer makes every arm that
/// runs a decode path read faster than the step it is meant to predict. The
/// pattern is not modelled on any weight distribution — it exists so that no
/// decode arm is skipped, and nothing here asserts on the results.
///
/// The `gap_*` arms are not gemvs: they are the grid-barrier trio
/// (`probe::GapArm`) at the grids a fold would use. Their subtractions give
/// the cooperative launch's price and the barrier's price in the same
/// instrument, and against the same `touch` floor, as the node price.
#[cfg(feature = "gpu")]
fn bench_kernels(model: &Deepseek2Model) -> Result<(), GateError> {
    use bloomery_gpu::probe::{GAP_THREADS, GapArm};
    use bloomery_gpu::{DeviceTensor, GpuError, Graph, Q8Act};
    use cuda_core::{CudaStream, DeviceBuffer};

    /// Launches per burst, and nodes per captured graph.
    const N: usize = 64;
    /// Bursts (or graph replays) per arm — the spread of these is printed.
    const ROUNDS: u32 = 7;
    /// Graph launches per round, so the one synchronize is amortized.
    const GREPS: usize = 4;

    fn burst(
        stream: &CudaStream,
        enq: &mut dyn FnMut(&CudaStream) -> Result<(), GpuError>,
    ) -> Result<(f64, f64, f64), GateError> {
        for _ in 0..N {
            enq(stream)?;
        }
        stream.synchronize()?;
        let (mut lo, mut hi, mut sum) = (f64::INFINITY, 0.0f64, 0.0f64);
        for _ in 0..ROUNDS {
            let t0 = std::time::Instant::now();
            for _ in 0..N {
                enq(stream)?;
            }
            stream.synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
            lo = lo.min(us);
            hi = hi.max(us);
            sum += us;
        }
        Ok((lo, sum / f64::from(ROUNDS), hi))
    }

    fn replay(stream: &CudaStream, g: &Graph) -> Result<(f64, f64, f64), GateError> {
        for _ in 0..2 {
            g.launch(stream)?;
        }
        stream.synchronize()?;
        let (mut lo, mut hi, mut sum) = (f64::INFINITY, 0.0f64, 0.0f64);
        for _ in 0..ROUNDS {
            let t0 = std::time::Instant::now();
            for _ in 0..GREPS {
                g.launch(stream)?;
            }
            stream.synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / (N * GREPS) as f64;
            lo = lo.min(us);
            hi = hi.max(us);
            sum += us;
        }
        Ok((lo, sum / f64::from(ROUNDS), hi))
    }

    let gpu = model.stages()[0].gpu();
    let stream = gpu.stream();
    let probe = bloomery_gpu::probe::Probe::load(gpu.context())?;

    // The f32 activation column every arm reads, and the source the q8
    // scratch below is quantized from: a real spread of exponents and signs
    // rather than a buffer of zeros.
    let x = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(2048))?;
    let mut tbuf = DeviceBuffer::<f32>::zeroed(stream, 32)?;
    let mut probs = DeviceBuffer::<f32>::zeroed(stream, 64)?;
    let mut ids = DeviceBuffer::<u32>::zeroed(stream, 6)?;
    let mut wts = DeviceBuffer::<f32>::zeroed(stream, 6)?;

    println!("bench n_per_burst={N} rounds={ROUNDS} graph_launches_per_round={GREPS} m_cols=1");

    // `--bench-arm <op>`: that one grouped-GEMM arm and nothing else.
    let only = {
        let mut args = std::env::args().skip_while(|a| a != "--bench-arm");
        match args.next() {
            None => None,
            Some(_) => Some(
                args.next()
                    .ok_or("gate_p8: --bench-arm wants an arm name (a `bench op=` value)")?,
            ),
        }
    };
    if let Some(arm) = only.as_deref() {
        return if gemm_arms(gpu, Some(arm))? {
            Ok(())
        } else {
            Err(format!("gate_p8: --bench-arm {arm} names no grouped-GEMM arm").into())
        };
    }

    {
        let mut enq = |s: &CudaStream| probe.enqueue_touch(s, &mut tbuf);
        let e = burst(stream, &mut enq)?;
        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
        let r = replay(stream, &g)?;
        print_arm("touch", 0, 0, g.node_count(), e, r);
    }

    for rows in [64usize, 576, 2048] {
        let w = DeviceTensor::<f32>::zeroed(stream, rows, 2048)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
        let mut enq = |s: &CudaStream| gpu.q8f32().enqueue_f32_gemv(s, &w, &x, 1, &mut y);
        let e = burst(stream, &mut enq)?;
        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
        let r = replay(stream, &g)?;
        // What the launch addresses: the whole weight plus one activation
        // column plus the outputs — the profile's own byte convention.
        let bytes = (rows * 2048 * 4 + 2048 * 4 + rows * 4) as u64;
        print_arm("f32_gemv", rows, bytes, g.node_count(), e, r);
    }

    {
        let sink = gpu.unlabelled_sink();
        let mut enq = |s: &CudaStream| {
            gpu.router()
                .enqueue_router_topk(s, &x, 1, 1.0, &mut probs, &mut ids, &mut wts, sink)
        };
        let e = burst(stream, &mut enq)?;
        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
        let r = replay(stream, &g)?;
        print_arm("router_topk", 1, 560, g.node_count(), e, r);
    }

    // Q3_K: attn_kv_a_mqa's 576 rows, attn_q's 3072, and the MoE expert
    // stack's six-expert width 6*1408 — the last one past this card's 6 MiB
    // L2, where the smaller two fit. Each shape paired with the
    // constant-folded `_sel` launch over the same rows.
    for (rows, k) in [(576usize, 2048usize), (3072, 2048), (8448, 2048)] {
        let n_sb = k / 256;
        let mut act = Q8Act::with_k(stream, 1, k)?;
        gpu.enqueue_quantize_q8_1(&x, &mut act)?;
        let cols = 110 * n_sb / 4;
        let w = DeviceTensor::<u32>::upload(stream, &fill_pattern(rows * cols), rows, cols)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
        // The profile's byte convention: the weight rows the launch reads,
        // the activation buffers it addresses, and the outputs it writes.
        let bytes =
            (rows * 110 * n_sb + 8 * 64 * n_sb.div_ceil(2) + 4 * 2 * n_sb + 4 * rows) as u64;
        {
            let mut enq = |_s: &CudaStream| gpu.enqueue_gemv_q3k(&w, &act, &mut y);
            let e = burst(stream, &mut enq)?;
            let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
            let r = replay(stream, &g)?;
            print_arm("q3k_gemv", rows, bytes, g.node_count(), e, r);
        }
        {
            let sel = DeviceBuffer::<u32>::zeroed(stream, 1)?;
            let mut enq =
                |_s: &CudaStream| gpu.enqueue_gemv_q3k_sel(&w, &act, &sel, 1, rows, &mut y);
            let e = burst(stream, &mut enq)?;
            let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
            let r = replay(stream, &g)?;
            print_arm("q3k_gemv_sel", rows, bytes, g.node_count(), e, r);
        }
    }

    // Q4_K: attn_output's 2048x2048 and shexp_down's 2048x2816 (n_sb 11,
    // the odd count whose last iteration the per-lane `sbp` guard trims),
    // plus a stack-sized 8192 rows past L2. No constant-folded twin exists
    // for this core — the guard lever itself is the measurement there.
    for (rows, k) in [(2048usize, 2048usize), (2048, 2816), (8192, 2048)] {
        let n_sb = k / 256;
        let mut act = Q8Act::with_k(stream, 1, k)?;
        // Bound, not a temporary: the quantize is asynchronous, so the
        // source has to outlive the launch rather than the statement.
        let xk = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(k))?;
        gpu.enqueue_quantize_q8_1(&xk, &mut act)?;
        let cols = 36 * n_sb;
        let w = DeviceTensor::<u32>::upload(stream, &fill_pattern(rows * cols), rows, cols)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
        let bytes = (rows * 144 * n_sb
            + 4 * 256 * n_sb.div_ceil(4)
            + 4 * 8 * n_sb
            + 4 * 2 * n_sb
            + 4 * rows) as u64;
        let mut enq = |_s: &CudaStream| gpu.enqueue_gemv_q4k(&w, &act, &mut y);
        let e = burst(stream, &mut enq)?;
        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
        let r = replay(stream, &g)?;
        print_arm(&format!("q4k_gemv_k{k}"), rows, bytes, g.node_count(), e, r);
        // The standing control for `fill_pattern`: the same shape over a
        // zeroed weight, which is what this bench fed every arm before the
        // pattern fill. It is here so the gap stays visible rather than
        // becoming a claim in a log — a zero f16 takes `half_to_f32`'s
        // cheapest arm, so a decode-path arm reads faster than the step it
        // is meant to predict.
        if (rows, k) == (2048, 2048) {
            let wz = DeviceTensor::<u32>::zeroed(stream, rows, cols)?;
            let mut enq = |_s: &CudaStream| gpu.enqueue_gemv_q4k(&wz, &act, &mut y);
            let e = burst(stream, &mut enq)?;
            let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
            let r = replay(stream, &g)?;
            print_arm(
                &format!("q4k_gemv_k{k}_zerow"),
                rows,
                bytes,
                g.node_count(),
                e,
                r,
            );
        }
    }

    // Q6_K at the lm_head's own width. The head launches once per step, not
    // once per layer, and it is the one K-quant gemv the per-op table never
    // shows, so its shape is priced here: `q6k_gemv` takes its column count
    // at launch like the two this round fixed, and nothing has folded it.
    {
        let (rows, k) = (102_400usize, 2048usize);
        let n_sb = k / 256;
        let mut act = Q8Act::with_k(stream, 1, k)?;
        gpu.enqueue_quantize_q8_1(&x, &mut act)?;
        let cols = 210 * n_sb / 4;
        let w = DeviceTensor::<u32>::upload(stream, &fill_pattern(rows * cols), rows, cols)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
        let bytes =
            (rows * 210 * n_sb + 4 * 128 * n_sb.div_ceil(2) + 4 * 2 * n_sb + 4 * rows) as u64;
        let mut enq = |_s: &CudaStream| gpu.enqueue_gemv_q6k(&w, &act, &mut y);
        let e = burst(stream, &mut enq)?;
        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
        let r = replay(stream, &g)?;
        print_arm("q6k_gemv_lm_head", rows, bytes, g.node_count(), e, r);
    }

    gemm_arms(gpu, None)?;
    // The grouped int8 GEMM (`bloomery_gpu::gemm`): Qwen3-30B-A3B's routed
    // gate shape — 128 experts of 768 x 2048 Q4_K, top-8, every slot reading
    // its token's column — and the dense 2048 x 2048 Q4_K case, each at T
    // tokens up to the largest ubatch (4096: 32,768 routed slots). The route
    // table is built once per T and its launch priced on its own; the GEMM
    // arm is the GEMM alone. Beside the usual row: the
    // arithmetic rate `2 * slots * rows * K` over the graph minimum, and that
    // rate against the card's int8 dense tensor peak. `only` keeps the one
    // arm of that name; the return says whether any arm ran.
    fn gemm_arms(gpu: &bloomery_gpu::Gpu, only: Option<&str>) -> Result<bool, GateError> {
        use bloomery_gpu::gemm::{GemmAct, GemmInput, GemmKernels, GemmRoute, GemmWeight};
        let stream = gpu.stream();
        let mut ran = false;
        let gk = GemmKernels::load(gpu.context())?;
        let name = gpu.device_name()?;
        let peak = int8_peak_tops(&name);
        println!(
            "bench gemm device={name:?} int8_dense_peak_tops={}",
            peak.map_or_else(|| "?".to_string(), |p| format!("{p}"))
        );
        let sink = gpu.unlabelled_sink();
        for (label, n_exp, top_k, rows, k) in [
            ("moe", 128usize, 8usize, 768usize, 2048usize),
            ("dense", 1, 1, 2048, 2048),
        ] {
            if only.is_some_and(|o| !o.starts_with(&format!("gemm_q4k_{label}_t"))) {
                continue;
            }
            let n_rows = n_exp * rows;
            let cols = 36 * k / 256;
            let w =
                DeviceTensor::<u32>::upload(stream, &fill_pattern(n_rows * cols), n_rows, cols)?;
            for t in [16usize, 64, 256, 512, 1024, 2048, 4096] {
                let op = format!("gemm_q4k_{label}_t{t}");
                if only.is_some_and(|o| o != op) {
                    continue;
                }
                ran = true;
                let n_slots = t * top_k;
                let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(t * k))?;
                let mut act = GemmAct::new(stream, t, k)?;
                gpu.enqueue_quantize_gemm(&xs, t, &mut act, sink)?;
                let ids = gemm_ids(t, top_k, n_exp);
                let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                let input = if n_exp == 1 {
                    gk.enqueue_route_dense(stream, n_slots, &mut route, sink)?;
                    GemmInput::PerSlot
                } else {
                    let mut enq =
                        |s: &CudaStream| gk.enqueue_route(s, &ids_d, n_slots, &mut route, sink);
                    let e = burst(stream, &mut enq)?;
                    let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                    let r = replay(stream, &g)?;
                    print_arm(
                        &format!("gemm_route_t{t}"),
                        n_slots,
                        4 * n_slots as u64,
                        g.node_count(),
                        e,
                        r,
                    );
                    GemmInput::Shared { top_k }
                };
                let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                let mut enq = |s: &CudaStream| {
                    gk.enqueue_gemm(s, GemmWeight::Q4K, &w, rows, &act, &route, input, &mut y)
                };
                let e = burst(stream, &mut enq)?;
                let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                let r = replay(stream, &g)?;
                let mut hit = vec![false; n_exp];
                for &i in &ids {
                    hit[i as usize] = true;
                }
                let experts = hit.iter().filter(|&&h| h).count();
                let n_sb = k / 256;
                // The weights of the experts a slot picked, the activation
                // buffers the launch reads, and the outputs it writes.
                let bytes = (experts * rows * 144 * n_sb
                    + t * (4 * 128 * n_sb.div_ceil(2) + 4 * 8 * n_sb + 4 * 2 * n_sb)
                    + 4 * n_slots * rows) as u64;
                print_arm(&op, rows, bytes, g.node_count(), e, r);
                let ops = 2.0 * (n_slots * rows * k) as f64;
                let tops = ops / r.0 / 1e6;
                println!(
                    "bench op={op} slots={n_slots} experts={experts} ops={ops:.0} tops_graph_min={tops:.2} \
                     pct_int8_peak={}",
                    peak.map_or_else(|| "?".to_string(), |p| format!("{:.1}", 100.0 * tops / p))
                );
            }
        }
        Ok(ran)
    }

    // The grid-barrier trio. The grids are the ones the three stopped folds
    // would have launched at — 16 and 22 of this card's 82 SMs — plus one
    // block per SM and two, so the reader can see whether either price is a
    // function of the grid at all. Arm order inside a grid is fixed
    // (plain, coop, coop_sync) and every arm is its own captured graph, so
    // the subtraction is between two graph replays of the same shape.
    for blocks in [16u32, 22, 82, 164] {
        let mut y =
            DeviceBuffer::<f32>::zeroed(stream, 2 * blocks as usize * GAP_THREADS as usize)?;
        for arm in [GapArm::Plain, GapArm::Coop, GapArm::CoopSync] {
            let mut enq = |s: &CudaStream| probe.enqueue_gap(s, arm, blocks, &mut y);
            let e = burst(stream, &mut enq)?;
            let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
            let r = replay(stream, &g)?;
            // Two stores per thread, both halves written; the phase-B half
            // holding 3.0 everywhere is what says the barrier arm got past
            // its `grid::sync`.
            let got = y.to_host_vec(stream)?;
            let n = blocks as usize * GAP_THREADS as usize;
            let phases_ok =
                got[..n].iter().all(|&v| v == 1.0) && got[n..].iter().all(|&v| v == 3.0);
            print_arm(
                &format!("{}_b{blocks}_ok{}", arm.name(), u8::from(phases_ok)),
                blocks as usize,
                8 * n as u64,
                g.node_count(),
                e,
                r,
            );
            if !phases_ok {
                return Err(format!(
                    "gate_p8 bench: {} at {blocks} blocks did not write both phases",
                    arm.name()
                )
                .into());
            }
        }
    }
    Ok(())
}

/// A fixed non-trivial u32 pattern — a multiplicative hash of the index,
/// which spreads bits through every byte and every f16 field a K-quant
/// super-block carries. Not a model of any weight distribution: its one job
/// is that no value-dependent decode path (`half_to_f32`'s zero arm above
/// all) is skipped by a buffer of zeros.
#[cfg(feature = "gpu")]
fn fill_pattern(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9)
        .collect()
}

/// The int8 dense tensor peak of the card named `name`, in TOPS — GA102
/// whitepaper figures, the denominator the prefill literature report uses —
/// or `None` for a card not listed.
#[cfg(feature = "gpu")]
fn int8_peak_tops(name: &str) -> Option<f64> {
    if name.contains("A6000") {
        Some(309.7)
    } else if name.contains("3090") {
        Some(284.0)
    } else {
        None
    }
}

/// Expert ids for `t` tokens of `top_k` distinct experts out of `n_exp`,
/// drawn from a fixed LCG: the bench's uniform routing.
#[cfg(feature = "gpu")]
fn gemm_ids(t: usize, top_k: usize, n_exp: usize) -> Vec<u32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut ids = Vec::with_capacity(t * top_k);
    for _ in 0..t {
        let mut pick: Vec<u32> = Vec::with_capacity(top_k);
        while pick.len() < top_k {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let e = ((s >> 33) % n_exp as u64) as u32;
            if !pick.contains(&e) {
                pick.push(e);
            }
        }
        ids.extend(pick);
    }
    ids
}

/// The same pattern as f32 activations, mapped into roughly ±1 so a
/// quantization of it exercises rounding rather than saturation.
#[cfg(feature = "gpu")]
fn fill_pattern_f32(n: usize) -> Vec<f32> {
    fill_pattern(n)
        .into_iter()
        .map(|w| (w >> 8) as f32 / 8_388_608.0 - 1.0)
        .collect()
}

/// One `bench` row: the eager burst and the graph replay of the same launch,
/// each as min/mean/max over the arm's rounds, with the graph GB/s beside it.
#[cfg(feature = "gpu")]
fn print_arm(
    name: &str,
    rows: usize,
    bytes: u64,
    nodes: usize,
    eager: (f64, f64, f64),
    graph: (f64, f64, f64),
) {
    let gbps = match (bytes > 0, graph.0 > 0.0) {
        (true, true) => format!("{:.1}", bytes as f64 / graph.0 / 1e3),
        _ => "?".to_string(),
    };
    println!(
        "bench op={name} rows={rows} nodes={nodes} bytes={bytes} \
         eager_us_min={:.3} eager_us_mean={:.3} eager_us_max={:.3} \
         graph_us_min={:.3} graph_us_mean={:.3} graph_us_max={:.3} graph_gbps={gbps}",
        eager.0, eager.1, eager.2, graph.0, graph.1, graph.2
    );
}

/// One `<flag> <u32>` pair off the command line: `--profile-pos` names the
/// position the `--profile` table is taken at, `--profile-ctx` the cache
/// height to allocate. Absent, the profile stays at the dump's last position
/// on the gate's own cache height.
#[cfg(feature = "gpu")]
fn parse_u32_flag(
    args: impl Iterator<Item = String>,
    flag: &str,
) -> Result<Option<u32>, GateError> {
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
