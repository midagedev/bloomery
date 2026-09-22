//! GPU gate for package P8b (docs/gpu-design.md work package): the assembled
//! MoE decode step of layer 1 (`GpuModel::step_layer_taps`), m = 1, against
//! the ik CUDA oracle dump. A lone layer has no embedding in front of it, so
//! its input residual is the oracle's own `l_out-0` column and its KV cache
//! is seeded from the oracle's `kv_cache-1` rows, exactly as `gate_p8` seeds
//! layer 0 — the tap table then isolates this layer's own ops. The tap rels
//! are PRINTED, never banded; the lead pins the MoE block bands from this
//! table. What is asserted:
//! - (a) every tap finite, and a structural fence per tap (`rel <= 0.25`,
//!   the same fence `gate_p8` carries: an order of magnitude above the
//!   measured rels and below the O(0.5..1) a wrong concat or a wrong expert
//!   produces). A violation here is a layout or routing defect, not noise;
//! - (b) the router's expert ids integer-equal to the dump's
//!   `ffn_moe_topk-1` last-token ids AND to the host routing reference, in
//!   rank order, with the router weights within `ROUTER_BAND` of it;
//! - (c) an eager rerun bit-identical;
//! - (d) the captured graph replays bit-identical to eager, and the node
//!   count equals its pin (`NODES_LAYER1`);
//! - (e) the SAME graph at a second position (pos 4, n_keys 5) reproducing
//!   an eager run there bit for bit — what the device-side
//!   `pos_buf`/`n_keys_buf` buy;
//! - (f) the SAME graph on a DIFFERENT input vector that routes to at least
//!   one different expert, still bit-identical to its own eager run — what
//!   the device-resident `sel` buys. A captured graph that froze the routing
//!   passes (d) and (e) and fails here. How many slots that probe moves is
//!   pinned too (`PROBE_SLOTS_CHANGED`): it is a property of the dump and
//!   the router, not of a sample.
//!
//! Two taps of the MoE block are not exposed: `moe_combine` folds the
//! weighted expert sum, the shared expert and the residual into one store,
//! so neither `ffn_moe_out-1` nor `ffn_out-1` exists as a buffer. `l_out-1`
//! carries that span; `ffn_moe_out-1` is additionally compared against the
//! host recombination of the engine's own `expert_down` and router weights,
//! which is the operand pair the fused store consumes — the table marks that
//! row `recombined`.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p8b: built without the `gpu` feature; see `just gate-gpu-p8b`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::GpuModel;
#[cfg(feature = "gpu")]
use bloomery_gpu::model::StepProbe;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::block::{self, BlockKind, M_TOKENS, TapKind, TapResult};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, find_ref_row, find_ref_row_in, max_rel_err, open_model, ref_dir, ref_manifest,
    ref_tensor_logical_in, route_ref, topk_ids_logical, verdict, widened_f16_bits,
};

/// The layer this gate assembles — the first MoE block of the model.
#[cfg(feature = "gpu")]
const LAYER: usize = 1;
/// Cache rows allocated for the gate's runs (the dump uses 6).
#[cfg(feature = "gpu")]
const CTX_MAX: usize = 64;
/// Structural fence — see the module doc. Not a band.
#[cfg(feature = "gpu")]
const FENCE: f32 = 0.25;
/// Router weights against `route_ref` on the same logits: the device `exp`
/// and the host libm differ by a few ulp, everything else in that chain is
/// bit-mirrored (`gate_moe_fused`'s constant). Ids are exact.
#[cfg(feature = "gpu")]
const ROUTER_BAND: f32 = 1e-5;
/// PIN(2026-09-21): the MoE layer's chain captures exactly this many graph
/// nodes — sixteen attention ops (block 0's seventeen without the embed) and
/// ten routed FFN ops, on this gate's one-segment cache. Measured by this
/// gate's own `graph graph_nodes=26` line, identical on three consecutive
/// runs. Not a band: the count is a deterministic property
/// of the chain, so the pin is exact and its margin is zero. A silently
/// added launch keeps the bit-identity arms green and fails only here.
///
/// PIN(2026-09-21, lfold round): 26 → 25. The attention half's two
/// `gemv_q3k_heads` launches became one `gemv_q3k_heads_pair` over all
/// sixteen heads, so the sixteen attention ops are fifteen. Derivation:
/// 15 + 10 routed FFN ops = 25.
///
/// PIN(2026-09-21, lfold round): 25 → 24. The attention half's two
/// `quantize_q8_1(kqvc_*)` launches became one whose grid covers both
/// halves. Derivation: 14 attention ops + 10 routed FFN ops = 24.
///
/// PIN(2026-09-22, fmerge round): 24 → 22, one launch from each half. The
/// attention half's `quantize_q8_1(kqvc)` is now a side output of the
/// attention launch itself (13 attention ops). The FFN half's
/// `moe_expert_quantize_q8` and `shexp_quantize_q8_1` became one
/// `moe_quantize_pair` carrying both geometries, which the shared expert's
/// `gate_up_swiglu` moving earlier makes adjacent (9 routed FFN ops).
/// Derivation: 13 + 9 = 22. FAIL-first held: the same source with this
/// constant still 24 printed `FAIL: layer 1 captures 22 nodes, the pin is
/// 24 — a launch was added or removed` while every bit-identity arm stayed
/// green, including the two new `merge` arms that compare each shape
/// against the launches it replaced. Both carry a value-neutral rollback
/// lever (`StepProbe::split_flash_quant`, `split_moe_quant`).
#[cfg(feature = "gpu")]
const NODES_LAYER1: usize = 22;
/// PIN(2026-09-21): slots the routing probe's input moves. The probe is the
/// first `l_out-0` column that routes differently from the last token's, so
/// both id vectors are fixed by the dump and the router alone: `[5, 38, 8,
/// 20, 26, 27]` against `[57, 60, 56, 8, 26, 43]` differ in five of six
/// slots. Measured identical on three consecutive runs, which is what makes
/// it pinnable — nothing here samples. Margin zero for the same reason as the
/// node pin. What it catches: a routing change that still leaves (f) green
/// because the replay follows it. A legitimate change of the dump or of the
/// router's tie-breaking re-pins this line with its own measurement.
#[cfg(feature = "gpu")]
const PROBE_SLOTS_CHANGED: usize = 5;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p8b", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let mut ok = true;
    let gguf = open_model()?;
    let man = ref_manifest()?;
    let dir = ref_dir();
    let mut model = GpuModel::load_blocks(&gguf, CTX_MAX, LAYER..LAYER + 1)?;
    println!(
        "resident stage_bytes={} ctx_max={CTX_MAX} m=1 layer={LAYER}",
        model.stages()[0].resident_bytes()
    );

    // The layer's input residual is the previous block's output: the dump's
    // own `l_out-0`, one column per token.
    let lo0_row = find_ref_row_in(&dir, &man, "l_out-0", 0)?;
    if lo0_row.ty != "f32" || lo0_row.op != "ADD" || lo0_row.ne[1] != M_TOKENS as u64 {
        return Err(format!(
            "gate_p8b: l_out-0 is {} {} {:?}, want f32 ADD with {M_TOKENS} token columns",
            lo0_row.op, lo0_row.ty, lo0_row.ne
        )
        .into());
    }
    let hidden = lo0_row.ne[0] as usize;
    let l_out0 = ref_tensor_logical_in(&dir, lo0_row)?;
    let column = |t: usize| l_out0[t * hidden..(t + 1) * hidden].to_vec();

    // This layer's own cache rows for tokens 0..4, the way gate_p8 seeds
    // layer 0's: the last token's attention then reads exactly the keys ik's
    // did, and the tap table measures this step's ops alone.
    let ik_cache = find_ref_row(&man, &format!("kv_cache-{LAYER}"), 0)?;
    if ik_cache.ty != "f16" || ik_cache.op != "VIEW" || ik_cache.ne != [576, 256, 1, 1] {
        return Err(format!(
            "gate_p8b: kv_cache-{LAYER} is {} {} {:?}, want VIEW f16 [576, 256]",
            ik_cache.op, ik_cache.ty, ik_cache.ne
        )
        .into());
    }

    // ---- eager run at the last position, tap table, finiteness
    let last = M_TOKENS - 1;
    let seed5 = widened_f16_bits(ik_cache, last)?;
    model.seed_layer_cache(LAYER, &seed5)?;
    let taps1 = model.step_layer_taps(LAYER, &column(last), last as u32)?;
    if let Some(name) = taps1.non_finite() {
        eprintln!("FAIL: tap {name} holds a non-finite value");
        ok = false;
    }

    // The fused combine's operands, recombined the way its store does —
    // `Σ_s w[s] · down[s*rows + d]` in slot order — so the dump's
    // `ffn_moe_out-1` still has something to be compared against.
    let moe_out_recombined = recombine(&taps1.expert_down, &taps1.moe_weights, hidden)?;

    let mut results: Vec<TapResult> = Vec::new();
    for tap in block::taps(BlockKind::Moe, LAYER) {
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
            TapKind::FfnNorm => &taps1.ffn_norm,
            TapKind::FfnMoeLogits => &taps1.moe_logits,
            TapKind::FfnMoeWeights => &taps1.moe_weights,
            TapKind::FfnMoeOut => &moe_out_recombined,
            TapKind::FfnShexp => &taps1.ffn_shexp,
            TapKind::LOut => &taps1.l_out,
            // `moe_combine` writes one store; the moe+shexp sum before the
            // residual never exists as a buffer.
            TapKind::FfnOut => {
                println!(
                    "tap {} op={} unexposed=moe_combine covered_by=l_out-{LAYER}",
                    tap.name(),
                    tap.op
                );
                continue;
            }
            _ => unreachable!("the MoE tap list holds only the kinds above"),
        };
        if tap.kind == TapKind::FfnMoeOut {
            println!(
                "tap {} op={} unexposed=moe_combine compared=recombined(expert_down,moe_weights)",
                tap.name(),
                tap.op
            );
        }
        let row = find_ref_row_in(&dir, &man, &tap.name(), tap.occurrence)?;
        block::check_row(row, &tap, M_TOKENS)?;
        let ref_all = ref_tensor_logical_in(&dir, row)?;
        let per = tap.kind.per_token();
        let ref_last = &ref_all[last * per..M_TOKENS * per];
        if got.len() != per {
            return Err(format!(
                "gate_p8b: {} got {} values, the tap's token column is {per}",
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
        "moe_layer_step",
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

    // ---- (b) routing: ids integer-equal to the dump AND to route_ref, in
    // rank order; weights inside the router band of route_ref.
    let topk_row = find_ref_row(&man, &format!("ffn_moe_topk-{LAYER}"), 0)?;
    let n_used = taps1.moe_ids.len();
    if topk_row.ty != "i32"
        || topk_row.op != "VIEW"
        || topk_row.ne != [n_used as u64, M_TOKENS as u64, 1, 1]
    {
        return Err(format!(
            "gate_p8b: ffn_moe_topk-{LAYER} is {} {} {:?}, want i32 VIEW [{n_used}, {M_TOKENS}]",
            topk_row.ty, topk_row.op, topk_row.ne
        )
        .into());
    }
    let ik_ids = topk_ids_logical(topk_row)?;
    // The router weight multiplier, read by the engine's own MoE metadata
    // reader so the gate cannot drift from what the step applies.
    let scale = model::moe::Meta::read(&gguf)?.scale;
    let (_, ids_ref, w_ref) = route_ref(&taps1.moe_logits, 1, scale)?;
    let ours: Vec<i32> = taps1.moe_ids.iter().map(|&e| e as i32).collect();
    let ik_last = &ik_ids[last * n_used..M_TOKENS * n_used];
    let ids_match_dump = ours == ik_last;
    let ids_match_ref = ours == ids_ref;
    let weights_err = max_rel_err(&taps1.moe_weights, &w_ref)?;
    let router_ok = ids_match_dump && ids_match_ref && weights_err <= ROUTER_BAND;
    println!(
        "router scale={scale} n_used={n_used} ids={ours:?} dump_ids={ik_last:?} \
         ids_equal_dump={ids_match_dump} ids_equal_route_ref={ids_match_ref} \
         weights_err={weights_err:.3e} band={ROUTER_BAND:.0e} {}",
        verdict(router_ok)
    );
    if !router_ok {
        ok = false;
    }

    // ---- (c) eager rerun bit-identical
    model.seed_layer_cache(LAYER, &seed5)?;
    let taps2 = model.step_layer_taps(LAYER, &column(last), last as u32)?;
    let rerun_same = taps1.bits_equal(&taps2);
    println!(
        "rerun eager_bit_identical={rerun_same} {}",
        verdict(rerun_same)
    );
    if !rerun_same {
        ok = false;
    }

    // ---- (c2) the two merged launch shapes of this layer are bit-identical
    // to the launch shapes they replaced. `split_flash_quant` puts the
    // `kqvc` quantization back on a launch of its own instead of riding
    // inside the attention launch; `split_moe_quant` splits the MoE half's
    // one mixed-geometry quantize back into its two. Both are launch moves
    // that may not move a value, and this is the arm that says so — a scale
    // reduced over a different set of values changes every byte quantized
    // with it, which no band on `l_out` sees reliably.
    for (what, probe) in [
        (
            "split_flash_quant",
            StepProbe {
                split_flash_quant: true,
                ..StepProbe::default()
            },
        ),
        (
            "split_moe_quant",
            StepProbe {
                split_moe_quant: true,
                ..StepProbe::default()
            },
        ),
    ] {
        model.set_probe(probe)?;
        model.seed_layer_cache(LAYER, &seed5)?;
        let taps_split = model.step_layer_taps(LAYER, &column(last), last as u32)?;
        model.set_probe(StepProbe::default())?;
        let same = taps1.bits_equal(&taps_split);
        println!("merge {what}_bit_identical={same} {}", verdict(same));
        if !same {
            ok = false;
        }
    }

    // ---- (d) captured graph: replay bit-identical to eager
    model.seed_layer_cache(LAYER, &seed5)?;
    let nodes = model.capture_layer(LAYER)?;
    model.seed_layer_cache(LAYER, &seed5)?;
    model.replay_layer(LAYER, &column(last), last as u32)?;
    let taps3 = model.layer_taps(LAYER)?;
    let replay_same = taps3.bits_equal(&taps1);
    let nodes_pinned = nodes == NODES_LAYER1;
    println!(
        "graph graph_nodes={nodes} pinned_nodes={NODES_LAYER1} nodes_equal_pin={nodes_pinned} \
         eager_vs_replay_bit_identical={replay_same} {}",
        verdict(replay_same && nodes_pinned)
    );
    if !replay_same {
        ok = false;
    }
    if !nodes_pinned {
        eprintln!(
            "FAIL: layer {LAYER} captures {nodes} nodes, the pin is {NODES_LAYER1} — a launch \
             was added or removed"
        );
        ok = false;
    }

    // ---- (e) the same graph at a second position: reseed tokens 0..3,
    // pos 4 / n_keys 5; the replay must equal a fresh eager run there.
    {
        let pos = last - 1;
        let seed4 = widened_f16_bits(ik_cache, pos)?;
        model.seed_layer_cache(LAYER, &seed4)?;
        let eager = model.step_layer_taps(LAYER, &column(pos), pos as u32)?;
        model.seed_layer_cache(LAYER, &seed4)?;
        model.replay_layer(LAYER, &column(pos), pos as u32)?;
        let replayed = model.layer_taps(LAYER)?;
        let same = eager.bits_equal(&replayed);
        println!(
            "second_pos pos={pos} n_keys={} replay_bit_identical_to_eager={same} {}",
            pos + 1,
            verdict(same)
        );
        if !same {
            ok = false;
        }
    }

    // ---- (f) the same graph on an input that routes differently. The
    // candidate inputs are the other `l_out-0` columns; the first one whose
    // eager routing differs from the last token's is the probe. Everything
    // else about the run is held fixed (same position, same seeded cache),
    // so the only thing that moved is the routing.
    {
        let mut probe: Option<(usize, Vec<u32>)> = None;
        for t in 0..last {
            model.seed_layer_cache(LAYER, &seed5)?;
            let taps = model.step_layer_taps(LAYER, &column(t), last as u32)?;
            if taps.moe_ids != taps1.moe_ids {
                probe = Some((t, taps.moe_ids.clone()));
                break;
            }
        }
        let Some((t, probe_ids)) = probe else {
            return Err(format!(
                "gate_p8b: none of the l_out-0 columns 0..{last} routes differently from the \
                 last token's {:?} — (f) has no probe on this dump",
                taps1.moe_ids
            )
            .into());
        };
        let changed = probe_ids
            .iter()
            .zip(&taps1.moe_ids)
            .filter(|(a, b)| a != b)
            .count();
        model.seed_layer_cache(LAYER, &seed5)?;
        let eager = model.step_layer_taps(LAYER, &column(t), last as u32)?;
        model.seed_layer_cache(LAYER, &seed5)?;
        model.replay_layer(LAYER, &column(t), last as u32)?;
        let replayed = model.layer_taps(LAYER)?;
        let same = eager.bits_equal(&replayed);
        let routed_on_device = changed > 0 && replayed.moe_ids == probe_ids;
        let dump_ids = &ik_ids[t * n_used..(t + 1) * n_used];
        let dump_agrees = eager
            .moe_ids
            .iter()
            .map(|&e| e as i32)
            .eq(dump_ids.iter().copied());
        let slots_pinned = changed == PROBE_SLOTS_CHANGED;
        println!(
            "routing_probe src=l_out-0[token_{t}] ids={:?} base_ids={:?} slots_changed={changed} \
             pinned_slots_changed={PROBE_SLOTS_CHANGED} slots_equal_pin={slots_pinned} \
             replay_bit_identical_to_eager={same} replay_ids_follow_router={routed_on_device} \
             dump_ids={dump_ids:?} eager_ids_equal_dump={dump_agrees} (dump_ids printed, not \
             asserted) {}",
            probe_ids,
            taps1.moe_ids,
            verdict(same && routed_on_device && slots_pinned)
        );
        if !(same && routed_on_device) {
            ok = false;
        }
        if !slots_pinned {
            eprintln!(
                "FAIL: the routing probe moves {changed} slots, the pin is \
                 {PROBE_SLOTS_CHANGED} — the routing of this dump's columns changed"
            );
            ok = false;
        }
    }

    if !ok {
        eprintln!("FAILED: gate_p8b");
        std::process::exit(1);
    }
    println!(
        "PASSED: gate_p8b — MoE layer {LAYER} step assembled: taps printed against the dump's \
         last token, routing exact, rerun/replay/second-position/second-routing bit-identical, \
         inside the structural fence"
    );
    Ok(())
}

/// `Σ_s w[s] · down[s*rows + d]` over the slots, in slot order — the sum
/// `moe_combine` folds into its store, recomputed on the host from the
/// engine's own operands.
#[cfg(feature = "gpu")]
fn recombine(down: &[f32], w: &[f32], rows: usize) -> Result<Vec<f32>, GateError> {
    if w.is_empty() || down.len() != w.len() * rows {
        return Err(format!(
            "gate_p8b: recombine: down {} values for {} slots of {rows}",
            down.len(),
            w.len()
        )
        .into());
    }
    let mut y = vec![0.0f32; rows];
    for (s, &ws) in w.iter().enumerate() {
        for (d, o) in y.iter_mut().enumerate() {
            *o += ws * down[s * rows + d];
        }
    }
    Ok(y)
}
