//! The qwen3moe end-to-end gate: the whole chain — 48 layers, the head and
//! the argmax — on one card, against ik's CPU oracle and its greedy answers.
//!
//! Every prompt runs from position 0 (`GpuModel::reset`), its tokens fed one
//! position at a time through the decode path: this chain has no prefill
//! kernel, and the greedy reference was written the same way
//! (`argmax_ref --step-prefill`).
//!
//! What is asserted:
//! - (s) structure: the captured step holds [`NODES_CHAIN`] nodes, 48 of
//!   them memcpy (the residual crossing each layer boundary) and none a host
//!   node.
//! - (f) teacher-forced, per layer: each layer run alone on ik's own input
//!   row (`inp_embd` for layer 0, `l_out-(L−1)` after) at each of the
//!   oracle's five positions. The attention half against ik's `attn_out-L`
//!   (`‖(ffn_inp − x_in) − attn_out‖ / ‖attn_out‖`), and the FFN half run
//!   alone on ik's own FFN input `x_in + attn_out` against ik's
//!   `routed_out-L`, measured on the magnitude of the routed sum's terms
//!   (`Σ_s |w_s · down_s|`, which the eight terms' cancellation cannot
//!   shrink). Each half's error over the gap between the two sides' 8-bit
//!   inputs to it must stay within [`RATIO_BAND`]. A site where the router
//!   picks an expert ik does not is counted and printed, not judged.
//! - (c) free-running: the five tokens through the whole chain, every
//!   layer's `l_out` against ik's, `‖ours − ik‖ / ‖ik‖` within
//!   [`FREE_BAND`]; the last position's logits against ik's
//!   `result_output`: the same argmax, the relative distance printed.
//! - (g) greedy: prompts `0..PROMPTS` of `tools/ref/prompts.tsv` under this
//!   model's tokenizer (`just ik-greedy-qwen3moe` writes them and ik's
//!   continuations), [`GEN`] tokens each in graph mode: no prompt diverges
//!   from ik at a position where ik's own top1-top2 margin is at or above
//!   [`MARGIN_FLOOR`] — `gpu_gates::prompts::compare_greedy`'s classes, the
//!   V2-Lite gate's criterion; later divergences and near ties are printed.
//! - (r) graph replay equals the eager body: the same prompts in eager mode
//!   give the same tokens, and the last step's logits are bit-identical.
//!
//! The flash pass is read once per process (`BLOOMERY_GQA_MMA`), so each
//! pass is its own run; the `load` line names it.
//!
//! `--ppl TAG` instead scores the chain against ik's KL-divergence base file
//! `$BLOOMERY_DATA/ikppl/TAG.kld` (`tools/ref/ik-ppl.sh --kld-base`): chunk by
//! chunk from a reset, one step per id, at every scored position our NLL,
//! ik's, KL(ik‖ours) and the top-1 agreement. Printed, not judged.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_e2e: built without the `gpu` feature; see `just gate-gpu-qwen3moe-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Qwen3moeModel;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::kld::KldBase;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::prompts::{GreedyClass, compare_greedy, read_greedy, read_prompts};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, checks_failed, data_dir, ik_q8_2, open_split,
        q8_1_dequant, ref_ints, ref_tensor_logical_in, topk_ids_logical_within, verdict,
    };
    use cuda_core::sys;
    use model::arch::Arch;
    use std::path::PathBuf;
    use std::time::Instant;

    /// Generated tokens per prompt: the greedy files' width.
    const GEN: usize = 32;

    /// Prompts of the greedy arm: rows `0..PROMPTS` of `tools/ref/prompts.tsv`.
    const PROMPTS: usize = 8;

    /// Cache rows: the oracle's five tokens, and the longest greedy prompt
    /// plus `GEN` with room.
    const CTX: usize = 256;

    /// The captured step's node count, derived before the chain was built:
    /// the embedding row, 18 nodes per layer (attention norm+quant, q, k, v,
    /// QK-norm+rope+append, flash segment pass, flash merge, q8_1 of the
    /// attention rows, attn_output, the residual add, FFN norm+quant, the
    /// router gemv, the router, gate·up·SwiGLU, q8_1 of the SwiGLU rows,
    /// down, combine, the residual copy) and the head's four (norm, q8_1,
    /// Q6_K gemv, argmax): 1 + 48·18 + 4.
    const NODES_CHAIN: usize = 869;

    /// The layer boundaries' residual copies in the captured step.
    const MEMCPY_CHAIN: usize = 48;

    /// PIN(2026-09-24): the teacher-forced bound on each half's error ratio
    /// — its relative error over the relative distance between the two
    /// sides' 8-bit inputs to that half (`quant_gap`: ours q8_1 per 128
    /// values against ik's q8_2 per 32, at the attention's q/k/v and
    /// attn_output inputs, the FFN's gate·up and down inputs, in
    /// quadrature). To first order a linear map carries its input's relative
    /// perturbation to its output unchanged, so a correct chain reads about
    /// 1; measured on this set: median about 1, worst 2.8 in attention (the
    /// softmax sharpens a score error) and 6.1 in the FFN (a layer whose
    /// input has one dominant channel, where the isotropic prediction
    /// undercounts). The kernel gates bound each kernel exactly; this band
    /// only has to separate that class from a wiring fault, which reads an
    /// error of order one over a gap of about 2e-2.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-24): the free-running bound on a layer output's relative
    /// distance. Derivation: the forced arm's per-layer errors (up to about
    /// 4e-2 of a layer's update, whose norm is of the residual's order)
    /// added in quadrature over 48 layers, independent: √48 · 4e-2 ≈ 0.28.
    const FREE_BAND: f64 = 0.28;

    /// PIN(2026-09-21): the V2-Lite gate's floor — a first difference where
    /// the reference's own top1-top2 margin is below this is a near-tie
    /// re-lottery, not a fault.
    const MARGIN_FLOOR: f32 = 0.5;

    fn rel(a: &[f32], b: &[f32], base: impl Fn(usize) -> f64) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += base(i).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// ik's `name` rows, `k` values each.
    fn rows(man: &RefManifest, name: &str, k: usize) -> Result<Vec<Vec<f32>>, GateError> {
        let row = man.tensor(name, 0)?;
        let v = ref_tensor_logical_in(&man.dir, row)?;
        if row.ne[0] as usize != k || v.len() % k != 0 {
            return Err(format!("{name} is {:?}, want rows of {k}", row.ne).into());
        }
        Ok(v.chunks(k).map(<[f32]>::to_vec).collect())
    }

    fn open(ctx: usize, mode: StepMode) -> Result<Qwen3moeModel, GateError> {
        let file = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-e2e")?;
        let t = Instant::now();
        let mut m = Qwen3moeModel::load_full(file, ctx)?;
        m.set_mode(mode);
        let mma = m.body("gate_qwen3moe_e2e")?.flash_mma();
        println!(
            "load resident_bytes={} ctx={ctx} layers={} flash_mma={mma} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.stages()[0].layers().len(),
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().collect();
        if let Some(i) = args.iter().position(|a| a == "--ppl") {
            let tag = args.get(i + 1).ok_or("--ppl needs a tag")?;
            return ppl(tag);
        }
        let mut m = open(CTX, StepMode::Graph)?;
        let mut ok = true;
        ok &= structure(&mut m)?;
        let o = oracle::for_arch(Arch::Qwen3moe)?;
        let man = o.open(Set::Cpu)?;
        ok &= forced(&mut m, &man)?;
        ok &= free(&mut m, &man)?;
        ok &= greedy(&mut m)?;
        println!("gate_qwen3moe_e2e: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ------------------------------------------------------- (s) structure

    fn structure(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        let nodes = m.capture_step()?;
        let ([kernel, memcpy, memset, host], other) = count_kinds(
            &m.step_graph_nodes()?,
            [
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST,
            ],
        );
        let pass = nodes == NODES_CHAIN && memcpy == MEMCPY_CHAIN && host == 0;
        println!(
            "structure graph_nodes={nodes} (want {NODES_CHAIN}) kernel={kernel} memcpy={memcpy} \
             (want {MEMCPY_CHAIN}) memset={memset} host={host} (want 0) other={other} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------ (f) forced

    /// ik's rows of `name` at every layer, and the offset of its first row
    /// in the prefill's positions (the last layer keeps only the output
    /// token's rows past its attention).
    fn per_layer(
        man: &RefManifest,
        stem: &str,
        n_layer: usize,
        k: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>, GateError> {
        (0..n_layer)
            .map(|l| rows(man, &format!("{stem}-{l}"), k))
            .collect()
    }

    /// The row of `v` (a layer's rows) for prefill position `t` of `t_n`, if
    /// the layer kept it.
    fn at(v: &[Vec<f32>], t: usize, t_n: usize) -> Option<&[f32]> {
        t.checked_sub(t_n - v.len()).map(|i| v[i].as_slice())
    }

    /// `‖x̂_ours − x̂_ik‖ / ‖x‖` over the `k`-value rows of `x`: how far apart
    /// the two sides' 8-bit activations of the same input sit (ours q8_1 per
    /// 128 values, ik q8_2 per 32).
    fn quant_gap(x: &[f32], k: usize) -> f64 {
        let o = q8_1_dequant(x, k, x.len() / k);
        let i = ik_q8_2::reconstruct(x);
        rel(&o, &i, |j| f64::from(x[j]))
    }

    fn forced(m: &mut Qwen3moeModel, man: &RefManifest) -> Result<bool, GateError> {
        let hp = m.body("forced")?.hparams().clone();
        let (h, n_layer, n_exp) = (hp.n_embd, hp.n_layer, hp.experts.n_expert as u32);
        let embd = rows(man, "inp_embd", h)?;
        let t_n = embd.len();
        let ik_out = per_layer(man, "l_out", n_layer, h)?;
        let ik_attn = per_layer(man, "attn_out", n_layer, h)?;
        let ik_routed = per_layer(man, "routed_out", n_layer, h)?;
        let ik_anorm = per_layer(man, "attn_norm", n_layer, h)?;
        let ik_fnorm = per_layer(man, "ffn_inp_normed", n_layer, h)?;
        let q_len = hp.n_head * hp.head_dim;
        let ik_fa: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| rows(man, &format!("fa-{l} (reshaped)"), q_len))
            .collect::<Result<_, _>>()?;
        let ff = hp.experts.ff;
        let ik_par: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| {
                let slots = rows(man, &format!("ffn_moe_gate_par-{l}"), ff)?;
                Ok(slots
                    .chunks(hp.experts.n_used)
                    .map(<[Vec<f32>]>::concat)
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        // The routed sum's terms: each slot's down output times its weight,
        // summed in magnitude — the scale the FFN's error is measured on,
        // since the eight terms can cancel in `routed_out`.
        let n_used = hp.experts.n_used;
        let ik_mag: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| {
                let down = rows(man, &format!("ffn_moe_down-{l}"), h)?;
                let w = ref_tensor_logical_in(
                    &man.dir,
                    man.tensor(&format!("ffn_moe_weights_norm-{l}"), 0)?,
                )?;
                Ok(down
                    .chunks(n_used)
                    .zip(w.chunks(n_used))
                    .map(|(d, w)| {
                        (0..h)
                            .map(|i| (0..n_used).map(|s| (w[s] * d[s][i]).abs()).sum::<f32>())
                            .collect()
                    })
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        let mut ik_ids = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, trow, n_exp)?;
            let n_used = hp.experts.n_used;
            ik_ids.push(ids.chunks(n_used).map(<[i32]>::to_vec).collect::<Vec<_>>());
        }
        m.reset()?;
        let mut ok = true;
        let (mut w_up, mut w_attn, mut w_ffn) = (0.0f64, 0.0f64, 0.0f64);
        let (mut w_attn_r, mut w_ffn_r) = (0.0f64, 0.0f64);
        let (mut flips, mut sites) = (0usize, 0usize);
        for l in 0..n_layer {
            let (mut l_up, mut l_attn, mut l_ffn, mut l_flip) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            let (mut l_attn_r, mut l_attn_p, mut l_ffn_r, mut l_ffn_p) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for t in 0..t_n {
                let x_in: &[f32] = if l == 0 {
                    &embd[t]
                } else {
                    at(&ik_out[l - 1], t, t_n).ok_or("a layer input row is missing")?
                };
                let run = m.step_layer(l, x_in, u32::try_from(t)?)?;
                if let Some(a) = at(&ik_attn[l], t, t_n) {
                    let got: Vec<f32> =
                        run.ffn_inp.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
                    let e = rel(&got, a, |i| f64::from(a[i]));
                    let g_in = at(&ik_anorm[l], t, t_n).map_or(0.0, |x| quant_gap(x, h));
                    let g_fa = at(&ik_fa[l], t, t_n).map_or(0.0, |x| quant_gap(x, q_len));
                    let pred = g_in.hypot(g_fa);
                    l_attn = l_attn.max(e);
                    l_attn_r = l_attn_r.max(e / pred.max(f64::MIN_POSITIVE));
                    l_attn_p = l_attn_p.max(pred);
                }
                let (Some(want), Some(a), Some(r)) = (
                    at(&ik_out[l], t, t_n),
                    at(&ik_attn[l], t, t_n),
                    at(&ik_routed[l], t, t_n),
                ) else {
                    continue;
                };
                l_up = l_up.max(rel(&run.l_out, want, |i| {
                    f64::from(want[i]) - f64::from(x_in[i])
                }));
                // The FFN half alone, on ik's own FFN input.
                let ik_inp: Vec<f32> = a.iter().zip(x_in).map(|(&a, &x)| a + x).collect();
                let f = m.step_ffn(l, &ik_inp)?;
                let got: Vec<f32> = f.l_out.iter().zip(&ik_inp).map(|(&o, &i)| o - i).collect();
                let theirs = &ik_ids[l][ik_ids[l].len() - (t_n - t)];
                let flipped = f
                    .ids
                    .iter()
                    .filter(|&&e| !theirs.contains(&(e as i32)))
                    .count();
                sites += 1;
                let mag = at(&ik_mag[l], t, t_n).ok_or("a routed magnitude row is missing")?;
                let e = rel(&got, r, |i| f64::from(mag[i]));
                let g_in = at(&ik_fnorm[l], t, t_n).map_or(0.0, |x| quant_gap(x, h));
                let g_par = at(&ik_par[l], t, t_n).map_or(0.0, |x| quant_gap(x, ff));
                let pred = g_in.hypot(g_par);
                l_ffn_p = l_ffn_p.max(pred);
                if flipped == 0 {
                    l_ffn_r = l_ffn_r.max(e / pred.max(f64::MIN_POSITIVE));
                }
                if flipped > 0 {
                    flips += 1;
                    l_flip += 1;
                    println!(
                        "forced layer={l} pos={t}: {flipped} router id(s) differ from ik's, ffn_rel={e:.3e} (printed)"
                    );
                } else {
                    l_ffn = l_ffn.max(e);
                }
            }
            println!(
                "forced layer={l} attn_rel={l_attn:.3e} (quant gap {l_attn_p:.3e}, worst ratio {l_attn_r:.2}) \
                 ffn_rel={l_ffn:.3e} (quant gap {l_ffn_p:.3e}, worst ratio {l_ffn_r:.2}) \
                 update_rel={l_up:.3e} flip_sites={l_flip}"
            );
            w_up = w_up.max(l_up);
            w_attn = w_attn.max(l_attn);
            w_ffn = w_ffn.max(l_ffn);
            w_attn_r = w_attn_r.max(l_attn_r);
            w_ffn_r = w_ffn_r.max(l_ffn_r);
            if l_attn_r > RATIO_BAND || l_ffn_r > RATIO_BAND {
                ok = false;
                println!("forced layer={l}: an error ratio passes {RATIO_BAND} FAIL");
            }
        }
        println!(
            "forced: {n_layer} layers x {t_n} positions; attention half worst rel {w_attn:.3e}, ratio \
             {w_attn_r:.2}; FFN half on ik's input worst rel {w_ffn:.3e}, ratio {w_ffn_r:.2} over {} \
             sites without a routing difference (band ratio {RATIO_BAND}); {flips} sites route an \
             expert ik does not (printed); whole-layer update worst {w_up:.3e} (printed) {}",
            sites - flips,
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (c) free-running

    fn free(m: &mut Qwen3moeModel, man: &RefManifest) -> Result<bool, GateError> {
        let hp = m.body("free")?.hparams().clone();
        let (h, n_layer) = (hp.n_embd, hp.n_layer);
        let toks: Vec<u32> = ref_ints(man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
            .iter()
            .map(|&i| u32::try_from(i))
            .collect::<Result<_, _>>()?;
        let mut ik_out = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            ik_out.push(rows(man, &format!("l_out-{l}"), h)?);
        }
        let ik_logits = rows(man, "result_output", hp.n_vocab)?;
        m.set_layer_taps(true)?;
        m.reset()?;
        let t_n = toks.len();
        let mut per_layer = vec![0.0f64; n_layer];
        let mut last = 0u32;
        for (t, &tok) in toks.iter().enumerate() {
            last = m.step(&[tok])?;
            let taps = m.layer_taps()?;
            for l in 0..n_layer {
                let off = t_n - ik_out[l].len();
                let Some(t_ik) = t.checked_sub(off) else {
                    continue;
                };
                let want = &ik_out[l][t_ik];
                per_layer[l] = per_layer[l].max(rel(&taps[l], want, |i| f64::from(want[i])));
            }
        }
        let logits = m.logits()?;
        m.set_layer_taps(false)?;
        let want = ik_logits.last().ok_or("result_output has no row")?;
        let ik_top = argmax(want);
        let lrel = rel(&logits, want, |i| f64::from(want[i]));
        let worst = per_layer.iter().copied().fold(0.0f64, f64::max);
        let ok = worst <= FREE_BAND && last == ik_top;
        for (l, &e) in per_layer.iter().enumerate() {
            if l % 8 == 0 || l == n_layer - 1 || e > FREE_BAND {
                println!("free layer={l} l_out_rel={e:.3e}");
            }
        }
        println!(
            "free: {t_n} tokens {toks:?}, worst l_out_rel={worst:.3e} (band {FREE_BAND:.0e}); last \
             position argmax ours={last} ik={ik_top} logits_rel={lrel:.3e} (printed) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    fn argmax(v: &[f32]) -> u32 {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best as u32
    }

    // ------------------------------------------------ (g) greedy, (r) replay

    fn greedy_dir() -> PathBuf {
        data_dir().join("qwen3moe").join("greedy")
    }

    /// Our greedy continuation of `ids`: `GEN` tokens, the first the
    /// prompt's own next token, and the last step's logits.
    fn continue_greedy(
        m: &mut Qwen3moeModel,
        ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        m.reset()?;
        let mut out = Vec::with_capacity(GEN);
        let mut next = m.step(ids)?;
        out.push(next);
        for _ in 1..GEN {
            next = m.step(&[next])?;
            out.push(next);
        }
        Ok((out, m.logits()?))
    }

    fn greedy(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        let dir = greedy_dir();
        let mut prompts = Vec::new();
        let mut reference = Vec::new();
        for p in 0..PROMPTS {
            let pr = read_prompts(&dir.join(format!("prompt{p}.tsv")))?;
            let gr = read_greedy(&dir.join(format!("greedy-ik-cpu-{GEN}-p{p}.tsv")))?;
            let (Some(pr), Some(gr)) = (pr.into_iter().next(), gr.into_iter().next()) else {
                return Err(format!("{}: prompt {p} has no row", dir.display()).into());
            };
            if gr.n_tokens != pr.tokens.len() {
                return Err(format!(
                    "prompt {p}: {} ids, the greedy file's row says {}",
                    pr.tokens.len(),
                    gr.n_tokens
                )
                .into());
            }
            prompts.push(pr);
            reference.push(gr);
        }
        let mut graph = Vec::with_capacity(PROMPTS);
        let mut graph_logits = Vec::with_capacity(PROMPTS);
        m.set_mode(StepMode::Graph);
        for p in &prompts {
            let (toks, logits) = continue_greedy(m, &p.tokens)?;
            graph.push(toks);
            graph_logits.push(logits);
        }
        m.set_mode(StepMode::Eager);
        let mut replay_ok = true;
        for (i, p) in prompts.iter().enumerate() {
            let (toks, logits) = continue_greedy(m, &p.tokens)?;
            let same_t = toks == graph[i];
            let same_l = logits
                .iter()
                .zip(&graph_logits[i])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            if !(same_t && same_l) {
                replay_ok = false;
                println!(
                    "replay prompt {}: tokens_equal={same_t} logits_bit_equal={same_l} FAIL",
                    p.id
                );
            }
        }
        m.set_mode(StepMode::Graph);
        println!(
            "replay: {PROMPTS} prompts x {GEN} tokens, graph = eager tokens and last logits bit for bit {}",
            verdict(replay_ok)
        );
        let rep = compare_greedy(&graph, &reference, MARGIN_FLOOR);
        for (r, p) in rep.rows.iter().zip(&prompts) {
            println!(
                "greedy prompt {} ({} ids): {:?} first_diff={:?} ik_margin={:?} ours={}/{} tokens",
                r.id,
                p.tokens.len(),
                r.class,
                r.first_diff,
                r.ref_margin,
                r.n_ours,
                r.n_ref
            );
        }
        let greedy_ok = rep.rows.iter().all(|r| r.class != GreedyClass::Diverged);
        println!(
            "greedy: identical={} near_tie={} diverged={} (want 0, margin floor {MARGIN_FLOOR}) {}",
            rep.n_identical,
            rep.n_near_tie,
            rep.n_diverged,
            verdict(greedy_ok)
        );
        Ok(replay_ok && greedy_ok)
    }

    // ------------------------------------------------------------- ppl

    /// The top two of `v` by (value desc, id asc): ids and values.
    fn top2(v: &[f32]) -> (usize, f32, usize, f32) {
        let ahead = |a: usize, b: usize| v[a] > v[b] || (v[a] == v[b] && a < b);
        let (mut best, mut second) = (0usize, usize::MAX);
        for i in 1..v.len() {
            if ahead(i, best) {
                second = best;
                best = i;
            } else if second == usize::MAX || ahead(i, second) {
                second = i;
            }
        }
        (best, v[best], second, v[second])
    }

    fn ppl(tag: &str) -> Result<(), GateError> {
        let path: PathBuf = data_dir().join("ikppl").join(format!("{tag}.kld"));
        let base = KldBase::open_own_vocab(&path)?;
        println!(
            "ppl base {}: ctx {} chunks {} scored per chunk {} from {}",
            path.display(),
            base.n_ctx(),
            base.n_chunk(),
            base.scored_per_chunk(),
            base.first_scored()
        );
        let mut m = open(base.n_ctx(), StepMode::Graph)?;
        let n_vocab = m.body("ppl")?.hparams().n_vocab;
        if base.n_vocab() != n_vocab {
            return Err(format!(
                "{} holds {} vocabulary entries per record, the model {n_vocab}",
                path.display(),
                base.n_vocab()
            )
            .into());
        }
        let t = Instant::now();
        let (mut n, mut sd, mut sd2) = (0usize, 0.0f64, 0.0f64);
        let (mut nll_o, mut nll_i, mut kld, mut kld2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut same_top, mut dm2) = (0usize, 0.0f64);
        let mut lq = vec![0.0f64; n_vocab];
        for c in 0..base.n_chunk() {
            let toks = base
                .chunk_tokens(c)
                .ok_or_else(|| format!("chunk {c} has no ids"))?
                .to_vec();
            m.reset()?;
            for pos in 0..base.n_ctx() - 1 {
                m.step(&toks[pos..=pos])?;
                let Some(rec) = base.record(c, pos) else {
                    continue;
                };
                let logits = m.logits()?;
                if logits.len() != n_vocab {
                    return Err(format!("{} logits, the vocabulary {n_vocab}", logits.len()).into());
                }
                let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let lse = f64::from(mx)
                    + logits
                        .iter()
                        .map(|&v| (f64::from(v) - f64::from(mx)).exp())
                        .sum::<f64>()
                        .ln();
                for (q, &v) in lq.iter_mut().zip(&logits) {
                    *q = f64::from(v) - lse;
                }
                let no = -lq[rec.next as usize];
                let ni = rec.nll();
                let d = no - ni;
                n += 1;
                sd += d;
                sd2 += d * d;
                nll_o += no;
                nll_i += ni;
                let mut k = 0.0f64;
                let (mut top_i, mut best_i, mut second_i) =
                    (0usize, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for (i, lp) in rec.log_probs().enumerate() {
                    let lp = f64::from(lp);
                    k += lp.exp() * (lp - lq[i]);
                    if lp > best_i {
                        second_i = best_i;
                        (top_i, best_i) = (i, lp);
                    } else if lp > second_i {
                        second_i = lp;
                    }
                }
                kld += k;
                kld2 += k * k;
                let (a, av, _, bv) = top2(&logits);
                same_top += usize::from(a == top_i);
                let dm = f64::from(av - bv) - (best_i - second_i);
                dm2 += dm * dm;
            }
            let nf = n as f64;
            println!(
                "ppl chunk {c}: {n} positions, mean d {:+.5}, ppl ours {:.4} ik {:.4}, {:.0} s",
                sd / nf,
                (nll_o / nf).exp(),
                (nll_i / nf).exp(),
                t.elapsed().as_secs_f64()
            );
        }
        let nf = n as f64;
        let mean = sd / nf;
        let var = (sd2 / nf - mean * mean).max(0.0) * nf / (nf - 1.0);
        let se = (var / nf).sqrt();
        let kmean = kld / nf;
        let kse = ((kld2 / nf - kmean * kmean).max(0.0) / (nf - 1.0)).sqrt();
        println!(
            "ppl: {n} positions; PPL ours {:.4} ik {:.4}; d = NLL_ours − NLL_ik mean {mean:+.5} ± \
             {se:.5} (SE), sd(d) {:.4}; Δ_PPL {:+.3} %; KLD(ik‖ours) {kmean:.5} ± {kse:.5}; same top \
             {:.2} %; σ_rel (rms of margin ours − ik) {:.4}; {:.0} s (printed, not judged)",
            (nll_o / nf).exp(),
            (nll_i / nf).exp(),
            var.sqrt(),
            100.0 * (mean.exp() - 1.0),
            100.0 * same_top as f64 / nf,
            (dm2 / nf).sqrt(),
            t.elapsed().as_secs_f64()
        );
        Ok(())
    }
}
