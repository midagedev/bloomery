//! GPU gate for the qwen3moe router (`arch::qwen3moe::router`): softmax
//! over 128 experts, the top 8, and the weights renormalized
//! (`norm_topk_prob`).
//!
//! The routing alone (`qwen3moe_router`, one token's logits in) against the
//! host rule (`route_ref_within` at 128/8 — the serial max fold, f32 `exp`,
//! the f64 ascending sum, the f32 divide, ties to the smaller id — then the
//! chosen probabilities summed in f64 in slot order and each divided by the
//! f32 sum): ids EXACT, probabilities and weights within [`BAND`] (the
//! device `exp` and the host's differ in their last ulps, and the device
//! sums the exps as a lane tree, which rounds to the same f32 normalizer
//! unless the two f64 sums straddle an f32 rounding boundary; everything
//! else is mirrored op for op), and a rerun bit-identical.
//! Against ik, on every token of every layer of every set (`ffn_moe_logits-L`
//! in): the ids EQUAL `ffn_moe_topk-L` (its logical twin, every token's
//! eight in rank order); the weights' distance to `ffn_moe_weights_norm-L`
//! is printed, not asserted.
//!
//! Constructed ties pin the tie rule where the dump cannot: all 128 logits
//! equal; an eighth place shared by three experts, two of them in one lane
//! of the selecting warp; a first place shared across lanes. And the
//! routing as a captured graph: one node, the replay bit-identical to the
//! eager launch.
//!
//! The engine's launch (`qwen3moe_router_fused`: the logit gemv and the
//! routing in one launch, the last block to finish routing every token) at
//! m = 1, 5 and 8 tokens (`ffn_inp_normed-13` of the prefill set, then
//! `-14`'s): logits, probabilities, ids and weights BIT-EQUAL to
//! `f32_gemv` followed by the routing alone on each token's column, and the
//! block ticket count back at zero. As a captured graph at m = 8: one node,
//! two replays bit-identical to the eager launch, the count zero after each.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_router: built without the `gpu` feature; see `just gate-gpu-qwen3moe-router`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_router", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::router::{
        MAX_TOKENS, N_EXPERT, N_USED, RouterKernels, RouterOut,
    };
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_gates::qwen3moe::sets;
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, open_model, open_split, ref_tensor_logical_in,
        route_ref_within, tensor_bytes_as, topk_ids_logical_within, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::GgmlType;
    use model::arch::Arch;
    use model::arch::qwen3moe::hparams::{Hparams, Score};
    use model::arch::qwen3moe::names;

    /// Probabilities and weights against the host rule, absolute: both are
    /// in [0, 1] and the only divergence is `exp`'s last ulps.
    const BAND: f32 = 1e-6;

    /// One launch's results.
    struct Routed {
        probs: Vec<f32>,
        ids: Vec<u32>,
        weights: Vec<f32>,
    }

    fn route(
        k: &RouterKernels,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut RouterOut,
    ) -> Result<Routed, GateError> {
        k.enqueue(stream, x, out)?;
        stream.synchronize()?;
        Ok(Routed {
            probs: out.probs.to_host_vec(stream)?,
            ids: out.ids.to_host_vec(stream)?,
            weights: out.weights.to_host_vec(stream)?,
        })
    }

    /// The host rule for one token: probabilities, ids and renormalized
    /// weights.
    fn host(logits: &[f32]) -> Result<Routed, GateError> {
        let (probs, ids, w) = route_ref_within(logits, 1, N_EXPERT, N_USED, 1.0)?;
        let sum = w.iter().fold(0.0f64, |a, &v| a + f64::from(v)) as f32;
        Ok(Routed {
            probs,
            ids: ids.into_iter().map(|i| i as u32).collect(),
            weights: w.iter().map(|&v| v / sum).collect(),
        })
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// A fused launch's results for its first `m` tokens.
    struct Fused {
        logits: Vec<f32>,
        routed: Routed,
    }

    fn read_fused(stream: &CudaStream, out: &RouterOut, m: usize) -> Result<Fused, GateError> {
        let mut logits = out.logits.to_host_vec(stream)?;
        let mut probs = out.probs.to_host_vec(stream)?;
        let mut ids = out.ids.to_host_vec(stream)?;
        let mut weights = out.weights.to_host_vec(stream)?;
        logits.truncate(m * N_EXPERT);
        probs.truncate(m * N_EXPERT);
        ids.truncate(m * N_USED);
        weights.truncate(m * N_USED);
        Ok(Fused {
            logits,
            routed: Routed {
                probs,
                ids,
                weights,
            },
        })
    }

    fn fused_equal(a: &Fused, b: &Fused) -> bool {
        bits_equal(&a.logits, &b.logits)
            && bits_equal(&a.routed.probs, &b.routed.probs)
            && a.routed.ids == b.routed.ids
            && bits_equal(&a.routed.weights, &b.routed.weights)
    }

    /// The split reference for `m` token columns `x`: `f32_gemv` into the
    /// row-major logits, then the routing alone on each token's column.
    fn split_pair(
        gpu: &Gpu,
        k: &RouterKernels,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: &mut RouterOut,
    ) -> Result<Fused, GateError> {
        let stream = gpu.stream();
        let mut y = DeviceBuffer::<f32>::zeroed(stream, N_EXPERT * m)?;
        gpu.q8f32().enqueue_f32_gemv(stream, w, x, m, &mut y)?;
        stream.synchronize()?;
        let y = y.to_host_vec(stream)?;
        let mut want = Fused {
            logits: Vec::new(),
            routed: Routed {
                probs: Vec::new(),
                ids: Vec::new(),
                weights: Vec::new(),
            },
        };
        for t in 0..m {
            let col: Vec<f32> = (0..N_EXPERT).map(|e| y[e * m + t]).collect();
            let r = route(k, stream, &DeviceBuffer::from_host(stream, &col)?, out)?;
            want.logits.extend_from_slice(&col);
            want.routed.probs.extend_from_slice(&r.probs);
            want.routed.ids.extend_from_slice(&r.ids);
            want.routed.weights.extend_from_slice(&r.weights);
        }
        Ok(want)
    }

    /// Layer `l`'s router weight as f32 on the card.
    fn router_weight(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        l: usize,
    ) -> Result<DeviceTensor<f32>, GateError> {
        let name = names::ffn_gate_inp(l);
        let (info, b) = tensor_bytes_as(gguf, &name, GgmlType::F32, None)?;
        if info.dims.len() != 2 || info.dims[1] as usize != N_EXPERT {
            return Err(format!("{name} is {:?}, want [k, {N_EXPERT}]", info.dims).into());
        }
        let w: Vec<f32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(DeviceTensor::upload(
            gpu.stream(),
            &w,
            N_EXPERT,
            info.dims[0] as usize,
        )?)
    }

    pub fn run() -> Result<(), GateError> {
        let split = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-router")?;
        let hp = Hparams::read(&split)?;
        let e = hp.experts;
        if e.n_expert != N_EXPERT
            || e.n_used != N_USED
            || e.score != Score::Softmax
            || !e.weights_norm
        {
            return Err(format!(
                "the file routes {:?}; the kernel is softmax, {N_USED} of {N_EXPERT}, renormalized",
                e
            )
            .into());
        }
        let gpu = Gpu::new()?;
        let k = RouterKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let mut out = RouterOut::new(stream)?;
        println!("gate_qwen3moe_router: device {}", gpu.device_name()?);
        let mut ok = true;

        // ---- every token of every layer of every set.
        let (mut tokens, mut ik_mismatch, mut worst_w, mut worst_p, mut worst_ik) =
            (0usize, 0usize, 0.0f32, 0.0f32, 0.0f32);
        for (label, man) in sets()? {
            let mut set_ok = true;
            for layer in 0..hp.n_layer {
                let lrow = man.tensor(&format!("ffn_moe_logits-{layer}"), 0)?;
                let trow = man.tensor(&format!("ffn_moe_topk-{layer}"), 0)?;
                let wrow = man.tensor(&format!("ffn_moe_weights_norm-{layer}"), 0)?;
                let logits = ref_tensor_logical_in(&man.dir, lrow)?;
                let ik_ids = topk_ids_logical_within(&man, trow, N_EXPERT as u32)?;
                let ik_w = ref_tensor_logical_in(&man.dir, wrow)?;
                let m = lrow.ne[1] as usize;
                if lrow.ne[0] as usize != N_EXPERT
                    || ik_ids.len() != N_USED * m
                    || ik_w.len() != N_USED * m
                {
                    return Err(format!(
                        "{label} layer {layer}: logits {:?}, {} ids, {} weights",
                        lrow.ne,
                        ik_ids.len(),
                        ik_w.len()
                    )
                    .into());
                }
                for t in 0..m {
                    let lt = &logits[t * N_EXPERT..(t + 1) * N_EXPERT];
                    let x = DeviceBuffer::from_host(stream, lt)?;
                    let a = route(&k, stream, &x, &mut out)?;
                    let b = route(&k, stream, &x, &mut out)?;
                    let h = host(lt)?;
                    let rerun = bits_equal(&a.probs, &b.probs)
                        && a.ids == b.ids
                        && bits_equal(&a.weights, &b.weights);
                    let ids_exact = a.ids == h.ids;
                    let (pe, we) = (max_abs(&a.probs, &h.probs), max_abs(&a.weights, &h.weights));
                    let want: Vec<u32> = ik_ids[t * N_USED..(t + 1) * N_USED]
                        .iter()
                        .map(|&i| i as u32)
                        .collect();
                    let ik_same = a.ids == want;
                    let ik_rel = a
                        .weights
                        .iter()
                        .zip(&ik_w[t * N_USED..(t + 1) * N_USED])
                        .fold(0.0f32, |m, (x, y)| {
                            m.max((x - y).abs() / y.abs().max(f32::MIN_POSITIVE))
                        });
                    let pass = rerun && ids_exact && pe <= BAND && we <= BAND && ik_same;
                    tokens += 1;
                    ik_mismatch += usize::from(!ik_same);
                    worst_p = worst_p.max(pe);
                    worst_w = worst_w.max(we);
                    worst_ik = worst_ik.max(ik_rel);
                    if !pass {
                        set_ok = false;
                        println!(
                            "router set={label} layer={layer} t={t} ids={:?} host={:?} ik={want:?} \
                             probs_err={pe:.3e} weights_err={we:.3e} rerun={rerun} FAIL",
                            a.ids, h.ids
                        );
                    }
                }
            }
            println!(
                "router set={label}: {} layers, ids = host and = ik on every token: {set_ok}",
                hp.n_layer
            );
            ok &= set_ok;
        }
        println!(
            "router real: {tokens} tokens, ik_ids_mismatch={ik_mismatch}/{tokens} probs_err={worst_p:.3e} \
             weights_err={worst_w:.3e} (band {BAND:.0e}) ik_weights_rel={worst_ik:.3e} (printed, not pinned) {}",
            verdict(ok)
        );

        // ---- constructed ties.
        let mut cases: Vec<(&str, Vec<f32>, &[u32])> = Vec::new();
        cases.push(("all-equal", vec![0.5; N_EXPERT], &[0, 1, 2, 3, 4, 5, 6, 7]));
        let mut l = vec![-4.0f32; N_EXPERT];
        for (i, &e) in [100usize, 90, 64, 33, 5, 127, 70].iter().enumerate() {
            l[e] = 3.0 - 0.25 * i as f32;
        }
        for e in [96usize, 32, 1] {
            l[e] = 1.0;
        }
        cases.push((
            "eighth-shared-by-96-32-1",
            l,
            &[100, 90, 64, 33, 5, 127, 70, 1],
        ));
        let mut l = vec![-4.0f32; N_EXPERT];
        for (i, e) in (40..47).enumerate() {
            l[e] = 1.0 - 0.1 * i as f32;
        }
        l[127] = 2.0;
        l[31] = 2.0;
        cases.push((
            "first-shared-by-127-31",
            l,
            &[31, 127, 40, 41, 42, 43, 44, 45],
        ));
        for (name, logits, want) in &cases {
            let x = DeviceBuffer::from_host(stream, logits)?;
            let a = route(&k, stream, &x, &mut out)?;
            let h = host(logits)?;
            let pass = a.ids == *want && h.ids == *want && max_abs(&a.weights, &h.weights) <= BAND;
            println!(
                "tie case={name} ids={:?} host={:?} want={want:?} {}",
                a.ids,
                h.ids,
                verdict(pass)
            );
            ok &= pass;
        }

        // ---- the router as a captured graph.
        let man = &sets()?[0].1;
        let lrow = man.tensor("ffn_moe_logits-13", 0)?;
        let logits = ref_tensor_logical_in(&man.dir, lrow)?;
        let x = DeviceBuffer::from_host(stream, &logits[..N_EXPERT])?;
        let eager = route(&k, stream, &x, &mut out)?;
        out.probs.zero_async(stream)?;
        out.ids.zero_async(stream)?;
        out.weights.zero_async(stream)?;
        stream.synchronize()?;
        let graph = gpu.capture(|s| k.enqueue(s, &x, &mut out))?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let replay = Routed {
            probs: out.probs.to_host_vec(stream)?,
            ids: out.ids.to_host_vec(stream)?,
            weights: out.weights.to_host_vec(stream)?,
        };
        let identical = bits_equal(&eager.probs, &replay.probs)
            && eager.ids == replay.ids
            && bits_equal(&eager.weights, &replay.weights);
        let nodes = graph.node_count();
        let pass = identical && nodes == 1;
        println!(
            "graph op=qwen3moe_router src=ffn_moe_logits-13 eager_vs_graph_bit_identical={identical} \
             graph_nodes={nodes} {}",
            verdict(pass)
        );
        ok &= pass;

        // ---- the engine's fused launch against the split pair.
        let gguf = open_model()?;
        let w = router_weight(&gpu, &gguf, 13)?;
        let kk = w.cols();
        let cpu = &sets()?[0].1;
        let mut cols = ref_tensor_logical_in(&cpu.dir, cpu.tensor("ffn_inp_normed-13", 0)?)?;
        cols.extend(ref_tensor_logical_in(
            &cpu.dir,
            cpu.tensor("ffn_inp_normed-14", 0)?,
        )?);
        if cols.len() < MAX_TOKENS * kk {
            return Err(format!(
                "ffn_inp_normed-13/-14 hold {} values, want {MAX_TOKENS} x {kk}",
                cols.len()
            )
            .into());
        }
        let mut fout = RouterOut::with_tokens(stream, MAX_TOKENS)?;
        for m in [1usize, 5, MAX_TOKENS] {
            let x = DeviceBuffer::from_host(stream, &cols[..m * kk])?;
            let want = split_pair(&gpu, &k, &w, &x, m, &mut out)?;
            k.enqueue_fused(stream, &w, &x, m, &mut fout)?;
            stream.synchronize()?;
            let got = read_fused(stream, &fout, m)?;
            let tickets = fout.tickets(stream)?;
            let logits = bits_equal(&got.logits, &want.logits);
            let probs = bits_equal(&got.routed.probs, &want.routed.probs);
            let ids = got.routed.ids == want.routed.ids;
            let weights = bits_equal(&got.routed.weights, &want.routed.weights);
            let pass = logits && probs && ids && weights && tickets == 0;
            println!(
                "fused m={m} src=ffn_inp_normed-13/14 vs f32_gemv+router: logits_bits={logits} \
                 probs_bits={probs} ids={ids} weights_bits={weights} tickets={tickets} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        let x = DeviceBuffer::from_host(stream, &cols[..MAX_TOKENS * kk])?;
        k.enqueue_fused(stream, &w, &x, MAX_TOKENS, &mut fout)?;
        stream.synchronize()?;
        let eager = read_fused(stream, &fout, MAX_TOKENS)?;
        fout.logits.zero_async(stream)?;
        fout.probs.zero_async(stream)?;
        fout.ids.zero_async(stream)?;
        fout.weights.zero_async(stream)?;
        stream.synchronize()?;
        let graph = gpu.capture(|s| k.enqueue_fused(s, &w, &x, MAX_TOKENS, &mut fout))?;
        let mut replays_ok = true;
        let mut tickets_after = Vec::new();
        for _ in 0..2 {
            graph.launch(stream)?;
            stream.synchronize()?;
            replays_ok &= fused_equal(&eager, &read_fused(stream, &fout, MAX_TOKENS)?);
            tickets_after.push(fout.tickets(stream)?);
        }
        let nodes = graph.node_count();
        let pass = replays_ok && nodes == 1 && tickets_after.iter().all(|&t| t == 0);
        println!(
            "graph op=qwen3moe_router_fused m={MAX_TOKENS} two_replays_bit_identical={replays_ok} \
             tickets_after={tickets_after:?} graph_nodes={nodes} {}",
            verdict(pass)
        );
        ok &= pass;

        println!("gate_qwen3moe_router: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
