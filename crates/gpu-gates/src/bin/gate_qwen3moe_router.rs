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
//!
//! The one-token launch with the FFN norm folded in (`qwen3moe_router_norm`,
//! the chain's at m = 1) against the two launches it replaces —
//! `fused::norm_quant` then `qwen3moe_router_fused` at one token — on every
//! token of `l_out-12` through layer 13's `ffn_norm` and router weight and of
//! `l_out-46` through layer 47's: the q8_1 bytes (all five planes, the scales
//! by bits), logits, probabilities, ids and weights BIT-EQUAL, the ticket
//! count zero. A refused input — a NaN, and a finite value whose square
//! overflows the sum of squares — raises `NormQuant` and `Router` in both,
//! every block's q8_1 scale NaN in both (the same bytes), and the token's
//! routing NaN. `norm_quant` alone, labelled with layer 13, over three
//! columns (a NaN in column 1, the overflow in column 2): `NormQuant` with
//! that layer, the refused columns' normed values and scales NaN, column 0
//! the clean run's bit for bit, the word clean before and after a clean
//! run; then one column whose gain is `+inf` at value 200: only its block 1
//! (values 128..256) is refused, every other block the clean run's. As a
//! captured graph: one node, two replays bit-identical to the eager launch,
//! the count zero after each.
//!
//! Refusals, each launch labelled with layer 13: a token with a non-finite
//! logit raises `FaultSite::Router` with that layer, its probabilities and
//! weights are NaN and its ids those it held before the launch, and every
//! other token is bit for bit the clean run's. The routing alone on a NaN, a
//! +inf and a −inf logit, a clean rerun after each (the word clean, the clean
//! bits); the fused launch at five tokens with one token's column finite but
//! overflowing to +inf at one expert (the column is `1e38` with the sign of
//! that expert's weights); the ubatch pair at nine tokens with the ninth so;
//! a NaN in the router weight, which refuses every token of the fused launch
//! and the norm-fused launch's token (its q8_1 bytes unchanged).

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
    use bloomery_gpu::fused::{FusedKernels, Q8ActHost, readback_q8act};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, LAYER_NONE, Q8Act};
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
        sink: FaultSink,
        out: &mut RouterOut,
    ) -> Result<Routed, GateError> {
        k.enqueue(stream, x, sink, out)?;
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
            let r = route(
                k,
                stream,
                &DeviceBuffer::from_host(stream, &col)?,
                gpu.unlabelled_sink(),
                out,
            )?;
            want.logits.extend_from_slice(&col);
            want.routed.probs.extend_from_slice(&r.probs);
            want.routed.ids.extend_from_slice(&r.ids);
            want.routed.weights.extend_from_slice(&r.weights);
        }
        Ok(want)
    }

    /// Layer `l`'s router weight as f32 on the host ([`N_EXPERT`] rows of
    /// `k`), and `k`.
    fn router_weight_host(gguf: &gguf::Gguf, l: usize) -> Result<(Vec<f32>, usize), GateError> {
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
        Ok((w, info.dims[0] as usize))
    }

    /// Layer `l`'s router weight as f32 on the card.
    fn router_weight(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        l: usize,
    ) -> Result<DeviceTensor<f32>, GateError> {
        let (w, k) = router_weight_host(gguf, l)?;
        Ok(DeviceTensor::upload(gpu.stream(), &w, N_EXPERT, k)?)
    }

    /// Two q8_1 readbacks equal plane for plane, the scales by bits.
    fn q8_equal(a: &Q8ActHost, b: &Q8ActHost) -> bool {
        a.q3 == b.q3 && a.q4 == b.q4 && a.q6 == b.q6 && a.s8 == b.s8 && bits_equal(&a.d8, &b.d8)
    }

    /// Layer `l`'s FFN norm gain on the host.
    fn ffn_gain_host(gguf: &gguf::Gguf, l: usize) -> Result<Vec<f32>, GateError> {
        let (_, b) = tensor_bytes_as(gguf, &names::ffn_norm(l), GgmlType::F32, None)?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// Layer `l`'s FFN norm gain on the card.
    fn ffn_gain(gpu: &Gpu, gguf: &gguf::Gguf, l: usize) -> Result<DeviceBuffer<f32>, GateError> {
        Ok(DeviceBuffer::from_host(
            gpu.stream(),
            &ffn_gain_host(gguf, l)?,
        )?)
    }

    /// One path's results for one token: the q8_1 of the normed row and the
    /// router's outputs.
    struct NormRouted {
        act: Q8ActHost,
        routed: Fused,
    }

    /// What one layer's norm-fused check reads: the router weight, the norm
    /// gain and epsilon, and the module `norm_quant` lives in.
    struct NormLayer<'a> {
        w: &'a DeviceTensor<f32>,
        gain: &'a DeviceBuffer<f32>,
        eps: f32,
        norm: &'a FusedKernels,
    }

    /// The two launches the norm-fused entry replaces, on one token `x`.
    fn norm_split(
        gpu: &Gpu,
        k: &RouterKernels,
        nl: &NormLayer<'_>,
        x: &DeviceBuffer<f32>,
        out: &mut RouterOut,
    ) -> Result<NormRouted, GateError> {
        let stream = gpu.stream();
        let kk = nl.w.cols();
        let mut act = Q8Act::with_k(stream, 1, kk)?;
        let mut normed = DeviceBuffer::<f32>::zeroed(stream, kk)?;
        nl.norm.enqueue_norm_quant(
            stream,
            x,
            nl.gain,
            nl.eps,
            &mut act,
            &mut normed,
            gpu.unlabelled_sink(),
        )?;
        k.enqueue_fused(stream, nl.w, &normed, 1, gpu.unlabelled_sink(), out)?;
        stream.synchronize()?;
        Ok(NormRouted {
            act: readback_q8act(stream, &act)?,
            routed: read_fused(stream, out, 1)?,
        })
    }

    /// The norm-fused entry on one token `x`, and the ticket count after it.
    fn norm_fused(
        gpu: &Gpu,
        k: &RouterKernels,
        nl: &NormLayer<'_>,
        x: &DeviceBuffer<f32>,
        out: &mut RouterOut,
    ) -> Result<(NormRouted, u32), GateError> {
        let stream = gpu.stream();
        let mut act = Q8Act::with_k(stream, 1, nl.w.cols())?;
        k.enqueue_norm_fused(
            stream,
            nl.w,
            x,
            nl.gain,
            nl.eps,
            &mut act,
            gpu.unlabelled_sink(),
            out,
        )?;
        stream.synchronize()?;
        let run = NormRouted {
            act: readback_q8act(stream, &act)?,
            routed: read_fused(stream, out, 1)?,
        };
        Ok((run, out.tickets(stream)?))
    }

    /// Whether token 0's routing in `f` is refused: NaN probabilities and
    /// weights.
    fn routing_nan(f: &Fused) -> bool {
        f.routed.probs[..N_EXPERT].iter().all(|v| v.is_nan())
            && f.routed.weights[..N_USED].iter().all(|v| v.is_nan())
    }

    /// `norm_quant` over the `x.len() / gain.len()` columns `x`, raising on
    /// `sink`: the normed values and the q8_1 planes, read back.
    fn norm_quant_run(
        gpu: &Gpu,
        norm: &FusedKernels,
        x: &[f32],
        gain: &[f32],
        eps: f32,
        sink: FaultSink,
    ) -> Result<(Vec<f32>, Q8ActHost), GateError> {
        let stream = gpu.stream();
        let kk = gain.len();
        let m = x.len() / kk;
        let mut act = Q8Act::with_k(stream, m, kk)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * kk)?;
        norm.enqueue_norm_quant(
            stream,
            &DeviceBuffer::from_host(stream, x)?,
            &DeviceBuffer::from_host(stream, gain)?,
            eps,
            &mut act,
            &mut y,
            sink,
        )?;
        stream.synchronize()?;
        Ok((y.to_host_vec(stream)?, readback_q8act(stream, &act)?))
    }

    /// Column `c` of `m` of a plane sized for `m` columns.
    fn col<T>(plane: &[T], m: usize, c: usize) -> &[T] {
        let n = plane.len() / m;
        &plane[c * n..(c + 1) * n]
    }

    /// Column `c` of `m` of two q8_1 readbacks equal plane for plane, the
    /// scales by bits.
    fn q8_col_equal(a: &Q8ActHost, b: &Q8ActHost, m: usize, c: usize) -> bool {
        col(&a.q3, m, c) == col(&b.q3, m, c)
            && col(&a.q4, m, c) == col(&b.q4, m, c)
            && col(&a.q6, m, c) == col(&b.q6, m, c)
            && col(&a.s8, m, c) == col(&b.s8, m, c)
            && bits_equal(col(&a.d8, m, c), col(&b.d8, m, c))
    }

    /// `norm_quant`'s refusals (module doc), each launch labelled with
    /// layer 13: whether they held.
    fn norm_quant_refusals(
        gpu: &Gpu,
        norm: &FusedKernels,
        rows: &[f32],
        gain: &[f32],
        eps: f32,
    ) -> Result<bool, GateError> {
        let kk = gain.len();
        let m = 3usize;
        if rows.len() < m * kk {
            return Err(format!(
                "the refusal columns want {m} tokens of {kk}, got {}",
                rows.len()
            )
            .into());
        }
        let layer = 13usize;
        let sink = gpu.layer_sink(layer)?;
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::NormQuant));
        let show = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let mut ok = true;

        let x = &rows[..m * kk];
        let before = gpu.fault()?;
        let (yc, qc) = norm_quant_run(gpu, norm, x, gain, eps, sink)?;
        let after_clean = gpu.fault()?;
        let mut xb = x.to_vec();
        xb[kk + 77] = f32::NAN;
        xb[2 * kk + 77] = 1.0e30;
        let finite_overflow = xb[2 * kk..].iter().all(|v| v.is_finite());
        let (yb, qb) = norm_quant_run(gpu, norm, &xb, gain, eps, sink)?;
        let word = gpu.take_fault()?;
        let refused = [1usize, 2].iter().all(|&c| {
            col(&yb, m, c).iter().all(|v| v.is_nan())
                && col(&qb.d8, m, c).iter().all(|d| d.is_nan())
        });
        let clean_col = bits_equal(col(&yb, m, 0), col(&yc, m, 0)) && q8_col_equal(&qb, &qc, m, 0);
        let (ya, qa) = norm_quant_run(gpu, norm, x, gain, eps, sink)?;
        let clean_again =
            bits_equal(&ya, &yc) && q8_col_equal(&qa, &qc, 1, 0) && gpu.fault()?.is_none();
        let pass = before.is_none()
            && after_clean.is_none()
            && finite_overflow
            && word == want
            && refused
            && clean_col
            && clean_again;
        println!(
            "fault op=norm_quant m={m} column 1 NaN, column 2 finite={finite_overflow} with a square \
             that overflows: word \"{}\" (want \"{}\", clean before {} and after the clean run {}) \
             refused columns' normed values and scales NaN {refused}, column 0 bit-identical \
             {clean_col}, clean rerun bits and word clean {clean_again} {}",
            show(word),
            show(want),
            before.is_none(),
            after_clean.is_none(),
            verdict(pass)
        );
        ok &= pass;

        // One column, its gain +inf at value 200: block 1 alone is refused.
        let x = &rows[..kk];
        let (yc, qc) = norm_quant_run(gpu, norm, x, gain, eps, sink)?;
        let mut gb = gain.to_vec();
        gb[200] = f32::INFINITY;
        let (yb, qb) = norm_quant_run(gpu, norm, x, &gb, eps, sink)?;
        let word = gpu.take_fault()?;
        let block = 128..256;
        let block_nan = yb[block.clone()].iter().all(|v| v.is_nan()) && qb.d8[1].is_nan();
        let rest_y = bits_equal(&yb[..block.start], &yc[..block.start])
            && bits_equal(&yb[block.end..], &yc[block.end..]);
        let rest_q = bits_equal(&qb.d8[..1], &qc.d8[..1])
            && bits_equal(&qb.d8[2..], &qc.d8[2..])
            && qb.s8[..4] == qc.s8[..4]
            && qb.s8[8..] == qc.s8[8..];
        let clean_after = gpu.fault()?.is_none();
        let pass = word == want && block_nan && rest_y && rest_q && clean_after;
        println!(
            "fault op=norm_quant gain[200]=inf: word \"{}\" (want \"{}\") block 1's normed values and \
             scale NaN {block_nan}, other blocks' normed values {rest_y} and scales and group sums \
             {rest_q} bit-identical, word clean after {clean_after} {}",
            show(word),
            show(want),
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    /// The norm-fused section (module doc).
    fn norm_section(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        k: &RouterKernels,
        eps: f32,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let cpu = &sets()?[0].1;
        let mut ok = true;
        let mut out_s = RouterOut::new(stream)?;
        let mut out_f = RouterOut::new(stream)?;
        let norm = FusedKernels::load(gpu.context())?;
        let mut last = None;
        for (l, src) in [(13usize, "l_out-12"), (47, "l_out-46")] {
            let w = router_weight(gpu, gguf, l)?;
            let gain = ffn_gain(gpu, gguf, l)?;
            let nl = NormLayer {
                w: &w,
                gain: &gain,
                eps,
                norm: &norm,
            };
            let kk = w.cols();
            let rows = ref_tensor_logical_in(&cpu.dir, cpu.tensor(src, 0)?)?;
            if rows.is_empty() || rows.len() % kk != 0 {
                return Err(format!("{src} holds {} values, want rows of {kk}", rows.len()).into());
            }
            let m = rows.len() / kk;
            let mut same = 0usize;
            let mut tickets_ok = true;
            for t in 0..m {
                let x = DeviceBuffer::from_host(stream, &rows[t * kk..(t + 1) * kk])?;
                let want = norm_split(gpu, k, &nl, &x, &mut out_s)?;
                let (got, tickets) = norm_fused(gpu, k, &nl, &x, &mut out_f)?;
                let eq = q8_equal(&got.act, &want.act) && fused_equal(&got.routed, &want.routed);
                same += usize::from(eq);
                tickets_ok &= tickets == 0;
                if !eq {
                    println!(
                        "norm layer={l} src={src} t={t}: q8_1_bits={} logits_bits={} ids={:?} vs {:?} FAIL",
                        q8_equal(&got.act, &want.act),
                        bits_equal(&got.routed.logits, &want.routed.logits),
                        got.routed.routed.ids,
                        want.routed.routed.ids
                    );
                }
            }
            let pass = same == m && tickets_ok;
            println!(
                "norm layer={l} src={src} vs norm_quant+router_fused: {same}/{m} tokens bit-identical \
                 (q8_1 planes, logits, probs, ids, weights) tickets_zero={tickets_ok} {}",
                verdict(pass)
            );
            ok &= pass;
            last = Some((l, w, gain, rows));
        }
        let (last_l, w, gain, rows) = last.ok_or("no layer ran")?;
        let kk = w.cols();
        let x0 = rows[..kk].to_vec();
        let nl = NormLayer {
            w: &w,
            gain: &gain,
            eps,
            norm: &norm,
        };

        // A refused input: both paths raise the norm's site and the router's,
        // refuse every block's scale, write the same bytes and refuse the
        // routing.
        let want = Some(Fault::of_sites(
            LAYER_NONE,
            &[FaultSite::NormQuant, FaultSite::Router],
        ));
        for (name, v) in [("nan", f32::NAN), ("overflow", 1.0e30f32)] {
            let mut xn = x0.clone();
            xn[77] = v;
            let xn = DeviceBuffer::from_host(stream, &xn)?;
            gpu.clear_fault()?;
            let split = norm_split(gpu, k, &nl, &xn, &mut out_s)?;
            let f_split = gpu.take_fault()?;
            let (fused, tickets) = norm_fused(gpu, k, &nl, &xn, &mut out_f)?;
            let f_fused = gpu.take_fault()?;
            let scales_nan =
                split.act.d8.iter().all(|d| d.is_nan()) && fused.act.d8.iter().all(|d| d.is_nan());
            let q8_same = q8_equal(&split.act, &fused.act);
            let refused = routing_nan(&split.routed) && routing_nan(&fused.routed);
            let pass = f_split == want
                && f_fused == want
                && scales_nan
                && q8_same
                && refused
                && tickets == 0;
            println!(
                "norm {name}-input: split fault={f_split:?} fused fault={f_fused:?} (want {want:?}) \
                 every scale NaN {scales_nan}, q8_1 bytes equal {q8_same}, routing refused {refused}, \
                 tickets={tickets} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        let gain_host = ffn_gain_host(gguf, last_l)?;
        ok &= norm_quant_refusals(gpu, &norm, &rows, &gain_host, eps)?;

        // The graph: one node, two replays equal to the eager launch.
        let x = DeviceBuffer::from_host(stream, &x0)?;
        let (eager, _) = norm_fused(gpu, k, &nl, &x, &mut out_f)?;
        let mut act = Q8Act::with_k(stream, 1, w.cols())?;
        let graph = gpu.capture(|s| {
            k.enqueue_norm_fused(
                s,
                &w,
                &x,
                &gain,
                eps,
                &mut act,
                gpu.unlabelled_sink(),
                &mut out_f,
            )
        })?;
        let mut replays_ok = true;
        let mut tickets_after = Vec::new();
        for _ in 0..2 {
            graph.launch(stream)?;
            stream.synchronize()?;
            let r = NormRouted {
                act: readback_q8act(stream, &act)?,
                routed: read_fused(stream, &out_f, 1)?,
            };
            replays_ok &= q8_equal(&r.act, &eager.act) && fused_equal(&r.routed, &eager.routed);
            tickets_after.push(out_f.tickets(stream)?);
        }
        let nodes = graph.node_count();
        drop(graph);
        let pass = replays_ok && nodes == 1 && tickets_after.iter().all(|&t| t == 0);
        println!(
            "graph op=qwen3moe_router_norm two_replays_bit_identical={replays_ok} \
             tickets_after={tickets_after:?} graph_nodes={nodes} {}",
            verdict(pass)
        );
        Ok(ok && pass)
    }

    /// The model layer the refusal cases label their launches with, so the
    /// word shows each launcher carried its caller's label.
    const FAULT_LAYER: usize = 13;

    /// A finite activation whose products with the router weights stay
    /// finite while one row's dot does not: `Σ_i |w_ei| · BIG` is past
    /// `f32::MAX` for every row of the file.
    const BIG: f32 = 1.0e38;

    /// The expert whose logit the overflow column drives to `+inf`.
    const HOT: usize = 42;

    /// The overflow column for router row `e`: `BIG` with the sign of each of
    /// the row's weights, so every value is finite and the row's logit is
    /// `Σ_i |w_ei| · BIG = +inf`.
    fn overflow_col(wh: &[f32], k: usize, e: usize) -> Vec<f32> {
        wh[e * k..(e + 1) * k]
            .iter()
            .map(|&w| if w < 0.0 { -BIG } else { BIG })
            .collect()
    }

    /// Token `t` of `a` and `b` bit for bit: logits, probabilities, ids and
    /// weights.
    fn token_equal(a: &Fused, b: &Fused, t: usize) -> bool {
        let (e, u) = (
            t * N_EXPERT..(t + 1) * N_EXPERT,
            t * N_USED..(t + 1) * N_USED,
        );
        bits_equal(&a.logits[e.clone()], &b.logits[e.clone()])
            && bits_equal(&a.routed.probs[e.clone()], &b.routed.probs[e])
            && a.routed.ids[u.clone()] == b.routed.ids[u.clone()]
            && bits_equal(&a.routed.weights[u.clone()], &b.routed.weights[u])
    }

    /// Whether token `t` of `f` is refused: NaN probabilities and weights,
    /// and the ids `kept` held for it before the launch.
    fn token_refused(f: &Fused, t: usize, kept: &Fused) -> bool {
        let (e, u) = (
            t * N_EXPERT..(t + 1) * N_EXPERT,
            t * N_USED..(t + 1) * N_USED,
        );
        f.routed.probs[e].iter().all(|v| v.is_nan())
            && f.routed.weights[u.clone()].iter().all(|v| v.is_nan())
            && f.routed.ids[u.clone()] == kept.routed.ids[u]
    }

    /// A multi-token refusal's verdict: `got` against the clean run `clean`
    /// of the same launch into the same buffers, the tokens in `bad` refused
    /// and every other token bit for bit the clean run's.
    fn tokens_verdict(got: &Fused, clean: &Fused, m: usize, bad: &[usize]) -> (bool, bool) {
        let refused = bad.iter().all(|&t| token_refused(got, t, clean));
        let others = (0..m)
            .filter(|t| !bad.contains(t))
            .all(|t| token_equal(got, clean, t));
        (refused, others)
    }

    /// The router's refusals (module doc): every entry, each undefined
    /// input, the word before and after a clean run.
    fn fault_section(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        k: &RouterKernels,
        eps: f32,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sink = gpu.layer_sink(FAULT_LAYER)?;
        let want = Some(Fault::at(u32::try_from(FAULT_LAYER)?, FaultSite::Router));
        let (wh, kk) = router_weight_host(gguf, FAULT_LAYER)?;
        let w = DeviceTensor::upload(stream, &wh, N_EXPERT, kk)?;
        let cpu = &sets()?[0].1;
        let logits = ref_tensor_logical_in(&cpu.dir, cpu.tensor("ffn_moe_logits-13", 0)?)?;
        let mut cols = ref_tensor_logical_in(&cpu.dir, cpu.tensor("ffn_inp_normed-13", 0)?)?;
        cols.extend(ref_tensor_logical_in(
            &cpu.dir,
            cpu.tensor("ffn_inp_normed-14", 0)?,
        )?);
        let ub_n = MAX_TOKENS + 1;
        if logits.len() < N_EXPERT || cols.len() < ub_n * kk {
            return Err(format!(
                "ffn_moe_logits-13 holds {}, ffn_inp_normed-13/-14 {} values; want {N_EXPERT} and \
                 {ub_n} x {kk}",
                logits.len(),
                cols.len()
            )
            .into());
        }
        let mut ok = true;

        // The routing alone: a NaN, a +inf and a −inf logit.
        let before = gpu.fault()?;
        let mut out = RouterOut::new(stream)?;
        let x = DeviceBuffer::from_host(stream, &logits[..N_EXPERT])?;
        let clean = route(k, stream, &x, sink, &mut out)?;
        let after_clean = gpu.fault()?;
        let pass = before.is_none() && after_clean.is_none();
        println!(
            "fault clean: word before {before:?} and after a clean routing {after_clean:?} (want \
             None) {}",
            verdict(pass)
        );
        ok &= pass;
        for (name, v) in [
            ("nan", f32::NAN),
            ("+inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            let mut l = logits[..N_EXPERT].to_vec();
            l[37] = v;
            let r = route(
                k,
                stream,
                &DeviceBuffer::from_host(stream, &l)?,
                sink,
                &mut out,
            )?;
            let word = gpu.take_fault()?;
            let refused =
                r.probs.iter().all(|p| p.is_nan()) && r.weights.iter().all(|p| p.is_nan());
            let ids_kept = r.ids == clean.ids;
            let again = route(k, stream, &x, sink, &mut out)?;
            let clean_after = gpu.fault()?.is_none()
                && bits_equal(&again.probs, &clean.probs)
                && again.ids == clean.ids
                && bits_equal(&again.weights, &clean.weights);
            let pass = word == want && refused && ids_kept && clean_after;
            println!(
                "fault op=qwen3moe_router logit[37]={name}: word \"{}\" (want \"{}\") probs_weights_nan={refused} \
                 ids_kept={ids_kept} clean_rerun_word_none_and_bits={clean_after} {}",
                word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
                want.map_or_else(String::new, |f| f.to_string()),
                verdict(pass)
            );
            ok &= pass;
        }

        // The fused launch: token 2 of 5 overflows to +inf from a finite row.
        let m = 5usize;
        let bad_t = 2usize;
        let mut fout = RouterOut::with_tokens(stream, MAX_TOKENS)?;
        let xc = DeviceBuffer::from_host(stream, &cols[..m * kk])?;
        k.enqueue_fused(stream, &w, &xc, m, sink, &mut fout)?;
        stream.synchronize()?;
        let clean_f = read_fused(stream, &fout, m)?;
        let clean_word = gpu.fault()?;
        let hot = overflow_col(&wh, kk, HOT);
        let mut xb = cols[..m * kk].to_vec();
        xb[bad_t * kk..(bad_t + 1) * kk].copy_from_slice(&hot);
        let finite_in = xb.iter().all(|v| v.is_finite());
        k.enqueue_fused(
            stream,
            &w,
            &DeviceBuffer::from_host(stream, &xb)?,
            m,
            sink,
            &mut fout,
        )?;
        stream.synchronize()?;
        let got = read_fused(stream, &fout, m)?;
        let word = gpu.take_fault()?;
        let tickets = fout.tickets(stream)?;
        let hot_logit = got.logits[bad_t * N_EXPERT + HOT];
        let (refused, others) = tokens_verdict(&got, &clean_f, m, &[bad_t]);
        let pass = clean_word.is_none()
            && finite_in
            && hot_logit == f32::INFINITY
            && word == want
            && refused
            && others
            && tickets == 0;
        println!(
            "fault op=qwen3moe_router_fused m={m} token {bad_t} column finite={finite_in} logit[{HOT}]={hot_logit} \
             (want inf): word {word:?} (want {want:?}, clean run {clean_word:?}) token refused={refused} \
             other tokens bit-identical={others} tickets={tickets} {}",
            verdict(pass)
        );
        ok &= pass;

        // The ubatch router: token 8 of 9 (the second chunk) overflows.
        let mut ub = RouterOut::for_ubatch(stream, ub_n)?;
        let xu = DeviceBuffer::from_host(stream, &cols[..ub_n * kk])?;
        k.enqueue_ubatch(stream, &w, &xu, ub_n, sink, &mut ub)?;
        stream.synchronize()?;
        let clean_u = read_fused(stream, &ub, ub_n)?;
        let clean_word = gpu.fault()?;
        let mut xb = cols[..ub_n * kk].to_vec();
        xb[(ub_n - 1) * kk..].copy_from_slice(&hot);
        k.enqueue_ubatch(
            stream,
            &w,
            &DeviceBuffer::from_host(stream, &xb)?,
            ub_n,
            sink,
            &mut ub,
        )?;
        stream.synchronize()?;
        let got = read_fused(stream, &ub, ub_n)?;
        let word = gpu.take_fault()?;
        let (refused, others) = tokens_verdict(&got, &clean_u, ub_n, &[ub_n - 1]);
        let pass = clean_word.is_none() && word == want && refused && others;
        println!(
            "fault op=qwen3moe_router_logits+route n={ub_n} token {} overflows: word {word:?} (want {want:?}, \
             clean run {clean_word:?}) token refused={refused} other tokens bit-identical={others} {}",
            ub_n - 1,
            verdict(pass)
        );
        ok &= pass;

        // A NaN router weight: every token of the fused launch is refused, and
        // the norm-fused launch's routing too, its q8_1 bytes the clean run's.
        let mut wn = wh.clone();
        wn[17 * kk + 3] = f32::NAN;
        let wnd = DeviceTensor::upload(stream, &wn, N_EXPERT, kk)?;
        k.enqueue_fused(stream, &wnd, &xc, m, sink, &mut fout)?;
        stream.synchronize()?;
        let got = read_fused(stream, &fout, m)?;
        let word = gpu.take_fault()?;
        let all: Vec<usize> = (0..m).collect();
        let (refused, _) = tokens_verdict(&got, &clean_f, m, &all);
        let pass = word == want && refused;
        println!(
            "fault op=qwen3moe_router_fused m={m} router weight [17][3]=NaN: word {word:?} (want {want:?}) \
             every token refused={refused} {}",
            verdict(pass)
        );
        ok &= pass;
        let gain = ffn_gain(gpu, gguf, FAULT_LAYER)?;
        let x0 = DeviceBuffer::from_host(stream, &cols[..kk])?;
        let mut run_norm =
            |w: &DeviceTensor<f32>| -> Result<(NormRouted, Option<Fault>), GateError> {
                let mut act = Q8Act::with_k(stream, 1, kk)?;
                k.enqueue_norm_fused(stream, w, &x0, &gain, eps, &mut act, sink, &mut out)?;
                stream.synchronize()?;
                let r = NormRouted {
                    act: readback_q8act(stream, &act)?,
                    routed: read_fused(stream, &out, 1)?,
                };
                Ok((r, gpu.take_fault()?))
            };
        let (clean_n, clean_word) = run_norm(&w)?;
        let (got, word) = run_norm(&wnd)?;
        let q8_same = q8_equal(&got.act, &clean_n.act);
        let refused = token_refused(&got.routed, 0, &clean_n.routed);
        let pass = clean_word.is_none() && word == want && refused && q8_same;
        println!(
            "fault op=qwen3moe_router_norm router weight [17][3]=NaN: word {word:?} (want {want:?}, clean run \
             {clean_word:?}) token refused={refused} q8_1 bytes = clean run's {q8_same} {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
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
        let sink = gpu.unlabelled_sink();
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
                    let a = route(&k, stream, &x, sink, &mut out)?;
                    let b = route(&k, stream, &x, sink, &mut out)?;
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
            let a = route(&k, stream, &x, sink, &mut out)?;
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
        let eager = route(&k, stream, &x, sink, &mut out)?;
        out.probs.zero_async(stream)?;
        out.ids.zero_async(stream)?;
        out.weights.zero_async(stream)?;
        stream.synchronize()?;
        let graph = gpu.capture(|s| k.enqueue(s, &x, sink, &mut out))?;
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
            k.enqueue_fused(stream, &w, &x, m, sink, &mut fout)?;
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
        k.enqueue_fused(stream, &w, &x, MAX_TOKENS, sink, &mut fout)?;
        stream.synchronize()?;
        let eager = read_fused(stream, &fout, MAX_TOKENS)?;
        fout.logits.zero_async(stream)?;
        fout.probs.zero_async(stream)?;
        fout.ids.zero_async(stream)?;
        fout.weights.zero_async(stream)?;
        stream.synchronize()?;
        let graph = gpu.capture(|s| k.enqueue_fused(s, &w, &x, MAX_TOKENS, sink, &mut fout))?;
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

        ok &= norm_section(&gpu, &gguf, &k, hp.rms_eps)?;
        ok &= fault_section(&gpu, &gguf, &k, hp.rms_eps)?;

        println!("gate_qwen3moe_router: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
