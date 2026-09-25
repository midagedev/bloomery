//! GPU gate for qwen3moe's routed down projection: the new Q6_K `_sel`
//! kernel (`q6k_sel::q6k_gemv_sel`) on its own, then every layer's
//! `ffn_moe_down-L` tap through the kernel its stack's type takes (Q6_K on
//! half the layers, `q4k_gemv_sel` on the rest).
//!
//! 1. The body: `output.weight` (Q6_K, K = 2048) viewed as a stack of four
//!    2048-row experts; for a sel vector with a repeated id, slot `s` equals
//!    `q6k_gemv` over expert `sel[s]`'s rows alone against a one-column
//!    activation quantized from column `s` alone, bit for bit; an id past
//!    the stack leaves its slot untouched; a rerun is bit-identical.
//! 2. The kernel's arithmetic on the real K = 768 stacks (630-byte rows,
//!    every other row at 2 mod 4): each slot against the f64 dot of the
//!    dequantized rows with the q8_1-reconstructed activation within
//!    [`KERNEL_BAND`] of the slot's largest value.
//! 3. The tap, every token of every set, `ffn_moe_gate_par-L` in with ik's
//!    `ffn_moe_topk-L` as `sel`. ik's rule simulated (the f64 dot of the
//!    dequantized rows with ik's q8_2 reconstruction of each column,
//!    `ik_q8_2::reconstruct`) against the dump within `F32_TERMS` roundings
//!    of the dot's magnitude per output — which proves rows, experts and
//!    columns on ik's own values. Then ours against ik: the difference the
//!    two activation quantizations predict, `Σ w·(x̂_ours − x̂_ik)` computed
//!    exactly, must match the measured difference within both sides'
//!    rounding (`F32_TERMS` of each side's magnitude). The magnitude is
//!    `Σ|w·x̂|` for a Q6_K stack and `Σ(|d1·q·x̂| + |m1·x̂|)` for a Q4_K one,
//!    whose dots sum the scale and min terms apart (the gate·up gate's
//!    bound; the split must reproduce the dequantized row bit for bit). The
//!    plain relative distance to ik is printed.
//!
//! And the kernel as a captured graph: one node, the replay equal to the
//! eager launch.
//!
//! 4. The head's projection with its argmax folded in
//!    (`head_argmax::qwen3moe_head_q6k_argmax`) on the whole `output.weight`
//!    against the shared head's two launches, `q6k_gemv` then
//!    `argmax_fault`: logits BIT-EQUAL, the readback pair (token, fault
//!    word) EQUAL, and the block reduction's key and ticket back at their
//!    seeds after every launch. Cases: the plain weight, twice; the winning
//!    row copied into row 3 (a tie across blocks) and into its pair row
//!    `best ^ 1` (a tie inside a block) — the lower index must win, and the
//!    host's scan of the reference logits must agree, and the packed key
//!    must order 60 edge candidates as the host rule does; the winning row's
//!    scales set to NaN (a NaN logit is never taken); a non-finite
//!    activation, whose raised fault word both readbacks must carry. As a
//!    captured graph: one node, two replays equal to the eager launch.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_down: built without the `gpu` feature; see `just gate-gpu-qwen3moe-down`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_down", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::head_argmax::{
        HeadArgmaxKernels, HeadArgmaxState, argmax_key,
    };
    use bloomery_gpu::q4k_sel::Q4kSelKernels;
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE, Q8Act};
    use bloomery_gpu_gates::qwen3moe::{q4k_parts, sets};
    use bloomery_gpu_gates::rounding::gamma;
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, activations, bits_equal, bytes_to_words, checks_failed, ik_q8_2,
        open_model, q8_1_dequant, ref_tensor_logical_in, tensor_bytes, tensor_bytes_as,
        topk_ids_logical_within, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, dequant_row};
    use model::arch::qwen3moe::names;

    /// f32 roundings each side's dot is allowed per output, on `Σ|w·x̂|`:
    /// ours adds a slot's per-lane products over at most `ceil(n_sb/2)`
    /// iterations (three roundings each: two scale products and the fused
    /// add) and five butterfly levels; ik adds 24 block products of 32 and
    /// its lane tree. Forty covers both at K = 768 with room.
    const F32_TERMS: usize = 40;

    /// What `y` holds before a launch, so a slot the kernel leaves alone
    /// reads back as these bits.
    const SENT: f32 = 1.0e30;

    /// A stack on the card: its rows as the byte stream in words, zero-padded
    /// at the tail to a whole number of words per row.
    fn upload(gpu: &Gpu, bytes: &[u8], rows: usize) -> Result<DeviceTensor<u32>, GateError> {
        let mut words = bytes_to_words(bytes);
        words.resize(words.len().div_ceil(rows) * rows, 0);
        let cols = words.len() / rows;
        Ok(DeviceTensor::upload(gpu.stream(), &words, rows, cols)?)
    }

    /// `x`'s `m` columns of `k` quantized on the card.
    fn quantize(gpu: &Gpu, x: &[f32], m: usize, k: usize) -> Result<Q8Act, GateError> {
        let mut act = Q8Act::with_k(gpu.stream(), m, k)?;
        let xd = DeviceBuffer::from_host(gpu.stream(), x)?;
        gpu.enqueue_quantize_q8_1(&xd, &mut act)?;
        gpu.stream().synchronize()?;
        Ok(act)
    }

    /// One head readback: the logits and the (token, fault word, site mask) triple.
    struct HeadRun {
        logits: Vec<f32>,
        out: Vec<u32>,
    }

    /// The shared head's two launches over `w` and `act`: `q6k_gemv`, then
    /// `argmax_fault`.
    fn head_ref(gpu: &Gpu, w: &DeviceTensor<u32>, act: &Q8Act) -> Result<HeadRun, GateError> {
        let stream = gpu.stream();
        let n = w.rows();
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; n])?;
        let mut out = DeviceBuffer::from_host(stream, &[7u32, 7, 7])?;
        gpu.enqueue_gemv_q6k(w, act, &mut y)?;
        gpu.elem()
            .enqueue_argmax_fault(stream, &y, n, gpu.unlabelled_sink(), &mut out)?;
        stream.synchronize()?;
        Ok(HeadRun {
            logits: y.to_host_vec(stream)?,
            out: out.to_host_vec(stream)?,
        })
    }

    /// The fused launch over `w` and `act`, and whether the key and the
    /// ticket stood at their seeds after it.
    fn head_fused(
        gpu: &Gpu,
        hk: &HeadArgmaxKernels,
        st: &mut HeadArgmaxState,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
    ) -> Result<(HeadRun, bool), GateError> {
        let stream = gpu.stream();
        let n = w.rows();
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; n])?;
        let mut out = DeviceBuffer::from_host(stream, &[7u32, 7, 7])?;
        hk.enqueue(stream, w, act, gpu.unlabelled_sink(), &mut y, st, &mut out)?;
        stream.synchronize()?;
        let run = HeadRun {
            logits: y.to_host_vec(stream)?,
            out: out.to_host_vec(stream)?,
        };
        Ok((run, st.at_seed(stream)?))
    }

    /// The host's argmax of `v`, ties to the lower index, NaN never taken —
    /// the rule both launches implement.
    fn host_argmax(v: &[f32]) -> u32 {
        let (mut bv, mut bi) = (f32::NEG_INFINITY, 0u32);
        for (i, &x) in (0u32..).zip(v) {
            if x > bv || (x == bv && i < bi) {
                (bv, bi) = (x, i);
            }
        }
        bi
    }

    /// Whether the packed key orders every pair of edge candidates as the
    /// host rule does: `a` beats `b` exactly when its key is larger. The
    /// values include both zeros (equal to the rule, one key), both
    /// infinities, the smallest subnormals and the extremes; the indices
    /// include 0 and `u32::MAX`.
    fn key_order_ok() -> bool {
        let vals = [
            f32::NEG_INFINITY,
            f32::MIN,
            -1.0,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            1.0,
            f32::MAX,
            f32::INFINITY,
        ];
        let idx = [0u32, 1, 7, 8, 151_935, u32::MAX];
        let cands: Vec<(f32, u32)> = vals
            .iter()
            .flat_map(|&v| idx.iter().map(move |&i| (v, i)))
            .collect();
        cands.iter().all(|&(av, ai)| {
            cands.iter().all(|&(bv, bi)| {
                let beats = av > bv || (av == bv && ai < bi);
                beats == (argmax_key(av, ai) > argmax_key(bv, bi))
            })
        })
    }

    /// Section 4 (module doc): the fused head against the shared head's two
    /// launches on `output.weight`'s bytes.
    fn head_argmax(gpu: &Gpu, bytes: &[u8], k: usize) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let key_ok = key_order_ok();
        println!(
            "head_argmax key order: every pair of 60 edge candidates (both zeros, both infinities, subnormals, \
             index 0 and u32::MAX) ordered as the host rule {}",
            verdict(key_ok)
        );
        let rb = 210 * k / 256;
        let n = bytes.len() / rb;
        let hk = HeadArgmaxKernels::load(gpu.context())?;
        let mut st = HeadArgmaxState::new(stream)?;
        let act = quantize(gpu, &activations(k, 1, 104_729), 1, k)?;
        let mut ok = key_ok;
        let mut judge = |case: &str, w: &DeviceTensor<u32>, act: &Q8Act, want: Option<u32>| {
            let r = head_ref(gpu, w, act)?;
            let (f, seed) = head_fused(gpu, &hk, &mut st, w, act)?;
            let logits = bits_equal(&f.logits, &r.logits);
            let host = host_argmax(&r.logits);
            let want_ok = want.is_none_or(|t| t == r.out[0]);
            let pass = logits && f.out == r.out && host == r.out[0] && want_ok && seed;
            println!(
                "head_argmax case={case} rows={n} logits_bits={logits} fused={:?} shared={:?} host_token={host} \
                 want={want:?} key_ticket_at_seed={seed} {}",
                f.out,
                r.out,
                verdict(pass)
            );
            ok &= pass;
            Ok::<u32, GateError>(r.out[0])
        };
        let words = bytes_to_words(bytes);
        let w = DeviceTensor::upload(stream, &words, n, rb / 4)?;
        let best = judge("plain", &w, &act, None)?;
        judge("plain-rerun", &w, &act, Some(best))?;
        drop(w);

        let b = best as usize;
        for (case, dst) in [("tie-across-blocks", 3usize), ("tie-in-block", b ^ 1)] {
            let mut tb = bytes.to_vec();
            tb.copy_within(b * rb..(b + 1) * rb, dst * rb);
            let w = DeviceTensor::upload(stream, &bytes_to_words(&tb), n, rb / 4)?;
            judge(case, &w, &act, Some(dst.min(b) as u32))?;
        }

        // The winning row's super-block scales `d` (f16 at byte 208 of each)
        // set to NaN: its logit is NaN and the next row wins.
        let mut nb = bytes.to_vec();
        for sb in 0..k / 256 {
            let at = b * rb + sb * 210 + 208;
            nb[at..at + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
        }
        let w = DeviceTensor::upload(stream, &bytes_to_words(&nb), n, rb / 4)?;
        let nan_best = judge("nan-best-row", &w, &act, None)?;
        ok &= nan_best != best;
        drop(w);

        let w = DeviceTensor::upload(stream, &words, n, rb / 4)?;
        // The graph: one node, two replays equal to the eager launch.
        let eager = head_ref(gpu, &w, &act)?;
        let mut yg = DeviceBuffer::from_host(stream, &vec![SENT; n])?;
        let mut og = DeviceBuffer::from_host(stream, &[7u32, 7, 7])?;
        let graph = gpu.capture(|s| {
            hk.enqueue(
                s,
                &w,
                &act,
                gpu.unlabelled_sink(),
                &mut yg,
                &mut st,
                &mut og,
            )
        })?;
        let mut replays = true;
        for _ in 0..2 {
            graph.launch(stream)?;
            stream.synchronize()?;
            replays &= bits_equal(&yg.to_host_vec(stream)?, &eager.logits)
                && og.to_host_vec(stream)? == eager.out
                && st.at_seed(stream)?;
        }
        let nodes = graph.node_count();
        drop(graph);
        let pass = replays && nodes == 1;
        println!(
            "graph op=qwen3moe_head_q6k_argmax two_replays_equal_and_at_seed={replays} graph_nodes={nodes} {}",
            verdict(pass)
        );
        ok &= pass;

        // A non-finite activation raises the fault word in the quantizer;
        // both readbacks copy it.
        let mut xf = activations(k, 1, 104_729);
        xf[17] = f32::NAN;
        let act_f = quantize(gpu, &xf, 1, k)?;
        let r = head_ref(gpu, &w, &act_f)?;
        let (f, seed) = head_fused(gpu, &hk, &mut st, &w, &act_f)?;
        gpu.clear_fault()?;
        let raised = Fault::from_words(r.out[1], r.out[2]).is_some();
        let pass = raised && f.out == r.out && seed;
        println!(
            "head_argmax case=fault-word fused={:?} shared={:?} raised={raised} (want raised, equal) \
             key_ticket_at_seed={seed} {}",
            f.out,
            r.out,
            verdict(pass)
        );
        ok &= pass;
        println!("head_argmax: {}", verdict(ok));
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gguf = open_model()?;
        let gpu = Gpu::new()?;
        let q6 = Q6kSelKernels::load(gpu.context(), gpu.fault_word())?;
        let q4 = Q4kSelKernels::load(gpu.context(), gpu.fault_word())?;
        let stream = gpu.stream();
        println!("gate_qwen3moe_down: device {}", gpu.device_name()?);
        let mut ok = true;

        // ---- 1. the body against q6k_gemv, on output.weight as 4 experts.
        let (info, bytes) = tensor_bytes_as(&gguf, &names::output(), GgmlType::Q6_K, None)?;
        let k_out = info.dims[0] as usize;
        let rpe = 2048usize;
        let rb = 210 * k_out / 256;
        let stack = upload(&gpu, &bytes[..4 * rpe * rb], 4 * rpe)?;
        let sel_h: [u32; 4] = [3, 0, 2, 3];
        let sel = DeviceBuffer::from_host(stream, &sel_h)?;
        let x = activations(k_out, 4, 7919);
        let act = quantize(&gpu, &x, 4, k_out)?;
        let mut runs = Vec::new();
        for _ in 0..2 {
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; 4 * rpe])?;
            q6.enqueue_gemv_q6k_sel(stream, &stack, &act, &sel, 4, rpe, &mut y)?;
            stream.synchronize()?;
            runs.push(y.to_host_vec(stream)?);
        }
        let mut body_ok = bits_equal(&runs[0], &runs[1]);
        for (s, &id) in sel_h.iter().enumerate() {
            let rows = &bytes[id as usize * rpe * rb..(id as usize + 1) * rpe * rb];
            let w1 = DeviceTensor::upload(stream, &bytes_to_words(rows), rpe, rb / 4)?;
            let a1 = quantize(&gpu, &x[s * k_out..(s + 1) * k_out], 1, k_out)?;
            let mut y1 = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
            gpu.enqueue_gemv_q6k(&w1, &a1, &mut y1)?;
            gpu.stream().synchronize()?;
            let want = y1.to_host_vec(stream)?;
            let same = bits_equal(&runs[0][s * rpe..(s + 1) * rpe], &want);
            body_ok &= same;
            println!("body slot={s} id={id} bit_identical_to_q6k_gemv={same}");
        }
        let sel_oor = DeviceBuffer::from_host(stream, &[1u32, 4, 0, u32::MAX])?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; 4 * rpe])?;
        q6.enqueue_gemv_q6k_sel(stream, &stack, &act, &sel_oor, 4, rpe, &mut y)?;
        stream.synchronize()?;
        let yo = y.to_host_vec(stream)?;
        // Id 4 is past the stack and raises the named fault; u32::MAX is a
        // host-served slot and raises nothing. Taking the word here also
        // leaves it clean for the head's fault-word case below.
        let fault = gpu.take_fault()?;
        let want_fault = Fault::at(LAYER_NONE, FaultSite::ExpertId);
        let untouched = |s: usize| {
            yo[s * rpe..(s + 1) * rpe]
                .iter()
                .all(|v| v.to_bits() == SENT.to_bits())
        };
        let oor_ok = untouched(1)
            && untouched(3)
            && !untouched(0)
            && !untouched(2)
            && fault == Some(want_fault);
        body_ok &= oor_ok;
        println!(
            "body output.weight K={k_out} as 4x{rpe}: rerun and slots vs q6k_gemv, ids 4 and u32::MAX untouched, \
             fault={fault:?} (want expert_id at id 4)={oor_ok} {}",
            verdict(body_ok)
        );
        ok &= body_ok;

        // ---- the graph: the eager launch above, captured.
        let mut yg = DeviceBuffer::from_host(stream, &vec![SENT; 4 * rpe])?;
        let graph =
            gpu.capture(|s| q6.enqueue_gemv_q6k_sel(s, &stack, &act, &sel, 4, rpe, &mut yg))?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let g_same = bits_equal(&yg.to_host_vec(stream)?, &runs[0]);
        let nodes = graph.node_count();
        let pass = g_same && nodes == 1;
        println!(
            "graph op=q6k_gemv_sel eager_vs_graph_bit_identical={g_same} graph_nodes={nodes} {}",
            verdict(pass)
        );
        ok &= pass;
        drop(stack);

        // ---- 4. the head's projection with its argmax.
        ok &= head_argmax(&gpu, bytes, k_out)?;

        // ---- 2 and 3. every layer's stack, every token of every set.
        let sets = sets()?;
        let (mut n_tok, mut worst_k, mut worst_sim, mut worst_pred, mut worst_ik) =
            (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f32);
        let (mut widen, mut parts_bad) = (0.0f64, 0usize);
        let (mut dq, mut mn) = (Vec::new(), Vec::new());
        let mut layers = 0usize;
        loop {
            let name = names::ffn_down_exps(layers);
            if gguf.find(&name).is_none() {
                break;
            }
            let (info, bytes) = tensor_bytes(&gguf, &name)?;
            let [k, rpe, n_exp] = info.dims[..] else {
                return Err(format!("{name} is {:?}, want [K, rows, experts]", info.dims).into());
            };
            let (k, rpe, n_exp) = (k as usize, rpe as usize, n_exp as usize);
            let rb = bytes.len() / (rpe * n_exp);
            let stack = upload(&gpu, bytes, rpe * n_exp)?;
            let mut layer_ok = true;
            let mut row = vec![0.0f32; k];
            for (label, man) in &sets {
                let l = layers;
                let xrow = man.tensor(&format!("ffn_moe_gate_par-{l}"), 0)?;
                let yrow = man.tensor(&format!("ffn_moe_down-{l}"), 0)?;
                let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
                let xs = ref_tensor_logical_in(&man.dir, xrow)?;
                let ys = ref_tensor_logical_in(&man.dir, yrow)?;
                let ids = topk_ids_logical_within(man, trow, n_exp as u32)?;
                let (slots, m) = (xrow.ne[1] as usize, xrow.ne[2] as usize);
                if xrow.ne[0] as usize != k || yrow.ne[0] as usize != rpe || ids.len() != slots * m
                {
                    return Err(format!(
                        "{label} layer {l}: gate_par {:?}, down {:?}, {} ids",
                        xrow.ne,
                        yrow.ne,
                        ids.len()
                    )
                    .into());
                }
                for t in 0..m {
                    let x = &xs[t * slots * k..(t + 1) * slots * k];
                    let ik = &ys[t * slots * rpe..(t + 1) * slots * rpe];
                    let sel_h: Vec<u32> = ids[t * slots..(t + 1) * slots]
                        .iter()
                        .map(|&i| i as u32)
                        .collect();
                    let sel = DeviceBuffer::from_host(stream, &sel_h)?;
                    let act = quantize(&gpu, x, slots, k)?;
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; slots * rpe])?;
                    match info.ty {
                        GgmlType::Q6_K => {
                            q6.enqueue_gemv_q6k_sel(stream, &stack, &act, &sel, slots, rpe, &mut y)?
                        }
                        GgmlType::Q4_K => {
                            q4.enqueue_gemv_q4k_sel(stream, &stack, &act, &sel, slots, rpe, &mut y)?
                        }
                        ty => {
                            return Err(format!(
                                "{name} is {ty:?}; the chain has Q4_K and Q6_K down kernels"
                            )
                            .into());
                        }
                    }
                    stream.synchronize()?;
                    let ours = y.to_host_vec(stream)?;
                    let xo = q8_1_dequant(x, k, slots);
                    let xi = ik_q8_2::reconstruct(x);
                    for s in 0..slots {
                        let e = sel_h[s] as usize;
                        let (mut mx_ref, mut d_ref) = (0.0f64, 0.0f64);
                        for r in 0..rpe {
                            let wr = &bytes[(e * rpe + r) * rb..(e * rpe + r + 1) * rb];
                            dequant_row(info.ty, wr, &mut row)?;
                            let q4 = info.ty == GgmlType::Q4_K;
                            if q4 {
                                q4k_parts(wr, &mut dq, &mut mn);
                            }
                            let (mut dot_o, mut dot_i, mut abs_o, mut abs_i, mut plain) =
                                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                            for i in 0..k {
                                let w = f64::from(row[i]);
                                let mag = if q4 {
                                    if (dq[i] - mn[i]).to_bits() != row[i].to_bits() {
                                        parts_bad += 1;
                                    }
                                    f64::from(dq[i]).abs() + f64::from(mn[i]).abs()
                                } else {
                                    w.abs()
                                };
                                let (a, b) = (f64::from(xo[s * k + i]), f64::from(xi[s * k + i]));
                                dot_o += w * a;
                                dot_i += w * b;
                                abs_o += mag * a.abs();
                                abs_i += mag * b.abs();
                                plain += (w * a).abs();
                            }
                            if q4 {
                                widen = widen.max(abs_o / plain.max(f64::MIN_POSITIVE));
                            }
                            let o = f64::from(ours[s * rpe + r]);
                            let iv = f64::from(ik[s * rpe + r]);
                            mx_ref = mx_ref.max(dot_o.abs());
                            d_ref = d_ref.max((o - dot_o).abs());
                            let tol_i = gamma(F32_TERMS) * abs_i;
                            let tol = gamma(F32_TERMS) * (abs_o + abs_i);
                            let sim = (iv - dot_i).abs();
                            let gap = ((o - iv) - (dot_o - dot_i)).abs();
                            worst_sim = worst_sim.max(sim / tol_i.max(f64::MIN_POSITIVE));
                            worst_pred = worst_pred.max(gap / tol.max(f64::MIN_POSITIVE));
                            if sim > tol_i || gap > tol {
                                layer_ok = false;
                            }
                        }
                        let rel = d_ref / mx_ref.max(f64::MIN_POSITIVE);
                        worst_k = worst_k.max(rel);
                        if rel > f64::from(KERNEL_BAND) {
                            layer_ok = false;
                        }
                    }
                    let mx = ik.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    let d = ours
                        .iter()
                        .zip(ik)
                        .fold(0.0f32, |a, (&p, &q)| a.max((p - q).abs()));
                    worst_ik = worst_ik.max(d / mx);
                    n_tok += 1;
                }
            }
            println!(
                "down layer={layers} type={:?} K={k} rows={rpe} experts={n_exp} row_bytes={rb} {}",
                info.ty,
                verdict(layer_ok)
            );
            ok &= layer_ok;
            layers += 1;
        }
        println!(
            "down tap: {layers} layers, {n_tok} (layer, token) sites — kernel vs q8_1 reference max_rel={worst_k:.3e} \
             (band {KERNEL_BAND:.0e}); ik sim vs dump at {worst_sim:.3} of its rounding bound; ours - ik vs the \
             predicted quantization difference at {worst_pred:.3} of both bounds; plain ours vs ik rel={worst_ik:.3e} \
             (printed)"
        );
        let parts_ok = parts_bad == 0;
        println!(
            "down tap Q4_K magnitude: Σ(|d1·q·x̂| + |m1·x̂|) over Σ|w·x̂| per output at most {widen:.3} \
             (printed); the split off the dequantized row on {parts_bad} values (want 0) {}",
            verdict(parts_ok)
        );
        ok &= parts_ok;
        println!("gate_qwen3moe_down: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
