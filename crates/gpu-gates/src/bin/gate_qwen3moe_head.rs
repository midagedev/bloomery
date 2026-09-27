//! GPU gate for qwen3moe's head: the projection with its argmax folded in
//! (`head_argmax::qwen3moe_head_q6k_argmax`) on the whole `output.weight`
//! against the shared head's two launches, `q6k_gemv` then `argmax_fault`:
//! logits BIT-EQUAL, the readback pair (token, fault word) EQUAL, and the
//! block reduction's key and ticket back at their seeds after every launch.
//! Cases: the plain weight, twice; the winning row copied into row 3 (a tie
//! across blocks) and into its pair row `best ^ 1` (a tie inside a block) —
//! the lower index must win, and the host's scan of the reference logits
//! must agree, and the packed key must order 60 edge candidates as the host
//! rule does; the winning row's scales set to NaN (a NaN logit is never
//! taken); a non-finite activation, whose raised fault word both readbacks
//! must carry. As a captured graph: one node, two replays equal to the eager
//! launch.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_head: built without the `gpu` feature; see `just gate-gpu-qwen3moe-head`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_head", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::head_argmax::{
        HeadArgmaxKernels, HeadArgmaxState, argmax_key,
    };
    use bloomery_gpu::{DeviceTensor, Fault, Gpu, Q8Act};
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, bytes_to_words, checks_failed, open_model,
        tensor_bytes_as, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::GgmlType;
    use model::arch::qwen3moe::names;

    /// What the logits hold before a launch, so a row the kernel leaves
    /// alone reads back as these bits.
    const SENT: f32 = 1.0e30;

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

    /// The fused head against the shared head's two launches on
    /// `output.weight`'s bytes (module doc): whether every check held.
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
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        bloomery_levers::at_main(&[])?;
        let gguf = open_model()?;
        let gpu = Gpu::new()?;
        println!("gate_qwen3moe_head: device {}", gpu.device_name()?);
        let (info, bytes) = tensor_bytes_as(&gguf, &names::output(), GgmlType::Q6_K, None)?;
        let ok = head_argmax(&gpu, bytes, info.dims[0] as usize)?;
        println!("gate_qwen3moe_head: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
