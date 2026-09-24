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
//!    of `Σ|w·x̂|` per output — which proves rows, experts and columns on
//!    ik's own values. Then ours against ik: the difference the two
//!    activation quantizations predict, `Σ w·(x̂_ours − x̂_ik)` computed
//!    exactly, must match the measured difference within both sides'
//!    rounding (`F32_TERMS` of `Σ|w·x̂|` each). The plain relative distance
//!    to ik is printed.
//!
//! And the kernel as a captured graph: one node, the replay equal to the
//! eager launch.

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
    use bloomery_gpu::q4k_sel::Q4kSelKernels;
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_gates::qwen3moe::sets;
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

    pub fn run() -> Result<(), GateError> {
        let gguf = open_model()?;
        let gpu = Gpu::new()?;
        let q6 = Q6kSelKernels::load(gpu.context())?;
        let q4 = Q4kSelKernels::load(gpu.context())?;
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
        let untouched = |s: usize| {
            yo[s * rpe..(s + 1) * rpe]
                .iter()
                .all(|v| v.to_bits() == SENT.to_bits())
        };
        let oor_ok = untouched(1) && untouched(3) && !untouched(0) && !untouched(2);
        body_ok &= oor_ok;
        println!(
            "body output.weight K={k_out} as 4x{rpe}: rerun and slots vs q6k_gemv, ids 4 and u32::MAX untouched={oor_ok} {}",
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

        // ---- 2 and 3. every layer's stack, every token of every set.
        let sets = sets()?;
        let (mut n_tok, mut worst_k, mut worst_sim, mut worst_pred, mut worst_ik) =
            (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f32);
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
                            let (mut dot_o, mut dot_i, mut abs_o, mut abs_i) =
                                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                            for i in 0..k {
                                let w = f64::from(row[i]);
                                let (a, b) = (f64::from(xo[s * k + i]), f64::from(xi[s * k + i]));
                                dot_o += w * a;
                                dot_i += w * b;
                                abs_o += (w * a).abs();
                                abs_i += (w * b).abs();
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
        println!("gate_qwen3moe_down: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
