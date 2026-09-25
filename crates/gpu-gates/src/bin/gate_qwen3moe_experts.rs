//! GPU gate for qwen3moe's three chain kernels beside the phase-2 set: the
//! Q4_K embedding row, the routed experts' gate·up·SwiGLU `_sel`, and the
//! combine (`bloomery_gpu::arch::qwen3moe::experts`).
//!
//! 1. Embedding: `embed_rows_q4k` against `gguf::quant::dequant_row` on the
//!    same rows, bit for bit, for ids 0, 1, the last row and 61 spread
//!    between them; then every oracle set's `inp_embd` from its `inp_tokens`
//!    (ik's GET_ROWS through the same dequantizer), bit for bit; an id past
//!    the table raises `FaultSite::TokenId` and its row is NaN, every value,
//!    the other rows unchanged.
//! 2. Gate·up body: for a sel vector with a repeated id, slot `s` equals
//!    `q4k_gemv` of expert `sel[s]`'s gate rows and up rows alone, combined
//!    by `elem::swiglu` (the same `silu_mul` core), bit for bit; a rerun is
//!    bit-identical. An id past the stack, each in a launch of its own (the
//!    expert count in slot 1, then `u32::MAX` in slot 3: this chain has no
//!    host tier, so `hybrid::HOST` is no exemption here), launched with
//!    layer 13's sink, raises `FaultSite::ExpertId` with that layer and
//!    writes NaN into exactly its slot's rows, every other slot bit for bit
//!    the clean run's; the word is clean before and after a clean run.
//! 3. The tap, every token of every set: `ffn_inp_normed-L` in, ik's
//!    `ffn_moe_topk-L` as `sel`, against `ffn_moe_gate_par-L`. Per output,
//!    `g` and `u` are the f64 dots of the dequantized rows with each side's
//!    activation reconstruction (ours `q8_1_dequant`, ik `ik_q8_2`), and
//!    `h = silu(g)·u` in f64. Each side against its own `h` within
//!    `tol = 1.1·|u|·γ(F32_TERMS)·A_g + |silu(g)|·γ(F32_TERMS)·A_u +
//!    SILU_ULPS·u·|h|` (the dot roundings through `∂h/∂g ≤ 1.1|u|` and
//!    `∂h/∂u = silu(g)`, plus the exponential and the two products). `A` is
//!    the magnitude both sides' f32 sums run over: a Q4_K value is
//!    `d1·q − m1`, and each side sums the scale term and the min term of a
//!    block apart (ours as the two chains of `q4k_coeff`, ik as the block
//!    dot and `d·isum`), so `A = Σ_i (|d1·q_i·x̂_i| + |m1·x̂_i|)`, not
//!    `Σ_i |w_i·x̂_i|`, which a near-cancelling value undercounts; then
//!    ours − ik against the predicted difference `h_ours − h_ik` within
//!    both sides' `tol`. The plain relative distance to ik is printed.
//! 4. Combine: ik's `ffn_moe_down-L` and `ffn_moe_weights_norm-L` in with a
//!    zero residual, against ik's `routed_out-L` (its weighted sum): max
//!    relative distance within `γ(8)` of `Σ|w·d|` per value, and the count
//!    of bit-equal values printed.
//!
//! PIN(2026-09-25): removed — the engine no longer runs `qwen3moe_gate_up_swiglu_quant_q4k` (gate·up with the down's q8_1 folded in through per-group tickets, section 5): decode was slower with it (rig-log 2026-09-25.md#qwen3fuse-regression-nsys), the step quantizes `h` in its own launch again, and the kernel is gone with its check.
//!
//! And each kernel as a captured graph: one node, the replay equal to the
//! eager launch.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_experts: built without the `gpu` feature; see `just gate-gpu-qwen3moe-experts`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_experts", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::experts::{ExpertKernels, GateUpArgs};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, LAYER_NONE, Q8Act};
    use bloomery_gpu_gates::qwen3moe::{q4k_parts, sets};
    use bloomery_gpu_gates::rounding::gamma;
    use bloomery_gpu_gates::{
        GateError, Layout, RowKind, bits_equal, bytes_to_words, checks_failed, ik_q8_2, open_model,
        q8_1_dequant, ref_ints, ref_tensor_logical_in, tensor_bytes_as, topk_ids_logical_within,
        verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, dequant_row};
    use model::arch::qwen3moe::names;

    /// f32 roundings each side's dot is allowed per output, on `Σ|w·x̂|`
    /// (the down gate's count: ours adds each lane's super-block products
    /// over `ceil(n_sb/4)` iterations and five butterfly levels, ik adds 64
    /// block products of 32 and its lane tree; forty covers both at K = 2048).
    const F32_TERMS: usize = 40;

    /// Unit roundings of `|h|` for the SwiGLU itself on either side: the
    /// exponential (a few ulp in ik's vector `expf` and in the device's),
    /// the add, the divide and the two products.
    const SILU_ULPS: f64 = 16.0;

    /// f32 unit roundoff.
    const U: f64 = f32::EPSILON as f64 / 2.0;

    /// What an output buffer holds before a launch, so a slot the kernel
    /// leaves alone reads back as these bits.
    const SENT: f32 = 1.0e30;

    /// A stack on the card: its rows as the byte stream in words.
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

    fn silu64(g: f64) -> f64 {
        g / (1.0 + (-g).exp())
    }

    pub fn run() -> Result<(), GateError> {
        let gguf = open_model()?;
        let gpu = Gpu::new()?;
        let ex = ExpertKernels::load(gpu.context())?;
        let stream = gpu.stream();
        println!("gate_qwen3moe_experts: device {}", gpu.device_name()?);
        let sets = sets()?;
        let mut ok = true;
        ok &= embed(&gpu, &gguf, &sets)?;
        ok &= gate_up_body(&gpu, &gguf, &ex)?;
        ok &= gate_up_tap(&gpu, &gguf, &ex, &sets)?;
        ok &= combine(&gpu, &ex, &sets, stream)?;
        println!("gate_qwen3moe_experts: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ------------------------------------------------------------ 1. embed

    fn embed(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        sets: &[(&'static str, bloomery_gpu_gates::RefManifest)],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (info, bytes) = tensor_bytes_as(gguf, &names::token_embd(), GgmlType::Q4_K, None)?;
        let (k, n_rows) = (info.dims[0] as usize, info.dims[1] as usize);
        let rb = bytes.len() / n_rows;
        let table = upload(gpu, bytes, n_rows)?;
        let mut ids: Vec<u32> = vec![0, 1, (n_rows - 1) as u32];
        ids.extend((1..62u64).map(|i| ((i * 2_654_435_761) % n_rows as u64) as u32));
        let run = |ids: &[u32]| -> Result<Vec<f32>, GateError> {
            let idb = DeviceBuffer::from_host(stream, ids)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; ids.len() * k])?;
            gpu.elem()
                .enqueue_embed_rows_q4k(stream, &table, &idb, &mut y)?;
            stream.synchronize()?;
            Ok(y.to_host_vec(stream)?)
        };
        let got = run(&ids)?;
        let mut want = vec![0.0f32; ids.len() * k];
        for (t, &id) in ids.iter().enumerate() {
            let row = &bytes[id as usize * rb..(id as usize + 1) * rb];
            dequant_row(info.ty, row, &mut want[t * k..(t + 1) * k])?;
        }
        let host_ok = bits_equal(&got, &want) && bits_equal(&got, &run(&ids)?);
        println!(
            "embed Q4_K K={k} rows={n_rows}: {} ids vs dequant_row bit_identical (and rerun)={host_ok} {}",
            ids.len(),
            verdict(host_ok)
        );
        let mut ok = host_ok;
        for (label, man) in sets {
            let erow = man.tensor("inp_embd", 0)?;
            let toks = ref_ints(man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
                .iter()
                .map(|&i| u32::try_from(i).map_err(|_| format!("{label}: token id {i}")))
                .collect::<Result<Vec<u32>, _>>()?;
            let ik = ref_tensor_logical_in(&man.dir, erow)?;
            let ours = run(&toks)?;
            let same = bits_equal(&ours, &ik);
            println!(
                "embed {label}: {} tokens vs ik inp_embd bit_identical={same} {}",
                toks.len(),
                verdict(same)
            );
            ok &= same;
        }
        // The graph: one node, replay = eager.
        let idb = DeviceBuffer::from_host(stream, &ids)?;
        let mut yg = DeviceBuffer::from_host(stream, &vec![SENT; ids.len() * k])?;
        let graph = gpu.capture(|s| gpu.elem().enqueue_embed_rows_q4k(s, &table, &idb, &mut yg))?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let g_same = bits_equal(&yg.to_host_vec(stream)?, &got);
        let nodes = graph.node_count();
        let pass = g_same && nodes == 1;
        println!(
            "graph op=embed_rows_q4k eager_vs_graph_bit_identical={g_same} graph_nodes={nodes} {}",
            verdict(pass)
        );
        // An id past the table: the named fault; the other rows unchanged.
        let clean = gpu.take_fault()?;
        let bad_ids = [ids[3], n_rows as u32 + 3, ids[4]];
        let y = run(&bad_ids)?;
        let fault = gpu.take_fault()?;
        let want_fault = Fault::at(LAYER_NONE, FaultSite::TokenId);
        let others = bits_equal(&y[..k], &got[3 * k..4 * k])
            && bits_equal(&y[2 * k..3 * k], &got[4 * k..5 * k]);
        let nan = y[k..2 * k].iter().filter(|v| v.is_nan()).count();
        let oor_ok = clean.is_none() && fault == Some(want_fault) && others && nan == k;
        println!(
            "embed Q4_K id past the table ids={bad_ids:?}: word_before={clean:?} fault=\"{}\" \
             (want \"{want_fault}\") other_rows_bit_identical={others} bad_row_nan={nan}/{k} \
             (want all) {}",
            fault.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            verdict(oor_ok)
        );
        Ok(ok && pass && oor_ok)
    }

    // ---------------------------------------------------- 2. gate·up body

    fn gate_up_body(gpu: &Gpu, gguf: &gguf::Gguf, ex: &ExpertKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (gi, gb) = tensor_bytes_as(gguf, &names::ffn_gate_exps(0), GgmlType::Q4_K, None)?;
        let (_, ub) = tensor_bytes_as(gguf, &names::ffn_up_exps(0), GgmlType::Q4_K, None)?;
        let [k, rpe, n_exp] = gi.dims[..] else {
            return Err(format!("gate stack is {:?}", gi.dims).into());
        };
        let (k, rpe, n_exp) = (k as usize, rpe as usize, n_exp as usize);
        let rb = gb.len() / (rpe * n_exp);
        let wg = upload(gpu, gb, rpe * n_exp)?;
        let wu = upload(gpu, ub, rpe * n_exp)?;
        let sel_h: [u32; 8] = [5, 127, 0, 5, 64, 3, 99, 1];
        let sel = DeviceBuffer::from_host(stream, &sel_h)?;
        let x = bloomery_gpu_gates::activations(k, 1, 4099);
        let act = quantize(gpu, &x, 1, k)?;
        let launch = |sel: &DeviceBuffer<u32>, fault: FaultSink| -> Result<Vec<f32>, GateError> {
            let mut h = DeviceBuffer::from_host(stream, &vec![SENT; 8 * rpe])?;
            ex.enqueue_gate_up(
                stream,
                GateUpArgs {
                    wg: &wg,
                    wu: &wu,
                    act: &act,
                    sel,
                    n_slots: 8,
                    rows_per_expert: rpe,
                    fault,
                    h: &mut h,
                },
            )?;
            stream.synchronize()?;
            Ok(h.to_host_vec(stream)?)
        };
        let unl = gpu.unlabelled_sink();
        let h = launch(&sel, unl)?;
        let mut ok = bits_equal(&h, &launch(&sel, unl)?);
        for (s, &id) in sel_h.iter().enumerate() {
            let span = id as usize * rpe * rb..(id as usize + 1) * rpe * rb;
            let g1 = DeviceTensor::upload(stream, &bytes_to_words(&gb[span.clone()]), rpe, rb / 4)?;
            let u1 = DeviceTensor::upload(stream, &bytes_to_words(&ub[span]), rpe, rb / 4)?;
            let mut yg = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
            let mut yu = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
            let mut hs = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
            gpu.enqueue_gemv_q4k(&g1, &act, &mut yg)?;
            gpu.enqueue_gemv_q4k(&u1, &act, &mut yu)?;
            gpu.elem().enqueue_swiglu(stream, &yg, &yu, rpe, &mut hs)?;
            stream.synchronize()?;
            let same = bits_equal(&h[s * rpe..(s + 1) * rpe], &hs.to_host_vec(stream)?);
            ok &= same;
            println!("body slot={s} id={id} bit_identical_to_q4k_gemv+swiglu={same}");
        }
        println!(
            "body ffn_gate/up_exps.0 K={k} rows={rpe} experts={n_exp}: rerun and slots vs q4k_gemv+swiglu {}",
            verdict(ok)
        );
        // An id past the stack in one slot per launch, against the same
        // launch with a valid id there.
        let layer = 13usize;
        let sink = gpu.layer_sink(layer)?;
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::ExpertId));
        let clean_ids = [2u32, 9, 7, 11, 0, 0, 0, 0];
        let clean_sel = DeviceBuffer::from_host(stream, &clean_ids)?;
        let slot = |v: &[f32], s: usize| v[s * rpe..(s + 1) * rpe].to_vec();
        for (bad, id) in [(1usize, n_exp as u32), (3, u32::MAX)] {
            let before = gpu.fault()?;
            let hc = launch(&clean_sel, sink)?;
            let after_clean = gpu.fault()?;
            let mut ids = clean_ids;
            ids[bad] = id;
            let ho = launch(&DeviceBuffer::from_host(stream, &ids)?, sink)?;
            let word = gpu.take_fault()?;
            let bad_nan = slot(&ho, bad).iter().all(|v| v.is_nan());
            let others = (0..8)
                .filter(|&s| s != bad)
                .all(|s| bits_equal(&slot(&ho, s), &slot(&hc, s)));
            let clean_again = bits_equal(&launch(&clean_sel, sink)?, &hc) && gpu.fault()?.is_none();
            let oor_ok = before.is_none()
                && after_clean.is_none()
                && word == want
                && bad_nan
                && others
                && clean_again;
            ok &= oor_ok;
            println!(
                "body id {id} past the stack of {n_exp} in slot {bad}: word \"{}\" (want \"{}\", clean \
                 before {} and after the clean run {}) that slot NaN {bad_nan}, other slots bit-identical \
                 {others}, clean rerun bits and word clean {clean_again} {}",
                word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
                want.map_or_else(String::new, |f| f.to_string()),
                before.is_none(),
                after_clean.is_none(),
                verdict(oor_ok)
            );
        }
        let mut hg = DeviceBuffer::from_host(stream, &vec![SENT; 8 * rpe])?;
        let graph = gpu.capture(|s| {
            ex.enqueue_gate_up(
                s,
                GateUpArgs {
                    wg: &wg,
                    wu: &wu,
                    act: &act,
                    sel: &sel,
                    n_slots: 8,
                    rows_per_expert: rpe,
                    fault: unl,
                    h: &mut hg,
                },
            )
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let g_same = bits_equal(&hg.to_host_vec(stream)?, &h);
        let nodes = graph.node_count();
        let pass = g_same && nodes == 1;
        println!(
            "graph op=gate_up_swiglu_q4k eager_vs_graph_bit_identical={g_same} graph_nodes={nodes} {}",
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ----------------------------------------------------- 3. gate·up tap

    fn gate_up_tap(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        ex: &ExpertKernels,
        sets: &[(&'static str, bloomery_gpu_gates::RefManifest)],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mut ok = true;
        let (mut n_tok, mut worst_o, mut worst_i, mut worst_pred, mut worst_ik) =
            (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f32);
        let mut worst_i_at = String::new();
        let mut parts_bad = 0usize;
        let mut layer = 0usize;
        let (mut rg, mut ru) = (Vec::new(), Vec::new());
        let (mut gq, mut gm, mut uq, mut um) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        loop {
            let gname = names::ffn_gate_exps(layer);
            if gguf.find(&gname).is_none() {
                break;
            }
            let (gi, gb) = tensor_bytes_as(gguf, &gname, GgmlType::Q4_K, None)?;
            let (_, ub) = tensor_bytes_as(gguf, &names::ffn_up_exps(layer), GgmlType::Q4_K, None)?;
            let (k, rpe, n_exp) = (
                gi.dims[0] as usize,
                gi.dims[1] as usize,
                gi.dims[2] as usize,
            );
            let rb = gb.len() / (rpe * n_exp);
            let wg = upload(gpu, gb, rpe * n_exp)?;
            let wu = upload(gpu, ub, rpe * n_exp)?;
            rg.resize(k, 0.0f32);
            ru.resize(k, 0.0f32);
            let mut layer_ok = true;
            for (label, man) in sets {
                let l = layer;
                let xrow = man.tensor(&format!("ffn_inp_normed-{l}"), 0)?;
                let hrow = man.tensor(&format!("ffn_moe_gate_par-{l}"), 0)?;
                let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
                let xs = ref_tensor_logical_in(&man.dir, xrow)?;
                let hs = ref_tensor_logical_in(&man.dir, hrow)?;
                let ids = topk_ids_logical_within(man, trow, n_exp as u32)?;
                let (slots, m) = (hrow.ne[1] as usize, hrow.ne[2] as usize);
                if hrow.ne[0] as usize != rpe || ids.len() != slots * m || xs.len() < m * k {
                    return Err(format!(
                        "{label} layer {l}: ffn_inp_normed {:?}, gate_par {:?}, {} ids",
                        xrow.ne,
                        hrow.ne,
                        ids.len()
                    )
                    .into());
                }
                // The FFN runs only the tokens its output keeps: the last
                // layer of a prefill runs the last token alone.
                let x0 = xs.len() / k - m;
                for t in 0..m {
                    let x = &xs[(x0 + t) * k..(x0 + t + 1) * k];
                    let ik = &hs[t * slots * rpe..(t + 1) * slots * rpe];
                    let sel_h: Vec<u32> = ids[t * slots..(t + 1) * slots]
                        .iter()
                        .map(|&i| i as u32)
                        .collect();
                    let sel = DeviceBuffer::from_host(stream, &sel_h)?;
                    let act = quantize(gpu, x, 1, k)?;
                    let mut h = DeviceBuffer::from_host(stream, &vec![SENT; slots * rpe])?;
                    ex.enqueue_gate_up(
                        stream,
                        GateUpArgs {
                            wg: &wg,
                            wu: &wu,
                            act: &act,
                            sel: &sel,
                            n_slots: slots,
                            rows_per_expert: rpe,
                            fault: gpu.unlabelled_sink(),
                            h: &mut h,
                        },
                    )?;
                    stream.synchronize()?;
                    let ours = h.to_host_vec(stream)?;
                    let xo = q8_1_dequant(x, k, 1);
                    let xi = ik_q8_2::reconstruct(x);
                    for s in 0..slots {
                        let e = sel_h[s] as usize;
                        for r in 0..rpe {
                            let span = (e * rpe + r) * rb..(e * rpe + r + 1) * rb;
                            dequant_row(GgmlType::Q4_K, &gb[span.clone()], &mut rg)?;
                            dequant_row(GgmlType::Q4_K, &ub[span.clone()], &mut ru)?;
                            q4k_parts(&gb[span.clone()], &mut gq, &mut gm);
                            q4k_parts(&ub[span], &mut uq, &mut um);
                            let mut acc = [0.0f64; 8];
                            for i in 0..k {
                                let (g, u) = (f64::from(rg[i]), f64::from(ru[i]));
                                let (a, b) = (f64::from(xo[i]), f64::from(xi[i]));
                                if (gq[i] - gm[i]).to_bits() != rg[i].to_bits()
                                    || (uq[i] - um[i]).to_bits() != ru[i].to_bits()
                                {
                                    parts_bad += 1;
                                }
                                let mag_g = f64::from(gq[i]).abs() + f64::from(gm[i]).abs();
                                let mag_u = f64::from(uq[i]).abs() + f64::from(um[i]).abs();
                                acc[0] += g * a;
                                acc[1] += u * a;
                                acc[2] += mag_g * a.abs();
                                acc[3] += mag_u * a.abs();
                                acc[4] += g * b;
                                acc[5] += u * b;
                                acc[6] += mag_g * b.abs();
                                acc[7] += mag_u * b.abs();
                            }
                            let tol = |g: f64, u: f64, ag: f64, au: f64| {
                                let h = silu64(g) * u;
                                1.1 * u.abs() * gamma(F32_TERMS) * ag
                                    + silu64(g).abs() * gamma(F32_TERMS) * au
                                    + SILU_ULPS * U * h.abs()
                            };
                            let (ho, hi) = (silu64(acc[0]) * acc[1], silu64(acc[4]) * acc[5]);
                            let (to, ti) = (
                                tol(acc[0], acc[1], acc[2], acc[3]),
                                tol(acc[4], acc[5], acc[6], acc[7]),
                            );
                            let o = f64::from(ours[s * rpe + r]);
                            let iv = f64::from(ik[s * rpe + r]);
                            let (d_o, d_i) = ((o - ho).abs(), (iv - hi).abs());
                            let gap = ((o - iv) - (ho - hi)).abs();
                            let tiny = f64::MIN_POSITIVE;
                            worst_o = worst_o.max(d_o / to.max(tiny));
                            if d_i / ti.max(tiny) > worst_i {
                                worst_i = d_i / ti.max(tiny);
                                worst_i_at = format!(
                                    "layer {l} {label} t {t} slot {s} row {r}: g {:.6e} u {:.6e} h_ik_sim {hi:.6e} \
                                     ik {iv:.6e} ours {o:.6e} h_ours_sim {ho:.6e} tol_i {ti:.3e} A_g {:.3e} A_u {:.3e}",
                                    acc[4], acc[5], acc[6], acc[7]
                                );
                            }
                            worst_pred = worst_pred.max(gap / (to + ti).max(tiny));
                            if d_o > to || d_i > ti || gap > to + ti {
                                layer_ok = false;
                            }
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
                "gate_up layer={layer} K={k} rows={rpe} experts={n_exp} {}",
                verdict(layer_ok)
            );
            ok &= layer_ok;
            layer += 1;
        }
        println!(
            "gate_up tap: {layer} layers, {n_tok} (layer, token) sites — ours vs its f64 rule at {worst_o:.3} \
             of its bound; ik sim vs dump at {worst_i:.3} of its bound; ours - ik vs the predicted \
             quantization difference at {worst_pred:.3} of both bounds; plain ours vs ik rel={worst_ik:.3e} \
             (printed) {}",
            verdict(ok)
        );
        println!(
            "gate_up worst ik-sim site: {worst_i_at}; Q4_K split d1·q − m1 ≠ dequant_row at \
             {parts_bad} values (want 0)"
        );
        Ok(ok && parts_bad == 0)
    }

    // --------------------------------------------------------- 4. combine

    fn combine(
        gpu: &Gpu,
        ex: &ExpertKernels,
        sets: &[(&'static str, bloomery_gpu_gates::RefManifest)],
        stream: &cuda_core::CudaStream,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let (mut n_tok, mut n_same, mut n_val, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
        let mut graph_checked = false;
        for (label, man) in sets {
            let mut l = 0usize;
            while man.tensor(&format!("ffn_moe_down-{l}"), 0).is_ok() {
                let drow = man.tensor(&format!("ffn_moe_down-{l}"), 0)?;
                let wrow = man.tensor(&format!("ffn_moe_weights_norm-{l}"), 0)?;
                let orow = man.tensor(&format!("routed_out-{l}"), 0)?;
                let down = ref_tensor_logical_in(&man.dir, drow)?;
                let wts = ref_tensor_logical_in(&man.dir, wrow)?;
                let outs = ref_tensor_logical_in(&man.dir, orow)?;
                let (rows, slots, m) = (
                    drow.ne[0] as usize,
                    drow.ne[1] as usize,
                    drow.ne[2] as usize,
                );
                if wts.len() != slots * m || outs.len() != rows * m {
                    return Err(format!(
                        "{label} layer {l}: down {:?}, weights {:?}, routed_out {:?}",
                        drow.ne, wrow.ne, orow.ne
                    )
                    .into());
                }
                let zero = DeviceBuffer::<f32>::zeroed(stream, rows)?;
                for t in 0..m {
                    let d = &down[t * slots * rows..(t + 1) * slots * rows];
                    let w = &wts[t * slots..(t + 1) * slots];
                    let ik = &outs[t * rows..(t + 1) * rows];
                    let dd = DeviceBuffer::from_host(stream, d)?;
                    let wd = DeviceBuffer::from_host(stream, w)?;
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows])?;
                    ex.enqueue_combine(stream, &dd, &wd, &zero, slots, &mut y)?;
                    stream.synchronize()?;
                    let ours = y.to_host_vec(stream)?;
                    for r in 0..rows {
                        let abs: f64 = (0..slots)
                            .map(|s| (f64::from(w[s]) * f64::from(d[s * rows + r])).abs())
                            .sum();
                        let diff = (f64::from(ours[r]) - f64::from(ik[r])).abs();
                        let tol = gamma(2 * slots) * abs;
                        worst = worst.max(diff / tol.max(f64::MIN_POSITIVE));
                        if diff > tol {
                            ok = false;
                        }
                        n_same += usize::from(ours[r].to_bits() == ik[r].to_bits());
                        n_val += 1;
                    }
                    if !graph_checked {
                        let mut yg = DeviceBuffer::from_host(stream, &vec![SENT; rows])?;
                        let graph = gpu
                            .capture(|s| ex.enqueue_combine(s, &dd, &wd, &zero, slots, &mut yg))?;
                        graph.launch(stream)?;
                        stream.synchronize()?;
                        let g_same = bits_equal(&yg.to_host_vec(stream)?, &ours);
                        let nodes = graph.node_count();
                        let pass = g_same && nodes == 1;
                        println!(
                            "graph op=combine eager_vs_graph_bit_identical={g_same} graph_nodes={nodes} {}",
                            verdict(pass)
                        );
                        ok &= pass;
                        graph_checked = true;
                    }
                    n_tok += 1;
                }
                l += 1;
            }
        }
        println!(
            "combine vs ik routed_out: {n_tok} (layer, token) sites, {n_same}/{n_val} values bit-equal, \
             worst at {worst:.3} of γ(2·slots)·Σ|w·d| {}",
            verdict(ok)
        );
        Ok(ok)
    }
}
