//! GPU kernel gate for package P6 (docs/gpu-design.md 작업 꾸러미): the MoE
//! router — softmax over 64, top-6, weights — and the expert offset table.
//!
//! Asserted (implementation correctness) against a host scalar reference
//! transcribed from the CPU engine's routing (`model::moe::route_inner`:
//! serial ascending f32 max fold, f32 `exp`, f64 sum accumulated ascending,
//! f32 divide by `sum as f32`; top-k by prob desc with ties to the smaller
//! id — its argsort comparator; weights are the chosen probs times the
//! file's `expert_weights_scale`): ids EXACT, probs and weights within
//! 1e-6 — tighter than the package's KERNEL_BAND because these feed the
//! token directly and the only device/host divergence is `exp`'s last ulps.
//! Plus a bit-identical rerun per shape, and eager vs captured-graph byte
//! identity for the router -> table chain (the decode step's shape).
//!
//! Printed, NOT asserted: distance to the ik CUDA oracle on real inputs —
//! `ffn_moe_logits-L` in; probs vs `ffn_moe_probs-L` (SOFT_MAX), weights vs
//! `ffn_moe_weights-L` (GET_ROWS). Ids are categorical and ARE asserted:
//! the distance is a mismatch count, not a `max_rel_err`, and all of it
//! must be zero. The topk rows are VIEWs of the argsort output whose plain
//! files are flat reads (token 0's ranking only), so ids come from the v2
//! logical twins (`ffn_moe_topk-L.0.logical.f32`, every token's top-6 ids
//! cast to f32) and every token's ids must equal the routing reference.
//! Synthetic inputs cover what the dump cannot supply (an exact tie at the
//! 6th/7th rank, all-equal logits, ±80 extreme logits, and m = 8) and pin
//! the tie rule and overflow behaviour against the same host reference.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p6: built without the `gpu` feature; see `just gate-gpu-p6`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::router::{N_EXPERT, N_USED, RouterKernels};
    use bloomery_gpu_gates::{
        find_ref_row, max_rel_err, open_model, ref_manifest, ref_tensor_of, route_ref,
        tensor_bytes, topk_ids_logical,
    };
    use cuda_core::DeviceBuffer;

    // Router weights band: device `exp` vs host libm differ by at most a few
    // ulp, and everything else in the chain is bit-mirrored, so a correct
    // kernel sits far below this. Ids are exact by construction.
    const BAND: f32 = 1e-6;

    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let router = RouterKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let man = ref_manifest()?;

    // The kernels' fixed 64/top-6 geometry, cross-checked against the file.
    let n_expert = gguf
        .arch_get_u64("expert_count")
        .ok_or("gate_p6: metadata key expert_count missing")? as usize;
    let n_used = gguf
        .arch_get_u64("expert_used_count")
        .ok_or("gate_p6: metadata key expert_used_count missing")? as usize;
    if n_expert != N_EXPERT || n_used != N_USED {
        return Err(format!(
            "gate_p6: model routes {n_used} of {n_expert} experts; the kernels are fixed 6 of 64"
        )
        .into());
    }
    let scale = gguf
        .architecture()
        .and_then(|a| gguf.value(&format!("{a}.expert_weights_scale")))
        .and_then(|v| v.as_f32())
        .unwrap_or(1.0);

    // Rows per expert of the three stacks the offset table addresses.
    let (gate_info, _) = tensor_bytes(&gguf, "blk.1.ffn_gate_exps.weight")?;
    let (down_info, _) = tensor_bytes(&gguf, "blk.1.ffn_down_exps.weight")?;
    if gate_info.dims.len() != 3
        || gate_info.dims[2] != n_expert as u64
        || down_info.dims.len() != 3
        || down_info.dims[2] != n_expert as u64
    {
        return Err(format!(
            "gate_p6: expert stacks are {:?} / {:?}, want [K, rows, {n_expert}]",
            gate_info.dims, down_info.dims
        )
        .into());
    }
    let (rows_gu, rows_dn) = (gate_info.dims[1] as usize, down_info.dims[1] as usize);

    // ---- real layers: logits from the dump, all tokens it holds.
    for l in [1usize, 13, 26] {
        let logits_row = find_ref_row(&man, &format!("ffn_moe_logits-{l}"), 0)?;
        if logits_row.ty != "f32"
            || logits_row.op != "MUL_MAT"
            || logits_row.ne[0] != N_EXPERT as u64
            || logits_row.ne[2] != 1
            || logits_row.ne[3] != 1
        {
            return Err(format!(
                "gate_p6: ffn_moe_logits-{l} is {} {} {:?}, want f32 MUL_MAT [64, t, 1, 1]",
                logits_row.ty, logits_row.op, logits_row.ne
            )
            .into());
        }
        let m = logits_row.ne[1] as usize;
        if !(1..=8).contains(&m) {
            return Err(format!(
                "gate_p6: ffn_moe_logits-{l} holds {m} tokens; the router drives m in 1..=8"
            )
            .into());
        }
        let x_ik = ref_tensor_of(logits_row)?; // ggml t-major: x[t*64 + e]
        let (probs_ref, ids_ref, w_ref) = route_ref(&x_ik, m, scale)?;

        // The router's oracle outputs, chain-checked against the logits row.
        let probs_row = find_ref_row(&man, &format!("ffn_moe_probs-{l}"), 0)?;
        let weights_row = find_ref_row(&man, &format!("ffn_moe_weights-{l}"), 0)?;
        let topk_row = find_ref_row(&man, &format!("ffn_moe_topk-{l}"), 0)?;
        if probs_row.ty != "f32"
            || probs_row.op != "SOFT_MAX"
            || probs_row.ne != [N_EXPERT as u64, m as u64, 1, 1]
        {
            return Err(format!(
                "gate_p6: ffn_moe_probs-{l} is {} {} {:?}, want f32 SOFT_MAX [64, {m}, 1, 1]",
                probs_row.ty, probs_row.op, probs_row.ne
            )
            .into());
        }
        if weights_row.ty != "f32"
            || weights_row.op != "GET_ROWS"
            || weights_row.ne != [1, N_USED as u64, m as u64, 1]
        {
            return Err(format!(
                "gate_p6: ffn_moe_weights-{l} is {} {} {:?}, want f32 GET_ROWS [1, 6, {m}, 1]",
                weights_row.ty, weights_row.op, weights_row.ne
            )
            .into());
        }
        if topk_row.ty != "i32"
            || topk_row.op != "VIEW"
            || topk_row.ne != [N_USED as u64, m as u64, 1, 1]
        {
            return Err(format!(
                "gate_p6: ffn_moe_topk-{l} is {} {} {:?}, want i32 VIEW [6, {m}, 1, 1]",
                topk_row.ty, topk_row.op, topk_row.ne
            )
            .into());
        }
        let ik_probs = ref_tensor_of(probs_row)?; // t-major [64, m]
        let ik_w = ref_tensor_of(weights_row)?; // t-major [6, m]
        // The topk ids are the row's LOGICAL twin: every token's top-6,
        // each id cast to f32 by the dumper (see `topk_ids_logical`). The
        // plain file is the flat read of the argsort parent — token 0's
        // ranking only — and is not the tensor.
        let ik_ids = topk_ids_logical(topk_row)?;
        // Per-token softmax sanity of the loaded probs (sum 1 within f32
        // rounding) — catches a layout mix-up before any comparison.
        for t in 0..m {
            let s: f64 = ik_probs[t * N_EXPERT..(t + 1) * N_EXPERT]
                .iter()
                .map(|&v| f64::from(v))
                .sum();
            if (s - 1.0).abs() > 1e-3 {
                return Err(
                    format!("gate_p6: ffn_moe_probs-{l} token {t} sums to {s}, want 1").into(),
                );
            }
        }
        // GET_ROWS proof, dump-internal: every token's weights are bit-exact
        // copies of that token's probs at that token's ids.
        for t in 0..m {
            for s in 0..N_USED {
                let e = ik_ids[t * N_USED + s] as usize;
                let wv = ik_w[t * N_USED + s];
                if e >= N_EXPERT || wv.to_bits() != ik_probs[t * N_EXPERT + e].to_bits() {
                    return Err(format!(
                        "gate_p6: ffn_moe_weights-{l} ({t},{s}) is not probs[{e}]: {wv} vs {}",
                        if e < N_EXPERT {
                            ik_probs[t * N_EXPERT + e]
                        } else {
                            f32::NAN
                        }
                    )
                    .into());
                }
            }
        }

        // Device run on the same logits (transposed into the gemv layout).
        let x_dev_layout = to_expert_major(&x_ik, N_EXPERT, m);
        let out = run_router(&router, stream, &x_dev_layout, m, scale)?;
        let probs_t = transpose(&out.probs, N_EXPERT, m);
        let ids_t: Vec<i32> = transpose(&out.ids, N_USED, m)
            .iter()
            .map(|&v| v as i32)
            .collect();
        let w_t = transpose(&out.weights, N_USED, m);
        let ids_exact = ids_t == ids_ref;
        let probs_err = max_rel_err(&probs_t, &probs_ref)?;
        let w_err = max_rel_err(&w_t, &w_ref)?;
        let ik_probs_rel = max_rel_err(&probs_t, &ik_probs)?;
        let ik_w_rel = max_rel_err(&w_t, &ik_w)?;
        let ik_ids_exact = ik_ids == ids_ref;
        let ik_ids_mis = ids_t.iter().zip(&ik_ids).filter(|(a, b)| a != b).count();
        let pass =
            ids_exact && ik_ids_exact && probs_err <= BAND && w_err <= BAND && out.rerun_bit_same;
        println!(
            "shape op=router_topk src=ffn_moe_logits-{l} m={m} ids_exact={ids_exact} ik_ids_exact={ik_ids_exact} probs_err={probs_err:.3e} weights_err={w_err:.3e} bit_identical_rerun={} ik_probs_rel={ik_probs_rel:.3e} ik_weights_rel={ik_w_rel:.3e} ik_ids_mismatch={ik_ids_mis}/{} {}",
            out.rerun_bit_same,
            N_USED * m,
            verdict(pass)
        );
        if !ik_ids_exact {
            for t in 0..m {
                for s in 0..N_USED {
                    let i = t * N_USED + s;
                    if ik_ids[i] != ids_ref[i] {
                        println!(
                            "divergence op=router_topk layer={l} token={t} slot={s} dump_id={} route_ref_id={} device_id={}",
                            ik_ids[i], ids_ref[i], ids_t[i]
                        );
                    }
                }
            }
        }
        if !pass {
            ok = false;
        }

        // The decode (m=1) table over this layer's token-0 ids: slot-major
        // ids put token 0 in the first 6 elements.
        let t0: Vec<u32> = out.ids[..N_USED].to_vec();
        let table_ok = run_table(&router, stream, &t0, rows_gu, rows_dn)?;
        println!(
            "shape op=expert_table src=router_ids_L{l}_t0 rows_gu={rows_gu} rows_dn={rows_dn} exact={} bit_identical_rerun={} ik_rel=na(no oracle tensor) {}",
            table_ok.0,
            table_ok.1,
            verdict(table_ok.0 && table_ok.1)
        );
        if !(table_ok.0 && table_ok.1) {
            ok = false;
        }
    }
    println!(
        "dump ffn_moe_topk: manifest type i32 op VIEW; the plain file is the flat read of the argsort parent (token 0's ranking) — ids are the v2 logical twin's, every token's top-6, each id cast to f32, integral 0..63 checked"
    );

    // ---- synthetic: shapes and values the dump cannot supply.
    // An exact tie at the 6th/7th rank: experts 0..4 lead distinctly,
    // experts 5 and 6 share the 6th-largest logit — the tie rule must pick 5.
    let mut tie = vec![0.0f32; N_EXPERT];
    for (e, v) in tie.iter_mut().enumerate() {
        *v = if e < 5 {
            3.0 - 0.1 * e as f32
        } else if e == 5 || e == 6 {
            1.5
        } else {
            1.4 - 0.01 * (e - 7) as f32
        };
    }
    synth_case(&router, stream, "tie_6th_7th", &tie, scale, &mut ok)?;

    // All-equal logits: probs all 1/64, ids 0..=5 (every rank a tie).
    let flat = vec![1.25f32; N_EXPERT];
    synth_case(&router, stream, "all_equal", &flat, scale, &mut ok)?;

    // Extreme finite logits (+80 vs -80, rest moderate): exp underflows to
    // 0 for the far tail and nothing becomes non-finite.
    let mut ext = vec![0.0f32; N_EXPERT];
    for (e, v) in ext.iter_mut().enumerate() {
        *v = match e {
            0 => 80.0,
            1 => -80.0,
            _ => e as f32 * 0.25 - 7.75,
        };
    }
    synth_case(&router, stream, "extreme_pm80", &ext, scale, &mut ok)?;

    // m = 8 (the layout bound): distinct logits per token, one exact tie in
    // token 2, extremes in token 6.
    let mut xt = vec![0.0f32; N_EXPERT * 8];
    for t in 0..8usize {
        for e in 0..N_EXPERT {
            xt[t * N_EXPERT + e] = (((e * 7 + t * 13) % N_EXPERT) as f32) / 8.0 - 4.0;
        }
    }
    xt[2 * N_EXPERT + 10] = 0.5;
    xt[2 * N_EXPERT + 11] = 0.5;
    xt[6 * N_EXPERT + 3] = 80.0;
    xt[6 * N_EXPERT + 40] = -80.0;
    let x8 = to_expert_major(&xt, N_EXPERT, 8);
    let out = run_router(&router, stream, &x8, 8, scale)?;
    let probs_t = transpose(&out.probs, N_EXPERT, 8);
    let ids_t: Vec<i32> = transpose(&out.ids, N_USED, 8)
        .iter()
        .map(|&v| v as i32)
        .collect();
    let w_t = transpose(&out.weights, N_USED, 8);
    let (probs_ref, ids_ref, w_ref) = route_ref(&xt, 8, scale)?;
    let ids_exact = ids_t == ids_ref;
    let probs_err = max_rel_err(&probs_t, &probs_ref)?;
    let w_err = max_rel_err(&w_t, &w_ref)?;
    let pass = ids_exact && probs_err <= BAND && w_err <= BAND && out.rerun_bit_same;
    println!(
        "shape op=router_topk synthetic=m8_stride_with_tie m=8 ids_exact={ids_exact} probs_err={probs_err:.3e} weights_err={w_err:.3e} bit_identical_rerun={} {}",
        out.rerun_bit_same,
        verdict(pass)
    );
    if !pass {
        ok = false;
    }

    // ---- expert_table on synthetic ids including the stack edges 0 and 63.
    let edge_ids: Vec<u32> = vec![0, 63, 5, 32, 1, 62];
    let table_ok = run_table(&router, stream, &edge_ids, rows_gu, rows_dn)?;
    println!(
        "shape op=expert_table synthetic=edge_ids_0_63 rows_gu={rows_gu} rows_dn={rows_dn} exact={} bit_identical_rerun={} ik_rel=na(no oracle tensor) {}",
        table_ok.0,
        table_ok.1,
        verdict(table_ok.0 && table_ok.1)
    );
    if !(table_ok.0 && table_ok.1) {
        ok = false;
    }

    // ---- eager vs captured graph: the decode chain router -> table on
    // resident buffers (layer 13's real logits, all 6 tokens). The router's
    // ids buffer feeds the table with no host round trip — the step's shape.
    let logits_row = find_ref_row(&man, "ffn_moe_logits-13", 0)?;
    let m = logits_row.ne[1] as usize;
    let x_ik = ref_tensor_of(logits_row)?;
    let x_dev = DeviceBuffer::from_host(stream, &to_expert_major(&x_ik, N_EXPERT, m))?;
    let mut probs_dev = DeviceBuffer::<f32>::zeroed(stream, N_EXPERT * m)?;
    let mut ids_dev = DeviceBuffer::<u32>::zeroed(stream, N_USED * m)?;
    let mut w_dev = DeviceBuffer::<f32>::zeroed(stream, N_USED * m)?;
    let mut r0g = DeviceBuffer::<u32>::zeroed(stream, N_USED)?;
    let mut r0u = DeviceBuffer::<u32>::zeroed(stream, N_USED)?;
    let mut r0d = DeviceBuffer::<u32>::zeroed(stream, N_USED)?;
    let chain = |probs: &mut DeviceBuffer<f32>,
                 ids: &mut DeviceBuffer<u32>,
                 w: &mut DeviceBuffer<f32>,
                 r0g: &mut DeviceBuffer<u32>,
                 r0u: &mut DeviceBuffer<u32>,
                 r0d: &mut DeviceBuffer<u32>|
     -> Result<(), Box<dyn std::error::Error>> {
        router.enqueue_router_topk(stream, &x_dev, m, scale, probs, ids, w)?;
        router.enqueue_expert_table(stream, ids, rows_gu, rows_dn, r0g, r0u, r0d)?;
        Ok(())
    };
    chain(
        &mut probs_dev,
        &mut ids_dev,
        &mut w_dev,
        &mut r0g,
        &mut r0u,
        &mut r0d,
    )?;
    stream.synchronize()?;
    let eager = (
        probs_dev.to_host_vec(stream)?,
        ids_dev.to_host_vec(stream)?,
        w_dev.to_host_vec(stream)?,
        r0g.to_host_vec(stream)?,
        r0u.to_host_vec(stream)?,
        r0d.to_host_vec(stream)?,
    );
    probs_dev.zero_async(stream)?;
    ids_dev.zero_async(stream)?;
    w_dev.zero_async(stream)?;
    r0g.zero_async(stream)?;
    r0u.zero_async(stream)?;
    r0d.zero_async(stream)?;
    stream.synchronize()?;
    let graph = gpu.capture(|_s| {
        chain(
            &mut probs_dev,
            &mut ids_dev,
            &mut w_dev,
            &mut r0g,
            &mut r0u,
            &mut r0d,
        )
    })?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let replay = (
        probs_dev.to_host_vec(stream)?,
        ids_dev.to_host_vec(stream)?,
        w_dev.to_host_vec(stream)?,
        r0g.to_host_vec(stream)?,
        r0u.to_host_vec(stream)?,
        r0d.to_host_vec(stream)?,
    );
    let identical = eager.0 == replay.0
        && eager.1 == replay.1
        && eager.2 == replay.2
        && eager.3 == replay.3
        && eager.4 == replay.4
        && eager.5 == replay.5;
    let nodes = graph.node_count();
    let pass = identical && nodes == 2;
    println!(
        "graph op=router_topk+expert_table src=ffn_moe_logits-13 m={m} eager_vs_graph_bit_identical={identical} graph_nodes={nodes} {}",
        verdict(pass)
    );
    if !pass {
        ok = false;
    }

    if !ok {
        eprintln!("FAILED: gate_p6");
        std::process::exit(1);
    }
    println!(
        "PASSED: router ids exact / probs+weights within 1e-6 of the route_inner reference; expert table exact; eager == graph replay"
    );
    Ok(())
}

/// One synthetic m=1 shape, asserted against the same host reference.
#[cfg(feature = "gpu")]
fn synth_case(
    router: &bloomery_gpu::router::RouterKernels,
    stream: &cuda_core::CudaStream,
    name: &str,
    logits: &[f32],
    scale: f32,
    ok: &mut bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use bloomery_gpu_gates::{max_rel_err, route_ref};

    const BAND: f32 = 1e-6;
    let out = run_router(router, stream, logits, 1, scale)?;
    let (probs_ref, ids_ref, w_ref) = route_ref(logits, 1, scale)?;
    let ids_dev: Vec<i32> = out.ids.iter().map(|&v| v as i32).collect();
    let ids_exact = ids_dev == ids_ref;
    let probs_err = max_rel_err(&out.probs, &probs_ref)?;
    let w_err = max_rel_err(&out.weights, &w_ref)?;
    let pass = ids_exact && probs_err <= BAND && w_err <= BAND && out.rerun_bit_same;
    println!(
        "shape op=router_topk synthetic={name} m=1 ids_exact={ids_exact} probs_err={probs_err:.3e} weights_err={w_err:.3e} bit_identical_rerun={} {}",
        out.rerun_bit_same,
        verdict(pass)
    );
    if !pass {
        *ok = false;
    }
    Ok(())
}

/// The router on resident-free (fresh) buffers, run twice: the outputs of
/// the first run and whether the second run reproduced them bit for bit.
#[cfg(feature = "gpu")]
struct RouterOut {
    probs: Vec<f32>,
    ids: Vec<u32>,
    weights: Vec<f32>,
    rerun_bit_same: bool,
}

#[cfg(feature = "gpu")]
fn run_router(
    router: &bloomery_gpu::router::RouterKernels,
    stream: &cuda_core::CudaStream,
    x: &[f32],
    m: usize,
    scale: f32,
) -> Result<RouterOut, Box<dyn std::error::Error>> {
    use bloomery_gpu::router::{N_EXPERT, N_USED};
    use cuda_core::DeviceBuffer;

    let x_dev = DeviceBuffer::from_host(stream, x)?;
    let mut probs = DeviceBuffer::<f32>::zeroed(stream, N_EXPERT * m)?;
    let mut ids = DeviceBuffer::<u32>::zeroed(stream, N_USED * m)?;
    let mut w = DeviceBuffer::<f32>::zeroed(stream, N_USED * m)?;
    router.enqueue_router_topk(stream, &x_dev, m, scale, &mut probs, &mut ids, &mut w)?;
    stream.synchronize()?;
    let p1 = probs.to_host_vec(stream)?;
    let i1 = ids.to_host_vec(stream)?;
    let w1 = w.to_host_vec(stream)?;
    router.enqueue_router_topk(stream, &x_dev, m, scale, &mut probs, &mut ids, &mut w)?;
    stream.synchronize()?;
    let rerun = p1 == probs.to_host_vec(stream)?
        && i1 == ids.to_host_vec(stream)?
        && w1 == w.to_host_vec(stream)?;
    if let Some(i) = p1.iter().position(|v| !v.is_finite()) {
        return Err(format!("gate_p6: non-finite prob at {i}").into());
    }
    Ok(RouterOut {
        probs: p1,
        ids: i1,
        weights: w1,
        rerun_bit_same: rerun,
    })
}

/// The expert table on `ids` (6 u32): (exact vs the host computation,
/// bit-identical rerun).
#[cfg(feature = "gpu")]
fn run_table(
    router: &bloomery_gpu::router::RouterKernels,
    stream: &cuda_core::CudaStream,
    ids: &[u32],
    rows_gu: usize,
    rows_dn: usize,
) -> Result<(bool, bool), Box<dyn std::error::Error>> {
    use bloomery_gpu::router::N_USED;
    use cuda_core::DeviceBuffer;

    let ids_dev = DeviceBuffer::from_host(stream, ids)?;
    let mut r0g = DeviceBuffer::<u32>::zeroed(stream, N_USED)?;
    let mut r0u = DeviceBuffer::<u32>::zeroed(stream, N_USED)?;
    let mut r0d = DeviceBuffer::<u32>::zeroed(stream, N_USED)?;
    router.enqueue_expert_table(
        stream, &ids_dev, rows_gu, rows_dn, &mut r0g, &mut r0u, &mut r0d,
    )?;
    stream.synchronize()?;
    let (g1, u1, d1) = (
        r0g.to_host_vec(stream)?,
        r0u.to_host_vec(stream)?,
        r0d.to_host_vec(stream)?,
    );
    router.enqueue_expert_table(
        stream, &ids_dev, rows_gu, rows_dn, &mut r0g, &mut r0u, &mut r0d,
    )?;
    stream.synchronize()?;
    let rerun = g1 == r0g.to_host_vec(stream)?
        && u1 == r0u.to_host_vec(stream)?
        && d1 == r0d.to_host_vec(stream)?;
    let expect =
        |rows: usize| -> Vec<u32> { ids.iter().map(|&e| e * rows as u32).collect::<Vec<_>>() };
    let exact = g1 == expect(rows_gu) && u1 == expect(rows_gu) && d1 == expect(rows_dn);
    Ok((exact, rerun))
}

/// Transpose from the kernels' layouts to ggml's: `src` as `per` rows of
/// `m` (expert/slot-major, the kernels' buffers) becomes `m` rows of `per`
/// (token-major, the dump's tensors).
#[cfg(feature = "gpu")]
fn transpose<T: Copy>(src: &[T], per: usize, m: usize) -> Vec<T> {
    let mut out = vec![src[0]; per * m];
    for t in 0..m {
        for i in 0..per {
            out[t * per + i] = src[i * m + t];
        }
    }
    out
}

/// The inverse direction: `src` token-major (`m` rows of `per`, the dump's
/// tensors) becomes the kernels' expert-major layout (`per` rows of `m`).
#[cfg(feature = "gpu")]
fn to_expert_major<T: Copy>(src: &[T], per: usize, m: usize) -> Vec<T> {
    let mut out = vec![src[0]; per * m];
    for t in 0..m {
        for i in 0..per {
            out[i * m + t] = src[t * per + i];
        }
    }
    out
}

#[cfg(feature = "gpu")]
fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}
