//! Real-activation deviation probe (the real-x track): the q8_1 kernel
//! families driven on the activations the model actually produced — the ik
//! CUDA oracle dump — instead of the synthetic LCG, so the lead can pin the
//! kernel band from the real distribution.
//!
//! An instrument, not a gate: no band assertion anywhere. One line per
//! (site, layer), each number `max|a-b| / max|b|` over all rows and tokens:
//! - `raw_x_rel`   our kernel vs `ref_gemv` on the raw real activation —
//!                 the number a raw-activation band would gate;
//! - `ik_rel`      our kernel vs ik CUDA's own output for the same matmul;
//! - `ik_vs_exact` ik CUDA's output vs the same `ref_gemv` — ik's own
//!                 activation-quantization noise, the design's floor.
//! Per line the input's amax and amax/rms say how spiky the real input is;
//! one synthetic-`activations()` contrast line per family follows the real
//! table (no ik column exists there).
//!
//! Exits 1 only on a non-finite value, a missing/mis-sized dump file, or a
//! chain whose dims do not prove it. A site whose input or output is not in
//! the dump is skipped with a printed `skip` line, never guessed.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("real_x: built without the `gpu` feature; see `just probe-gpu-real-x`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::q5::Q5Kernels;
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, RefRow};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    activations, bytes_to_words, find_ref_row, max_rel_err, open_model, ref_dir, ref_gemv,
    ref_manifest, ref_tensor_of, row_bytes, tensor_bytes, tensor_bytes_as,
};
#[cfg(feature = "gpu")]
use cuda_core::{CudaStream, DeviceBuffer};
#[cfg(feature = "gpu")]
use gguf::Gguf;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;
#[cfg(feature = "gpu")]
use model::arch::deepseek2::names;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("real_x", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    // Synthetic-contrast seed, fixed: the contrast lines must be
    // reproducible like the gates' pinned draws.
    const SEED: u32 = 1;
    // output.weight rows driven (and given an exact reference): the head's
    // full 102400 rows add nothing the first 4096 do not show.
    const HEAD_ROWS: usize = 4096;

    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let q5 = Q5Kernels::load(gpu.context())?;
    let stream = gpu.stream();
    let man = ref_manifest()?;
    println!(
        "real_x: dump {} ({} manifest rows)",
        ref_dir().display(),
        man.len()
    );

    // Family summary collectors, in print order; [raw_x_rel, ik_rel,
    // ik_vs_exact] per real line.
    let mut fams: Vec<(GgmlType, Vec<[f32; 3]>)> = vec![
        (GgmlType::Q3_K, Vec::new()),
        (GgmlType::Q4_K, Vec::new()),
        (GgmlType::Q6_K, Vec::new()),
        (GgmlType::Q5_1, Vec::new()),
        (GgmlType::Q5_0, Vec::new()),
    ];
    let mut push = |ty: GgmlType, v: [f32; 3]| {
        if let Some(f) = fams.iter_mut().find(|(t, _)| *t == ty) {
            f.1.push(v);
        }
    };

    // ---- real sites. Chains (dims prove each: input ne0 = weight K,
    // output ne0 = weight rows, output op = the matmul that produced it):
    //   attn_norm-L [2048,t] -> blk.L.attn_q.weight        -> q-L [3072,t]
    //   attn_norm-L [2048,t] -> blk.L.attn_kv_a_mqa.weight -> kv_rope_compressed-L [576,t]
    //   kqv_2d-L    [2048,t] -> blk.L.attn_output.weight   -> kqv_out-L [2048,t]
    //   ffn_norm-L  [2048,t] -> blk.L.ffn_{gate,up}[_shexp].weight -> ffn_up_gate-L
    //   result_norm [2048,1] -> output.weight              -> result_output [102400,1]
    for l in [0usize, 1, 13, 26] {
        let in_norm = find_ref_row(&man, &format!("attn_norm-{l}"), 0)?;
        for (site, tensor, out, out_op) in [
            ("attn_q", names::attn_q(l), format!("q-{l}"), "MUL_MAT"),
            (
                "attn_kv_a_mqa",
                names::attn_kv_a_mqa(l),
                format!("kv_rope_compressed-{l}"),
                "MUL_MAT",
            ),
        ] {
            if let Some(v) = real_kq_site(
                &gpu,
                &gguf,
                &man,
                &format!("{site} L={l}"),
                &tensor,
                in_norm,
                &out,
                out_op,
                None,
            )? {
                push(GgmlType::Q3_K, v);
            }
        }

        // The o-proj input is the reshape VIEW kqv_2d-L of the flash output
        // kqv-L: same buffer, so their manifest sums must agree — this
        // blocks picking up a reordered or post-op tensor by accident.
        let kqv_2d = find_ref_row(&man, &format!("kqv_2d-{l}"), 0)?;
        let kqv = find_ref_row(&man, &format!("kqv-{l}"), 0)?;
        if (kqv_2d.sum - kqv.sum).abs() > 1e-3 * kqv.sum.abs().max(1.0) {
            return Err(format!(
                "real_x: kqv_2d-{l} sum {} != kqv-{l} sum {} — not the same buffer",
                kqv_2d.sum, kqv.sum
            )
            .into());
        }
        if let Some(v) = real_kq_site(
            &gpu,
            &gguf,
            &man,
            &format!("attn_output L={l}"),
            &names::attn_output(l),
            kqv_2d,
            &format!("kqv_out-{l}"),
            "MUL_MAT",
            None,
        )? {
            push(GgmlType::Q4_K, v);
        }
    }

    // ---- fused gate/up sites. The dump's FUSED_UP_GATE tensor is the
    // swiglu of the two projections ([I,t], not a raw half of [2I,t]); the
    // operand order is asserted from the manifest (src0 = up, src1 = gate)
    // and our chain is both gemvs over ONE quantization + host swiglu
    // (see `swiglu_site`).
    let dense_norm = find_ref_row(&man, "ffn_norm-0", 0)?;
    let dense_up_gate = find_ref_row(&man, "ffn_up_gate-0", 0)?;
    if let Some(v) = swiglu_site(
        &gpu,
        &gguf,
        "ffn_swiglu_dense L=0",
        dense_norm,
        dense_up_gate,
        &names::ffn_gate(0),
        &names::ffn_up(0),
    )? {
        push(GgmlType::Q3_K, v);
    }
    for l in [1usize, 13, 26] {
        let norm = find_ref_row(&man, &format!("ffn_norm-{l}"), 0)?;
        let up_gate = find_ref_row(&man, &format!("ffn_up_gate-{l}"), 0)?;
        if let Some(v) = swiglu_site(
            &gpu,
            &gguf,
            &format!("ffn_swiglu_shexp L={l}"),
            norm,
            up_gate,
            &names::ffn_gate_shexp(l),
            &names::ffn_up_shexp(l),
        )? {
            push(GgmlType::Q3_K, v);
        }
    }

    // ---- the head. result_norm carries the last token only (m=1).
    if let Some(v) = real_kq_site(
        &gpu,
        &gguf,
        &man,
        "head",
        "output.weight",
        find_ref_row(&man, "result_norm", 0)?,
        "result_output",
        "MUL_MAT",
        Some(HEAD_ROWS),
    )? {
        push(GgmlType::Q6_K, v);
    }

    // ---- sites the dump cannot serve: skipped, printed, never guessed.
    println!(
        "skip site=ffn_down_shexp T=Q4_K K=2816 reason=\"input (swiglu of shexp gate/up) absent; only the down output ffn_shexp-L is dumped\""
    );
    println!(
        "skip site=ffn_down_dense0 T=Q5_1 K=10944 reason=\"input (swiglu of dense gate/up) absent; only the down output ffn_out-0 is dumped\""
    );
    println!(
        "skip site=ffn_exps T=Q5_0/Q3_K reason=\"routed-expert outputs are MUL_MAT_ID aggregates ([2048,tokens,topk]); an expert id cannot be attributed without ik's routing axis convention\""
    );

    // ---- synthetic contrast, one line per kernel family on `activations()`
    // (the generator the kernel gates print their raw-x deviation on), same
    // geometry as the family's real site where one exists.
    synth_kq_site(
        &gpu,
        &gguf,
        "attn_q_synth",
        &names::attn_q(1),
        GgmlType::Q3_K,
        None,
        6,
        SEED,
    )?;
    synth_kq_site(
        &gpu,
        &gguf,
        "attn_output_synth",
        &names::attn_output(1),
        GgmlType::Q4_K,
        None,
        6,
        SEED,
    )?;
    synth_kq_site(
        &gpu,
        &gguf,
        "head_synth",
        "output.weight",
        GgmlType::Q6_K,
        Some(HEAD_ROWS),
        1,
        SEED,
    )?;
    synth_q5_1_site(&q5, &gguf, stream, "ffn_down_dense0_synth", 6, SEED)?;
    synth_q5_0_site(&q5, &gguf, stream, "ffn_down_exps_e0_synth", 0, 6, SEED)?;

    // ---- summary: per family the max and median of each column over the
    // real lines.
    for (ty, vals) in &fams {
        if vals.is_empty() {
            println!("summary T={ty:?} n=0 real lines (synthetic only this round)");
            continue;
        }
        let col = |i: usize| {
            let mut c: Vec<f32> = vals.iter().map(|v| v[i]).collect();
            c.sort_by(|a, b| a.total_cmp(b));
            let med = if c.len() % 2 == 1 {
                c[c.len() / 2]
            } else {
                0.5 * (c[c.len() / 2 - 1] + c[c.len() / 2])
            };
            (c[c.len() - 1], med)
        };
        let (rmax, rmed) = col(0);
        let (imax, imed) = col(1);
        let (emax, emed) = col(2);
        println!(
            "summary T={ty:?} n={} raw_x_rel max={rmax:.3e} med={rmed:.3e} ik_rel max={imax:.3e} med={imed:.3e} ik_vs_exact max={emax:.3e} med={emed:.3e}",
            vals.len()
        );
    }
    println!("real_x: instrument done (no band asserted)");
    Ok(())
}

/// Reshape a dump tensor from ggml order ([ne0=rows, ne1=tokens]: value
/// (r, t) at t*rows + r) into the kernels' output order (r*m + t), so a
/// dump row can be compared element-for-element with a kernel output.
#[cfg(feature = "gpu")]
fn ik_to_ours(ik: &[f32], rows: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * m];
    for t in 0..m {
        for r in 0..rows {
            out[r * m + t] = ik[t * rows + r];
        }
    }
    out
}

/// (amax, amax/rms) of an activation block; a zero rms is a broken input.
#[cfg(feature = "gpu")]
fn spikiness(x: &[f32], site: &str) -> Result<(f32, f32), GateError> {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let sumsq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let rms = (sumsq / x.len() as f64).sqrt();
    if !(amax > 0.0 && rms > 0.0) {
        return Err(format!("real_x: {site}: activation is all zero (amax {amax})").into());
    }
    Ok((amax, (f64::from(amax) / rms) as f32))
}

/// Run one K-quant (Q3_K/Q4_K/Q6_K) site on real dump activations: load and
/// chain-check input and output rows, drive the device path exactly as the
/// gate binaries do (quantize + gemv on the engine stream), and print the
/// line. `row_cap` limits the rows driven (and compared, and given an exact
/// reference) — the head's line prints `rows=<cap>/<total>`. Returns the
/// three columns for the family summary, or None with a skip line printed.
#[cfg(feature = "gpu")]
#[allow(
    clippy::too_many_arguments,
    reason = "a gate-local measurement site threading the dump row, tensor and geometry it measures; folding them into a params struct is R8's axis"
)]
fn real_kq_site(
    gpu: &Gpu,
    gguf: &Gguf,
    man: &[RefRow],
    site: &str,
    tensor: &str,
    in_row: &RefRow,
    out_name: &str,
    out_op: &str,
    row_cap: Option<usize>,
) -> Result<Option<[f32; 3]>, GateError> {
    let (info, bytes) = tensor_bytes(gguf, tensor)?;
    let ty = info.ty;
    let k = info.dims[0] as usize;
    let rows_total: usize = info.dims[1..].iter().product::<u64>() as usize;
    let rows = row_cap.map_or(rows_total, |c| rows_total.min(c));
    let rb = row_bytes(ty, k)?;
    if bytes.len() < rb * rows {
        return Err(format!("real_x: {site}: {tensor} too small for {rows} rows").into());
    }

    // Input chain check: f32, ne0 = K, a plain 2-D token block, m within
    // the kernels' 1..=8 column range.
    if in_row.ty != "f32" || in_row.ne[0] != k as u64 || in_row.ne[2] != 1 || in_row.ne[3] != 1 {
        return Err(format!(
            "real_x: {site}: input {} is f32 {:?}, want f32 [{k}, t, 1, 1]",
            in_row.name, in_row.ne
        )
        .into());
    }
    let m = in_row.ne[1] as usize;
    if !(1..=8).contains(&m) {
        return Err(format!(
            "real_x: {site}: {} has {m} tokens; this probe drives m in 1..=8",
            in_row.name
        )
        .into());
    }
    let x = ref_tensor_of(in_row)?;

    // Output chain check: the op that produced it, ne0 = the weight's rows,
    // ne1 = the input's tokens.
    let out_row = find_ref_row(man, out_name, 0)?;
    if out_row.op != out_op || out_row.ne[0] != rows_total as u64 || out_row.ne[1] != m as u64 {
        return Err(format!(
            "real_x: {site}: output {} is op {} {:?}, want {out_op} [{rows_total}, {m}, 1, 1]",
            out_row.name, out_row.op, out_row.ne
        )
        .into());
    }
    let yk = ref_tensor_of(out_row)?;
    let ik = ik_to_ours(&yk, rows_total, m);

    let y = kq_device_run(gpu, ty, &bytes[..rb * rows], k, rows, &x, m)?;
    let y_exact = ref_gemv(ty, &bytes[..rb * rows], k, rows, &x, m)?;
    let raw_x_rel = max_rel_err(&y, &y_exact)?;
    let ik_rel = max_rel_err(&y, &ik[..rows * m])?;
    let ik_vs_exact = max_rel_err(&ik[..rows * m], &y_exact)?;
    let (amax, ratio) = spikiness(&x, site)?;
    let cap_note = match row_cap {
        Some(c) if c < rows_total => format!("{rows}/{rows_total}"),
        _ => rows.to_string(),
    };
    println!(
        "real site={site:<24} T={ty:?} K={k} rows={cap_note} m={m} w={tensor} raw_x_rel={raw_x_rel:.3e} ik_rel={ik_rel:.3e} ik_vs_exact={ik_vs_exact:.3e} amax={amax:.3e} amax/rms={ratio:.1}"
    );
    Ok(Some([raw_x_rel, ik_rel, ik_vs_exact]))
}

/// The quantize + gemv sequence of the gate binaries over resident buffers:
/// upload the weight words, quantize `x` to q8_1 on the device, run the
/// family's gemv, read `rows * m` outputs back. Errors on a non-finite
/// output before anything is compared.
#[cfg(feature = "gpu")]
fn kq_device_run(
    gpu: &Gpu,
    ty: GgmlType,
    w_bytes: &[u8],
    k: usize,
    rows: usize,
    x: &[f32],
    m: usize,
) -> Result<Vec<f32>, GateError> {
    let words = bytes_to_words(w_bytes);
    if words.len() % rows != 0 {
        return Err(format!(
            "kq_device_run: {rows} rows do not divide {} words",
            words.len()
        )
        .into());
    }
    let stream = gpu.stream();
    let w = DeviceTensor::upload(stream, &words, rows, words.len() / rows)?;
    let x_dev = DeviceBuffer::from_host(stream, x)?;
    let mut act = Q8Act::with_k(stream, m, k)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    match ty {
        GgmlType::Q3_K => gpu.enqueue_gemv_q3k(&w, &act, &mut y_dev)?,
        GgmlType::Q4_K => gpu.enqueue_gemv_q4k(&w, &act, &mut y_dev)?,
        GgmlType::Q6_K => gpu.enqueue_gemv_q6k(&w, &act, &mut y_dev)?,
        other => return Err(format!("kq_device_run: no gemv for {other:?}").into()),
    }
    stream.synchronize()?;
    let y = y_dev.to_host_vec(stream)?;
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(format!("kq_device_run: non-finite output at index {i} ({ty:?})").into());
    }
    Ok(y)
}

/// A fused gate/up site. The dump's FUSED_UP_GATE tensor is NOT a raw half
/// of the [gate; up] concat — it is [I, t] where I is the FFN intermediate,
/// i.e. the swiglu of the two projections (the fused kernel applies it).
/// The operand order is the manifest's, asserted before any number is read:
/// `src0` names the up projection, `src1` the gate projection — the fused
/// `silu(gate)·up`. Our chain mirrors the engine's: ONE q8_1 quantization
/// of the norm output, both Q3_K gemvs against it, then the host swiglu
/// silu(gate)·up in f32.
#[cfg(feature = "gpu")]
fn swiglu_site(
    gpu: &Gpu,
    gguf: &Gguf,
    site: &str,
    in_row: &RefRow,
    up_gate_row: &RefRow,
    gate_tensor: &str,
    up_tensor: &str,
) -> Result<Option<[f32; 3]>, GateError> {
    if up_gate_row.op != "FUSED_UP_GATE" {
        return Err(format!(
            "real_x: {site}: {} is op {}, want FUSED_UP_GATE",
            up_gate_row.name, up_gate_row.op
        )
        .into());
    }
    // src0 = up, src1 = gate — anything else is not the fused
    // silu(gate)·up this chain models.
    if up_gate_row.src0.as_deref() != Some(up_tensor)
        || up_gate_row.src1.as_deref() != Some(gate_tensor)
    {
        return Err(format!(
            "real_x: {site}: {} src0={:?} src1={:?}, want src0 {up_tensor} / src1 {gate_tensor}",
            up_gate_row.name, up_gate_row.src0, up_gate_row.src1
        )
        .into());
    }
    // The chain: `in_row` [K, t, 1, 1] through both weights [K, rows] to
    // `up_gate_row` [rows, t]. K and rows are the dump's; each weight must
    // be exactly [K, rows].
    if up_gate_row.ne[1] != in_row.ne[1] || in_row.ne[2] != 1 || in_row.ne[3] != 1 {
        return Err(format!(
            "real_x: {site}: chain dims {} {:?} vs {} {:?} do not prove the chain",
            in_row.name, in_row.ne, up_gate_row.name, up_gate_row.ne
        )
        .into());
    }
    let w_dims = [in_row.ne[0], up_gate_row.ne[0]];
    let (_, g_bytes) = tensor_bytes_as(gguf, gate_tensor, GgmlType::Q3_K, Some(&w_dims))?;
    let (_, u_bytes) = tensor_bytes_as(gguf, up_tensor, GgmlType::Q3_K, Some(&w_dims))?;
    let (k, rows) = (w_dims[0] as usize, w_dims[1] as usize);
    let m = in_row.ne[1] as usize;
    let x = ref_tensor_of(in_row)?;
    let yk = ref_tensor_of(up_gate_row)?;
    let ik = ik_to_ours(&yk, rows, m);
    let rb = row_bytes(GgmlType::Q3_K, k)?;

    // Our chain: one quantization, both gemvs, host swiglu.
    let (y_g, y_u) = q3k_two_gemv(
        gpu,
        &g_bytes[..rb * rows],
        &u_bytes[..rb * rows],
        k,
        rows,
        &x,
        m,
    )?;
    let s: Vec<f32> = y_g.iter().zip(&y_u).map(|(&g, &u)| silu(g) * u).collect();
    if let Some(i) = s.iter().position(|v| !v.is_finite()) {
        return Err(format!("real_x: {site}: non-finite swiglu output at index {i}").into());
    }
    let g_exact = ref_gemv(GgmlType::Q3_K, &g_bytes[..rb * rows], k, rows, &x, m)?;
    let u_exact = ref_gemv(GgmlType::Q3_K, &u_bytes[..rb * rows], k, rows, &x, m)?;
    let s_ref: Vec<f32> = g_exact
        .iter()
        .zip(&u_exact)
        .map(|(&g, &u)| silu(g) * u)
        .collect();
    let raw_x_rel = max_rel_err(&s, &s_ref)?;
    let ik_rel = max_rel_err(&s, &ik)?;
    let ik_vs_exact = max_rel_err(&ik, &s_ref)?;
    let (amax, ratio) = spikiness(&x, site)?;
    println!(
        "real site={site:<28} T=Q3_K K={k} rows={rows} m={m} w={gate_tensor}|{up_tensor} raw_x_rel={raw_x_rel:.3e} ik_rel={ik_rel:.3e} ik_vs_exact={ik_vs_exact:.3e} amax={amax:.3e} amax/rms={ratio:.1}"
    );
    Ok(Some([raw_x_rel, ik_rel, ik_vs_exact]))
}

/// Two Q3_K gemvs over ONE quantization of `x` (the gate·up sharing the
/// engine's step does): upload both weights, quantize once, run both gemvs
/// into separate outputs, read both back. Same finite check as
/// `kq_device_run`.
#[cfg(feature = "gpu")]
fn q3k_two_gemv(
    gpu: &Gpu,
    w1_bytes: &[u8],
    w2_bytes: &[u8],
    k: usize,
    rows: usize,
    x: &[f32],
    m: usize,
) -> Result<(Vec<f32>, Vec<f32>), GateError> {
    let words1 = bytes_to_words(w1_bytes);
    let words2 = bytes_to_words(w2_bytes);
    if words1.len() % rows != 0 || words2.len() % rows != 0 {
        return Err(format!(
            "q3k_two_gemv: {rows} rows do not divide {}/{} words",
            words1.len(),
            words2.len()
        )
        .into());
    }
    let stream = gpu.stream();
    let w1 = DeviceTensor::upload(stream, &words1, rows, words1.len() / rows)?;
    let w2 = DeviceTensor::upload(stream, &words2, rows, words2.len() / rows)?;
    let x_dev = DeviceBuffer::from_host(stream, x)?;
    let mut act = Q8Act::with_k(stream, m, k)?;
    let mut y1_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    let mut y2_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    gpu.enqueue_gemv_q3k(&w1, &act, &mut y1_dev)?;
    gpu.enqueue_gemv_q3k(&w2, &act, &mut y2_dev)?;
    stream.synchronize()?;
    let y1 = y1_dev.to_host_vec(stream)?;
    let y2 = y2_dev.to_host_vec(stream)?;
    for (n, y) in [("first", &y1), ("second", &y2)] {
        if let Some(i) = y.iter().position(|v| !v.is_finite()) {
            return Err(format!("q3k_two_gemv: non-finite {n} output at index {i}").into());
        }
    }
    Ok((y1, y2))
}

/// silu in f32, the elementwise form the reference kernels use: v/(1+e^-v).
#[cfg(feature = "gpu")]
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// One synthetic-contrast line for a K-quant family: the same weight and
/// device path as the real sites, `activations()` input, `raw_x_rel` only
/// (there is no ik answer for a synthetic input).
#[cfg(feature = "gpu")]
fn synth_kq_site(
    gpu: &Gpu,
    gguf: &Gguf,
    site: &str,
    tensor: &str,
    ty: GgmlType,
    row_cap: Option<usize>,
    m: usize,
    seed: u32,
) -> Result<(), GateError> {
    let (info, bytes) = tensor_bytes_as(gguf, tensor, ty, None)?;
    let k = info.dims[0] as usize;
    let rows_total: usize = info.dims[1..].iter().product::<u64>() as usize;
    let rows = row_cap.map_or(rows_total, |c| rows_total.min(c));
    let rb = row_bytes(ty, k)?;
    let x = activations(k, m, seed);
    let y = kq_device_run(gpu, ty, &bytes[..rb * rows], k, rows, &x, m)?;
    let y_exact = ref_gemv(ty, &bytes[..rb * rows], k, rows, &x, m)?;
    let raw_x_rel = max_rel_err(&y, &y_exact)?;
    let (amax, ratio) = spikiness(&x, site)?;
    let cap_note = match row_cap {
        Some(c) if c < rows_total => format!("{rows}/{rows_total}"),
        _ => rows.to_string(),
    };
    println!(
        "synth site={site:<24} T={ty:?} K={k} rows={cap_note} m={m} raw_x_rel={raw_x_rel:.3e} amax={amax:.3e} amax/rms={ratio:.1}"
    );
    Ok(())
}

/// Synthetic-contrast line for Q5_1 on the dense down projection
/// (`blk.0.ffn_down.weight`, K=10944) — the family's real input is not in
/// the dump, so this is the only Q5_1 line of the round.
#[cfg(feature = "gpu")]
fn synth_q5_1_site(
    q5: &Q5Kernels,
    gguf: &Gguf,
    stream: &CudaStream,
    site: &str,
    m: usize,
    seed: u32,
) -> Result<(), GateError> {
    use bloomery_gpu::q5::{Q8Blocks32, pack_q5_1};

    // dims [K, rows]
    let (_, bytes) = tensor_bytes_as(
        gguf,
        &names::ffn_down(0),
        GgmlType::Q5_1,
        Some(&[10944, 2048]),
    )?;
    let (k, rows) = (10944usize, 2048usize);
    let k_blocks = k / 32;
    let q_stride = 256 * k_blocks.div_ceil(32);
    let w_dev = DeviceTensor::upload(
        stream,
        &pack_q5_1(bytes, k, rows)?,
        rows,
        q_stride + 2 * k_blocks,
    )?;
    let x = activations(k, m, seed);
    let y_exact = ref_gemv(GgmlType::Q5_1, bytes, k, rows, &x, m)?;
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Blocks32::new(stream, k, m)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    q5.enqueue_quantize_q8(stream, &x_dev, &mut act)?;
    q5.enqueue_gemv_q5_1(stream, &w_dev, &act, 0, rows, 0, m, &mut y_dev, 0)?;
    stream.synchronize()?;
    let y = y_dev.to_host_vec(stream)?;
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(format!("real_x: {site}: non-finite output at index {i}").into());
    }
    let raw_x_rel = max_rel_err(&y, &y_exact)?;
    let (amax, ratio) = spikiness(&x, site)?;
    println!(
        "synth site={site:<24} T=Q5_1 K={k} rows={rows} m={m} raw_x_rel={raw_x_rel:.3e} amax={amax:.3e} amax/rms={ratio:.1}"
    );
    Ok(())
}

/// Synthetic-contrast line for Q5_0 on expert 0 of the routed stack
/// (`blk.1.ffn_down_exps.weight`, K=1408), reached by `row0` like the P2
/// gate — the family's real input is a MUL_MAT_ID aggregate this round.
#[cfg(feature = "gpu")]
fn synth_q5_0_site(
    q5: &Q5Kernels,
    gguf: &Gguf,
    stream: &CudaStream,
    site: &str,
    expert: usize,
    m: usize,
    seed: u32,
) -> Result<(), GateError> {
    use bloomery_gpu::q5::{Q8Blocks32, pack_q5_0};

    // dims [K, rows, experts]
    let (_, bytes) = tensor_bytes_as(
        gguf,
        &names::ffn_down_exps(1),
        GgmlType::Q5_0,
        Some(&[1408, 2048, 64]),
    )?;
    let (k, rows, n_exp) = (1408usize, 2048usize, 64usize);
    let rb = row_bytes(GgmlType::Q5_0, k)?;
    let k_blocks = k / 32;
    let q_stride = 256 * k_blocks.div_ceil(32);
    let w_dev = DeviceTensor::upload(
        stream,
        &pack_q5_0(bytes, k, n_exp * rows)?,
        n_exp * rows,
        q_stride + k_blocks,
    )?;
    let x = activations(k, m, seed);
    let expert_bytes = &bytes[expert * rows * rb..][..rows * rb];
    let y_exact = ref_gemv(GgmlType::Q5_0, expert_bytes, k, rows, &x, m)?;

    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Blocks32::new(stream, k, m)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    q5.enqueue_quantize_q8(stream, &x_dev, &mut act)?;
    q5.enqueue_gemv_q5_0(
        stream,
        &w_dev,
        &act,
        expert * rows,
        rows,
        0,
        m,
        &mut y_dev,
        0,
    )?;
    stream.synchronize()?;
    let y = y_dev.to_host_vec(stream)?;
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(format!("real_x: {site}: non-finite output at index {i}").into());
    }
    let raw_x_rel = max_rel_err(&y, &y_exact)?;
    let (amax, ratio) = spikiness(&x, site)?;
    println!(
        "synth site={site:<24} T=Q5_0 K={k} rows={rows} m={m} expert={expert} raw_x_rel={raw_x_rel:.3e} amax={amax:.3e} amax/rms={ratio:.1}"
    );
    Ok(())
}
