//! GPU gate for the MoE fused kernels (docs/gpu-design.md P8 prep): the MoE
//! block's routed-expert half as FOUR fused launches vs the EIGHT-launch op
//! path, on layer 1's real tensors and the CUDA oracle dump's own
//! activations (input = dump `ffn_norm-1`'s LAST token column; the shared
//! expert and the residual arrive as dump inputs `ffn_shexp-1`/`ffn_inp-1`,
//! their pairing proven by element sums as the P4 add chains did). The
//! contract is BIT IDENTITY, not a band: both paths run the same cores
//! (`cores::q3k_row_dot` twice, `elem::silu_mul`,
//! `elem::weighted_expert_sum` plus two plain adds, the unchanged 32-value
//! quantizer and `q5_0_gemv_sel`), so any differing bit is a fusion defect.
//! Asserted: the h intermediates, the `Q8Blocks32` contents, the down
//! outputs, the final y, bit-identical reruns of both paths, eager vs
//! captured-graph replay for both paths, node counts 8 and 4, the router's
//! ids/weights against the host reference (`route_ref`), and an out-of-range
//! `sel` id leaving its slot untouched in both paths. Printed, never
//! asserted: `ik_rel` of y against dump `l_out-1`'s last column and of the
//! op-path weighted sum against `ffn_moe_out-1`'s. `--time` (lead-only,
//! under the machine lease) replays each captured graph 2000x and prints
//! us/replay plus 4- and 8-node empty-graph references; without the flag
//! nothing is timed or printed about time.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_moe_fused: built without the `gpu` feature; see `just gate-gpu-moe`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu_gates::oracle::deepseek2::L_OUT_1;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, bits_equal, load_ref, max_rel_err, open_model, ref_manifest, ref_model_path,
    route_ref, tensor_bytes, tensor_bytes_as, us_per_replay, verdict,
};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::Split;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_moe_fused", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    use bloomery_gpu::arch::deepseek2::Body;
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu::moe_fused::MoeFusedKernels;
    use bloomery_gpu::probe::Probe;
    use bloomery_gpu::q5::Q8Blocks32;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE, Q8Act};

    // Router band against `route_ref` on the same logits: the device `exp`
    // and the host libm differ by a few ulp, everything else in the chain is
    // bit-mirrored. Ids are exact by construction.
    const ROUTER_BAND: f32 = 1e-6;
    // Out-of-range probe sentinel: the bad slot's rows must still hold it.
    const SENT: f32 = 1.0e30;

    let time_mode = std::env::args().any(|a| a == "--time");
    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let moe = MoeFusedKernels::load(gpu.context(), gpu.fault_word())?;
    let stream = gpu.stream();
    let man = ref_manifest()?;

    // Shape from the file, never assumed: experts, routing width, the expert
    // intermediate width, the combine scale.
    let n_expert = gguf
        .arch_get_u64("expert_count")
        .ok_or("gate_moe_fused: metadata key expert_count missing")? as usize;
    let n_used = gguf
        .arch_get_u64("expert_used_count")
        .ok_or("gate_moe_fused: metadata key expert_used_count missing")? as usize;
    let ff = gguf
        .arch_get_u64("expert_feed_forward_length")
        .ok_or("gate_moe_fused: metadata key expert_feed_forward_length missing")?
        as usize;
    let scale = gguf.arch_get_f32("expert_weights_scale").unwrap_or(1.0);

    // K is the gate stack's row width and `rows` the down stack's row count,
    // both read from the file; every other extent is the metadata above.
    let k = tensor_bytes(&gguf, "blk.1.ffn_gate_exps.weight")?.0.dims[0];
    let (dn_info, _) = tensor_bytes(&gguf, "blk.1.ffn_down_exps.weight")?;
    let rows = *dn_info.dims.get(1).ok_or_else(|| {
        format!(
            "gate_moe_fused: blk.1.ffn_down_exps.weight is {:?}, want [expert_ff, rows, experts]",
            dn_info.dims
        )
    })?;
    // Dims in ggml order: gate and up [K, expert_ff, experts], down
    // [expert_ff, rows, experts], the router [K, experts].
    let (ff_d, n_d) = (ff as u64, n_expert as u64);
    let want: [(&str, GgmlType, &[u64]); 4] = [
        (
            "blk.1.ffn_gate_exps.weight",
            GgmlType::Q3_K,
            &[k, ff_d, n_d],
        ),
        ("blk.1.ffn_up_exps.weight", GgmlType::Q3_K, &[k, ff_d, n_d]),
        (
            "blk.1.ffn_down_exps.weight",
            GgmlType::Q5_0,
            &[ff_d, rows, n_d],
        ),
        ("blk.1.ffn_gate_inp.weight", GgmlType::F32, &[k, n_d]),
    ];
    for (name, ty, dims) in want {
        tensor_bytes_as(&gguf, name, ty, Some(dims))?;
    }
    let (k, rows) = (k as usize, rows as usize);
    if !(1..=8).contains(&n_used) {
        return Err(format!(
            "gate_moe_fused: n_used {n_used} outside 1..=8 (the activation scratch width)"
        )
        .into());
    }

    // ---- dump inputs: layer 1, the LAST token column of each.
    let (norm_row, ffn_norm) = load_ref(&man, "ffn_norm-1", 0)?;
    if norm_row.ty != "f32"
        || norm_row.ne[0] != k as u64
        || norm_row.ne[2] != 1
        || norm_row.ne[3] != 1
        || norm_row.ne[1] < 1
    {
        return Err(format!(
            "gate_moe_fused: ffn_norm-1 is {} {:?}, want f32 [K, t, 1, 1] with K = {k}",
            norm_row.ty, norm_row.ne
        )
        .into());
    }
    let tokens = norm_row.ne[1] as usize;
    let t0 = tokens - 1;
    let x = ffn_norm[t0 * k..tokens * k].to_vec();
    let (sh_row, sh_vals) = load_ref(&man, "ffn_shexp-1", 0)?;
    sh_row.expect("ffn_shexp-1", "f32", [k as u64, tokens as u64, 1, 1], "in")?;
    let shexp = sh_vals[t0 * k..tokens * k].to_vec();
    let (in_row, in_vals) = load_ref(&man, "ffn_inp-1", 0)?;
    in_row.expect("ffn_inp-1", "f32", [k as u64, tokens as u64, 1, 1], "in")?;
    let resid = in_vals[t0 * k..tokens * k].to_vec();
    let (fo_row, fo_vals) = load_ref(&man, "ffn_out-1", 0)?;
    fo_row.expect("ffn_out-1", "f32", [k as u64, tokens as u64, 1, 1], "ADD")?;
    let (lo_row, l_out) = load_ref(&man, L_OUT_1, 0)?;
    lo_row.expect(L_OUT_1, "f32", [k as u64, tokens as u64, 1, 1], "ADD")?;
    let (mm_row, moe_exp) = load_ref(&man, "ffn_moe_out-1", 0)?;
    mm_row.expect(
        "ffn_moe_out-1",
        "f32",
        [rows as u64, tokens as u64, 1, 1],
        "MUL_MULTI_ADD",
    )?;
    println!(
        "input op=ffn_norm-1 dims={:?} used=last_token_position_{t0} K={k} rows={rows} \
         n_expert={n_expert} n_used={n_used} expert_ff={ff} scale={scale}",
        norm_row.ne
    );

    // Operand pairing of the combine, by element sums (the P4 add chains):
    // ffn_out-1 = ffn_moe_out-1 + ffn_shexp-1, then l_out-1 = ffn_out-1 +
    // ffn_inp-1 — the grouping the fused kernel's parentheses must match.
    {
        let s = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
        let (sa, sb, so) = (
            s(&moe_exp[t0 * rows..tokens * rows]),
            s(&shexp),
            s(&fo_vals[t0 * k..tokens * k]),
        );
        if (sa + sb - so).abs() > 1e-3 * so.abs().max(1.0) {
            return Err(format!(
                "gate_moe_fused: sum(ffn_moe_out-1)+sum(ffn_shexp-1) = {} != sum(ffn_out-1) = \
                 {so} — dump does not pair these operands",
                sa + sb
            )
            .into());
        }
        let (sc, sd) = (s(&resid), s(&l_out[t0 * k..tokens * k]));
        if (so + sc - sd).abs() > 1e-3 * sd.abs().max(1.0) {
            return Err(format!(
                "gate_moe_fused: sum(ffn_out-1)+sum(ffn_inp-1) = {} != sum(l_out-1) = {sd} — \
                 dump does not pair these operands",
                so + sc
            )
            .into());
        }
        println!(
            "chain ffn_out-1=ffn_moe_out-1+ffn_shexp-1 l_out-1=ffn_out-1+ffn_inp-1 sums_pair=PASS"
        );
    }

    // ---- resident weights: layer 1 only, as a stage holds it — the tensors
    // this block consumes, plus the block's derived weights.
    let file = Split::open(ref_model_path()?)?;
    let mut wts = Weights::load(stream, &file, 1..2, false)?;
    Body::derive(stream, &file, 1..2, &mut wts)?;
    fn kq<'a>(wts: &'a Weights, name: &str) -> Result<&'a DeviceTensor<u32>, GateError> {
        let Some(DevWeight::KQuant { ty, w, .. }) = wts.get(name) else {
            return Err(format!("gate_moe_fused: {name} is not a KQuant resident weight").into());
        };
        if *ty != GgmlType::Q3_K {
            return Err(format!("gate_moe_fused: {name} is {ty}, want Q3_K").into());
        }
        Ok(w)
    }
    let wg = kq(&wts, "blk.1.ffn_gate_exps.weight")?;
    let wu = kq(&wts, "blk.1.ffn_up_exps.weight")?;
    let wd = match wts.get("blk.1.ffn_down_exps.weight") {
        Some(DevWeight::Q5_0 { w, .. }) => w,
        _ => {
            return Err(
                "gate_moe_fused: blk.1.ffn_down_exps.weight is not a Q5_0 resident weight".into(),
            );
        }
    };
    let wr = match wts.get("blk.1.ffn_gate_inp.weight") {
        Some(DevWeight::F32 { w, .. }) => w,
        _ => {
            return Err(
                "gate_moe_fused: blk.1.ffn_gate_inp.weight is not an F32 resident weight".into(),
            );
        }
    };

    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let shexp_dev = DeviceBuffer::from_host(stream, &shexp)?;
    let resid_dev = DeviceBuffer::from_host(stream, &resid)?;

    // ---- router on device: F32 gemv over the normed input, then top-6. Its
    // ids are the `sel` BOTH paths read; its weights feed both combines.
    let mut logits = DeviceBuffer::<f32>::zeroed(stream, n_expert)?;
    let mut probs = DeviceBuffer::<f32>::zeroed(stream, n_expert)?;
    let mut ids = DeviceBuffer::<u32>::zeroed(stream, n_used)?;
    let mut weights = DeviceBuffer::<f32>::zeroed(stream, n_used)?;
    gpu.q8f32()
        .enqueue_f32_gemv(stream, wr, &x_dev, 1, &mut logits)?;
    gpu.router().enqueue_router_topk(
        stream,
        &logits,
        1,
        scale,
        &mut probs,
        &mut ids,
        &mut weights,
        gpu.unlabelled_sink(),
    )?;
    stream.synchronize()?;
    let (logits_h, probs_h, ids_h, weights_h) = (
        logits.to_host_vec(stream)?,
        probs.to_host_vec(stream)?,
        ids.to_host_vec(stream)?,
        weights.to_host_vec(stream)?,
    );
    let (probs_ref, ids_ref, w_ref) = route_ref(&logits_h, 1, scale)?;
    let ids_exact = ids_h.iter().zip(&ids_ref).all(|(&a, &b)| a as i32 == b);
    let probs_err = max_rel_err(&probs_h, &probs_ref)?;
    let w_err = max_rel_err(&weights_h, &w_ref)?;
    let router_ok = ids_exact && probs_err <= ROUTER_BAND && w_err <= ROUTER_BAND;
    println!(
        "router op=router_topk src=ffn_norm-1[last]·ffn_gate_inp m=1 ids_exact={ids_exact} \
         probs_err={probs_err:.3e} weights_err={w_err:.3e} sel={ids_h:?} {}",
        verdict(router_ok)
    );
    if !router_ok {
        ok = false;
    }

    // ---- the shared input quantization both paths consume (outside the
    // fused/op split and outside both graphs — the eight and the four count
    // from here).
    let mut act = Q8Act::with_k(stream, 1, k)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    stream.synchronize()?;

    // Resident buffers, one set per path downstream of the shared act (the
    // graphs freeze addresses).
    let mut gate_y = DeviceBuffer::<f32>::zeroed(stream, n_used * ff)?;
    let mut up_y = DeviceBuffer::<f32>::zeroed(stream, n_used * ff)?;
    let mut h_op = DeviceBuffer::<f32>::zeroed(stream, n_used * ff)?;
    let mut h_fu = DeviceBuffer::<f32>::zeroed(stream, n_used * ff)?;
    let mut act32_op = Q8Blocks32::new(stream, ff, n_used)?;
    let mut act32_fu = Q8Blocks32::new(stream, ff, n_used)?;
    let mut down_op = DeviceBuffer::<f32>::zeroed(stream, n_used * rows)?;
    let mut down_fu = DeviceBuffer::<f32>::zeroed(stream, n_used * rows)?;
    let mut moe_y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
    let mut ffn_out = DeviceBuffer::<f32>::zeroed(stream, rows)?;
    let mut y_op = DeviceBuffer::<f32>::zeroed(stream, rows)?;
    let mut y_fu = DeviceBuffer::<f32>::zeroed(stream, rows)?;

    // The op path: eight enqueues. The adds are add(a, b) in the dump's own
    // operand order — ffn_out = moe + shexp, then y = ffn_out + resid.
    #[allow(
        clippy::too_many_arguments,
        reason = "a gate-local runner threading the buffer set its launches take; folding them into a params struct is R8's axis"
    )]
    fn run_op(
        gpu: &Gpu,
        stream: &cuda_core::CudaStream,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
        wd: &DeviceTensor<u32>,
        sel: &DeviceBuffer<u32>,
        weights: &DeviceBuffer<f32>,
        shexp_dev: &DeviceBuffer<f32>,
        resid_dev: &DeviceBuffer<f32>,
        act: &Q8Act,
        n_used: usize,
        ff: usize,
        rows: usize,
        gate_y: &mut DeviceBuffer<f32>,
        up_y: &mut DeviceBuffer<f32>,
        h_op: &mut DeviceBuffer<f32>,
        act32_op: &mut Q8Blocks32,
        down_op: &mut DeviceBuffer<f32>,
        moe_y: &mut DeviceBuffer<f32>,
        ffn_out: &mut DeviceBuffer<f32>,
        y_op: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        gpu.enqueue_gemv_q3k_sel(wg, act, sel, n_used, ff, gate_y)?;
        gpu.enqueue_gemv_q3k_sel(wu, act, sel, n_used, ff, up_y)?;
        gpu.elem()
            .enqueue_swiglu(stream, gate_y, up_y, n_used * ff, h_op)?;
        gpu.q5()
            .enqueue_quantize_q8(stream, h_op, act32_op, gpu.unlabelled_sink())?;
        gpu.q5()
            .enqueue_gemv_q5_0_sel(stream, wd, act32_op, sel, n_used, rows, down_op)?;
        gpu.elem()
            .enqueue_weighted_sum(stream, down_op, weights, rows, n_used as u32, 1, moe_y)?;
        gpu.elem()
            .enqueue_add(stream, moe_y, shexp_dev, rows, ffn_out)?;
        gpu.elem()
            .enqueue_add(stream, ffn_out, resid_dev, rows, y_op)
    }

    // The fused path: four enqueues.
    #[allow(
        clippy::too_many_arguments,
        reason = "a gate-local runner threading the buffer set its launches take; folding them into a params struct is R8's axis"
    )]
    fn run_fu(
        gpu: &Gpu,
        moe: &MoeFusedKernels,
        stream: &cuda_core::CudaStream,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
        wd: &DeviceTensor<u32>,
        sel: &DeviceBuffer<u32>,
        weights: &DeviceBuffer<f32>,
        shexp_dev: &DeviceBuffer<f32>,
        resid_dev: &DeviceBuffer<f32>,
        act: &Q8Act,
        n_used: usize,
        ff: usize,
        rows: usize,
        h_fu: &mut DeviceBuffer<f32>,
        act32_fu: &mut Q8Blocks32,
        down_fu: &mut DeviceBuffer<f32>,
        y_fu: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        moe.enqueue_expert_gate_up_swiglu(stream, wg, wu, act, sel, n_used, ff, h_fu)?;
        gpu.q5()
            .enqueue_quantize_q8(stream, h_fu, act32_fu, gpu.unlabelled_sink())?;
        gpu.q5()
            .enqueue_gemv_q5_0_sel(stream, wd, act32_fu, sel, n_used, rows, down_fu)?;
        moe.enqueue_moe_combine(
            stream, down_fu, weights, shexp_dev, resid_dev, rows, n_used, y_fu,
        )
    }

    // ---- eager runs with readbacks
    run_op(
        &gpu,
        stream,
        wg,
        wu,
        wd,
        &ids,
        &weights,
        &shexp_dev,
        &resid_dev,
        &act,
        n_used,
        ff,
        rows,
        &mut gate_y,
        &mut up_y,
        &mut h_op,
        &mut act32_op,
        &mut down_op,
        &mut moe_y,
        &mut ffn_out,
        &mut y_op,
    )?;
    stream.synchronize()?;
    let gate_y_1 = gate_y.to_host_vec(stream)?;
    let up_y_1 = up_y.to_host_vec(stream)?;
    let h_op_1 = h_op.to_host_vec(stream)?;
    let act32_op_h = act32_op.readback(stream)?;
    let down_op_1 = down_op.to_host_vec(stream)?;
    let moe_1 = moe_y.to_host_vec(stream)?;
    let y_op_1 = y_op.to_host_vec(stream)?;

    run_fu(
        &gpu,
        &moe,
        stream,
        wg,
        wu,
        wd,
        &ids,
        &weights,
        &shexp_dev,
        &resid_dev,
        &act,
        n_used,
        ff,
        rows,
        &mut h_fu,
        &mut act32_fu,
        &mut down_fu,
        &mut y_fu,
    )?;
    stream.synchronize()?;
    let h_fu_1 = h_fu.to_host_vec(stream)?;
    let act32_fu_h = act32_fu.readback(stream)?;
    let down_fu_1 = down_fu.to_host_vec(stream)?;
    let y_fu_1 = y_fu.to_host_vec(stream)?;

    // ---- the bit-identity table
    let h_same = bits_equal(&h_op_1, &h_fu_1);
    println!(
        "step1 op=gate_sel+up_sel+swiglu vs fused=expert_gate_up_swiglu_q3k h_bit_identical={h_same} n={}",
        n_used * ff
    );
    if !h_same {
        ok = false;
    }
    let act32_same = act32_op_h.q == act32_fu_h.q
        && act32_op_h.s8 == act32_fu_h.s8
        && act32_op_h
            .d8
            .iter()
            .zip(&act32_fu_h.d8)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "step2 op=quantize_q8(32) vs same q_bit_identical={} s8_bit_identical={} d8_bit_identical={} \
         (q {} u32, s8 {} i32, d8 {} f32)",
        act32_op_h.q == act32_fu_h.q,
        act32_op_h.s8 == act32_fu_h.s8,
        act32_op_h
            .d8
            .iter()
            .zip(&act32_fu_h.d8)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        act32_op_h.q.len(),
        act32_op_h.s8.len(),
        act32_op_h.d8.len()
    );
    if !act32_same {
        ok = false;
    }
    let down_same = bits_equal(&down_op_1, &down_fu_1);
    println!(
        "step3 op=down_q5_0_sel vs same down_bit_identical={down_same} n={}",
        n_used * rows
    );
    if !down_same {
        ok = false;
    }
    let y_same = bits_equal(&y_op_1, &y_fu_1);
    println!(
        "final op=weighted_sum+add(shexp)+add(resid) vs fused=moe_combine y_bit_identical={y_same} n={rows}"
    );
    if !y_same {
        ok = false;
    }

    // ---- bit-identical reruns of both paths
    run_op(
        &gpu,
        stream,
        wg,
        wu,
        wd,
        &ids,
        &weights,
        &shexp_dev,
        &resid_dev,
        &act,
        n_used,
        ff,
        rows,
        &mut gate_y,
        &mut up_y,
        &mut h_op,
        &mut act32_op,
        &mut down_op,
        &mut moe_y,
        &mut ffn_out,
        &mut y_op,
    )?;
    stream.synchronize()?;
    let y_op_2 = y_op.to_host_vec(stream)?;
    run_fu(
        &gpu,
        &moe,
        stream,
        wg,
        wu,
        wd,
        &ids,
        &weights,
        &shexp_dev,
        &resid_dev,
        &act,
        n_used,
        ff,
        rows,
        &mut h_fu,
        &mut act32_fu,
        &mut down_fu,
        &mut y_fu,
    )?;
    stream.synchronize()?;
    let y_fu_2 = y_fu.to_host_vec(stream)?;
    let rerun_same = bits_equal(&y_op_1, &y_op_2) && bits_equal(&y_fu_1, &y_fu_2);
    println!(
        "rerun op_bit_identical={} fused_bit_identical={}",
        bits_equal(&y_op_1, &y_op_2),
        bits_equal(&y_fu_1, &y_fu_2)
    );
    if !rerun_same {
        ok = false;
    }

    // ---- captured graphs: node counts and replay byte identity
    let graph_op = gpu.capture(|_| {
        run_op(
            &gpu,
            stream,
            wg,
            wu,
            wd,
            &ids,
            &weights,
            &shexp_dev,
            &resid_dev,
            &act,
            n_used,
            ff,
            rows,
            &mut gate_y,
            &mut up_y,
            &mut h_op,
            &mut act32_op,
            &mut down_op,
            &mut moe_y,
            &mut ffn_out,
            &mut y_op,
        )
    })?;
    let op_nodes = graph_op.node_count();
    graph_op.launch(stream)?;
    stream.synchronize()?;
    let op_replay_same = bits_equal(&y_op_1, &y_op.to_host_vec(stream)?)
        && bits_equal(&h_op_1, &h_op.to_host_vec(stream)?);

    let graph_fu = gpu.capture(|_| {
        run_fu(
            &gpu,
            &moe,
            stream,
            wg,
            wu,
            wd,
            &ids,
            &weights,
            &shexp_dev,
            &resid_dev,
            &act,
            n_used,
            ff,
            rows,
            &mut h_fu,
            &mut act32_fu,
            &mut down_fu,
            &mut y_fu,
        )
    })?;
    let fu_nodes = graph_fu.node_count();
    graph_fu.launch(stream)?;
    stream.synchronize()?;
    let fu_replay_same = bits_equal(&y_fu_1, &y_fu.to_host_vec(stream)?)
        && bits_equal(&h_fu_1, &h_fu.to_host_vec(stream)?);

    let nodes_ok = op_nodes == 8 && fu_nodes == 4;
    println!(
        "graph op_nodes={op_nodes} op_eager_vs_replay_bit_identical={op_replay_same} fused_nodes={fu_nodes} fused_eager_vs_replay_bit_identical={fu_replay_same} {}",
        verdict(nodes_ok && op_replay_same && fu_replay_same)
    );
    if !(nodes_ok && op_replay_same && fu_replay_same) {
        ok = false;
    }

    // ---- printed, never asserted: distance to the CUDA oracle's own
    // outputs on the same position.
    {
        let ik_y = max_rel_err(&y_fu_1, &l_out[t0 * k..tokens * k])?;
        let ik_moe = max_rel_err(&moe_1, &moe_exp[t0 * rows..tokens * rows])?;
        println!(
            "ik l_out-1[last] ik_rel={ik_y:.3e} ffn_moe_out-1[last] ik_rel={ik_moe:.3e} (printed, not asserted)"
        );
    }

    // ---- out-of-range sel id: slot n_used-1 carries n_expert (one past the
    // last expert); the op path's `q3k_gemv_sel` outputs and the fused kernel
    // must both leave that slot's rows at the sentinel and reproduce the
    // clean run's bits everywhere else, and each raises the named fault on
    // its own; the same slot at `HOST` (a host-served slot) raises nothing in
    // the fused kernel.
    {
        let mut sel_oor = ids_h.clone();
        sel_oor[n_used - 1] = n_expert as u32;
        let sel_oor_dev = DeviceBuffer::from_host(stream, &sel_oor)?;
        let span = n_used * ff;
        let mut g_oor = DeviceBuffer::from_host(stream, &vec![SENT; span])?;
        let mut u_oor = DeviceBuffer::from_host(stream, &vec![SENT; span])?;
        let mut h_oor = DeviceBuffer::from_host(stream, &vec![SENT; span])?;
        gpu.enqueue_gemv_q3k_sel(wg, &act, &sel_oor_dev, n_used, ff, &mut g_oor)?;
        gpu.enqueue_gemv_q3k_sel(wu, &act, &sel_oor_dev, n_used, ff, &mut u_oor)?;
        let fault_op = gpu.take_fault()?;
        moe.enqueue_expert_gate_up_swiglu(
            stream,
            wg,
            wu,
            &act,
            &sel_oor_dev,
            n_used,
            ff,
            &mut h_oor,
        )?;
        stream.synchronize()?;
        let fault = gpu.take_fault()?;
        let mut sel_host = ids_h.clone();
        sel_host[n_used - 1] = HOST;
        let sel_host_dev = DeviceBuffer::from_host(stream, &sel_host)?;
        let mut h_host = DeviceBuffer::from_host(stream, &vec![SENT; span])?;
        moe.enqueue_expert_gate_up_swiglu(
            stream,
            wg,
            wu,
            &act,
            &sel_host_dev,
            n_used,
            ff,
            &mut h_host,
        )?;
        let fault_host = gpu.take_fault()?;
        let want = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
        let fault_ok = fault_op == want && fault == want && fault_host.is_none();
        let (g_v, u_v, h_v) = (
            g_oor.to_host_vec(stream)?,
            u_oor.to_host_vec(stream)?,
            h_oor.to_host_vec(stream)?,
        );
        let untouched = |v: &[f32]| {
            v[(n_used - 1) * ff..]
                .iter()
                .all(|&x| x.to_bits() == SENT.to_bits())
        };
        let good = |v: &[f32], r: &[f32]| {
            (0..n_used - 1).all(|s| bits_equal(&v[s * ff..(s + 1) * ff], &r[s * ff..(s + 1) * ff]))
        };
        let op_untouched = untouched(&g_v) && untouched(&u_v);
        let fu_untouched = untouched(&h_v);
        let good_same = good(&g_v, &gate_y_1) && good(&u_v, &up_y_1) && good(&h_v, &h_op_1);
        let pass = op_untouched && fu_untouched && good_same && fault_ok;
        println!(
            "oor sel[{}]={n_expert} op_gemv_bad_slot_untouched={op_untouched} \
             fused_bad_slot_untouched={fu_untouched} good_slots_bit_identical={good_same} \
             op_fault={fault_op:?} fused_fault={fault:?} (want expert_id each) \
             fused_host_slot_fault={fault_host:?} (want none): {fault_ok} {}",
            n_used - 1,
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    // ---- lead-only timing under the machine lease; correctness runs never
    // reach this. Replays each captured graph N times, one synchronize at the
    // end, plus 4- and 8-node empty-graph references.
    if time_mode {
        const N: u32 = 2000;
        let probe = Probe::load(gpu.context())?;
        let mut tbuf = DeviceBuffer::<f32>::zeroed(stream, 32)?;
        let g4 =
            gpu.capture(|_| (0..4).try_for_each(|_| probe.enqueue_touch(stream, &mut tbuf)))?;
        let g8 =
            gpu.capture(|_| (0..8).try_for_each(|_| probe.enqueue_touch(stream, &mut tbuf)))?;
        println!(
            "time n={N} op_us_per_replay={:.3} fused_us_per_replay={:.3} touch4_us_per_replay={:.3} touch8_us_per_replay={:.3}",
            us_per_replay(&graph_op, stream, N)?,
            us_per_replay(&graph_fu, stream, N)?,
            us_per_replay(&g4, stream, N)?,
            us_per_replay(&g8, stream, N)?,
        );
    }

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_moe_fused fused 4-launch MoE routed half bit-identical to the 8-launch op \
         path (h, Q8Blocks32, down, y, reruns, graph replays); node counts 8 and 4; router and \
         out-of-range sel checks pass"
    );
    Ok(())
}
