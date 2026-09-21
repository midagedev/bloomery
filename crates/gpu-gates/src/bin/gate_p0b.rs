//! GPU gate for package P0b (docs/gpu-design.md 작업 꾸러미): the dense
//! block-0 FFN half as FOUR fused launches vs the EIGHT-launch op path, on
//! the real block-0 tensors and the CUDA oracle dump's own activations
//! (input = dump `ffn_inp-0`'s LAST token column). The contract is BIT
//! IDENTITY, not a band: both paths run the same cores
//! (`cores::q3k_row_dot`, `elem::{rms_scale, silu_mul}`, `q5::q5_row_dot`,
//! the quantizer tail), so any differing bit is a fusion defect.
//! Asserted: the step-1 Q8Act contents, the step-2 h, the final y, eager vs
//! captured-graph replay for both paths, node counts 8 and 4, and
//! bit-identical reruns of both paths. Printed, never asserted: `ik_rel` of
//! y against dump `l_out-0`'s last column and of h against `ffn_up_gate-0`'s
//! (the block-layer bands are pinned by the lead from these numbers, not
//! here). `--time` (lead-only, under the machine lease) replays each
//! captured graph 2000x and prints us/replay plus 4- and 8-node empty-graph
//! references; without the flag nothing is timed or printed about time.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p0b: built without the `gpu` feature; see `just gate-gpu-p0b`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu_gates::RefRow;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    bytes_to_words, find_ref_row, max_rel_err, open_model, ref_manifest, ref_tensor_of, row_bytes,
    tensor_bytes,
};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bloomery_gpu::fused::{FusedKernels, readback_q8act};
    use bloomery_gpu::probe::Probe;
    use bloomery_gpu::q5::{Q8Blocks32, pack_q5_1};
    use bloomery_gpu::{DeviceTensor, Gpu, Graph, Q8Act};

    // The block-0 shapes (asserted from the file below): hidden width 2048,
    // FFN intermediate 10944 (gate/up rows, down K), down rows 2048. The
    // dump was made with a six-token prompt; the decode shape takes the
    // LAST position.
    const K: usize = 2048;
    const FF: usize = 10944;
    const ROWS: usize = 2048;
    const TOKENS: usize = 6;

    let time_mode = std::env::args().any(|a| a == "--time");
    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let fused = FusedKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let man = ref_manifest()?;

    // The one architecture-wide rms epsilon (as gate_p4 reads it).
    let eps: f32 = gguf
        .architecture()
        .and_then(|a| gguf.value(&format!("{a}.attention.layer_norm_rms_epsilon")))
        .and_then(gguf::Value::as_f32)
        .ok_or("gate_p0b: metadata <arch>.attention.layer_norm_rms_epsilon missing")?;

    // The input: dump ffn_inp-0's last token column, dims proven first.
    let (inp_row, ffn_inp) = load_ref(&man, "ffn_inp-0", 0)?;
    expect(
        &inp_row,
        "ffn_inp-0",
        "f32",
        [K as u64, TOKENS as u64, 1, 1],
        "in",
    )?;
    let x = ffn_inp[(TOKENS - 1) * K..TOKENS * K].to_vec();
    println!(
        "input op=ffn_inp-0 dims={:?} used=last_token_position_{} K={K} m=1 eps={eps:e}",
        inp_row.ne,
        TOKENS - 1
    );

    // Weights: the real block-0 FFN tensors.
    let gain = f32_tensor(&gguf, "blk.0.ffn_norm.weight", K)?;
    let (wg_info, wg_bytes) = tensor_bytes(&gguf, "blk.0.ffn_gate.weight")?;
    assert_eq!(wg_info.ty, GgmlType::Q3_K, "blk.0.ffn_gate type");
    assert_eq!(
        wg_info.dims,
        [K as u64, FF as u64],
        "blk.0.ffn_gate dims [K, rows]"
    );
    let (wu_info, wu_bytes) = tensor_bytes(&gguf, "blk.0.ffn_up.weight")?;
    assert_eq!(wu_info.ty, GgmlType::Q3_K, "blk.0.ffn_up type");
    assert_eq!(
        wu_info.dims,
        [K as u64, FF as u64],
        "blk.0.ffn_up dims [K, rows]"
    );
    let (wd_info, wd_bytes) = tensor_bytes(&gguf, "blk.0.ffn_down.weight")?;
    assert_eq!(wd_info.ty, GgmlType::Q5_1, "blk.0.ffn_down type");
    assert_eq!(
        wd_info.dims,
        [FF as u64, ROWS as u64],
        "blk.0.ffn_down dims [K, rows]"
    );

    let rb3 = row_bytes(GgmlType::Q3_K, K)?;
    assert!(
        wg_bytes.len() >= rb3 * FF && wu_bytes.len() >= rb3 * FF,
        "Q3_K file too small"
    );
    let wg_words = bytes_to_words(&wg_bytes[..rb3 * FF]);
    let wu_words = bytes_to_words(&wu_bytes[..rb3 * FF]);
    assert!(
        wg_words.len() % FF == 0 && wu_words.len() % FF == 0,
        "Q3_K words not row-divisible"
    );
    let wg_dev = DeviceTensor::upload(stream, &wg_words, FF, rb3 / 4)?;
    let wu_dev = DeviceTensor::upload(stream, &wu_words, FF, rb3 / 4)?;
    let k_blocks = FF / 32;
    let q_stride = 256 * k_blocks.div_ceil(32);
    let wd_dev = DeviceTensor::upload(
        stream,
        &pack_q5_1(wd_bytes, FF, ROWS)?,
        ROWS,
        q_stride + 2 * k_blocks,
    )?;

    // Resident buffers, one set per path (the graphs freeze addresses).
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let gain_dev = DeviceBuffer::from_host(stream, &gain)?;
    let mut norm_y = DeviceBuffer::<f32>::zeroed(stream, K)?;
    let mut act_op = Q8Act::with_k(stream, 1, K)?;
    let mut act_fu = Q8Act::with_k(stream, 1, K)?;
    let mut gate_y = DeviceBuffer::<f32>::zeroed(stream, FF)?;
    let mut up_y = DeviceBuffer::<f32>::zeroed(stream, FF)?;
    let mut h_op = DeviceBuffer::<f32>::zeroed(stream, FF)?;
    let mut h_fu = DeviceBuffer::<f32>::zeroed(stream, FF)?;
    let mut act32_op = Q8Blocks32::new(stream, FF, 1)?;
    let mut act32_fu = Q8Blocks32::new(stream, FF, 1)?;
    let mut down_y = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;
    let mut y_op = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;
    let mut y_fu = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;

    // The op path: eight enqueues, intermediates read back after its steps
    // 1-2 (norm+quantize) and 5 (swiglu). The residual add is
    // add(a=down output, b=ffn_inp) — the dump's `l_out-0 = ffn_out-0 +
    // ffn_inp-0` operand order.
    #[allow(clippy::too_many_arguments)]
    fn run_op(
        gpu: &Gpu,
        stream: &cuda_core::CudaStream,
        x_dev: &DeviceBuffer<f32>,
        gain_dev: &DeviceBuffer<f32>,
        wg_dev: &DeviceTensor<u32>,
        wu_dev: &DeviceTensor<u32>,
        wd_dev: &DeviceTensor<u32>,
        eps: f32,
        norm_y: &mut DeviceBuffer<f32>,
        act_op: &mut Q8Act,
        gate_y: &mut DeviceBuffer<f32>,
        up_y: &mut DeviceBuffer<f32>,
        h_op: &mut DeviceBuffer<f32>,
        act32_op: &mut Q8Blocks32,
        down_y: &mut DeviceBuffer<f32>,
        y_op: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        gpu.elem()
            .enqueue_rms_norm(stream, x_dev, gain_dev, eps, K, 1, norm_y)?;
        gpu.enqueue_quantize_q8_1(norm_y, act_op)?;
        gpu.enqueue_gemv_q3k(wg_dev, act_op, gate_y)?;
        gpu.enqueue_gemv_q3k(wu_dev, act_op, up_y)?;
        gpu.elem().enqueue_swiglu(stream, gate_y, up_y, FF, h_op)?;
        gpu.q5().enqueue_quantize_q8(stream, h_op, act32_op)?;
        gpu.q5()
            .enqueue_gemv_q5_1(stream, wd_dev, act32_op, 0, ROWS, 0, 1, down_y, 0)?;
        gpu.elem().enqueue_add(stream, down_y, x_dev, ROWS, y_op)
    }

    // The fused path: four enqueues, the same intermediates.
    #[allow(clippy::too_many_arguments)]
    fn run_fu(
        gpu: &Gpu,
        fused: &FusedKernels,
        stream: &cuda_core::CudaStream,
        x_dev: &DeviceBuffer<f32>,
        gain_dev: &DeviceBuffer<f32>,
        wg_dev: &DeviceTensor<u32>,
        wu_dev: &DeviceTensor<u32>,
        wd_dev: &DeviceTensor<u32>,
        eps: f32,
        act_fu: &mut Q8Act,
        h_fu: &mut DeviceBuffer<f32>,
        act32_fu: &mut Q8Blocks32,
        y_fu: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fused.enqueue_norm_quant(stream, x_dev, gain_dev, eps, act_fu)?;
        fused.enqueue_gate_up_swiglu(stream, wg_dev, wu_dev, act_fu, h_fu)?;
        gpu.q5().enqueue_quantize_q8(stream, h_fu, act32_fu)?;
        fused.enqueue_down_add_q5_1(stream, wd_dev, act32_fu, x_dev, y_fu)
    }

    // ---- eager runs with readbacks
    run_op(
        &gpu,
        stream,
        &x_dev,
        &gain_dev,
        &wg_dev,
        &wu_dev,
        &wd_dev,
        eps,
        &mut norm_y,
        &mut act_op,
        &mut gate_y,
        &mut up_y,
        &mut h_op,
        &mut act32_op,
        &mut down_y,
        &mut y_op,
    )?;
    stream.synchronize()?;
    let act_op_h = readback_q8act(stream, &act_op)?;
    let h_op_1 = h_op.to_host_vec(stream)?;
    let y_op_1 = y_op.to_host_vec(stream)?;

    run_fu(
        &gpu,
        &fused,
        stream,
        &x_dev,
        &gain_dev,
        &wg_dev,
        &wu_dev,
        &wd_dev,
        eps,
        &mut act_fu,
        &mut h_fu,
        &mut act32_fu,
        &mut y_fu,
    )?;
    stream.synchronize()?;
    let act_fu_h = readback_q8act(stream, &act_fu)?;
    let h_fu_1 = h_fu.to_host_vec(stream)?;
    let y_fu_1 = y_fu.to_host_vec(stream)?;

    // ---- the bit-identity table
    let act_same = act_op_h.q3 == act_fu_h.q3
        && act_op_h.q4 == act_fu_h.q4
        && act_op_h.q6 == act_fu_h.q6
        && act_op_h.s8 == act_fu_h.s8
        && act_op_h
            .d8
            .iter()
            .zip(&act_fu_h.d8)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "step1 op=norm+quantize_q8_1 vs fused=norm_quant Q8Act_bit_identical={act_same} \
         (q3 {} u64, q4 {} u32, q6 {} u32, s8 {} i32, d8 {} f32)",
        act_op_h.q3.len(),
        act_op_h.q4.len(),
        act_op_h.q6.len(),
        act_op_h.s8.len(),
        act_op_h.d8.len()
    );
    if !act_same {
        ok = false;
    }
    let h_same = bits_equal(&h_op_1, &h_fu_1);
    println!(
        "step2 op=gate_gemv+up_gemv+swiglu vs fused=gate_up_swiglu_q3k h_bit_identical={h_same} n={FF}"
    );
    if !h_same {
        ok = false;
    }
    let y_same = bits_equal(&y_op_1, &y_fu_1);
    println!("final op=down_gemv+add vs fused=down_add_q5_1 y_bit_identical={y_same} n={ROWS}");
    if !y_same {
        ok = false;
    }

    // ---- bit-identical reruns of both paths
    run_op(
        &gpu,
        stream,
        &x_dev,
        &gain_dev,
        &wg_dev,
        &wu_dev,
        &wd_dev,
        eps,
        &mut norm_y,
        &mut act_op,
        &mut gate_y,
        &mut up_y,
        &mut h_op,
        &mut act32_op,
        &mut down_y,
        &mut y_op,
    )?;
    stream.synchronize()?;
    let y_op_2 = y_op.to_host_vec(stream)?;
    run_fu(
        &gpu,
        &fused,
        stream,
        &x_dev,
        &gain_dev,
        &wg_dev,
        &wu_dev,
        &wd_dev,
        eps,
        &mut act_fu,
        &mut h_fu,
        &mut act32_fu,
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
            &x_dev,
            &gain_dev,
            &wg_dev,
            &wu_dev,
            &wd_dev,
            eps,
            &mut norm_y,
            &mut act_op,
            &mut gate_y,
            &mut up_y,
            &mut h_op,
            &mut act32_op,
            &mut down_y,
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
            &fused,
            stream,
            &x_dev,
            &gain_dev,
            &wg_dev,
            &wu_dev,
            &wd_dev,
            eps,
            &mut act_fu,
            &mut h_fu,
            &mut act32_fu,
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
        let (l_row, l_out) = load_ref(&man, "l_out-0", 0)?;
        expect(
            &l_row,
            "l_out-0",
            "f32",
            [K as u64, TOKENS as u64, 1, 1],
            "ADD",
        )?;
        let ik_y = max_rel_err(&y_fu_1, &l_out[(TOKENS - 1) * K..TOKENS * K])?;
        let (ug_row, up_gate) = load_ref(&man, "ffn_up_gate-0", 0)?;
        // The dump's fused gate/up output: dims match h exactly.
        expect(
            &ug_row,
            "ffn_up_gate-0",
            "f32",
            [FF as u64, TOKENS as u64, 1, 1],
            "FUSED_UP_GATE",
        )?;
        let ik_h = max_rel_err(&h_fu_1, &up_gate[(TOKENS - 1) * FF..TOKENS * FF])?;
        println!(
            "ik l_out-0[last] ik_rel={ik_y:.3e} ffn_up_gate-0[last] ik_rel={ik_h:.3e} (printed, not asserted)"
        );
    }

    // ---- lead-only timing under the machine lease; correctness runs never
    // reach this. Replays each captured graph N times, one synchronize at
    // the end, plus 4- and 8-node empty-graph references.
    if time_mode {
        const N: u32 = 2000;
        let probe = Probe::load(gpu.context())?;
        let mut tbuf = DeviceBuffer::<f32>::zeroed(stream, 32)?;
        let g4 =
            gpu.capture(|_| (0..4).try_for_each(|_| probe.enqueue_touch(stream, &mut tbuf)))?;
        let g8 =
            gpu.capture(|_| (0..8).try_for_each(|_| probe.enqueue_touch(stream, &mut tbuf)))?;
        fn us_per_replay(g: &Graph, stream: &cuda_core::CudaStream, n: u32) -> f64 {
            // Correctness-run code never calls this; unwrap keeps the probe
            // readable.
            g.launch(stream).unwrap();
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                g.launch(stream).unwrap();
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / f64::from(n)
        }
        println!(
            "time n={N} op_us_per_replay={:.3} fused_us_per_replay={:.3} touch4_us_per_replay={:.3} touch8_us_per_replay={:.3}",
            us_per_replay(&graph_op, stream, N),
            us_per_replay(&graph_fu, stream, N),
            us_per_replay(&g4, stream, N),
            us_per_replay(&g8, stream, N),
        );
    }

    if !ok {
        eprintln!("FAILED: gate_p0b");
        std::process::exit(1);
    }
    println!(
        "PASSED: gate_p0b fused 4-launch block-0 FFN bit-identical to the 8-launch op path \
         (Q8Act, h, y, reruns, graph replays); node counts 8 and 4"
    );
    Ok(())
}

/// Load `(name, occurrence)` from the dump with its manifest row.
#[cfg(feature = "gpu")]
fn load_ref(
    man: &[RefRow],
    name: &str,
    occ: u32,
) -> Result<(RefRow, Vec<f32>), Box<dyn std::error::Error>> {
    let row = find_ref_row(man, name, occ)?;
    Ok((row.clone(), ref_tensor_of(row)?))
}

/// Prove a dump row's type, dims and op — the chain check every consumer
/// runs before trusting a tensor. `op = "in"` only checks type and dims.
#[cfg(feature = "gpu")]
fn expect(
    row: &RefRow,
    what: &str,
    ty: &str,
    ne: [u64; 4],
    op: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if row.ty != ty || row.ne != ne || (op != "in" && row.op != op) {
        return Err(format!(
            "gate_p0b: {what}: {} is {} {:?} op {}, want {} {:?} op {}",
            row.name, row.ty, row.ne, row.op, ty, ne, op
        )
        .into());
    }
    Ok(())
}

/// An F32 tensor from the file as f32 (norm gains), length-checked.
#[cfg(feature = "gpu")]
fn f32_tensor(
    gguf: &gguf::Gguf,
    name: &str,
    want: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (t, b) = tensor_bytes(gguf, name)?;
    if t.ty != GgmlType::F32 || b.len() != want * 4 {
        return Err(format!(
            "gate_p0b: {name} is {:?} with {} bytes, want F32 x {want}",
            t.ty,
            b.len()
        )
        .into());
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

#[cfg(feature = "gpu")]
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(feature = "gpu")]
fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}
