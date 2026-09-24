//! GPU gate for package P0b (docs/gpu-design.md work package): the dense
//! block-0 FFN half as FOUR fused launches vs the EIGHT-launch op path, on
//! the real block-0 tensors and the CUDA oracle dump's own activations
//! (input = dump `ffn_inp-0`'s LAST token column). The contract is BIT
//! IDENTITY, not a band: both paths run the same cores
//! (`cores::q3k_row_dot`, `elem::{rms_scale, silu_mul}`, `q5::q5_row_dot`,
//! the quantizer tail), so any differing bit is a fusion defect.
//! Asserted: the step-1 Q8Act contents AND the f32 normed side output, the
//! step-2 h, the final y, eager vs captured-graph replay for both paths,
//! node counts 8 and 4, and bit-identical reruns of both paths. A second
//! section does the same for the attention chain's MLA key path: the fused
//! `kv_norm_rope_append` against the four launches it replaces
//! (rope + rms_norm + kvr gather + kv_append), on the dump's own
//! `kv_rope_compressed-0` column and the model's real rope cache — `kv_s`,
//! `kvr` and the appended f16 cache row bit for bit, rerun, eager vs
//! captured-graph replay, and the SAME captured graph at a second position. Printed, never asserted: `ik_rel` of
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
use bloomery_gpu_gates::oracle::deepseek2::L_OUT_0;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, bits_equal, bytes_to_words, f32_tensor, load_ref, max_rel_err, open_model,
    ref_manifest, row_bytes, tensor_bytes_as, us_per_replay, verdict,
};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p0b", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    use bloomery_gpu::fused::{FusedKernels, readback_q8act};
    use bloomery_gpu::probe::Probe;
    use bloomery_gpu::q5::{Q8Blocks32, pack_q5_1};
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};

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
        .arch_get_f32("attention.layer_norm_rms_epsilon")
        .ok_or("gate_p0b: metadata <arch>.attention.layer_norm_rms_epsilon missing")?;

    // The input: dump ffn_inp-0's last token column, dims proven first.
    let (inp_row, ffn_inp) = load_ref(&man, "ffn_inp-0", 0)?;
    inp_row.expect("ffn_inp-0", "f32", [K as u64, TOKENS as u64, 1, 1], "in")?;
    let x = ffn_inp[(TOKENS - 1) * K..TOKENS * K].to_vec();
    println!(
        "input op=ffn_inp-0 dims={:?} used=last_token_position_{} K={K} m=1 eps={eps:e}",
        inp_row.ne,
        TOKENS - 1
    );

    // Weights: the real block-0 FFN tensors, each dims [K, rows].
    let gain = f32_tensor(&gguf, "blk.0.ffn_norm.weight", K)?;
    let (_, wg_bytes) = tensor_bytes_as(
        &gguf,
        "blk.0.ffn_gate.weight",
        GgmlType::Q3_K,
        Some(&[K as u64, FF as u64]),
    )?;
    let (_, wu_bytes) = tensor_bytes_as(
        &gguf,
        "blk.0.ffn_up.weight",
        GgmlType::Q3_K,
        Some(&[K as u64, FF as u64]),
    )?;
    let (_, wd_bytes) = tensor_bytes_as(
        &gguf,
        "blk.0.ffn_down.weight",
        GgmlType::Q5_1,
        Some(&[FF as u64, ROWS as u64]),
    )?;

    let rb3 = row_bytes(GgmlType::Q3_K, K)?;
    assert!(
        wg_bytes.len() >= rb3 * FF && wu_bytes.len() >= rb3 * FF,
        "Q3_K file too small"
    );
    let wg_words = bytes_to_words(&wg_bytes[..rb3 * FF]);
    let wu_words = bytes_to_words(&wu_bytes[..rb3 * FF]);
    assert!(
        wg_words.len().is_multiple_of(FF) && wu_words.len().is_multiple_of(FF),
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
    let mut norm_fu = DeviceBuffer::<f32>::zeroed(stream, K)?;
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
    #[allow(
        clippy::too_many_arguments,
        reason = "a gate-local runner threading the buffer set its launches take; folding them into a params struct is R8's axis"
    )]
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
    ) -> Result<(), bloomery_gpu::GpuError> {
        gpu.elem()
            .enqueue_rms_norm(stream, x_dev, gain_dev, eps, K, 1, norm_y)?;
        gpu.enqueue_quantize_q8_1(norm_y, act_op)?;
        gpu.enqueue_gemv_q3k(wg_dev, act_op, gate_y)?;
        gpu.enqueue_gemv_q3k(wu_dev, act_op, up_y)?;
        gpu.elem().enqueue_swiglu(stream, gate_y, up_y, FF, h_op)?;
        gpu.q5()
            .enqueue_quantize_q8(stream, h_op, act32_op, gpu.unlabelled_sink())?;
        gpu.q5()
            .enqueue_gemv_q5_1(stream, wd_dev, act32_op, 0, ROWS, 0, 1, down_y, 0)?;
        gpu.elem().enqueue_add(stream, down_y, x_dev, ROWS, y_op)
    }

    // The fused path: four enqueues, the same intermediates.
    #[allow(
        clippy::too_many_arguments,
        reason = "a gate-local runner threading the buffer set its launches take; folding them into a params struct is R8's axis"
    )]
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
        norm_fu: &mut DeviceBuffer<f32>,
        h_fu: &mut DeviceBuffer<f32>,
        act32_fu: &mut Q8Blocks32,
        y_fu: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        fused.enqueue_norm_quant(
            stream,
            x_dev,
            gain_dev,
            eps,
            act_fu,
            norm_fu,
            gpu.unlabelled_sink(),
        )?;
        fused.enqueue_gate_up_swiglu(stream, wg_dev, wu_dev, act_fu, h_fu)?;
        gpu.q5()
            .enqueue_quantize_q8(stream, h_fu, act32_fu, gpu.unlabelled_sink())?;
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
        &mut norm_fu,
        &mut h_fu,
        &mut act32_fu,
        &mut y_fu,
    )?;
    stream.synchronize()?;
    let act_fu_h = readback_q8act(stream, &act_fu)?;
    let norm_fu_1 = norm_fu.to_host_vec(stream)?;
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
    let norm_same = bits_equal(&norm_y.to_host_vec(stream)?, &norm_fu_1);
    println!(
        "step1b op=rms_norm vs fused=norm_quant[f32_side_output] normed_bit_identical={norm_same} n={K}"
    );
    if !norm_same {
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
        &mut norm_fu,
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
            &mut norm_fu,
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

    // ---- the MLA key path: `kv_norm_rope_append` vs the four launches it
    // replaces, on the dump's own kv_a column and the model's real rope
    // cache. Same shape of proof as above: bits, rerun, eager vs replay,
    // and the same captured graph at a second position.
    {
        use bloomery_gpu::arch::deepseek2::MlaParams;
        use bloomery_gpu::model::StepKernels;

        const CACHE_ROWS: usize = 64;
        let mla = MlaParams::read(&gguf, 0)?;
        let (latent, rope) = (mla.latent, mla.rope_dims);
        let width = latent + rope;
        let step = StepKernels::load(gpu.context())?;

        let (kv_row, kv_all) = load_ref(&man, "kv_rope_compressed-0", 0)?;
        kv_row.expect(
            "kv_rope_compressed-0",
            "f32",
            [width as u64, TOKENS as u64, 1, 1],
            "MUL_MAT",
        )?;
        let kv_a = kv_all[(TOKENS - 1) * width..TOKENS * width].to_vec();
        let kv_gain = f32_tensor(&gguf, "blk.0.attn_kv_a_norm.weight", latent)?;

        let kv_a_dev = DeviceBuffer::from_host(stream, &kv_a)?;
        let kv_gain_dev = DeviceBuffer::from_host(stream, &kv_gain)?;
        let mut cs_dev = DeviceBuffer::from_host(stream, &mla.rope.cache((TOKENS - 1) as u32))?;
        let mut pos_dev = DeviceBuffer::from_host(stream, &[(TOKENS - 1) as u32])?;
        // The `kvr` permutation the op path needs as a gather pair table:
        // [k_rope | kv_compressed], the oracle's CONCAT order.
        let g_src: Vec<u32> = (0..rope)
            .map(|i| (latent + i) as u32)
            .chain((0..latent).map(|i| i as u32))
            .collect();
        let g_dst: Vec<u32> = (0..rope)
            .map(|i| i as u32)
            .chain((0..latent).map(|i| (rope + i) as u32))
            .collect();
        let g_src_dev = DeviceBuffer::from_host(stream, &g_src)?;
        let g_dst_dev = DeviceBuffer::from_host(stream, &g_dst)?;

        let mut kv_s_op = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let mut kvr_op = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let mut cache_op = DeviceTensor::<u16>::zeroed(stream, CACHE_ROWS, width)?;
        let mut kv_s_fu = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let mut kvr_fu = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let mut cache_fu = DeviceTensor::<u16>::zeroed(stream, CACHE_ROWS, width)?;

        #[allow(
            clippy::too_many_arguments,
            reason = "a gate-local runner threading the buffer set its launches take; folding them into a params struct is R8's axis"
        )]
        fn key_op(
            gpu: &Gpu,
            step: &StepKernels,
            stream: &cuda_core::CudaStream,
            kv_a: &DeviceBuffer<f32>,
            gain: &DeviceBuffer<f32>,
            cs: &DeviceBuffer<f32>,
            pos: &DeviceBuffer<u32>,
            src: &DeviceBuffer<u32>,
            dst: &DeviceBuffer<u32>,
            eps: f32,
            latent: usize,
            rope: usize,
            kv_s: &mut DeviceBuffer<f32>,
            kvr: &mut DeviceBuffer<f32>,
            cache: &mut DeviceTensor<u16>,
        ) -> Result<(), bloomery_gpu::GpuError> {
            let width = latent + rope;
            gpu.elem()
                .enqueue_rope(stream, kv_a, cs, rope, (width / rope) as u32, 1, kv_s)?;
            gpu.elem()
                .enqueue_rms_norm(stream, kv_a, gain, eps, latent, 1, kv_s)?;
            step.enqueue_gather(stream, kv_s, src, dst, width, kvr)?;
            gpu.flash()
                .enqueue_kv_append_pos_buf(stream, kvr, pos, cache, 1)?;
            Ok(())
        }

        let run_op_key = |kv_s: &mut DeviceBuffer<f32>,
                          kvr: &mut DeviceBuffer<f32>,
                          cache: &mut DeviceTensor<u16>|
         -> Result<(), bloomery_gpu::GpuError> {
            key_op(
                &gpu,
                &step,
                stream,
                &kv_a_dev,
                &kv_gain_dev,
                &cs_dev,
                &pos_dev,
                &g_src_dev,
                &g_dst_dev,
                mla.eps,
                latent,
                rope,
                kv_s,
                kvr,
                cache,
            )
        };
        run_op_key(&mut kv_s_op, &mut kvr_op, &mut cache_op)?;
        stream.synchronize()?;
        let (kv_s_op_1, kvr_op_1) = (kv_s_op.to_host_vec(stream)?, kvr_op.to_host_vec(stream)?);
        let cache_op_1 = cache_op.buf().to_host_vec(stream)?;

        fused.enqueue_kv_norm_rope_append(
            stream,
            &kv_a_dev,
            &kv_gain_dev,
            &cs_dev,
            &pos_dev,
            mla.eps,
            latent,
            rope,
            &mut kv_s_fu,
            &mut kvr_fu,
            &mut cache_fu,
        )?;
        stream.synchronize()?;
        let (kv_s_fu_1, kvr_fu_1) = (kv_s_fu.to_host_vec(stream)?, kvr_fu.to_host_vec(stream)?);
        let cache_fu_1 = cache_fu.buf().to_host_vec(stream)?;

        let key_same = bits_equal(&kv_s_op_1, &kv_s_fu_1)
            && bits_equal(&kvr_op_1, &kvr_fu_1)
            && cache_op_1 == cache_fu_1;
        println!(
            "key op=rope+rms_norm+gather+kv_append vs fused=kv_norm_rope_append \
             kv_s_bit_identical={} kvr_bit_identical={} cache_bit_identical={} pos={} \
             latent={latent} rope={rope} cache_rows={CACHE_ROWS} {}",
            bits_equal(&kv_s_op_1, &kv_s_fu_1),
            bits_equal(&kvr_op_1, &kvr_fu_1),
            cache_op_1 == cache_fu_1,
            TOKENS - 1,
            verdict(key_same)
        );
        if !key_same {
            ok = false;
        }

        // Reruns: both paths are functions of their inputs alone.
        run_op_key(&mut kv_s_op, &mut kvr_op, &mut cache_op)?;
        fused.enqueue_kv_norm_rope_append(
            stream,
            &kv_a_dev,
            &kv_gain_dev,
            &cs_dev,
            &pos_dev,
            mla.eps,
            latent,
            rope,
            &mut kv_s_fu,
            &mut kvr_fu,
            &mut cache_fu,
        )?;
        stream.synchronize()?;
        let key_rerun = bits_equal(&kvr_op_1, &kvr_op.to_host_vec(stream)?)
            && bits_equal(&kvr_fu_1, &kvr_fu.to_host_vec(stream)?)
            && cache_fu_1 == cache_fu.buf().to_host_vec(stream)?;
        println!("key rerun_bit_identical={key_rerun} {}", verdict(key_rerun));
        if !key_rerun {
            ok = false;
        }

        // Captured graphs: four nodes against one, and each replay equal to
        // its own eager run.
        let g_key_op = gpu.capture(|_| {
            key_op(
                &gpu,
                &step,
                stream,
                &kv_a_dev,
                &kv_gain_dev,
                &cs_dev,
                &pos_dev,
                &g_src_dev,
                &g_dst_dev,
                mla.eps,
                latent,
                rope,
                &mut kv_s_op,
                &mut kvr_op,
                &mut cache_op,
            )
        })?;
        let key_op_nodes = g_key_op.node_count();
        let g_key_fu = gpu.capture(|_| {
            fused.enqueue_kv_norm_rope_append(
                stream,
                &kv_a_dev,
                &kv_gain_dev,
                &cs_dev,
                &pos_dev,
                mla.eps,
                latent,
                rope,
                &mut kv_s_fu,
                &mut kvr_fu,
                &mut cache_fu,
            )
        })?;
        let key_fu_nodes = g_key_fu.node_count();
        g_key_op.launch(stream)?;
        g_key_fu.launch(stream)?;
        stream.synchronize()?;
        let key_replay = bits_equal(&kvr_op_1, &kvr_op.to_host_vec(stream)?)
            && bits_equal(&kv_s_fu_1, &kv_s_fu.to_host_vec(stream)?)
            && bits_equal(&kvr_fu_1, &kvr_fu.to_host_vec(stream)?)
            && cache_fu_1 == cache_fu.buf().to_host_vec(stream)?;
        let key_nodes_ok = key_op_nodes == 4 && key_fu_nodes == 1;
        println!(
            "key graph op_nodes={key_op_nodes} fused_nodes={key_fu_nodes} \
             eager_vs_replay_bit_identical={key_replay} {}",
            verdict(key_nodes_ok && key_replay)
        );
        if !(key_nodes_ok && key_replay) {
            ok = false;
        }

        // The same captured graphs at a second position: the append reads
        // `pos_buf` on the device, so rewriting it moves the landing row.
        // The row written at the first position must survive untouched.
        const POS2: u32 = (TOKENS - 2) as u32;
        pos_dev.copy_from_host(stream, &[POS2])?;
        cs_dev.copy_from_host(stream, &mla.rope.cache(POS2))?;
        g_key_op.launch(stream)?;
        g_key_fu.launch(stream)?;
        stream.synchronize()?;
        let cache_op_2 = cache_op.buf().to_host_vec(stream)?;
        let cache_fu_2 = cache_fu.buf().to_host_vec(stream)?;
        let row1 = (TOKENS - 1) * width;
        let row2 = POS2 as usize * width;
        let second_same = cache_op_2 == cache_fu_2
            && cache_fu_2[row2..row2 + width] != cache_fu_1[row2..row2 + width]
            && cache_fu_2[row1..row1 + width] == cache_fu_1[row1..row1 + width];
        println!(
            "key second_pos pos={POS2} replay_cache_bit_identical_to_op_path={} \
             new_row_written={} first_row_untouched={} {}",
            cache_op_2 == cache_fu_2,
            cache_fu_2[row2..row2 + width] != cache_fu_1[row2..row2 + width],
            cache_fu_2[row1..row1 + width] == cache_fu_1[row1..row1 + width],
            verdict(second_same)
        );
        if !second_same {
            ok = false;
        }
    }

    // ---- printed, never asserted: distance to the CUDA oracle's own
    // outputs on the same position.
    {
        let (l_row, l_out) = load_ref(&man, L_OUT_0, 0)?;
        l_row.expect(L_OUT_0, "f32", [K as u64, TOKENS as u64, 1, 1], "ADD")?;
        let ik_y = max_rel_err(&y_fu_1, &l_out[(TOKENS - 1) * K..TOKENS * K])?;
        let (ug_row, up_gate) = load_ref(&man, "ffn_up_gate-0", 0)?;
        // The dump's fused gate/up output: dims match h exactly.
        ug_row.expect(
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
        "PASSED: gate_p0b fused 4-launch block-0 FFN bit-identical to the 8-launch op path \
         (Q8Act, f32 normed, h, y, reruns, graph replays); node counts 8 and 4. The MLA key \
         path fused to 1 launch bit-identical to its 4 (kv_s, kvr, cache row, rerun, replay, \
         second position)"
    );
    Ok(())
}
