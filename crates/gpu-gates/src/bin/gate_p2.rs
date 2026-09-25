//! GPU kernel gate for package P2 (docs/gpu-design.md work package): Q5_0 gemv
//! on `blk.1.ffn_down_exps.weight` (the flat 64-expert stack, experts reached
//! by `row0`) and Q5_1 on `blk.0.ffn_down.weight`, each within `KERNEL_BAND`
//! against `ref_gemv` over the same rows. Also pins: bit-identical rerun per
//! shape, `col0`/`y0` addressing with untouched neighbours, and eager vs
//! captured-graph byte identity for one shape.
//!
//! The reference is fed the q8_1-quantized activations (`q8_hat`, the exact
//! host mirror of the device quantizer's semantics: per 32-value block
//! d = amax/127, x̂ = round(x/d)·d), so the gate measures the implementation
//! against its own design the way the stage-0 gates measured against the
//! quantizing oracle. Against raw activations this design's floor is far
//! higher at these shapes (each shape also prints `raw_x_rel`, non-gating):
//! PIN(2026-09-21): measured with `activations()` at K=1408/K=10944,
//! 32-value q8 blocks, the raw-x deviation of a correct kernel is
//! ~8e-3..1.1e-2 (synthetic-floor probe 8.0–9.3e-3; gate runs 7.0e-3–1.1e-2),
//! i.e. the stage-0 "3–5e-3 floor" premise does not transfer to this
//! generator and geometry — the raw-x comparison cannot sit under 1e-2
//! here, so it is reported, not asserted.
//! On real activations (`just rawx-floor`, Q5_0 moe_down) the same
//! deviation measures 1.3–1.4e-2.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p2: built without the `gpu` feature; see `just gate-gpu-p2`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, bits_equal, verdict};

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p2", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    use bloomery_gpu::q5::{Q5Kernels, Q8Blocks32, pack_q5_0, pack_q5_1};
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_gates::{
        KERNEL_BAND, activations, max_rel_err, open_model, ref_gemv, row_bytes, tensor_bytes_as,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::GgmlType;

    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let q5 = Q5Kernels::load(gpu.context(), gpu.fault_word())?;
    let stream = gpu.stream();

    // ---- Q5_0: the expert stack, experts 0/5/63 by row0 (flat-stack
    // addressing, no gather copy), m in {1, 8}.
    // dims [K, rows, experts]
    let (_, w_bytes) = tensor_bytes_as(
        &gguf,
        "blk.1.ffn_down_exps.weight",
        GgmlType::Q5_0,
        Some(&[1408, 2048, 64]),
    )?;
    let (k0, rows0, n_exp) = (1408usize, 2048usize, 64usize);
    let rb0 = row_bytes(GgmlType::Q5_0, k0)?;
    let k_blocks0 = k0 / 32;
    let q_stride0 = 256 * k_blocks0.div_ceil(32);
    let w_dev = DeviceTensor::upload(
        stream,
        &pack_q5_0(w_bytes, k0, n_exp * rows0)?,
        n_exp * rows0,
        q_stride0 + k_blocks0,
    )?;
    let mut case = 0u32;
    for expert in [0usize, 5, 63] {
        for m in [1usize, 8] {
            case += 1;
            let x = activations(k0, m, 7000 + case * 17);
            let expert_bytes = &w_bytes[expert * rows0 * rb0..][..rows0 * rb0];
            let y_ref = ref_gemv(GgmlType::Q5_0, expert_bytes, k0, rows0, &q8_hat(&x, k0), m)?;
            let y_raw = ref_gemv(GgmlType::Q5_0, expert_bytes, k0, rows0, &x, m)?;
            let x_dev = DeviceBuffer::from_host(stream, &x)?;
            let mut act = Q8Blocks32::new(stream, k0, m)?;
            let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows0 * m)?;
            q5.enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
            q5.enqueue_gemv_q5_0(
                stream,
                &w_dev,
                &act,
                expert * rows0,
                rows0,
                0,
                m,
                &mut y_dev,
                0,
            )?;
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            q5.enqueue_gemv_q5_0(
                stream,
                &w_dev,
                &act,
                expert * rows0,
                rows0,
                0,
                m,
                &mut y_dev,
                0,
            )?;
            stream.synchronize()?;
            let y2 = y_dev.to_host_vec(stream)?;
            let bit_same = bits_equal(&y, &y2);
            let rel = max_rel_err(&y, &y_ref)?;
            let raw = max_rel_err(&y, &y_raw)?;
            let pass = rel <= KERNEL_BAND && bit_same;
            println!(
                "shape type=q5_0 K={k0} rows={rows0} m={m} expert={expert} max_rel_err={rel:.3e} raw_x_rel={raw:.3e} bit_identical_rerun={bit_same} {}",
                verdict(pass)
            );
            if !pass {
                ok = false;
            }
        }
    }

    // ---- col0/y0 addressing: quantize 8 columns once, gemv reads column 3
    // (m = 1) of expert 5 into y0 = 2*rows0 of a 4-segment buffer; the
    // segments on both sides of the span must keep the sentinel.
    let expert = 5usize;
    let x8 = activations(k0, 8, 9100);
    let expert_bytes = &w_bytes[expert * rows0 * rb0..][..rows0 * rb0];
    let y_ref8 = ref_gemv(GgmlType::Q5_0, expert_bytes, k0, rows0, &q8_hat(&x8, k0), 8)?;
    const SENT: f32 = 1.0e30;
    let x_dev = DeviceBuffer::from_host(stream, &x8)?;
    let mut act = Q8Blocks32::new(stream, k0, 8)?;
    q5.enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
    let (col0, y0) = (3usize, 2 * rows0);
    let mut y_dev = DeviceBuffer::from_host(stream, &vec![SENT; 4 * rows0])?;
    q5.enqueue_gemv_q5_0(
        stream,
        &w_dev,
        &act,
        expert * rows0,
        rows0,
        col0,
        1,
        &mut y_dev,
        y0,
    )?;
    stream.synchronize()?;
    let y = y_dev.to_host_vec(stream)?;
    let col_ref: Vec<f32> = (0..rows0).map(|r| y_ref8[r * 8 + col0]).collect();
    let rel = max_rel_err(&y[y0..y0 + rows0], &col_ref)?;
    let neighbours = y[..y0]
        .iter()
        .chain(&y[y0 + rows0..])
        .all(|&v| v.to_bits() == SENT.to_bits());
    let pass = rel <= KERNEL_BAND && neighbours;
    println!(
        "shape type=q5_0 K={k0} rows={rows0} m=1 col0={col0} y0={y0} max_rel_err={rel:.3e} neighbours_untouched={neighbours} {}",
        verdict(pass)
    );
    if !pass {
        ok = false;
    }

    // ---- Q5_1: the dense down projection, all rows, m in {1, 8}.
    // dims [K, rows]
    let (_, w1_bytes) = tensor_bytes_as(
        &gguf,
        "blk.0.ffn_down.weight",
        GgmlType::Q5_1,
        Some(&[10944, 2048]),
    )?;
    let (k1, rows1) = (10944usize, 2048usize);
    let k_blocks1 = k1 / 32;
    let q_stride1 = 256 * k_blocks1.div_ceil(32);
    let w1_dev = DeviceTensor::upload(
        stream,
        &pack_q5_1(w1_bytes, k1, rows1)?,
        rows1,
        q_stride1 + 2 * k_blocks1,
    )?;
    for m in [1usize, 8] {
        let x = activations(k1, m, 8300 + m as u32);
        let y_ref = ref_gemv(GgmlType::Q5_1, w1_bytes, k1, rows1, &q8_hat(&x, k1), m)?;
        let y_raw = ref_gemv(GgmlType::Q5_1, w1_bytes, k1, rows1, &x, m)?;
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut act = Q8Blocks32::new(stream, k1, m)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows1 * m)?;
        q5.enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
        q5.enqueue_gemv_q5_1(stream, &w1_dev, &act, 0, rows1, 0, m, &mut y_dev, 0)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        q5.enqueue_gemv_q5_1(stream, &w1_dev, &act, 0, rows1, 0, m, &mut y_dev, 0)?;
        stream.synchronize()?;
        let y2 = y_dev.to_host_vec(stream)?;
        let bit_same = bits_equal(&y, &y2);
        let rel = max_rel_err(&y, &y_ref)?;
        let raw = max_rel_err(&y, &y_raw)?;
        let pass = rel <= KERNEL_BAND && bit_same;
        println!(
            "shape type=q5_1 K={k1} rows={rows1} m={m} max_rel_err={rel:.3e} raw_x_rel={raw:.3e} bit_identical_rerun={bit_same} {}",
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    // ---- eager vs captured graph (q5_1, m = 8): the step shape. Resident
    // buffers, the two-kernel sequence enqueued eagerly vs captured once and
    // replayed onto the same addresses — byte identity, two graph nodes.
    let x = activations(k1, 8, 9900);
    let y_ref = ref_gemv(GgmlType::Q5_1, w1_bytes, k1, rows1, &q8_hat(&x, k1), 8)?;
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Blocks32::new(stream, k1, 8)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows1 * 8)?;
    q5.enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
    q5.enqueue_gemv_q5_1(stream, &w1_dev, &act, 0, rows1, 0, 8, &mut y_dev, 0)?;
    stream.synchronize()?;
    let y_eager = y_dev.to_host_vec(stream)?;

    y_dev.zero_async(stream)?;
    stream.synchronize()?;
    let graph = gpu.capture(|_s| {
        q5.enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
        q5.enqueue_gemv_q5_1(stream, &w1_dev, &act, 0, rows1, 0, 8, &mut y_dev, 0)
    })?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let y_graph = y_dev.to_host_vec(stream)?;

    let identical = bits_equal(&y_eager, &y_graph);
    let rel = max_rel_err(&y_graph, &y_ref)?;
    let nodes = graph.node_count();
    let pass = identical && nodes == 2 && rel <= KERNEL_BAND;
    println!(
        "graph type=q5_1 K={k1} rows={rows1} m=8 eager_vs_graph_bit_identical={identical} graph_nodes={nodes} max_rel_err={rel:.3e} {}",
        verdict(pass)
    );
    if !pass {
        ok = false;
    }

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: q5_0/q5_1 gemv within KERNEL_BAND of the quantized-input reference; col0/y0 addressing clean; eager == graph replay"
    );
    Ok(())
}

/// Host mirror of the device q8_1 quantizer's semantics: per 32-value block
/// of each of the m columns, d = amax/127 (1.0 for an all-zero block) and
/// x̂ = round(x/d)·d clamped to ±127 levels. Independent of the device path
/// (plain f32 scalar code), so the gate still catches a wrong d, a wrong
/// block split, or a layout mix-up on either side.
#[cfg(feature = "gpu")]
fn q8_hat(x: &[f32], k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for col in 0..x.len() / k {
        for b in 0..k / 32 {
            let vals = &x[col * k + 32 * b..][..32];
            let amax = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (o, &v) in out[col * k + 32 * b..][..32].iter_mut().zip(vals) {
                *o = (v / d).round().clamp(-127.0, 127.0) * d;
            }
        }
    }
    out
}
