//! gpu-spike — a binary in its own package calling the bloomery-gpu library,
//! and the P0 gate: an eagerly enqueued kernel sequence and its captured
//! graph replay must produce bit-identical output bytes.
//!
//! Correctness uses the same files the stage-0 binary's q4k rows use
//! (crates/q3k-gemv/src/main.rs): `$BLOOMERY_DATA/attn.q4k` (27 concatenated
//! [2048,2048] Q4_K tensors; attn0 = the first), `x_m1.f32`/`x_m8.f32` and
//! `y_ref_attn0_*.f32`, gated at max |err| / max |ref| <= 1e-2 (the stage-0
//! q8_1-activation design gate).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gpu-spike: built without the `gpu` feature; the GPU path is off.");
    eprintln!("gpu build: cargo oxide build --arch sm_86 -- -p bloomery-gpu-spike --features gpu");
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use cuda_core::DeviceBuffer;

    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    let gpu = Gpu::new()?;
    let profile = if cfg!(debug_assertions) {
        "dev"
    } else {
        "release"
    };
    println!("host build profile: {profile}");

    let w_bytes = std::fs::read(format!("{data}/attn.q4k"))?;
    assert!(w_bytes.len() % 4 == 0, "attn.q4k not u32-divisible");
    assert_eq!(
        w_bytes.len(),
        55296 * 1152,
        "attn.q4k vs 27 x [2048,2048] Q4_K rows"
    );
    let w_host: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let x_m1 = read_f32(&format!("{data}/x_m1.f32"));
    let x_m8 = read_f32(&format!("{data}/x_m8.f32"));
    assert_eq!(x_m1.len(), 2048);
    assert_eq!(x_m8.len(), 2048 * 8);

    // attn0 = the first [2048,2048] tensor of the stack; the m=8 row reads
    // the same x_m8 draw the stage-0 binary does.
    let mut all_ok = true;
    for (name, n, m, x) in [
        ("attn0_m1", 2048usize, 1usize, &x_m1),
        ("attn0_m8", 2048, 8, &x_m8),
    ] {
        let y = gpu.gemv_q4k(&w_host[..n * 288], x, n, m)?;
        let y2 = gpu.gemv_q4k(&w_host[..n * 288], x, n, m)?;
        let bit_same = y
            .iter()
            .zip(y2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let yref = read_f32(&format!("{data}/y_ref_{name}.f32"));
        assert_eq!(yref.len(), n * m);
        let denom = yref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let maxerr = y
            .iter()
            .zip(yref.iter())
            .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
        let rel = maxerr / denom;
        println!(
            "shape {name:<10} N={n:>6} M={m} max_rel_err={rel:.3e} bit_identical_rerun={bit_same}"
        );
        if rel > 1e-2 {
            eprintln!("FAIL: {name} rel err {rel:.3e} exceeds 1e-2");
            all_ok = false;
        }
    }

    // P0 gate: the step shape. Resident weight, resident activation scratch,
    // resident output; the two-kernel sequence enqueued eagerly on the engine
    // stream vs the same sequence captured once and replayed. Both write the
    // same addresses, so the outputs must be byte-identical; the graph's node
    // count is the launch-count column of the first lease measurement.
    let (n, m) = (2048usize, 1usize);
    let stream = gpu.stream();
    let w_dev = DeviceTensor::upload(stream, &w_host[..n * 288], n, 288)?;
    let x_dev = DeviceBuffer::from_host(stream, &x_m1)?;
    let mut act = Q8Act::new(stream, m)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n * m)?;

    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    gpu.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
    stream.synchronize()?;
    let y_eager = y_dev.to_host_vec(stream)?;

    y_dev.zero_async(stream)?;
    stream.synchronize()?;
    let graph = gpu.capture(|_s| {
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        gpu.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
        Ok(())
    })?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let y_graph = y_dev.to_host_vec(stream)?;

    let identical = y_eager.len() == y_graph.len()
        && y_eager
            .iter()
            .zip(y_graph.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let ref_ok = {
        let yref = read_f32(&format!("{data}/y_ref_attn0_m1.f32"));
        let denom = yref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let maxerr = y_graph
            .iter()
            .zip(yref.iter())
            .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
        maxerr / denom <= 1e-2
    };
    println!(
        "P0 gate: eager_vs_graph_bit_identical={identical} graph_nodes={} graph_within_1e-2_of_ref={ref_ok}",
        graph.node_count()
    );
    if !identical || !ref_ok || graph.node_count() != 2 {
        eprintln!("FAIL: P0 gate (expected bit-identical, 2 nodes, within 1e-2)");
        all_ok = false;
    }

    // Host submission cost, step shape: enqueue the two-kernel sequence
    // `iters` times with ONE synchronize at the end, eager vs graph replay.
    // Design figures (shared box, dev or release host build as printed
    // above), not benchmarks of record; the lease measurement is P8's.
    let iters = 1000u32;
    for _ in 0..10 {
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        gpu.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
        graph.launch(stream)?;
    }
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        gpu.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
    }
    let eager_submit_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    stream.synchronize()?;
    let eager_total_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        graph.launch(stream)?;
    }
    let graph_submit_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    stream.synchronize()?;
    let graph_total_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    println!(
        "submission probe n={n} m={m} x{iters}: eager submit {eager_submit_us:.2} us/step (incl. device {eager_total_us:.2}), \
         graph submit {graph_submit_us:.2} us/step (incl. device {graph_total_us:.2})"
    );

    // Node-gap probe: one graph of N nodes (the two-kernel pair repeated),
    // replayed with a synchronize per replay, so us/node is what a step of N
    // op-kernels costs on the device once host submission is gone. Two row
    // counts: at 8 rows the gemv body is near-empty and the figure reads as
    // the per-node gap; at 2048 rows it carries a real attn-sized body.
    // Design figures, same standing as the submission probe above.
    for rows in [8usize, 2048] {
        let w_probe = DeviceTensor::upload(stream, &w_host[..rows * 288], rows, 288)?;
        let mut y_probe = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
        for pairs in [50usize, 150, 350] {
            let g = gpu.capture(|_s| {
                for _ in 0..pairs {
                    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
                    gpu.enqueue_gemv_q4k(&w_probe, &act, &mut y_probe)?;
                }
                Ok(())
            })?;
            for _ in 0..5 {
                g.launch(stream)?;
            }
            stream.synchronize()?;
            let reps = 50u32;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                g.launch(stream)?;
                stream.synchronize()?;
            }
            let replay_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            println!(
                "node-gap probe rows={rows} nodes={}: replay+sync {replay_us:.1} us = {:.2} us/node",
                g.node_count(),
                replay_us / g.node_count() as f64
            );
        }
    }

    // The same probe over a one-store kernel from the crate's second device
    // module: the per-node cost with no body to speak of.
    let probe = bloomery_gpu::probe::Probe::load(gpu.context())?;
    let mut y_touch = DeviceBuffer::<f32>::zeroed(stream, 32)?;
    for nodes in [100usize, 700] {
        let g = gpu.capture(|s| {
            for _ in 0..nodes {
                probe.enqueue_touch(s, &mut y_touch)?;
            }
            Ok(())
        })?;
        for _ in 0..5 {
            g.launch(stream)?;
        }
        stream.synchronize()?;
        let reps = 50u32;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            g.launch(stream)?;
            stream.synchronize()?;
        }
        let replay_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        println!(
            "node-gap probe touch nodes={}: replay+sync {replay_us:.1} us = {:.2} us/node",
            g.node_count(),
            replay_us / g.node_count() as f64
        );
    }
    let touched = y_touch.to_host_vec(stream)?;
    if touched.iter().any(|&v| v != 1.0) {
        eprintln!("FAIL: touch kernel from the second device module did not write its 32 elements");
        all_ok = false;
    }

    // Bare launch+sync and full-call figures, kept from the packaging spike.
    let (n, m) = (8usize, 1usize);
    let launch_us = gpu.probe_q4k_launch_us(&w_host[..n * 288], &x_m1, n, m, 1000)?;
    let t0 = std::time::Instant::now();
    for _ in 0..1000 {
        gpu.gemv_q4k(&w_host[..n * 288], &x_m1, n, m)?;
    }
    let call_us = t0.elapsed().as_secs_f64() * 1e6 / 1000.0;
    println!(
        "launch probe n={n} m={m}: mean launch+sync = {launch_us:.2} us, mean full gemv_q4k call = {call_us:.2} us"
    );

    if !all_ok {
        std::process::exit(1);
    }
    println!("PASSED: bloomery-gpu gemv_q4k within 1e-2 of y_ref; P0 eager == graph replay");
    Ok(())
}

#[cfg(feature = "gpu")]
fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert!(b.len() % 4 == 0, "odd size for {path}");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
