//! gpu-spike — a binary in its own package calling the bloomery-gpu library.
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
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    let gpu = bloomery_gpu::Gpu::new()?;

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

    // Launch-cost probe on a tiny shape (one 256-thread block): bare
    // kernel launch+sync from a library call, then the full gemv_q4k call
    // (which adds uploads, quantize, allocations and the copy-back per
    // call). Design figures, not benchmarks of record.
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
    println!("PASSED: bloomery-gpu gemv_q4k within 1e-2 of y_ref (q8_1 activation design)");
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
