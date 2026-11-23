//! GPU kernel gate for package P3 (docs/gpu-design.md work package): the F32
//! router gemv and the derived-Q8_0 gemv, f32 activations, against an f64
//! reference over dequantized rows. The reference side is
//! `bloomery_gpu_gates`; the band is 1e-5 — far tighter than
//! `KERNEL_BAND`, because these sites feed the router's top-6 (a flipped
//! near-tie changes the token) and the kernels accumulate in f32 against
//! at most 2048 terms, which leaves orders of margin. If a shape measures
//! above the band the gate FAILs and the figure is reported, never absorbed.
//!
//! The Q8_0 weights are quantized here, not read: the engine's Q8_0 on this
//! model is derived at load (`gguf::quant::Q8Block`), not a file tensor, so the
//! shared quantize rule in `gguf::quant` supplies the blocks, and each
//! reference value is `code × scale` as the block is written — the product
//! `dequant_row`'s Q8_0 arm computes, exact in f32.

#[cfg(feature = "gpu")]
use bloomery_gpu::DeviceBufferRaw;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::GateError;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p3", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    use bloomery_gpu::q8f32::Q8F32Kernels;
    use bloomery_gpu::{DeviceTensor, Gpu};
    use cuda_core::DeviceBuffer;
    use gguf::quant::GgmlType;

    // f32 accumulation against an f64 reference; the band for both kernels,
    // fixed before the first run.
    const BAND: f32 = 1e-5;

    let gguf = bloomery_gpu_gates::open_model()?;
    let gpu = Gpu::new()?;
    let kernels = Q8F32Kernels::load(gpu.context())?;
    let stream = gpu.stream();
    let mut all_ok = true;

    // F32, real router weight: blk.1.ffn_gate_inp.weight, F32 [2048, 64].
    let (_, w_bytes) = bloomery_gpu_gates::tensor_bytes_as(
        &gguf,
        "blk.1.ffn_gate_inp.weight",
        GgmlType::F32,
        Some(&[2048, 64]),
    )?;
    let w_router: Vec<f32> = w_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let w_dev = DeviceTensor::upload(stream, &w_router, 64, 2048)?;
    for m in [1usize, 8] {
        let x = bloomery_gpu_gates::activations(2048, m, 3);
        let y_ref = bloomery_gpu_gates::ref_gemv(GgmlType::F32, w_bytes, 2048, 64, &x, m)?;
        let (rel, bit_same) = run_f32(&kernels, stream, &w_dev, 64, m, &x, &y_ref)?;
        all_ok &= report("f32", 2048, 64, m, rel, bit_same, BAND);
    }

    // F32, synthetic shapes: weights from the shared LCG. K = 512 is whole
    // hoisted trips (`LANE_UNROLL` chunks each); 480 and 992 (15 and 31
    // chunks) end in the single-column body's one-chunk tail, and their row
    // counts leave the last block short.
    for (k, rows, w_seed, x_seed) in [
        (512usize, 1000usize, 5u32, 11u32),
        (480, 100, 25, 27),
        (992, 100, 29, 31),
    ] {
        let w_synth = bloomery_gpu_gates::activations(k * rows, 1, w_seed);
        let w_synth_dev = DeviceTensor::upload(stream, &w_synth, rows, k)?;
        for m in [1usize, 8] {
            let x = bloomery_gpu_gates::activations(k, m, x_seed);
            let y_ref = ref_f64_dot(&w_synth, k, &x, m);
            let (rel, bit_same) = run_f32(&kernels, stream, &w_synth_dev, rows, m, &x, &y_ref)?;
            all_ok &= report("f32", k, rows, m, rel, bit_same, BAND);
        }
    }

    // Q8_0: weights quantized here from LCG rows, reference = the
    // dequantized rows dotted in f64. At m = 1 each K takes its own path
    // through the single-column body: 128 the one-word row, 2048 whole
    // trips of four 128-value steps, 480 (three steps and 96 values) a pair,
    // a single and the partial word, 992 (seven steps and 96 values) a trip,
    // a pair, a single and the partial word in one row.
    for (k, rows, w_seed, x_seed) in [
        (128usize, 3072usize, 7u32, 13u32),
        (2048, 64, 9, 15),
        (480, 100, 17, 19),
        (992, 100, 21, 23),
    ] {
        let src = bloomery_gpu_gates::activations(k * rows, 1, w_seed);
        let mut qs_words = Vec::with_capacity(rows * k / 4);
        let mut d_scales = Vec::with_capacity(rows * k / 32);
        let mut deq = Vec::with_capacity(rows * k);
        for row in src.chunks_exact(k) {
            quant_q8_0_row(row, &mut qs_words, &mut d_scales, &mut deq);
        }
        let qs_dev = DeviceTensor::upload(stream, &qs_words, rows, k / 4)?;
        let d_dev = DeviceTensor::upload(stream, &d_scales, rows, k / 32)?;
        for m in [1usize, 8] {
            let x = bloomery_gpu_gates::activations(k, m, x_seed);
            let y_ref = ref_f64_dot(&deq, k, &x, m);
            let (rel, bit_same) = run_q8(&kernels, stream, &qs_dev, &d_dev, (rows, m), &x, &y_ref)?;
            all_ok &= report("q8_0", k, rows, m, rel, bit_same, BAND);
        }

        // Eager vs captured-graph byte identity, once per Q8_0 shape: the
        // enqueue path is allocation-free, so the same launch must capture
        // and must reproduce the eager bytes bit for bit.
        let x = bloomery_gpu_gates::activations(k, 8, x_seed);
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * 8)?;
        kernels.enqueue_q8_0_gemv(stream, &qs_dev, &d_dev, &x_dev, 8, &mut y_dev)?;
        stream.synchronize()?;
        let y_eager = y_dev.to_host_vec(stream)?;
        y_dev.zero_raw(stream)?;
        stream.synchronize()?;
        let graph =
            gpu.capture(|s| kernels.enqueue_q8_0_gemv(s, &qs_dev, &d_dev, &x_dev, 8, &mut y_dev))?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let y_graph = y_dev.to_host_vec(stream)?;
        let identical = y_eager.len() == y_graph.len()
            && y_eager
                .iter()
                .zip(y_graph.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "graph gate: type=q8_0 K={k} rows={rows} m=8 eager_vs_graph_bit_identical={identical} nodes={}",
            graph.node_count()
        );
        let pass = identical && graph.node_count() == 1;
        if !pass {
            eprintln!(
                "FAIL: graph gate K={k} rows={rows}: identical={identical} nodes={}",
                graph.node_count()
            );
        }
        all_ok &= pass;
    }

    if !all_ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!("PASSED: f32 and q8_0 gemv within 1e-5 of the f64 reference; eager == graph replay");
    Ok(())
}

/// Two enqueues of the F32 gemv over resident buffers: (max_rel_err of the
/// first run against `y_ref`, bit-identity of the rerun).
#[cfg(feature = "gpu")]
fn run_f32(
    kernels: &bloomery_gpu::q8f32::Q8F32Kernels,
    stream: &cuda_core::CudaStream,
    w: &bloomery_gpu::DeviceTensor<f32>,
    rows: usize,
    m: usize,
    x: &[f32],
    y_ref: &[f32],
) -> Result<(f32, bool), GateError> {
    use cuda_core::DeviceBuffer;
    let x_dev = DeviceBuffer::from_host(stream, x)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    kernels.enqueue_f32_gemv(stream, w, &x_dev, m, &mut y_dev)?;
    stream.synchronize()?;
    let y1 = y_dev.to_host_vec(stream)?;
    kernels.enqueue_f32_gemv(stream, w, &x_dev, m, &mut y_dev)?;
    stream.synchronize()?;
    let y2 = y_dev.to_host_vec(stream)?;
    let bit_same = y1.iter().zip(&y2).all(|(a, b)| a.to_bits() == b.to_bits());
    Ok((bloomery_gpu_gates::max_rel_err(&y1, y_ref)?, bit_same))
}

/// Two enqueues of the Q8_0 gemv over resident buffers; returns as `run_f32`.
#[cfg(feature = "gpu")]
fn run_q8(
    kernels: &bloomery_gpu::q8f32::Q8F32Kernels,
    stream: &cuda_core::CudaStream,
    qs: &bloomery_gpu::DeviceTensor<u32>,
    d: &bloomery_gpu::DeviceTensor<u16>,
    (rows, m): (usize, usize),
    x: &[f32],
    y_ref: &[f32],
) -> Result<(f32, bool), GateError> {
    use cuda_core::DeviceBuffer;
    let x_dev = DeviceBuffer::from_host(stream, x)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
    kernels.enqueue_q8_0_gemv(stream, qs, d, &x_dev, m, &mut y_dev)?;
    stream.synchronize()?;
    let y1 = y_dev.to_host_vec(stream)?;
    kernels.enqueue_q8_0_gemv(stream, qs, d, &x_dev, m, &mut y_dev)?;
    stream.synchronize()?;
    let y2 = y_dev.to_host_vec(stream)?;
    let bit_same = y1.iter().zip(&y2).all(|(a, b)| a.to_bits() == b.to_bits());
    Ok((bloomery_gpu_gates::max_rel_err(&y1, y_ref)?, bit_same))
}

/// One shape line; returns its verdict (a FAIL also goes to stderr).
#[cfg(feature = "gpu")]
fn report(ty: &str, k: usize, rows: usize, m: usize, rel: f32, bit_same: bool, band: f32) -> bool {
    println!(
        "shape type={ty} K={k} rows={rows} m={m} max_rel_err={rel:.3e} bit_identical_rerun={bit_same}"
    );
    let fail = rel > band || !bit_same;
    if fail {
        eprintln!(
            "FAIL: {ty} K={k} rows={rows} m={m}: rel {rel:.3e} (band {band:.0e}), bit_identical_rerun={bit_same}"
        );
    }
    !fail
}

/// Packs the shared Q8_0 rule into device words and a dequantized reference row.
#[cfg(feature = "gpu")]
fn quant_q8_0_row(x: &[f32], qs: &mut Vec<u32>, d: &mut Vec<u16>, deq: &mut Vec<f32>) {
    assert!(x.len().is_multiple_of(32), "Q8_0 row needs whole blocks");
    let blocks: Vec<_> = x.chunks(32).map(gguf::quant::quantize_q8_0).collect();
    let (words, scales) = bloomery_gpu::weights::q8_0_planes(&blocks);
    qs.extend(words);
    d.extend(scales);
    deq.extend(blocks.iter().flat_map(|b| {
        let scale = gguf::quant::half_to_f32(b.d);
        b.q.map(move |q| f32::from(q) * scale)
    }));
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    #[test]
    fn q8_0_rows_refuse_non_finite_blocks() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = [1.0; 32];
            x[17] = bad;
            for x in [x, [bad; 32]] {
                assert!(
                    std::panic::catch_unwind(|| {
                        super::quant_q8_0_row(
                            &x,
                            &mut Vec::new(),
                            &mut Vec::new(),
                            &mut Vec::new(),
                        );
                    })
                    .is_err(),
                    "non-finite Q8_0 input must fail"
                );
            }
        }
        assert!(
            std::panic::catch_unwind(|| {
                super::quant_q8_0_row(
                    &[1.0; 31],
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
            })
            .is_err()
        );
        let (mut q, mut d, mut deq) = (Vec::new(), Vec::new(), Vec::new());
        super::quant_q8_0_row(&[127.0; 32], &mut q, &mut d, &mut deq);
        assert_eq!(q, [0x7f7f7f7f; 8]);
        assert_eq!(d, [0x3c00]);
        assert_eq!(deq, [127.0; 32]);
        let x: [f32; 32] = std::array::from_fn(|i| [100.0, -100.0, 0.0, 50.0][i % 4]);
        super::quant_q8_0_row(&x, &mut q, &mut d, &mut deq);
        assert_eq!(&q[8..], &[0x4000_817f; 8]);
        assert_eq!(d, [0x3c00, 0x3a4d]);
        assert_eq!(&deq[32..36], &[100.0249, -100.0249, 0.0, 50.40625]);
    }
}

/// Reference `y = W · x` over dequantized f32 rows: per output, a
/// sequential f64 dot (`ref_gemv`'s arithmetic, without its quantized-byte
/// front end).
#[cfg(feature = "gpu")]
fn ref_f64_dot(w: &[f32], k: usize, x: &[f32], m: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity(w.len() / k * m);
    for row in w.chunks_exact(k) {
        for c in 0..m {
            let xc = &x[c * k..(c + 1) * k];
            let dot: f64 = row
                .iter()
                .zip(xc)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            y.push(dot as f32);
        }
    }
    y
}
