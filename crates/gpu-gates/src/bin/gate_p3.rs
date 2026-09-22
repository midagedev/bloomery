//! GPU kernel gate for package P3 (docs/gpu-design.md work package): the F32
//! router gemv and the derived-Q8_0 gemv, f32 activations, against an f64
//! reference over dequantized rows. The reference side is
//! `bloomery_gpu_gates`; the band is 1e-5 — far tighter than
//! `KERNEL_BAND`, because these sites feed the router's top-6 (a flipped
//! near-tie changes the token) and the kernels accumulate in f32 against
//! at most 2048 terms, which leaves orders of margin. If a shape measures
//! above the band the gate FAILs and the figure is reported, never absorbed.
//!
//! The Q8_0 weights are quantized here, not read: the shared `dequant_row`
//! has no Q8_0 arm (`GgmlType` models the file's storage types; the engine's
//! derived Q8_0 lives in `model` as `Q8Block`), so the ggml quantize rule
//! the engine's load-time requant runs and its dequantizer are transcribed
//! below, verbatim from `model::attn`.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p3: built without the `gpu` feature; see `just gate-gpu-p3`.");
    std::process::exit(2);
}

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
    let (t, w_bytes) = bloomery_gpu_gates::tensor_bytes(&gguf, "blk.1.ffn_gate_inp.weight")?;
    if t.ty != GgmlType::F32 || t.dims.len() != 2 || t.dims[0] != 2048 || t.dims[1] != 64 {
        return Err(format!(
            "router tensor is {:?} {:?}, want F32 [2048, 64]",
            t.ty, t.dims
        )
        .into());
    }
    let w_router: Vec<f32> = w_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w_dev = DeviceTensor::upload(stream, &w_router, 64, 2048)?;
    for m in [1usize, 8] {
        let x = bloomery_gpu_gates::activations(2048, m, 3);
        let y_ref = bloomery_gpu_gates::ref_gemv(GgmlType::F32, w_bytes, 2048, 64, &x, m)?;
        let (rel, bit_same) = run_f32(&kernels, stream, &w_dev, 64, m, &x, &y_ref)?;
        report("f32", 2048, 64, m, rel, bit_same, BAND, &mut all_ok);
    }

    // F32, synthetic shape (K=512, rows=1000): weights from the shared LCG.
    let w_synth = bloomery_gpu_gates::activations(512 * 1000, 1, 5);
    let w_synth_dev = DeviceTensor::upload(stream, &w_synth, 1000, 512)?;
    for m in [1usize, 8] {
        let x = bloomery_gpu_gates::activations(512, m, 11);
        let y_ref = ref_f64_dot(&w_synth, 512, &x, m);
        let (rel, bit_same) = run_f32(&kernels, stream, &w_synth_dev, 1000, m, &x, &y_ref)?;
        report("f32", 512, 1000, m, rel, bit_same, BAND, &mut all_ok);
    }

    // Q8_0: weights quantized here from LCG rows, reference = the
    // dequantized rows dotted in f64.
    for (k, rows, w_seed, x_seed) in [(128usize, 3072usize, 7u32, 13u32), (2048, 64, 9, 15)] {
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
            let (rel, bit_same) = run_q8(&kernels, stream, &qs_dev, &d_dev, rows, m, &x, &y_ref)?;
            report("q8_0", k, rows, m, rel, bit_same, BAND, &mut all_ok);
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
        y_dev.zero_async(stream)?;
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
        if !identical || graph.node_count() != 1 {
            eprintln!(
                "FAIL: graph gate K={k} rows={rows}: identical={identical} nodes={}",
                graph.node_count()
            );
            all_ok = false;
        }
    }

    if !all_ok {
        std::process::exit(1);
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
    d: &bloomery_gpu::DeviceTensor<f32>,
    rows: usize,
    m: usize,
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

/// One shape line and its verdict.
#[cfg(feature = "gpu")]
fn report(
    ty: &str,
    k: usize,
    rows: usize,
    m: usize,
    rel: f32,
    bit_same: bool,
    band: f32,
    all_ok: &mut bool,
) {
    println!(
        "shape type={ty} K={k} rows={rows} m={m} max_rel_err={rel:.3e} bit_identical_rerun={bit_same}"
    );
    if rel > band || !bit_same {
        eprintln!(
            "FAIL: {ty} K={k} rows={rows} m={m}: rel {rel:.3e} (band {band:.0e}), bit_identical_rerun={bit_same}"
        );
        *all_ok = false;
    }
}

// ------------------------------------------------- local Q8_0 quantizer

/// `f32_to_f16_bits` — IEEE round-to-nearest-even, transcribed from
/// `model::attn` (the engine's load-time requant stores this rounding).
#[cfg(feature = "gpu")]
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return sign | 0x7c00;
    }
    let exp = ((a >> 23) as i32) - 127;
    let frac = a & 0x007f_ffff;
    if exp > 15 {
        // Past f16's finite range; the LCG weights never reach it.
        return sign | 0x7c00;
    }
    if exp >= -14 {
        // Normal f16: keep 11 significand bits, ties-to-even carry.
        let v = (((exp + 15) as u32) << 23) | frac;
        let t = v + 0x0fff + ((v >> 13) & 1);
        let h = t >> 13;
        if h & 0x7c00 == 0x7c00 {
            return sign | 0x7c00;
        }
        sign | h as u16
    } else {
        // Subnormal f16.
        if exp < -25 {
            return sign;
        }
        let shift = (-1 - exp) as u32;
        let v = 0x0080_0000 | frac;
        let half = 1u32 << (shift - 1);
        let rem = v & ((1 << shift) - 1);
        let mut h = v >> shift;
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        sign | h as u16
    }
}

/// IEEE half to f32, integer-only — every f16 value is exact in f32, so any
/// correct conversion is bit-identical to the engine's.
#[cfg(feature = "gpu")]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let mag = if exp == 0x1f {
        0x7f800000 | (mant << 13)
    } else if exp == 0 {
        if mant == 0 {
            0
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (e << 23) | ((m & 0x3ff) << 13)
        }
    } else {
        ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(sign | mag)
}

/// `nearest_int` — the f32 magic-round the engine's quantizers use
/// (`qdot::nearest_int`, ggml's `nearest_int`).
#[cfg(feature = "gpu")]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = f32::to_bits(val);
    ((i & 0x007f_ffff) as i32) - 0x0040_0000
}

/// ggml `quantize_row_q8_0` x86 branch, as the engine's load-time requant
/// runs it: `d = amax/127` stored as f16, `id = 127/amax` (a different f32
/// than `1/d`), codes `nearest_int(v*id)` clamped to [-128, 127]. Appends
/// one row's device layout (k/4 u32 words, k/32 f32 scales) and the
/// dequantized reference row (k values of `q as f32 * scale` — the exact
/// bits the kernel reconstructs).
#[cfg(feature = "gpu")]
fn quant_q8_0_row(x: &[f32], qs: &mut Vec<u32>, d: &mut Vec<f32>, deq: &mut Vec<f32>) {
    for blk in x.chunks_exact(32) {
        let mut amax = 0.0f32;
        for &v in blk {
            amax = amax.max(v.abs());
        }
        let scale = half_to_f32(f32_to_f16_bits(amax / 127.0));
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        let mut word = [0u32; 8];
        for (j, &v) in blk.iter().enumerate() {
            let q = nearest_int(v * id).clamp(-128, 127) as i8;
            word[j / 4] |= u32::from(q as u8) << (8 * (j % 4));
            deq.push(q as f32 * scale);
        }
        qs.extend_from_slice(&word);
        d.push(scale);
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
