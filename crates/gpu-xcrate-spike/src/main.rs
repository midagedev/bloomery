//! gpu-xcrate-spike — one process, two device bundles: `bloomery_gpu`'s and
//! this package's. The spike's `#[cuda_module]` holds copies of two
//! `bloomery_gpu` entries whose bodies are reached across the crate
//! boundary; this binary runs both sides on the same synthetic Q3_K rows and
//! the same synthetic activation and compares the outputs bit for bit.
//!
//! The input is synthetic on purpose: the question is whether a core
//! compiled from another crate produces the same instructions, not whether
//! Q3_K is right (the reference gates own that). Weight bytes are an LCG
//! draw with a finite f16 super-block scale, so every value is ordinary.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gpu-xcrate-spike: built without the `gpu` feature; the GPU path is off.");
    eprintln!(
        "gpu build: cargo oxide build --arch sm_86 -- -p bloomery-gpu-xcrate-spike --features gpu"
    );
}

#[cfg(feature = "gpu")]
mod device;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use cuda_core::DeviceBuffer;
    use device::{SpikeAct, XcrateKernels};

    const N_SB: usize = 8; // K = 2048
    const ROWS: usize = 64;
    const WORDS_PER_ROW: usize = 110 * N_SB / 4;

    let w_words = synth_q3k_rows(ROWS, N_SB);
    assert_eq!(w_words.len(), ROWS * WORDS_PER_ROW);
    let x = synth_activation(256 * N_SB);

    let gpu = Gpu::new()?;
    let stream = gpu.stream();

    let w_dev = DeviceTensor::upload(stream, &w_words, ROWS, WORDS_PER_ROW)?;
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let sel = DeviceBuffer::from_host(stream, &[0u32])?;

    // Side A: the in-crate entries, through the public engine API.
    let mut act = Q8Act::with_k(stream, 1, 256 * N_SB)?;
    let mut y_ref = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    gpu.enqueue_gemv_q3k_sel(&w_dev, &act, &sel, 1, ROWS, &mut y_ref)?;
    stream.synchronize()?;
    let y_ref = y_ref.to_host_vec(stream)?;

    // Side B: this package's copies, against scratch this package owns.
    // Loading happens into the context `bloomery_gpu` already loaded its own
    // bundles into — the second `.oxart` of the process.
    let xk = XcrateKernels::load(gpu.context())?;
    let mut spike_act = SpikeAct::new(stream, 1, N_SB)?;
    let mut y_spike = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;
    xk.enqueue_quantize(stream, &x_dev, &mut spike_act)?;
    xk.enqueue_gemv_sel(stream, w_dev.buf(), &spike_act, &sel, ROWS, &mut y_spike)?;
    stream.synchronize()?;
    let y_spike = y_spike.to_host_vec(stream)?;

    let mismatch = y_ref
        .iter()
        .zip(y_spike.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let finite = y_ref.iter().all(|v| v.is_finite()) && y_ref.iter().any(|&v| v != 0.0);
    println!(
        "xcrate spike rows={ROWS} n_sb={N_SB}: in_crate[0]={:?} spike[0]={:?} outputs_finite_and_nonzero={finite}",
        y_ref[0], y_spike[0]
    );
    match mismatch {
        None => {
            println!(
                "PASSED: {ROWS} rows bit-identical between the in-crate entries and this package's copies (two .oxart bundles, one process)"
            );
            if !finite {
                eprintln!(
                    "FAIL: outputs are all zero or non-finite — the comparison proves nothing"
                );
                std::process::exit(1);
            }
            Ok(())
        }
        Some(i) => {
            eprintln!(
                "FAIL: row {i} differs: in-crate {:?} (bits {:#010x}) vs spike {:?} (bits {:#010x})",
                y_ref[i],
                y_ref[i].to_bits(),
                y_spike[i],
                y_spike[i].to_bits()
            );
            std::process::exit(1);
        }
    }
}

/// `rows` Q3_K rows of `n_sb` super-blocks as the gemv reads them: 110 bytes
/// per super-block, the row stream as u32 words. Super-block layout is
/// hmask[32] @ +0, qs[64] @ +32, scales[12] @ +96, d (f16) @ +108; the
/// quantized fields are an LCG draw and d is a small normal f16, so every
/// decoded weight is finite.
#[cfg(feature = "gpu")]
fn synth_q3k_rows(rows: usize, n_sb: usize) -> Vec<u32> {
    let mut bytes = vec![0u8; rows * 110 * n_sb];
    let mut s = 0x1234_5678u32;
    let mut next = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 16) as u8
    };
    for sb in 0..rows * n_sb {
        let base = sb * 110;
        for b in bytes[base..base + 108].iter_mut() {
            *b = next();
        }
        // f16 2^-7 nudged per super-block: exponent field 8, low mantissa
        // bits varying — normal, positive, and well inside f32.
        let d = 0x2000u16 | (sb as u16 & 0x00ff);
        bytes[base + 108..base + 110].copy_from_slice(&d.to_le_bytes());
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// `k` activation values in [-1, 1) from the same LCG.
#[cfg(feature = "gpu")]
fn synth_activation(k: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9u32;
    (0..k)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}
