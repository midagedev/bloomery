//! GPU kernel gate for package P1 (docs/gpu-design.md work package): the
//! K-quant gemvs and the shared q8_1 quantizer generalized from K=2048 to K
//! as a launch argument.
//!
//! Three assertions per shape (type × K × rows × m, on real model rows):
//! `max_rel_err <= KERNEL_BAND` against a reference dot computed on the
//! SAME q8_1-quantized activations the kernel consumes (the gate measures
//! the gemv arithmetic; the activation-quantization noise the exact
//! reference would add measures ~1.5e-2 on the verbatim port and is printed
//! ungated as `exact_ref_err`), a bit-identical rerun, and — for K=2048 —
//! an FNV-1a 64 hash of the output bytes pinned to the verbatim stage-0
//! port, so generalization and the cores extraction are proven to change
//! no K=2048 bit.
//!
//! And the quantizers' refusal (`bloomery_gpu::fault`): a column holding a
//! non-finite value has no q8 form, so `q3k_quantize_q8_1`, `norm_quant` and
//! the Q5 path's 32-value quantizer each raise the fault word at their own
//! site instead of rounding the value to code 0 — planted NaN and inf lanes,
//! the word read back and cleared per case; the pinned shapes leave it clean.
//! The two q8_1 quantizers store a refused block with a NaN scale, so the
//! q3_K gemv over the planted column is NaN in every row.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p1: built without the `gpu` feature; see `just gate-gpu-p1`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu_gates::GateError;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p1", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_gates::{
        Fnv1a64, KERNEL_BAND, activations, bytes_to_words, max_rel_err, open_model, ref_gemv,
        row_bytes, tensor_bytes_as,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::GgmlType;

    // Fixed activation seed for every shape: the pinned hashes cover the
    // whole pipeline, activations included, so the draw must be fixed.
    const SEED: u32 = 1;

    // PIN(2026-09-21): K=2048 output bits of the verbatim stage-0 port,
    // activations(k, m, SEED) on the exact tensors and row spans below
    // ((m=1, m=8) per type). Generalization must reproduce every bit.
    const PIN_Q3K_K2048: (u64, u64) = (0x2404_4dc6_74fe_c873, 0x54ad_6fdb_a9c2_1a63);
    const PIN_Q4K_K2048: (u64, u64) = (0xe4e9_931f_39ac_4346, 0xa47c_3723_65b0_08c3);
    const PIN_Q6K_K2048: (u64, u64) = (0xf488_ed75_9aff_b869, 0xa4b6_6ba0_05f5_012d);
    // PIN(2026-09-21): the shapes generalization opened (K=512, the K=2816
    // partial group, a row count off the 8-lane grid), taken from the lead's
    // rerun of the landed P1 tree (cbf46c3) — equal to the delegate's run.
    // These guard the paths against later fusion work; they prove nothing
    // about stage 0, which never ran these shapes.
    const PIN_Q3K_K512: (u64, u64) = (0xf378_f732_ab02_4bfa, 0xa32a_f561_6481_a47e);
    const PIN_Q4K_K2816: (u64, u64) = (0x4729_e493_2486_9646, 0x1779_5da0_e13a_3350);
    const PIN_Q3K_R1407: (u64, u64) = (0xfe53_dbd4_e64b_d903, 0xe9bb_5923_c457_b5ff);

    // (name, tensor, type, K, row cap: None = all rows, (m=1, m=8) pin pair)
    let shapes: &[(
        &str,
        &str,
        GgmlType,
        usize,
        Option<usize>,
        Option<(u64, u64)>,
    )] = &[
        (
            "q3k_k2048",
            "blk.1.attn_q.weight",
            GgmlType::Q3_K,
            2048,
            Some(2048),
            Some(PIN_Q3K_K2048),
        ),
        (
            "q4k_k2048",
            "blk.0.attn_output.weight",
            GgmlType::Q4_K,
            2048,
            None,
            Some(PIN_Q4K_K2048),
        ),
        (
            "q6k_k2048",
            "output.weight",
            GgmlType::Q6_K,
            2048,
            Some(4096),
            Some(PIN_Q6K_K2048),
        ),
        // The shapes the stage-0 constants could not express: K=512
        // (attn_kv_b), K=2816 with 11 super-blocks per row (shexp down),
        // and a row count that is not a multiple of 8.
        (
            "q3k_k512",
            "blk.1.attn_kv_b.weight",
            GgmlType::Q3_K,
            512,
            None,
            Some(PIN_Q3K_K512),
        ),
        (
            "q4k_k2816",
            "blk.1.ffn_down_shexp.weight",
            GgmlType::Q4_K,
            2816,
            None,
            Some(PIN_Q4K_K2816),
        ),
        (
            "q3k_k2048_r1407",
            "blk.1.ffn_gate_shexp.weight",
            GgmlType::Q3_K,
            2048,
            Some(1407),
            Some(PIN_Q3K_R1407),
        ),
    ];

    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let mut all_ok = true;

    for &(name, tensor, ty, k, cap, pins) in shapes {
        let (info, bytes) = tensor_bytes_as(&gguf, tensor, ty, None)?;
        if info.dims[0] != k as u64 {
            return Err(format!("{tensor} is {:?}, want K = {k} at dims[0]", info.dims).into());
        }
        let rows_total: usize = info.dims[1..].iter().product::<u64>() as usize;
        let rows = cap.map_or(rows_total, |c| rows_total.min(c));
        assert!(rows >= 1, "{tensor}: no rows");
        let rb = row_bytes(ty, k)?;
        assert!(
            bytes.len() >= rb * rows,
            "{tensor}: file too small for {rows} rows"
        );
        let words = bytes_to_words(&bytes[..rb * rows]);
        assert_eq!(words.len() % rows, 0, "{tensor}: words not row-divisible");
        let w = DeviceTensor::upload(stream, &words, rows, words.len() / rows)?;

        for m in [1usize, 8] {
            let x = activations(k, m, SEED);
            let x_dev = DeviceBuffer::from_host(stream, &x)?;
            let mut act = Q8Act::with_k(stream, m, k)?;
            let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
            let mut run = |y: &mut DeviceBuffer<f32>| -> Result<(), GateError> {
                gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
                match ty {
                    GgmlType::Q3_K => gpu.enqueue_gemv_q3k(&w, &act, y)?,
                    GgmlType::Q4_K => gpu.enqueue_gemv_q4k(&w, &act, y)?,
                    GgmlType::Q6_K => gpu.enqueue_gemv_q6k(&w, &act, y)?,
                    other => return Err(format!("gate_p1: no gemv for {other:?}").into()),
                }
                stream.synchronize()?;
                Ok(())
            };
            run(&mut y_dev)?;
            let y1 = y_dev.to_host_vec(stream)?;
            run(&mut y_dev)?;
            let y2 = y_dev.to_host_vec(stream)?;
            let bit_same = y1
                .iter()
                .zip(y2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            // Gated: the reference dot on the same q8_1-quantized
            // activations the kernel consumes.
            let xq = q8_1_dequant(&x, k, m);
            let y_ref = ref_gemv(ty, &bytes[..rb * rows], k, rows, &xq, m)?;
            let rel = max_rel_err(&y1, &y_ref)?;
            // Ungated information: the error against the exact activations,
            // i.e. the q8_1 activation-quantization noise of this design.
            let y_exact = ref_gemv(ty, &bytes[..rb * rows], k, rows, &x, m)?;
            let exact_ref_err = max_rel_err(&y1, &y_exact)?;
            let hash = Fnv1a64::default().f32s(&y1).value();
            println!(
                "shape {name:<16} T={ty:?} K={k} rows={rows} m={m} max_rel_err={rel:.3e} exact_ref_err={exact_ref_err:.3e} bit_identical_rerun={bit_same} fnv1a64={hash:#018x}"
            );
            if rel > KERNEL_BAND {
                eprintln!("FAIL: {name} m={m} rel err {rel:.3e} exceeds {KERNEL_BAND}");
                all_ok = false;
            }
            if !bit_same {
                eprintln!("FAIL: {name} m={m} rerun is not bit-identical");
                all_ok = false;
            }
            if let Some(pins) = pins {
                let pin = if m == 1 { pins.0 } else { pins.1 };
                if hash != pin {
                    eprintln!(
                        "FAIL: {name} m={m} fnv1a64 {hash:#018x} != pinned {pin:#018x} \
                         (pinned output bits changed)"
                    );
                    all_ok = false;
                }
            }
        }
    }

    all_ok &= quant_faults(&gpu, &gguf)?;

    if !all_ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_p1 K-quant gemvs within {KERNEL_BAND} of the q8_1 reference, \
         reruns bit-identical, pins matched, every planted non-finite lane raised its \
         quantizer's fault and the gemv over a refused q8_1 column is NaN"
    );
    Ok(())
}

/// The quantizers' fault contract on planted input. The pinned shapes above
/// ran on finite activations, so the word must still be clean; then per case
/// the planted column goes through one quantizer, and the word read back
/// (and cleared) must name that quantizer's site with no layer. For the two
/// q8_1 quantizers the refused block is a contract too: the q3_K gemv of the
/// planted column must be NaN in every row, where a block rounded to code 0
/// would give the gemv of the same column with the planted lanes set to 0.0
/// and nothing downstream could tell.
#[cfg(feature = "gpu")]
fn quant_faults(gpu: &bloomery_gpu::Gpu, gguf: &gguf::Gguf) -> Result<bool, GateError> {
    use bloomery_gpu::fused::FusedKernels;
    use bloomery_gpu::q5::{Q5Kernels, Q8Blocks32};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, LAYER_NONE, Q8Act};
    use bloomery_gpu_gates::{
        activations, bits_equal, bytes_to_words, row_bytes, tensor_bytes_as, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::GgmlType;

    const K: usize = 2048;
    const ROWS: usize = 256;
    const LANE: usize = 37;
    let stream = gpu.stream();
    let clean = gpu.fault()?;
    println!(
        "fault case=after_shapes want=none got={} {}",
        clean.map_or("none".to_string(), |f| f.to_string()),
        verdict(clean.is_none())
    );
    let mut ok = clean.is_none();
    gpu.clear_fault()?;

    let (_, bytes) = tensor_bytes_as(gguf, "blk.1.attn_q.weight", GgmlType::Q3_K, None)?;
    let rb = row_bytes(GgmlType::Q3_K, K)?;
    let words = bytes_to_words(&bytes[..rb * ROWS]);
    let w = DeviceTensor::upload(stream, &words, ROWS, words.len() / ROWS)?;
    let gemv = |x: &[f32], act: &mut Q8Act| -> Result<Vec<f32>, GateError> {
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;
        gpu.enqueue_quantize_q8_1(&x_dev, act)?;
        gpu.enqueue_gemv_q3k(&w, act, &mut y)?;
        Ok(y.to_host_vec(stream)?)
    };
    let want = |site: FaultSite| Fault::at(LAYER_NONE, site);
    let base = activations(K, 1, 1);
    let planted = |lanes: std::ops::Range<usize>, v: f32| {
        let mut x = base.clone();
        let mut z = base.clone();
        for i in lanes {
            x[i] = v;
            z[i] = 0.0;
        }
        (x, z)
    };
    let mut act = Q8Act::with_k(stream, 1, K)?;
    for (name, (x, z)) in [
        ("nan_lane", planted(LANE..LANE + 1, f32::NAN)),
        ("inf_lane", planted(LANE..LANE + 1, f32::INFINITY)),
        ("nan_block", planted(0..128, f32::NAN)),
    ] {
        let y_bad = gemv(&x, &mut act)?;
        let got = gpu.take_fault()?;
        let y_zero = gemv(&z, &mut act)?;
        let after = gpu.take_fault()?;
        let gemv_nan = y_bad.iter().all(|v| v.is_nan());
        let pass = got == Some(want(FaultSite::QuantColumn)) && after.is_none() && gemv_nan;
        println!(
            "fault case={name} entry=q3k_quantize_q8_1 want=quant_column got={} zero_lane_clean={} \
             gemv_nan={gemv_nan} silent_zero={} {}",
            got.map_or("none".to_string(), |f| f.to_string()),
            after.is_none(),
            bits_equal(&y_bad, &y_zero),
            verdict(pass)
        );
        ok &= pass;
    }

    // norm_quant: one NaN lane makes the column's sum of squares NaN, so
    // every normalized value is NaN and every block of the column is
    // refused — the gemv over it is NaN, not the zero a column of code 0
    // would give.
    let fused = FusedKernels::load(gpu.context())?;
    let gain = DeviceBuffer::from_host(stream, &vec![1.0f32; K])?;
    let (x, _) = planted(LANE..LANE + 1, f32::NAN);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut normed = DeviceBuffer::<f32>::zeroed(stream, K)?;
    let mut y = DeviceBuffer::<f32>::zeroed(stream, ROWS)?;
    fused.enqueue_norm_quant(
        stream,
        &x_dev,
        &gain,
        1e-6,
        &mut act,
        &mut normed,
        gpu.unlabelled_sink(),
    )?;
    gpu.enqueue_gemv_q3k(&w, &act, &mut y)?;
    let y = y.to_host_vec(stream)?;
    let got = gpu.take_fault()?;
    let gemv_nan = y.iter().all(|v| v.is_nan());
    let pass = got == Some(want(FaultSite::NormQuant)) && gemv_nan;
    println!(
        "fault case=nan_lane entry=norm_quant want=norm_quant got={} gemv_nan={gemv_nan} {}",
        got.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    ok &= pass;

    // The Q5 path's 32-value quantizer, the same planted lane.
    let q5 = Q5Kernels::load(gpu.context(), gpu.fault_word())?;
    let mut act32 = Q8Blocks32::new(stream, K, 1)?;
    q5.enqueue_quantize_q8(stream, &x_dev, &mut act32, gpu.unlabelled_sink())?;
    let got = gpu.take_fault()?;
    let pass = got == Some(want(FaultSite::Q5Quant));
    println!(
        "fault case=nan_lane entry=q5_quantize_q8 want=q5_quant got={} {}",
        got.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    ok &= pass;
    Ok(ok)
}

/// Host transcription of the device quantizer's activation treatment: per
/// 128-value block, d = amax/127 and q = round(x/d) clamped to ±127; the
/// reference dot runs on the reconstructed q·d values. The kernel keeps q
/// as the integer the dp4a chains consume and factors d in once, so the
/// two differ only by float rounding of that factoring — real gemv defects
/// stay far above KERNEL_BAND, while the shared activation-quantization
/// noise (the `exact_ref_err` column) cancels.
#[cfg(feature = "gpu")]
fn q8_1_dequant(x: &[f32], k: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k * m];
    for c in 0..m {
        for b in 0..k / 128 {
            let blk = &x[c * k + b * 128..c * k + b * 128 + 128];
            let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (i, &v) in blk.iter().enumerate() {
                let q = (v / d).round().clamp(-127.0, 127.0);
                out[c * k + b * 128 + i] = q * d;
            }
        }
    }
    out
}
