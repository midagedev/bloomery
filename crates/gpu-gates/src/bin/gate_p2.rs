//! GPU kernel gate for package P2 (docs/gpu-design.md work package): Q5_0 gemv
//! on `blk.1.ffn_down_exps.weight` (the flat 64-expert stack, experts reached
//! by `row0`) and Q5_1 on `blk.0.ffn_down.weight`, each within `KERNEL_BAND`
//! against `ref_gemv` over the same rows. Also pins: bit-identical rerun per
//! shape, `col0`/`y0` addressing with untouched neighbours, and eager vs
//! captured-graph byte identity for one shape.
//!
//! `q5_0_resident_is_compact`: the resident Q5_0 (`pack_q5_0`) holds the
//! file's 22 bytes a 32-value block — a row of `kb` blocks is `22·kb` bytes,
//! two more when `kb` is odd (the scale plane's last half-word) — and the
//! profile's byte count (`model::weight_bytes`), the loaded buffer and the
//! placement figure (`weights::resident_size`) say so; and the gemv over it is
//! bit for bit `q5_0_gemv_mirror`, the host transcription of the int8 dot
//! from the file's bytes and the device quantizer's codes. On expert 5 of the
//! file's stack (`kb` 44) and on synthetic rows of `kb` 33 (odd, so a lane
//! walks two blocks and the scale plane ends on a half), m ∈ {1, 3, 8}.
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
//! On real activations (Q5_0 moe_down) the same
//! deviation measures 1.3–1.4e-2.

#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, bits_equal, verdict};

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p2", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    use bloomery_gpu::q5::{Q5Kernels, Q8Blocks32, pack_q5_0, pack_q5_1, q5_0_row_words};
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
    let w_dev = DeviceTensor::upload(
        stream,
        &pack_q5_0(w_bytes, k0, n_exp * rows0)?,
        n_exp * rows0,
        q5_0_row_words(k0 / 32),
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

    // ---- q5_0_resident_is_compact (module doc).
    let expert = 5usize;
    let real = &w_bytes[expert * rows0 * rb0..][..rows0 * rb0];
    ok &= resident_is_compact(&gpu, &q5, (real, k0, rows0), &[1, 8], "file_expert5")?;
    let (k_odd, rows_odd) = (1056usize, 256usize);
    let synth = synthetic_q5_0(k_odd, rows_odd, 0x5e0d_c0de);
    ok &= resident_is_compact(&gpu, &q5, (&synth, k_odd, rows_odd), &[1, 3], "synthetic")?;

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
        "PASSED: q5_0/q5_1 gemv within KERNEL_BAND of the quantized-input reference; col0/y0 addressing clean; eager == graph replay; resident q5_0 compact and bit for bit the int8 dot"
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

/// One `q5_0_resident_is_compact` shape (module doc): `rows` rows of `k`
/// values of raw Q5_0 `bytes` packed and uploaded as the loader does, their
/// byte counts against `22 · k/32` a row, and the gemv at each `m` of `ms`
/// against [`q5_0_gemv_mirror`] on activations the device quantized. One
/// line a `m`; `false` on any miss.
#[cfg(feature = "gpu")]
fn resident_is_compact(
    gpu: &bloomery_gpu::Gpu,
    q5: &bloomery_gpu::q5::Q5Kernels,
    (bytes, k, rows): (&[u8], usize, usize),
    ms: &[usize],
    what: &str,
) -> Result<bool, GateError> {
    use bloomery_gpu::DeviceTensor;
    use bloomery_gpu::model::weight_bytes;
    use bloomery_gpu::q5::{Q8Blocks32, pack_q5_0};
    use bloomery_gpu::weights::{DevWeight, resident_size};
    use bloomery_gpu_gates::activations;
    use cuda_core::DeviceBuffer;
    use gguf::quant::GgmlType;

    let stream = gpu.stream();
    let kb = k / 32;
    let packed = pack_q5_0(bytes, k, rows)?;
    let file = rows * kb * 22;
    let pad = if kb % 2 == 1 { 2 * rows } else { 0 };
    let packed_bytes = 4 * packed.len();
    let dw = DevWeight::Q5_0 {
        w: DeviceTensor::upload(stream, &packed, rows, packed.len() / rows)?,
        k,
    };
    let profile = weight_bytes(&dw, rows);
    let resident = dw.resident_bytes();
    let placement = resident_size(GgmlType::Q5_0, k, rows);
    let bytes_ok = packed_bytes == file + pad
        && profile == Some(file)
        && resident == packed_bytes
        && placement == Some(resident);
    let DevWeight::Q5_0 { w, .. } = &dw else {
        return Err("resident_is_compact: the weight built above is Q5_0".into());
    };
    let mut ok = true;
    for &m in ms {
        let x = activations(k, m, 0x50c0 + (k + m) as u32);
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut act = Q8Blocks32::new(stream, k, m)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
        q5.enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
        q5.enqueue_gemv_q5_0(stream, w, &act, 0, rows, 0, m, &mut y_dev, 0)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let want = q5_0_gemv_mirror(bytes, k, rows, &act.readback(stream)?, m);
        let differ = y
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let pass = bytes_ok && differ == 0 && y.len() == want.len();
        println!(
            "case=q5_0_resident_is_compact {what} K={k} kb={kb} rows={rows} m={m} \
             packed_bytes={packed_bytes} file_bytes={file} pad_bytes={pad} \
             profile_bytes={profile:?} resident_bytes={resident} placement_bytes={placement:?} \
             bits_differ={differ} of={n} {}",
            verdict(pass),
            n = want.len()
        );
        ok &= pass;
    }
    Ok(ok)
}

/// The Q5_0 gemv as the host transcribes it, `y[r·m + c]` for `rows` rows of
/// raw Q5_0 `bytes` (`k` values a row) and the first `m` columns of `act`
/// (the device quantizer's codes and scales, read back): per block `b` the
/// exact i32 sum `isum` of value `j`'s code `q − 16` (the low nibble of
/// `qs[j]` for `j < 16`, else the high nibble of `qs[j − 16]`, `qh` bit `j`
/// its fifth bit — ggml's `block_q5_0`) times the column's int8 code of
/// value `j` (byte `j % 4` of word `256·(b/32) + 32·(j/4) + b%32` of the
/// column's `256·⌈kb/32⌉` words), then `t = (isum as f32)·d + 0` with `d`
/// the block's f16 scale, and lane `b % 32`'s partial `p = fma(t, e, p)` with
/// `e` the column's block scale, in increasing `b` — the two fused
/// multiply-adds the gemv's PTX carries a column (`fma(a, d, 0)` rounds as
/// the product does) — and the lanes summed by the xor butterfly over 16,
/// 8, 4, 2, 1 (lane 0's sum).
#[cfg(feature = "gpu")]
fn q5_0_gemv_mirror(
    bytes: &[u8],
    k: usize,
    rows: usize,
    act: &bloomery_gpu::q5::Q8Blocks32Host,
    m: usize,
) -> Vec<f32> {
    use gguf::quant::half_to_f32;
    let kb = k / 32;
    let q_stride = 256 * kb.div_ceil(32);
    let mut y = vec![0.0f32; rows * m];
    for r in 0..rows {
        for c in 0..m {
            let mut lanes = [0.0f32; 32];
            for b in 0..kb {
                let blk = &bytes[(r * kb + b) * 22..][..22];
                let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let mut isum = 0i32;
                for j in 0..32 {
                    let nib = if j < 16 {
                        blk[6 + j] & 0x0f
                    } else {
                        blk[6 + j - 16] >> 4
                    };
                    let code = i32::from(nib | ((((qh >> j) & 1) as u8) << 4)) - 16;
                    let word = act.q[c * q_stride + 256 * (b / 32) + 32 * (j / 4) + b % 32];
                    isum += code * i32::from((word >> (8 * (j % 4))) as u8 as i8);
                }
                let e = act.d8[c * kb + b];
                lanes[b % 32] = (isum as f32 * d + 0.0).mul_add(e, lanes[b % 32]);
            }
            for mask in [16, 8, 4, 2, 1] {
                let prev = lanes;
                for (l, v) in lanes.iter_mut().enumerate() {
                    *v = prev[l] + prev[l ^ mask];
                }
            }
            y[r * m + c] = lanes[0];
        }
    }
    y
}

/// `rows` rows of `k` values of synthetic Q5_0 blocks from `seed`: random
/// `qh` and `qs` bytes and a finite f16 `d` of either sign, exponent 8..20.
#[cfg(feature = "gpu")]
fn synthetic_q5_0(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    let mut rng = bloomery_gpu_gates::GateRng(seed);
    let mut out = Vec::with_capacity(rows * k / 32 * 22);
    for _ in 0..rows * k / 32 {
        let sign = rng.below(2) << 15;
        let exp = (8 + rng.below(13)) << 10;
        let d = (sign | exp | rng.below(1 << 10)) as u16;
        out.extend_from_slice(&d.to_le_bytes());
        out.extend((0..20).map(|_| rng.below(256) as u8));
    }
    out
}
