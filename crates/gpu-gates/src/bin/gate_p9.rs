//! GPU kernel gate for package P9 (docs/gpu-design.md work package):
//! device-indirect expert addressing — one launch computes all `n_slots`
//! selected experts of a resident 3-D stack, the ids read from a device
//! buffer, so the launch is addressable from inside a captured CUDA graph
//! (the property a launch scalar like `row0` cannot have: a captured graph
//! freezes every scalar, but a buffer the kernel dereferences per replay
//! does not freeze).
//!
//! The correctness reference is the EXISTING per-expert kernels, and the
//! contract is bit identity, not a band: `q3k_gemv_sel` slot s must equal
//! `q3k_gemv` run on an upload of just expert sel[s]'s 1408 rows (m = 1,
//! the one shared activation column), `q5_0_gemv_sel` slot s must equal
//! `q5_0_gemv` with row0 = sel[s]*2048, col0 = s (each expert's down input
//! differs). Layer-1 tensors, real weights, `activations()` inputs, two sel
//! vectors each (one carrying a duplicate id), a bit-identical rerun, the
//! in-graph replay proof for both kernels, and out-of-range ids (64,
//! u32::MAX) in device slots: the run must not fault, the bad slot's output
//! must stay untouched (sentinel) and every other slot bit-identical.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p9: built without the `gpu` feature; see `just gate-gpu-p9`.");
    std::process::exit(2);
}

// Module-level because the reference helpers below `main` share them.
#[cfg(feature = "gpu")]
use bloomery_gpu::q5::{Q5Kernels, Q8Blocks32, pack_q5_0};
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, activations, bits_equal, bytes_to_words, open_model, tensor_bytes_as, verdict,
};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;
#[cfg(feature = "gpu")]
use std::collections::HashMap;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p9", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    // Fixed seeds: the gate is bit identity, so every input is fixed.
    const SEED_Q3K: u32 = 4211;
    const SEED_Q5: u32 = 5327;
    // sel vectors: the first carries the duplicate id 17 twice, the second
    // 63 twice; both stay within 0..64.
    const SEL_A: [u32; 6] = [0, 5, 63, 17, 17, 2];
    const SEL_B: [u32; 6] = [63, 0, 31, 1, 63, 5];
    // Out-of-range probe: slot 1 carries 64 (one past the last expert),
    // slot 3 carries u32::MAX; the rest are ordinary ids.
    const SEL_OOR: [u32; 6] = [3, 64, 0, u32::MAX, 7, 1];
    const SENT: f32 = 1.0e30;
    const N_SLOTS: usize = 6;

    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let q5 = Q5Kernels::load(gpu.context())?;
    let stream = gpu.stream();

    // ------------------------------------------------ Q3_K: ffn_gate_exps
    // dims [K, rows, experts]
    let (_, w3_bytes) = tensor_bytes_as(
        &gguf,
        "blk.1.ffn_gate_exps.weight",
        GgmlType::Q3_K,
        Some(&[2048, 1408, 64]),
    )?;
    let (k3, rpe3, nexp3) = (2048usize, 1408usize, 64usize);
    let wpm3 = 110 * (k3 / 256) / 4; // 220 u32 words per row
    let words3 = bytes_to_words(w3_bytes);
    assert_eq!(
        words3.len(),
        nexp3 * rpe3 * wpm3,
        "blk.1.ffn_gate_exps: word count"
    );
    let w3_dev = DeviceTensor::upload(stream, &words3, nexp3 * rpe3, wpm3)?;
    // The ONE activation column every slot shares (m = 1: gate and up read
    // the same input).
    let x3 = activations(k3, 1, SEED_Q3K);
    let x3_dev = DeviceBuffer::from_host(stream, &x3)?;
    let mut act3 = Q8Act::with_k(stream, 1, k3)?;
    gpu.enqueue_quantize_q8_1(&x3_dev, &mut act3)?;
    stream.synchronize()?;

    let mut q3k_cache: HashMap<usize, Vec<f32>> = HashMap::new();
    for (tag, sel) in [("a", &SEL_A), ("b", &SEL_B)] {
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N_SLOTS * rpe3)?;
        gpu.enqueue_gemv_q3k_sel(&w3_dev, &act3, &sel_dev, N_SLOTS, rpe3, &mut y_dev)?;
        stream.synchronize()?;
        let y1 = y_dev.to_host_vec(stream)?;
        gpu.enqueue_gemv_q3k_sel(&w3_dev, &act3, &sel_dev, N_SLOTS, rpe3, &mut y_dev)?;
        stream.synchronize()?;
        let y2 = y_dev.to_host_vec(stream)?;
        let rerun_same = bits_equal(&y1, &y2);
        let mut slot_same = true;
        for (s, &id) in sel.iter().enumerate() {
            let r = q3k_expert_ref(
                &gpu,
                &act3,
                &words3,
                id as usize,
                rpe3,
                wpm3,
                &mut q3k_cache,
            )?;
            slot_same &= bits_equal(&y1[s * rpe3..(s + 1) * rpe3], &r);
        }
        let pass = slot_same && rerun_same;
        println!(
            "q3k_sel[{tag}] sel={sel:?} slot_bit_identical={slot_same} bit_identical_rerun={rerun_same} {}",
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    // In-graph proof (the point of the package): capture the _sel launch,
    // replay for sel A, overwrite the sel DEVICE buffer outside the graph,
    // replay the SAME graph — the output must follow the new ids.
    let mut sel_dev = DeviceBuffer::from_host(stream, &SEL_A)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N_SLOTS * rpe3)?;
    let graph = gpu.capture(|_s| {
        gpu.enqueue_gemv_q3k_sel(&w3_dev, &act3, &sel_dev, N_SLOTS, rpe3, &mut y_dev)
    })?;
    let nodes = graph.node_count();
    graph.launch(stream)?;
    stream.synchronize()?;
    let ya = y_dev.to_host_vec(stream)?;
    sel_dev.copy_from_host(stream, &SEL_B)?; // host->device, OUTSIDE the graph
    graph.launch(stream)?;
    stream.synchronize()?;
    let yb = y_dev.to_host_vec(stream)?;
    let mut a_same = true;
    for (s, &id) in SEL_A.iter().enumerate() {
        let r = q3k_expert_ref(
            &gpu,
            &act3,
            &words3,
            id as usize,
            rpe3,
            wpm3,
            &mut q3k_cache,
        )?;
        a_same &= bits_equal(&ya[s * rpe3..(s + 1) * rpe3], &r);
    }
    let mut b_same = true;
    for (s, &id) in SEL_B.iter().enumerate() {
        let r = q3k_expert_ref(
            &gpu,
            &act3,
            &words3,
            id as usize,
            rpe3,
            wpm3,
            &mut q3k_cache,
        )?;
        b_same &= bits_equal(&yb[s * rpe3..(s + 1) * rpe3], &r);
    }
    let pass = a_same && b_same && nodes == 1;
    println!(
        "q3k_graph replay_a_bit_identical={a_same} replay_b_bit_identical={b_same} graph_nodes={nodes} {}",
        verdict(pass)
    );
    if !pass {
        ok = false;
    }

    // Out-of-range ids: the host contract cannot validate a device-side id;
    // the kernel must not fault, the bad slot's output stays untouched and
    // every other slot is bit-identical to its reference.
    {
        let sel_dev = DeviceBuffer::from_host(stream, &SEL_OOR)?;
        let mut y_dev = DeviceBuffer::from_host(stream, &vec![SENT; N_SLOTS * rpe3])?;
        gpu.enqueue_gemv_q3k_sel(&w3_dev, &act3, &sel_dev, N_SLOTS, rpe3, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let (mut good_same, mut bad_untouched) = (true, true);
        for (s, &id) in SEL_OOR.iter().enumerate() {
            if (id as usize) < nexp3 {
                let r = q3k_expert_ref(
                    &gpu,
                    &act3,
                    &words3,
                    id as usize,
                    rpe3,
                    wpm3,
                    &mut q3k_cache,
                )?;
                good_same &= bits_equal(&y[s * rpe3..(s + 1) * rpe3], &r);
            } else {
                bad_untouched &= y[s * rpe3..(s + 1) * rpe3]
                    .iter()
                    .all(|&v| v.to_bits() == SENT.to_bits());
                println!(
                    "q3k_oor slot {s} id={id} produced {} (untouched sentinel)",
                    y[s * rpe3]
                );
            }
        }
        let pass = good_same && bad_untouched;
        println!(
            "q3k_oor good_slots_bit_identical={good_same} bad_slots_untouched={bad_untouched} {}",
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    // ------------------------------------------------ Q5_0: ffn_down_exps
    // dims [K, rows, experts]
    let (_, w5_bytes) = tensor_bytes_as(
        &gguf,
        "blk.1.ffn_down_exps.weight",
        GgmlType::Q5_0,
        Some(&[1408, 2048, 64]),
    )?;
    let (k5, rpe5, nexp5) = (1408usize, 2048usize, 64usize);
    let k_blocks5 = k5 / 32;
    let q_stride5 = 256 * k_blocks5.div_ceil(32);
    let packed5 = pack_q5_0(w5_bytes, k5, nexp5 * rpe5)?;
    let w5_dev = DeviceTensor::upload(stream, &packed5, nexp5 * rpe5, q_stride5 + k_blocks5)?;
    // One activation column per slot (each expert's down input differs).
    let x5 = activations(k5, N_SLOTS, SEED_Q5);
    let x5_dev = DeviceBuffer::from_host(stream, &x5)?;
    let mut act5 = Q8Blocks32::new(stream, k5, N_SLOTS)?;
    q5.enqueue_quantize_q8(stream, &x5_dev, &mut act5, gpu.unlabelled_sink())?;
    stream.synchronize()?;

    for (tag, sel) in [("a", &SEL_A), ("b", &SEL_B)] {
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N_SLOTS * rpe5)?;
        q5.enqueue_gemv_q5_0_sel(stream, &w5_dev, &act5, &sel_dev, N_SLOTS, rpe5, &mut y_dev)?;
        stream.synchronize()?;
        let y1 = y_dev.to_host_vec(stream)?;
        q5.enqueue_gemv_q5_0_sel(stream, &w5_dev, &act5, &sel_dev, N_SLOTS, rpe5, &mut y_dev)?;
        stream.synchronize()?;
        let y2 = y_dev.to_host_vec(stream)?;
        let rerun_same = bits_equal(&y1, &y2);
        let mut slot_same = true;
        for (s, &id) in sel.iter().enumerate() {
            let r = q5_expert_col_ref(&q5, stream, &w5_dev, &act5, id as usize, s, rpe5)?;
            slot_same &= bits_equal(&y1[s * rpe5..(s + 1) * rpe5], &r);
        }
        let pass = slot_same && rerun_same;
        println!(
            "q5_sel[{tag}] sel={sel:?} slot_bit_identical={slot_same} bit_identical_rerun={rerun_same} {}",
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    // In-graph proof for the down projection, same shape as the q3k one.
    let mut sel_dev = DeviceBuffer::from_host(stream, &SEL_A)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N_SLOTS * rpe5)?;
    let graph = gpu.capture(|_s| {
        q5.enqueue_gemv_q5_0_sel(stream, &w5_dev, &act5, &sel_dev, N_SLOTS, rpe5, &mut y_dev)
    })?;
    let nodes = graph.node_count();
    graph.launch(stream)?;
    stream.synchronize()?;
    let ya = y_dev.to_host_vec(stream)?;
    sel_dev.copy_from_host(stream, &SEL_B)?; // host->device, OUTSIDE the graph
    graph.launch(stream)?;
    stream.synchronize()?;
    let yb = y_dev.to_host_vec(stream)?;
    let mut a_same = true;
    for (s, &id) in SEL_A.iter().enumerate() {
        let r = q5_expert_col_ref(&q5, stream, &w5_dev, &act5, id as usize, s, rpe5)?;
        a_same &= bits_equal(&ya[s * rpe5..(s + 1) * rpe5], &r);
    }
    let mut b_same = true;
    for (s, &id) in SEL_B.iter().enumerate() {
        let r = q5_expert_col_ref(&q5, stream, &w5_dev, &act5, id as usize, s, rpe5)?;
        b_same &= bits_equal(&yb[s * rpe5..(s + 1) * rpe5], &r);
    }
    let pass = a_same && b_same && nodes == 1;
    println!(
        "q5_graph replay_a_bit_identical={a_same} replay_b_bit_identical={b_same} graph_nodes={nodes} {}",
        verdict(pass)
    );
    if !pass {
        ok = false;
    }

    // Out-of-range ids for the down projection.
    {
        let sel_dev = DeviceBuffer::from_host(stream, &SEL_OOR)?;
        let mut y_dev = DeviceBuffer::from_host(stream, &vec![SENT; N_SLOTS * rpe5])?;
        q5.enqueue_gemv_q5_0_sel(stream, &w5_dev, &act5, &sel_dev, N_SLOTS, rpe5, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let (mut good_same, mut bad_untouched) = (true, true);
        for (s, &id) in SEL_OOR.iter().enumerate() {
            if (id as usize) < nexp5 {
                let r = q5_expert_col_ref(&q5, stream, &w5_dev, &act5, id as usize, s, rpe5)?;
                good_same &= bits_equal(&y[s * rpe5..(s + 1) * rpe5], &r);
            } else {
                bad_untouched &= y[s * rpe5..(s + 1) * rpe5]
                    .iter()
                    .all(|&v| v.to_bits() == SENT.to_bits());
                println!(
                    "q5_oor slot {s} id={id} produced {} (untouched sentinel)",
                    y[s * rpe5]
                );
            }
        }
        let pass = good_same && bad_untouched;
        println!(
            "q5_oor good_slots_bit_identical={good_same} bad_slots_untouched={bad_untouched} {}",
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_p9 _sel outputs bit-identical to the per-expert kernels slot by slot; \
         graph replay follows ids overwritten between replays; out-of-range ids fault-free"
    );
    Ok(())
}

/// Reference for one Q3_K slot: the EXISTING `q3k_gemv` on an upload of
/// just expert `id`'s `rpe` rows, m = 1 (the shared activation column).
/// Results are cached per expert — duplicate ids in a sel vector must map
/// to one and the same reference bytes.
#[cfg(feature = "gpu")]
fn q3k_expert_ref(
    gpu: &Gpu,
    act: &Q8Act,
    words: &[u32],
    id: usize,
    rpe: usize,
    wpm: usize,
    cache: &mut HashMap<usize, Vec<f32>>,
) -> Result<Vec<f32>, GateError> {
    if let Some(v) = cache.get(&id) {
        return Ok(v.clone());
    }
    let lo = id * rpe * wpm;
    let dev = DeviceTensor::upload(gpu.stream(), &words[lo..lo + rpe * wpm], rpe, wpm)?;
    let mut y = DeviceBuffer::<f32>::zeroed(gpu.stream(), rpe)?;
    gpu.enqueue_gemv_q3k(&dev, act, &mut y)?;
    gpu.stream().synchronize()?;
    let v = y.to_host_vec(gpu.stream())?;
    cache.insert(id, v.clone());
    Ok(v)
}

/// Reference for one Q5_0 slot: the EXISTING `q5_0_gemv` with
/// `row0 = id*rpe`, `col0 = slot`, m_cols = 1 — reading the same resident
/// stack and the same `slot` activation column the `_sel` kernel reads.
#[cfg(feature = "gpu")]
fn q5_expert_col_ref(
    q5: &Q5Kernels,
    stream: &cuda_core::CudaStream,
    w: &DeviceTensor<u32>,
    act: &Q8Blocks32,
    id: usize,
    slot: usize,
    rpe: usize,
) -> Result<Vec<f32>, GateError> {
    let mut y = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
    q5.enqueue_gemv_q5_0(stream, w, act, id * rpe, rpe, slot, 1, &mut y, 0)?;
    stream.synchronize()?;
    Ok(y.to_host_vec(stream)?)
}
