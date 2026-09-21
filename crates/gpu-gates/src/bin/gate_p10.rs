//! GPU gate for package P10 (docs/gpu-design.md 작업 꾸러미): resident
//! weights — the whole model file uploaded once, in the device format each
//! kernel consumes, proven against the per-gate uploads P1/P2/P3/P9 made.
//! Every assertion is bit-exact (no bands: this gate is about bytes and
//! addressing, not arithmetic).
//!
//! 1. Census: every tensor of the file classified; per type count, file
//!    bytes, resident bytes, plus totals. A type with no device format
//!    fails the gate outright.
//! 2. Read-back identity: EVERY uploaded tensor read back device->host
//!    equals an independently packed host copy bit for bit — packed here
//!    with the gates' own reference helpers (`bytes_to_words`, `pack_q5_*`,
//!    a local transcription of the Q8_0 planes), not with the loader's
//!    code, so a wrong packing cannot pass against itself. Compared tensor
//!    by tensor; the whole model is never held on the host twice.
//! 3. Staging: `load(0..14)` + `load(14..n)` + globals cover exactly the
//!    full load's names, disjointly, and their resident bytes sum to the
//!    full load's (each dropped before the next — the card is shared).
//! 4. Kernel identity: per format, the owning kernel run from the resident
//!    `DevWeight` equals the same kernel on a gate-style upload of the same
//!    tensor under the same `activations()` input, bit for bit — and the
//!    whole table reruns bit-identically.
//! 5. Derived parity: layer 1's resident derived q_nope2, read back, is the
//!    CPU crate's `Derived` blocks byte for byte (the f16 scale <-> f32
//!    plane round trip is bijective, so plane-bits equality is block-bytes
//!    equality).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p10: built without the `gpu` feature; see `just gate-gpu-p10`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::q5::{Q5Kernels, Q8Blocks32, pack_q5_0, pack_q5_1};
#[cfg(feature = "gpu")]
use bloomery_gpu::q8f32::Q8F32Kernels;
#[cfg(feature = "gpu")]
use bloomery_gpu::weights::{Derived, DevWeight, Q8Block, Weights, resident_size};
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{activations, bytes_to_words, open_model, row_bytes, tensor_bytes};
#[cfg(feature = "gpu")]
use cuda_core::{CudaStream, DeviceBuffer};
#[cfg(feature = "gpu")]
use gguf::Gguf;
#[cfg(feature = "gpu")]
use gguf::quant::{GgmlType, half_to_f32};
#[cfg(feature = "gpu")]
use std::collections::{BTreeMap, BTreeSet};

// Fixed: every check is bit identity, so every input is fixed.
#[cfg(feature = "gpu")]
const SEED: u32 = 10;
// Expert ids for the _sel rows, as gate_p9 used them (one duplicate id).
#[cfg(feature = "gpu")]
const SEL: [u32; 6] = [0, 5, 63, 17, 17, 2];

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The staging cut; layer 1 (the kernel-identity rows) sits below it.
    const CUT: usize = 14;

    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let q5 = Q5Kernels::load(gpu.context())?;
    let q8f32 = Q8F32Kernels::load(gpu.context())?;
    let stream = gpu.stream();
    let mut ok = true;

    let n_layers = gguf
        .block_count()
        .ok_or("gate_p10: metadata key block_count missing")? as usize;
    if n_layers <= CUT {
        return Err(format!("gate_p10: need more than {CUT} blocks, got {n_layers}").into());
    }

    // ------------------------------------------------------------- 1. census
    let mut per_ty: BTreeMap<GgmlType, (usize, u64, u64)> = BTreeMap::new();
    let mut unknown: Vec<(String, GgmlType)> = Vec::new();
    let (mut n_tensors, mut file_total, mut res_total) = (0usize, 0u64, 0u64);
    for t in gguf.iter_tensors() {
        n_tensors += 1;
        file_total += t.nbytes;
        let k = t.dims[0] as usize;
        let rows = tensor_rows(&t.dims);
        match resident_size(t.ty, k, rows) {
            Some(r) => {
                let e = per_ty.entry(t.ty).or_default();
                e.0 += 1;
                e.1 += t.nbytes;
                e.2 += r as u64;
                res_total += r as u64;
            }
            None => unknown.push((t.name.clone(), t.ty)),
        }
    }
    for (ty, (c, fb, rb)) in &per_ty {
        println!("census ty={ty} tensors={c} file_bytes={fb} resident_bytes={rb}");
    }
    // The derived q_nope2 planes are no file tensors: 36 resident bytes per
    // Q8_0 block (8 code words + 1 f32 scale).
    let derived = Derived::new(&gguf)?;
    let derived_total: usize = (0..n_layers)
        .map(|l| derived.wk_b_all_heads(l).map(|b| b.len() * 36).unwrap_or(0))
        .sum();
    println!(
        "census ty=derived_q_nope2 tensors={n_layers} file_bytes=0 resident_bytes={derived_total}"
    );
    println!(
        "census TOTAL tensors={n_tensors} file_bytes={file_total} resident_bytes={} (files {res_total} + derived {derived_total})",
        res_total + derived_total as u64
    );
    if !unknown.is_empty() {
        for (name, ty) in &unknown {
            eprintln!("FAIL: census: tensor {name} has type {ty} with no device format");
        }
        eprintln!(
            "FAILED: gate_p10 census — {} unclassifiable tensor(s)",
            unknown.len()
        );
        std::process::exit(1);
    }

    // ------------------------------------------ 2. full load + read-back
    let full = Weights::load(stream, &gguf, 0..n_layers, true)?;
    if full.resident_bytes() as u64 != res_total + derived_total as u64 {
        eprintln!(
            "FAIL: resident totals disagree: loaded {} vs census {}",
            full.resident_bytes(),
            res_total + derived_total as u64
        );
        ok = false;
    }
    let mut agg: BTreeMap<String, (usize, u64, bool)> = BTreeMap::new();
    let mut readback_ok = true;
    for name in full.names() {
        let Some(dw) = full.get(name) else { continue };
        let (kind, bytes, same, detail) = match expected_planes(&gguf, &derived, name, n_layers)? {
            HostPlanes::Words(want) => {
                let words = match dw {
                    DevWeight::KQuant { ty, w, .. } => (format!("{ty}"), w),
                    DevWeight::Q5_0 { w, .. } => ("q5_0".to_string(), w),
                    DevWeight::Q5_1 { w, .. } => ("q5_1".to_string(), w),
                    _ => {
                        return Err(
                            format!("gate_p10: {name}: word plane on another variant").into()
                        );
                    }
                };
                let got = words.1.buf().to_host_vec(stream)?;
                let same = got == want;
                let detail = (!same).then(|| {
                    format!(
                        "first differing word at {}",
                        got.iter()
                            .zip(want.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(got.len().min(want.len()))
                    )
                });
                (words.0, want.len() as u64 * 4, same, detail)
            }
            HostPlanes::F32(want) => {
                let DevWeight::F32 { w, .. } = dw else {
                    return Err(format!("gate_p10: {name}: f32 plane on another variant").into());
                };
                let got = w.buf().to_host_vec(stream)?;
                (
                    "f32".to_string(),
                    want.len() as u64 * 4,
                    bits_equal_f32(&got, &want),
                    None,
                )
            }
            HostPlanes::Q8(qs_want, d_want) => {
                let DevWeight::Q8_0Derived { qs, d, .. } = dw else {
                    return Err(format!("gate_p10: {name}: q8 planes on another variant").into());
                };
                let qs_got = qs.buf().to_host_vec(stream)?;
                let d_got = d.buf().to_host_vec(stream)?;
                (
                    "derived_q_nope2".to_string(),
                    qs_want.len() as u64 * 4 + d_want.len() as u64 * 4,
                    qs_got == qs_want && bits_equal_f32(&d_got, &d_want),
                    None,
                )
            }
        };
        if let Some(detail) = detail {
            eprintln!("FAIL: readback {name}: {detail}");
            readback_ok = false;
        }
        let e = agg.entry(kind).or_insert((0, 0, true));
        e.0 += 1;
        e.1 += bytes;
        e.2 &= same;
        // 5. derived parity, block level, on layer 1's read-back: ggml's
        // Q8_0 block bytes are the f16 scale bits plus the 32 codes, and
        // both map to the planes bijectively (every f16 is exact in f32),
        // so plane-bits equality is block-bytes equality.
        if name == bloomery_gpu::weights::derived_name(1) {
            let DevWeight::Q8_0Derived { qs, d, .. } = dw else {
                return Err(format!("gate_p10: {name}: not the derived variant").into());
            };
            let qs_got = qs.buf().to_host_vec(stream)?;
            let d_got = d.buf().to_host_vec(stream)?;
            let blocks = derived.wk_b_all_heads(1)?;
            let (mut codes, mut scales) = (true, true);
            for (i, b) in blocks.iter().enumerate() {
                scales &= d_got[i].to_bits() == half_to_f32(b.d).to_bits();
                for j in 0..32 {
                    codes &= (qs_got[i * 8 + j / 4] >> (8 * (j % 4))) as u8 == b.q[j] as u8;
                }
            }
            println!(
                "derived_parity layer=1 blocks={} codes_bit_identical={codes} scales_bit_identical={scales} {}",
                blocks.len(),
                verdict(codes && scales)
            );
            if !(codes && scales) {
                ok = false;
            }
        }
    }
    for (kind, (c, b, same)) in &agg {
        println!("readback kind={kind} tensors={c} host_packed_bytes={b} identical={same}");
        if !same {
            readback_ok = false;
        }
    }
    println!(
        "readback tensors={} identical={readback_ok} {}",
        n_tensors + n_layers,
        verdict(readback_ok)
    );
    if !readback_ok {
        ok = false;
    }

    // The full load's names and resident total; then it goes away.
    let full_names: BTreeSet<String> = full.names().map(str::to_owned).collect();
    let full_resident = full.resident_bytes();
    drop(full);

    // -------------------------------------------------------- 3. staging
    let mut staged: BTreeSet<String> = BTreeSet::new();
    let mut staged_resident = 0usize;
    for (range, globals) in [(0..CUT, false), (CUT..n_layers, false), (0..0, true)] {
        let w = Weights::load(stream, &gguf, range.clone(), globals)?;
        let names: BTreeSet<String> = w.names().map(str::to_owned).collect();
        let res = w.resident_bytes();
        println!(
            "staging range={:?} globals={globals} tensors={} resident_bytes={res}",
            range,
            names.len()
        );
        drop(w);
        for n in names {
            if !staged.insert(n.clone()) {
                eprintln!("FAIL: staging: name {n} covered twice");
                ok = false;
            }
        }
        staged_resident += res;
    }
    let cover = staged == full_names;
    let res_sum = staged_resident == full_resident;
    if !cover {
        eprintln!(
            "FAIL: staging name cover: missing {:?} extra {:?}",
            full_names.difference(&staged).collect::<Vec<_>>(),
            staged.difference(&full_names).collect::<Vec<_>>()
        );
    }
    if !res_sum {
        eprintln!("FAIL: staging resident sum {staged_resident} != full {full_resident}");
    }
    println!(
        "staging cover_exact={cover} disjoint_resident_sum={res_sum} full_resident_bytes={full_resident} {}",
        verdict(cover && res_sum)
    );
    if !(cover && res_sum) {
        ok = false;
    }

    // ------------------------------------------ 4. kernel identity table
    // Q8_0 as a FILE type cannot appear: this reader's GgmlType has no Q8_0
    // arm and refuses such a file at open. The q8f32 format still runs, one
    // row below, through the derived variant.
    println!("skip q8_0 absent");
    let w = Weights::load(stream, &gguf, 0..2, true)?;
    let t1 = run_table(&gguf, &gpu, &q5, &q8f32, stream, &w, &derived, SEED)?;
    let t2 = run_table(&gguf, &gpu, &q5, &q8f32, stream, &w, &derived, SEED)?;
    let mut rerun_same = t1.len() == t2.len();
    for ((l1, y1), (l2, y2)) in t1.iter().zip(t2.iter()) {
        let same = l1 == l2 && y1 == y2;
        if !same {
            eprintln!("FAIL: kernel identity rerun: row {l1} differs from its rerun");
        }
        rerun_same &= same;
    }
    println!(
        "kernel_identity rerun_of_whole_table_bit_identical={rerun_same} rows={} {}",
        t1.len(),
        verdict(rerun_same)
    );
    if !rerun_same {
        ok = false;
    }
    drop(w);

    if !ok {
        eprintln!("FAILED: gate_p10");
        std::process::exit(1);
    }
    println!(
        "PASSED: gate_p10 — census complete, every resident tensor reads back as its independent \
         host packing, staging covers the model disjointly at equal bytes, and each format's \
         kernel is bit-identical between the resident weight and the gate-style upload"
    );
    Ok(())
}

/// Rows of a tensor as the kernels count them: dims[1]·dims[2]·… (1 for a
/// 1-D tensor); a 3-D expert stack is its experts' rows stacked.
#[cfg(feature = "gpu")]
fn tensor_rows(dims: &[u64]) -> usize {
    dims[1..].iter().product::<u64>() as usize
}

/// The host-side packing a resident tensor must read back as, computed with
/// the gates' own reference helpers — never the loader's code.
#[cfg(feature = "gpu")]
enum HostPlanes {
    Words(Vec<u32>),
    F32(Vec<f32>),
    Q8(Vec<u32>, Vec<f32>),
}

#[cfg(feature = "gpu")]
fn expected_planes(
    gguf: &Gguf,
    derived: &Derived,
    name: &str,
    n_layers: usize,
) -> Result<HostPlanes, Box<dyn std::error::Error>> {
    if let Some(rest) = name.strip_prefix("derived.blk.") {
        let l: usize = rest
            .strip_suffix(".q_nope2")
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| format!("gate_p10: malformed derived name {name:?}"))?;
        if l >= n_layers {
            return Err(format!("gate_p10: derived layer {l} out of range").into());
        }
        let (qs, d) = gate_q8_planes(derived.wk_b_all_heads(l)?);
        return Ok(HostPlanes::Q8(qs, d));
    }
    let (info, bytes) = tensor_bytes(gguf, name)?;
    let k = info.dims[0] as usize;
    let rows = tensor_rows(&info.dims);
    Ok(match info.ty {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K => {
            let rb = row_bytes(info.ty, k)?;
            HostPlanes::Words(bytes_to_words(&bytes[..rb * rows]))
        }
        GgmlType::Q5_0 => HostPlanes::Words(pack_q5_0(bytes, k, rows)?),
        GgmlType::Q5_1 => HostPlanes::Words(pack_q5_1(bytes, k, rows)?),
        GgmlType::F32 => HostPlanes::F32(
            bytes[..rows * k * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        other => {
            return Err(format!("gate_p10: {name}: type {other} has no reference packing").into());
        }
    })
}

/// The q8f32 two-plane transcription of Q8_0 blocks, written here from the
/// `Q8Block` fields so the read-back has an anchor the loader does not
/// share: code j in word j/4 byte j%4, scale = the block's f16 as f32.
#[cfg(feature = "gpu")]
fn gate_q8_planes(blocks: &[Q8Block]) -> (Vec<u32>, Vec<f32>) {
    let mut qs = Vec::with_capacity(blocks.len() * 8);
    let mut d = Vec::with_capacity(blocks.len());
    for b in blocks {
        let mut w = [0u32; 8];
        for (j, &q) in b.q.iter().enumerate() {
            w[j / 4] |= u32::from(q as u8) << (8 * (j % 4));
        }
        qs.extend_from_slice(&w);
        d.push(half_to_f32(b.d));
    }
    (qs, d)
}

/// One kernel-identity row set: per format, the owning kernel on the
/// resident `DevWeight` vs a gate-style upload of the same tensor under the
/// same `activations()` input. Returns each row's resident output bits (the
/// whole-table rerun compares these).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn run_table(
    gguf: &Gguf,
    gpu: &Gpu,
    q5: &Q5Kernels,
    q8f32: &bloomery_gpu::q8f32::Q8F32Kernels,
    stream: &CudaStream,
    w: &Weights,
    derived: &Derived,
    seed: u32,
) -> Result<Vec<(String, Vec<u32>)>, Box<dyn std::error::Error>> {
    let mut rows: Vec<(String, Vec<u32>)> = Vec::new();

    // K-quant plain rows: layer-1 tensors (attn_q Q3_K, attn_output Q4_K)
    // and the global Q6_K output matrix, each at m = 1 and m = 8, against a
    // gate_p1-style word upload.
    for (label, name) in [
        ("q3k", "blk.1.attn_q.weight"),
        ("q4k", "blk.1.attn_output.weight"),
        ("q6k", "output.weight"),
    ] {
        let (info, bytes) = tensor_bytes(gguf, name)?;
        let (k, nrows) = (info.dims[0] as usize, tensor_rows(&info.dims));
        let rb = row_bytes(info.ty, k)?;
        let words = bytes_to_words(&bytes[..rb * nrows]);
        let rref = DeviceTensor::upload(stream, &words, nrows, words.len() / nrows)?;
        let DevWeight::KQuant { ty, w: res, .. } = w.get(name).unwrap() else {
            return Err(format!("run_table: {name} is not a KQuant resident weight").into());
        };
        if *ty != info.ty {
            return Err(format!(
                "run_table: {name}: resident type {ty} != file type {}",
                info.ty
            )
            .into());
        }
        for m in [1usize, 8] {
            let (same, rerun, bits) =
                kid_kquant(gpu, stream, info.ty, res, &rref, nrows, k, m, seed)?;
            println!(
                "kid row={label}_m{m} K={k} rows={nrows} resident_vs_gate_upload_bit_identical={same} resident_rerun_bit_identical={rerun} {}",
                verdict(same && rerun)
            );
            rows.push((format!("{label}_m{m}"), bits));
            if !(same && rerun) {
                return Err(format!("run_table: row {label}_m{m} failed").into());
            }
        }
    }

    // Q5_1: the dense down projection of block 0 (the only Q5_1 site), m =
    // 1 and 8, against a gate_p2-style packed upload.
    {
        let (info, bytes) = tensor_bytes(gguf, "blk.0.ffn_down.weight")?;
        assert_eq!(info.ty, GgmlType::Q5_1, "blk.0.ffn_down type");
        let (k, nrows) = (info.dims[0] as usize, tensor_rows(&info.dims));
        let kb = k / 32;
        let cols = 256 * kb.div_ceil(32) + 2 * kb;
        let rref = DeviceTensor::upload(stream, &pack_q5_1(bytes, k, nrows)?, nrows, cols)?;
        let DevWeight::Q5_1 { w: res, .. } = w.get("blk.0.ffn_down.weight").unwrap() else {
            return Err("run_table: blk.0.ffn_down is not Q5_1".into());
        };
        for m in [1usize, 8] {
            let (same, rerun, bits) = kid_q5_1(q5, stream, res, &rref, nrows, k, m, seed)?;
            println!(
                "kid row=q5_1_m{m} K={k} rows={nrows} resident_vs_gate_upload_bit_identical={same} resident_rerun_bit_identical={rerun} {}",
                verdict(same && rerun)
            );
            rows.push((format!("q5_1_m{m}"), bits));
            if !(same && rerun) {
                return Err(format!("run_table: row q5_1_m{m} failed").into());
            }
        }
    }

    // Q5_0 expert stack through the device-indirect down projection, sel as
    // gate_p9 used it, against a gate_p9-style packed upload of the stack.
    {
        let (info, bytes) = tensor_bytes(gguf, "blk.1.ffn_down_exps.weight")?;
        assert_eq!(info.ty, GgmlType::Q5_0, "blk.1.ffn_down_exps type");
        let (k, rpe, nexp) = (
            info.dims[0] as usize,
            info.dims[1] as usize,
            info.dims[2] as usize,
        );
        let kb = k / 32;
        let rref = DeviceTensor::upload(
            stream,
            &pack_q5_0(bytes, k, nexp * rpe)?,
            nexp * rpe,
            256 * kb.div_ceil(32) + kb,
        )?;
        let DevWeight::Q5_0 { w: res, .. } = w.get("blk.1.ffn_down_exps.weight").unwrap() else {
            return Err("run_table: blk.1.ffn_down_exps is not Q5_0".into());
        };
        let x = activations(k, SEL.len(), seed);
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut act = Q8Blocks32::new(stream, k, SEL.len())?;
        q5.enqueue_quantize_q8(stream, &x_dev, &mut act)?;
        let sel_dev = DeviceBuffer::from_host(stream, &SEL)?;
        let (same, rerun, bits) = kid_q5_0_sel(q5, stream, res, &rref, &act, &sel_dev, rpe)?;
        println!(
            "kid row=q5_0_sel sel={SEL:?} K={k} rows_per_expert={rpe} resident_vs_gate_upload_bit_identical={same} resident_rerun_bit_identical={rerun} {}",
            verdict(same && rerun)
        );
        rows.push(("q5_0_sel".into(), bits));
        if !(same && rerun) {
            return Err("run_table: row q5_0_sel failed".into());
        }
    }

    // Q3_K expert stack through the device-indirect gate projection, same
    // sel, against a gate_p9-style word upload of the stack.
    {
        let (info, bytes) = tensor_bytes(gguf, "blk.1.ffn_gate_exps.weight")?;
        assert_eq!(info.ty, GgmlType::Q3_K, "blk.1.ffn_gate_exps type");
        let (k, rpe, nexp) = (
            info.dims[0] as usize,
            info.dims[1] as usize,
            info.dims[2] as usize,
        );
        let wpm = 110 * (k / 256) / 4;
        let words = bytes_to_words(bytes);
        if words.len() != nexp * rpe * wpm {
            return Err(format!(
                "run_table: blk.1.ffn_gate_exps: {} words != {nexp}*{rpe}*{wpm}",
                words.len()
            )
            .into());
        }
        let rref = DeviceTensor::upload(stream, &words, nexp * rpe, wpm)?;
        let DevWeight::KQuant { w: res, .. } = w.get("blk.1.ffn_gate_exps.weight").unwrap() else {
            return Err("run_table: blk.1.ffn_gate_exps is not KQuant".into());
        };
        let x = activations(k, 1, seed);
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut act = Q8Act::with_k(stream, 1, k)?;
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        stream.synchronize()?;
        let sel_dev = DeviceBuffer::from_host(stream, &SEL)?;
        let (same, rerun, bits) = kid_q3k_sel(gpu, stream, res, &rref, &act, &sel_dev, rpe)?;
        println!(
            "kid row=q3k_sel sel={SEL:?} K={k} rows_per_expert={rpe} resident_vs_gate_upload_bit_identical={same} resident_rerun_bit_identical={rerun} {}",
            verdict(same && rerun)
        );
        rows.push(("q3k_sel".into(), bits));
        if !(same && rerun) {
            return Err("run_table: row q3k_sel failed".into());
        }
    }

    // F32 router: blk.1.ffn_gate_inp against a gate_p3-style f32 upload.
    {
        let (info, bytes) = tensor_bytes(gguf, "blk.1.ffn_gate_inp.weight")?;
        assert_eq!(info.ty, GgmlType::F32, "blk.1.ffn_gate_inp type");
        let (k, nrows) = (info.dims[0] as usize, tensor_rows(&info.dims));
        let vals: Vec<f32> = bytes[..nrows * k * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let rref = DeviceTensor::upload(stream, &vals, nrows, k)?;
        let DevWeight::F32 { w: res, .. } = w.get("blk.1.ffn_gate_inp.weight").unwrap() else {
            return Err("run_table: blk.1.ffn_gate_inp is not F32".into());
        };
        for m in [1usize, 8] {
            let (same, rerun, bits) = kid_f32(q8f32, stream, res, &rref, nrows, k, m, seed)?;
            println!(
                "kid row=f32_router_m{m} K={k} rows={nrows} resident_vs_gate_upload_bit_identical={same} resident_rerun_bit_identical={rerun} {}",
                verdict(same && rerun)
            );
            rows.push((format!("f32_router_m{m}"), bits));
            if !(same && rerun) {
                return Err(format!("run_table: row f32_router_m{m} failed").into());
            }
        }
    }

    // The q8f32 format through the derived variant: layer 1's q_nope2
    // planes, resident vs a gate_p3-style upload of the same planes.
    {
        let DevWeight::Q8_0Derived { qs: rqs, d: rd, k } = w
            .get(&bloomery_gpu::weights::derived_name(1))
            .ok_or("run_table: derived.blk.1.q_nope2 missing")?
        else {
            return Err("run_table: derived.blk.1.q_nope2 is not the derived variant".into());
        };
        let (qs, d) = gate_q8_planes(derived.wk_b_all_heads(1)?);
        let nrows = d.len();
        let rqs_ref = DeviceTensor::upload(stream, &qs, nrows, qs.len() / nrows)?;
        let rd_ref = DeviceTensor::upload(stream, &d, nrows, d.len() / nrows)?;
        for m in [1usize, 8] {
            let (same, rerun, bits) = kid_q8_derived(
                q8f32, stream, rqs, rd, &rqs_ref, &rd_ref, nrows, *k, m, seed,
            )?;
            println!(
                "kid row=q8_0_derived_m{m} K={k} rows={nrows} resident_vs_gate_upload_bit_identical={same} resident_rerun_bit_identical={rerun} {}",
                verdict(same && rerun)
            );
            rows.push((format!("q8_0_derived_m{m}"), bits));
            if !(same && rerun) {
                return Err(format!("run_table: row q8_0_derived_m{m} failed").into());
            }
        }
    }

    Ok(rows)
}

/// One K-quant gemv shape on two uploads of the same tensor, one shared
/// quantized activation: (resident vs reference, resident rerun, resident
/// output bits).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn kid_kquant(
    gpu: &Gpu,
    stream: &CudaStream,
    ty: GgmlType,
    res: &DeviceTensor<u32>,
    rref: &DeviceTensor<u32>,
    nrows: usize,
    k: usize,
    m: usize,
    seed: u32,
) -> Result<(bool, bool, Vec<u32>), Box<dyn std::error::Error>> {
    if res.rows() != rref.rows() || res.cols() != rref.cols() {
        return Err(format!(
            "kid_kquant: resident {}x{} vs reference {}x{}",
            res.rows(),
            res.cols(),
            rref.rows(),
            rref.cols()
        )
        .into());
    }
    let x = activations(k, m, seed);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Act::with_k(stream, m, k)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    stream.synchronize()?;
    let run = |w: &DeviceTensor<u32>,
               y: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        match ty {
            GgmlType::Q3_K => gpu.enqueue_gemv_q3k(w, &act, y),
            GgmlType::Q4_K => gpu.enqueue_gemv_q4k(w, &act, y),
            GgmlType::Q6_K => gpu.enqueue_gemv_q6k(w, &act, y),
            other => Err(format!("kid_kquant: no gemv for {other:?}").into()),
        }
    };
    let mut y1 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1v = y1.to_host_vec(stream)?;
    let mut y2 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(rref, &mut y2)?;
    stream.synchronize()?;
    let y2v = y2.to_host_vec(stream)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1b = y1.to_host_vec(stream)?;
    Ok((
        bits_equal_f32(&y1v, &y2v),
        bits_equal_f32(&y1v, &y1b),
        y1v.iter().map(|v| v.to_bits()).collect(),
    ))
}

/// One Q5_1 gemv shape on two uploads, one shared quantized activation:
/// returns as `kid_kquant`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn kid_q5_1(
    q5: &Q5Kernels,
    stream: &CudaStream,
    res: &DeviceTensor<u32>,
    rref: &DeviceTensor<u32>,
    nrows: usize,
    k: usize,
    m: usize,
    seed: u32,
) -> Result<(bool, bool, Vec<u32>), Box<dyn std::error::Error>> {
    if res.rows() != rref.rows() || res.cols() != rref.cols() {
        return Err(format!(
            "kid_q5_1: resident {}x{} vs reference {}x{}",
            res.rows(),
            res.cols(),
            rref.rows(),
            rref.cols()
        )
        .into());
    }
    let x = activations(k, m, seed);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Blocks32::new(stream, k, m)?;
    q5.enqueue_quantize_q8(stream, &x_dev, &mut act)?;
    stream.synchronize()?;
    let run = |w: &DeviceTensor<u32>,
               y: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        q5.enqueue_gemv_q5_1(stream, w, &act, 0, nrows, 0, m, y, 0)
    };
    let mut y1 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1v = y1.to_host_vec(stream)?;
    let mut y2 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(rref, &mut y2)?;
    stream.synchronize()?;
    let y2v = y2.to_host_vec(stream)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1b = y1.to_host_vec(stream)?;
    Ok((
        bits_equal_f32(&y1v, &y2v),
        bits_equal_f32(&y1v, &y1b),
        y1v.iter().map(|v| v.to_bits()).collect(),
    ))
}

/// One Q5_0 `_sel` launch on two uploads of the expert stack: six slots
/// read six quantized columns and select rows of the stack by `sel`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn kid_q5_0_sel(
    q5: &Q5Kernels,
    stream: &CudaStream,
    res: &DeviceTensor<u32>,
    rref: &DeviceTensor<u32>,
    act: &Q8Blocks32,
    sel: &DeviceBuffer<u32>,
    rpe: usize,
) -> Result<(bool, bool, Vec<u32>), Box<dyn std::error::Error>> {
    if res.rows() != rref.rows() || res.cols() != rref.cols() {
        return Err(format!(
            "kid_q5_0_sel: resident {}x{} vs reference {}x{}",
            res.rows(),
            res.cols(),
            rref.rows(),
            rref.cols()
        )
        .into());
    }
    let run = |w: &DeviceTensor<u32>,
               y: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        q5.enqueue_gemv_q5_0_sel(stream, w, act, sel, SEL.len(), rpe, y)
    };
    let mut y1 = DeviceBuffer::<f32>::zeroed(stream, SEL.len() * rpe)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1v = y1.to_host_vec(stream)?;
    let mut y2 = DeviceBuffer::<f32>::zeroed(stream, SEL.len() * rpe)?;
    run(rref, &mut y2)?;
    stream.synchronize()?;
    let y2v = y2.to_host_vec(stream)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1b = y1.to_host_vec(stream)?;
    Ok((
        bits_equal_f32(&y1v, &y2v),
        bits_equal_f32(&y1v, &y1b),
        y1v.iter().map(|v| v.to_bits()).collect(),
    ))
}

/// One Q3_K `_sel` launch on two uploads of the expert stack (m = 1: every
/// slot shares the one quantized column).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn kid_q3k_sel(
    gpu: &Gpu,
    stream: &CudaStream,
    res: &DeviceTensor<u32>,
    rref: &DeviceTensor<u32>,
    act: &Q8Act,
    sel: &DeviceBuffer<u32>,
    rpe: usize,
) -> Result<(bool, bool, Vec<u32>), Box<dyn std::error::Error>> {
    if res.rows() != rref.rows() || res.cols() != rref.cols() {
        return Err(format!(
            "kid_q3k_sel: resident {}x{} vs reference {}x{}",
            res.rows(),
            res.cols(),
            rref.rows(),
            rref.cols()
        )
        .into());
    }
    let run = |w: &DeviceTensor<u32>,
               y: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        gpu.enqueue_gemv_q3k_sel(w, act, sel, SEL.len(), rpe, y)
    };
    let mut y1 = DeviceBuffer::<f32>::zeroed(stream, SEL.len() * rpe)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1v = y1.to_host_vec(stream)?;
    let mut y2 = DeviceBuffer::<f32>::zeroed(stream, SEL.len() * rpe)?;
    run(rref, &mut y2)?;
    stream.synchronize()?;
    let y2v = y2.to_host_vec(stream)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1b = y1.to_host_vec(stream)?;
    Ok((
        bits_equal_f32(&y1v, &y2v),
        bits_equal_f32(&y1v, &y1b),
        y1v.iter().map(|v| v.to_bits()).collect(),
    ))
}

/// One F32 gemv shape on two uploads: returns as `kid_kquant`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn kid_f32(
    q8f32: &bloomery_gpu::q8f32::Q8F32Kernels,
    stream: &CudaStream,
    res: &DeviceTensor<f32>,
    rref: &DeviceTensor<f32>,
    nrows: usize,
    k: usize,
    m: usize,
    seed: u32,
) -> Result<(bool, bool, Vec<u32>), Box<dyn std::error::Error>> {
    if res.rows() != rref.rows() || res.cols() != rref.cols() {
        return Err(format!(
            "kid_f32: resident {}x{} vs reference {}x{}",
            res.rows(),
            res.cols(),
            rref.rows(),
            rref.cols()
        )
        .into());
    }
    let x = activations(k, m, seed);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let run = |w: &DeviceTensor<f32>,
               y: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        q8f32.enqueue_f32_gemv(stream, w, &x_dev, m, y)
    };
    let mut y1 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1v = y1.to_host_vec(stream)?;
    let mut y2 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(rref, &mut y2)?;
    stream.synchronize()?;
    let y2v = y2.to_host_vec(stream)?;
    run(res, &mut y1)?;
    stream.synchronize()?;
    let y1b = y1.to_host_vec(stream)?;
    Ok((
        bits_equal_f32(&y1v, &y2v),
        bits_equal_f32(&y1v, &y1b),
        y1v.iter().map(|v| v.to_bits()).collect(),
    ))
}

/// One Q8_0 gemv shape on two uploads of the two planes: returns as
/// `kid_kquant`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn kid_q8_derived(
    q8f32: &bloomery_gpu::q8f32::Q8F32Kernels,
    stream: &CudaStream,
    rqs: &DeviceTensor<u32>,
    rd: &DeviceTensor<f32>,
    qs_ref: &DeviceTensor<u32>,
    d_ref: &DeviceTensor<f32>,
    nrows: usize,
    k: usize,
    m: usize,
    seed: u32,
) -> Result<(bool, bool, Vec<u32>), Box<dyn std::error::Error>> {
    if rqs.rows() != qs_ref.rows()
        || rqs.cols() != qs_ref.cols()
        || rd.rows() != d_ref.rows()
        || rd.cols() != d_ref.cols()
    {
        return Err(format!(
            "kid_q8_derived: resident qs {}x{} d {}x{} vs reference qs {}x{} d {}x{}",
            rqs.rows(),
            rqs.cols(),
            rd.rows(),
            rd.cols(),
            qs_ref.rows(),
            qs_ref.cols(),
            d_ref.rows(),
            d_ref.cols()
        )
        .into());
    }
    let x = activations(k, m, seed);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let run = |qs: &DeviceTensor<u32>,
               d: &DeviceTensor<f32>,
               y: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        q8f32.enqueue_q8_0_gemv(stream, qs, d, &x_dev, m, y)
    };
    let mut y1 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(rqs, rd, &mut y1)?;
    stream.synchronize()?;
    let y1v = y1.to_host_vec(stream)?;
    let mut y2 = DeviceBuffer::<f32>::zeroed(stream, nrows * m)?;
    run(qs_ref, d_ref, &mut y2)?;
    stream.synchronize()?;
    let y2v = y2.to_host_vec(stream)?;
    run(rqs, rd, &mut y1)?;
    stream.synchronize()?;
    let y1b = y1.to_host_vec(stream)?;
    Ok((
        bits_equal_f32(&y1v, &y2v),
        bits_equal_f32(&y1v, &y1b),
        y1v.iter().map(|v| v.to_bits()).collect(),
    ))
}

#[cfg(feature = "gpu")]
fn bits_equal_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(feature = "gpu")]
fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}
