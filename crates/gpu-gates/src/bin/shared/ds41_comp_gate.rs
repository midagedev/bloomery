//! GPU gate for DeepSeek-V4.1's index key of a layer whose keys are projected
//! from its input (`ds41_index_key_input`, Mistral Large 4's
//! `IndexKeys::FromInput`) — the input form of B4 op block E
//! (`docs/research/v41-b4-plan-report.md` §1-E) — on synthetic rows from fixed
//! seeds (`idx_input`): the kernel against its host rule bit for bit (the tail
//! rope at 64 dims and at none), against `ds41_index_key` on rows whose
//! LayerNorm is exactly the RMS norm, against llama.cpp's own `[rope | nope]`
//! head order at the score level, and its refusals and faults.
//!
//! No clause here reads ik's dump.
//! PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause

use std::f32::consts::FRAC_1_SQRT_2;

use bloomery_gpu::index_key::{self, HT_SCALE, IndexKeyInputArgs, IndexKeyKernels};
use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE};
use bloomery_gpu_deepseek41::compress::{CompGeom, StepInts};
use bloomery_gpu_deepseek41::index_key::{CompIndexKeyKernels, IndexKeyArgs};
use bloomery_gpu_deepseek41::rope::{Direction, RopeSpec, RopeTable};
use bloomery_gpu_gates::ds41_meta::RopeMeta;
use bloomery_gpu_gates::rounding::butterfly;
use bloomery_gpu_gates::{GateError, activations, checks_failed, ref_model_path, verdict};
use cuda_core::DeviceBuffer;
use gguf::Split;
use gguf::quant::f32_to_f16_bits;
use model::arch::mistrallarge4::rows::rope_rows_to_tail;

/// What the gate fills a cache with before a launch: an f16 NaN that
/// `f32_to_f16_bits` never writes (it turns every NaN into the canonical
/// `sign | 0x7e00`, whose payload differs), so a row still holding it was
/// not written.
const SENTINEL: u16 = 0xffff;

// ------------------------------------------------------ host rules

/// The tail rope of one row: its last `cs.len()` values turned in pairs
/// by `cs` (`[cos, sin, …]`), every product and sum rounded on its own —
/// the kernels' `rope_pair_rn` and ik's unfused rotation alike.
fn rotate(x: &[f32], cs: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    let tail = y.len() - cs.len();
    for (p, &[c, s]) in y[tail..]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(cs.as_chunks::<2>().0)
    {
        let [x0, x1] = *p;
        *p = [x0 * c - x1 * s, x0 * s + x1 * c];
    }
    y
}

/// ik's `fast_ht` (`iqk_cpu_ops.cpp`): butterflies at `h = 1, 2, 4, …`,
/// `x + y` and `x − y`, the scale multiplied by `0.707106781f` (the f32
/// `FRAC_1_SQRT_2`) per stage, then every value by it.
fn fast_ht(v: &mut [f32]) -> f32 {
    let n = v.len();
    let mut scale = 1.0f32;
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (x, y) = (v[j], v[j + h]);
                v[j] = x + y;
                v[j + h] = x - y;
            }
        }
        scale *= FRAC_1_SQRT_2;
        h <<= 1;
    }
    for x in v.iter_mut() {
        *x *= scale;
    }
    scale
}

/// The YaRN table of each position, `n_dims` values apiece
/// ([`RopeTable::push`]).
fn tables(meta: &RopeMeta<RopeSpec>, pos: &[u32]) -> Result<Vec<Vec<f32>>, GateError> {
    let ours = RopeTable::new(&meta.yarn)?;
    Ok(pos
        .iter()
        .map(|&p| {
            let mut cs = Vec::with_capacity(meta.n_dims);
            ours.push(p, Direction::Forward, &mut cs);
            cs
        })
        .collect())
}

// ------------------------------------------------------------ run

/// What the clauses read besides their inputs.
struct Cx {
    gpu: Gpu,
    idx: CompIndexKeyKernels,
    idx_in: IndexKeyKernels,
    meta: RopeMeta<RopeSpec>,
}

pub fn run() -> Result<(), GateError> {
    let split = Split::open(ref_model_path()?)?;
    let meta = RopeMeta::read(
        &split,
        "gate-gpu-ds41-kernels",
        RopeSpec::window,
        RopeSpec::yarn,
    )?;
    let gpu = Gpu::new()?;
    let idx = CompIndexKeyKernels::load(gpu.context())?;
    let idx_in = IndexKeyKernels::load(gpu.context())?;
    println!(
        "gate_deepseek41_comp: device {} — n_dims {} eps {:e}; yarn rope {:?}; ht_scale {:e}",
        gpu.device_name()?,
        meta.n_dims,
        meta.eps,
        meta.yarn,
        HT_SCALE
    );
    let cx = Cx {
        gpu,
        idx,
        idx_in,
        meta,
    };
    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
    let pass = input_key_section(&cx)?;
    println!("gate_deepseek41_comp: idx_input — {}", verdict(pass));
    if !pass {
        return Err(checks_failed());
    }
    Ok(())
}

// ----------------------------------------------- the key from the input

/// The LayerNorm's epsilon the input-form clauses run at: Mistral Large 4's
/// `attention.layer_norm_epsilon`, 1e-6.
const INPUT_EPS: f32 = 1e-6;

/// The kernel's LayerNorm of one key, op for op (`index_key`'s module doc):
/// the lane sums (lane `l` the values `4l ..`) in f64, the mean rounded to
/// f32, the f32 deviations, the f32 squares added in f64, the variance, the
/// scale, then `((dev · scale) · gain) + bias` each rounded.
fn input_norm_host(x: &[f32], gain: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let lanes = |f: &dyn Fn(usize) -> f64| -> [f64; 32] {
        std::array::from_fn(|l| (0..4).fold(0.0f64, |a, j| a + f(4 * l + j)))
    };
    let mean = (butterfly(lanes(&|i| f64::from(x[i]))) / 128.0) as f32;
    let dev: Vec<f32> = x.iter().map(|&v| v - mean).collect();
    let var = (butterfly(lanes(&|i| f64::from(dev[i] * dev[i]))) / 128.0) as f32;
    let scale = if var.is_finite() {
        1.0 / (var + eps).sqrt()
    } else {
        f32::NAN
    };
    (0..128)
        .map(|i| ((dev[i] * scale) * gain[i]) + bias[i])
        .collect()
}

/// The host rule of `ds41_index_key_input` for one token: the LayerNorm, the
/// tail turn by `cs` (none when `cs` is empty), the transform, the f16 row.
fn input_key_host(x: &[f32], gain: &[f32], bias: &[f32], eps: f32, cs: &[f32]) -> Vec<u16> {
    let mut v = input_norm_host(x, gain, bias, eps);
    if !cs.is_empty() {
        v = rotate(&v, cs);
    }
    fast_ht(&mut v);
    v.iter().map(|&e| f32_to_f16_bits(e)).collect()
}

/// One launch of the input-form kernel into a fresh cache of `rows` sentinel
/// rows, read back.
#[allow(
    clippy::too_many_arguments,
    reason = "a gate helper over the launch's own arguments"
)]
fn input_run(
    cx: &Cx,
    x: &[f32],
    gain: &[f32],
    bias: &[f32],
    pos: &[u32],
    cs: &[f32],
    n_dims: usize,
    rows: usize,
) -> Result<Vec<u16>, GateError> {
    let stream = cx.gpu.stream();
    let w = index_key::WIDTH;
    let x = DeviceBuffer::from_host(stream, x)?;
    let gain = DeviceBuffer::from_host(stream, gain)?;
    let bias = DeviceBuffer::from_host(stream, bias)?;
    let pos_dev = DeviceBuffer::from_host(stream, pos)?;
    let cs = DeviceBuffer::from_host(stream, cs)?;
    let mut cache = DeviceTensor::upload(stream, &vec![SENTINEL; rows * w], rows, w)?;
    cx.idx_in.enqueue_index_key_input(
        stream,
        IndexKeyInputArgs {
            x: &x,
            gain: &gain,
            bias: &bias,
            pos: &pos_dev,
            cs: &cs,
            eps: INPUT_EPS,
            n_dims,
            m: pos.len(),
            cache: &mut cache,
            fault: cx.gpu.unlabelled_sink(),
        },
    )?;
    stream.synchronize()?;
    Ok(cache.buf().to_host_vec(stream)?)
}

/// The rows of `cache` that hold no sentinel.
fn written_rows(cache: &[u16], rows: usize) -> Vec<usize> {
    let w = index_key::WIDTH;
    (0..rows)
        .filter(|&r| cache[r * w..(r + 1) * w].iter().any(|&h| h != SENTINEL))
        .collect()
}

/// Mean of a row's pairs rotated in f64 by `cs`, over the pairs that start at
/// `at` (`[cos, sin, …]`): the reference's rope, no rounding of its own.
fn rotate64(x: &[f64], cs: &[f32], at: usize) -> Vec<f64> {
    let mut y = x.to_vec();
    for (i, c) in cs.as_chunks::<2>().0.iter().enumerate() {
        let (c, s) = (f64::from(c[0]), f64::from(c[1]));
        let (x0, x1) = (x[at + 2 * i], x[at + 2 * i + 1]);
        y[at + 2 * i] = x0 * c - x1 * s;
        y[at + 2 * i + 1] = x0 * s + x1 * c;
    }
    y
}

/// The orthonormal Walsh–Hadamard transform of 128 values, in f64.
fn hadamard64(v: &mut [f64]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h <<= 1;
    }
    let s = 1.0 / (n as f64).sqrt();
    for e in v.iter_mut() {
        *e *= s;
    }
}

/// `ds41_index_key_input` (Mistral Large 4's `IndexKeys::FromInput`) on
/// synthetic rows.
fn input_key_section(cx: &Cx) -> Result<bool, GateError> {
    let w = index_key::WIDTH;
    let n_dims = cx.meta.n_dims;
    let rope = 64usize;
    if n_dims != rope {
        return Err(
            format!("the clauses run at {rope} rope dims; the file's table is {n_dims}").into(),
        );
    }
    let mut ok = true;
    let stream = cx.gpu.stream();

    // ---- the kernel against its host rule, bit for bit, with the tail rope at
    // 64 dims and at none (a NoPE layer); positions scattered and not in row
    // order, the other rows of the cache untouched.
    let pos: Vec<u32> = vec![40, 3, 17, 0, 29, 8, 44, 22, 11];
    let (m, rows) = (pos.len(), 48usize);
    let tab = tables(&cx.meta, &pos)?;
    let cs: Vec<f32> = tab.iter().flatten().copied().collect();
    let x = activations(m * w, 1, 4101);
    let gain: Vec<f32> = activations(w, 1, 4102)
        .iter()
        .map(|&a| 1.0 + 0.25 * a)
        .collect();
    let bias: Vec<f32> = activations(w, 1, 4103).iter().map(|&a| 0.05 * a).collect();
    for (label, dims) in [("rope64", rope), ("nope", 0usize)] {
        let table: &[f32] = if dims == 0 { &[0.0; 4] } else { &cs };
        let got = input_run(cx, &x, &gain, &bias, &pos, table, dims, rows)?;
        let got2 = input_run(cx, &x, &gain, &bias, &pos, table, dims, rows)?;
        let rerun = got == got2;
        let rows_exact = pos.iter().enumerate().all(|(t, &p)| {
            let cs_t: &[f32] = if dims == 0 { &[] } else { &tab[t] };
            let host = input_key_host(&x[t * w..(t + 1) * w], &gain, &bias, INPUT_EPS, cs_t);
            got[p as usize * w..(p as usize + 1) * w] == host[..]
        });
        let mut want_rows: Vec<usize> = pos.iter().map(|&p| p as usize).collect();
        want_rows.sort_unstable();
        let only = written_rows(&got, rows) == want_rows;
        let pass = rows_exact && only && rerun;
        println!(
            "idx_input rule={label} tokens={m} rows={rows} n_dims={dims}: rows = host rule in f16 \
             {rows_exact} other_rows_sentinel={only} bit_identical_rerun={rerun} {}",
            verdict(pass)
        );
        ok &= pass;
    }

    // ---- the file's own head order. Keys in the file's `[rope | nope]` layout
    // go through the load's row move (`rope_rows_to_tail`) into the kernel; the
    // reference is llama.cpp's graph in f64 on the unmoved rows — LayerNorm,
    // the rope of the head's first 64 values, no transform (it is orthogonal
    // and the same on both sides of a score) — and the scores of every key
    // against four queries (the file's order, roped first; the kernel's side
    // moved, roped at the tail and transformed) must agree within the f16
    // rounding of the stored key. The control is the same score with the
    // query left unmoved: wrong by the size of the score itself.
    let kpos: [u32; 6] = [5, 17, 2, 40, 31, 8];
    let qpos: [u32; 4] = [29, 3, 44, 17];
    let all_pos: Vec<u32> = kpos.iter().chain(&qpos).copied().collect();
    let tabs = tables(&cx.meta, &all_pos)?;
    let km = kpos.len();
    let kx_file = activations(km * w, 1, 4201);
    let kgain: Vec<f32> = activations(w, 1, 4202)
        .iter()
        .map(|&a| 1.0 + 0.25 * a)
        .collect();
    let kbias: Vec<f32> = activations(w, 1, 4203).iter().map(|&a| 0.05 * a).collect();
    let mut kx_moved = kx_file.clone();
    rope_rows_to_tail(&mut kx_moved, w, rope);
    let (mut gain_moved, mut bias_moved) = (kgain.clone(), kbias.clone());
    rope_rows_to_tail(&mut gain_moved, w, rope);
    rope_rows_to_tail(&mut bias_moved, w, rope);
    let cs_k: Vec<f32> = tabs[..km].iter().flatten().copied().collect();
    let got = input_run(
        cx,
        &kx_moved,
        &gain_moved,
        &bias_moved,
        &kpos,
        &cs_k,
        rope,
        48,
    )?;
    let queries: Vec<Vec<f64>> = (0..qpos.len())
        .map(|j| {
            activations(w, 1, 4300 + j as u32)
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        })
        .collect();
    let (mut worst, mut control, mut cells) = (0.0f64, 0.0f64, 0usize);
    for (t, &p) in kpos.iter().enumerate() {
        let xk = &kx_file[t * w..(t + 1) * w];
        let mean = xk.iter().map(|&v| f64::from(v)).sum::<f64>() / 128.0;
        let var = xk
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / 128.0;
        let sc = 1.0 / (var + f64::from(INPUT_EPS)).sqrt();
        let normed: Vec<f64> = (0..w)
            .map(|i| (f64::from(xk[i]) - mean) * sc * f64::from(kgain[i]) + f64::from(kbias[i]))
            .collect();
        let k_ref = rotate64(&normed, &tabs[t][..rope], 0);
        let k_ours: Vec<f64> = got[p as usize * w..(p as usize + 1) * w]
            .iter()
            .map(|&h| f64::from(gguf::quant::half_to_f32(h)))
            .collect();
        let kn = k_ref.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (j, q) in queries.iter().enumerate() {
            let qn = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            // The file's order: rope on the first 64 values.
            let q_ref = rotate64(q, &tabs[km + j][..rope], 0);
            let s_ref: f64 = q_ref.iter().zip(&k_ref).map(|(a, b)| a * b).sum();
            // The kernel's side: the move, the tail rope, the transform.
            let mut q_moved = q.clone();
            rope_rows_to_tail(&mut q_moved, w, rope);
            let mut q_ours = rotate64(&q_moved, &tabs[km + j][..rope], w - rope);
            hadamard64(&mut q_ours);
            let s_ours: f64 = q_ours.iter().zip(&k_ours).map(|(a, b)| a * b).sum();
            // The control: the query roped in the file's order, not moved.
            let mut q_wrong = q_ref.clone();
            hadamard64(&mut q_wrong);
            let s_wrong: f64 = q_wrong.iter().zip(&k_ours).map(|(a, b)| a * b).sum();
            let den = (qn * kn).max(f64::MIN_POSITIVE);
            worst = worst.max((s_ours - s_ref).abs() / den);
            control += (s_wrong - s_ref).abs() / den;
            cells += 1;
        }
    }
    let control = control / cells as f64;
    // The f16 store rounds each key value by at most 2^-11 of it, so the score
    // moves by at most 2^-11 of |q|·|k|; the rest is f32 arithmetic.
    let band = 2.0f64.powi(-11) * 1.25;
    let pass = worst <= band && control > 20.0 * band;
    println!(
        "idx_input order keys={km} queries={} head=[rope|nope]->moved: max |score_kernel - \
         score_file_order| / (|q||k|) = {worst:.3e} (band {band:.3e}); control (query unmoved) mean \
         {control:.3e} {}",
        qpos.len(),
        verdict(pass)
    );
    ok &= pass;

    // ---- the whole kernel is `ds41_index_key` on rows whose LayerNorm is
    // exactly its RMS norm: integer rows that sum to zero (mean 0, every sum
    // exact), gain 1 and bias 0, the file's order moved first, at ascending
    // positions and the same tables.
    let n = 8usize;
    let rows = 16usize;
    let pos: Vec<u32> = (0..n as u32).map(|i| 2 * i + 1).collect();
    let tab = tables(&cx.meta, &pos)?;
    let cs: Vec<f32> = tab.iter().flatten().copied().collect();
    let mut x = Vec::with_capacity(n * w);
    for t in 0..n {
        let half: Vec<f32> = activations(w / 2, 1, 4400 + t as u32)
            .iter()
            .map(|&a| (a * 7.0).round())
            .collect();
        let mut row: Vec<f32> = half.iter().flat_map(|&v| [v, 0.0 - v]).collect();
        row[0] = 8.0;
        row[1] = -8.0;
        rope_rows_to_tail(&mut row, w, rope);
        x.extend(row);
    }
    let ones = vec![1.0f32; w];
    let zeros = vec![0.0f32; w];
    let ours = input_run(cx, &x, &ones, &zeros, &pos, &cs, rope, rows)?;
    let geom = CompGeom {
        ratio: 1,
        max_groups: n,
        tokens: n,
        rows,
    };
    let write_row: Vec<u64> = pos.iter().map(|&p| u64::from(p)).collect();
    let read: Vec<u32> = (0..n as u32).map(|g| 1 + g).collect();
    let mut words = vec![0u32; geom.words()];
    geom.pack(
        &StepInts {
            write_row: &write_row,
            read: &read,
            persist_src: &[],
            persist_dst: &[],
        },
        &mut words,
    )?;
    let theirs = {
        let step = DeviceBuffer::from_host(stream, &words)?;
        let k = DeviceBuffer::from_host(stream, &x)?;
        let gain = DeviceBuffer::from_host(stream, &ones)?;
        let cs_dev = DeviceBuffer::from_host(stream, &cs)?;
        let mut cache = DeviceTensor::upload(stream, &vec![SENTINEL; rows * w], rows, w)?;
        cx.idx.enqueue_index_key(
            stream,
            IndexKeyArgs {
                geom,
                step: &step,
                k: &k,
                gain: &gain,
                cs: &cs_dev,
                eps: INPUT_EPS,
                n_dims: rope,
                cache: &mut cache,
                fault: cx.gpu.unlabelled_sink(),
            },
        )?;
        stream.synchronize()?;
        cache.buf().to_host_vec(stream)?
    };
    let same = ours == theirs;
    let wrote = written_rows(&ours, rows).len();
    let pass = same && wrote == n;
    println!(
        "idx_input vs_ds41_index_key tokens={n} rows written={wrote} (integer rows, mean 0, gain 1, \
         bias 0, file order moved): cache bits = ds41_index_key's {same} {}",
        verdict(pass)
    );
    ok &= pass;

    // ---- refusals and faults.
    let (rows, m) = (8usize, 2usize);
    let xr = activations(m * w, 1, 4501);
    let g1 = vec![1.0f32; w];
    let b0 = vec![0.0f32; w];
    let cs2 = vec![0.5f32; m * rope];
    // A position past the cache: CachePos, the token writes nothing, the
    // other token's row is the clean run's.
    let before = cx.gpu.take_fault()?;
    let got = input_run(cx, &xr, &g1, &b0, &[3, rows as u32], &cs2, rope, rows)?;
    let word = cx.gpu.take_fault()?;
    let rows_set = written_rows(&got, rows);
    let pass = before.is_none()
        && word == Some(Fault::at(LAYER_NONE, FaultSite::CachePos))
        && rows_set == [3];
    println!(
        "idx_input fault pos_past_cache word={} rows_written={rows_set:?} (want only row 3) {}",
        word.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    ok &= pass;
    // A NaN in a key: every value of that token's row is non-finite and
    // CacheValue is raised; the other token's row is the clean run's.
    let mut xn = xr.clone();
    xn[w + 5] = f32::NAN;
    let clean = input_run(cx, &xr, &g1, &b0, &[1, 2], &cs2, rope, rows)?;
    let clean_word = cx.gpu.take_fault()?;
    let got = input_run(cx, &xn, &g1, &b0, &[1, 2], &cs2, rope, rows)?;
    let word = cx.gpu.take_fault()?;
    let bad_row = got[2 * w..3 * w]
        .iter()
        .all(|&h| h != SENTINEL && !gguf::quant::half_to_f32(h).is_finite());
    let other = got[w..2 * w] == clean[w..2 * w];
    let pass = clean_word.is_none()
        && word == Some(Fault::at(LAYER_NONE, FaultSite::CacheValue))
        && bad_row
        && other;
    println!(
        "idx_input fault nan_key clean_run_word={} word={} nonfinite_row={bad_row} \
         other_row_clean={other} {}",
        clean_word.map_or("none".to_string(), |f| f.to_string()),
        word.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    ok &= pass;
    // A finite key whose stored values pass f16's range: CacheValue too.
    let big: Vec<f32> = vec![1.0e6; w];
    let got = input_run(cx, &xr, &big, &b0, &[1, 2], &cs2, rope, rows)?;
    let word = cx.gpu.take_fault()?;
    let inf_row = got[w..2 * w]
        .iter()
        .any(|&h| gguf::quant::half_to_f32(h).is_infinite());
    let pass = word == Some(Fault::at(LAYER_NONE, FaultSite::CacheValue)) && inf_row;
    println!(
        "idx_input fault past_f16 word={} inf_in_row={inf_row} {}",
        word.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    ok &= pass;
    // The launcher's refusals, by name, before any launch.
    let dev = |v: &[f32]| DeviceBuffer::from_host(stream, v);
    let (xd, gd, bd, cd) = (dev(&xr)?, dev(&g1)?, dev(&b0)?, dev(&cs2)?);
    let pos2 = DeviceBuffer::from_host(stream, &[1u32, 2])?;
    let mut cache = DeviceTensor::upload(stream, &vec![SENTINEL; rows * w], rows, w)?;
    let mut narrow = DeviceTensor::upload(stream, &vec![SENTINEL; rows * 64], rows, 64)?;
    let sink = cx.gpu.unlabelled_sink();
    let mut try_with = |x: &DeviceBuffer<f32>,
                        pos: &DeviceBuffer<u32>,
                        cs: &DeviceBuffer<f32>,
                        n_dims: usize,
                        m: usize,
                        wide: bool| {
        let cache = if wide { &mut cache } else { &mut narrow };
        cx.idx_in
            .enqueue_index_key_input(
                stream,
                IndexKeyInputArgs {
                    x,
                    gain: &gd,
                    bias: &bd,
                    pos,
                    cs,
                    eps: INPUT_EPS,
                    n_dims,
                    m,
                    cache,
                    fault: sink,
                },
            )
            .is_err()
    };
    let short_x = dev(&xr[..w])?;
    let short_cs = dev(&cs2[..rope])?;
    let pos1 = DeviceBuffer::from_host(stream, &[1u32])?;
    let refusals = [
        ("m_zero", try_with(&xd, &pos2, &cd, rope, 0, true)),
        (
            "n_dims_not_a_lane_multiple",
            try_with(&xd, &pos2, &cd, 6, m, true),
        ),
        (
            "n_dims_past_the_key",
            try_with(&xd, &pos2, &cd, w + 4, m, true),
        ),
        (
            "cache_not_128_wide",
            try_with(&xd, &pos2, &cd, rope, m, false),
        ),
        ("x_short", try_with(&short_x, &pos2, &cd, rope, m, true)),
        ("pos_short", try_with(&xd, &pos1, &cd, rope, m, true)),
        ("cs_short", try_with(&xd, &pos2, &short_cs, rope, m, true)),
    ];
    for (why, refused) in refusals {
        println!(
            "idx_input refusal {why} refused={refused} {}",
            verdict(refused)
        );
        ok &= refused;
    }
    Ok(ok)
}
