//! GPU kernel gate for `q6k_gemv_ids` and `q6k_gemv_ids_mcol`
//! (`bloomery_gpu::q6k_ids`): the MTP draft's borrowed head over a Q6_K
//! `output.weight`, through a row map. No model file — a synthetic
//! "vocabulary" of 70,000 Q6_K rows at K = 2,560 (ten super-blocks, the
//! UD-Q3_K_XL head's geometry), finite `d`, random codes, fixed seeds.
//!
//! What is asserted:
//! - (a) **Bits.** For m ∈ {1, 2, 4, 8} and maps of 1, 7 and 65,536 sorted
//!   ids, plus no map (the full head): every logit equals
//!   `enqueue_gemv_q6k` over an upload of the gathered rows against the same
//!   quantized activation, bit for bit; every column of an m-column launch
//!   is the m = 1 launch of that column's own one-column activation, bit for
//!   bit; and a rerun is bit-identical.
//! - (b) **f64 band.** Against `gguf::dequant_row` times the f32 input,
//!   summed in f64: every sampled (row, column) within the derived bound of
//!   the q8_1 quantization and the f32 accumulation ([`BAND_DOC`]), and the
//!   relative error under [`REL_PIN`].
//! - (c) **Faults.** An id at or past the matrix's rows raises
//!   `FaultSite::TokenId` on the fault word as an unlabelled launch and
//!   writes its row's outputs NaN, the other rows still bit-identical to
//!   their references; the launch after the fault's read-and-clear raises
//!   nothing.
//! - (d) **Capture.** A captured launch replayed equals the eager launch,
//!   bit for bit, and follows ids overwritten between replays; the graph is
//!   the one launch.
//! - (e) **Refusals by name.** m of 0 and 9, m past the activation's
//!   columns, a K not a multiple of 512, a map shorter than the rows and a y
//!   shorter than rows·m are each refused as the launcher's own `Shape`
//!   error.
//!
//! FAIL-first for this gate: (i) the map ignored — the kernel reading row
//! `r` for `ids[r]` — turns (a) red; (ii) a column stride off by one word
//! turns (a)'s per-column comparison red; (iii) the kernel's id check
//! removed turns (c) red.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_q6k_ids: built without the `gpu` feature; see `just gate-gpu-q6k-ids`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::q6k_ids::{Q6kIdsArgs, Q6kIdsKernels};
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::rounding::{U, gamma};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, verdict};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::quant::{GgmlType, dequant_row};

/// The synthetic vocabulary's rows.
#[cfg(feature = "gpu")]
const ROWS: usize = 70_000;
/// K, the file's head width: ten super-blocks, an even count.
#[cfg(feature = "gpu")]
const K: usize = 2_560;
/// Words per Q6_K row at K: 210·10/4.
#[cfg(feature = "gpu")]
const WPR: usize = 525;
/// The maps' lengths, and the seeds everything fixed depends on.
#[cfg(feature = "gpu")]
const MAP_LENS: [usize; 3] = [1, 7, 65_536];
#[cfg(feature = "gpu")]
const SEED_W: u64 = 0x9e37_79b9_7f4a_7c15;
#[cfg(feature = "gpu")]
const SEED_X: u32 = 6101;
#[cfg(feature = "gpu")]
const SEED_MAP: u32 = 6203;
/// The rows of each map the f64 band walks.
#[cfg(feature = "gpu")]
const BAND_ROWS: usize = 24;
/// What a `SENT`-filled y holds before every launch, so a row the kernel
/// leaves alone reads back as these bits.
#[cfg(feature = "gpu")]
const SENT: f32 = 1.0e30;
/// The bound's derivation, (b)'s: the q8_1 quantization moves each value by
/// at most its block's `d/2` (`d = amax/127`), so the dot by
/// `Σ_b (d_b/2)(1+256U)·Σ|w|`; the kernel accumulates one f32 term a
/// super-block pair a lane (`iters` of them) and reduces the 32 lanes by the
/// butterfly, so the arithmetic sits within `γ(2·iters + 6)` of the terms'
/// magnitude `Σ|w·x|`, the quantization term included once for the terms it
/// feeds. PIN(2026-10-06): max|ours − ref| / max|ref| at most [`REL_PIN`]
/// over m ∈ {1, 2, 4, 8} and every map; measured 1.5094e-2 at m = 4 (the
/// q8_32 model's 1.2858e-2 RMS bound is the scale).
#[cfg(feature = "gpu")]
const BAND_DOC: &str = "quant + gamma(2*iters + 6) * (wx + quant)";
#[cfg(feature = "gpu")]
const REL_PIN: f64 = 1.6e-2;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_q6k_ids", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let gpu = Gpu::new()?;
    let kern = Q6kIdsKernels::load(gpu.context())?;
    let stream = gpu.stream();

    let words = synthetic_q6k(ROWS, K / 256, SEED_W);
    let w = DeviceTensor::upload(stream, &words, ROWS, WPR)?;
    let no_map = DeviceBuffer::from_host(stream, &[0u32; 1])?;

    let mut ok = true;
    ok &= check_bits(&gpu, &kern, &w, &words, &no_map)?;
    ok &= check_band(&gpu, &kern, &w, &words, &no_map)?;
    ok &= check_fault(&gpu, &kern, &w, &words, &no_map)?;
    ok &= check_capture(&gpu, &kern, &w, &words, &no_map)?;
    ok &= check_refusals(&gpu, &kern, &w, &no_map)?;

    if !ok {
        return Err(checks_failed());
    }
    println!(
        "PASSED: gate_q6k_ids q6k_gemv_ids and _mcol bit-identical to enqueue_gemv_q6k over the \
         gathered rows at m in {{1, 2, 4, 8}} over maps of 1, 7 and 65,536 ids and no map, each \
         column the m = 1 launch of its own; within the derived f64 band ({BAND_DOC}); an id \
         past the rows raising token_id with a NaN row and a clean launch after; the captured \
         replay eager and following overwritten ids; every launcher misuse refused by name"
    );
    Ok(())
}

/// One launch's shape: the map over the matrix's rows (`None`: no map, the
/// stand-in word in its place), the output rows and the columns.
#[cfg(feature = "gpu")]
struct Form<'a> {
    map: Option<&'a DeviceBuffer<u32>>,
    no_map: &'a DeviceBuffer<u32>,
    rows: usize,
    m: usize,
}

/// One launch of `form`'s shape into a `SENT`-filled y, read back.
#[cfg(feature = "gpu")]
fn launch(
    gpu: &Gpu,
    kern: &Q6kIdsKernels,
    w: &DeviceTensor<u32>,
    act: &Q8Act,
    form: Form<'_>,
) -> Result<Vec<f32>, GateError> {
    let s = gpu.stream();
    let mut y = DeviceBuffer::from_host(s, &vec![SENT; form.rows * form.m])?;
    kern.enqueue_gemv_q6k_ids(
        s,
        Q6kIdsArgs {
            w,
            act,
            map: form.map,
            no_map: form.no_map,
            rows: form.rows,
            m: form.m,
            y: &mut y,
            fault: gpu.unlabelled_sink(),
        },
    )?;
    s.synchronize()?;
    Ok(y.to_host_vec(s)?)
}

/// The reference over `ids` (`None`: the whole matrix): `enqueue_gemv_q6k`
/// on an upload of the gathered rows against `act`.
#[cfg(feature = "gpu")]
fn reference(
    gpu: &Gpu,
    words: &[u32],
    ids: Option<&[u32]>,
    rows: usize,
    act: &Q8Act,
) -> Result<Vec<f32>, GateError> {
    let s = gpu.stream();
    let mut y = DeviceBuffer::from_host(s, &vec![SENT; rows * act.m()])?;
    let mut stack = |rows_words: &[u32], rows: usize| -> Result<(), GateError> {
        let w = DeviceTensor::upload(s, rows_words, rows, WPR)?;
        gpu.enqueue_gemv_q6k(&w, act, &mut y)?;
        Ok(())
    };
    match ids {
        None => stack(words, ROWS)?,
        Some(ids) => {
            let mut gathered = Vec::with_capacity(rows * WPR);
            for &id in ids {
                gathered.extend_from_slice(&words[id as usize * WPR..(id as usize + 1) * WPR]);
            }
            stack(&gathered, rows)?;
        }
    }
    s.synchronize()?;
    Ok(y.to_host_vec(s)?)
}

/// The fixed-seed map of `len` sorted distinct ids below [`ROWS`].
#[cfg(feature = "gpu")]
fn map_of(len: usize) -> Vec<u32> {
    let mut s = SEED_MAP.wrapping_add(len as u32);
    let mut seen = std::collections::HashSet::with_capacity(len);
    let mut ids = Vec::with_capacity(len);
    while ids.len() < len {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let id = (s >> 8) % ROWS as u32;
        if seen.insert(id) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids
}

/// (a): every m and map against the reference, per column, and a rerun.
#[cfg(feature = "gpu")]
fn check_bits(
    gpu: &Gpu,
    kern: &Q6kIdsKernels,
    w: &DeviceTensor<u32>,
    words: &[u32],
    no_map: &DeviceBuffer<u32>,
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let mut ok = true;
    for m in [1usize, 2, 4, 8] {
        let x = activations(K, m, SEED_X + m as u32);
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut act = Q8Act::with_k(stream, m, K)?;
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        stream.synchronize()?;
        // The one-column activations, quantized from each column alone.
        let mut ones = Vec::with_capacity(m);
        for c in 0..m {
            let xd = DeviceBuffer::from_host(stream, &x[c * K..(c + 1) * K])?;
            let mut a = Q8Act::with_k(stream, 1, K)?;
            gpu.enqueue_quantize_q8_1(&xd, &mut a)?;
            ones.push(a);
        }
        stream.synchronize()?;
        let cases: Vec<(String, Option<Vec<u32>>)> = MAP_LENS
            .iter()
            .map(|&len| (format!("map{len}"), Some(map_of(len))))
            .chain([(format!("nomap{ROWS}"), None)])
            .collect();
        for (tag, ids) in &cases {
            let rows = ids.as_ref().map_or(ROWS, Vec::len);
            let ids_dev = ids
                .as_ref()
                .map(|v| DeviceBuffer::from_host(stream, v))
                .transpose()?;
            let map = ids_dev.as_ref();
            let y = launch(
                gpu,
                kern,
                w,
                &act,
                Form {
                    map,
                    no_map,
                    rows,
                    m,
                },
            )?;
            let again = launch(
                gpu,
                kern,
                w,
                &act,
                Form {
                    map,
                    no_map,
                    rows,
                    m,
                },
            )?;
            let want = reference(gpu, words, ids.as_deref(), rows, &act)?;
            let same = bits_equal(&y, &want);
            let rerun = bits_equal(&y, &again);
            // Each column against its own one-column launch, strided out of
            // the m-column y.
            let mut col_same = true;
            for (c, one) in ones.iter().enumerate() {
                let yc = launch(
                    gpu,
                    kern,
                    w,
                    one,
                    Form {
                        map,
                        no_map,
                        rows,
                        m: 1,
                    },
                )?;
                let stride: Vec<f32> = y[c..].iter().step_by(m).copied().collect();
                col_same &= bits_equal(&stride, &yc);
            }
            let pass = same && rerun && col_same;
            ok &= pass;
            println!(
                "q6k_ids[{tag}:m{m}] rows={rows} as_enqueue_gemv_q6k={same} \
                 bit_identical_rerun={rerun} each_column_the_m1_launch={col_same} {}{}{}",
                verdict(pass),
                mismatch("first_mismatch_vs_ref", &y, &want, m),
                mismatch("first_mismatch_rerun", &again, &y, m),
            );
        }
    }
    Ok(ok)
}

/// (b): the f64 band over the first [`BAND_ROWS`] rows of every map at m = 4
/// against `dequant_row` × the f32 input.
#[cfg(feature = "gpu")]
fn check_band(
    gpu: &Gpu,
    kern: &Q6kIdsKernels,
    w: &DeviceTensor<u32>,
    words: &[u32],
    no_map: &DeviceBuffer<u32>,
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let m = 4usize;
    let iters = K / 256 / 2;
    let x = activations(K, m, SEED_X + m as u32);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Act::with_k(stream, m, K)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    let (mut err_max, mut ref_max) = (0.0f64, 0.0f64);
    let mut over: Option<(usize, usize, f64, f64)> = None;
    let mut w_f32 = vec![0.0f32; K];
    let mut rb = vec![0u8; WPR * 4];
    for &len in MAP_LENS.iter() {
        let ids = map_of(len);
        let rows = BAND_ROWS.min(ids.len());
        let ids_dev = DeviceBuffer::from_host(stream, &ids[..rows])?;
        let y = launch(
            gpu,
            kern,
            w,
            &act,
            Form {
                map: Some(&ids_dev),
                no_map,
                rows,
                m,
            },
        )?;
        for r in 0..rows {
            rb.clear();
            words[ids[r] as usize * WPR..(ids[r] as usize + 1) * WPR]
                .iter()
                .for_each(|word| rb.extend_from_slice(&word.to_le_bytes()));
            dequant_row(GgmlType::Q6_K, &rb, &mut w_f32)?;
            for c in 0..m {
                let (mut exact, mut wx, mut quant) = (0.0f64, 0.0f64, 0.0f64);
                for (wb, xb) in w_f32.chunks(32).zip(x[c * K..(c + 1) * K].chunks(32)) {
                    let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                    let d = f64::from(amax) / 127.0;
                    for (&wi, &xi) in wb.iter().zip(xb) {
                        exact += f64::from(wi) * f64::from(xi);
                        wx += f64::from(wi) * f64::from(xi).abs();
                    }
                    quant += d / 2.0 * (1.0 + 256.0 * U) * sw_per_block(wb);
                }
                let bound = quant + gamma(2 * iters + 6) * (wx + quant);
                let at = r * m + c;
                let err = (f64::from(y[at]) - exact).abs();
                if err > bound && over.is_none() {
                    over = Some((r, c, err, bound));
                }
                err_max = err_max.max(err);
                ref_max = ref_max.max(exact.abs());
            }
        }
    }
    let rel = if ref_max > 0.0 {
        err_max / ref_max
    } else {
        0.0
    };
    let pass = over.is_none() && rel <= REL_PIN;
    println!(
        "q6k_band[m{m}] rows={BAND_ROWS} of each map bound=\"{BAND_DOC}\" max_abs={err_max:e} \
         max_ref={ref_max:e} rel={rel:e} pin={REL_PIN:e} {}{}",
        verdict(pass),
        over.map_or_else(String::new, |(r, c, e, b)| {
            format!(" first_over=row {r} column {c} err {e:e} bound {b:e}")
        }),
    );
    Ok(pass)
}

/// `Σ|w|` of one 32-value block.
#[cfg(feature = "gpu")]
fn sw_per_block(wb: &[f32]) -> f64 {
    wb.iter().map(|w| f64::from(*w).abs()).sum()
}

/// (c): the named fault, its NaN row, the other rows correct, and the clean
/// launch after the fault's read-and-clear.
#[cfg(feature = "gpu")]
fn check_fault(
    gpu: &Gpu,
    kern: &Q6kIdsKernels,
    w: &DeviceTensor<u32>,
    words: &[u32],
    no_map: &DeviceBuffer<u32>,
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let m = 4usize;
    let x = activations(K, m, SEED_X + 7081);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Act::with_k(stream, m, K)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    stream.synchronize()?;
    if gpu.take_fault()?.is_some() {
        println!(
            "q6k_fault the fault word held a fault before the case {}",
            verdict(false)
        );
        return Ok(false);
    }
    // Row 1 names an id past the matrix; the others are ordinary.
    let ids: Vec<u32> = vec![5, ROWS as u32 + 3, 7, 11, 2];
    let rows = ids.len();
    let ids_dev = DeviceBuffer::from_host(stream, &ids)?;
    let y = launch(
        gpu,
        kern,
        w,
        &act,
        Form {
            map: Some(&ids_dev),
            no_map,
            rows,
            m,
        },
    )?;
    let fault = gpu.take_fault()?;
    let want = Fault::at(LAYER_NONE, FaultSite::TokenId);
    let fault_ok = fault == Some(want);
    let nan_row = y[m..2 * m].iter().all(|v| v.is_nan());
    let mut clean = Vec::new();
    for (r, &id) in ids.iter().enumerate() {
        if (id as usize) < ROWS {
            clean.extend_from_slice(&y[r * m..(r + 1) * m]);
        }
    }
    let mut clean_want = Vec::new();
    {
        let good: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| (id as usize) < ROWS)
            .collect();
        let y_ref = reference(gpu, words, Some(&good), good.len(), &act)?;
        clean_want.extend_from_slice(&y_ref);
    }
    let others_ok = bits_equal(&clean, &clean_want);
    // The run after the fault's reset: a clean map raises nothing.
    let clean_ids: Vec<u32> = vec![5, 7, 11];
    let clean_dev = DeviceBuffer::from_host(stream, &clean_ids)?;
    launch(
        gpu,
        kern,
        w,
        &act,
        Form {
            map: Some(&clean_dev),
            no_map,
            rows: clean_ids.len(),
            m,
        },
    )?;
    let after = gpu.take_fault()?;
    let after_ok = after.is_none();
    let pass = fault_ok && nan_row && others_ok && after_ok;
    let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
    println!(
        "q6k_fault[m{m}] id_row=1 fault=\"{}\" want=\"{}\" nan_row={nan_row} \
         other_rows_bit_identical={others_ok} fault_after_reset=\"{}\" {}",
        shown(fault),
        shown(Some(want)),
        shown(after),
        verdict(pass),
    );
    Ok(pass)
}

/// (d): a captured launch replayed eager, following ids overwritten between
/// replays, its graph the one launch.
#[cfg(feature = "gpu")]
fn check_capture(
    gpu: &Gpu,
    kern: &Q6kIdsKernels,
    w: &DeviceTensor<u32>,
    words: &[u32],
    no_map: &DeviceBuffer<u32>,
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let m = 4usize;
    let x = activations(K, m, SEED_X + 7121);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Act::with_k(stream, m, K)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    stream.synchronize()?;
    let a = map_of(7);
    let b: Vec<u32> = a.iter().map(|&id| (id + 1) % ROWS as u32).collect();
    let rows = a.len();
    let mut ids_dev = DeviceBuffer::from_host(stream, &a)?;
    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * m])?;
    // Declared after the buffers it addresses, so it is dropped first.
    let graph = gpu.capture(|s| {
        kern.enqueue_gemv_q6k_ids(
            s,
            Q6kIdsArgs {
                w,
                act: &act,
                map: Some(&ids_dev),
                no_map,
                rows,
                m,
                y: &mut y,
                fault: gpu.unlabelled_sink(),
            },
        )
    })?;
    let nodes = graph.node_count();
    graph.launch(stream)?;
    stream.synchronize()?;
    let ya = y.to_host_vec(stream)?;
    ids_dev.copy_from_host(stream, &b)?; // host→device, OUTSIDE the graph
    graph.launch(stream)?;
    stream.synchronize()?;
    let yb = y.to_host_vec(stream)?;
    let ea = reference(gpu, words, Some(&a), rows, &act)?;
    let eb = reference(gpu, words, Some(&b), rows, &act)?;
    let a_same = bits_equal(&ya, &ea);
    let b_same = bits_equal(&yb, &eb);
    let fault_clean = gpu.take_fault()?.is_none();
    let pass = a_same && b_same && nodes == 1 && fault_clean;
    println!(
        "q6k_graph[m{m}] replay_a_bit_identical={a_same} replay_b_bit_identical={b_same} \
         graph_nodes={nodes} fault_after=\"{}\" {}{}{}",
        fault_clean,
        verdict(pass),
        mismatch("first_mismatch_a", &ya, &ea, m),
        mismatch("first_mismatch_b", &yb, &eb, m),
    );
    Ok(pass)
}

/// (e): the launcher's refusals, each its own `Shape` error.
#[cfg(feature = "gpu")]
fn check_refusals(
    gpu: &Gpu,
    kern: &Q6kIdsKernels,
    w: &DeviceTensor<u32>,
    no_map: &DeviceBuffer<u32>,
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let (rows, m) = (7usize, 4usize);
    let x = activations(K, m, SEED_X + 7331);
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Act::with_k(stream, m, K)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    stream.synchronize()?;
    let ids = map_of(rows);
    let ids_dev = DeviceBuffer::from_host(stream, &ids)?;
    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * m])?;
    // A 2-column activation, and one at K = 2,304: a multiple of 256 the
    // Q8Act takes but not of 512, so Q6_K rows would not start word-aligned.
    let x2 = activations(K, 2, SEED_X + 7451);
    let x2_dev = DeviceBuffer::from_host(stream, &x2)?;
    let mut two = Q8Act::with_k(stream, 2, K)?;
    gpu.enqueue_quantize_q8_1(&x2_dev, &mut two)?;
    let x23 = activations(2_304, m, SEED_X + 7561);
    let x23_dev = DeviceBuffer::from_host(stream, &x23)?;
    let mut odd = Q8Act::with_k(stream, m, 2_304)?;
    gpu.enqueue_quantize_q8_1(&x23_dev, &mut odd)?;
    stream.synchronize()?;
    let short_map = DeviceBuffer::from_host(stream, &ids[..3])?;
    let mut short_y = DeviceBuffer::from_host(stream, &vec![SENT; rows * m - 1])?;
    let cases: [(&str, Result<(), GpuError>); 6] = [
        ("m_0", {
            kern.enqueue_gemv_q6k_ids(
                stream,
                Q6kIdsArgs {
                    w,
                    act: &act,
                    map: Some(&ids_dev),
                    no_map,
                    rows,
                    m: 0,
                    y: &mut y,
                    fault: gpu.unlabelled_sink(),
                },
            )
        }),
        ("m_9", {
            kern.enqueue_gemv_q6k_ids(
                stream,
                Q6kIdsArgs {
                    w,
                    act: &act,
                    map: Some(&ids_dev),
                    no_map,
                    rows,
                    m: 9,
                    y: &mut y,
                    fault: gpu.unlabelled_sink(),
                },
            )
        }),
        ("m_past_the_acts_columns", {
            kern.enqueue_gemv_q6k_ids(
                stream,
                Q6kIdsArgs {
                    w,
                    act: &two,
                    map: Some(&ids_dev),
                    no_map,
                    rows,
                    m: 4,
                    y: &mut y,
                    fault: gpu.unlabelled_sink(),
                },
            )
        }),
        ("k_2304_not_a_multiple_of_512", {
            kern.enqueue_gemv_q6k_ids(
                stream,
                Q6kIdsArgs {
                    w,
                    act: &odd,
                    map: Some(&ids_dev),
                    no_map,
                    rows,
                    m,
                    y: &mut y,
                    fault: gpu.unlabelled_sink(),
                },
            )
        }),
        ("map_shorter_than_rows", {
            kern.enqueue_gemv_q6k_ids(
                stream,
                Q6kIdsArgs {
                    w,
                    act: &act,
                    map: Some(&short_map),
                    no_map,
                    rows,
                    m,
                    y: &mut y,
                    fault: gpu.unlabelled_sink(),
                },
            )
        }),
        ("y_shorter_than_rows_times_m", {
            kern.enqueue_gemv_q6k_ids(
                stream,
                Q6kIdsArgs {
                    w,
                    act: &act,
                    map: Some(&ids_dev),
                    no_map,
                    rows,
                    m,
                    y: &mut short_y,
                    fault: gpu.unlabelled_sink(),
                },
            )
        }),
    ];
    let mut ok = true;
    for (case, r) in cases {
        let pass = matches!(
            r,
            Err(GpuError::Shape {
                what: "enqueue_gemv_q6k_ids",
                ..
            })
        );
        let seen = match r {
            Ok(()) => "Ok (accepted)".to_string(),
            Err(e) => format!("Err: {e}"),
        };
        println!(
            "q6k_host[{case}] want=Err(Shape enqueue_gemv_q6k_ids) got={seen} {}",
            verdict(pass)
        );
        ok &= pass;
    }
    Ok(ok)
}

/// ` <label>=row R column C got 0x… want 0x…` at the first index where `got`
/// and `want` differ in bits, `m` columns a row; empty when they agree.
#[cfg(feature = "gpu")]
fn mismatch(label: &str, got: &[f32], want: &[f32], m: usize) -> String {
    if got.len() != want.len() {
        return format!(" {label}=length got {} want {}", got.len(), want.len());
    }
    match got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits())
    {
        Some(i) => format!(
            " {label}=row {} column {} got {:#010x} want {:#010x}",
            i / m,
            i % m,
            got[i].to_bits(),
            want[i].to_bits()
        ),
        None => String::new(),
    }
}

/// `rows` synthetic Q6_K rows of `n_sb` super-blocks as u32 words, from a
/// fixed-seed xorshift64: each super-block's 210 bytes are ql[128], qh[64],
/// 16 scale bytes and a `d` — a positive normal f16 (sign 0, exponent field
/// 15..=20, random mantissa), so no NaN or Inf enters. Every pattern of the
/// scale and quant bytes is a valid Q6_K block. `seed` must be nonzero.
#[cfg(feature = "gpu")]
fn synthetic_q6k(rows: usize, n_sb: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let half = |r: u64| -> u16 { (((15 + (r % 6)) as u16) << 10) | ((r >> 32) as u16 & 0x3ff) };
    let mut bytes = Vec::with_capacity(rows * n_sb * 210);
    for _ in 0..rows * n_sb {
        let d = half(next());
        // ql[128] @0, qh[64] @128, scales int8[16] @192 — random, every
        // pattern a valid block — then `d` f16 at 208.
        for _ in 0..208 {
            bytes.push((next() >> 32) as u8);
        }
        bytes.extend_from_slice(&d.to_le_bytes());
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}
