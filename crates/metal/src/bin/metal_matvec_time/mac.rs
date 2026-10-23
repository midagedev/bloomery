//! The mat-vec timing probe's body (macOS only).
//!
//! The method is the probe's (`tools/metal-probe/probe.swift`, `gemvRun`): a
//! matrix is copied into a working set of at least 1 GiB so that a run of
//! dispatches never reads one place twice inside the cache's reach; a command
//! buffer holds enough dispatches to read 768 MiB; two buffers warm the clocks
//! and `--trials` (16) are timed, all committed back to back; a dispatch's time
//! is its buffer's time over the dispatches in it.
//!
//! What differs is the clock. The crate's recording does not give out its
//! command buffer, so a buffer's end is the moment the stream's event recorded
//! after it fires, read by a host that spins on it; a buffer's time is the gap
//! between two such moments (the first timed buffer starts where the last warm
//! one ends). Every buffer is encoded before the first is committed, so the
//! host is never what the device waits for. The signal-only buffer between two
//! timed ones is the stream's own and is in every gap. The figure is the median
//! of the per-buffer rates (see [`Timing`]); `--gaps` prints every gap.
//!
//! Three launches are timed: the plain mat-vec of one tensor, the segmented
//! mat-vec of several tensors in one dispatch, and the head-normed residual
//! mat-vec. The f64 oracle's check of the first dispatch against the band runs
//! for the plain mat-vec only (`--no-check` skips it).
//!
//! Flags: `--only matvec|seg|headnorm|q4k|q6k`, `--trials N`, `--gaps`,
//! `--no-check` (skip the f64 oracle's check of the first dispatch against the
//! band).

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, Event, LAYER_NONE, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::{
    BLOCK_VALUES, HeadNormParams, Kernels, MatvecParams, Mode, NormOnLoad, PARTIALS_PER_HEAD,
    Quant, SegParams, Segment, VALUE_HEADS,
};

#[path = "../shared/oracle.rs"]
#[allow(
    dead_code,
    reason = "the probe reads the oracle's reference and band; the gate reads the rest"
)]
mod oracle;

/// The probe's working set, window, warm-up and trial counts.
const WORKING_SET_BYTES: usize = 1 << 30;
const WINDOW_BYTES: usize = 768 << 20;
const WARMUPS: usize = 2;
/// How long the timing loop spins on one buffer's event before it gives up
/// by name: the stream's own sync bound.
const EVENT_BOUND: Duration = Duration::from_secs(60);

/// The probe's shapes: (rows, columns) of an expert's down, a layer's
/// projections and the large ones.
const SHAPES: [(usize, usize); 5] = [
    (2048, 2048),
    (4096, 4096),
    (512, 2048),
    (1536, 4096),
    (8192, 4096),
];

/// What a timed dispatch reads and writes.
enum Shape {
    /// A plain mat-vec of one tensor.
    Matvec {
        quant: Quant,
        rows: usize,
        cols: usize,
    },
    /// A segmented mat-vec: the row segments of several tensors of one type
    /// and one width, in one dispatch.
    Seg {
        quant: Quant,
        cols: usize,
        /// The rows of every segment.
        segments: Vec<usize>,
    },
    /// The head-normed residual mat-vec: a gated delta net's out projection.
    HeadNorm { rows: usize, cols: usize },
}

impl Shape {
    /// The name the line prints.
    fn kind(&self) -> &'static str {
        match self {
            Self::Matvec { .. } => "matvec",
            Self::Seg { .. } => "seg",
            Self::HeadNorm { .. } => "headnorm",
        }
    }

    /// The weight type the dispatch reads.
    fn quant(&self) -> Quant {
        match self {
            Self::Matvec { quant, .. } | Self::Seg { quant, .. } => *quant,
            Self::HeadNorm { .. } => Quant::Q4K,
        }
    }

    /// The columns the dispatch reads.
    fn cols(&self) -> usize {
        match self {
            Self::Matvec { cols, .. } | Self::Seg { cols, .. } => *cols,
            Self::HeadNorm { cols, .. } => *cols,
        }
    }

    /// The rows the dispatch writes.
    fn rows(&self) -> usize {
        match self {
            Self::Matvec { rows, .. } => *rows,
            Self::Seg { segments, .. } => segments.iter().sum(),
            Self::HeadNorm { rows, .. } => *rows,
        }
    }

    /// The bytes a dispatch streams: its weights.
    fn bytes(&self) -> usize {
        match self {
            Self::Matvec { quant, rows, cols } => rows * cols / BLOCK_VALUES * quant.block_bytes(),
            Self::Seg {
                quant,
                cols,
                segments,
            } => segments.iter().sum::<usize>() * cols / BLOCK_VALUES * quant.block_bytes(),
            Self::HeadNorm { rows, cols } => rows * cols / BLOCK_VALUES * Quant::Q4K.block_bytes(),
        }
    }
}

/// The shapes of the segmented and the head-normed launches: a layer's
/// projections, and the out projection of a gated delta net.
fn other_shapes() -> Vec<(&'static str, Shape)> {
    vec![
        (
            "seg delta qkv+gate",
            Shape::Seg {
                quant: Quant::Q4K,
                cols: 2048,
                segments: vec![8192, 4096],
            },
        ),
        (
            "seg attn q+gate+k+v",
            Shape::Seg {
                quant: Quant::Q4K,
                cols: 2048,
                segments: vec![8192, 4096, 512, 512],
            },
        ),
        (
            "seg q6k qkv",
            Shape::Seg {
                quant: Quant::Q6K,
                cols: 2048,
                segments: vec![8192],
            },
        ),
        (
            "headnorm out",
            Shape::HeadNorm {
                rows: 2048,
                cols: 4096,
            },
        ),
    ]
}

/// One matrix of random blocks the probe's way: random bytes, with `d` (and
/// `dmin`) a finite f16 in [0.0005, 0.004].
fn matrix(quant: Quant, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let block = quant.block_bytes();
    let blocks = rows * cols / BLOCK_VALUES;
    let mut rng = oracle::Rng::new(seed);
    let mut bytes = vec![0u8; blocks * block];
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&word[..n]);
    }
    let mut small = || {
        let unit = (rng.signed() + 1.0) / 2.0;
        gguf::quant::f32_to_f16_bits(0.0005 + 0.0035 * unit).to_le_bytes()
    };
    for b in bytes.chunks_exact_mut(block) {
        match quant {
            Quant::Q4K => {
                b[0..2].copy_from_slice(&small());
                b[2..4].copy_from_slice(&small());
            }
            Quant::Q6K => b[208..210].copy_from_slice(&small()),
        }
    }
    bytes
}

/// One shape's timing: bytes per second over a dispatch, and every timed
/// buffer's gap. The median of the per-buffer rates is the figure: the host
/// that spins on the events wakes late at times, and a late wake stamps two
/// neighbouring buffers at one moment (a gap near zero beside a gap of two
/// buffers), which the median ignores and a mean over the span would not.
struct Timing {
    dispatches: usize,
    copies: usize,
    working_set: usize,
    median: f64,
    /// The gaps within a quarter of the median gap: the buffers read cleanly.
    clean: usize,
    /// Every timed buffer's gap, in seconds.
    gaps: Vec<f64>,
    /// The host's time to encode one dispatch, in seconds.
    encode: f64,
}

fn time_shape(
    device: &MetalDevice,
    kernels: &Kernels,
    shape: &Shape,
    trials: usize,
    check: bool,
) -> Result<Timing, String> {
    let stream = device.stream();
    let (quant, cols, rows) = (shape.quant(), shape.cols(), shape.rows());
    let seed = 0x6B70_726F_6265 ^ ((rows as u64) << 24) ^ ((cols as u64) << 4);
    // The weights of the dispatch: one matrix, or the segments' matrices side
    // by side.
    let one: Vec<u8> = match shape {
        Shape::Matvec { rows, cols, .. } => matrix(quant, *rows, *cols, seed),
        Shape::Seg { cols, segments, .. } => {
            let mut all = Vec::new();
            for (i, s) in segments.iter().enumerate() {
                all.extend_from_slice(&matrix(quant, *s, *cols, seed ^ (i as u64)));
            }
            all
        }
        Shape::HeadNorm { rows, cols } => matrix(Quant::Q4K, *rows, *cols, seed),
    };
    let bytes = one.len();
    let stride = bytes.div_ceil(256) * 256;
    let copies = WORKING_SET_BYTES.div_ceil(stride).max(1);
    let mut all = vec![0u8; copies * stride];
    for c in 0..copies {
        all[c * stride..c * stride + bytes].copy_from_slice(&one);
    }
    let weights = device.upload(stream, &all).map_err(why)?;
    drop(all);
    let x_values = oracle::activations(cols, 0x78, 1.0);
    let x = device.upload(stream, &x_values).map_err(why)?;
    // The norm's gain, and the head norm's gated numerator and partial sums.
    let gain = device.upload(stream, &vec![1.0f32; cols]).map_err(why)?;
    let g = device.upload(stream, &x_values).map_err(why)?;
    let partials = device
        .upload(stream, &vec![1.0f32; VALUE_HEADS * PARTIALS_PER_HEAD])
        .map_err(why)?;
    // The segments' places in the weights and in the output.
    let mut segments: Vec<Segment> = Vec::new();
    if let Shape::Seg {
        cols,
        segments: rows_of,
        ..
    } = shape
    {
        let mut weight_blocks = 0usize;
        let mut out_offset = 0usize;
        for r in rows_of {
            segments.push(Segment {
                cols: *cols,
                rows: *r,
                weight_blocks,
                out_offset,
            });
            weight_blocks += r * (cols / BLOCK_VALUES);
            out_offset += r;
        }
    }

    let dispatches = WINDOW_BYTES.div_ceil(bytes).max(30);
    let buffers = WARMUPS + trials;
    let mut y = device
        .zeroed::<f32>(stream, Elems(buffers * dispatches * rows))
        .map_err(why)?;

    // Encode every buffer, then commit them back to back, an event after each.
    let encode_start = Instant::now();
    let mut recordings = Vec::with_capacity(buffers);
    for b in 0..buffers {
        let mut recording = stream.begin().map_err(why)?;
        {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            for j in 0..dispatches {
                let n = b * dispatches + j;
                let w = weights
                    .window(Elems((n % copies) * stride), Elems(bytes))
                    .map_err(why)?;
                let out = y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?;
                match shape {
                    Shape::Matvec { quant, rows, cols } => {
                        kernels
                            .matvec(
                                &mut encoder,
                                MatvecParams {
                                    quant: *quant,
                                    rows: *rows,
                                    cols: *cols,
                                    weights: w,
                                    x: x.window(Elems(0), Elems(*cols)).map_err(why)?,
                                    y: out,
                                    mode: Mode::Store,
                                },
                            )
                            .map_err(why)?;
                    }
                    Shape::Seg { quant, cols, .. } => {
                        kernels
                            .matvec_seg(
                                &mut encoder,
                                SegParams {
                                    quant: *quant,
                                    segments: segments.clone(),
                                    weights: w,
                                    x: x.window(Elems(0), Elems(*cols)).map_err(why)?,
                                    y: out,
                                    norm: NormOnLoad {
                                        gain: gain.window(Elems(0), Elems(*cols)).map_err(why)?,
                                        eps: 1e-6,
                                        fault: device.fault_word(),
                                        layer: LAYER_NONE,
                                    },
                                },
                            )
                            .map_err(why)?;
                    }
                    Shape::HeadNorm { rows, cols } => {
                        kernels
                            .headnorm_resid(
                                &mut encoder,
                                HeadNormParams {
                                    rows: *rows,
                                    cols: *cols,
                                    weights: w,
                                    g: g.window(Elems(0), Elems(*cols)).map_err(why)?,
                                    partials: partials
                                        .window(Elems(0), Elems(VALUE_HEADS * PARTIALS_PER_HEAD))
                                        .map_err(why)?,
                                    y: out,
                                    eps: 1e-6,
                                    fault: device.fault_word(),
                                    layer: LAYER_NONE,
                                },
                            )
                            .map_err(why)?;
                    }
                }
            }
            encoder.end();
        }
        recordings.push(recording);
    }
    let encode = encode_start.elapsed().as_secs_f64() / (buffers * dispatches) as f64;
    let mut events = Vec::with_capacity(buffers);
    for recording in recordings {
        stream.submit(recording).map_err(why)?;
        events.push(stream.record().map_err(why)?);
    }
    let mut ends = Vec::with_capacity(buffers);
    for event in &events {
        let deadline = Instant::now() + EVENT_BOUND;
        while !event.done().map_err(why)? {
            if Instant::now() >= deadline {
                // A failed command buffer stops the signal: its error first.
                stream.sync().map_err(why)?;
                return Err(format!(
                    "event {} was not signalled within {} s",
                    event.value(),
                    EVENT_BOUND.as_secs()
                ));
            }
            std::hint::spin_loop();
        }
        ends.push(Instant::now());
    }
    stream.sync().map_err(why)?;

    let mut out = vec![0f32; buffers * dispatches * rows];
    device
        .read(stream, y.whole().map_err(why)?, &mut out)
        .map_err(why)?;
    let first = &out[..rows];
    for (n, slice) in out.chunks_exact(rows).enumerate() {
        if slice
            .iter()
            .zip(first)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(format!(
                "dispatch {n} read the same weights and gave other bits than dispatch 0"
            ));
        }
    }
    if check && matches!(shape, Shape::Matvec { .. }) {
        let x64: Vec<f64> = x_values.iter().map(|&v| f64::from(v)).collect();
        let reference = oracle::reference(quant, &one, rows, cols, &x64)?;
        let band = oracle::band(quant, cols / BLOCK_VALUES, oracle::Kind::Store);
        oracle::check(first, &reference, band, oracle::Kind::Store, None)?;
    }

    let gaps: Vec<f64> = (WARMUPS..buffers)
        .map(|b| ends[b].duration_since(ends[b - 1]).as_secs_f64())
        .collect();
    let per_buffer: Vec<f64> = gaps
        .iter()
        .map(|&g| bytes as f64 * dispatches as f64 / g / 1e9)
        .collect();
    let median_gbps = median(&per_buffer);
    let median_gap = median(&gaps);
    let clean = gaps
        .iter()
        .filter(|&&g| (g - median_gap).abs() <= 0.25 * median_gap)
        .count();
    Ok(Timing {
        dispatches,
        copies,
        working_set: copies * stride,
        median: median_gbps,
        clean,
        gaps,
        encode,
    })
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The most timed buffers: every buffer, warm-ups included, is encoded before
/// any is committed, and each takes a signal buffer beside it, so both must fit
/// the default queue's 64 command buffers or `begin` blocks.
const MAX_TRIALS: usize = 64 / 2 - WARMUPS;

/// The `--trials` value: a whole number in `1..=MAX_TRIALS`, refused by name
/// otherwise (no trial has no median to take; more would block the queue).
fn trials_of(value: Option<&String>) -> Result<usize, String> {
    match value.map(|v| (v, v.parse::<usize>())) {
        None => Err("--trials takes a value".to_string()),
        Some((_, Ok(n))) if (1..=MAX_TRIALS).contains(&n) => Ok(n),
        Some((value, _)) => Err(format!(
            "--trials takes a whole number in 1..={MAX_TRIALS}, not {value:?}"
        )),
    }
}

/// The `--only` value: one of the probe's weight types, refused by name
/// otherwise (a selection that matches no shape would skip every shape and
/// exit as a success).
fn only_of(value: Option<&String>) -> Result<&str, String> {
    let Some(value) = value else {
        return Err("--only takes a value".to_string());
    };
    if [Quant::Q4K, Quant::Q6K]
        .iter()
        .any(|q| q.name() == value.as_str())
    {
        Ok(value)
    } else {
        Err(format!(
            "--only takes {} or {}, not {value:?}",
            Quant::Q4K.name(),
            Quant::Q6K.name()
        ))
    }
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trials = match args
        .iter()
        .position(|a| a == "--trials")
        .map_or(Ok(16), |at| trials_of(args.get(at + 1)))
    {
        Ok(trials) => trials,
        Err(e) => {
            println!("metal_matvec_time: {e}");
            return ExitCode::from(64);
        }
    };
    let only = match args
        .iter()
        .position(|a| a == "--only")
        .map_or(Ok(None), |at| only_of(args.get(at + 1)).map(Some))
    {
        Ok(only) => only,
        Err(e) => {
            println!("metal_matvec_time: {e}");
            return ExitCode::from(64);
        }
    };
    let check = !args.iter().any(|a| a == "--no-check");
    let show_gaps = args.iter().any(|a| a == "--gaps");

    let device = match MetalDevice::open() {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("metal_matvec_time: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let kernels = match Kernels::new(&device) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let info = match device.info() {
        Ok(info) => info,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "info\tdevice={}\tlib={}\tbackend={}\ttrials={trials}\twarmups={WARMUPS}\twindow_bytes={WINDOW_BYTES}\tworking_set_bytes={WORKING_SET_BYTES}",
        info.name,
        device.library().path().name(),
        <MetalDevice as Device>::BACKEND
    );
    let wanted = |shape: &Shape, label: &str| -> bool {
        match only {
            None => true,
            Some(o) => {
                o == shape.kind()
                    || (shape.kind() == "matvec" && o == shape.quant().name())
                    || label.contains(o)
            }
        }
    };
    let mut failed = false;
    let mut shapes: Vec<(String, Shape)> = Vec::new();
    for quant in [Quant::Q4K, Quant::Q6K] {
        for (rows, cols) in SHAPES {
            shapes.push((
                format!("matvec {} {rows}x{cols}", quant.name()),
                Shape::Matvec { quant, rows, cols },
            ));
        }
    }
    for (label, shape) in other_shapes() {
        shapes.push((label.to_string(), shape));
    }
    for (label, shape) in &shapes {
        if !wanted(shape, label) {
            continue;
        }
        match time_shape(&device, &kernels, shape, trials, check) {
            Ok(t) => {
                if show_gaps {
                    let us: Vec<String> =
                        t.gaps.iter().map(|g| format!("{:.0}", g * 1e6)).collect();
                    println!("gaps_us\t{}", us.join(" "));
                }
                let bytes = shape.bytes();
                println!(
                    "matvec\tkind={}\tshape={label}\ttype={}\trows={}\tcols={}\tdispatches={}\ttrials={trials}\tclean={}\tbytes={bytes}\tcopies={}\tworking_set_bytes={}\tgbps_median={:.2}\tus_median={:.2}\tencode_us={:.2}",
                    shape.kind(),
                    shape.quant().name(),
                    shape.rows(),
                    shape.cols(),
                    t.dispatches,
                    t.clean,
                    t.copies,
                    t.working_set,
                    t.median,
                    bytes as f64 / t.median / 1e3,
                    t.encode * 1e6,
                );
            }
            Err(e) => {
                println!("matvec\tkind={}\tshape={label}\tFAIL\t{e}", shape.kind());
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
