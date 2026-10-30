//! The mat-vec timing probe's body (macOS only).
//!
//! The method is the probe's (`tools/metal-probe/probe.swift`, `gemvRun`): a
//! matrix is copied into a working set of at least 1 GiB so that a run of
//! dispatches never reads one place twice inside the cache's reach; a command
//! buffer holds enough dispatches to read 768 MiB; two buffers warm the clocks
//! and `--trials` (16) are timed, all committed back to back; a dispatch's time
//! is its buffer's time over the dispatches in it.
//!
//! What differs is the clock. Each timed command buffer's time is its own
//! `GPUEndTime` minus `GPUStartTime`, read after the buffer has completed.
//! Those stamps belong to the GPU, so a host that arrives late, or was busy
//! while the GPU ran, still reads the same interval. A row fails for a GPU
//! reason: the buffer failed, it was still running when the wait ended, or
//! the interval is not a positive finite duration. The figure is the median
//! of the per-buffer rates (see [`Timing`]); `--gaps` prints every timed
//! buffer's GPU time.
//!
//! This binary is a development-loop instrument. A figure it prints is this
//! machine's, taken with no timing lease, and is not a published number.
//!
//! Five launches are timed: the plain mat-vec of one tensor, the segmented
//! mat-vec of several tensors in one dispatch, the head-normed residual
//! mat-vec, the router (`router35`, 257 rows of K 2048 f32, top-8) and the
//! head (`q6k_head`, 248320 rows of K 2048 Q6_K: the logits dispatch, a
//! barrier and the pick). The f64 oracle's
//! check of the first dispatch against the band runs for the plain mat-vec
//! only (`--no-check` skips it). The router and the head run on synthetic
//! inputs of that shape; they open no model file.
//!
//! Each row is one line, tab-separated: `kernel`, `shape`, `arm`, the GPU
//! microseconds per dispatch (`us_median` and `us_p90` over the repeats) and
//! `gbps`. The `window` arm is one command buffer of enough dispatches to
//! read the probe's window, and its time is the buffer's time over those
//! dispatches. `--chain N` adds three arms of each selected kernel. `lone`
//! is one dispatch in its own command buffer. `serial` is N dispatches in
//! one serial encoder with a memory barrier between each. `concurrent` is
//! the same N dispatches in one concurrent encoder, again with a barrier
//! between each. `cb_us` on those two is (chain buffer time − N × lone
//! time) / N, from the medians.
//!
//! Beside the router and the head-normed mat-vec, the probe times their
//! phase cuts: a copy of the shipped source with one counted text edit, the
//! way the kernels gate compiles its mutants, compiled by the runtime and
//! launched by the same host module. A row whose shape ends in `[name]` is
//! that cut: the router without its fused norm's input pass and without its
//! tail, the head-normed mat-vec with the scales precomputed. The cut is a
//! development instrument, not a kernel: the edit's find text must occur
//! exactly as often as named, so a kernel edit that moves it is a named
//! error, not a silent no-op.
//!
//! Router and head rows also carry the time their weight bytes take at the
//! stream bandwidth measured in this run — the `window` arm of `q4k
//! 8192x4096`, run for that purpose when the selection does not already
//! time it — and the review's predicted tail span. The measured tail of a
//! row is its `us_median` minus that byte time. When that arm has no
//! bandwidth, the row still prints its own microseconds and marks the byte
//! time unavailable, by name.
//!
//! The routed experts are timed beside the mat-vecs: the gate-up pair
//! (`q4k_pair_swiglu`, nine slots of 512 by 2048, the session's gate_up) and
//! the folded down (`q4k_down_fold`, 2048 rows of the slots' 512, the
//! session's down), at eight routed experts and the shared one, on synthetic
//! stacks. `--only pair` / `--only fold` select them.
//!
//! `--only lfm3` times LFM3-1.7B's decode launches instead (`lfm3.rs`): the
//! session's own host calls over A4 synthetic weights, on a device opened at
//! LFM3's instance, with the diagnostic rows and source cuts that split the
//! gap to MLX's `qmv` into terms. It runs no other row. `--only d27` times
//! Qwen3.8-27B's the same way (`d27.rs`), with each normed row's scaled twin.
//!
//! `--geometry rows=N,sg=M,router_sg=K` compiles the kernels' own source with
//! other values of the geometry constants — the rows a SIMD group owns
//! (2/4/8, a `#define` of the source), the SIMD groups in a mat-vec
//! threadgroup (the kernels' function constant) and the router's own group
//! count (another `#define`) — and launches every arm through that variant,
//! so one binary sweeps the geometry: the rows of a run under a variant carry
//! `[rows=N sg=M router_sg=K]` in their shape label. The pair and the fold
//! follow the mat-vec geometry: a pair's gate and up rows are two of the
//! geometry's rows, the fold owns the geometry's rows a SIMD group.
//!
//! Flags: `--only` (repeatable; comma-separated `matvec`, `seg`, `headnorm`,
//! `pair`, `fold`, `q4k`, `q6k`, `a4`, `router`, `head`, or a shape label),
//! `--trials N`, `--chain N`, `--geometry rows=N,sg=M,router_sg=K`, `--gaps`,
//! `--no-check`, `--help`.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord, LAYER_NONE, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::instance::HOST;
use bloomery_metal::kernels::router::{self, RouterParams};
use bloomery_metal::kernels::{
    EXPERT_K, EXPERT_SLOTS, FoldParams, HEAD_K, HEAD_SIMD_GROUPS, HeadNormParams, HeadParams,
    Kernels, LoadNorm, MatvecParams, Mode, PARTIALS_PER_HEAD, PairParams, Quant, ROUTED_IDS,
    ROWS_PER_SIMD_GROUP, SIMD_GROUPS_PER_GROUP, SegParams, Segment, SharedPlan, VALUE_HEADS,
};
use bloomery_metal::library::{Library, SOURCE};
use bloomery_metal::{GpuInterval, MetalStream};

#[path = "../shared/oracle.rs"]
#[allow(
    dead_code,
    reason = "the probe reads the oracle's reference and band; the gate reads the rest"
)]
mod oracle;

#[path = "../shared/synthetic.rs"]
mod synthetic;
use synthetic::matrix;

#[path = "lfm3.rs"]
mod lfm3;

#[path = "d27.rs"]
mod d27;

/// The probe's working set, window, warm-up and trial counts.
const WORKING_SET_BYTES: usize = 1 << 30;
const WINDOW_BYTES: usize = 768 << 20;
const WARMUPS: usize = 2;
/// The heat a window arm of a shape row ([`time_shape`]: the model rows of
/// `lfm3.rs` and `d27.rs` among them) commits ahead of its warm-ups: its own dispatches, enough to read `HEAT_BYTES` in
/// `HEAT_BUFFERS` command buffers, so the GPU runs the timed buffers at the
/// clock a sustained stream holds and not at the one it idled down to while
/// the host built the arm's weights. A small shape's heat stops at
/// `HEAT_MAX` dispatches.
const HEAT_BYTES: usize = 96 << 30;
const HEAT_BUFFERS: usize = 4;
const HEAT_MAX: usize = 16_384;

/// The heat dispatches of `arm` over a shape of `bytes`: none for a chain
/// arm, whose short buffers are the point.
fn heat_of(arm: &Arm, bytes: usize) -> usize {
    if arm.dispatches.is_some() {
        return 0;
    }
    HEAT_BYTES.div_ceil(bytes).min(HEAT_MAX)
}
/// How long the host waits for one timed command buffer. A buffer still
/// running when this ends is a named failure, and later arms are not
/// submitted behind it. The wait only gates the read of the GPU stamps; it
/// is not itself the measurement.
const BUFFER_WAIT: Duration = Duration::from_secs(30);

/// The most dispatches `--chain` accepts. Past this the head's per-dispatch
/// outputs stop fitting a development run.
const CHAIN_MAX: usize = 256;

/// The router's weight bytes: 257 rows of K 2048 f32. The design's per-layer
/// matrix, and the kernel's own shape.
const ROUTER_BYTES: usize = router::ROWS * router::K * core::mem::size_of::<f32>();

/// The timed head's rows: Qwen3.6's output matrix. The launch takes any row
/// count; this is the shape the design's head is timed at.
const HEAD_ROWS: usize = 248_320;

/// The head's weight bytes: 248320 rows of K 2048 Q6_K. The design's head
/// matrix, and the kernel's own shape; the block bytes are gguf's row.
fn head_bytes() -> usize {
    HEAD_ROWS * (HEAD_K / Quant::Q6K.values_per_block()) * Quant::Q6K.block_bytes()
}

/// The review's predicted tail span for one router dispatch, in the row's
/// microsecond fields.
const ROUTER_TAIL_US: (f64, f64) = (11.0, 123.0);

/// Whether this build's geometry is the one the predictions above were derived at: Qwen3.6's
/// router (257 rows of K 2048, top-8) and head K 2048. A build of another host refuses at `main`.
const PREDICTED_GEOMETRY: bool =
    router::ROWS == 257 && router::K == 2048 && router::TOP_K_MAX == 8 && HEAD_K == 2048;

const _: () =
    assert!(!PREDICTED_GEOMETRY || (ROUTER_BYTES == 2_105_344 && ROUTER_BYTES.is_multiple_of(256)));
const _: () = assert!(HEAD_ROWS.is_multiple_of(SIMD_GROUPS_PER_GROUP * ROWS_PER_SIMD_GROUP));

/// How one kernel is launched into a timed command buffer.
struct Arm {
    name: &'static str,
    /// Dispatches in the buffer. `None` fills the probe's window.
    dispatches: Option<usize>,
    concurrent: bool,
    /// A memory barrier before every dispatch after the first.
    barrier: bool,
}

fn window_arm() -> Arm {
    Arm {
        name: "window",
        dispatches: None,
        concurrent: false,
        barrier: false,
    }
}

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
        /// Whether the launch adds into its output: the residual epilogue.
        resid: bool,
    },
    /// A segmented mat-vec: the row segments of several tensors of one type
    /// and one width, in one dispatch.
    Seg {
        quant: Quant,
        cols: usize,
        /// The rows of every segment.
        segments: Vec<usize>,
        /// The input is already `x · gain` and the norm's scale is one word
        /// (the scaled entry, what a hand-on feeds), else the norm on load.
        scaled: bool,
    },
    /// The head-normed residual mat-vec: a gated delta net's out projection.
    HeadNorm { rows: usize, cols: usize },
    /// A dense FFN's gate-up pair at its one slot: the experts' launch over
    /// one gate and one up of `rows` by `cols`, the weights holding the gate
    /// and then the up. The launch refuses a `rows` that is not the device's
    /// instance's expert width.
    DensePair {
        quant: Quant,
        rows: usize,
        cols: usize,
        /// The scaled entry, as the segmented mat-vec's.
        scaled: bool,
    },
    /// A dense FFN's folded down at its one slot: `rows` by `k`, added into
    /// the output, the route weight 1.
    DenseFold { quant: Quant, rows: usize, k: usize },
    /// The output head: `rows` logits over `cols`, the norm on the load.
    Head {
        quant: Quant,
        rows: usize,
        cols: usize,
        part: HeadPart,
        /// The scaled entry, as the segmented mat-vec's.
        scaled: bool,
    },
}

/// What of the output head a row launches.
#[derive(Clone, Copy)]
enum HeadPart {
    /// The logits dispatch, the barrier and the pick.
    Pair,
    /// The pick alone, over the zero logits the output buffer holds: the
    /// bytes it streams are the logits'.
    Pick,
}

impl Shape {
    /// The name the line prints.
    fn kind(&self) -> &'static str {
        match self {
            Self::Matvec { .. } => "matvec",
            Self::Seg { .. } => "seg",
            Self::HeadNorm { .. } => "headnorm",
            Self::DensePair { .. } => "pair",
            Self::DenseFold { .. } => "fold",
            Self::Head { .. } => "head",
        }
    }

    /// The weight type the dispatch reads.
    fn quant(&self) -> Quant {
        match self {
            Self::Matvec { quant, .. }
            | Self::Seg { quant, .. }
            | Self::DensePair { quant, .. }
            | Self::DenseFold { quant, .. }
            | Self::Head { quant, .. } => *quant,
            Self::HeadNorm { .. } => Quant::Q4K,
        }
    }

    /// The columns the dispatch reads.
    fn cols(&self) -> usize {
        match self {
            Self::Matvec { cols, .. }
            | Self::Seg { cols, .. }
            | Self::DensePair { cols, .. }
            | Self::Head { cols, .. } => *cols,
            Self::HeadNorm { cols, .. } => *cols,
            Self::DenseFold { k, .. } => *k,
        }
    }

    /// The rows the dispatch writes.
    fn rows(&self) -> usize {
        match self {
            Self::Matvec { rows, .. }
            | Self::DensePair { rows, .. }
            | Self::DenseFold { rows, .. }
            | Self::Head { rows, .. } => *rows,
            Self::Seg { segments, .. } => segments.iter().sum(),
            Self::HeadNorm { rows, .. } => *rows,
        }
    }

    /// The bytes a dispatch streams: its weights.
    fn bytes(&self) -> usize {
        let per =
            |quant: Quant, values: usize| values / quant.values_per_block() * quant.block_bytes();
        match self {
            Self::Matvec {
                quant, rows, cols, ..
            } => per(*quant, rows * cols),
            Self::Seg {
                quant,
                cols,
                segments,
                ..
            } => per(*quant, segments.iter().sum::<usize>() * cols),
            Self::HeadNorm { rows, cols } => per(Quant::Q4K, rows * cols),
            Self::DensePair {
                quant, rows, cols, ..
            } => 2 * per(*quant, rows * cols),
            Self::DenseFold { quant, rows, k } => per(*quant, rows * k),
            Self::Head {
                quant,
                rows,
                cols,
                part: HeadPart::Pair,
                ..
            } => per(*quant, rows * cols),
            Self::Head {
                rows,
                part: HeadPart::Pick,
                ..
            } => rows * core::mem::size_of::<f32>(),
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
                scaled: false,
            },
        ),
        (
            "seg attn q+gate+k+v",
            Shape::Seg {
                quant: Quant::Q4K,
                cols: 2048,
                segments: vec![8192, 4096, 512, 512],
                scaled: false,
            },
        ),
        (
            "seg q6k qkv",
            Shape::Seg {
                quant: Quant::Q6K,
                cols: 2048,
                segments: vec![8192],
                scaled: false,
            },
        ),
        (
            "headnorm out",
            Shape::HeadNorm {
                rows: 2048,
                cols: 4096,
            },
        ),
        // The attention output projection's shape: the plain Q4_K residual
        // mat-vec beside the head-normed one, the same weights stream.
        (
            "matvec q4k 2048x4096 resid",
            Shape::Matvec {
                quant: Quant::Q4K,
                rows: 2048,
                cols: 4096,
                resid: true,
            },
        ),
    ]
}

/// One shape's timing. `gaps` are the timed buffers' own GPU intervals, in
/// seconds. The median of the per-buffer rates is the figure.
struct Timing {
    dispatches: usize,
    copies: usize,
    working_set: usize,
    /// The median of the per-buffer rates, in GB/s.
    median: f64,
    /// The median of the per-dispatch times, in microseconds.
    us_median: f64,
    /// The 90th percentile of the per-dispatch times, nearest rank, in
    /// microseconds.
    us_p90: f64,
    /// The gaps within a quarter of the median gap: the buffers read cleanly.
    clean: usize,
    /// Every timed buffer's GPU interval, in seconds.
    gaps: Vec<f64>,
    /// The host's time to encode one dispatch, in seconds.
    encode: f64,
}

/// The bytes of one copy, aligned up to 256, and how many copies fill the
/// working set.
fn spread(bytes: usize) -> (usize, usize, usize) {
    let stride = bytes.div_ceil(256) * 256;
    let copies = WORKING_SET_BYTES.div_ceil(stride).max(1);
    (stride, copies, copies * stride)
}

fn dispatch_count(arm: &Arm, bytes: usize) -> usize {
    arm.dispatches
        .unwrap_or_else(|| WINDOW_BYTES.div_ceil(bytes).max(30))
}

/// What one timed arm launches. The closure that encodes a dispatch is the
/// work; this is the description of it.
struct TimedWork<'a> {
    /// The weight bytes one dispatch reads.
    bytes: usize,
    /// Dispatches encoded into one command buffer.
    dispatches: usize,
    /// Copies of the weights in the working set.
    copies: usize,
    /// Bytes of the working set those copies fill.
    working_set_bytes: usize,
    /// Timed command buffers after the warm-up.
    trials: usize,
    arm: &'a Arm,
    /// Dispatches of the heat ahead of the warm-ups: `encode` gets them as
    /// indices `0..heat` and the buffers' dispatches after them, so every
    /// index is one output written once.
    heat: usize,
}

/// Encode every buffer, commit them back to back, and read each timed
/// buffer's GPU interval. `encode` launches dispatch `n` (counted across
/// the buffers).
fn time_dispatches(
    stream: &MetalStream,
    work: &TimedWork<'_>,
    mut encode: impl FnMut(&mut ComputeEncoder<'_>, usize) -> Result<(), String>,
) -> Result<Timing, String> {
    if work.dispatches == 0 || work.trials == 0 {
        return Err(format!(
            "a timed arm needs at least one dispatch and one repeat, not {} and {}",
            work.dispatches, work.trials
        ));
    }
    let buffers = WARMUPS + work.trials;
    let per_heat = work.heat.div_ceil(HEAT_BUFFERS);
    for h in 0..HEAT_BUFFERS {
        let first = h * per_heat;
        let last = work.heat.min(first + per_heat);
        if first >= last {
            break;
        }
        let mut recording = stream.begin().map_err(why)?;
        {
            let mut encoder = if work.arm.concurrent {
                recording.compute_encoder_concurrent().map_err(why)?
            } else {
                recording.compute_encoder().map_err(why)?
            };
            for n in first..last {
                if work.arm.barrier && n > first {
                    encoder.memory_barrier();
                }
                encode(&mut encoder, n)?;
            }
            encoder.end();
        }
        stream.submit(recording).map_err(why)?;
    }
    let encode_start = Instant::now();
    let mut pending = Vec::with_capacity(buffers);
    for b in 0..buffers {
        let mut recording = stream.begin().map_err(why)?;
        let stamps = recording.stamps();
        {
            let mut encoder = if work.arm.concurrent {
                recording.compute_encoder_concurrent().map_err(why)?
            } else {
                recording.compute_encoder().map_err(why)?
            };
            for j in 0..work.dispatches {
                if work.arm.barrier && j > 0 {
                    encoder.memory_barrier();
                }
                encode(&mut encoder, work.heat + b * work.dispatches + j)?;
            }
            encoder.end();
        }
        pending.push((recording, stamps));
    }
    let encode_s = encode_start.elapsed().as_secs_f64() / (buffers * work.dispatches) as f64;
    let mut all_stamps = Vec::with_capacity(buffers);
    for (recording, stamps) in pending {
        stream.submit(recording).map_err(why)?;
        all_stamps.push(stamps);
    }
    let mut gaps = Vec::with_capacity(work.trials);
    for (i, stamps) in all_stamps.iter().enumerate() {
        let gpu = match stamps.interval(BUFFER_WAIT) {
            Ok(GpuInterval::Ran(gpu)) => gpu,
            // A buffer still running is not waited on again here: the caller
            // submits no later arm behind it.
            Ok(GpuInterval::StillRunning) => {
                return Err(
                    "the command buffer was still running on the GPU when the wait ended"
                        .to_string(),
                );
            }
            // Any other failure has finished; drain it so the next arm does
            // not report it again.
            Err(e) => {
                let _drained = stream.sync();
                return Err(why(e));
            }
        };
        if i >= WARMUPS {
            gaps.push(gpu);
        }
    }
    stream.sync().map_err(why)?;
    if gaps.is_empty() {
        return Err("a timed arm recorded no buffer".to_string());
    }
    let per_dispatch: Vec<f64> = gaps
        .iter()
        .map(|&g| g / work.dispatches as f64 * 1e6)
        .collect();
    let per_buffer: Vec<f64> = gaps
        .iter()
        .map(|&g| work.bytes as f64 * work.dispatches as f64 / g / 1e9)
        .collect();
    let rate = median(&per_buffer);
    if !(rate.is_finite() && rate > 0.0) {
        return Err(format!(
            "the GPU intervals give {rate} GB/s, not a positive finite rate"
        ));
    }
    let median_gap = median(&gaps);
    let clean = gaps
        .iter()
        .filter(|&&g| (g - median_gap).abs() <= 0.25 * median_gap)
        .count();
    Ok(Timing {
        dispatches: work.dispatches,
        copies: work.copies,
        working_set: work.working_set_bytes,
        median: rate,
        us_median: median(&per_dispatch),
        us_p90: percentile(&per_dispatch, 0.9),
        clean,
        gaps,
        encode: encode_s,
    })
}

fn time_shape(
    device: &MetalDevice,
    kernels: &Kernels,
    shape: &Shape,
    arm: &Arm,
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
        // The gate, then the up: two tensors of one expert's shape.
        Shape::DensePair { rows, cols, .. } => {
            let mut both = matrix(quant, *rows, *cols, seed);
            both.extend_from_slice(&matrix(quant, *rows, *cols, seed ^ 1));
            both
        }
        Shape::DenseFold { rows, k, .. } => matrix(quant, *rows, *k, seed),
        Shape::Head { rows, cols, .. } => matrix(quant, *rows, *cols, seed),
    };
    // The bytes the working set is spread by: the weights'. A row that
    // launches the pick alone still holds the head's weights, so its buffers
    // are the pair's.
    let bytes = one.len();
    let (stride, copies, working_set) = spread(bytes);
    let mut all = vec![0u8; copies * stride];
    for c in 0..copies {
        all[c * stride..c * stride + bytes].copy_from_slice(&one);
    }
    let weights = device.upload(stream, &all).map_err(why)?;
    drop(all);
    let x_values = oracle::activations(cols, 0x78, 1.0);
    let x = device.upload(stream, &x_values).map_err(why)?;
    // The norm's gain, the scaled entries' scale word, and the head norm's
    // gated numerator and partial sums.
    let gain = device.upload(stream, &vec![1.0f32; cols]).map_err(why)?;
    let scale_word = device.upload(stream, &[1.0f32]).map_err(why)?;
    let load_norm = |scaled: bool| -> Result<LoadNorm<'_>, String> {
        Ok(if scaled {
            LoadNorm::Scaled {
                scale: scale_word.window(Elems(0), Elems(1)).map_err(why)?,
            }
        } else {
            LoadNorm::Gain {
                gain: gain.window(Elems(0), Elems(cols)).map_err(why)?,
                eps: 1e-6,
            }
        })
    };
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
            weight_blocks += r * (cols / quant.values_per_block());
            out_offset += r;
        }
    }

    let dispatches = dispatch_count(arm, bytes);
    let buffers = WARMUPS + trials;
    // Every dispatch, the heat's first, writes an output of its own.
    let heat = heat_of(arm, bytes);
    let outputs = heat + buffers * dispatches;
    let mut y = device
        .zeroed::<f32>(stream, Elems(outputs * rows))
        .map_err(why)?;
    // The experts' and the head's launches: the instance's experts, the one
    // expert id and route weight of a dense FFN, and one pick word per
    // committed buffer for the head. A shape that launches neither holds the
    // one element each.
    let experts = &kernels.geom().experts;
    let faults_read = matches!(
        shape,
        Shape::DensePair { .. } | Shape::DenseFold { .. } | Shape::Head { .. }
    );
    if faults_read {
        device.fault_word().clear(stream).map_err(why)?;
    }
    let ids = device.upload(stream, &[0i32]).map_err(why)?;
    let route = device.upload(stream, &[1.0f32]).map_err(why)?;
    let mut step_id = device
        .zeroed::<u32>(stream, Elems(outputs.div_ceil(dispatches)))
        .map_err(why)?;

    let timing = time_dispatches(
        stream,
        &TimedWork {
            bytes,
            dispatches,
            copies,
            working_set_bytes: working_set,
            trials,
            arm,
            heat,
        },
        |encoder, n| {
            let w = weights
                .window(Elems((n % copies) * stride), Elems(bytes))
                .map_err(why)?;
            let out = y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?;
            match shape {
                Shape::Matvec {
                    quant,
                    rows,
                    cols,
                    resid,
                } => {
                    kernels
                        .matvec(
                            encoder,
                            MatvecParams {
                                quant: *quant,
                                rows: *rows,
                                cols: *cols,
                                weights: w,
                                x: x.window(Elems(0), Elems(*cols)).map_err(why)?,
                                y: out,
                                mode: if *resid { Mode::Residual } else { Mode::Store },
                            },
                        )
                        .map_err(why)?;
                }
                Shape::Seg {
                    quant,
                    cols,
                    scaled,
                    ..
                } => {
                    kernels
                        .matvec_seg(
                            encoder,
                            SegParams {
                                quant: *quant,
                                segments: &segments,
                                weights: w,
                                x: x.window(Elems(0), Elems(*cols)).map_err(why)?,
                                y: out,
                                norm: load_norm(*scaled)?,
                                fault: device.fault_word(),
                                layer: LAYER_NONE,
                            },
                        )
                        .map_err(why)?;
                }
                Shape::HeadNorm { rows, cols } => {
                    kernels
                        .headnorm_resid(
                            encoder,
                            HeadNormParams {
                                quant: Quant::Q4K,
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
                Shape::DensePair {
                    quant,
                    rows,
                    cols,
                    scaled,
                } => {
                    let half = bytes / 2;
                    let base = (n % copies) * stride;
                    kernels
                        .pair(
                            encoder,
                            PairParams {
                                quant: *quant,
                                shared_quant: *quant,
                                slots: experts.slots,
                                k: *rows,
                                cols: *cols,
                                n_experts: 1,
                                ids: ids
                                    .window(Elems(0), Elems(experts.routed_ids))
                                    .map_err(why)?,
                                gate: weights.window(Elems(base), Elems(half)).map_err(why)?,
                                up: weights
                                    .window(Elems(base + half), Elems(half))
                                    .map_err(why)?,
                                shared_gate: None,
                                shared_up: None,
                                x: x.window(Elems(0), Elems(*cols)).map_err(why)?,
                                norm: load_norm(*scaled)?,
                                y: out,
                                out_stride: *rows,
                                fault: device.fault_word(),
                                layer: LAYER_NONE,
                            },
                        )
                        .map_err(why)?;
                }
                Shape::DenseFold { quant, rows, k } => {
                    kernels
                        .fold(
                            encoder,
                            FoldParams {
                                quant: *quant,
                                shared: SharedPlan::None,
                                rows: *rows,
                                slots: experts.slots,
                                k: *k,
                                n_experts: 1,
                                ids: ids
                                    .window(Elems(0), Elems(experts.routed_ids))
                                    .map_err(why)?,
                                down: w,
                                h: x.window(Elems(0), Elems(*k)).map_err(why)?,
                                w: route.window(Elems(0), Elems(experts.slots)).map_err(why)?,
                                y: &out,
                                fault: device.fault_word(),
                                layer: LAYER_NONE,
                            },
                        )
                        .map_err(why)?;
                }
                Shape::Head {
                    quant,
                    rows,
                    cols,
                    part,
                    scaled,
                } => {
                    let b = n / dispatches;
                    let params = HeadParams {
                        quant: *quant,
                        rows: *rows,
                        k: *cols,
                        simdgroups: HEAD_SIMD_GROUPS,
                        weights: w,
                        x: x.window(Elems(0), Elems(*cols)).map_err(why)?,
                        norm: load_norm(*scaled)?,
                        logits: out,
                        step_id: step_id.window_mut(Elems(b), Elems(1)).map_err(why)?,
                        fault: device.fault_word(),
                        layer: LAYER_NONE,
                    };
                    let launched = match part {
                        HeadPart::Pair => kernels.head(encoder, params),
                        HeadPart::Pick => kernels.head_pick(encoder, &params),
                    };
                    launched.map_err(why)?;
                }
            }
            Ok(())
        },
    )?;
    let mut out = vec![0f32; y.len().get()];
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
    if faults_read {
        ticket_and_fault_free(device, stream)?;
    }
    if check && matches!(shape, Shape::Matvec { resid: false, .. }) {
        let x64: Vec<f64> = x_values.iter().map(|&v| f64::from(v)).collect();
        let reference = oracle::reference(quant, &one, rows, cols, &x64)?;
        let band = oracle::band(quant, cols / quant.values_per_block(), oracle::Kind::Store);
        oracle::check(first, &reference, band, oracle::Kind::Store, None)?;
    }
    Ok(timing)
}

/// Every dispatch wrote the first dispatch's bits.
fn same_outputs(
    device: &MetalDevice,
    stream: &MetalStream,
    y: &bloomery_metal::buffer::MetalBuffer<f32>,
    rows: usize,
) -> Result<(), String> {
    let mut out = vec![0f32; y.len().get()];
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
    Ok(())
}

/// Every in-flight command buffer's ticket is 0, and the kernel raised nothing.
///
/// The buffers of one arm are committed before the first is waited on, so each
/// has its own ticket. A shared counter is the next buffer's handshake.
fn ticket_and_fault(
    device: &MetalDevice,
    stream: &MetalStream,
    ticket: &bloomery_metal::buffer::MetalBuffer<u32>,
) -> Result<(), String> {
    let mut back = vec![1u32; ticket.len().get()];
    device
        .read(stream, ticket.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if let Some((i, n)) = back.into_iter().enumerate().find(|(_, n)| *n != 0) {
        return Err(format!("the ticket counter of buffer {i} is {n}, not 0"));
    }
    if let Some(fault) = device.fault_word().read(stream).map_err(why)? {
        return Err(format!("the kernel raised {fault}"));
    }
    Ok(())
}

/// One counted text edit of the kernels' source: a phase cut's copy, the way
/// the gate's mutants edit theirs. The find text must occur exactly once, so
/// a kernel edit that moves it is a named error, not a silent no-op.
struct Phase {
    name: &'static str,
    find: &'static str,
    with: &'static str,
}

/// The kernels of a phase cut's copy of the shipped source, compiled by the
/// runtime.
fn phase_kernels(device: &MetalDevice, phase: &Phase) -> Result<Kernels, String> {
    let found = SOURCE.matches(phase.find).count();
    if found != 1 {
        return Err(format!(
            "phase {}: its find text occurs {found} times, not 1; the kernel moved",
            phase.name
        ));
    }
    let text = SOURCE.replacen(phase.find, phase.with, 1);
    let library = Library::from_source_text(device, &text).map_err(why)?;
    Kernels::from_library(&library).map_err(why)
}

/// The geometry a run compiles the kernels' source at: the rows a SIMD group
/// owns, the SIMD groups in a mat-vec's threadgroup, and the router's own
/// group count. The default is the shipped source's own geometry; a value
/// beside it is a variant source, compiled the way a phase cut's is.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Geometry {
    rows: usize,
    simdgroups: usize,
    router_sg: usize,
}

impl Geometry {
    const DEFAULT: Self = Self {
        rows: ROWS_PER_SIMD_GROUP,
        simdgroups: SIMD_GROUPS_PER_GROUP,
        router_sg: router::SG,
    };

    fn is_default(self) -> bool {
        self == Self::DEFAULT
    }

    /// The marker a variant run's rows carry, empty for the default.
    fn label(self) -> String {
        if self.is_default() {
            return String::new();
        }
        format!(
            " [rows={} sg={} router_sg={}]",
            self.rows, self.simdgroups, self.router_sg
        )
    }
}

/// The kernels of a geometry variant: `source` (the device's instance's
/// program text) with the two `#define`s of the geometry replaced, compiled by
/// the runtime, one occurrence of each named so a header edit that moves one
/// is an error, not a silent no-op.
fn geometry_kernels(
    device: &MetalDevice,
    source: &str,
    geometry: Geometry,
) -> Result<Kernels, String> {
    let edits = [
        (
            format!(
                "#define BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP {}",
                ROWS_PER_SIMD_GROUP
            ),
            format!(
                "#define BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP {}",
                geometry.rows
            ),
        ),
        (
            format!("#define BLOOMERY_ROUTER_SG {}", router::SG),
            format!("#define BLOOMERY_ROUTER_SG {}", geometry.router_sg),
        ),
    ];
    let mut text = source.to_string();
    for (find, with) in edits {
        let found = text.matches(find.as_str()).count();
        if found != 1 {
            return Err(format!(
                "the geometry define {find:?} occurs {found} times, not 1; the header moved"
            ));
        }
        text = text.replacen(find.as_str(), with.as_str(), 1);
    }
    let library = Library::from_source_text(device, &text).map_err(why)?;
    Kernels::with_geometry(
        &library,
        geometry.rows,
        geometry.simdgroups,
        geometry.router_sg,
    )
    .map_err(why)
}

/// The experts a pair or fold stack holds: big enough that eight routed ids
/// and the shared expert read distinct weights, small enough to build fast.
const PROBE_EXPERTS: usize = 16;

/// The ids a probe pair or fold launches: eight distinct routed experts.
const PROBE_IDS: [i32; ROUTED_IDS] = [0, 1, 2, 3, 4, 5, 6, 7];

/// The bytes of `experts` experts of `rows` rows over `cols` columns.
fn expert_stack_bytes(quant: Quant, experts: usize, rows: usize, cols: usize) -> usize {
    experts * rows * cols / quant.values_per_block() * quant.block_bytes()
}

/// One tensor's share of the working set: `copies` copies at a 256-aligned
/// stride, sized so every tensor of the dispatch together fills about
/// `budget` bytes.
fn spread_share(bytes: usize, budget: usize) -> (usize, usize, usize) {
    let stride = bytes.div_ceil(256) * 256;
    let copies = (budget / stride).max(1);
    (stride, copies, copies * stride)
}

/// Upload `copies` copies of `one` at `stride`, the probe's working-set way.
fn upload_copies(
    device: &MetalDevice,
    stream: &MetalStream,
    one: &[u8],
    stride: usize,
    copies: usize,
) -> Result<bloomery_metal::buffer::MetalBuffer<u8>, String> {
    let mut all = vec![0u8; copies * stride];
    for c in 0..copies {
        all[c * stride..c * stride + one.len()].copy_from_slice(one);
    }
    let buffer = device.upload(stream, &all).map_err(why)?;
    Ok(buffer)
}

/// The gate-up pair at the session's shape: nine slots of 512 by 2048, eight
/// routed experts and the shared one, synthetic stacks. Every dispatch reads
/// the same bytes of another copy, so every dispatch must give the same bits.
fn time_pair(
    device: &MetalDevice,
    kernels: &Kernels,
    quant: Quant,
    arm: &Arm,
    trials: usize,
) -> Result<Timing, String> {
    let stream = device.stream();
    device.fault_word().clear(stream).map_err(why)?;
    let rows = EXPERT_K;
    let cols = 2048usize;
    let one_expert = expert_stack_bytes(quant, 1, rows, cols);
    let stack = expert_stack_bytes(quant, PROBE_EXPERTS, rows, cols);
    let gate = matrix(quant, PROBE_EXPERTS * rows, cols, 0x70A1);
    let up = matrix(quant, PROBE_EXPERTS * rows, cols, 0x70B2);
    let shared_gate = matrix(quant, rows, cols, 0x70C3);
    let shared_up = matrix(quant, rows, cols, 0x70D4);
    let bytes = EXPERT_SLOTS * 2 * one_expert;
    // Four weight tensors: the stacks get the budget's lion share, the shared
    // experts one expert's worth of copies each.
    let budget = WORKING_SET_BYTES / 4;
    let (stack_stride, stack_copies, stack_set) = spread_share(stack, budget);
    let (one_stride, one_copies, one_set) = spread_share(one_expert, budget);
    let gate_w = upload_copies(device, stream, &gate, stack_stride, stack_copies)?;
    let up_w = upload_copies(device, stream, &up, stack_stride, stack_copies)?;
    let shg_w = upload_copies(device, stream, &shared_gate, one_stride, one_copies)?;
    let shu_w = upload_copies(device, stream, &shared_up, one_stride, one_copies)?;
    let working_set = 2 * stack_set + 2 * one_set;
    let x = device
        .upload(stream, &oracle::activations(cols, 0x78, 1.0))
        .map_err(why)?;
    let gain = device.upload(stream, &vec![1.0f32; cols]).map_err(why)?;
    let ids = device.upload(stream, &PROBE_IDS).map_err(why)?;
    let dispatches = dispatch_count(arm, bytes);
    let buffers = WARMUPS + trials;
    let out_values = EXPERT_SLOTS * EXPERT_K;
    let mut y = device
        .zeroed::<f32>(stream, Elems(buffers * dispatches * out_values))
        .map_err(why)?;
    let timing = time_dispatches(
        stream,
        &TimedWork {
            bytes,
            dispatches,
            copies: stack_copies,
            working_set_bytes: working_set,
            trials,
            arm,
            heat: 0,
        },
        |encoder, n| {
            let sg = (n % one_copies) * one_stride;
            kernels
                .pair(
                    encoder,
                    PairParams {
                        quant,
                        shared_quant: quant,
                        slots: EXPERT_SLOTS,
                        k: EXPERT_K,
                        cols,
                        n_experts: PROBE_EXPERTS,
                        ids: ids.window(Elems(0), Elems(ROUTED_IDS)).map_err(why)?,
                        gate: gate_w
                            .window(Elems((n % stack_copies) * stack_stride), Elems(stack))
                            .map_err(why)?,
                        up: up_w
                            .window(Elems((n % stack_copies) * stack_stride), Elems(stack))
                            .map_err(why)?,
                        shared_gate: Some(shg_w.window(Elems(sg), Elems(one_expert)).map_err(why)?),
                        shared_up: Some(shu_w.window(Elems(sg), Elems(one_expert)).map_err(why)?),
                        x: x.window(Elems(0), Elems(cols)).map_err(why)?,
                        norm: LoadNorm::Gain {
                            gain: gain.window(Elems(0), Elems(cols)).map_err(why)?,
                            eps: 1e-6,
                        },
                        y: y.window_mut(Elems(n * out_values), Elems(out_values))
                            .map_err(why)?,
                        out_stride: EXPERT_K,
                        fault: device.fault_word(),
                        layer: LAYER_NONE,
                    },
                )
                .map_err(why)?;
            Ok(())
        },
    )?;
    same_outputs(device, stream, &y, out_values)?;
    ticket_and_fault_free(device, stream)?;
    Ok(timing)
}

/// The folded down at the session's shape: 2048 rows over the nine slots'
/// 512-wide hidden states, the route weights applied on the load.
fn time_fold(
    device: &MetalDevice,
    kernels: &Kernels,
    quant: Quant,
    arm: &Arm,
    trials: usize,
) -> Result<Timing, String> {
    let stream = device.stream();
    device.fault_word().clear(stream).map_err(why)?;
    let rows = 2048usize;
    let k = EXPERT_K;
    let one_expert = expert_stack_bytes(quant, 1, rows, k);
    let stack = expert_stack_bytes(quant, PROBE_EXPERTS, rows, k);
    let down = matrix(quant, PROBE_EXPERTS * rows, k, 0x70E5);
    let shared = matrix(quant, rows, k, 0x70F6);
    let bytes = EXPERT_SLOTS * one_expert;
    let budget = WORKING_SET_BYTES / 3;
    let (stack_stride, stack_copies, stack_set) = spread_share(stack, budget);
    let (one_stride, one_copies, one_set) = spread_share(one_expert, budget);
    let down_w = upload_copies(device, stream, &down, stack_stride, stack_copies)?;
    let sh_w = upload_copies(device, stream, &shared, one_stride, one_copies)?;
    let working_set = stack_set + one_set;
    let mut h = oracle::activations(EXPERT_SLOTS * k, 0x78, 1.0);
    for slot in 0..EXPERT_SLOTS {
        for v in &mut h[slot * k..(slot + 1) * k] {
            *v *= 1.0 + 0.125 * slot as f32;
        }
    }
    let h = device.upload(stream, &h).map_err(why)?;
    let route: Vec<f32> = (0..EXPERT_SLOTS).map(|s| 0.1 * (s as f32 + 1.0)).collect();
    let w = device.upload(stream, &route).map_err(why)?;
    let ids = device.upload(stream, &PROBE_IDS).map_err(why)?;
    let dispatches = dispatch_count(arm, bytes);
    let buffers = WARMUPS + trials;
    let mut y = device
        .zeroed::<f32>(stream, Elems(buffers * dispatches * rows))
        .map_err(why)?;
    let timing = time_dispatches(
        stream,
        &TimedWork {
            bytes,
            dispatches,
            copies: stack_copies,
            working_set_bytes: working_set,
            trials,
            arm,
            heat: 0,
        },
        |encoder, n| {
            let sh = (n % one_copies) * one_stride;
            let out = y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?;
            kernels
                .fold(
                    encoder,
                    FoldParams {
                        quant,
                        shared: SharedPlan::Same(
                            sh_w.window(Elems(sh), Elems(one_expert)).map_err(why)?,
                        ),
                        rows,
                        slots: EXPERT_SLOTS,
                        k,
                        n_experts: PROBE_EXPERTS,
                        ids: ids.window(Elems(0), Elems(ROUTED_IDS)).map_err(why)?,
                        down: down_w
                            .window(Elems((n % stack_copies) * stack_stride), Elems(stack))
                            .map_err(why)?,
                        h: h.window(Elems(0), Elems(EXPERT_SLOTS * k)).map_err(why)?,
                        w: w.window(Elems(0), Elems(EXPERT_SLOTS)).map_err(why)?,
                        y: &out,
                        fault: device.fault_word(),
                        layer: LAYER_NONE,
                    },
                )
                .map_err(why)?;
            Ok(())
        },
    )?;
    same_outputs(device, stream, &y, rows)?;
    ticket_and_fault_free(device, stream)?;
    Ok(timing)
}

/// The pair, the fold and the head hold no ticket; the probe's own launches
/// carry none, and the kernel raised nothing.
fn ticket_and_fault_free(device: &MetalDevice, stream: &MetalStream) -> Result<(), String> {
    if let Some(fault) = device.fault_word().read(stream).map_err(why)? {
        return Err(format!("the kernel raised {fault}"));
    }
    Ok(())
}

/// The router's phase cuts: the norm's load pass (the gained input a constant,
/// the squares skipped; the reduction stays), and the tail (the last
/// threadgroup resets its ticket and selects nothing).
const ROUTER_PHASES: &[Phase] = &[
    Phase {
        name: "nonorm",
        find: "    float sum_sq = 0.f;\n    for (uint i = 0u; i < ROUTER_K / 4u / ROUTER_THREADS; ++i) {\n        const uint f4 = tid + ROUTER_THREADS * i;\n        float4 xv = router_load4(x, f4);\n        float4 gv = router_load4(gain, f4);\n        tg_h[f4] = xv * gv;\n        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;\n    }",
        with: "    float sum_sq = 0.f;\n    for (uint i = 0u; i < ROUTER_K / 4u / ROUTER_THREADS; ++i) {\n        const uint f4 = tid + ROUTER_THREADS * i;\n        tg_h[f4] = 1.f;\n    }",
    },
    Phase {
        name: "notail",
        find: "    ticket_is_last(ticket, groups, tid, last_flag);\n    if (last_flag == 1u) {\n        route_last(partials, ticket, logits, ids, weights, faults, args,\n                   tid, lane, sg_in_tg, tg_logits, tg_keys, tg_bad);\n    }\n",
        with: "    ticket_is_last(ticket, groups, tid, last_flag);\n    if (last_flag == 1u && tid == 0u) {\n        ticket_reset(ticket);\n    }\n",
    },
];

/// The head-normed mat-vec's phase cuts: the scales precomputed by the
/// producer (the pass reads them, the barrier stays), and the apply cut (the
/// row loop the plain residual path runs: no head scale on the block).
const HEADNORM_PHASES: &[Phase] = &[
    Phase {
        name: "pre",
        find: "    threadgroup float scales[BLOOMERY_DELTA_VALUE_HEADS];\n    // A lane a head, a further round for each SIMD width of heads.\n    if (sgitg == 0) {\n        for (uint head = tiisg; head < BLOOMERY_DELTA_VALUE_HEADS; head += 32u) {\n            float sum = 0.f;\n            for (short j = 0; j < BLOOMERY_DELTA_COLUMN_BLOCKS; ++j) {\n                sum += partials[head * BLOOMERY_DELTA_COLUMN_BLOCKS + j];\n            }\n            const float mean = sum / float(BLOOMERY_DELTA_HEAD_COLS);\n            const float s = rsqrt(mean + args.eps);\n            if (!isfinite(sum) || !isfinite(s)) {\n                // A head the norm cannot take is NaN, not a zero head: its values\n                // are then not finite and the rows that read them are NaN.\n                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);\n                scales[head] = NAN;\n            } else {\n                scales[head] = s;\n            }\n        }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        with: "    threadgroup float scales[BLOOMERY_DELTA_VALUE_HEADS];\n    if (sgitg == 0 && tiisg < BLOOMERY_DELTA_VALUE_HEADS) {\n        scales[tiisg] = partials[tiisg];\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
    },
    Phase {
        name: "noapply",
        find: "    threadgroup float scales[BLOOMERY_DELTA_VALUE_HEADS];\n    // A lane a head, a further round for each SIMD width of heads.\n    if (sgitg == 0) {\n        for (uint head = tiisg; head < BLOOMERY_DELTA_VALUE_HEADS; head += 32u) {\n            float sum = 0.f;\n            for (short j = 0; j < BLOOMERY_DELTA_COLUMN_BLOCKS; ++j) {\n                sum += partials[head * BLOOMERY_DELTA_COLUMN_BLOCKS + j];\n            }\n            const float mean = sum / float(BLOOMERY_DELTA_HEAD_COLS);\n            const float s = rsqrt(mean + args.eps);\n            if (!isfinite(sum) || !isfinite(s)) {\n                // A head the norm cannot take is NaN, not a zero head: its values\n                // are then not finite and the rows that read them are NaN.\n                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);\n                scales[head] = NAN;\n            } else {\n                scales[head] = s;\n            }\n        }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    const int first_row = simd_first_row(tgpig.x, sgitg);\n    matvec_rows<Q, false, true, true>(W, g, y, args.nblocks, args.eps, args.layer,\n                                      nullptr, scales, faults, first_row, (int)args.rows, tiisg);",
        with: "    const int first_row = simd_first_row(tgpig.x, sgitg);\n    matvec_rows<Q, false, true, false>(W, g, y, args.nblocks, args.eps, args.layer,\n                                       nullptr, nullptr, faults, first_row, (int)args.rows, tiisg);",
    },
];

/// The dense mat-vec's inner-loop cuts, beside its own rows: the block loop
/// with two blocks a step (a 512-value pass, MLX's loop shape), a compiler
/// unroll of the shipped loop, so the lane's block order and the row's float
/// order do not move. The two rows find the one shared loop now: the cut
/// unrolls the template's body, and the row measured is the named quant's.
const MATVEC_UNROLL: &str = "_Pragma(\"clang loop unroll_count(2)\")\n";

const MATVEC_PHASES: &[(&str, Quant, &str)] = &[
    (
        "unroll2",
        Quant::Q4K,
        "    for (int ib = Q::first_block(lane); ib < nb; ib += Q::block_stride()) {",
    ),
    (
        "unroll2",
        Quant::Q6K,
        "    for (int ib = Q::first_block(lane); ib < nb; ib += Q::block_stride()) {",
    ),
];

/// The kernels of a mat-vec phase cut: the pragma in front of the quant's
/// block loop.
fn matvec_phase_kernels(
    device: &MetalDevice,
    phase: (&str, Quant, &str),
) -> Result<Kernels, String> {
    let (name, _, find) = phase;
    let found = SOURCE.matches(find).count();
    if found != 1 {
        return Err(format!(
            "phase {name}: its find text occurs {found} times, not 1; the kernel moved"
        ));
    }
    let text = SOURCE.replacen(find, &format!("{MATVEC_UNROLL}{find}"), 1);
    let library = Library::from_source_text(device, &text).map_err(why)?;
    Kernels::from_library(&library).map_err(why)
}

fn time_router(
    device: &MetalDevice,
    kernels: &Kernels,
    arm: &Arm,
    trials: usize,
) -> Result<Timing, String> {
    let stream = device.stream();
    device.fault_word().clear(stream).map_err(why)?;
    let (stride, copies, working_set) = spread(ROUTER_BYTES);
    let stride_elems = stride / core::mem::size_of::<f32>();
    let weight_elems = router::ROWS * router::K;
    let mut one = vec![0f32; weight_elems];
    let mut rng = oracle::Rng::new(0x7200_0001);
    for value in &mut one {
        *value = rng.signed() * 0.02;
    }
    let mut all = vec![0f32; copies * stride_elems];
    for c in 0..copies {
        let at = c * stride_elems;
        all[at..at + weight_elems].copy_from_slice(&one);
    }
    let weights = device.upload(stream, &all).map_err(why)?;
    drop(all);
    let x = device
        .upload(stream, &oracle::activations(router::K, 0x78, 1.0))
        .map_err(why)?;
    let gain = device
        .upload(stream, &vec![1.0f32; router::K])
        .map_err(why)?;
    let dispatches = dispatch_count(arm, ROUTER_BYTES);
    let buffers = WARMUPS + trials;
    // One scratch set per committed buffer. The dispatches inside a buffer are
    // ordered; the buffers of the arm are in flight together.
    let mut partials = device
        .zeroed::<f32>(stream, Elems(buffers * router::PARTIALS))
        .map_err(why)?;
    let mut ticket = device.zeroed::<u32>(stream, Elems(buffers)).map_err(why)?;
    let slots = router::TOP_K_MAX + 1;
    let mut ids = device
        .zeroed::<u32>(stream, Elems(buffers * slots))
        .map_err(why)?;
    let mut chosen = device
        .zeroed::<f32>(stream, Elems(buffers * slots))
        .map_err(why)?;
    let rows = router::ROWS;
    let mut y = device
        .zeroed::<f32>(stream, Elems((WARMUPS + trials) * dispatches * rows))
        .map_err(why)?;
    let timing = time_dispatches(
        stream,
        &TimedWork {
            bytes: ROUTER_BYTES,
            dispatches,
            copies,
            working_set_bytes: working_set,
            trials,
            arm,
            heat: 0,
        },
        |encoder, n| {
            let b = n / dispatches;
            let base = (n % copies) * stride_elems;
            kernels
                .router(
                    encoder,
                    RouterParams {
                        rows: router::ROWS,
                        k: router::K,
                        top_k: router::TOP_K_MAX,
                        routed: weights
                            .window(Elems(base), Elems(router::EXPERTS * router::K))
                            .map_err(why)?,
                        shared: (router::SHARED == 1)
                            .then(|| {
                                weights.window(
                                    Elems(base + router::EXPERTS * router::K),
                                    Elems(router::K),
                                )
                            })
                            .transpose()
                            .map_err(why)?,
                        x: x.window(Elems(0), Elems(router::K)).map_err(why)?,
                        gain: gain.window(Elems(0), Elems(router::K)).map_err(why)?,
                        partials: partials
                            .window_mut(Elems(b * router::PARTIALS), Elems(router::PARTIALS))
                            .map_err(why)?,
                        ticket: ticket.window_mut(Elems(b), Elems(1)).map_err(why)?,
                        logits: y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?,
                        ids: ids
                            .window_mut(Elems(b * slots), Elems(slots))
                            .map_err(why)?,
                        weights: chosen
                            .window_mut(Elems(b * slots), Elems(slots))
                            .map_err(why)?,
                        eps: 1e-6,
                        fault: device.fault_word(),
                        layer: LAYER_NONE,
                    },
                )
                .map_err(why)?;
            Ok(())
        },
    )?;
    same_outputs(device, stream, &y, rows)?;
    ticket_and_fault(device, stream, &ticket)?;
    Ok(timing)
}

fn time_head(
    device: &MetalDevice,
    kernels: &Kernels,
    arm: &Arm,
    trials: usize,
) -> Result<Timing, String> {
    let stream = device.stream();
    device.fault_word().clear(stream).map_err(why)?;
    let one = matrix(Quant::Q6K, HEAD_ROWS, HEAD_K, 0x6EAD);
    let bytes = one.len();
    if bytes != head_bytes() {
        return Err(format!(
            "the head weights are {bytes} bytes, the kernel's shape is {}",
            head_bytes()
        ));
    }
    let (stride, copies, working_set) = spread(bytes);
    let mut all = vec![0u8; copies * stride];
    for c in 0..copies {
        all[c * stride..c * stride + bytes].copy_from_slice(&one);
    }
    let weights = device.upload(stream, &all).map_err(why)?;
    drop(all);
    let x = device
        .upload(stream, &oracle::activations(HEAD_K, 0x78, 1.0))
        .map_err(why)?;
    let gain = device.upload(stream, &vec![1.0f32; HEAD_K]).map_err(why)?;
    let dispatches = dispatch_count(arm, bytes);
    let buffers = WARMUPS + trials;
    // One pick word per committed buffer.
    let mut step_id = device.zeroed::<u32>(stream, Elems(buffers)).map_err(why)?;
    let rows = HEAD_ROWS;
    let mut y = device
        .zeroed::<f32>(stream, Elems((WARMUPS + trials) * dispatches * rows))
        .map_err(why)?;
    let timing = time_dispatches(
        stream,
        &TimedWork {
            bytes,
            dispatches,
            copies,
            working_set_bytes: working_set,
            trials,
            arm,
            heat: 0,
        },
        |encoder, n| {
            let b = n / dispatches;
            kernels
                .head(
                    encoder,
                    HeadParams {
                        quant: Quant::Q6K,
                        rows: HEAD_ROWS,
                        k: HEAD_K,
                        simdgroups: HEAD_SIMD_GROUPS,
                        weights: weights
                            .window(Elems((n % copies) * stride), Elems(bytes))
                            .map_err(why)?,
                        x: x.window(Elems(0), Elems(HEAD_K)).map_err(why)?,
                        norm: LoadNorm::Gain {
                            gain: gain.window(Elems(0), Elems(HEAD_K)).map_err(why)?,
                            eps: 1e-6,
                        },
                        logits: y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?,
                        step_id: step_id.window_mut(Elems(b), Elems(1)).map_err(why)?,
                        fault: device.fault_word(),
                        layer: LAYER_NONE,
                    },
                )
                .map_err(why)?;
            Ok(())
        },
    )?;
    same_outputs(device, stream, &y, rows)?;
    ticket_and_fault_free(device, stream)?;
    Ok(timing)
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Nearest rank of `p`: the value at `ceil(p * len)`, counting from one.
fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = (p * v.len() as f64).ceil() as usize;
    v[rank.saturating_sub(1)]
}

/// The `--geometry` value: `key=value` pairs of `rows`, `sg` and `router_sg`
/// over a comma, each optional, refused by name when a key is unknown or a
/// value is not a whole number.
fn geometry_of(value: Option<&String>) -> Result<Geometry, String> {
    let Some(value) = value else {
        return Err("--geometry takes a value".to_string());
    };
    let mut geometry = Geometry::DEFAULT;
    let mut seen = 0usize;
    for tok in value.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let (key, raw) = tok
            .split_once('=')
            .ok_or_else(|| format!("--geometry takes key=value pairs, not {tok:?}"))?;
        let n: usize = raw
            .parse()
            .map_err(|_| format!("--geometry's {key} takes a whole number, not {raw:?}"))?;
        match key.trim() {
            "rows" => geometry.rows = n,
            "sg" => geometry.simdgroups = n,
            "router_sg" => geometry.router_sg = n,
            other => {
                return Err(format!(
                    "--geometry takes rows, sg or router_sg, not {other:?}"
                ));
            }
        }
        seen += 1;
    }
    if seen == 0 {
        return Err("--geometry takes at least one of rows, sg, router_sg".to_string());
    }
    Ok(geometry)
}

/// The most timed buffers: every buffer, warm-ups included, is encoded before
/// any is committed, so all of them must fit the default queue's 64 command
/// buffers or `begin` blocks.
const MAX_TRIALS: usize = 64 - WARMUPS;

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

/// The `--chain` value: at least two dispatches, and no more than [`CHAIN_MAX`].
fn chain_of(value: Option<&String>) -> Result<usize, String> {
    match value.map(|v| (v, v.parse::<usize>())) {
        None => Err("--chain takes a value".to_string()),
        Some((_, Ok(n))) if (2..=CHAIN_MAX).contains(&n) => Ok(n),
        Some((value, _)) => Err(format!(
            "--chain takes a whole number from 2 to {CHAIN_MAX}, not {value:?}"
        )),
    }
}

fn arms_of(chain: Option<usize>) -> Vec<Arm> {
    let mut arms = vec![window_arm()];
    if let Some(n) = chain {
        arms.push(Arm {
            name: "lone",
            dispatches: Some(1),
            concurrent: false,
            barrier: false,
        });
        arms.push(Arm {
            name: "serial",
            dispatches: Some(n),
            concurrent: false,
            barrier: true,
        });
        arms.push(Arm {
            name: "concurrent",
            dispatches: Some(n),
            concurrent: true,
            barrier: true,
        });
    }
    arms
}

/// The shape the stream bandwidth comes from: a large streaming mat-vec.
const STREAM_LABEL: &str = "matvec q4k 8192x4096";

fn stream_shape() -> Shape {
    Shape::Matvec {
        quant: Quant::Q4K,
        rows: 8192,
        cols: 4096,
        resid: false,
    }
}

struct Run {
    only: Vec<String>,
    trials: usize,
    chain: Option<usize>,
    geometry: Geometry,
    gaps: bool,
    check: bool,
}

fn help() {
    println!(
        "\
metal_matvec_time: development-loop timings of the Metal kernels.

A number this prints is this machine's, from no timing lease, and is not a
published result. Rows are tab-separated: kernel, shape, arm, us_median,
us_p90, gbps.

  --help                 print this and exit 0
  --only KIND            a kernel (matvec, seg, headnorm, router, head), a
                         mat-vec type (q4k, q6k, a4), or a shape label.
                         Repeatable, and one value may be comma-separated.
                         Unset times every shape. `lfm3` runs LFM3-1.7B's
                         decode rows alone (lfm3.rs); a further value there
                         narrows them to a kernel (pair, fold, head, seg,
                         matvec) or a label part (`[plain]`, `[scalarx]`);
                         `sweep` times each row under the geometry variants
                         of lfm3.rs too, `overlap` adds an arm of the window
                         in a concurrent encoder with no barrier. `d27` runs
                         Qwen3.8-27B's rows the same way (d27.rs).
  --trials N             timed command buffers per arm (default 16)
  --chain N              also time N back-to-back dispatches: arm lone (one
                         dispatch), serial and concurrent (a memory barrier
                         between each). cb_us is (chain time − N × lone time) / N
  --geometry K=V,...     compile the kernels' own source at another geometry:
                         rows (a SIMD group's output rows, 2/4/8), sg (SIMD
                         groups in a mat-vec threadgroup), router_sg (the
                         router's). Every arm of the run launches the variant.
  --gaps                 print every timed buffer's GPU time, in microseconds
  --no-check             skip the plain mat-vec's f64 oracle check"
    );
}

fn parse(args: &[String]) -> Result<Run, String> {
    let mut only = Vec::new();
    let mut trials = 16usize;
    let mut chain = None;
    let mut geometry = None;
    let mut gaps = false;
    let mut check = true;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = || args.get(i + 1);
        match arg {
            "--gaps" => gaps = true,
            "--no-check" => check = false,
            "--only" => {
                let value = next().ok_or("--only takes a value")?;
                if value.split(',').any(|tok| tok.trim().is_empty()) {
                    return Err("--only takes a kernel, a type or a shape label".to_string());
                }
                only.push(value.clone());
                i += 1;
            }
            "--trials" => {
                trials = trials_of(next())?;
                i += 1;
            }
            "--chain" => {
                chain = Some(chain_of(next())?);
                i += 1;
            }
            "--geometry" => {
                geometry = Some(geometry_of(next())?);
                i += 1;
            }
            other => {
                return Err(format!(
                    "unknown argument {other}; --help lists the arguments"
                ));
            }
        }
        i += 1;
    }
    Ok(Run {
        only,
        trials,
        chain,
        geometry: geometry.unwrap_or(Geometry::DEFAULT),
        gaps,
        check,
    })
}

/// The selection names `kind`, or a label that starts with `label_head`. An
/// empty selection names every kernel.
fn named(only: &[String], kind: &str, label_head: &str) -> bool {
    only.is_empty()
        || only.iter().any(|spec| {
            spec.split(',')
                .map(str::trim)
                .any(|tok| tok == kind || tok.starts_with(label_head))
        })
}

/// `tok` selects this launch. A mat-vec type selects plain mat-vecs only, and
/// `head` does not select the head-normed residual.
fn selects(only: &[String], kind: &str, quant: &str, label: &str) -> bool {
    if only.is_empty() {
        return true;
    }
    only.iter().any(|spec| {
        spec.split(',').map(str::trim).any(|tok| {
            tok == kind
                || tok == label
                || (kind == "matvec" && tok == quant)
                || (tok != "head" && tok != "router" && label.contains(tok))
        })
    })
}

fn tail_extra(us_median: f64, bytes: usize, stream_gbps: Option<f64>, pred: (f64, f64)) -> String {
    let Some(gbps) = stream_gbps.filter(|g| g.is_finite() && *g > 0.0) else {
        // The row's own microseconds still print. The byte time needs the
        // stream rate, and the name says it is not in this run.
        return format!(
            "\tstream_gbps=unavailable\tbytes_us=unavailable\ttail_us=unavailable\treason=this run has no stream bandwidth\ttail_pred_us={:.0}-{:.0}",
            pred.0, pred.1
        );
    };
    let bytes_us = bytes as f64 / gbps / 1e3;
    let tail_us = us_median - bytes_us;
    format!(
        "\tstream_gbps={gbps:.2}\tbytes_us={bytes_us:.3}\ttail_us={tail_us:.3}\ttail_pred_us={:.0}-{:.0}",
        pred.0, pred.1
    )
}

fn print_row(
    kernel: &str,
    label: &str,
    arm: &str,
    trials: usize,
    t: &Timing,
    bytes: usize,
    extra: &str,
) {
    println!(
        "time\tkernel={kernel}\tshape={label}\tarm={arm}\tn={}\trepeats={trials}\tus_median={:.3}\tus_p90={:.3}\tgbps={:.2}\tbytes={bytes}\tclean={}\tcopies={}\tworking_set_bytes={}\tencode_us={:.2}{extra}",
        t.dispatches,
        t.us_median,
        t.us_p90,
        t.median,
        t.clean,
        t.copies,
        t.working_set,
        t.encode * 1e6,
    );
}

fn print_fail(kernel: &str, label: &str, arm: &str, err: &str) {
    println!("time\tkernel={kernel}\tshape={label}\tarm={arm}\tFAIL\t{err}");
}

/// One kernel's arms: the window arm's rate, whether an arm failed, and
/// whether a buffer was still running (later arms stay unsubmitted).
struct ArmRun {
    window_gbps: Option<f64>,
    failed: bool,
    stuck: bool,
}

/// Time every arm.
fn time_arms(
    run: &Run,
    kernel: &str,
    label: &str,
    bytes: usize,
    tail: Option<(f64, f64)>,
    stream_gbps: Option<f64>,
    time_one: impl Fn(&Arm) -> Result<Timing, String>,
) -> ArmRun {
    let mut window_gbps = None;
    let mut lone_us = None;
    let mut failed = false;
    let mut stuck = false;
    for arm in arms_of(run.chain) {
        match time_one(&arm) {
            Ok(t) => {
                if run.gaps {
                    let us: Vec<String> =
                        t.gaps.iter().map(|g| format!("{:.0}", g * 1e6)).collect();
                    println!(
                        "gaps_us\tkernel={kernel}\tarm={}\t{}",
                        arm.name,
                        us.join(" ")
                    );
                }
                let mut extra = String::new();
                if let Some(pred) = tail {
                    extra = tail_extra(t.us_median, bytes, stream_gbps, pred);
                }
                if arm.name == "lone" {
                    lone_us = Some(t.us_median);
                }
                if arm.name == "serial" || arm.name == "concurrent" {
                    match lone_us {
                        Some(lone) => {
                            let cb = t.us_median - lone;
                            extra.push_str(&format!("\tlone_us={lone:.3}\tcb_us={cb:.3}"));
                        }
                        None => {
                            print_fail(
                                kernel,
                                label,
                                arm.name,
                                "the lone arm did not time, so the chain has no per-dispatch cost",
                            );
                            failed = true;
                            continue;
                        }
                    }
                }
                if arm.name == "window" {
                    window_gbps = Some(t.median);
                }
                print_row(kernel, label, arm.name, run.trials, &t, bytes, &extra);
            }
            Err(e) => {
                print_fail(kernel, label, arm.name, &e);
                failed = true;
                if e.contains("still running") {
                    stuck = true;
                    break;
                }
            }
        }
    }
    ArmRun {
        window_gbps,
        failed,
        stuck,
    }
}

pub fn main() -> ExitCode {
    if !PREDICTED_GEOMETRY {
        eprintln!(
            "metal_matvec_time: refused: its byte predictions are Qwen3.6's, and this build's host \
             instance is {}",
            HOST.name
        );
        return ExitCode::from(78);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        help();
        return ExitCode::SUCCESS;
    }
    let run = match parse(&args) {
        Ok(run) => run,
        Err(e) => {
            println!("metal_matvec_time: {e}");
            return ExitCode::from(64);
        }
    };

    if lfm3::selected(&run.only) {
        return lfm3::main(&run);
    }
    if d27::selected(&run.only) {
        return d27::main(&run);
    }

    let device = match MetalDevice::open(&HOST) {
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
    let kernels = if run.geometry.is_default() {
        match Kernels::new(&device) {
            Ok(kernels) => kernels,
            Err(e) => {
                println!("metal_matvec_time: FAIL ({e})");
                return ExitCode::FAILURE;
            }
        }
    } else {
        match geometry_kernels(&device, SOURCE, run.geometry) {
            Ok(kernels) => kernels,
            Err(e) => {
                println!(
                    "metal_matvec_time: FAIL (geometry rows={} sg={} router_sg={}: {e})",
                    run.geometry.rows, run.geometry.simdgroups, run.geometry.router_sg
                );
                return ExitCode::FAILURE;
            }
        }
    };
    let info = match device.info() {
        Ok(info) => info,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let chain = run
        .chain
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "info\tinstrument=development-loop\tdevice={}\tlib={}\tbackend={}\ttrials={}\twarmups={WARMUPS}\twindow_bytes={WINDOW_BYTES}\tworking_set_bytes={WORKING_SET_BYTES}\tchain={chain}\tgeom=rows{},sg{},router_sg{}",
        info.name,
        device.library().path().name(),
        <MetalDevice as Device>::BACKEND,
        run.trials,
        run.geometry.rows,
        run.geometry.simdgroups,
        run.geometry.router_sg,
    );

    let mut shapes: Vec<(String, Shape)> = Vec::new();
    for quant in [Quant::Q4K, Quant::Q6K, Quant::A4] {
        for (rows, cols) in SHAPES {
            shapes.push((
                format!("matvec {} {rows}x{cols}", quant.name()),
                Shape::Matvec {
                    quant,
                    rows,
                    cols,
                    resid: false,
                },
            ));
        }
    }
    for (label, shape) in other_shapes() {
        shapes.push((label.to_string(), shape));
    }
    let geom = run.geometry.label();
    let want_router = named(&run.only, "router", "router35");
    let want_head = named(&run.only, "head", "q6k_head");
    let probe_selected = selects(&run.only, "matvec", Quant::Q4K.name(), STREAM_LABEL);

    let mut failed = false;
    let mut stream_gbps = None;
    // The bandwidth probe is the window arm only. Chain arms of a shape the
    // selection did not ask for would time a kernel the row is not about.
    if (want_router || want_head) && !probe_selected {
        let shape = stream_shape();
        let probe_run = Run {
            only: Vec::new(),
            trials: run.trials,
            chain: None,
            geometry: run.geometry,
            gaps: run.gaps,
            check: run.check,
        };
        let outcome = time_arms(
            &probe_run,
            shape.kind(),
            STREAM_LABEL,
            shape.bytes(),
            None,
            None,
            |arm| time_shape(&device, &kernels, &shape, arm, run.trials, run.check),
        );
        stream_gbps = outcome.window_gbps;
        failed |= outcome.failed;
        if outcome.stuck {
            return ExitCode::FAILURE;
        }
    }

    for (label, shape) in &shapes {
        if !selects(&run.only, shape.kind(), shape.quant().name(), label) {
            continue;
        }
        let outcome = time_arms(
            &run,
            shape.kind(),
            &format!("{label}{geom}"),
            shape.bytes(),
            None,
            None,
            |arm| time_shape(&device, &kernels, shape, arm, run.trials, run.check),
        );
        if label == STREAM_LABEL {
            stream_gbps = stream_gbps.or(outcome.window_gbps);
        }
        failed |= outcome.failed;
        if outcome.stuck {
            return ExitCode::FAILURE;
        }
    }

    // The routed experts at the session's shapes, both quants: the pair
    // (nine slots of 512 rows over 2048 columns) and the folded down (2048
    // rows over the slots' 512), eight routed experts and the shared one.
    for (kind, quant, rows, cols) in [
        ("pair", Quant::Q4K, EXPERT_K, 2048usize),
        ("pair", Quant::Q6K, EXPERT_K, 2048),
        ("pair", Quant::A4, EXPERT_K, 2048),
        ("fold", Quant::Q4K, 2048, EXPERT_K),
        ("fold", Quant::Q6K, 2048, EXPERT_K),
        ("fold", Quant::A4, 2048, EXPERT_K),
    ] {
        let label = match kind {
            "pair" => format!("pair {} 9x{rows}x{cols}", quant.name()),
            _ => format!("fold {} {rows}x{cols}x9", quant.name()),
        };
        if !selects(&run.only, kind, quant.name(), &label) {
            continue;
        }
        let bytes = EXPERT_SLOTS
            * expert_stack_bytes(quant, 1, rows, cols)
            * if kind == "pair" { 2 } else { 1 };
        let outcome = time_arms(
            &run,
            kind,
            &format!("{label}{geom}"),
            bytes,
            None,
            None,
            |arm| {
                if kind == "pair" {
                    time_pair(&device, &kernels, quant, arm, run.trials)
                } else {
                    time_fold(&device, &kernels, quant, arm, run.trials)
                }
            },
        );
        failed |= outcome.failed;
        if outcome.stuck {
            return ExitCode::FAILURE;
        }
    }

    // The dense mat-vec's phase cuts: the block loop unrolled two blocks a
    // step, at the stream shape each.
    for phase in MATVEC_PHASES {
        let (name, quant, _) = *phase;
        let shape = Shape::Matvec {
            quant,
            rows: 8192,
            cols: 4096,
            resid: false,
        };
        let label = format!("matvec {} 8192x4096 [{name}]{geom}", quant.name());
        if !selects(&run.only, "matvec", quant.name(), &label) {
            continue;
        }
        let phase_kernels = match matvec_phase_kernels(&device, *phase) {
            Ok(kernels) => kernels,
            Err(e) => {
                println!("phase {name}: FAIL ({e})");
                failed = true;
                continue;
            }
        };
        let outcome = time_arms(&run, "matvec", &label, shape.bytes(), None, None, |arm| {
            time_shape(&device, &phase_kernels, &shape, arm, run.trials, false)
        });
        failed |= outcome.failed;
        if outcome.stuck {
            return ExitCode::FAILURE;
        }
    }

    // The head-normed mat-vec's phase cuts, beside its own row: the scales
    // precomputed by the producer, and the apply cut down to the plain
    // residual path.
    let headnorm_shape = Shape::HeadNorm {
        rows: 2048,
        cols: 4096,
    };
    if selects(&run.only, "headnorm", Quant::Q4K.name(), "headnorm out") {
        for phase in HEADNORM_PHASES {
            let phase_kernels = match phase_kernels(&device, phase) {
                Ok(kernels) => kernels,
                Err(e) => {
                    println!("phase {}: FAIL ({e})", phase.name);
                    failed = true;
                    continue;
                }
            };
            let outcome = time_arms(
                &run,
                "headnorm",
                &format!("headnorm out 2048x4096 [{}]{}", phase.name, geom),
                headnorm_shape.bytes(),
                None,
                None,
                |arm| {
                    time_shape(
                        &device,
                        &phase_kernels,
                        &headnorm_shape,
                        arm,
                        run.trials,
                        false,
                    )
                },
            );
            failed |= outcome.failed;
            if outcome.stuck {
                return ExitCode::FAILURE;
            }
        }
    }

    let router_label = format!(
        "router35 {}x{} f32 top{}{geom}",
        router::ROWS,
        router::K,
        router::TOP_K_MAX
    );
    if want_router {
        let outcome = time_arms(
            &run,
            "router",
            &router_label,
            ROUTER_BYTES,
            Some(ROUTER_TAIL_US),
            stream_gbps,
            |arm| time_router(&device, &kernels, arm, run.trials),
        );
        failed |= outcome.failed;
        if outcome.stuck {
            return ExitCode::FAILURE;
        }
        for phase in ROUTER_PHASES {
            let phase_kernels = match phase_kernels(&device, phase) {
                Ok(kernels) => kernels,
                Err(e) => {
                    println!("phase {}: FAIL ({e})", phase.name);
                    failed = true;
                    continue;
                }
            };
            let outcome = time_arms(
                &run,
                "router",
                &format!("{router_label} [{}]", phase.name),
                ROUTER_BYTES,
                Some(ROUTER_TAIL_US),
                stream_gbps,
                |arm| time_router(&device, &phase_kernels, arm, run.trials),
            );
            failed |= outcome.failed;
            if outcome.stuck {
                return ExitCode::FAILURE;
            }
        }
    }
    let head_label = format!("q6k_head {HEAD_ROWS}x{HEAD_K} q6k{geom}");
    if want_head {
        let outcome = time_arms(
            &run,
            "head",
            &head_label,
            head_bytes(),
            None,
            stream_gbps,
            |arm| time_head(&device, &kernels, arm, run.trials),
        );
        failed |= outcome.failed;
        if outcome.stuck {
            return ExitCode::FAILURE;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
