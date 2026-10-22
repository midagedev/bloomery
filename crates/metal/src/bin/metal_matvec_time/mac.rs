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
//! head (`q6k_head_argmax`, 248320 rows of K 2048 Q6_K). The f64 oracle's
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
//! Router and head rows also carry the time their weight bytes take at the
//! stream bandwidth measured in this run — the `window` arm of `q4k
//! 8192x4096`, run for that purpose when the selection does not already
//! time it — and the review's predicted tail span. The measured tail of a
//! row is its `us_median` minus that byte time. When that arm has no
//! bandwidth, the row still prints its own microseconds and marks the byte
//! time unavailable, by name.
//!
//! Flags: `--only` (repeatable; comma-separated `matvec`, `seg`, `headnorm`,
//! `q4k`, `q6k`, `router`, `head`, or a shape label), `--trials N`,
//! `--chain N`, `--gaps`, `--no-check`, `--help`.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord, LAYER_NONE, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::kernels::router::{self, RouterParams};
use bloomery_metal::kernels::{
    BLOCK_VALUES, HEAD_GROUPS, HEAD_K, HEAD_SIMD_GROUPS, HEAD_VOCAB, HeadNormParams, HeadParams,
    Kernels, MatvecParams, Mode, NormOnLoad, PARTIALS_PER_HEAD, Quant, SegParams, Segment,
    VALUE_HEADS,
};
use bloomery_metal::{GpuInterval, MetalStream};

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

/// The head's weight bytes: 248320 rows of K 2048 Q6_K. The design's head
/// matrix, and the kernel's own shape.
const HEAD_BYTES: usize = HEAD_VOCAB * (HEAD_K / BLOCK_VALUES) * Quant::Q6K.block_bytes();

/// The review's predicted tail span for one router dispatch, in the row's
/// microsecond fields.
const ROUTER_TAIL_US: (f64, f64) = (11.0, 123.0);

/// The review's predicted tail span for one head dispatch, in the row's
/// microsecond fields.
const HEAD_TAIL_US: (f64, f64) = (62.0, 683.0);

const _: () = assert!(router::ROWS == 257 && router::K == 2048);
const _: () = assert!(router::TOP_K_MAX == 8);
const _: () = assert!(HEAD_VOCAB == 248_320 && HEAD_K == 2048);
const _: () = assert!(ROUTER_BYTES == 2_105_344 && ROUTER_BYTES.is_multiple_of(256));
const _: () = assert!(HEAD_BYTES == 417_177_600 && HEAD_BYTES.is_multiple_of(256));
const _: () = assert!(HEAD_GROUPS * 16 == HEAD_VOCAB);

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
                encode(&mut encoder, b * work.dispatches + j)?;
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
    };
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
            copies,
            working_set_bytes: working_set,
            trials,
            arm,
        },
        |encoder, n| {
            let w = weights
                .window(Elems((n % copies) * stride), Elems(bytes))
                .map_err(why)?;
            let out = y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?;
            match shape {
                Shape::Matvec { quant, rows, cols } => {
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
                                mode: Mode::Store,
                            },
                        )
                        .map_err(why)?;
                }
                Shape::Seg { quant, cols, .. } => {
                    kernels
                        .matvec_seg(
                            encoder,
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
                            encoder,
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
    if check && matches!(shape, Shape::Matvec { .. }) {
        let x64: Vec<f64> = x_values.iter().map(|&v| f64::from(v)).collect();
        let reference = oracle::reference(quant, &one, rows, cols, &x64)?;
        let band = oracle::band(quant, cols / BLOCK_VALUES, oracle::Kind::Store);
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
                        shared: weights
                            .window(Elems(base + router::EXPERTS * router::K), Elems(router::K))
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
    let one = matrix(Quant::Q6K, HEAD_VOCAB, HEAD_K, 0x6EAD);
    let bytes = one.len();
    if bytes != HEAD_BYTES {
        return Err(format!(
            "the head weights are {bytes} bytes, the kernel's shape is {HEAD_BYTES}"
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
    // One scratch set per committed buffer, as for the router.
    let mut candidates = device
        .zeroed::<u64>(stream, Elems(buffers * HEAD_GROUPS))
        .map_err(why)?;
    let mut ticket = device.zeroed::<u32>(stream, Elems(buffers)).map_err(why)?;
    let mut step_id = device.zeroed::<u32>(stream, Elems(buffers)).map_err(why)?;
    let rows = HEAD_VOCAB;
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
        },
        |encoder, n| {
            let b = n / dispatches;
            kernels
                .head(
                    encoder,
                    HeadParams {
                        rows: HEAD_VOCAB,
                        k: HEAD_K,
                        simdgroups: HEAD_SIMD_GROUPS,
                        weights: weights
                            .window(Elems((n % copies) * stride), Elems(bytes))
                            .map_err(why)?,
                        x: x.window(Elems(0), Elems(HEAD_K)).map_err(why)?,
                        gain: gain.window(Elems(0), Elems(HEAD_K)).map_err(why)?,
                        logits: y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?,
                        candidates: candidates
                            .window_mut(Elems(b * HEAD_GROUPS), Elems(HEAD_GROUPS))
                            .map_err(why)?,
                        ticket: ticket.window_mut(Elems(b), Elems(1)).map_err(why)?,
                        step_id: step_id.window_mut(Elems(b), Elems(1)).map_err(why)?,
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

/// The `--trials` value: a whole number of at least one, refused by name
/// otherwise (no trial has no median to take).
fn trials_of(value: Option<&String>) -> Result<usize, String> {
    match value.map(|v| (v, v.parse::<usize>())) {
        None => Err("--trials takes a value".to_string()),
        Some((_, Ok(n))) if n >= 1 => Ok(n),
        Some((value, _)) => Err(format!(
            "--trials takes a whole number of at least 1, not {value:?}"
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
    }
}

struct Run {
    only: Vec<String>,
    trials: usize,
    chain: Option<usize>,
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
                         mat-vec type (q4k, q6k), or a shape label. Repeatable,
                         and one value may be comma-separated. Unset times every
                         shape.
  --trials N             timed command buffers per arm (default 16)
  --chain N              also time N back-to-back dispatches: arm lone (one
                         dispatch), serial and concurrent (a memory barrier
                         between each). cb_us is (chain time − N × lone time) / N
  --gaps                 print every timed buffer's GPU time, in microseconds
  --no-check             skip the plain mat-vec's f64 oracle check"
    );
}

fn parse(args: &[String]) -> Result<Run, String> {
    let mut only = Vec::new();
    let mut trials = 16usize;
    let mut chain = None;
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
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let chain = run
        .chain
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "info\tinstrument=development-loop\tdevice={}\tlib={}\tbackend={}\ttrials={}\twarmups={WARMUPS}\twindow_bytes={WINDOW_BYTES}\tworking_set_bytes={WORKING_SET_BYTES}\tchain={chain}",
        info.name,
        device.library().path().name(),
        <MetalDevice as Device>::BACKEND,
        run.trials,
    );

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
    let want_router = named(&run.only, "router", "router35");
    let want_head = named(&run.only, "head", "q6k_head_argmax");
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
            label,
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

    let router_label = format!(
        "router35 {}x{} f32 top{}",
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
    }
    let head_label = format!("q6k_head_argmax {HEAD_VOCAB}x{HEAD_K} q6k");
    if want_head {
        let outcome = time_arms(
            &run,
            "head",
            &head_label,
            HEAD_BYTES,
            Some(HEAD_TAIL_US),
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
