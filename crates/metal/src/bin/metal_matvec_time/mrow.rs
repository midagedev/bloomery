//! The M-row rows of the mat-vec timing probe (`--only mrow`): the M-row
//! mat-vec family beside the 1-row launches it stands on, at the dense
//! Qwen3.8-27B decode shapes, on a device opened at the 27B's own instance
//! (as `d27.rs` opens it: a dense FFN of 17,408 values over a 5,120-wide
//! residual, 24/4 attention heads of 256, a gated delta net of 48 value
//! heads), so the pair runs at the one dense slot the session runs it at.
//!
//! Every shape times two arms over the same weight working set and the same
//! activation rows: the 1-row launch the session makes (`m=1`) and the M-row
//! launch at M = 2. A shape's figure is the window arm's median per-dispatch
//! GPU time, and each shape prints `t(2)/t(1)` beside it. The pair and the
//! head run their scaled entries, in both arms: the verify and the step read
//! the join's normed rows, the only form the M-row pair and head have.
//!
//! The shapes are the 27B step's own mat-vec launches — the two P1 segment
//! groups, the attention out, the dense FFN's down and gate-up, and the head
//! over the whole vocabulary — and the `mean` row is the step's M-row
//! mat-vec time over its 1-row time: each shape's ratio weighted by its share
//! of the 1-row step, its launches a step (48 delta layers, 16 attention
//! layers, 64 FFNs, one head) times the 1-row arm's own median. The delta
//! layers' out projection — the head-normed residual mat-vec — has no M-row
//! entry and no row here: its share is the one the mean leaves out.
//!
//! The bands the `expect` lines name are the MTP design's M-row decomposition
//! of a verify pass: the family earns its place only while the measured ratio
//! sits in them.

use std::process::ExitCode;

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord, LAYER_NONE};
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::{FfnAxes, Key, QWEN35MOE_GDN48, QWEN35MOE_GQA6, open_key};
use bloomery_metal::kernels::{
    HEAD_SIMD_GROUPS, HeadMParams, HeadParams, Kernels, LoadNorm, MROWS, MatvecMParams,
    MatvecParams, Mode, PairMParams, PairParams, Quant, SIMD_GROUPS_PER_GROUP, SegMParams,
    SegParams, Segment,
};

use super::{
    Run, WARMUPS, arms_of, dispatch_count, heat_of, print_fail, print_row, spread, time_dispatches,
    window_arm,
};

/// The 27B's residual width, dense FFN width and vocabulary.
const HIDDEN: usize = 5_120;
const FF: usize = 17_408;
const VOCAB: usize = 248_320;

/// The delta layer's P1 segments and the attention layer's, and the out
/// projections' columns: 24 heads of 256 and 48 value heads of 128.
const DELTA_SEGMENTS: [usize; 4] = [10_240, 6144, 48, 48];
const ATTN_SEGMENTS: [usize; 3] = [12_288, 1024, 1024];
const OUT_COLS: usize = 6144;

const QUANT: Quant = Quant::A4;

/// The M the family runs, beside the 1-row arm.
const M: usize = MROWS;

/// The epsilon every normed launch takes.
const EPS: f32 = 1e-6;

/// One launch of the 27B's step, as the probe times it.
#[derive(Clone, Copy)]
enum MShape {
    /// The segmented norm-on-load mat-vec of a P1 group.
    Seg {
        name: &'static str,
        segments: &'static [usize],
    },
    /// The plain mat-vec, adding into the stream.
    Matvec {
        name: &'static str,
        rows: usize,
        cols: usize,
    },
    /// The dense FFN's gate-up pair at the instance's one slot.
    Pair,
    /// The head: the logits dispatch and the pick, a row an activation row.
    Head,
}

/// The shapes of the 27B's step.
const SHAPES: [MShape; 6] = [
    MShape::Seg {
        name: "delta_p1",
        segments: &DELTA_SEGMENTS,
    },
    MShape::Seg {
        name: "attn_p1",
        segments: &ATTN_SEGMENTS,
    },
    MShape::Matvec {
        name: "attn_out",
        rows: HIDDEN,
        cols: OUT_COLS,
    },
    MShape::Matvec {
        name: "down",
        rows: HIDDEN,
        cols: FF,
    },
    MShape::Pair,
    MShape::Head,
];

/// Each shape's launches in one 27B step, in the shapes' order: the 48 gated
/// delta layers' P1, the 16 attention layers' P1 and out projection, the 64
/// layers' dense FFNs (a gate-up and a down each), and the one head.
const STEP_LAUNCHES: [u32; SHAPES.len()] = [48, 16, 16, 64, 64, 1];

/// Where the `mean` row's weights come from.
const STEP_BASIS: &str = "the 27B step's launches a shape times the 1-row arm's median";

/// The band the design's M-row decomposition predicts for the weighted mean at
/// M = 2.
const EXPECT: (f64, f64) = (1.05, 1.20);

/// The band the kernels' own instruction counts predict for the same mean:
/// each shape's time is the larger of its DRAM time and its arithmetic's, the
/// arithmetic being the loop's issue slots at 65 to 100 % of the GPU's issue
/// rate (16 cores of four schedulers at 1.296 GHz), and the 1-row time the
/// shape's own 1-row median.
const DERIVED: (f64, f64) = (0.95, 1.50);

impl MShape {
    /// The input columns.
    fn cols(&self) -> usize {
        match self {
            Self::Seg { .. } | Self::Pair | Self::Head => HIDDEN,
            Self::Matvec { cols, .. } => *cols,
        }
    }

    /// The output words an activation row takes.
    fn out_words(&self) -> usize {
        match self {
            Self::Seg { segments, .. } => segments.iter().sum(),
            Self::Matvec { .. } | Self::Pair => self.rows_of(),
            Self::Head => VOCAB,
        }
    }

    /// The weight rows (the pair's one tensor of gate and up).
    fn rows_of(&self) -> usize {
        match self {
            Self::Matvec { rows, .. } => *rows,
            Self::Pair => FF,
            _ => self.out_words(),
        }
    }

    /// The label the rows print.
    fn label(&self) -> String {
        match self {
            Self::Seg { name, segments } => format!(
                "mrow {name} seg a4 ({})x{HIDDEN}",
                segments
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("+")
            ),
            Self::Matvec { name, rows, cols } => {
                format!("mrow {name} a4 {rows}x{cols} resid")
            }
            Self::Pair => format!("mrow gate_up pair a4 scaled 1x{FF}x{HIDDEN}"),
            Self::Head => format!("mrow head a4_head_logits_scaled+argmax {VOCAB}x{HIDDEN}"),
        }
    }

    /// The weights one dispatch reads: the segments' tensors side by side,
    /// one tensor, or the pair's gate and then its up.
    fn weights(&self, seed: u64) -> Vec<u8> {
        let matrix =
            |rows: usize, seed: u64| super::synthetic::matrix(QUANT, rows, self.cols(), seed);
        match self {
            Self::Seg { segments, .. } => {
                let mut all = Vec::new();
                for (i, rows) in segments.iter().enumerate() {
                    all.extend_from_slice(&matrix(*rows, seed + i as u64));
                }
                all
            }
            Self::Pair => {
                // The gate, then the up: the working set spreads both.
                let mut both = matrix(FF, seed);
                both.extend_from_slice(&matrix(FF, seed ^ 1));
                both
            }
            _ => matrix(self.rows_of(), seed),
        }
    }

    /// The segments' places in the weights and in the output, the session's
    /// own layout.
    fn segments(&self) -> Vec<Segment> {
        let mut segs = Vec::new();
        let mut weight_blocks = 0usize;
        let mut out_offset = 0usize;
        if let Self::Seg { segments, .. } = self {
            for rows in segments.iter() {
                segs.push(Segment {
                    cols: HIDDEN,
                    rows: *rows,
                    weight_blocks,
                    out_offset,
                });
                weight_blocks += rows * (HIDDEN / QUANT.values_per_block());
                out_offset += rows;
            }
        }
        segs
    }
}

/// One shape's shared buffers: the working set every arm reads, the
/// activation rows every arm's M rows come from, the gain (whose first words
/// are the scaled entries' scale words), and the pair's one id.
struct ShapeBufs {
    weights: MetalBuffer<u8>,
    /// The one tensor's bytes (the window every dispatch takes).
    bytes: usize,
    stride: usize,
    copies: usize,
    x: MetalBuffer<f32>,
    gain: MetalBuffer<f32>,
    ids: MetalBuffer<i32>,
}

/// Whether the selection names the M-row rows.
pub(super) fn selected(only: &[String]) -> bool {
    super::lfm3::names(only, "mrow")
}

pub(super) fn main(run: &Run) -> ExitCode {
    let key = Key {
        hidden: HIDDEN as u32,
        ffn: FfnAxes::Dense { ff: FF as u32 },
        attn: QWEN35MOE_GQA6.key.attn,
        delta: QWEN35MOE_GDN48.key.delta,
    };
    let instance = match open_key(key) {
        Ok(instance) => instance,
        Err(e) => {
            println!("metal_matvec_time: FAIL (the 27B's instance: {e})");
            return ExitCode::FAILURE;
        }
    };
    let device = match MetalDevice::open(instance) {
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
    println!(
        "info\tinstrument=development-loop\tdevice={}\tlib={}\tbackend={}\tinstance={instance}\ttrials={}\tmrows=1,{M}",
        info.name,
        device.library().path().name(),
        <MetalDevice as Device>::BACKEND,
        run.trials,
    );

    print_pipelines(&device);

    let mut failed = false;
    // The window arm's median of each (shape, arm): [m=1, m=M].
    let mut medians: [Vec<Option<f64>>; 2] = Default::default();
    for (at, shape) in SHAPES.iter().enumerate() {
        let label = shape.label();
        let bufs = match shape_bufs(&device, shape, at as u64) {
            Ok(bufs) => bufs,
            Err(e) => {
                println!("mrow {label}: FAIL ({e})");
                failed = true;
                for row in &mut medians {
                    row.push(None);
                }
                continue;
            }
        };
        if let Err(e) = device.fault_word().clear(device.stream()).map_err(why) {
            println!("mrow {label}: FAIL (the fault word: {e})");
            failed = true;
        }
        for (row, m) in [1usize, M].into_iter().enumerate() {
            match time_one(run, &device, &kernels, shape, &bufs, m, &label) {
                Ok(us) => medians[row].push(Some(us)),
                Err(e) => {
                    println!("mrow {label} m={m}: FAIL ({e})");
                    failed = true;
                    medians[row].push(None);
                }
            }
        }
        match device.fault_word().read(device.stream()).map_err(why) {
            Ok(None) => {}
            Ok(Some(fault)) => {
                println!("mrow {label}: FAIL (a kernel raised {fault})");
                failed = true;
            }
            Err(e) => {
                println!("mrow {label}: FAIL (the fault word: {e})");
                failed = true;
            }
        }
    }

    // The ratio rows, one a shape, then the step-weighted mean.
    for (at, shape) in SHAPES.iter().enumerate() {
        let label = shape.label();
        let Some(one) = medians[0][at] else {
            println!("ratio\tshape={label}\tm={M}\tFAIL (the 1-row arm did not time)");
            continue;
        };
        match medians[1][at] {
            Some(us) => println!(
                "ratio\tshape={label}\tm={M}\tt1_us={one:.3}\tt{M}_us={us:.3}\tratio={:.4}",
                us / one
            ),
            None => println!("ratio\tshape={label}\tm={M}\tFAIL"),
        }
    }
    println!(
        "expect\tshape=mrow weighted mean\tm={M}\tratio={:.2}-{:.2}\tbasis=the MTP design's M-row decomposition",
        EXPECT.0, EXPECT.1
    );
    println!(
        "expect\tshape=mrow weighted mean\tm={M}\tratio={:.2}-{:.2}\tbasis=the kernels' instruction counts, issue at 65-100 % [derived]",
        DERIVED.0, DERIVED.1
    );
    match step_mean(&STEP_LAUNCHES, &medians[0], &medians[1]) {
        Some(mean) => println!(
            "mean\tshape=mrow weighted mean\tm={M}\tratio={:.4}\tstep1_us={:.1}\tshapes={}\tbasis={STEP_BASIS}",
            mean.ratio, mean.step_one_us, mean.shapes
        ),
        None => {
            println!("mean\tshape=mrow weighted mean\tm={M}\tFAIL (no shape timed both arms)");
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The pipelines whose resources the probe prints: the 1-row entries of the
/// four launches and the M-row entries the verify runs.
const PIPELINES: [&str; 9] = [
    "a4_matvec_resid",
    "a4_matvec_seg",
    "a4_pair_swiglu",
    "a4_head_logits",
    "a4_matvec_resid_m2",
    "a4_matvec_seg_m2",
    "a4_matvec_seg_scaled_m2",
    "a4_pair_swiglu_scaled_m2",
    "a4_head_logits_scaled_m2",
];

/// One `pipeline` row an entry: the most threads a threadgroup of it may hold
/// (the driver lowers it as the entry's registers rise), its SIMD width and
/// its threadgroup memory.
fn print_pipelines(device: &MetalDevice) {
    for entry in PIPELINES {
        match device
            .library()
            .pipeline_with_simdgroups(entry, SIMD_GROUPS_PER_GROUP as u32)
        {
            Ok(p) => println!(
                "pipeline\tentry={entry}\tmax_threads={}\twidth={}\tgroup_bytes={}",
                p.max_threads_per_group(),
                p.execution_width(),
                p.static_group_bytes()
            ),
            Err(e) => println!("pipeline\tentry={entry}\tFAIL ({e})"),
        }
    }
}

/// The step-weighted mean of the shapes' `t(m) / t(1)`.
struct StepMean {
    /// The step's M-row mat-vec time over its 1-row time.
    ratio: f64,
    /// The 1-row step's mat-vec time over the shapes that count, in µs.
    step_one_us: f64,
    /// The shapes that count: those whose two arms both timed.
    shapes: usize,
}

/// The mean ratio of the step's M-row mat-vec time to its 1-row time: each
/// shape's median is taken `launches` times a step, so a shape weighs its
/// share of the 1-row step and the mean is the sum of the M-row times over the
/// sum of the 1-row times. A shape either arm did not time leaves the mean.
/// `None` when no shape counts.
fn step_mean(launches: &[u32], one: &[Option<f64>], many: &[Option<f64>]) -> Option<StepMean> {
    let (mut step_one, mut step_many, mut shapes) = (0.0, 0.0, 0usize);
    for ((n, t1), tm) in launches.iter().zip(one).zip(many) {
        if let (Some(t1), Some(tm)) = (t1, tm) {
            step_one += f64::from(*n) * t1;
            step_many += f64::from(*n) * tm;
            shapes += 1;
        }
    }
    (step_one > 0.0).then(|| StepMean {
        ratio: step_many / step_one,
        step_one_us: step_one,
        shapes,
    })
}

/// One shape's shared buffers.
fn shape_bufs(device: &MetalDevice, shape: &MShape, at: u64) -> Result<ShapeBufs, String> {
    let stream = device.stream();
    let one = shape.weights(0x5D00 + at);
    let bytes = one.len();
    let (stride, copies, _) = spread(bytes);
    let mut all = vec![0u8; copies * stride];
    for c in 0..copies {
        all[c * stride..c * stride + bytes].copy_from_slice(&one);
    }
    let weights = device.upload(stream, &all).map_err(why)?;
    drop(all);
    // The family's activation rows; the 1-row arm reads the first of them.
    let x = device
        .upload(
            stream,
            &super::oracle::activations(M * shape.cols(), 0x5D40 + at, 1.0),
        )
        .map_err(why)?;
    let gain = device
        .upload(stream, &vec![1.0f32; shape.cols()])
        .map_err(why)?;
    let ids = device.upload(stream, &[0i32]).map_err(why)?;
    Ok(ShapeBufs {
        weights,
        bytes,
        stride,
        copies,
        x,
        gain,
        ids,
    })
}

/// Time every arm of one (shape, m) over the shared buffers, printing each
/// row; the window arm's median per-dispatch time is the figure.
fn time_one(
    run: &Run,
    device: &MetalDevice,
    kernels: &Kernels,
    shape: &MShape,
    bufs: &ShapeBufs,
    m: usize,
    label: &str,
) -> Result<f64, String> {
    let stream = device.stream();
    let rows = shape.out_words();
    let cols = shape.cols();
    // One output set for every arm: the arm with the most dispatches sizes
    // it, and the lone arm's per-buffer indices reach the whole set.
    let dispatches = arms_of(run.chain)
        .iter()
        .map(|a| dispatch_count(a, bufs.bytes))
        .max()
        .unwrap_or_else(|| dispatch_count(&window_arm(), bufs.bytes));
    let buffers = WARMUPS + run.trials;
    let heat = heat_of(&window_arm(), bufs.bytes);
    let outputs = heat + buffers * dispatches;
    let mut y = device
        .zeroed::<f32>(stream, Elems(outputs * m * rows))
        .map_err(why)?;
    let mut picks = device
        .zeroed::<u32>(stream, Elems(outputs * m))
        .map_err(why)?;
    let segments = shape.segments();
    let mut window_median = None;
    for arm in arms_of(run.chain) {
        let work = super::TimedWork {
            bytes: bufs.bytes,
            dispatches: dispatch_count(&arm, bufs.bytes),
            copies: bufs.copies,
            working_set_bytes: bufs.copies * bufs.stride,
            trials: run.trials,
            arm: &arm,
            heat: heat_of(&arm, bufs.bytes),
        };
        let one_dispatch = dispatch_count(&arm, bufs.bytes);
        let timing = time_dispatches(stream, &work, |encoder, n| {
            let w = bufs
                .weights
                .window(Elems((n % bufs.copies) * bufs.stride), Elems(bufs.bytes))
                .map_err(why)?;
            let out = y
                .window_mut(Elems(n * m * rows), Elems(m * rows))
                .map_err(why)?;
            let b = n / one_dispatch;
            let gain = || {
                bufs.gain
                    .window(Elems(0), Elems(cols))
                    .map_err(|e| e.to_string())
            };
            // The scaled entries' scale words, one a row.
            let scale = |words: usize| {
                bufs.gain
                    .window(Elems(0), Elems(words))
                    .map_err(|e| e.to_string())
            };
            match (shape, m) {
                (MShape::Matvec { rows, cols, .. }, 1) => kernels
                    .matvec(
                        encoder,
                        MatvecParams {
                            quant: QUANT,
                            rows: *rows,
                            cols: *cols,
                            weights: w,
                            x: bufs.x.window(Elems(0), Elems(*cols)).map_err(why)?,
                            y: out,
                            mode: Mode::Residual,
                        },
                    )
                    .map_err(|e| e.to_string()),
                (MShape::Matvec { rows, cols, .. }, _) => kernels
                    .matvec_m(
                        encoder,
                        MatvecMParams {
                            quant: QUANT,
                            m,
                            rows: *rows,
                            cols: *cols,
                            weights: w,
                            x: bufs.x.window(Elems(0), Elems(m * cols)).map_err(why)?,
                            y: out,
                        },
                    )
                    .map_err(|e| e.to_string()),
                (MShape::Seg { .. }, 1) => kernels
                    .matvec_seg(
                        encoder,
                        SegParams {
                            quant: QUANT,
                            segments: &segments,
                            weights: w,
                            x: bufs.x.window(Elems(0), Elems(cols)).map_err(why)?,
                            y: out,
                            norm: LoadNorm::Gain {
                                gain: gain()?,
                                eps: EPS,
                            },
                            fault: device.fault_word(),
                            layer: LAYER_NONE,
                        },
                    )
                    .map_err(|e| e.to_string()),
                (MShape::Seg { .. }, _) => kernels
                    .matvec_seg_m(
                        encoder,
                        SegMParams {
                            quant: QUANT,
                            m,
                            segments: &segments,
                            weights: w,
                            x: bufs.x.window(Elems(0), Elems(m * cols)).map_err(why)?,
                            y: out,
                            norm: LoadNorm::Gain {
                                gain: gain()?,
                                eps: EPS,
                            },
                            fault: device.fault_word(),
                            layer: LAYER_NONE,
                        },
                    )
                    .map_err(|e| e.to_string()),
                (MShape::Pair, 1) => {
                    let (gate, up) = pair_halves(&w)?;
                    kernels
                        .pair(
                            encoder,
                            PairParams {
                                quant: QUANT,
                                shared_quant: QUANT,
                                slots: 1,
                                k: FF,
                                cols,
                                n_experts: 1,
                                ids: bufs.ids.window(Elems(0), Elems(1)).map_err(why)?,
                                gate,
                                up,
                                shared_gate: None,
                                shared_up: None,
                                x: bufs.x.window(Elems(0), Elems(cols)).map_err(why)?,
                                norm: LoadNorm::Scaled { scale: scale(1)? },
                                y: out,
                                out_stride: FF,
                                fault: device.fault_word(),
                                layer: LAYER_NONE,
                            },
                        )
                        .map_err(|e| e.to_string())
                }
                (MShape::Pair, _) => {
                    let (gate, up) = pair_halves(&w)?;
                    kernels
                        .pair_m(
                            encoder,
                            PairMParams {
                                quant: QUANT,
                                shared_quant: QUANT,
                                m,
                                slots: 1,
                                k: FF,
                                cols,
                                n_experts: 1,
                                ids: bufs.ids.window(Elems(0), Elems(1)).map_err(why)?,
                                gate,
                                up,
                                shared_gate: None,
                                shared_up: None,
                                x: bufs.x.window(Elems(0), Elems(m * cols)).map_err(why)?,
                                scale: scale(m)?,
                                y: out,
                                out_stride: FF,
                                fault: device.fault_word(),
                                layer: LAYER_NONE,
                            },
                        )
                        .map_err(|e| e.to_string())
                }
                (MShape::Head, 1) => kernels
                    .head(
                        encoder,
                        HeadParams {
                            quant: QUANT,
                            rows: VOCAB,
                            k: HIDDEN,
                            simdgroups: HEAD_SIMD_GROUPS,
                            weights: w,
                            x: bufs.x.window(Elems(0), Elems(HIDDEN)).map_err(why)?,
                            norm: LoadNorm::Scaled { scale: scale(1)? },
                            logits: out,
                            step_id: picks.window_mut(Elems(b), Elems(1)).map_err(why)?,
                            fault: device.fault_word(),
                            layer: LAYER_NONE,
                        },
                    )
                    .map_err(|e| e.to_string()),
                (MShape::Head, _) => kernels
                    .head_m(
                        encoder,
                        HeadMParams {
                            quant: QUANT,
                            m,
                            rows: VOCAB,
                            k: HIDDEN,
                            simdgroups: HEAD_SIMD_GROUPS,
                            weights: w,
                            x: bufs.x.window(Elems(0), Elems(m * HIDDEN)).map_err(why)?,
                            scale: scale(m)?,
                            logits: out,
                            step_ids: picks.window_mut(Elems(b * m), Elems(m)).map_err(why)?,
                            fault: device.fault_word(),
                            layer: LAYER_NONE,
                        },
                    )
                    .map_err(|e| e.to_string()),
            }
        });
        match timing {
            Ok(t) => {
                print_row(
                    "mrow",
                    &format!("{label} m={m}"),
                    arm.name,
                    run.trials,
                    &t,
                    bufs.bytes,
                    "",
                );
                if arm.name == "window" {
                    window_median = Some(t.us_median);
                }
            }
            Err(e) => {
                print_fail("mrow", &format!("{label} m={m}"), arm.name, &e);
                return Err(e);
            }
        }
    }
    window_median.ok_or_else(|| "the window arm did not time".to_string())
}

/// A weights window of one tensor of the working set.
type Weights<'w> = bloomery_device::Window<'w, u8, MetalDevice>;

/// The pair's gate and up windows: the working set's tensor holds the gate,
/// then the up, half a half.
fn pair_halves<'w>(w: &Weights<'w>) -> Result<(Weights<'w>, Weights<'w>), String> {
    let half = w.len().get() / 2;
    Ok((
        w.sub(Elems(0), Elems(half)).map_err(|e| e.to_string())?,
        w.sub(Elems(half), Elems(half)).map_err(|e| e.to_string())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean is the M-row step time over the 1-row step time: a shape of
    /// many launches weighs its launches times its 1-row median, and a shape
    /// either arm did not time leaves the sum.
    #[test]
    fn the_mean_is_the_step_time_ratio() {
        // Ten launches of a shape of 100 over 1 and 120 over M, one of a shape
        // of 400 and 800: (1200 + 800) / (1000 + 400).
        let mean = step_mean(
            &[10, 1],
            &[Some(100.0), Some(400.0)],
            &[Some(120.0), Some(800.0)],
        )
        .expect("two shapes timed");
        assert_eq!(mean.shapes, 2);
        assert!((mean.step_one_us - 1400.0).abs() < 1e-9);
        assert!((mean.ratio - 2000.0 / 1400.0).abs() < 1e-12);
        // The second shape's M-row arm failed: the mean is the first's ratio.
        let mean = step_mean(&[10, 1], &[Some(100.0), Some(400.0)], &[Some(120.0), None])
            .expect("one shape timed");
        assert_eq!(mean.shapes, 1);
        assert!((mean.ratio - 1.2).abs() < 1e-12);
        assert!(step_mean(&[10], &[None], &[Some(1.0)]).is_none());
        assert!(step_mean(&[], &[], &[]).is_none());
    }

    /// The launches a step makes are one count a shape, in the shapes' order,
    /// and the P1 of 48 delta layers outweighs the 16 attention layers'.
    #[test]
    fn the_step_launches_are_a_count_a_shape() {
        assert_eq!(STEP_LAUNCHES.len(), SHAPES.len());
        assert_eq!(STEP_LAUNCHES[0], 48);
        assert_eq!(STEP_LAUNCHES[1], 16);
        assert!(matches!(
            SHAPES[0],
            MShape::Seg {
                name: "delta_p1",
                ..
            }
        ));
        assert!(matches!(
            SHAPES[1],
            MShape::Seg {
                name: "attn_p1",
                ..
            }
        ));
        assert!(matches!(SHAPES[5], MShape::Head));
    }
}
