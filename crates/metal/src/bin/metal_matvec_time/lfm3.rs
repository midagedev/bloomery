//! The LFM3 rows of the mat-vec timing probe (`--only lfm3`).
//!
//! The rows are LFM3-1.7B's decode launches, each through the host call the
//! session makes, over A4 synthetic weights (group 64, 4 bits, the MLX layout
//! the loader produces), on a device opened at LFM3's own instance (a dense
//! FFN of 7680 values over a 2048-wide residual), so the pair and the fold run
//! at the one dense slot the session runs them at:
//!
//! - conv P1: one 6144-row segment, the norm on the load (`matvec_seg`);
//! - attention P1: segments of 2048, 512, 512 and 16 rows (`matvec_seg`);
//! - an out projection: 2048 by 2048, `Mode::Store` (`matvec`);
//! - the dense FFN's gate-up: the experts' pair over one gate and one up of
//!   7680 by 2048 (`pair`);
//! - the dense FFN's down: 7680 columns by 2048 rows through `matvec` with
//!   `Mode::Store`, which is what a sandwich session launches, and through the
//!   experts' fold at its one slot, which is what a plain-residual dense
//!   session launches;
//! - the head: the logits dispatch, a barrier and `logits_argmax`, over 128000
//!   rows of 2048.
//!
//! Beside those the probe times diagnostic rows, labelled in brackets, that
//! split a row into terms. The head is also timed as a plain mat-vec of its
//! shape (`[plain]`: no norm, no pick), as a one-segment norm-on-load mat-vec
//! (`[norm]`: the norm, no pick) and as the pick alone (`[pick]`:
//! `logits_argmax` over the zero logits the output buffer holds), so that the
//! head's time is `[plain]` plus the norm's term (`[norm]` less `[plain]`)
//! plus the pick's term (the head less `[norm]`, which `[pick]` reads on its
//! own). A plain mat-vec of the gate's shape is timed beside its norm-on-load
//! form and four cuts of the A4 row loop: `[fma]` (the dot accumulated with
//! `fma`, the one float change), `[scalarx]` (the activation and gain loads as
//! scalar floats, the shipped loads' predecessor: the same bits), `[unroll]`
//! (the walk over a SIMD group's rows forced to unroll: the same bits) and
//! `[lite]` (the dot reduced to a conversion and an add a word, the weight
//! reads kept: what the loads alone cost).
//!
//! `--only lfm3,sweep` times every row (but the cuts) once more under each
//! geometry of `SWEEP` (rows a SIMD group and SIMD groups a threadgroup),
//! the variants compiled in this process and timed right after the default, so
//! a geometry is judged against the default in one clock state. A variant's
//! rows carry the geometry in their label. `--only lfm3,overlap` adds an
//! `overlap` arm to every row: the window's dispatches in one concurrent
//! encoder with no barrier between them. `d27.rs` runs its rows through the
//! same machinery ([`run_rows`]).
//!
//! A cut is a copy of the instance's program text with counted edits, the way
//! the phase cuts of the other rows are: each find text must occur exactly
//! once, so a kernel edit that moves one is a named error, not a silent no-op.
//! A cut is a development instrument, not a kernel; its rows skip the f64
//! oracle and are not timed under `--geometry`.
//!
//! Each row with a prediction prints an `expect` line first: the band its
//! window arm's `gbps` is predicted to fall in, and the measured line it was
//! derived from. The bands are derived from the session's `--profile` groups
//! (a line of fixed microseconds plus microseconds a megabyte per kernel
//! family), so a row outside its band names the term the line got wrong.

use std::process::ExitCode;

use bloomery_device::conformance::why;
use bloomery_device::{Device, DeviceError};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::{FfnAxes, Key, LFM3_ATTN, open_key};
use bloomery_metal::kernels::{Kernels, Quant};
use bloomery_metal::library::{Library, sources_for};

use super::{
    Arm, Geometry, HEAT_BYTES, HeadPart, Run, Shape, WARMUPS, WINDOW_BYTES, WORKING_SET_BYTES,
    geometry_kernels, print_fail, print_row, time_arms, time_shape,
};

/// LFM3-1.7B's residual width, dense FFN width and vocabulary.
const HIDDEN: u32 = 2048;
const FF: u32 = 7680;
const VOCAB: usize = 128_000;

/// The conv layer's input projection (B, C and x), and the attention
/// layer's P1 segments: the query, the key, the value, and one gate value a
/// head.
const CONV_ROWS: usize = 6144;
const ATTN_SEGMENTS: [usize; 4] = [2048, 512, 512, 16];

/// One counted text edit of a cut: the text to find, once, and its
/// replacement.
pub(super) type Edit = (&'static str, &'static str);

/// A phase cut of the instance's program text.
pub(super) struct Cut {
    pub(super) name: &'static str,
    pub(super) edits: &'static [Edit],
}

/// The A4 block's accumulate lines, in the dot's own order.
const ACC_FIND: &str = concat!(
    "            acc[0] += lane.yl[4 * m + 0] * (words[m] & 0x000F);\n",
    "            acc[1] += lane.yl[4 * m + 1] * (words[m] & 0x00F0);\n",
    "            acc[2] += lane.yl[4 * m + 2] * (words[m] & 0x0F00);\n",
    "            acc[3] += lane.yl[4 * m + 3] * (words[m] & 0xF000);\n",
);

/// The same four lines with `fma`: the contracted form of the products.
const ACC_FMA: &str = concat!(
    "            acc[0] = fma(lane.yl[4 * m + 0], float(words[m] & 0x000F), acc[0]);\n",
    "            acc[1] = fma(lane.yl[4 * m + 1], float(words[m] & 0x00F0), acc[1]);\n",
    "            acc[2] = fma(lane.yl[4 * m + 2], float(words[m] & 0x0F00), acc[2]);\n",
    "            acc[3] = fma(lane.yl[4 * m + 3], float(words[m] & 0xF000), acc[3]);\n",
);

/// The dot reduced to one conversion and one add a code word: the weight and
/// the scale and bias reads stay, and the per-value work goes.
const ACC_LITE: &str = "            acc[0] += float(words[m]);\n";

/// The activation and gain loads of every row loop: four values in one
/// `float4` load.
const ACT4_FIND: &str = "    return *(device const float4 *)p;\n";

/// The same four values as four scalar float loads: the loads the row loops
/// had before `act4`.
const ACT4_SCALAR: &str = "    return float4(p[0], p[1], p[2], p[3]);\n";

static UNROLL: Cut = Cut {
    name: "unroll",
    edits: &[(ROWS_FIND, ROWS_UNROLL)],
};
static FMA: Cut = Cut {
    name: "fma",
    edits: &[(ACC_FIND, ACC_FMA)],
};
/// The row loop's walk over a SIMD group's rows, left to the compiler's choice
/// of unrolling.
const ROWS_FIND: &str = concat!(
    "        for (short row = 0; row < NR0; ++row) {\n",
    "            if constexpr (HEADNORM) {\n",
);

/// The same walk with the unroll forced: the four rows' weight loads are then
/// all in sight of the scheduler at once. The same bits.
const ROWS_UNROLL: &str = concat!(
    "        FOR_UNROLL (short row = 0; row < NR0; ++row) {\n",
    "            if constexpr (HEADNORM) {\n",
);

static SCALARX: Cut = Cut {
    name: "scalarx",
    edits: &[(ACT4_FIND, ACT4_SCALAR)],
};
static LITE: Cut = Cut {
    name: "lite",
    edits: &[(ACC_FIND, ACC_LITE)],
};

/// The geometries a `sweep` run times beside the default: (rows a SIMD group,
/// SIMD groups a threadgroup), the counts the kernels take (2, 4 or 8 rows; a
/// threadgroup of up to 1024 threads).
const SWEEP: [(usize, usize); 5] = [(2, 2), (4, 4), (4, 8), (8, 2), (8, 4)];

/// The band a row's window arm is predicted to fall in, and the measured
/// line it was derived from.
pub(super) struct Expect {
    lo: f64,
    hi: f64,
    basis: &'static str,
}

/// One probe row.
pub(super) struct Row {
    pub(super) label: String,
    pub(super) shape: Shape,
    pub(super) expect: Option<Expect>,
    pub(super) cut: Option<&'static Cut>,
}

fn a4_matvec(rows: usize, cols: usize) -> Shape {
    Shape::Matvec {
        quant: Quant::A4,
        rows,
        cols,
        resid: false,
    }
}

fn a4_seg(segments: &[usize]) -> Shape {
    Shape::Seg {
        quant: Quant::A4,
        cols: HIDDEN as usize,
        segments: segments.to_vec(),
        scaled: false,
    }
}

pub(super) fn expect(lo: f64, hi: f64, basis: &'static str) -> Option<Expect> {
    Some(Expect { lo, hi, basis })
}

/// Every row, in the order the session meets them, then the diagnostics.
fn rows() -> Vec<Row> {
    let hidden = HIDDEN as usize;
    let ff = FF as usize;
    let head = |part: HeadPart| Shape::Head {
        quant: Quant::A4,
        rows: VOCAB,
        cols: hidden,
        part,
        scaled: false,
    };
    let loop_shape = || a4_matvec(ff, hidden);
    let mut rows = vec![
        Row {
            label: format!("lfm3 conv_p1 seg a4 {CONV_ROWS}x{hidden}"),
            shape: a4_seg(&[CONV_ROWS]),
            expect: expect(84.0, 100.0, "profile conv_p1 90.0 GB/s"),
            cut: None,
        },
        Row {
            label: format!("lfm3 attn_p1 seg a4 (2048+512+512+16)x{hidden}"),
            shape: a4_seg(&ATTN_SEGMENTS),
            expect: expect(65.0, 85.0, "profile attn_p1 73.2 GB/s"),
            cut: None,
        },
        Row {
            label: format!("lfm3 attn_out conv_out matvec a4 {hidden}x{hidden} store"),
            shape: a4_matvec(hidden, hidden),
            expect: expect(78.0, 95.0, "profile attn_out 86.5 and conv_out 84.2 GB/s"),
            cut: None,
        },
        Row {
            label: format!("lfm3 gate_up pair a4 1x{ff}x{hidden}"),
            shape: Shape::DensePair {
                quant: Quant::A4,
                rows: ff,
                cols: hidden,
                scaled: false,
            },
            expect: expect(98.0, 115.0, "profile gate_up 105.3 GB/s"),
            cut: None,
        },
        Row {
            label: format!("lfm3 down matvec a4 {hidden}x{ff} store"),
            shape: a4_matvec(hidden, ff),
            expect: expect(105.0, 122.0, "profile down 114.2 GB/s"),
            cut: None,
        },
        Row {
            label: format!("lfm3 down fold a4 {hidden}x{ff}x1"),
            shape: Shape::DenseFold {
                quant: Quant::A4,
                rows: hidden,
                k: ff,
            },
            expect: expect(
                80.0,
                114.0,
                "no profile row: the down's matvec plus a per-block id read and address",
            ),
            cut: None,
        },
        Row {
            label: format!("lfm3 head a4_head_logits+logits_argmax {VOCAB}x{hidden}"),
            shape: head(HeadPart::Pair),
            expect: expect(
                105.0,
                123.0,
                "derived: the round's predicted 1.2 to 1.4 ms over the head's 147.5 MB",
            ),
            cut: None,
        },
        Row {
            label: format!("lfm3 head [plain] matvec a4 {VOCAB}x{hidden} store"),
            shape: a4_matvec(VOCAB, hidden),
            expect: expect(
                112.0,
                145.0,
                "plain family line of the profile: 9.7 us + 7.67 us/MB",
            ),
            cut: None,
        },
        Row {
            label: format!("lfm3 head [norm] seg a4 1x{VOCAB}x{hidden}"),
            shape: a4_seg(&[VOCAB]),
            expect: expect(
                102.0,
                128.0,
                "norm family line of the profile: 19.2 us + 8.41 us/MB",
            ),
            cut: None,
        },
    ];
    rows.push(Row {
        label: format!("lfm3 head [pick] logits_argmax {VOCAB}"),
        shape: head(HeadPart::Pick),
        expect: None,
        cut: None,
    });
    rows.push(Row {
        label: format!("lfm3 loop matvec a4 {ff}x{hidden} store"),
        shape: loop_shape(),
        expect: expect(
            105.0,
            122.0,
            "plain family line of the profile: 9.7 us + 7.67 us/MB",
        ),
        cut: None,
    });
    rows.push(Row {
        label: format!("lfm3 loop [norm] seg a4 1x{ff}x{hidden}"),
        shape: a4_seg(&[ff]),
        expect: expect(
            86.0,
            104.0,
            "norm family line of the profile: 19.2 us + 8.41 us/MB",
        ),
        cut: None,
    });
    for cut in [&SCALARX, &UNROLL, &FMA, &LITE] {
        rows.push(Row {
            label: format!("lfm3 loop matvec a4 {ff}x{hidden} store [{}]", cut.name),
            shape: loop_shape(),
            expect: None,
            cut: Some(cut),
        });
    }
    // The norm-on-load rows carry the gain loads too, so the scalar loads'
    // cost is read on them as well: the segmented loop shape, the gate-up pair
    // and the head's norm row.
    rows.push(Row {
        label: format!("lfm3 loop [norm] seg a4 1x{ff}x{hidden} [{}]", SCALARX.name),
        shape: a4_seg(&[ff]),
        expect: None,
        cut: Some(&SCALARX),
    });
    rows.push(Row {
        label: format!("lfm3 gate_up pair a4 1x{ff}x{hidden} [{}]", SCALARX.name),
        shape: Shape::DensePair {
            quant: Quant::A4,
            rows: ff,
            cols: hidden,
            scaled: false,
        },
        expect: None,
        cut: Some(&SCALARX),
    });
    rows.push(Row {
        label: format!(
            "lfm3 head [norm] seg a4 1x{VOCAB}x{hidden} [{}]",
            SCALARX.name
        ),
        shape: a4_seg(&[VOCAB]),
        expect: None,
        cut: Some(&SCALARX),
    });
    rows
}

/// The selection names the LFM3 rows.
pub(super) fn selected(only: &[String]) -> bool {
    names(only, "lfm3")
}

/// The selection names `token`.
pub(super) fn names(only: &[String], token: &str) -> bool {
    only.iter()
        .flat_map(|spec| spec.split(','))
        .any(|tok| tok.trim() == token)
}

/// The selection also names the geometry sweep.
fn swept(only: &[String]) -> bool {
    names(only, "sweep")
}

/// The selection also names the overlap arm: the window's dispatches in one
/// concurrent encoder with no barrier, the condition of MLX's own window
/// (independent matmuls in one eval run concurrently).
fn overlapped(only: &[String]) -> bool {
    names(only, "overlap")
}

/// A row is timed when the selection names no further filter, or one of its
/// further values is the row's kernel or a part of its label. `family` is the
/// token that selected the rows.
fn wanted(only: &[String], family: &str, kind: &str, label: &str) -> bool {
    let filters: Vec<&str> = only
        .iter()
        .flat_map(|spec| spec.split(','))
        .map(str::trim)
        .filter(|tok| ![family, "sweep", "overlap"].contains(tok))
        .collect();
    filters.is_empty()
        || filters
            .iter()
            .any(|tok| *tok == kind || label.contains(*tok))
}

/// The kernels of a cut: the instance's program text with the cut's edits,
/// each find text occurring once, compiled by the runtime.
fn cut_kernels(device: &MetalDevice, source: &str, cut: &Cut) -> Result<Kernels, String> {
    let mut text = source.to_string();
    for &(find, with) in cut.edits {
        let found = text.matches(find).count();
        if found != 1 {
            return Err(format!(
                "cut {}: a find text occurs {found} times, not 1; the kernel moved",
                cut.name
            ));
        }
        text = text.replacen(find, with, 1);
    }
    let library = Library::from_source_text(device, &text).map_err(why)?;
    Kernels::from_library(&library).map_err(why)
}

pub(super) fn main(run: &Run) -> ExitCode {
    let key = Key {
        hidden: HIDDEN,
        ffn: FfnAxes::Dense { ff: FF },
        attn: LFM3_ATTN.key.attn,
        delta: None,
    };
    run_rows(run, key, "lfm3", rows())
}

/// Time `rows` on a device opened at `key`, the instance of the model the
/// selection token `family` names, under the run's geometry and, when the
/// selection names it, the sweep's.
pub(super) fn run_rows(run: &Run, key: Key, family: &str, rows: Vec<Row>) -> ExitCode {
    let instance = match open_key(key) {
        Ok(instance) => instance,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({family}'s instance: {e})");
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
    let source = match sources_for(instance) {
        Ok((source, _tensor)) => source,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let built = if run.geometry.is_default() {
        Kernels::new(&device).map_err(why)
    } else {
        geometry_kernels(&device, &source, run.geometry)
    };
    let kernels = match built {
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
        "info\tinstrument=development-loop\tdevice={}\tlib={}\tbackend={}\tinstance={instance}\ttrials={}\twarmups={WARMUPS}\theat_bytes={HEAT_BYTES}\twindow_bytes={WINDOW_BYTES}\tworking_set_bytes={WORKING_SET_BYTES}\tgeom=rows{},sg{},router_sg{}",
        info.name,
        device.library().path().name(),
        <MetalDevice as Device>::BACKEND,
        run.trials,
        run.geometry.rows,
        run.geometry.simdgroups,
        run.geometry.router_sg,
    );

    let geom = run.geometry.label();
    let mut failed = false;
    // The sweep's variants, compiled once: each row is timed under the
    // default and then under each of these.
    let mut variants: Vec<(String, Kernels)> = Vec::new();
    if swept(&run.only) {
        for (per_simd_group, simdgroups) in SWEEP {
            let variant = Geometry {
                rows: per_simd_group,
                simdgroups,
                ..Geometry::DEFAULT
            };
            if variant == run.geometry {
                continue;
            }
            match geometry_kernels(&device, &source, variant) {
                Ok(kernels) => variants.push((variant.label(), kernels)),
                Err(e) => {
                    println!("geometry rows={per_simd_group} sg={simdgroups}: FAIL ({e})");
                    failed = true;
                }
            }
        }
    }
    let overlap = Arm {
        name: "overlap",
        dispatches: None,
        concurrent: true,
        barrier: false,
    };
    for row in rows {
        if !wanted(&run.only, family, row.shape.kind(), &row.label) {
            continue;
        }
        // A cut compiles the instance's default geometry, so a geometry
        // variant's run leaves the cuts out.
        if row.cut.is_some() && !run.geometry.is_default() {
            continue;
        }
        let cut_built = match row.cut {
            None => None,
            Some(cut) => match cut_kernels(&device, &source, cut) {
                Ok(kernels) => Some(kernels),
                Err(e) => {
                    println!("cut {}: FAIL ({e})", cut.name);
                    failed = true;
                    continue;
                }
            },
        };
        let launched = cut_built.as_ref().unwrap_or(&kernels);
        // The default's arm, then the sweep's: a cut compiles the default
        // geometry only.
        let mut arms: Vec<(&str, &Kernels)> = vec![(geom.as_str(), launched)];
        if row.cut.is_none() {
            arms.extend(variants.iter().map(|(label, k)| (label.as_str(), k)));
        }
        for (suffix, arm_kernels) in arms {
            let label = format!("{}{suffix}", row.label);
            if let Some(e) = &row.expect {
                println!(
                    "expect\tshape={label}\tgbps={:.0}-{:.0}\tbasis={}",
                    e.lo, e.hi, e.basis
                );
            }
            let outcome = time_arms(
                run,
                row.shape.kind(),
                &label,
                row.shape.bytes(),
                None,
                None,
                |arm| {
                    time_shape(
                        &device,
                        arm_kernels,
                        &row.shape,
                        arm,
                        run.trials,
                        run.check && row.cut.is_none(),
                    )
                },
            );
            failed |= outcome.failed;
            if outcome.stuck {
                return ExitCode::FAILURE;
            }
            if overlapped(&run.only) {
                let shape = &row.shape;
                match time_shape(&device, arm_kernels, shape, &overlap, run.trials, false) {
                    Ok(t) => print_row(
                        shape.kind(),
                        &label,
                        overlap.name,
                        run.trials,
                        &t,
                        shape.bytes(),
                        "",
                    ),
                    Err(e) => {
                        print_fail(shape.kind(), &label, overlap.name, &e);
                        failed = true;
                        if e.contains("still running") {
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
