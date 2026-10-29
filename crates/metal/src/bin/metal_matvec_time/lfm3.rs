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
//! form and three cuts of the A4 row loop: `[fma]` (the dot accumulated with
//! `fma`, the one float change), `[vecx]` (the activations read as `float4`s:
//! the same bits) and `[lite]` (the dot reduced to a conversion and an add a
//! word, the weight reads kept: what the loads alone cost).
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
    HeadPart, Run, Shape, WARMUPS, WINDOW_BYTES, WORKING_SET_BYTES, geometry_kernels, time_arms,
    time_shape,
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
type Edit = (&'static str, &'static str);

/// A phase cut of the instance's program text.
struct Cut {
    name: &'static str,
    edits: &'static [Edit],
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

/// The A4 block's activation reads: sixteen scalar floats a lane.
const LOAD_FIND: &str = concat!(
    "        lane.sumy = 0.f;\n",
    "        for (short l = 0; l < 4; ++l) {\n",
    "            float a0 = block[base + 4 * l + 0];\n",
    "            float a1 = block[base + 4 * l + 1];\n",
    "            float a2 = block[base + 4 * l + 2];\n",
    "            float a3 = block[base + 4 * l + 3];\n",
);

/// The same reads as four `float4`s: the lane's sixteen values start at a
/// multiple of 64 bytes of the input.
const LOAD_VEC: &str = concat!(
    "        lane.sumy = 0.f;\n",
    "        device const float4 * block4 = (device const float4 *)(block + base);\n",
    "        for (short l = 0; l < 4; ++l) {\n",
    "            const float4 v4 = block4[l];\n",
    "            float a0 = v4.x;\n",
    "            float a1 = v4.y;\n",
    "            float a2 = v4.z;\n",
    "            float a3 = v4.w;\n",
);

static FMA: Cut = Cut {
    name: "fma",
    edits: &[(ACC_FIND, ACC_FMA)],
};
static VECX: Cut = Cut {
    name: "vecx",
    edits: &[(LOAD_FIND, LOAD_VEC)],
};
static LITE: Cut = Cut {
    name: "lite",
    edits: &[(ACC_FIND, ACC_LITE)],
};

/// The band a row's window arm is predicted to fall in, and the measured
/// line it was derived from.
struct Expect {
    lo: f64,
    hi: f64,
    basis: &'static str,
}

/// One probe row.
struct Row {
    label: String,
    shape: Shape,
    expect: Option<Expect>,
    cut: Option<&'static Cut>,
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
    }
}

fn expect(lo: f64, hi: f64, basis: &'static str) -> Option<Expect> {
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
    for cut in [&FMA, &VECX, &LITE] {
        rows.push(Row {
            label: format!("lfm3 loop matvec a4 {ff}x{hidden} store [{}]", cut.name),
            shape: loop_shape(),
            expect: None,
            cut: Some(cut),
        });
    }
    rows
}

/// The selection names the LFM3 rows.
pub(super) fn selected(only: &[String]) -> bool {
    only.iter()
        .flat_map(|spec| spec.split(','))
        .any(|tok| tok.trim() == "lfm3")
}

/// A row is timed when the selection names no further filter, or one of its
/// further values is the row's kernel or a part of its label.
fn wanted(only: &[String], kind: &str, label: &str) -> bool {
    let filters: Vec<&str> = only
        .iter()
        .flat_map(|spec| spec.split(','))
        .map(str::trim)
        .filter(|tok| *tok != "lfm3")
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
    let instance = match open_key(key) {
        Ok(instance) => instance,
        Err(e) => {
            println!("metal_matvec_time: FAIL (LFM3's instance: {e})");
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
        "info\tinstrument=development-loop\tdevice={}\tlib={}\tbackend={}\tinstance={instance}\ttrials={}\twarmups={WARMUPS}\twindow_bytes={WINDOW_BYTES}\tworking_set_bytes={WORKING_SET_BYTES}\tgeom=rows{},sg{},router_sg{}",
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
    for row in rows() {
        if !wanted(&run.only, row.shape.kind(), &row.label) {
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
        let label = format!("{}{geom}", row.label);
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
                    launched,
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
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
