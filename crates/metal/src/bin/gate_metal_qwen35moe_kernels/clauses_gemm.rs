//! The GEMM clauses: the dense quantized dot of R activation rows against
//! the fixture's real tensors, every weight type, both operand arms and both
//! epilogues, held to the f64 oracle's band; the segmented dispatch
//! bit-equal to its segments' own single launches; the f16 operand's range
//! fault; the half-row entries bit-equal to the f32-row ones; and the launch
//! module's refusals.
//!
//! # The band
//!
//! The oracle is `crates/gguf`'s dequantization of the same blocks with an
//! f64 dot, as the mat-vec clauses' (`super::oracle`); the band is the same
//! model over the GEMM's own evaluation tree:
//!
//! - every weight value carries its owner's dequant arithmetic in f32: four
//!   roundings for Q4_K and Q5_K (`d·sc`, `·q`, `dmin·min`, the
//!   subtraction), two for Q6_K (`d·sc`, `·(q−32)`), two for A4 and A8
//!   (`s·q`, `+b`) and one for Q8_0 (`q·d`);
//! - the f32 operand's product rounds once; the f16 operand's two casts
//!   round once each into half (2^-11 relative) and their products are exact
//!   in the f32 accumulator (two eleven-bit mantissas);
//! - the accumulation is the tile's K/8 simdgroup instructions, each
//!   accumulating its eight-product dot in f32: a term passes its own
//!   instruction's dot (at worst eight roundings, the sequential order) and
//!   every later instruction's accumulate, so K/8 + 8 roundings;
//! - the staged round-trip through threadgroup memory is exact (the same
//!   values), and the residual epilogue adds one rounding of the sum,
//!   `U |y|`, as the mat-vec's residual kind. The path's operations are the
//!   absolute floor's count, as the oracle's.
//!
//! So the f32 arm's factor is `G(base + 1 + K/8 + 8)` and the f16 arm's is
//! `(1 + 2^-11)² (1 + G(base + K/8 + 8)) − 1`, both over the oracle's
//! magnitude sum.
//!
//! The tensor-op kernels (`gemm_tensor.metal`, [`GemmPath::TensorOps`]) sum
//! in the op's order, which no document fixes, at the op's defined
//! precision (relaxed precision off): f32 accumulation, the f16 operands'
//! products exact in it and the f32 operands' rounded once. Any order of a
//! sum of K terms passes a term through at most K − 1 additions (every
//! addition on its path to the root of the summation tree), so the
//! accumulation term is K, whatever the order, in place of K/8 + 8: the f32
//! arm's factor is `G(base + 1 + K)`, the f16 arm's
//! `(1 + 2^-11)² (1 + G(base + K)) − 1` — at K = 2048 a G of 2052 terms,
//! 1.2e-4 of the magnitude sum, at K = 512 3.1e-5. The casts, the subnormal
//! terms, the staged round-trip and the residual's rounding are the
//! simdgroup kernels' own. The f16 arm carries two absolute terms besides: a cast
//! into the subnormal range rounds on the subnormal grid (spacing 2^-24),
//! so its error is at most 2^-25 whatever the magnitude — the weights' side
//! sums to `2^-25 Σ_c |x_c|` and the activations' to `2^-25 Σ_c |w_c|`. The
//! design's power-of-two prescale producer keeps the activations out of the
//! subnormal range (their amax in [2^14, 2^15)); it is not here yet, so both
//! sides are in the band. A value past the half range is a fault, not a band
//! case.

use std::process::Command;

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::kernels::{
    GEMM_ACTS, GemmMode, GemmParams, GemmSegParams, GemmX, GemmXParams, Operand, Quant, Segment,
};
use bloomery_metal::library::GemmPath;
use gguf::quant::dequant_row;

use super::clauses_a4::{self, Sites};
use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, U};
use super::{Harness, SENTINEL_BITS, flag};

/// The half operand's cast rounds once, relatively.
pub(super) const HALF_ROUND: f64 = 1.0 / 2048.0;

/// The most absolute error of a cast into the f16 subnormal range: half the
/// grid's spacing.
pub(super) const HALF_SUBNORMAL: f64 = 1.0 / 33_554_432.0;

/// The words the output buffer holds before its window, so a store before
/// the window's first row lands on them and the guard sees it.
const Y_GUARD: usize = 8;

/// The words the output buffer holds past its window: an edge tile's store
/// can run up to a tile's width past the matrix's last column, so a store
/// past the last column lands here, inside the buffer, and the guard sees
/// it.
const Y_TAIL: usize = 72;

/// One GEMM launch of the gate: the shape, the windows' contents and the
/// arm.
pub struct GemmLaunch<'a> {
    pub quant: Quant,
    /// The weight rows N, the output's columns.
    pub rows: usize,
    /// The input columns K.
    pub cols: usize,
    /// The activation rows R, the output's rows.
    pub acts: usize,
    pub weights: &'a [u8],
    /// The activations: `acts` rows of `x_stride`, `cols` live in each.
    pub x: &'a [f32],
    pub x_stride: usize,
    /// The output before the launch: `acts` rows of `y_stride`, `rows` live
    /// in each.
    pub y0: &'a [f32],
    pub y_stride: usize,
    pub mode: GemmMode,
    pub operand: Operand,
}

fn padded<T: Copy>(pad: T, before: usize, body: &[T], after: usize) -> Vec<T> {
    let mut out = vec![pad; before];
    out.extend_from_slice(body);
    out.extend(std::iter::repeat_n(pad, after));
    out
}

/// Run `launch` and read the output window back, after checking that the
/// words around the window are as they were. The activation buffer carries
/// a tile's worth of sentinel rows past its window: the staging clamp never
/// reads them, and a read past the last row lands on them, not outside the
/// buffer.
pub fn launch_gemm(h: &Harness<'_>, launch: &GemmLaunch<'_>) -> Result<Vec<f32>, String> {
    launch_gemm_as(h, launch, Rows::F32)
}

/// The activation rows a gate launch hands its GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The f32 rows, cast as the kernel stages them.
    F32,
    /// The rows through [`bloomery_metal::kernels::Kernels::gemm_x`] in the
    /// same command buffer: half rows where the path takes them (the gate
    /// refuses a path that does but handed f32 rows back, and one that does
    /// not but cast).
    Cast,
}

/// The cast's half rows, NaN bits until the cast writes them, so a value it
/// left unwritten reads as a NaN output.
fn half_rows(h: &Harness<'_>, len: usize) -> Result<bloomery_metal::MetalBuffer<u16>, String> {
    h.device
        .upload(h.stream, &vec![0x7E01u16; len])
        .map_err(why)
}

/// The rows `gemm_x` handed back, against the path's own rule.
fn check_cast(h: &Harness<'_>, x: &GemmX<'_>) -> Result<(), DeviceError> {
    let half = matches!(x, GemmX::F16 { .. });
    if half == h.kernels.gemm_takes_half() {
        return Ok(());
    }
    Err(DeviceError::Range {
        what: "gate cast",
        detail: format!(
            "gemm_x handed back {} rows on a path whose f16 GEMMs {} half rows",
            if half { "half" } else { "f32" },
            if h.kernels.gemm_takes_half() {
                "read"
            } else {
                "do not read"
            }
        ),
    })
}

/// [`launch_gemm`] over the rows `rows` names.
pub fn launch_gemm_as(
    h: &Harness<'_>,
    launch: &GemmLaunch<'_>,
    rows: Rows,
) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let weights = padded(
        0xFFu8,
        super::WEIGHT_PAD,
        launch.weights,
        super::WEIGHT_TAIL,
    );
    let weights = device.upload(stream, &weights).map_err(why)?;
    let x_pad = GEMM_ACTS * launch.x_stride + launch.cols;
    let x_words = padded(sentinel, super::VALUE_PAD, launch.x, x_pad);
    let x = device.upload(stream, &x_words).map_err(why)?;
    let y_words = padded(sentinel, Y_GUARD, launch.y0, Y_TAIL);
    let mut y = device.upload(stream, &y_words).map_err(why)?;
    let x_len = (launch.acts - 1) * launch.x_stride + launch.cols;
    let y_len = (launch.acts - 1) * launch.y_stride + launch.rows;
    let mut half = half_rows(h, launch.acts * launch.cols)?;
    let xw = x
        .window(Elems(super::VALUE_PAD), Elems(x_len))
        .map_err(why)?;
    let weights = weights
        .window(Elems(super::WEIGHT_PAD), Elems(launch.weights.len()))
        .map_err(why)?;
    let yw = y.window_mut(Elems(Y_GUARD), Elems(y_len)).map_err(why)?;
    h.encode(|encoder| {
        let x = match rows {
            Rows::F32 => GemmX::F32 {
                x: xw,
                stride: launch.x_stride,
            },
            Rows::Cast => {
                let x = h.kernels.gemm_x(
                    encoder,
                    GemmXParams {
                        x: xw,
                        x_stride: launch.x_stride,
                        cols: launch.cols,
                        acts: launch.acts,
                        operand: launch.operand,
                        half: &mut half,
                        fault: device.fault_word(),
                        layer: LAYER,
                    },
                )?;
                check_cast(h, &x)?;
                x
            }
        };
        h.kernels.gemm(
            encoder,
            GemmParams {
                quant: launch.quant,
                rows: launch.rows,
                cols: launch.cols,
                acts: launch.acts,
                weights,
                x,
                y: yw,
                y_stride: launch.y_stride,
                mode: launch.mode,
                operand: launch.operand,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut back = vec![0f32; y_len + Y_GUARD + Y_TAIL];
    device
        .read(stream, y.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    check_guards(&back, y_len)?;
    Ok(back[Y_GUARD..Y_GUARD + y_len].to_vec())
}

/// One segmented GEMM launch of the gate: the segments' weights side by side
/// in one buffer, their rows at their own column offsets in one output.
pub struct GemmSegLaunch<'a> {
    pub quant: Quant,
    /// The input columns, the same for every segment.
    pub cols: usize,
    /// The activation rows R, the output's rows.
    pub acts: usize,
    /// The segments: the weight rows of one tensor and the bytes of its
    /// weights.
    pub segments: Vec<(usize, &'a [u8])>,
    pub x: &'a [f32],
    pub y0: &'a [f32],
    pub y_stride: usize,
    pub mode: GemmMode,
    pub operand: Operand,
}

/// Run `launch` and read the output window back, after checking the words
/// around it are as they were ([`launch_gemm`]'s own guards).
pub fn launch_gemm_seg(h: &Harness<'_>, launch: &GemmSegLaunch<'_>) -> Result<Vec<f32>, String> {
    launch_gemm_seg_as(h, launch, Rows::F32)
}

/// [`launch_gemm_seg`] over the rows `rows` names.
pub fn launch_gemm_seg_as(
    h: &Harness<'_>,
    launch: &GemmSegLaunch<'_>,
    rows: Rows,
) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let block = launch.quant.block_bytes();
    let (cols, acts) = (launch.cols, launch.acts);
    let mut weights: Vec<u8> = Vec::new();
    let mut list: Vec<Segment> = Vec::new();
    let mut out_offset = 0usize;
    for (rows, bytes) in &launch.segments {
        list.push(Segment {
            cols,
            rows: *rows,
            weight_blocks: weights.len() / block,
            out_offset,
        });
        weights.extend_from_slice(bytes);
        out_offset += rows;
    }
    let weight_bytes = weights.len();
    let x_len = acts * cols;
    let y_len = (acts - 1) * launch.y_stride + out_offset;
    let weights = padded(0xFFu8, super::WEIGHT_PAD, &weights, super::WEIGHT_TAIL);
    let weights = device.upload(stream, &weights).map_err(why)?;
    let x_pad = GEMM_ACTS * cols + cols;
    let x_words = padded(sentinel, super::VALUE_PAD, launch.x, x_pad);
    let x = device.upload(stream, &x_words).map_err(why)?;
    let y_words = padded(sentinel, Y_GUARD, launch.y0, Y_TAIL);
    let mut y = device.upload(stream, &y_words).map_err(why)?;
    let mut half = half_rows(h, acts * cols)?;
    let xw = x
        .window(Elems(super::VALUE_PAD), Elems(x_len))
        .map_err(why)?;
    let weights = weights
        .window(Elems(super::WEIGHT_PAD), Elems(weight_bytes))
        .map_err(why)?;
    let yw = y.window_mut(Elems(Y_GUARD), Elems(y_len)).map_err(why)?;
    h.encode(|encoder| {
        let x = match rows {
            Rows::F32 => GemmX::F32 {
                x: xw,
                stride: cols,
            },
            Rows::Cast => {
                let x = h.kernels.gemm_x(
                    encoder,
                    GemmXParams {
                        x: xw,
                        x_stride: cols,
                        cols,
                        acts,
                        operand: launch.operand,
                        half: &mut half,
                        fault: device.fault_word(),
                        layer: LAYER,
                    },
                )?;
                check_cast(h, &x)?;
                x
            }
        };
        h.kernels.gemm_seg(
            encoder,
            GemmSegParams {
                quant: launch.quant,
                segments: &list,
                weights,
                x,
                y: yw,
                y_stride: launch.y_stride,
                acts,
                mode: launch.mode,
                operand: launch.operand,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut back = vec![0f32; y_len + Y_GUARD + Y_TAIL];
    device
        .read(stream, y.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    check_guards(&back, y_len)?;
    Ok(back[Y_GUARD..Y_GUARD + y_len].to_vec())
}

/// The words either side of the output window are still the sentinel.
fn check_guards(back: &[f32], y_len: usize) -> Result<(), String> {
    let sides = back[..Y_GUARD].iter().chain(&back[Y_GUARD + y_len..]);
    for (i, v) in sides.enumerate() {
        if v.to_bits() != SENTINEL_BITS {
            return Err(format!(
                "a word {} the output window was written: {v:?}",
                if i < Y_GUARD { "before" } else { "past" }
            ));
        }
    }
    Ok(())
}

/// Σ|w_c| per output column, in f64 over the reader's dequantized row: the
/// f16 operand's subnormal-cast absolute term's activation side.
pub(super) fn weight_abs_sums(
    quant: Quant,
    weights: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f64>, String> {
    let row_bytes = cols / quant.values_per_block() * quant.block_bytes();
    let mut w = vec![0f32; cols];
    let mut out = Vec::with_capacity(rows);
    for row in weights.chunks_exact(row_bytes) {
        dequant_row(quant.ggml(), row, &mut w).map_err(|e| format!("dequantizing a row: {e}"))?;
        out.push(w.iter().map(|&v| f64::from(v).abs()).sum());
    }
    Ok(out)
}

/// The GEMM's band: the relative factor over the oracle's magnitude sum and
/// the operations of the longest path (the absolute floor's count), for the
/// program `path` runs. The derivation is this module's doc's.
pub(super) fn gemm_band(
    quant: Quant,
    cols: usize,
    operand: Operand,
    path: GemmPath,
) -> (f64, usize) {
    let base = match quant {
        Quant::Q4K | Quant::Q5K => 4,
        Quant::Q6K | Quant::A4 | Quant::A8 => 2,
        Quant::Q8_0 => 1,
        Quant::F32 => panic!("F32 has no dense GEMM: its rows run as the router's f32 GEMM"),
    };
    let acc = match path {
        GemmPath::Simdgroup => cols / 8 + 8,
        GemmPath::TensorOps => cols,
    };
    match operand {
        Operand::F32 => {
            let n = base + 1 + acc;
            (oracle::gamma_bound(n), n)
        }
        Operand::F16 => {
            let n = base + acc;
            (
                (1.0 + HALF_ROUND) * (1.0 + HALF_ROUND) * (1.0 + oracle::gamma_bound(n)) - 1.0,
                n,
            )
        }
    }
}

/// Run `launch`, hold every output element to the oracle's band, and return
/// the worst ratio of the error to the band. `w_abs` is
/// [`weight_abs_sums`]'s, over the same weights.
fn run_checked(h: &Harness<'_>, launch: &GemmLaunch<'_>, w_abs: &[f64]) -> Result<f64, String> {
    let got = launch_gemm(h, launch)?;
    common::expect_clean(h, "the gemm launch")?;
    check_rows(launch, &got, w_abs, None, h.kernels.gemm_path())
}

/// The band check of `got` against the oracle, row by row: `skip` names the
/// one row whose band is not checked (the range clause's, whose outputs are
/// NaN by contract).
fn check_rows(
    launch: &GemmLaunch<'_>,
    got: &[f32],
    w_abs: &[f64],
    skip: Option<usize>,
    path: GemmPath,
) -> Result<f64, String> {
    let (factor, ops) = gemm_band(launch.quant, launch.cols, launch.operand, path);
    let floor = ops as f64 * f64::from(f32::MIN_POSITIVE);
    let mut worst = 0f64;
    for r in 0..launch.acts {
        if skip == Some(r) {
            continue;
        }
        let row = &launch.x[r * launch.x_stride..r * launch.x_stride + launch.cols];
        let x_f64: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        let x_abs: f64 = x_f64.iter().map(|v| v.abs()).sum();
        let reference = oracle::reference(
            launch.quant,
            launch.weights,
            launch.rows,
            launch.cols,
            &x_f64,
        )?;
        for (n, &w_abs_n) in w_abs.iter().enumerate().take(launch.rows) {
            let at = r * launch.y_stride + n;
            let carried = match launch.mode {
                GemmMode::Store => 0.0,
                GemmMode::Residual => f64::from(launch.y0[at]),
            };
            let want = carried + reference.y[n];
            let resid = match launch.mode {
                GemmMode::Store => 0.0,
                GemmMode::Residual => U * want.abs(),
            };
            let width = factor * reference.magnitude[n] * (1.0 + U)
                + HALF_SUBNORMAL * (x_abs + w_abs_n)
                + resid
                + floor;
            let err = (f64::from(got[at]) - want).abs();
            if !got[at].is_finite() || err > width {
                return Err(format!(
                    "row {r} column {n}: got {:e}, want {want:e}, error {err:e} over the band {width:e}",
                    got[at]
                ));
            }
            worst = worst.max(err / width);
        }
    }
    Ok(worst)
}

/// One case of a store or resid clause: the tensor's bytes, its shape and
/// the strides (the tight ones unless named).
pub(super) struct Case<'a> {
    pub(super) label: String,
    pub(super) weights: &'a [u8],
    pub(super) quant: Quant,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) acts: usize,
    pub(super) x_stride: usize,
    pub(super) y_stride: usize,
    pub(super) seed: u64,
}

/// The cases one store or resid clause runs, both operand arms: the
/// fixture's real tensors at the activation row counts that exercise the
/// tile's edges — 9 and 33 leave activation rows past the last, 512 fills
/// the tiles — and a weight-row tail no threadgroup divides.
pub(super) fn run_cases(h: &Harness<'_>, mode: GemmMode, cases: &[Case<'_>]) -> Clause {
    let mut parts = Vec::new();
    for case in cases {
        let x = oracle::activations(case.acts * case.x_stride, case.seed, 1.0);
        let y_len = (case.acts - 1) * case.y_stride + case.rows;
        let y0 = oracle::activations(y_len, case.seed ^ 0x9E37, 1.0);
        let w_abs = weight_abs_sums(case.quant, case.weights, case.rows, case.cols)
            .map_err(|e| format!("{}: {e}", case.label))?;
        for operand in [Operand::F16, Operand::F32] {
            let worst = run_checked(
                h,
                &GemmLaunch {
                    quant: case.quant,
                    rows: case.rows,
                    cols: case.cols,
                    acts: case.acts,
                    weights: case.weights,
                    x: &x,
                    x_stride: case.x_stride,
                    y0: &y0,
                    y_stride: case.y_stride,
                    mode,
                    operand,
                },
                &w_abs,
            )
            .map_err(|e| format!("{} {:?}: {e}", case.label, operand))?;
            parts.push(format!("{} {:?} {worst:.3}", case.label, operand));
        }
    }
    common::noted(parts)
}

/// The Q4_K cases, the main fixture's tensors.
fn q4k_cases(fixture: &Fixture) -> Result<Vec<Case<'_>>, String> {
    let tail = fixture.rows_at(
        "blk.0.attn_qkv.weight",
        gguf::quant::GgmlType::Q4_K,
        2048,
        100,
        511,
    )?;
    Ok(vec![
        Case {
            label: "blk.0.attn_qkv.weight 8192x2048 r9".to_string(),
            weights: fixture.rows(
                "blk.0.attn_qkv.weight",
                gguf::quant::GgmlType::Q4_K,
                2048,
                8192,
            )?,
            quant: Quant::Q4K,
            rows: 8192,
            cols: 2048,
            acts: 9,
            x_stride: 2048,
            y_stride: 8192,
            seed: 0x6E31,
        },
        Case {
            label: "blk.0.ssm_out.weight 2048x4096 r33 strided".to_string(),
            weights: fixture.rows(
                "blk.0.ssm_out.weight",
                gguf::quant::GgmlType::Q4_K,
                4096,
                2048,
            )?,
            quant: Quant::Q4K,
            rows: 2048,
            cols: 4096,
            acts: 33,
            x_stride: 4096 + 16,
            y_stride: 2048 + 4,
            seed: 0x6E32,
        },
        Case {
            label: "blk.0.ffn_down_shexp.weight 2048x512 r512".to_string(),
            weights: fixture.rows(
                "blk.0.ffn_down_shexp.weight",
                gguf::quant::GgmlType::Q4_K,
                512,
                2048,
            )?,
            quant: Quant::Q4K,
            rows: 2048,
            cols: 512,
            acts: 512,
            x_stride: 512,
            y_stride: 2048,
            seed: 0x6E33,
        },
        Case {
            label: "blk.0.attn_qkv.weight tail 511x2048 r33".to_string(),
            weights: tail,
            quant: Quant::Q4K,
            rows: 511,
            cols: 2048,
            acts: 33,
            x_stride: 2048,
            y_stride: 512,
            seed: 0x6E34,
        },
    ])
}

/// The Q6_K cases, the main fixture's tensors, with the head's own tail.
fn q6k_cases(fixture: &Fixture) -> Result<Vec<Case<'_>>, String> {
    let tail_rows = 1153;
    let tail = fixture.rows_at(
        "output.weight",
        gguf::quant::GgmlType::Q6_K,
        2048,
        248_320 - tail_rows,
        tail_rows,
    )?;
    Ok(vec![
        Case {
            label: "blk.2.attn_qkv.weight 8192x2048 r9".to_string(),
            weights: fixture.rows(
                "blk.2.attn_qkv.weight",
                gguf::quant::GgmlType::Q6_K,
                2048,
                8192,
            )?,
            quant: Quant::Q6K,
            rows: 8192,
            cols: 2048,
            acts: 9,
            x_stride: 2048,
            y_stride: 8192,
            seed: 0x6E41,
        },
        Case {
            label: "blk.2.ffn_down_shexp.weight 2048x512 r33".to_string(),
            weights: fixture.rows(
                "blk.2.ffn_down_shexp.weight",
                gguf::quant::GgmlType::Q6_K,
                512,
                2048,
            )?,
            quant: Quant::Q6K,
            rows: 2048,
            cols: 512,
            acts: 33,
            x_stride: 512,
            y_stride: 2048,
            seed: 0x6E42,
        },
        Case {
            label: "blk.3.attn_v.weight 512x2048 r512".to_string(),
            weights: fixture.rows(
                "blk.3.attn_v.weight",
                gguf::quant::GgmlType::Q6_K,
                2048,
                512,
            )?,
            quant: Quant::Q6K,
            rows: 512,
            cols: 2048,
            acts: 512,
            x_stride: 2048,
            y_stride: 512,
            seed: 0x6E43,
        },
        Case {
            label: format!("output.weight tail {tail_rows}x2048 r9"),
            weights: tail,
            quant: Quant::Q6K,
            rows: tail_rows,
            cols: 2048,
            acts: 9,
            x_stride: 2048,
            y_stride: 1156,
            seed: 0x6E44,
        },
    ])
}

/// The seed of the A4 head-tail case's activations.
const HEAD_TAIL_SEED: u64 = 0x6E54;

/// The weight rows of the A4 gate-projection cases: a per-head attention gate
/// projects the hidden width to one value a head, 16 weight rows, a quarter of
/// the 64 a tile holds. Its output rows are 16 wide, the tight stride.
const GATE_ROWS: usize = 16;

/// The activation rows and seeds of the gate-projection cases: 33 and 9 leave
/// activation rows past the last of a tile.
const GATE_ACTS: [(usize, u64); 2] = [(33, 0x6E55), (9, 0x6E56)];

/// The A4 gate-projection cases: the first [`GATE_ROWS`] rows of the MLX
/// split's head, a tile of 16 live weight rows, at the tight output stride.
fn gate_cases<'a>(h: &Harness<'_>, mlx: &'a Fixture) -> Result<Vec<Case<'a>>, String> {
    let hidden = h.geom().hidden;
    let weights = mlx.rows_at(
        "output.weight",
        gguf::quant::GgmlType::MLX_A4,
        hidden,
        0,
        GATE_ROWS,
    )?;
    Ok(GATE_ACTS
        .iter()
        .map(|&(acts, seed)| Case {
            label: format!("output.weight head {GATE_ROWS}x{hidden} r{acts} stride {GATE_ROWS}"),
            weights,
            quant: Quant::A4,
            rows: GATE_ROWS,
            cols: hidden,
            acts,
            x_stride: hidden,
            y_stride: GATE_ROWS,
            seed,
        })
        .collect())
}

/// The A4 cases at a row's sites, the MLX-format fixture split's tensors: its
/// whole tensors, then the head's last rows, a count no weight tile divides.
fn a4_cases<'a>(h: &Harness<'_>, mlx: &'a Fixture, sites: &Sites) -> Result<Vec<Case<'a>>, String> {
    let tail_rows = 1153;
    let mut cases = Vec::new();
    for site in sites.gemm {
        let t = &site.tensor;
        cases.push(Case {
            label: format!("{} r{}", t.label(), site.acts),
            weights: t.bytes(mlx)?,
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            acts: site.acts,
            x_stride: t.cols,
            y_stride: t.rows.div_ceil(4) * 4,
            seed: site.seed,
        });
    }
    let hidden = h.geom().hidden;
    let output_rows = mlx.output_rows()?;
    cases.push(Case {
        label: format!("output.weight tail {tail_rows}x{hidden} r33"),
        weights: mlx.rows_at(
            "output.weight",
            gguf::quant::GgmlType::MLX_A4,
            hidden,
            output_rows - tail_rows,
            tail_rows,
        )?,
        quant: Quant::A4,
        rows: tail_rows,
        cols: hidden,
        acts: 33,
        x_stride: hidden,
        y_stride: tail_rows.div_ceil(4) * 4,
        seed: HEAD_TAIL_SEED,
    });
    cases.extend(gate_cases(h, mlx)?);
    Ok(cases)
}

/// The store dot at every weight type's cases, both operand arms: every
/// element within the f64 oracle's band.
pub fn gemm_q4k_store(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    run_cases(h, GemmMode::Store, &q4k_cases(fixture)?)
}

/// The residual epilogue at the same cases: `y = y + W . x`, in place.
pub fn gemm_q4k_resid(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    run_cases(h, GemmMode::Residual, &q4k_cases(fixture)?)
}

pub fn gemm_q6k_store(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    run_cases(h, GemmMode::Store, &q6k_cases(fixture)?)
}

pub fn gemm_q6k_resid(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    run_cases(h, GemmMode::Residual, &q6k_cases(fixture)?)
}

/// What a child run of the gate under Metal's shader validation said on
/// stderr.
struct Validated {
    /// Whether the validation layer announced itself.
    started: bool,
    /// How many accesses it reported.
    reports: usize,
    /// The first report's lines, for a clause that fails on it.
    first: String,
    /// The child's own verdict and last words.
    exit: Result<(), String>,
}

/// The mutant a child of the control run carries: the one that stages the
/// weight rows past a tile's live rows unclamped.
const CLAMP_MUTANT: &str = "gemm_weight_edge";

/// The gate's own binary, run as `--child gemm_a4_store` at this row's instance
/// with Metal's shader validation on (its reports go to stderr), beside the
/// parent's fixture flags and a `--mutant` when one is named.
fn gate_under_validation(
    h: &Harness<'_>,
    args: &[String],
    mutant: Option<&str>,
) -> Result<Validated, String> {
    let exe = std::env::current_exe().map_err(|e| format!("the gate's own path: {e}"))?;
    let mut child = Command::new(exe);
    child.args([
        "--child",
        "gemm_a4_store",
        "--instance",
        h.library.instance().name,
    ]);
    for name in ["--fixture", "--mlx-fixture"] {
        if let Some(path) = flag(args, name) {
            child.args([name, path]);
        }
    }
    if let Some(name) = mutant {
        child.args(["--mutant", name]);
    }
    child
        .env("MTL_SHADER_VALIDATION", "1")
        .env("MTL_SHADER_VALIDATION_GLOBAL_MEMORY", "1")
        .env("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1")
        .env("MTL_SHADER_VALIDATION_ENABLE_VERBOSE_LOGGING", "1");
    let out = child
        .output()
        .map_err(|e| format!("the child gate did not start: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let first = stderr
        .split("=====")
        .find(|block| block.contains("Invalid Metal usage"))
        .map(|block| {
            block
                .lines()
                .filter(|l| !l.is_empty())
                .take(2)
                .collect::<Vec<_>>()
                .join(" / ")
        })
        .unwrap_or_default();
    Ok(Validated {
        started: stderr.contains("Metal GPU Validation Enabled"),
        reports: stderr.matches("Invalid Metal usage").count(),
        first,
        exit: if out.status.success() {
            Ok(())
        } else {
            Err(format!(
                "the child exited {}: {}",
                out.status,
                stdout.lines().last().unwrap_or("no output")
            ))
        },
    })
}

/// The A4 store dot at the row's sites, the head's tail and the gate
/// projection's 16 rows, every case within the band of the f64 oracle in both
/// operands; and the gate projection's launches read no weight row past their
/// buffer. The weight rows past a tile's live rows stage a clamped copy of the
/// last live row and are never stored, so no output shows an unclamped read
/// and only Metal's shader validation reports it; it must be on when the
/// process starts, so the 16-row launches run in a child gate. The control:
/// the same child with the `gemm_weight_edge` mutant, which reads those rows,
/// must be reported, or the validation checked nothing and the clause is red.
///
/// Mutant: `gemm_weight_edge`, the clamp of the weight row removed.
pub fn gemm_a4_store(h: &Harness<'_>, mlx: &Fixture, sites: &Sites) -> Clause {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--child") {
        return run_cases(h, GemmMode::Store, &gate_cases(h, mlx)?);
    }
    let Verdict::Noted(values) = run_cases(h, GemmMode::Store, &a4_cases(h, mlx, sites)?)? else {
        return Err("the A4 store cases ended in no note".to_string());
    };
    let control = gate_under_validation(h, &args, Some(CLAMP_MUTANT))?;
    if !control.started || control.reports == 0 {
        return Err(format!(
            "shader validation {} for the control, a kernel that reads past its weights: nothing \
             was checked",
            if control.started {
                "reported nothing"
            } else {
                "did not start"
            }
        ));
    }
    let own = gate_under_validation(h, &args, flag(&args, "--mutant"))?;
    if !own.started {
        return Err("shader validation did not start: nothing was checked".to_string());
    }
    if own.reports > 0 {
        return Err(format!(
            "{} accesses outside a buffer reported in the {GATE_ROWS}-row launches, the first: {}",
            own.reports, own.first
        ));
    }
    own.exit?;
    Ok(Verdict::Noted(format!(
        "{values}; no report in the {GATE_ROWS}-row launches under shader validation, {} for the \
         control",
        control.reports
    )))
}

pub fn gemm_a4_resid(h: &Harness<'_>, mlx: &Fixture, sites: &Sites) -> Clause {
    run_cases(h, GemmMode::Residual, &a4_cases(h, mlx, sites)?)
}

/// Every dense arm of one weight type on `h`'s kernels (the tensor-op
/// kernels, for the `tensor_gemm_*` clauses): both epilogues in both
/// operands, every case within the band of `h`'s GEMM path.
pub(super) fn dense_arms(h: &Harness<'_>, cases: &[Case<'_>]) -> Clause {
    let mut notes = Vec::new();
    for mode in [GemmMode::Store, GemmMode::Residual] {
        let Verdict::Noted(note) = run_cases(h, mode, cases)? else {
            return Err(format!("{mode:?}: the cases read no band"));
        };
        notes.push(format!("{mode:?} {note}"));
    }
    Ok(Verdict::Noted(format!(
        "{} kernels: {}",
        h.kernels.gemm_path().name(),
        notes.join("; ")
    )))
}

pub fn tensor_gemm_q4k(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    dense_arms(h, &q4k_cases(fixture)?)
}

pub fn tensor_gemm_q6k(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    dense_arms(h, &q6k_cases(fixture)?)
}

pub fn tensor_gemm_a4(h: &Harness<'_>, mlx: &Fixture, sites: &Sites) -> Clause {
    dense_arms(h, &a4_cases(h, mlx, sites)?)
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone: the kernel does not change the order of a row's sum, and a
/// segment's tiles are the ones a launch of it alone would run. One group
/// per weight type, a tail segment in two of them, both operand arms and
/// both epilogues; the A4 group is the row's own ([`Sites`]).
pub fn gemm_seg(h: &Harness<'_>, fixture: &Fixture, mlx: &Fixture, sites: &Sites) -> Clause {
    let cols = SEG_COLS;
    let q4k_tail = fixture.rows_at(
        "blk.0.attn_qkv.weight",
        gguf::quant::GgmlType::Q4_K,
        cols,
        100,
        511,
    )?;
    let groups = [
        SegGroup {
            quant: Quant::Q4K,
            segments: vec![
                (
                    8192,
                    fixture.rows(
                        "blk.0.attn_qkv.weight",
                        gguf::quant::GgmlType::Q4_K,
                        cols,
                        8192,
                    )?,
                ),
                (
                    4096,
                    fixture.rows(
                        "blk.0.attn_gate.weight",
                        gguf::quant::GgmlType::Q4_K,
                        cols,
                        4096,
                    )?,
                ),
                (511, q4k_tail),
            ],
            mode: GemmMode::Store,
        },
        SegGroup {
            quant: Quant::Q6K,
            segments: vec![(
                8192,
                fixture.rows(
                    "blk.2.attn_qkv.weight",
                    gguf::quant::GgmlType::Q6_K,
                    cols,
                    8192,
                )?,
            )],
            mode: GemmMode::Residual,
        },
        a4_seg_group(mlx, sites)?,
    ];
    Ok(Verdict::Noted(format!(
        "every segment the same bits as its tensor alone: {}",
        seg_groups(h, &groups)?.join(", ")
    )))
}

/// The A4 group of a segmented dispatch at a row's sites: its whole tensors'
/// segments, then the tail segment cut from the middle of a tensor.
fn a4_seg_group<'a>(mlx: &'a Fixture, sites: &Sites) -> Result<SegGroup<'a>, String> {
    let ty = gguf::quant::GgmlType::MLX_A4;
    let mut segments = Vec::new();
    for (name, rows) in sites.gemm_seg.whole {
        segments.push((*rows, mlx.rows(name, ty, SEG_COLS, *rows)?));
    }
    let (name, at, rows) = sites.gemm_seg.tail;
    segments.push((rows, mlx.rows_at(name, ty, SEG_COLS, at, rows)?));
    Ok(SegGroup {
        quant: Quant::A4,
        segments,
        mode: GemmMode::Store,
    })
}

/// [`gemm_seg`] for a row that runs only its A4 sites: the A4 group, the same
/// bits as each segment's tensor alone.
pub fn gemm_seg_a4(h: &Harness<'_>, mlx: &Fixture, sites: &Sites) -> Clause {
    let groups = [a4_seg_group(mlx, sites)?];
    Ok(Verdict::Noted(format!(
        "every segment the same bits as its tensor alone: {}",
        seg_groups(h, &groups)?.join(", ")
    )))
}

/// The columns of every segment [`seg_groups`] runs.
pub(super) const SEG_COLS: usize = 2048;

/// One group of a segmented dispatch: one weight type's segments, each
/// `SEG_COLS` columns, and the epilogue.
pub(super) struct SegGroup<'a> {
    pub(super) quant: Quant,
    pub(super) segments: Vec<(usize, &'a [u8])>,
    pub(super) mode: GemmMode,
}

/// Each group's segmented dispatch, both operand arms, every segment
/// against its own single launch, bit for bit; every column of the output
/// written. One note per group and arm.
pub(super) fn seg_groups(h: &Harness<'_>, groups: &[SegGroup<'_>]) -> Result<Vec<String>, String> {
    let acts = 33;
    let cols = SEG_COLS;
    let mut parts = Vec::new();
    for (g, group) in groups.iter().enumerate() {
        let rows: usize = group.segments.iter().map(|(r, _)| r).sum();
        // The edge store's stride rule: a whole number of float4s.
        let stride = rows.div_ceil(4) * 4;
        let x = oracle::activations(acts * cols, 0x6E61 + g as u64, 1.0);
        let y_len = (acts - 1) * stride + rows;
        let y0 = oracle::activations(y_len, 0x6E71 + g as u64, 1.0);
        for operand in [Operand::F16, Operand::F32] {
            let got = launch_gemm_seg(
                h,
                &GemmSegLaunch {
                    quant: group.quant,
                    cols,
                    acts,
                    segments: group.segments.clone(),
                    x: &x,
                    y0: &y0,
                    y_stride: stride,
                    mode: group.mode,
                    operand,
                },
            )
            .map_err(|e| format!("{:?} {:?}: {e}", group.quant, operand))?;
            common::expect_clean(h, "the segmented launch")?;
            if let Some(at) = got.iter().position(|v| v.to_bits() == SENTINEL_BITS) {
                return Err(format!(
                    "{:?} {:?}: column {at} of the output was not written",
                    group.quant, operand
                ));
            }
            // Each segment against its own single launch, bit for bit. The
            // single launch's output rows hold the segment's own columns,
            // and its residual input is the same values the segment's
            // columns of the segmented launch's input hold.
            let mut at = 0usize;
            for (seg_rows, bytes) in &group.segments {
                let alone_stride = seg_rows.div_ceil(4) * 4;
                let mut alone_y0 = Vec::with_capacity((acts - 1) * alone_stride + seg_rows);
                for r in 0..acts {
                    alone_y0.extend_from_slice(&y0[r * stride + at..r * stride + at + seg_rows]);
                    if r + 1 < acts {
                        alone_y0.extend(std::iter::repeat_n(0f32, alone_stride - seg_rows));
                    }
                }
                let alone = launch_gemm(
                    h,
                    &GemmLaunch {
                        quant: group.quant,
                        rows: *seg_rows,
                        cols,
                        acts,
                        weights: bytes,
                        x: &x,
                        x_stride: cols,
                        y0: &alone_y0,
                        y_stride: alone_stride,
                        mode: group.mode,
                        operand,
                    },
                )
                .map_err(|e| format!("a segment of {:?} alone: {e}", group.quant))?;
                for r in 0..acts {
                    for n in 0..*seg_rows {
                        let seg_at = r * stride + at + n;
                        let own_at = r * alone_stride + n;
                        if got[seg_at].to_bits() != alone[own_at].to_bits() {
                            return Err(format!(
                                "{:?} {:?} segment at column {at} row {r} column {n}: \
                                 the segment read {:?}, the tensor alone read {:?}",
                                group.quant, operand, got[seg_at], alone[own_at]
                            ));
                        }
                    }
                }
                at += seg_rows;
            }
            parts.push(format!(
                "{:?} {} segments {:?}",
                group.quant,
                group.segments.len(),
                operand
            ));
        }
    }
    Ok(parts)
}

/// The f16 operand's range check: an activation that is not finite or past
/// the half range raises the activation quantizer's site and its row's
/// outputs are NaN — never a silent inf — and every other row stays in band.
/// The boundary itself (the largest finite half) is taken, not faulted. The
/// f32 arm takes the same activations as they are, like the plain mat-vec:
/// no fault, and the non-finite row's outputs are the non-finite dot's.
pub fn gemm_range(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let quant = Quant::Q4K;
    let (rows, cols, acts) = (2048usize, 512, 33);
    let weights = fixture.rows(
        "blk.0.ffn_down_shexp.weight",
        gguf::quant::GgmlType::Q4_K,
        cols,
        rows,
    )?;
    let w_abs = weight_abs_sums(quant, weights, rows, cols)?;
    for (what, value, bad_row) in [
        ("an activation past the half range", 65_536.0f32, 7),
        ("a NaN activation", f32::NAN, 0),
    ] {
        let mut x = oracle::activations(acts * cols, 0x6E81, 1.0);
        x[bad_row * cols + 11] = value;
        let y0 = oracle::activations(acts * rows, 0x6E82, 1.0);
        let launch = GemmLaunch {
            quant,
            rows,
            cols,
            acts,
            weights,
            x: &x,
            x_stride: cols,
            y0: &y0,
            y_stride: rows,
            mode: GemmMode::Store,
            operand: Operand::F16,
        };
        h.clear_fault()?;
        let got = launch_gemm(h, &launch)?;
        common::expect_fault(h, what, FaultSite::QuantColumn)?;
        for n in 0..rows {
            if !got[bad_row * rows + n].is_nan() {
                return Err(format!(
                    "{what}: row {bad_row} column {n} is {:?}, not NaN",
                    got[bad_row * rows + n]
                ));
            }
        }
        // The other rows are finite and in band.
        check_rows(&launch, &got, &w_abs, Some(bad_row), h.kernels.gemm_path())
            .map_err(|e| format!("{what}, another row: {e}"))?;
    }
    // The boundary: the largest finite half is taken, not faulted.
    h.clear_fault()?;
    let mut x = oracle::activations(acts * cols, 0x6E83, 1.0);
    x[3 * cols + 100] = 65_504.0;
    let y0 = oracle::activations(acts * rows, 0x6E84, 1.0);
    let launch = GemmLaunch {
        quant,
        rows,
        cols,
        acts,
        weights,
        x: &x,
        x_stride: cols,
        y0: &y0,
        y_stride: rows,
        mode: GemmMode::Store,
        operand: Operand::F16,
    };
    let worst = run_checked(h, &launch, &w_abs)?;
    // The f32 arm: the same non-finite activation, no fault, its row the
    // non-finite dot's.
    h.clear_fault()?;
    let mut x = oracle::activations(acts * cols, 0x6E85, 1.0);
    x[7 * cols + 11] = 65_536.0;
    let y0 = oracle::activations(acts * rows, 0x6E86, 1.0);
    let launch = GemmLaunch {
        quant,
        rows,
        cols,
        acts,
        weights,
        x: &x,
        x_stride: cols,
        y0: &y0,
        y_stride: rows,
        mode: GemmMode::Store,
        operand: Operand::F32,
    };
    let got = launch_gemm(h, &launch)?;
    common::expect_clean(h, "a finite activation under the f32 arm")?;
    for n in 0..rows {
        if !got[7 * rows + n].is_finite() {
            return Err(format!(
                "the f32 arm's row 7 column {n} is {:?}, not the finite dot's",
                got[7 * rows + n]
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "the f16 arm's range fault on the half range and on NaN, \
         the boundary taken, the f32 arm faultless; worst {worst:.3}"
    )))
}

/// The half-row GEMMs: on a path whose f16 GEMMs read half rows, the
/// activations cast once ([`Rows::Cast`]) give the f32-row entries' outputs
/// bit for bit — the dense A4 cases at both epilogues, the residual's at f32
/// rows strided wider than their columns, and the segmented group — and the
/// cast's range check is the staging's: an activation past the half range,
/// or NaN, raises the activation quantizer's site and every output is the
/// f32-row entry's bits, the faulted row's NaN among them. A path whose f16
/// GEMMs do not read half rows gets its f32 rows back from the cast.
pub fn gemm_half_rows(h: &Harness<'_>, mlx: &Fixture, sites: &Sites) -> Clause {
    half_rows_over(h, &a4_cases(h, mlx, sites)?, a4_seg_group(mlx, sites)?)
}

/// [`gemm_half_rows`]'s body over given `cases` and one segmented `group`:
/// the A8 clause runs it at the split's tensors quantized at 8 bits
/// (`clauses_a8`).
pub(super) fn half_rows_over(h: &Harness<'_>, cases: &[Case<'_>], group: SegGroup<'_>) -> Clause {
    let same =
        |what: &str, want: &[f32], got: &[f32]| super::clauses_rows::same_bits(what, got, want);
    h.clear_fault()?;
    for case in cases {
        for (mode, x_stride) in [
            (GemmMode::Store, case.x_stride),
            (GemmMode::Residual, case.x_stride + 20),
        ] {
            let x = oracle::activations((case.acts - 1) * x_stride + case.cols, case.seed, 1.0);
            let y_len = (case.acts - 1) * case.y_stride + case.rows;
            let y0 = oracle::activations(y_len, case.seed ^ 0x5A5A, 1.0);
            let launch = GemmLaunch {
                quant: case.quant,
                rows: case.rows,
                cols: case.cols,
                acts: case.acts,
                weights: case.weights,
                x: &x,
                x_stride,
                y0: &y0,
                y_stride: case.y_stride,
                mode,
                operand: Operand::F16,
            };
            let want = launch_gemm_as(h, &launch, Rows::F32)?;
            let got = launch_gemm_as(h, &launch, Rows::Cast)?;
            common::expect_clean(h, &case.label)?;
            same(
                &format!("{} {mode:?} x stride {x_stride}", case.label),
                &want,
                &got,
            )?;
        }
    }
    let (acts, cols) = (33, SEG_COLS);
    let seg_rows: usize = group.segments.iter().map(|(r, _)| r).sum();
    let y_stride = seg_rows.div_ceil(4) * 4;
    let x = oracle::activations(acts * cols, 0x6E91, 1.0);
    let y0 = oracle::activations((acts - 1) * y_stride + seg_rows, 0x6E92, 1.0);
    for mode in [GemmMode::Store, GemmMode::Residual] {
        let launch = GemmSegLaunch {
            quant: group.quant,
            cols,
            acts,
            segments: group.segments.clone(),
            x: &x,
            y0: &y0,
            y_stride,
            mode,
            operand: Operand::F16,
        };
        let want = launch_gemm_seg_as(h, &launch, Rows::F32)?;
        let got = launch_gemm_seg_as(h, &launch, Rows::Cast)?;
        common::expect_clean(h, "the segmented launch")?;
        same(
            &format!("the segmented {:?} group {mode:?}", group.quant),
            &want,
            &got,
        )?;
    }
    let case = cases.first().ok_or("no case for the half-row launches")?;
    for (what, value, bad_row) in [
        ("an activation past the half range", 65_536.0f32, 7usize),
        ("a NaN activation", f32::NAN, 0),
    ] {
        let mut x = oracle::activations(acts * case.cols, 0x6E93, 1.0);
        x[bad_row * case.cols + 11] = value;
        let y0 = oracle::activations((acts - 1) * case.y_stride + case.rows, 0x6E94, 1.0);
        let launch = GemmLaunch {
            quant: case.quant,
            rows: case.rows,
            cols: case.cols,
            acts,
            weights: case.weights,
            x: &x,
            x_stride: case.cols,
            y0: &y0,
            y_stride: case.y_stride,
            mode: GemmMode::Store,
            operand: Operand::F16,
        };
        h.clear_fault()?;
        let want = launch_gemm_as(h, &launch, Rows::F32)?;
        common::expect_fault(h, what, FaultSite::QuantColumn)?;
        h.clear_fault()?;
        let got = launch_gemm_as(h, &launch, Rows::Cast)?;
        common::expect_fault(
            h,
            &format!("{what} through the cast"),
            FaultSite::QuantColumn,
        )?;
        same(what, &want, &got)?;
        if !got[bad_row * case.y_stride].is_nan() {
            return Err(format!(
                "{what}: row {bad_row}'s first output is {:?}, not NaN",
                got[bad_row * case.y_stride]
            ));
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "{} dense {:?} cases at both epilogues and the segmented group, the cast rows' bits the \
         f32 rows' ({}); the range fault and NaN rows through the cast",
        cases.len(),
        case.quant,
        if h.kernels.gemm_takes_half() {
            "half rows"
        } else {
            "f32 rows handed back"
        }
    )))
}

/// The buffers of the refusal table.
struct Refusals {
    cols: usize,
    bytes: usize,
    weights: bloomery_metal::MetalBuffer<u8>,
    x: bloomery_metal::MetalBuffer<f32>,
    y: bloomery_metal::MetalBuffer<f32>,
}

/// The launch of one case: the shape it is told and the lengths and offsets
/// of its windows, one of them bad.
#[derive(Clone, Copy)]
struct Args {
    rows: usize,
    cols: usize,
    acts: usize,
    weights: usize,
    weights_at: usize,
    x: usize,
    x_stride: usize,
    y: usize,
    y_at: usize,
    y_stride: usize,
}

impl Args {
    const fn whole(rows: usize, cols: usize, acts: usize, bytes: usize) -> Self {
        Self {
            rows,
            cols,
            acts,
            weights: bytes,
            weights_at: 0,
            x: (acts - 1) * cols + cols,
            x_stride: cols,
            y: (acts - 1) * rows + rows,
            y_at: 0,
            y_stride: rows,
        }
    }
}

fn refuse(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &Args,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), DeviceError> {
    h.kernels.gemm(
        encoder,
        GemmParams {
            quant: Quant::Q4K,
            rows: a.rows,
            cols: a.cols,
            acts: a.acts,
            weights: b.weights.window(Elems(a.weights_at), Elems(a.weights))?,
            x: GemmX::F32 {
                x: b.x.window(Elems(0), Elems(a.x))?,
                stride: a.x_stride,
            },
            y: b.y.window_mut(Elems(a.y_at), Elems(a.y))?,
            y_stride: a.y_stride,
            mode: GemmMode::Store,
            operand: Operand::F16,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Args) + 'a,
    h: &'a Harness<'_>,
    whole: Args,
) -> BadArg<'a, Refusals> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut a = whole;
            edit(&mut a);
            refuse(h, b, &a, encoder)
        }),
    )
}

/// The segment list of one seg case, one property of it bad.
#[derive(Clone, Copy)]
struct SegArgs {
    segments: [Option<Segment>; 5],
    weights_len: usize,
    y_len: usize,
}

impl SegArgs {
    /// The table's arguments: the y window spans the whole buffer, so a
    /// case's own check refuses before the window check can. Every segment
    /// is eight rows of the one width.
    const fn whole(weights_len: usize, y_len: usize) -> Self {
        Self {
            segments: [
                Some(Segment {
                    cols: 2048,
                    rows: 8,
                    weight_blocks: 0,
                    out_offset: 0,
                }),
                None,
                None,
                None,
                None,
            ],
            weights_len,
            y_len,
        }
    }

    fn list(&self) -> Vec<Segment> {
        self.segments.iter().filter_map(|s| *s).collect()
    }
}

fn refuse_seg(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &SegArgs,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), DeviceError> {
    let segments = a.list();
    h.kernels.gemm_seg(
        encoder,
        GemmSegParams {
            quant: Quant::Q4K,
            segments: &segments,
            weights: b.weights.window(Elems(0), Elems(a.weights_len))?,
            x: GemmX::F32 {
                x: b.x.window(Elems(0), Elems(b.cols * 33))?,
                stride: b.cols,
            },
            acts: 33,
            y: b.y.window_mut(Elems(0), Elems(a.y_len))?,
            y_stride: a
                .segments
                .iter()
                .filter_map(|s| *s)
                .map(|s| s.out_offset + s.rows)
                .max()
                .unwrap_or(0),
            mode: GemmMode::Store,
            operand: Operand::F16,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

fn bad_seg<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut SegArgs) + 'a,
    h: &'a Harness<'_>,
    weights_len: usize,
) -> BadArg<'a, Refusals> {
    let whole = SegArgs::whole(weights_len, 264);
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut a = whole;
            edit(&mut a);
            refuse_seg(h, b, &a, encoder)
        }),
    )
}

/// The launches the host module must refuse: the shape it is told and the
/// length or offset of each window it is given, each refused naming the
/// argument; and the segment lists it cannot run. The well-formed launch is
/// accepted and writes.
pub fn gemm_refusals(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let (rows, cols, acts) = (8usize, 512, 9);
    let bytes = rows * cols / 256 * Quant::Q4K.block_bytes();
    // The segment lists' weights: exactly the one well-formed segment's
    // span (eight rows of the one width), so a block offset past it is
    // refused by the window check.
    let seg_bytes = 8 * 2048 / 256 * Quant::Q4K.block_bytes();
    let mut bufs = Refusals {
        cols: 2048,
        bytes,
        weights: device
            .upload(stream, &vec![0u8; bytes.max(seg_bytes)])
            .map_err(why)?,
        x: device.upload(stream, &vec![0f32; 2048 * 64]).map_err(why)?,
        y: device.upload(stream, &vec![1f32; 4096]).map_err(why)?,
    };
    let whole = Args::whole(rows, cols, acts, bytes);
    let cases = vec![
        bad(
            "a short weights window",
            "weights",
            |a| a.weights -= 1,
            h,
            whole,
        ),
        bad(
            "a long weights window",
            "weights",
            |a| a.weights += 1,
            h,
            whole,
        ),
        bad(
            "weights at an odd byte offset",
            "16-bit words",
            |a| a.weights_at = 1,
            h,
            whole,
        ),
        bad("a short x window", "x window", |a| a.x -= 1, h, whole),
        bad("a short y window", "y window", |a| a.y -= 1, h, whole),
        bad(
            "an x stride narrower than its row",
            "x stride",
            |a| a.x_stride = a.cols - 1,
            h,
            whole,
        ),
        bad(
            "a y stride narrower than its row",
            "y stride",
            |a| {
                a.y_stride = a.rows - 1;
            },
            h,
            whole,
        ),
        bad(
            "a y stride the edge store cannot write",
            "multiple of the four elements",
            |a| a.y_stride = a.rows + 2,
            h,
            whole,
        ),
        bad(
            "a y window the edge store cannot write",
            "float4",
            |a| a.y_at = 1,
            h,
            whole,
        ),
        bad(
            "no weight rows",
            "empty",
            |a| {
                a.rows = 0;
                a.weights = 0;
                a.y = 0;
                a.y_stride = 0;
            },
            h,
            whole,
        ),
        bad(
            "no columns",
            "empty",
            |a| {
                a.cols = 0;
                a.weights = 0;
                a.x = 0;
                a.x_stride = 0;
            },
            h,
            whole,
        ),
        bad(
            "no activation rows",
            "empty",
            |a| {
                a.acts = 0;
                a.x = 0;
                a.y = 0;
            },
            h,
            whole,
        ),
        bad(
            "columns that are not whole blocks",
            "whole blocks",
            |a| {
                a.cols = 500;
                a.x = 8 * 500 + 500;
                a.x_stride = 500;
            },
            h,
            whole,
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Refusals| common::still(h, &b.y, |v| *v == 1.0);
    let count = common::refuse_launches(h, "gemm", &mut bufs, cases, unchanged)?;

    // The segment lists: a list the kernel cannot run.
    let seg_cases = vec![
        bad_seg(
            "segments of different widths",
            "columns",
            |a| {
                a.segments[1] = Some(Segment {
                    cols: 2048 + 256,
                    rows: 8,
                    weight_blocks: 64,
                    out_offset: 8,
                });
            },
            h,
            seg_bytes,
        ),
        bad_seg(
            "a segment of no rows",
            "no rows",
            |a| {
                a.segments[0] = Some(Segment {
                    cols: 2048,
                    rows: 0,
                    weight_blocks: 0,
                    out_offset: 0,
                });
            },
            h,
            seg_bytes,
        ),
        bad_seg(
            "more segments than a dispatch runs",
            "at most",
            |a| {
                for (i, slot) in a.segments.iter_mut().enumerate() {
                    *slot = Some(Segment {
                        cols: 2048,
                        rows: 8,
                        weight_blocks: i * 64,
                        out_offset: i * 8,
                    });
                }
            },
            h,
            seg_bytes,
        ),
        bad_seg(
            "segments whose outputs overlap",
            "overlap",
            |a| {
                a.segments[1] = Some(Segment {
                    cols: 2048,
                    rows: 8,
                    weight_blocks: 64,
                    out_offset: 4,
                });
            },
            h,
            seg_bytes,
        ),
        bad_seg(
            "a segment past the weights",
            "weights",
            |a| {
                a.segments[0] = Some(Segment {
                    cols: 2048,
                    rows: 8,
                    weight_blocks: 200,
                    out_offset: 0,
                });
            },
            h,
            seg_bytes,
        ),
    ];
    let seg_count = common::refuse_launches(h, "gemm", &mut bufs, seg_cases, unchanged)?;
    let good: Edit<'_, Refusals> = Box::new(|b, encoder| {
        let a = Args::whole(8, 512, 9, b.bytes);
        refuse(h, b, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.y, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals and {seg_count} segment-list refusals, each by name, nothing written"
    )))
}

/// The clause table of the whole row: one clause per entry, after the
/// mat-vec families.
pub fn table<'a>(
    h: &'a Harness<'a>,
    fixture: &'a Result<Fixture, String>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    let sites = &clauses_a4::QWEN35_SITES;
    let with = |f: fn(&Harness<'_>, &Fixture) -> Clause| {
        Box::new(move || match fixture {
            Ok(fix) => f(h, fix),
            Err(e) => Err(e.clone()),
        }) as super::ClauseRun<'a>
    };
    vec![
        ("gemm_q4k_store", with(gemm_q4k_store)),
        ("gemm_q4k_resid", with(gemm_q4k_resid)),
        ("gemm_q6k_store", with(gemm_q6k_store)),
        ("gemm_q6k_resid", with(gemm_q6k_resid)),
        (
            "gemm_a4_store",
            clauses_a4::with(h, mlx, move |h, f| gemm_a4_store(h, f, sites)),
        ),
        (
            "gemm_a4_resid",
            clauses_a4::with(h, mlx, move |h, f| gemm_a4_resid(h, f, sites)),
        ),
        (
            "gemm_seg",
            Box::new(move || match (fixture, mlx) {
                (Ok(fix), Ok(split)) => gemm_seg(h, fix, split, sites),
                (Err(e), _) | (_, Err(e)) => Err(e.clone()),
            }),
        ),
        ("gemm_range", with(gemm_range)),
        (
            "gemm_half_rows",
            clauses_a4::with(h, mlx, move |h, f| gemm_half_rows(h, f, sites)),
        ),
        ("gemm_refusals", Box::new(move || gemm_refusals(h))),
    ]
}

/// The clauses of a row that runs its A4 sites only: [`table`]'s A4 clauses at
/// that row's sites.
pub fn shaped<'a>(
    h: &'a Harness<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    let sites = &clauses_a4::QWEN3_SITES;
    vec![
        (
            "gemm_a4_store",
            clauses_a4::with(h, mlx, move |h, f| gemm_a4_store(h, f, sites)),
        ),
        (
            "gemm_a4_resid",
            clauses_a4::with(h, mlx, move |h, f| gemm_a4_resid(h, f, sites)),
        ),
        (
            "gemm_seg",
            clauses_a4::with(h, mlx, move |h, f| gemm_seg_a4(h, f, sites)),
        ),
        (
            "gemm_half_rows",
            clauses_a4::with(h, mlx, move |h, f| gemm_half_rows(h, f, sites)),
        ),
    ]
}
