//! The M-row mat-vec clauses: every M-row launch's activation row is the
//! 1-row kernel's bits on that row's input alone — the contract the verify's
//! "verify ids == plain ids" clause stands on — a row whose input is not
//! finite stays in its own rows, and the launches refuse an M they hold no
//! entry of and a window the M rows do not fit, by name. The readers that norm
//! their input take it scaled, as the step's readers do: M rows of the stream
//! times the reader's gain and M scale words, which the hand-on over M rows
//! writes; the pair and the head take it no other way, and the segmented
//! mat-vec takes it on the load too (the first layer's, which reads the
//! embed). The scaled clauses feed them the rows the 1-row hand-on writes row
//! by row and hold them to the 1-row norm-on-load launch's bits (the bits the
//! 1-row scaled launch stores), and the hand-on's own clause holds each row of
//! its output to the 1-row hand-on's.
//!
//! The M-row family holds A4, the kind an MLX 4-bit folder of a Bf16 scale
//! pair opens as, so every clause launches A4 over synthetic weights
//! ([`synthetic::matrix`], its groups' biases made live) at the dense Qwen3.8-27B decode shapes the probe
//! `metal_matvec_time --only mrow` times: the two P1 segment groups, the
//! attention out and the FFN down, and the head's whole vocabulary. The pair
//! runs at the instance's own experts (the width and slots its geometry
//! names), the one shape the launch checks against the instance.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::kernels::{
    HEAD_SIMD_GROUPS, HeadMParams, HeadParams, LoadNorm, MROWS, MatvecMParams, NormNextMParams,
    NormNextParams, PairMParams, Quant, SegMParams, Segment,
};

use super::clauses_experts::{
    N_EXPERTS, PAIR_COLS, PairIn, PairWeights, launch_pair, synthetic_norm, synthetic_pair_for,
};
use super::clauses_rows::same_bits;
use super::common::{self, BadArg, LAYER};
use super::oracle;
use super::synthetic;
use super::{
    ACT_PAD, Harness, Launch, ModeSpec, OUT_GUARD, SENTINEL_BITS, SegLaunch, VALUE_PAD,
    WEIGHT_TAIL, check_guards, padded, weight_pad,
};

/// The epsilon every M-row norm takes.
const EPS: f32 = 1e-6;

/// The rows every launch runs.
const M: usize = MROWS;

/// The 27B's residual width, dense FFN width and vocabulary.
const HIDDEN: usize = 5120;
const FF: usize = 17_408;
const VOCAB: usize = 248_320;

/// The delta layer's P1 segments and the attention layer's, the 27B step's
/// two segment groups.
const DELTA_SEGMENTS: [usize; 4] = [10_240, 6144, 48, 48];
const ATTN_SEGMENTS: [usize; 3] = [12_288, 1024, 1024];

/// The attention out projection's columns: 24 heads of 256.
const OUT_COLS: usize = 6144;

/// Output rows of the tail shape: an odd count, so no SIMD group's rows divide
/// it.
const TAIL_ROWS: usize = 1531;

const QUANT: Quant = Quant::A4;

/// Where an A4 group's bias sits in its block: after the 32 code bytes and the
/// 2-byte scale.
const BIAS_AT: usize = 34;

/// One plain M-row launch of a clause: `M` rows of `rows` outputs over `cols`
/// columns, `x` holding the input rows and `y0` the output's own rows, which
/// the launch adds into.
struct MatvecCase<'a> {
    rows: usize,
    cols: usize,
    weights: &'a [u8],
    x: &'a [f32],
    y0: &'a [f32],
}

impl MatvecCase<'_> {
    /// The launch's outputs read back, the window's guards checked.
    fn run(&self, h: &Harness<'_>) -> Result<Vec<f32>, String> {
        let (device, stream) = (h.device, h.stream);
        let (m, rows, cols) = (M, self.rows, self.cols);
        let sentinel = f32::from_bits(SENTINEL_BITS);
        let pad = weight_pad(QUANT);
        let weights_buf = device
            .upload(stream, &padded(0xFFu8, pad, self.weights, WEIGHT_TAIL))
            .map_err(why)?;
        let x_buf = device
            .upload(stream, &padded(sentinel, ACT_PAD, self.x, 1))
            .map_err(why)?;
        let mut y = device
            .upload(stream, &padded(sentinel, OUT_GUARD, self.y0, OUT_GUARD))
            .map_err(why)?;
        h.encode(|encoder| {
            h.kernels.matvec_m(
                encoder,
                MatvecMParams {
                    quant: QUANT,
                    m,
                    rows,
                    cols,
                    weights: weights_buf.window(Elems(pad), Elems(self.weights.len()))?,
                    x: x_buf.window(Elems(ACT_PAD), Elems(m * cols))?,
                    y: y.window_mut(Elems(OUT_GUARD), Elems(m * rows))?,
                },
            )
        })?;
        read_guards(h, &y, m * rows)
    }
}

/// What an M-row reader takes as its input: the raw rows with the norm's gain
/// (the norm on the load), or the rows a hand-on wrote and their scale words
/// (`xn`: `m` rows of the reader's columns, then `m` words).
#[derive(Clone, Copy)]
enum Feed<'a> {
    Gain { x: &'a [f32], gain: &'a [f32] },
    Scaled { xn: &'a [f32] },
}

impl Feed<'_> {
    /// The buffer's words: the input rows, and for the scaled feed the scale
    /// words after them.
    fn words(&self) -> &[f32] {
        match self {
            Self::Gain { x, .. } => x,
            Self::Scaled { xn } => xn,
        }
    }

    /// The gain's words, empty for the scaled feed.
    fn gain(&self) -> &[f32] {
        match self {
            Self::Gain { gain, .. } => gain,
            Self::Scaled { .. } => &[],
        }
    }

    /// The norm over `x_buf` (the input's buffer, at `x_at`) and `gain_buf`
    /// (at `gain_at`): `m` rows of `cols`.
    fn norm<'b>(
        &self,
        x_buf: &'b MetalBuffer<f32>,
        x_at: usize,
        gain_buf: &'b MetalBuffer<f32>,
        gain_at: usize,
        m: usize,
        cols: usize,
    ) -> Result<LoadNorm<'b>, DeviceError> {
        Ok(match self {
            Self::Gain { .. } => LoadNorm::Gain {
                gain: gain_buf.window(Elems(gain_at), Elems(cols))?,
                eps: EPS,
            },
            Self::Scaled { .. } => LoadNorm::Scaled {
                scale: x_buf.window(Elems(x_at + m * cols), Elems(m))?,
            },
        })
    }
}

/// The M-row segmented mat-vec over the segments' weights: the outputs read
/// back, the window's guards checked.
fn seg_m(
    h: &Harness<'_>,
    segments: &[(usize, &[u8])],
    cols: usize,
    feed: Feed<'_>,
) -> Result<Vec<f32>, String> {
    let m = M;
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let block = QUANT.block_bytes();
    let mut weights: Vec<u8> = Vec::new();
    let mut segs: Vec<Segment> = Vec::new();
    let mut out_offset = 0usize;
    for (rows, bytes) in segments {
        segs.push(Segment {
            cols,
            rows: *rows,
            weight_blocks: weights.len() / block,
            out_offset,
        });
        weights.extend_from_slice(bytes);
        out_offset += rows;
    }
    let pad = weight_pad(QUANT);
    let weights_buf = device
        .upload(stream, &padded(0xFFu8, pad, &weights, WEIGHT_TAIL))
        .map_err(why)?;
    let x_buf = device
        .upload(stream, &padded(sentinel, ACT_PAD, feed.words(), 1))
        .map_err(why)?;
    let gain_buf = device
        .upload(stream, &padded(sentinel, ACT_PAD, feed.gain(), 1))
        .map_err(why)?;
    let y_words = padded(
        sentinel,
        OUT_GUARD,
        &common::sentinel_rows(m * out_offset),
        OUT_GUARD,
    );
    let mut y = device.upload(stream, &y_words).map_err(why)?;
    h.encode(|encoder| {
        h.kernels.matvec_seg_m(
            encoder,
            SegMParams {
                quant: QUANT,
                m,
                segments: &segs,
                weights: weights_buf.window(Elems(pad), Elems(weights.len()))?,
                x: x_buf.window(Elems(ACT_PAD), Elems(m * cols))?,
                y: y.window_mut(Elems(OUT_GUARD), Elems(m * out_offset))?,
                norm: feed.norm(&x_buf, ACT_PAD, &gain_buf, ACT_PAD, m, cols)?,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    read_guards(h, &y, m * out_offset)
}

/// The M-row pair over the stacks and picks `launch` names, its buffers the
/// 1-row pair's own, reading the hand-on's rows `xn` (`M` rows, then `M` scale
/// words): the outputs read back, the window's guards checked.
fn pair_m(h: &Harness<'_>, launch: &PairIn<'_>, xn: &[f32]) -> Result<Vec<f32>, String> {
    let m = M;
    let e = h.geom().experts;
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let pad = weight_pad(launch.quant);
    let gate = device
        .upload(stream, &padded(0xFFu8, pad, launch.gate, WEIGHT_TAIL))
        .map_err(why)?;
    let up = device
        .upload(stream, &padded(0xFFu8, pad, launch.up, WEIGHT_TAIL))
        .map_err(why)?;
    let shared_gate = device
        .upload(
            stream,
            &padded(0xFFu8, pad, launch.shared_gate, WEIGHT_TAIL),
        )
        .map_err(why)?;
    let shared_up = device
        .upload(stream, &padded(0xFFu8, pad, launch.shared_up, WEIGHT_TAIL))
        .map_err(why)?;
    let x_buf = device
        .upload(stream, &padded(sentinel, ACT_PAD, xn, 1))
        .map_err(why)?;
    let ids = device
        .upload(stream, &padded(-1, VALUE_PAD, launch.ids, 1))
        .map_err(why)?;
    let cols = launch.norm.gain.len();
    let y_len = m * e.slots * e.k;
    let mut y = device
        .upload(
            stream,
            &padded(
                sentinel,
                OUT_GUARD,
                &common::sentinel_rows(y_len),
                OUT_GUARD,
            ),
        )
        .map_err(why)?;
    let n_experts = launch.n_experts;
    let quant = launch.quant;
    h.encode(|encoder| {
        h.kernels.pair_m(
            encoder,
            PairMParams {
                quant,
                shared_quant: quant,
                m,
                slots: e.slots,
                k: e.k,
                cols,
                n_experts,
                ids: ids.window(Elems(VALUE_PAD), Elems(launch.ids.len()))?,
                gate: gate.window(Elems(pad), Elems(launch.gate.len()))?,
                up: up.window(Elems(pad), Elems(launch.up.len()))?,
                shared_gate: e
                    .has_shared
                    .then(|| shared_gate.window(Elems(pad), Elems(launch.shared_gate.len())))
                    .transpose()?,
                shared_up: e
                    .has_shared
                    .then(|| shared_up.window(Elems(pad), Elems(launch.shared_up.len())))
                    .transpose()?,
                x: x_buf.window(Elems(ACT_PAD), Elems(m * cols))?,
                scale: x_buf.window(Elems(ACT_PAD + m * cols), Elems(m))?,
                y: y.window_mut(Elems(OUT_GUARD), Elems(y_len))?,
                out_stride: e.k,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    read_guards(h, &y, y_len)
}

/// The head's own uploaded buffers, shared by its M-row launch and the 1-row
/// references: the weights are the vocabulary's whole width.
struct HeadBufs {
    weights: MetalBuffer<u8>,
    x: MetalBuffer<f32>,
    gain: MetalBuffer<f32>,
    /// `m` rows of logits, one row a pick.
    logits: MetalBuffer<f32>,
    picks: MetalBuffer<u32>,
}

impl HeadBufs {
    /// One M-row head launch, the logits and the picks read back. `handed`:
    /// the hand-on's rows and their scale words, which the launch reads.
    fn run_m(
        &mut self,
        h: &Harness<'_>,
        handed: &MetalBuffer<f32>,
    ) -> Result<(Vec<f32>, Vec<u32>), String> {
        let m = M;
        h.encode(|encoder| {
            h.kernels.head_m(
                encoder,
                HeadMParams {
                    quant: QUANT,
                    m,
                    rows: VOCAB,
                    k: HIDDEN,
                    simdgroups: HEAD_SIMD_GROUPS,
                    weights: self
                        .weights
                        .window(Elems(0), Elems(self.weights.len().get()))?,
                    x: handed.window(Elems(0), Elems(m * HIDDEN))?,
                    scale: handed.window(Elems(m * HIDDEN), Elems(m))?,
                    logits: self.logits.window_mut(Elems(0), Elems(m * VOCAB))?,
                    step_ids: self.picks.window_mut(Elems(0), Elems(m))?,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        })?;
        self.read(h, m)
    }

    /// The 1-row head over activation row `at`, into that row's own logits
    /// and pick of the shared buffers.
    fn run_one(&mut self, h: &Harness<'_>, at: usize) -> Result<(Vec<f32>, u32), String> {
        h.encode(|encoder| {
            h.kernels.head(
                encoder,
                HeadParams {
                    quant: QUANT,
                    rows: VOCAB,
                    k: HIDDEN,
                    simdgroups: HEAD_SIMD_GROUPS,
                    weights: self
                        .weights
                        .window(Elems(0), Elems(self.weights.len().get()))?,
                    x: self.x.window(Elems(at * HIDDEN), Elems(HIDDEN))?,
                    norm: LoadNorm::Gain {
                        gain: self.gain.window(Elems(0), Elems(HIDDEN))?,
                        eps: EPS,
                    },
                    logits: self.logits.window_mut(Elems(at * VOCAB), Elems(VOCAB))?,
                    step_id: self.picks.window_mut(Elems(at), Elems(1))?,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        })?;
        let (logits, picks) = self.read(h, at + 1)?;
        Ok((logits[at * VOCAB..(at + 1) * VOCAB].to_vec(), picks[at]))
    }

    /// The first `rows` logits rows and picks read back.
    fn read(&self, h: &Harness<'_>, rows: usize) -> Result<(Vec<f32>, Vec<u32>), String> {
        let mut logits = vec![0f32; rows * VOCAB];
        let mut picks = vec![0u32; rows];
        h.device
            .read(
                h.stream,
                self.logits
                    .window(Elems(0), Elems(rows * VOCAB))
                    .map_err(why)?,
                &mut logits,
            )
            .map_err(why)?;
        h.device
            .read(
                h.stream,
                self.picks.window(Elems(0), Elems(rows)).map_err(why)?,
                &mut picks,
            )
            .map_err(why)?;
        Ok((logits, picks))
    }
}

/// The outputs read back with their guards checked.
fn read_guards(h: &Harness<'_>, y: &MetalBuffer<f32>, words: usize) -> Result<Vec<f32>, String> {
    let mut back = vec![0f32; words + 2 * OUT_GUARD];
    h.device
        .read(h.stream, y.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    check_guards(&back, words)?;
    Ok(back[OUT_GUARD..OUT_GUARD + words].to_vec())
}

/// `rows` activation rows of `cols`, each its own seed and scale.
fn x_rows(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for m in 0..rows {
        out.extend(oracle::activations(
            cols,
            seed + m as u64,
            1.0 + 0.25 * m as f32,
        ));
    }
    out
}

/// A positive synthetic gain, as the other synthetic clauses take.
fn gain(cols: usize, seed: u64) -> Vec<f32> {
    oracle::activations(cols, seed, 0.5)
        .iter()
        .map(|g| 1.0 + g.abs())
        .collect()
}

/// Activation row `at`'s outputs against the 1-row kernel's on that row's
/// input alone: every word the same bits.
fn row_diff(got: &[f32], want: &[f32], at: usize) -> Result<(), String> {
    same_bits(&format!("row {at} against the 1-row kernel"), got, want)
}

/// A4 weights whose biases are live. `synthetic::matrix` leaves every group's
/// bias zero, which would hide a defect in how the bias term takes its input
/// row's sum; each group here gets a small BF16 bias of a sign and size drawn
/// by its index.
fn biased(mut weights: Vec<u8>) -> Vec<u8> {
    for (i, block) in weights.chunks_exact_mut(QUANT.block_bytes()).enumerate() {
        let steps = ((i * 7 + 3) % 17) as f32 - 8.0;
        block[BIAS_AT..BIAS_AT + 2]
            .copy_from_slice(&gguf::quant::fp32_to_bf16_bits(steps * 2.5e-4).to_le_bytes());
    }
    weights
}

/// A matrix of `synthetic::matrix`'s random groups with live biases.
fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    biased(synthetic::matrix(QUANT, rows, cols, seed))
}

/// The plain M-row residual mat-vec at the 27B's attention out and FFN down
/// shapes and at row counts a SIMD group's rows do not divide, every row the
/// 1-row kernel's bits on that row's input and residual.
pub fn matvec(h: &Harness<'_>) -> Clause {
    for (name, rows, cols) in [
        ("attn_out", HIDDEN, OUT_COLS),
        ("down", HIDDEN, FF),
        // A row count a SIMD group's rows do not divide: the last group's
        // dead rows are never stored.
        ("tail_resid", TAIL_ROWS, OUT_COLS),
    ] {
        let weights = matrix(rows, cols, 0x4D31 + M as u64);
        let x = x_rows(M, cols, 0x4D40 + M as u64);
        let y0 = x_rows(M, rows, 0x4D50 + M as u64);
        h.clear_fault()?;
        let got = MatvecCase {
            rows,
            cols,
            weights: &weights,
            x: &x,
            y0: &y0,
        }
        .run(h)
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        common::expect_clean(h, "the M-row mat-vec")?;
        for row in 0..M {
            let alone = h
                .matvec(&Launch {
                    quant: QUANT,
                    rows,
                    cols,
                    weights: &weights,
                    x: &x[row * cols..(row + 1) * cols],
                    y0: &y0[row * rows..(row + 1) * rows],
                    mode: ModeSpec::Residual,
                })
                .map_err(|e| format!("{name}: {e}"))?;
            let at = row * rows;
            row_diff(&got[at..at + rows], &alone, row)
                .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        }
    }
    // One row's NaN input, each row in turn: that row's every output is NaN,
    // the other row's the 1-row kernel's bits on its own input and residual.
    let (rows, cols) = (HIDDEN, OUT_COLS);
    let weights = matrix(rows, cols, 0x4D61);
    let y0 = x_rows(M, rows, 0x4D63);
    for nan_row in 0..M {
        let mut bad = x_rows(M, cols, 0x4D62);
        bad[nan_row * cols + 17] = f32::NAN;
        let got = MatvecCase {
            rows,
            cols,
            weights: &weights,
            x: &bad,
            y0: &y0,
        }
        .run(h)
        .map_err(|e| format!("a NaN input row {nan_row}: {e}"))?;
        for row in 0..M {
            let mine = &got[row * rows..(row + 1) * rows];
            if row == nan_row {
                if let Some(r) = mine.iter().position(|v| !v.is_nan()) {
                    return Err(format!("the NaN row kept output {r} finite: {:?}", mine[r]));
                }
                continue;
            }
            let alone = h
                .matvec(&Launch {
                    quant: QUANT,
                    rows,
                    cols,
                    weights: &weights,
                    x: &bad[row * cols..(row + 1) * cols],
                    y0: &y0[row * rows..(row + 1) * rows],
                    mode: ModeSpec::Residual,
                })
                .map_err(|e| format!("a NaN input row {nan_row}: {e}"))?;
            row_diff(mine, &alone, row).map_err(|e| format!("a NaN input row {nan_row}: {e}"))?;
        }
    }
    Ok(Verdict::Noted(
        "residual at the out, down and row-tail shapes: every row of M = 2 is the 1-row kernel's \
         bits; a NaN input row is NaN in its own row only"
            .to_string(),
    ))
}

/// The segmented M-row mat-vec at the 27B's two P1 segment groups, every row
/// the 1-row segmented launch's bits; one row's NaN input raises the norm
/// site and is NaN in its own row only.
pub fn seg(h: &Harness<'_>) -> Clause {
    let groups: [(&str, &[usize]); 2] =
        [("delta_p1", &DELTA_SEGMENTS), ("attn_p1", &ATTN_SEGMENTS)];
    for (gi, (name, rows_of)) in groups.iter().enumerate() {
        let owned: Vec<Vec<u8>> = rows_of
            .iter()
            .enumerate()
            .map(|(j, n)| matrix(*n, HIDDEN, 0x4E10 + (8 * gi + j) as u64))
            .collect();
        let segments: Vec<(usize, &[u8])> = rows_of
            .iter()
            .zip(&owned)
            .map(|(n, w)| (*n, w.as_slice()))
            .collect();
        let g = gain(HIDDEN, 0x4E30 + gi as u64);
        let rows: usize = rows_of.iter().sum();
        let alone = |x: &[f32], row: usize| -> Result<Vec<f32>, String> {
            h.seg(&SegLaunch {
                quant: QUANT,
                cols: HIDDEN,
                segments: segments.clone(),
                x: &x[row * HIDDEN..(row + 1) * HIDDEN],
                y0: &common::sentinel_rows(rows),
                gain: &g,
                eps: EPS,
                layer: LAYER,
            })
        };
        let x = x_rows(M, HIDDEN, 0x4E20 + M as u64);
        h.clear_fault()?;
        let got = seg_m(h, &segments, HIDDEN, Feed::Gain { x: &x, gain: &g })
            .map_err(|e| format!("{name}: {e}"))?;
        common::expect_clean(h, "the M-row segmented mat-vec")?;
        for row in 0..M {
            let want = alone(&x, row).map_err(|e| format!("{name}: {e}"))?;
            let at = row * rows;
            row_diff(&got[at..at + rows], &want, row).map_err(|e| format!("{name}: {e}"))?;
        }
        // One row's NaN input, each row in turn: the norm site raised, that
        // row NaN, the other row the clean 1-row bits.
        for nan_row in 0..M {
            let mut bad = x_rows(M, HIDDEN, 0x4E2F);
            bad[nan_row * HIDDEN + 17] = f32::NAN;
            h.clear_fault()?;
            let got = seg_m(h, &segments, HIDDEN, Feed::Gain { x: &bad, gain: &g })?;
            common::expect_fault(h, "a NaN input row", FaultSite::NormQuant)?;
            for row in 0..M {
                let mine = &got[row * rows..(row + 1) * rows];
                if row == nan_row {
                    if let Some(r) = mine.iter().position(|v| !v.is_nan()) {
                        return Err(format!(
                            "{name}'s NaN row {nan_row} kept output {r} finite: {:?}",
                            mine[r]
                        ));
                    }
                } else {
                    let want = alone(&bad, row).map_err(|e| format!("{name}'s NaN case: {e}"))?;
                    row_diff(mine, &want, row).map_err(|e| format!("{name}'s NaN case: {e}"))?;
                }
            }
            h.clear_fault()?;
        }
    }
    Ok(Verdict::Noted(
        "both P1 groups: every row of M = 2 is the 1-row launch's bits; a NaN input row raised \
         norm_quant and stayed in its own row"
            .to_string(),
    ))
}

/// The 1-row pair over activation row `row` of `x`, the shared stacks and
/// picks of `launch`.
fn pair_row(
    h: &Harness<'_>,
    launch: &PairIn<'_>,
    x: &[f32],
    cols: usize,
    row: usize,
) -> Result<Vec<f32>, String> {
    launch_pair(
        h,
        PairIn {
            x: &x[row * cols..(row + 1) * cols],
            ..pair_in_of(launch)
        },
    )
}

/// A copy of `launch`'s borrowed fields, for the struct update syntax.
fn pair_in_of<'a>(launch: &PairIn<'a>) -> PairIn<'a> {
    PairIn {
        quant: launch.quant,
        shared_quant: launch.shared_quant,
        n_experts: launch.n_experts,
        ids: launch.ids,
        gate: launch.gate,
        up: launch.up,
        shared_gate: launch.shared_gate,
        shared_up: launch.shared_up,
        x: launch.x,
        norm: launch.norm,
        y0: launch.y0,
    }
}

/// The hand-on of one stream row by the 1-row launch, the step's own: the
/// row times `gain` and the scale word after it, one more word than the row.
fn next_one(h: &Harness<'_>, row: &[f32], gain: &[f32]) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let cols = row.len();
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let x = device.upload(stream, row).map_err(why)?;
    let g = device.upload(stream, gain).map_err(why)?;
    let mut out = device
        .upload(stream, &vec![sentinel; cols + 2])
        .map_err(why)?;
    h.encode(|encoder| {
        h.kernels.norm_next(
            encoder,
            NormNextParams {
                quant: QUANT,
                x: x.window(Elems(0), Elems(cols))?,
                gain: g.window(Elems(0), Elems(cols))?,
                eps: EPS,
                layer: LAYER,
                out: out.window_mut(Elems(0), Elems(cols + 1))?,
                fault: h.device.fault_word(),
            },
        )
    })?;
    let mut back = vec![0f32; cols + 2];
    device
        .read(stream, out.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if back[cols + 1].to_bits() != SENTINEL_BITS {
        return Err("the 1-row hand-on wrote past its scale word".to_string());
    }
    back.truncate(cols + 1);
    Ok(back)
}

/// What the step hands a reader of `M` rows, row by row with the 1-row
/// hand-on: the rows times `gain` (`cols` values each), then each row's scale
/// word. The faults the rows raise are left in the fault word.
fn hand_on_rows(h: &Harness<'_>, x: &[f32], cols: usize, gain: &[f32]) -> Result<Vec<f32>, String> {
    let mut rows = Vec::with_capacity(M * cols + M);
    let mut scales = Vec::with_capacity(M);
    for r in 0..M {
        let one = next_one(h, &x[r * cols..(r + 1) * cols], gain)?;
        rows.extend_from_slice(&one[..cols]);
        scales.push(one[cols]);
    }
    rows.extend(scales);
    Ok(rows)
}

/// The hand-on over `M` rows of `x`: the rows times `gain` and the `M` scale
/// words, the guards around the output window checked: the front guard is a
/// vector wide, so the window starts on 16 bytes as the launch needs.
fn next_m(h: &Harness<'_>, x: &[f32], cols: usize, gain: &[f32]) -> Result<Vec<f32>, String> {
    let m = M;
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let words = m * cols + m;
    let x_buf = device
        .upload(stream, &padded(sentinel, ACT_PAD, x, ACT_PAD))
        .map_err(why)?;
    let gain_buf = device
        .upload(stream, &padded(sentinel, ACT_PAD, gain, ACT_PAD))
        .map_err(why)?;
    let mut out = device
        .upload(
            stream,
            &padded(sentinel, ACT_PAD, &common::sentinel_rows(words), ACT_PAD),
        )
        .map_err(why)?;
    h.encode(|encoder| {
        h.kernels.norm_next_m(
            encoder,
            NormNextMParams {
                quant: QUANT,
                m,
                x: x_buf.window(Elems(ACT_PAD), Elems(m * cols))?,
                gain: gain_buf.window(Elems(ACT_PAD), Elems(cols))?,
                eps: EPS,
                layer: LAYER,
                out: out.window_mut(Elems(ACT_PAD), Elems(words))?,
                fault: h.device.fault_word(),
            },
        )
    })?;
    let mut back = vec![0f32; words + 2 * ACT_PAD];
    device
        .read(stream, out.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    let guards = back[..ACT_PAD].iter().chain(&back[ACT_PAD + words..]);
    if guards.into_iter().any(|v| v.to_bits() != SENTINEL_BITS) {
        return Err("the hand-on wrote outside its output window".to_string());
    }
    Ok(back[ACT_PAD..ACT_PAD + words].to_vec())
}

/// The hand-on over M rows of the 27B's stream: each row's values and scale
/// word the 1-row hand-on's bits on that row alone; a row whose stream is not
/// finite raises the norm site and hands on NaN, the other row the 1-row bits.
pub fn hand_on(h: &Harness<'_>) -> Clause {
    let g = gain(HIDDEN, 0x5010);
    let x = x_rows(M, HIDDEN, 0x5020 + M as u64);
    h.clear_fault()?;
    let want = hand_on_rows(h, &x, HIDDEN, &g)?;
    common::expect_clean(h, "the 1-row hand-on")?;
    let got = next_m(h, &x, HIDDEN, &g)?;
    common::expect_clean(h, "the M-row hand-on")?;
    for row in 0..M {
        same_bits(
            &format!("row {row}'s values against the 1-row hand-on"),
            &got[row * HIDDEN..(row + 1) * HIDDEN],
            &want[row * HIDDEN..(row + 1) * HIDDEN],
        )?;
    }
    same_bits(
        "the scale words against the 1-row hand-on's",
        &got[M * HIDDEN..],
        &want[M * HIDDEN..],
    )?;
    // One row's NaN stream, each row in turn: its scale NaN and the norm site
    // raised, the other row's words the clean 1-row words.
    for nan_row in 0..M {
        let mut bad = x_rows(M, HIDDEN, 0x502F);
        bad[nan_row * HIDDEN + 17] = f32::NAN;
        h.clear_fault()?;
        let want = hand_on_rows(h, &bad, HIDDEN, &g)?;
        h.clear_fault()?;
        let got = next_m(h, &bad, HIDDEN, &g)?;
        common::expect_fault(h, "a NaN stream row", FaultSite::NormQuant)?;
        if !got[M * HIDDEN + nan_row].is_nan() {
            return Err(format!(
                "the NaN row {nan_row}'s scale is {:?}, not NaN",
                got[M * HIDDEN + nan_row]
            ));
        }
        for row in (0..M).filter(|row| *row != nan_row) {
            same_bits(
                &format!("a NaN stream row {nan_row}: row {row}"),
                &got[row * HIDDEN..(row + 1) * HIDDEN],
                &want[row * HIDDEN..(row + 1) * HIDDEN],
            )?;
            if got[M * HIDDEN + row].to_bits() != want[M * HIDDEN + row].to_bits() {
                return Err(format!(
                    "a NaN stream row {nan_row} changed row {row}'s scale: {:?} against {:?}",
                    got[M * HIDDEN + row],
                    want[M * HIDDEN + row]
                ));
            }
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "{HIDDEN} values a row: every row of M = 2 and its scale word are the 1-row hand-on's \
         bits; a NaN stream row raised norm_quant and handed on a NaN scale, the other row's \
         words unchanged"
    )))
}

/// The scaled M-row segmented mat-vec at the 27B's two P1 segment groups:
/// the rows the 1-row hand-on wrote, read scaled, are the 1-row norm-on-load
/// launch's bits (the engine step's) on each row's stream; a NaN stream row
/// is NaN in its own row, with the norm site raised, and the other row keeps
/// the 1-row bits.
pub fn seg_scaled(h: &Harness<'_>) -> Clause {
    let groups: [(&str, &[usize]); 2] =
        [("delta_p1", &DELTA_SEGMENTS), ("attn_p1", &ATTN_SEGMENTS)];
    for (gi, (name, rows_of)) in groups.iter().enumerate() {
        let owned: Vec<Vec<u8>> = rows_of
            .iter()
            .enumerate()
            .map(|(j, n)| matrix(*n, HIDDEN, 0x5110 + (8 * gi + j) as u64))
            .collect();
        let segments: Vec<(usize, &[u8])> = rows_of
            .iter()
            .zip(&owned)
            .map(|(n, w)| (*n, w.as_slice()))
            .collect();
        let g = gain(HIDDEN, 0x5130 + gi as u64);
        let rows: usize = rows_of.iter().sum();
        let alone = |x: &[f32], row: usize| -> Result<Vec<f32>, String> {
            h.seg(&SegLaunch {
                quant: QUANT,
                cols: HIDDEN,
                segments: segments.clone(),
                x: &x[row * HIDDEN..(row + 1) * HIDDEN],
                y0: &common::sentinel_rows(rows),
                gain: &g,
                eps: EPS,
                layer: LAYER,
            })
        };
        let x = x_rows(M, HIDDEN, 0x5120 + M as u64);
        let xn = hand_on_rows(h, &x, HIDDEN, &g)?;
        h.clear_fault()?;
        let got = seg_m(h, &segments, HIDDEN, Feed::Scaled { xn: &xn })
            .map_err(|e| format!("{name}: {e}"))?;
        common::expect_clean(h, "the scaled M-row segmented mat-vec")?;
        for row in 0..M {
            let want = alone(&x, row).map_err(|e| format!("{name}: {e}"))?;
            let at = row * rows;
            row_diff(&got[at..at + rows], &want, row).map_err(|e| format!("{name}: {e}"))?;
        }
        for nan_row in 0..M {
            let mut bad = x_rows(M, HIDDEN, 0x512F);
            bad[nan_row * HIDDEN + 17] = f32::NAN;
            let xn = hand_on_rows(h, &bad, HIDDEN, &g)?;
            h.clear_fault()?;
            let got = seg_m(h, &segments, HIDDEN, Feed::Scaled { xn: &xn })?;
            common::expect_fault(h, "a NaN stream row", FaultSite::NormQuant)?;
            for row in 0..M {
                let mine = &got[row * rows..(row + 1) * rows];
                if row == nan_row {
                    if let Some(r) = mine.iter().position(|v| !v.is_nan()) {
                        return Err(format!(
                            "{name}'s NaN row {nan_row} kept output {r} finite: {:?}",
                            mine[r]
                        ));
                    }
                } else {
                    let want = alone(&bad, row).map_err(|e| format!("{name}'s NaN case: {e}"))?;
                    row_diff(mine, &want, row).map_err(|e| format!("{name}'s NaN case: {e}"))?;
                }
            }
            h.clear_fault()?;
        }
    }
    Ok(Verdict::Noted(
        "both P1 groups: every row of M = 2 read scaled is the 1-row norm-on-load launch's \
         bits; a NaN stream row raised norm_quant and stayed in its own row"
            .to_string(),
    ))
}

/// The scaled M-row gate-up pair at the instance's own experts: every row the
/// 1-row pair's bits over the same stacks and picks; a NaN stream row is NaN
/// in its own row's slots only; an id past the stack is NaN in its slot of
/// every row and raises the expert-id site.
pub fn pair_scaled(h: &Harness<'_>) -> Clause {
    let e = h.geom().experts;
    let cols = PAIR_COLS;
    let weights = synthetic_pair_for(&e, QUANT, 0x5210);
    let weights = PairWeights {
        gate: biased(weights.gate),
        up: biased(weights.up),
        shared_gate: biased(weights.shared_gate),
        shared_up: biased(weights.shared_up),
    };
    let norm = synthetic_norm(EPS);
    let ids: Vec<i32> = (0..e.routed_ids).map(|i| (i % 4) as i32).collect();
    let launch = PairIn {
        quant: QUANT,
        shared_quant: QUANT,
        n_experts: N_EXPERTS,
        ids: &ids,
        gate: &weights.gate,
        up: &weights.up,
        shared_gate: &weights.shared_gate,
        shared_up: &weights.shared_up,
        x: &x_rows(1, cols, 0x5220),
        norm: &norm,
        y0: &common::sentinel_rows(e.slots * e.k),
    };
    let out_words = e.slots * e.k;
    let x = x_rows(M, cols, 0x5220 + M as u64);
    let xn = hand_on_rows(h, &x, cols, &norm.gain)?;
    h.clear_fault()?;
    let got = pair_m(h, &launch, &xn)?;
    common::expect_clean(h, "the scaled M-row pair")?;
    for row in 0..M {
        let want = pair_row(h, &launch, &x, cols, row)?;
        let at = row * out_words;
        row_diff(&got[at..at + out_words], &want, row)?;
    }
    let mut bad = x_rows(M, cols, 0x522F);
    bad[cols + 17] = f32::NAN;
    let xn = hand_on_rows(h, &bad, cols, &norm.gain)?;
    h.clear_fault()?;
    let got = pair_m(h, &launch, &xn)?;
    for row in 0..M {
        let want = pair_row(h, &launch, &bad, cols, row)?;
        let at = row * out_words;
        let mine = &got[at..at + out_words];
        if row == 1 {
            if let Some(r) = mine.iter().position(|v| !v.is_nan()) {
                return Err(format!("the NaN row kept word {r} finite: {:?}", mine[r]));
            }
        } else {
            row_diff(mine, &want, row)?;
        }
    }
    h.clear_fault()?;
    // A routed id past the stack: that slot is NaN in every row and raises the
    // expert-id site, every row the 1-row pair's bits.
    let bad_slot = 2usize.min(e.routed_ids - 1);
    let mut bad_ids = ids.clone();
    bad_ids[bad_slot] = N_EXPERTS as i32;
    let bad_launch = PairIn {
        ids: &bad_ids,
        ..pair_in_of(&launch)
    };
    let x = x_rows(M, cols, 0x522E);
    let xn = hand_on_rows(h, &x, cols, &norm.gain)?;
    h.clear_fault()?;
    let got = pair_m(h, &bad_launch, &xn)?;
    common::expect_fault(h, "an id past the stack", FaultSite::ExpertId)?;
    for row in 0..M {
        let want = pair_row(h, &bad_launch, &x, cols, row)?;
        let at = row * out_words;
        row_diff(&got[at..at + out_words], &want, row)
            .map_err(|e| format!("an id past the stack: {e}"))?;
        let slot = &got[at + bad_slot * e.k..at + (bad_slot + 1) * e.k];
        if let Some(r) = slot.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "an id past the stack left row {row}'s slot {bad_slot} word {r} finite: {:?}",
                slot[r]
            ));
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "at the instance's {} slots of {}: every row of M = 2 read scaled is the 1-row pair's \
         bits; a NaN stream row stayed in its own row's slots; an id past the stack raised \
         expert_id in its slot of every row",
        e.slots, e.k
    )))
}

/// The scaled M-row head over the whole 27B vocabulary: every row's logits and
/// pick the 1-row head's on that row's stream; a NaN stream row leaves its own
/// logits NaN and its pick no token, the other row clean.
pub fn head_scaled(h: &Harness<'_>) -> Clause {
    let weights = matrix(VOCAB, HIDDEN, 0x5250);
    let g = gain(HIDDEN, 0x5260);
    let (device, stream) = (h.device, h.stream);
    let bufs = |x: &[f32]| -> Result<HeadBufs, String> {
        Ok(HeadBufs {
            weights: device.upload(stream, &weights).map_err(why)?,
            x: device.upload(stream, x).map_err(why)?,
            gain: device.upload(stream, &g).map_err(why)?,
            logits: device
                .upload(
                    stream,
                    &vec![f32::from_bits(SENTINEL_BITS); (x.len() / HIDDEN) * VOCAB],
                )
                .map_err(why)?,
            picks: device
                .upload(stream, &vec![u32::MAX; x.len() / HIDDEN])
                .map_err(why)?,
        })
    };
    let x = x_rows(M, HIDDEN, 0x5270 + M as u64);
    let xn = hand_on_rows(h, &x, HIDDEN, &g)?;
    let handed = device.upload(stream, &xn).map_err(why)?;
    let mut run = bufs(&x)?;
    h.clear_fault()?;
    let (logits, picks) = run.run_m(h, &handed)?;
    common::expect_clean(h, "the scaled M-row head")?;
    for (row, pick) in picks.iter().copied().enumerate() {
        let (alone, id) = run.run_one(h, row)?;
        let at = row * VOCAB;
        row_diff(&logits[at..at + VOCAB], &alone, row)?;
        if pick != id {
            return Err(format!("row {row}'s pick is {pick}, the 1-row head's {id}"));
        }
    }
    let mut bad = x_rows(M, HIDDEN, 0x527F);
    bad[HIDDEN + 17] = f32::NAN;
    let xn = hand_on_rows(h, &bad, HIDDEN, &g)?;
    let handed = device.upload(stream, &xn).map_err(why)?;
    let mut run = bufs(&bad)?;
    h.clear_fault()?;
    let (logits, picks) = run.run_m(h, &handed)?;
    if h.fault()?.is_none() {
        return Err("a NaN stream row raised nothing".to_string());
    }
    for (row, pick) in picks.iter().copied().enumerate() {
        let (alone, id) = run.run_one(h, row)?;
        let at = row * VOCAB;
        let mine = &logits[at..at + VOCAB];
        if row == 1 {
            if let Some(r) = mine.iter().position(|v| !v.is_nan()) {
                return Err(format!("the NaN row kept logit {r} finite: {:?}", mine[r]));
            }
            if pick != 0xFFFF_FFFF {
                return Err(format!(
                    "the NaN row's pick is {pick}, not the no-token word"
                ));
            }
        } else {
            row_diff(mine, &alone, row)?;
            if pick != id {
                return Err(format!(
                    "the clean row's pick is {pick}, the 1-row head's {id}"
                ));
            }
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "{VOCAB}x{HIDDEN}: every row of M = 2 read scaled has the 1-row head's logits and \
         pick; a NaN stream row's pick is no token"
    )))
}

/// The buffers the refusal cases launch over: the weights of one small
/// tensor (`weights_len` bytes at the window's offset in its padded buffer),
/// four activation rows, a gain, an output window, the hand-on's output
/// (`out`, whose window starts on 16 bytes), the routed ids and four pick
/// words.
struct RefusalBufs {
    weights: MetalBuffer<u8>,
    weights_len: usize,
    x: MetalBuffer<f32>,
    gain: MetalBuffer<f32>,
    y: MetalBuffer<f32>,
    out: MetalBuffer<f32>,
    ids: MetalBuffer<i32>,
    picks: MetalBuffer<u32>,
}

/// The tensor's weights in `weights`, the window exactly its `len` bytes: the
/// launches check the weights' length first, so a window with the buffer's
/// guards would be the refusal of every case after it.
fn tensor(
    weights: &MetalBuffer<u8>,
    len: usize,
) -> Result<bloomery_device::Window<'_, u8, bloomery_metal::MetalDevice>, DeviceError> {
    weights.window(Elems(weight_pad(QUANT)), Elems(len))
}

/// A refusal case's plain launch over the refusal buffers: `m` rows of `rows`
/// outputs over `cols` columns, with `x_words` input words and `y_words`
/// output words; the cases that name a window pass a count the shape does not
/// take.
#[derive(Clone, Copy)]
struct Plain {
    quant: Quant,
    m: usize,
    rows: usize,
    cols: usize,
    x_words: usize,
    y_words: usize,
}

impl Plain {
    fn params<'b>(&self, b: &'b mut RefusalBufs) -> Result<MatvecMParams<'b>, DeviceError> {
        Ok(MatvecMParams {
            quant: self.quant,
            m: self.m,
            rows: self.rows,
            cols: self.cols,
            weights: tensor(&b.weights, b.weights_len)?,
            x: b.x.window(Elems(ACT_PAD), Elems(self.x_words))?,
            y: b.y.window_mut(Elems(OUT_GUARD), Elems(self.y_words))?,
        })
    }
}

/// The M-row launches' refusals: an M the family holds no entry of, a window
/// the M rows do not fit, the pair's instance shapes and the head's argmax
/// quads, each refused by name with nothing written, and the well-formed
/// launch accepted and writing.
pub fn refusals(h: &Harness<'_>) -> Clause {
    let rows = 64usize;
    let cols = 128usize;
    let weights = synthetic::matrix(QUANT, rows, cols, 0x4F90);
    let x = x_rows(4, cols, 0x4F91);
    let g = gain(cols, 0x4F92);
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let mut bufs = RefusalBufs {
        weights: device
            .upload(
                stream,
                &padded(0xFFu8, weight_pad(QUANT), &weights, WEIGHT_TAIL),
            )
            .map_err(why)?,
        weights_len: weights.len(),
        x: device
            .upload(stream, &padded(sentinel, ACT_PAD, &x, 1))
            .map_err(why)?,
        gain: device
            .upload(stream, &padded(sentinel, ACT_PAD, &g, 1))
            .map_err(why)?,
        y: device
            .upload(
                stream,
                &padded(
                    sentinel,
                    OUT_GUARD,
                    &common::sentinel_rows(4 * rows),
                    OUT_GUARD,
                ),
            )
            .map_err(why)?,
        out: device
            .upload(
                stream,
                &padded(
                    sentinel,
                    ACT_PAD,
                    &common::sentinel_rows(4 * cols + 4),
                    ACT_PAD,
                ),
            )
            .map_err(why)?,
        ids: device.upload(stream, &[0i32; 8]).map_err(why)?,
        picks: device.upload(stream, &[u32::MAX; 4]).map_err(why)?,
    };
    let segs = [Segment {
        cols,
        rows,
        weight_blocks: 0,
        out_offset: 0,
    }];
    let untouched = |h: &Harness<'_>, b: &RefusalBufs| {
        common::still(h, &b.y, |v| v.to_bits() == SENTINEL_BITS)
            .map_err(|e| format!("the outputs: {e}"))
    };
    let e = h.geom().experts;
    let well_formed = 2 * cols;

    let plain = Plain {
        quant: QUANT,
        m: 2,
        rows,
        cols,
        x_words: well_formed,
        y_words: 2 * rows,
    };
    let case = |name: &'static str, needle: &'static str, p: Plain| {
        BadArg::new(
            name,
            needle,
            Box::new(move |b: &mut RefusalBufs, enc| h.kernels.matvec_m(enc, p.params(b)?)),
        )
    };
    let matvec_cases: Vec<BadArg<'_, RefusalBufs>> = vec![
        case(
            "M of 1",
            "M is 1",
            Plain {
                m: 1,
                x_words: cols,
                y_words: rows,
                ..plain
            },
        ),
        case(
            "M of 3",
            "M is 3",
            Plain {
                m: 3,
                x_words: 3 * cols,
                y_words: 3 * rows,
                ..plain
            },
        ),
        case("M of 5", "M is 5", Plain { m: 5, ..plain }),
        case(
            "the A8 family",
            "no M-row entry",
            Plain {
                quant: Quant::A8,
                ..plain
            },
        ),
        case(
            "a short x window",
            "the x window",
            Plain {
                x_words: well_formed - 1,
                ..plain
            },
        ),
        case(
            "a short y window",
            "the y window",
            Plain {
                y_words: 2 * rows - 1,
                ..plain
            },
        ),
    ];
    let count = common::refuse_launches(h, "matvec mrows", &mut bufs, matvec_cases, untouched)?;

    // One scale word for two rows: the scaled input holds a word a row.
    let seg_case = BadArg::new(
        "a scale word for two rows",
        "the scale window",
        Box::new(|b: &mut RefusalBufs, enc| {
            h.kernels.matvec_seg_m(
                enc,
                SegMParams {
                    quant: QUANT,
                    m: 2,
                    segments: &segs,
                    weights: tensor(&b.weights, b.weights_len)?,
                    x: b.x.window(Elems(ACT_PAD), Elems(well_formed))?,
                    y: b.y.window_mut(Elems(OUT_GUARD), Elems(2 * rows))?,
                    norm: LoadNorm::Scaled {
                        scale: b.gain.window(Elems(ACT_PAD), Elems(1))?,
                    },
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        }),
    );
    let count = count
        + common::refuse_launches(h, "matvec seg mrows", &mut bufs, vec![seg_case], untouched)?;

    let pair_case = BadArg::new(
        "one slot",
        "slots",
        Box::new(|b: &mut RefusalBufs, enc| {
            h.kernels.pair_m(
                enc,
                PairMParams {
                    quant: QUANT,
                    shared_quant: QUANT,
                    m: 2,
                    slots: 1,
                    k: e.k,
                    cols,
                    n_experts: 1,
                    ids: b.ids.window(Elems(0), Elems(1))?,
                    gate: b.weights.window(Elems(0), Elems(0))?,
                    up: b.weights.window(Elems(0), Elems(0))?,
                    shared_gate: None,
                    shared_up: None,
                    x: b.x.window(Elems(ACT_PAD), Elems(well_formed))?,
                    scale: b.gain.window(Elems(ACT_PAD), Elems(2))?,
                    y: b.y.window_mut(Elems(OUT_GUARD), Elems(2 * rows))?,
                    out_stride: e.k,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        }),
    );
    let count =
        count + common::refuse_launches(h, "pair mrows", &mut bufs, vec![pair_case], untouched)?;

    // The head's own cases: logits rows the argmax cannot read as quads, and
    // a picks window shorter than the M rows. The weights are the small
    // tensor's, too short for the vocabulary the head names, so the cases
    // whose refusal is a later check pass a row count the weights refuse
    // first — the two cases here name their own needle before it.
    let head_case = BadArg::new(
        "quads plus two",
        "float4 quads",
        Box::new(|b: &mut RefusalBufs, enc| {
            h.kernels.head_m(
                enc,
                HeadMParams {
                    quant: QUANT,
                    m: 2,
                    rows: rows + 2,
                    k: cols,
                    simdgroups: HEAD_SIMD_GROUPS,
                    weights: tensor(&b.weights, b.weights_len)?,
                    x: b.x.window(Elems(ACT_PAD), Elems(well_formed))?,
                    scale: b.gain.window(Elems(ACT_PAD), Elems(2))?,
                    logits: b.y.window_mut(Elems(OUT_GUARD), Elems(2 * (rows + 2)))?,
                    step_ids: b.picks.window_mut(Elems(0), Elems(2))?,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        }),
    );
    let count =
        count + common::refuse_launches(h, "head mrows", &mut bufs, vec![head_case], untouched)?;

    // The hand-on's own cases, over its own output buffer.
    let next_launch = |b: &mut RefusalBufs,
                       enc: &mut bloomery_metal::encode::ComputeEncoder<'_>,
                       quant: Quant,
                       m: usize,
                       cols: usize,
                       out_words: usize,
                       eps: f32| {
        h.kernels.norm_next_m(
            enc,
            NormNextMParams {
                quant,
                m,
                x: b.x.window(Elems(ACT_PAD), Elems(m * cols))?,
                gain: b.gain.window(Elems(ACT_PAD), Elems(cols))?,
                eps,
                layer: LAYER,
                out: b.out.window_mut(Elems(ACT_PAD), Elems(out_words))?,
                fault: h.device.fault_word(),
            },
        )
    };
    let untouched_out = |h: &Harness<'_>, b: &RefusalBufs| {
        common::still(h, &b.out, |v| v.to_bits() == SENTINEL_BITS)
            .map_err(|e| format!("the outputs: {e}"))
    };
    let next_cases: Vec<BadArg<'_, RefusalBufs>> = vec![
        BadArg::new(
            "M of 1",
            "M is 1",
            Box::new(|b: &mut RefusalBufs, enc| next_launch(b, enc, QUANT, 1, cols, cols + 1, EPS)),
        ),
        BadArg::new(
            "the A8 family",
            "no M-row entry",
            Box::new(|b: &mut RefusalBufs, enc| {
                next_launch(b, enc, Quant::A8, 2, cols, 2 * cols + 2, EPS)
            }),
        ),
        BadArg::new(
            "columns that are not whole blocks",
            "not whole blocks",
            Box::new(|b: &mut RefusalBufs, enc| {
                next_launch(b, enc, QUANT, 2, cols - 4, 2 * (cols - 4) + 2, EPS)
            }),
        ),
        BadArg::new(
            "a short out window",
            "the out window",
            Box::new(|b: &mut RefusalBufs, enc| {
                next_launch(b, enc, QUANT, 2, cols, 2 * cols + 1, EPS)
            }),
        ),
        BadArg::new(
            "an epsilon that is not finite",
            "is not finite",
            Box::new(|b: &mut RefusalBufs, enc| {
                next_launch(b, enc, QUANT, 2, cols, 2 * cols + 2, f32::NAN)
            }),
        ),
    ];
    let count = count
        + common::refuse_launches(h, "norm next mrows", &mut bufs, next_cases, untouched_out)?;

    // The launch adds into its outputs, so the well-formed one runs over
    // finite ones (a NaN sentinel stays a NaN), and a word that is not zero is
    // its write.
    bufs.y = device
        .upload(
            stream,
            &padded(0.0f32, OUT_GUARD, &vec![0.0f32; 4 * rows], OUT_GUARD),
        )
        .map_err(why)?;
    common::accepted_writes(
        h,
        &mut bufs,
        Box::new(move |b: &mut RefusalBufs, enc| h.kernels.matvec_m(enc, plain.params(b)?)),
        |h, b| common::moved(h, &b.y, |v| *v != 0.0),
    )?;
    common::accepted_writes(
        h,
        &mut bufs,
        Box::new(|b: &mut RefusalBufs, enc| next_launch(b, enc, QUANT, 2, cols, 2 * cols + 2, EPS)),
        |h, b| common::moved(h, &b.out, |v| v.to_bits() != SENTINEL_BITS),
    )?;
    Ok(Verdict::Noted(format!(
        "{count} refusals by name, nothing written; the well-formed launches write"
    )))
}

/// The M-row clauses in the order the gate runs them: the launches in the
/// step's order, the hand-on and the scaled readers, then the refusals.
pub fn table<'a>(h: &'a Harness<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        (
            "mrows_matvec",
            Box::new(move || matvec(h)) as super::ClauseRun<'a>,
        ),
        ("mrows_seg", Box::new(move || seg(h))),
        ("mrows_next", Box::new(move || hand_on(h))),
        ("mrows_seg_scaled", Box::new(move || seg_scaled(h))),
        ("mrows_pair_scaled", Box::new(move || pair_scaled(h))),
        ("mrows_head_scaled", Box::new(move || head_scaled(h))),
        ("mrows_refusals", Box::new(move || refusals(h))),
    ]
}
