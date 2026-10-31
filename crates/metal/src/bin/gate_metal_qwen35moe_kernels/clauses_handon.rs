//! The sandwich join that hands on, and the readers that take its input.
//!
//! A one-row join whose next launch norms the stream on load
//! ([`Kernels::norm_add_next`](bloomery_metal::kernels::Kernels::norm_add_next))
//! writes the stream as `norm_add_rows` does and, for the next launch, the
//! new row times that launch's gain, then the norm's scale summed in the next
//! launch's own lane order. A reader given that input ([`LoadNorm::Scaled`])
//! stores the bits its norm-on-load launch stores over the new row: the
//! segmented mat-vec, the gate-up pair and the head's logits, of every type
//! with an entry. A plain residual's hand-on
//! ([`Kernels::norm_next`](bloomery_metal::kernels::Kernels::norm_next))
//! writes the same input from the new stream as it stands, the join's bits.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems, Window};
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::kernels::{
    HEAD_SIMD_GROUPS, HeadParams, LoadNorm, NormAddNextParams, NormAddRowsParams, NormNextParams,
    NormRowsParams, PairParams, Quant, SegParams, Segment,
};
use bloomery_metal::{MetalBuffer, MetalDevice};

use super::clauses_experts::PairWeights;
use super::clauses_rows::same_bits;
use super::common::{self, BadArg, Edit, LAYER};
use super::oracle;
use super::synthetic;
use super::{ClauseRun, Harness, SENTINEL_BITS};

/// The stream's width: the hidden width of every registered row.
const K: usize = 2048;

/// The types a join hands on to: each has a scaled segmented mat-vec, a
/// scaled pair and a `norm_add_next`; all but Q4_K a scaled head.
const TYPES: [Quant; 8] = [
    Quant::Q4K,
    Quant::Q6K,
    Quant::A4,
    Quant::A8,
    Quant::A4H,
    Quant::A8H,
    Quant::Q8_0,
    Quant::Q5K,
];

/// The rows of the segmented mat-vec and of the head.
const ROWS: usize = 96;

/// The experts in the pair's stacks: the routed ids alternate between them.
const N_EXPERTS: usize = 2;

/// One join: the half's output, its gain, the stream before, the add scale
/// and the epsilon.
#[derive(Clone, Copy)]
struct Join<'a> {
    out: &'a [f32],
    gain: &'a [f32],
    stream: &'a [f32],
    add: f32,
    eps: f32,
}

/// The launch after the join: its type, its norm's gain and epsilon.
#[derive(Clone, Copy)]
struct Next<'a> {
    quant: Quant,
    gain: &'a [f32],
    eps: f32,
}

/// The join over `j`, and with `next` the input it hands on. Returns the new
/// stream and, with `next`, the hand-on buffer: `K` values and the scale. The
/// hand-on buffer is one word longer than the launch's window, and that word
/// is checked as it was.
fn join(
    h: &Harness<'_>,
    j: &Join<'_>,
    next: Option<&Next<'_>>,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (device, stream) = (h.device, h.stream);
    let pad = f32::from_bits(SENTINEL_BITS);
    let out_b = device.upload(stream, j.out).map_err(why)?;
    let gain_b = device.upload(stream, j.gain).map_err(why)?;
    let mut y_b = device.upload(stream, j.stream).map_err(why)?;
    let next_gain_b = device
        .upload(stream, next.map_or(j.gain, |n| n.gain))
        .map_err(why)?;
    let mut xn_b = device.upload(stream, &vec![pad; K + 2]).map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        let joined = NormAddRowsParams {
            norm: NormRowsParams {
                rows: 1,
                cols: K,
                x: out_b.window(Elems(0), Elems(K))?,
                x_stride: K,
                gain: gain_b.window(Elems(0), Elems(K))?,
                y: y_b.window_mut(Elems(0), Elems(K))?,
                y_stride: K,
                eps: j.eps,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
            add: j.add,
        };
        match next {
            None => h.kernels.norm_add_rows(encoder, joined),
            Some(n) => h.kernels.norm_add_next(
                encoder,
                NormAddNextParams {
                    join: joined,
                    quant: n.quant,
                    gain: next_gain_b.window(Elems(0), Elems(K))?,
                    eps: n.eps,
                    layer: LAYER,
                    out: xn_b.window_mut(Elems(0), Elems(K + 1))?,
                },
            ),
        }
    })?;
    let mut y = vec![0f32; K];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    let mut xn = vec![0f32; K + 2];
    device
        .read(stream, xn_b.whole().map_err(why)?, &mut xn)
        .map_err(why)?;
    if xn[K + 1].to_bits() != SENTINEL_BITS {
        return Err("the join wrote past the hand-on buffer's scale word".to_string());
    }
    xn.truncate(K + 1);
    Ok((y, xn))
}

/// The plain residual's hand-on over `stream` for `next`: the hand-on buffer,
/// `K` values and the scale, checked one word past it as [`join`] checks.
fn plain(h: &Harness<'_>, stream: &[f32], next: &Next<'_>) -> Result<Vec<f32>, String> {
    let (device, stream_q) = (h.device, h.stream);
    let pad = f32::from_bits(SENTINEL_BITS);
    let x_b = device.upload(stream_q, stream).map_err(why)?;
    let gain_b = device.upload(stream_q, next.gain).map_err(why)?;
    let mut xn_b = device.upload(stream_q, &vec![pad; K + 2]).map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.norm_next(
            encoder,
            NormNextParams {
                quant: next.quant,
                x: x_b.window(Elems(0), Elems(K))?,
                gain: gain_b.window(Elems(0), Elems(K))?,
                eps: next.eps,
                layer: LAYER,
                out: xn_b.window_mut(Elems(0), Elems(K + 1))?,
                fault: h.device.fault_word(),
            },
        )
    })?;
    let mut xn = vec![0f32; K + 2];
    device
        .read(stream_q, xn_b.whole().map_err(why)?, &mut xn)
        .map_err(why)?;
    if xn[K + 1].to_bits() != SENTINEL_BITS {
        return Err("the plain hand-on wrote past the hand-on buffer's scale word".to_string());
    }
    xn.truncate(K + 1);
    Ok(xn)
}

/// What a reader loads: the new stream with the norm on load, or the hand-on
/// buffer scaled.
#[derive(Clone, Copy)]
enum Input<'a> {
    Gain {
        x: &'a [f32],
        gain: &'a [f32],
        eps: f32,
    },
    Scaled {
        xn: &'a [f32],
    },
}

/// The reader's input buffers: the values (and, scaled, the scale word after
/// them) and the gain.
struct Inputs {
    x: MetalBuffer<f32>,
    gain: MetalBuffer<f32>,
}

impl Inputs {
    fn upload(h: &Harness<'_>, input: Input<'_>) -> Result<Self, String> {
        let (x, gain) = match input {
            Input::Gain { x, gain, .. } => (x, gain),
            Input::Scaled { xn } => (xn, &xn[K..]),
        };
        Ok(Self {
            x: h.device.upload(h.stream, x).map_err(why)?,
            gain: h.device.upload(h.stream, gain).map_err(why)?,
        })
    }

    /// The input window and the norm the launch takes.
    fn bind(
        &self,
        input: Input<'_>,
    ) -> Result<(Window<'_, f32, MetalDevice>, LoadNorm<'_>), DeviceError> {
        let x = self.x.window(Elems(0), Elems(K))?;
        let norm = match input {
            Input::Gain { eps, .. } => LoadNorm::Gain {
                gain: self.gain.window(Elems(0), Elems(K))?,
                eps,
            },
            Input::Scaled { .. } => LoadNorm::Scaled {
                scale: self.x.window(Elems(K), Elems(1))?,
            },
        };
        Ok((x, norm))
    }
}

/// The segmented mat-vec of `ROWS` rows of `quant` over `input`: one segment.
fn seg(h: &Harness<'_>, quant: Quant, w: &[u8], input: Input<'_>) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let w_b = device.upload(stream, w).map_err(why)?;
    let inputs = Inputs::upload(h, input)?;
    let mut y_b = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); ROWS])
        .map_err(why)?;
    let segments = [Segment {
        cols: K,
        rows: ROWS,
        weight_blocks: 0,
        out_offset: 0,
    }];
    h.encode(|encoder| {
        let (x, norm) = inputs.bind(input)?;
        h.kernels.matvec_seg(
            encoder,
            SegParams {
                quant,
                segments: &segments,
                weights: w_b.window(Elems(0), Elems(w.len()))?,
                x,
                y: y_b.window_mut(Elems(0), Elems(ROWS))?,
                norm,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut y = vec![0f32; ROWS];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    Ok(y)
}

/// The pair's weights: stacks of [`N_EXPERTS`] experts and, where the
/// instance runs one, a shared expert, all of `quant`.
impl PairWeights {
    fn of(h: &Harness<'_>, quant: Quant, seed: u64) -> Self {
        let e = h.geom().experts;
        Self {
            gate: synthetic::matrix(quant, N_EXPERTS * e.k, K, seed),
            up: synthetic::matrix(quant, N_EXPERTS * e.k, K, seed + 1),
            shared_gate: synthetic::matrix(quant, e.k, K, seed + 2),
            shared_up: synthetic::matrix(quant, e.k, K, seed + 3),
        }
    }
}

/// The gate-up pair of `quant` over `input`: every slot's rows.
fn pair(
    h: &Harness<'_>,
    quant: Quant,
    w: &PairWeights,
    input: Input<'_>,
) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let e = h.geom().experts;
    let y_len = e.slots * e.k;
    let ids: Vec<i32> = (0..e.routed_ids).map(|i| (i % N_EXPERTS) as i32).collect();
    let gate_b = device.upload(stream, &w.gate).map_err(why)?;
    let up_b = device.upload(stream, &w.up).map_err(why)?;
    let shared_gate_b = device.upload(stream, &w.shared_gate).map_err(why)?;
    let shared_up_b = device.upload(stream, &w.shared_up).map_err(why)?;
    let ids_b = device.upload(stream, &ids).map_err(why)?;
    let inputs = Inputs::upload(h, input)?;
    let mut y_b = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); y_len])
        .map_err(why)?;
    h.encode(|encoder| {
        let (x, norm) = inputs.bind(input)?;
        h.kernels.pair(
            encoder,
            PairParams {
                quant,
                slots: e.slots,
                k: e.k,
                cols: K,
                n_experts: N_EXPERTS,
                ids: ids_b.window(Elems(0), Elems(ids.len()))?,
                gate: gate_b.window(Elems(0), Elems(w.gate.len()))?,
                up: up_b.window(Elems(0), Elems(w.up.len()))?,
                shared_quant: quant,
                shared_gate: e
                    .has_shared
                    .then(|| shared_gate_b.window(Elems(0), Elems(w.shared_gate.len())))
                    .transpose()?,
                shared_up: e
                    .has_shared
                    .then(|| shared_up_b.window(Elems(0), Elems(w.shared_up.len())))
                    .transpose()?,
                x,
                norm,
                y: y_b.window_mut(Elems(0), Elems(y_len))?,
                out_stride: e.k,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut y = vec![0f32; y_len];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    Ok(y)
}

/// The head of `ROWS` rows of `quant` over `input`: the logits and the pick.
fn head(
    h: &Harness<'_>,
    quant: Quant,
    w: &[u8],
    input: Input<'_>,
) -> Result<(Vec<f32>, u32), String> {
    let (device, stream) = (h.device, h.stream);
    let w_b = device.upload(stream, w).map_err(why)?;
    let inputs = Inputs::upload(h, input)?;
    let mut logits_b = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); ROWS])
        .map_err(why)?;
    let mut id_b = device.upload(stream, &[u32::MAX]).map_err(why)?;
    h.encode(|encoder| {
        let (x, norm) = inputs.bind(input)?;
        h.kernels.head(
            encoder,
            HeadParams {
                quant,
                rows: ROWS,
                k: K,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: w_b.window(Elems(0), Elems(w.len()))?,
                x,
                norm,
                logits: logits_b.window_mut(Elems(0), Elems(ROWS))?,
                step_id: id_b.window_mut(Elems(0), Elems(1))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut logits = vec![0f32; ROWS];
    let mut id = [0u32];
    device
        .read(stream, logits_b.whole().map_err(why)?, &mut logits)
        .map_err(why)?;
    device
        .read(stream, id_b.whole().map_err(why)?, &mut id)
        .map_err(why)?;
    Ok((logits, id[0]))
}

/// A gain near 1, from `seed`.
fn gain(seed: u64) -> Vec<f32> {
    let mut rng = oracle::Rng::new(seed);
    (0..K).map(|_| 1.0 + 0.5 * rng.signed()).collect()
}

/// The join that hands on writes the stream's bits of `norm_add_rows` and,
/// for its reader, each new value times the reader's gain with one rounding;
/// every reader given that input stores the bits of its norm-on-load launch
/// over the new stream (the scale summed in the reader's own lane order):
/// the segmented mat-vec and the pair of each type, the head of each type with
/// an entry. The plain residual's hand-on over the new stream writes the
/// join's hand-on bits, values and scale. Two joins: LFM3's add 1 at eps 1e-5,
/// and a negative add. Mutants: `handon_xg_no_gain` (the hand-on value is the
/// stream's, without the gain), `handon_plain_no_gain` (the plain hand-on's
/// is), `handon_scale_ulp` (the hand-on scale, both hand-ons' one walk, is
/// one rounding off the reader's), `handon_seg_scale_ignored` (the scaled
/// mat-vec rows skip the scale), `handon_pair_scale_ignored` (the scaled pair
/// skips it).
pub fn handon_readers(h: &Harness<'_>) -> Clause {
    let post_gain = gain(0x4A01);
    let next_gain = gain(0x4A02);
    let mut parts = Vec::new();
    for (case, (add, eps)) in [(1.0f32, 1.0e-5f32), (-0.75, 1.0e-6)]
        .into_iter()
        .enumerate()
    {
        let seed = 0x4A10 + 0x100 * case as u64;
        let out = oracle::activations(K, seed, 3.7);
        let before = oracle::activations(K, seed + 1, 1.0);
        let j = Join {
            out: &out,
            gain: &post_gain,
            stream: &before,
            add,
            eps,
        };
        let (y, _) = join(h, &j, None)?;
        common::expect_clean(h, "the join")?;
        for (t, quant) in TYPES.into_iter().enumerate() {
            let name = format!("add {add}, {}", quant.name());
            let next = Next {
                quant,
                gain: &next_gain,
                eps: 1.0e-5,
            };
            let (y_next, xn) = join(h, &j, Some(&next))?;
            common::expect_clean(h, &format!("{name}: the join that hands on"))?;
            same_bits(&format!("{name}: the stream"), &y_next, &y)?;
            let xg: Vec<f32> = y.iter().zip(&next_gain).map(|(v, g)| v * g).collect();
            same_bits(&format!("{name}: the hand-on values"), &xn[..K], &xg)?;
            let xn_plain = plain(h, &y, &next)?;
            common::expect_clean(h, &format!("{name}: the plain hand-on"))?;
            same_bits(&format!("{name}: the plain hand-on"), &xn_plain, &xn)?;
            let scale = xn[K];
            if !(scale.is_finite() && scale > 0.0) {
                return Err(format!("{name}: the hand-on scale is {scale:e}"));
            }
            let normed = Input::Gain {
                x: &y,
                gain: &next_gain,
                eps: next.eps,
            };
            let scaled = Input::Scaled { xn: &xn };
            let w_seed = seed + 0x10 * (t as u64 + 1);

            let w = synthetic::matrix(quant, ROWS, K, w_seed);
            let want = seg(h, quant, &w, normed)?;
            common::expect_clean(h, &format!("{name}: the mat-vec on load"))?;
            let got = seg(h, quant, &w, scaled)?;
            common::expect_clean(h, &format!("{name}: the scaled mat-vec"))?;
            same_bits(&format!("{name}: the scaled mat-vec"), &got, &want)?;

            let pw = PairWeights::of(h, quant, w_seed + 4);
            let want = pair(h, quant, &pw, normed)?;
            common::expect_clean(h, &format!("{name}: the pair on load"))?;
            let got = pair(h, quant, &pw, scaled)?;
            common::expect_clean(h, &format!("{name}: the scaled pair"))?;
            same_bits(&format!("{name}: the scaled pair"), &got, &want)?;

            if quant != Quant::Q4K {
                let want = head(h, quant, &w, normed)?;
                common::expect_clean(h, &format!("{name}: the head on load"))?;
                let got = head(h, quant, &w, scaled)?;
                common::expect_clean(h, &format!("{name}: the scaled head"))?;
                same_bits(&format!("{name}: the scaled logits"), &got.0, &want.0)?;
                if got.1 != want.1 {
                    return Err(format!(
                        "{name}: the scaled head picks {}, not {}",
                        got.1, want.1
                    ));
                }
            }
        }
        parts.push(format!("add {add}"));
    }
    Ok(Verdict::Noted(format!(
        "the stream's bits, the values times the gain, the plain hand-on's bits equal to the join's, and the scaled readers' bits (mat-vec and pair of five types, head of four) at {}",
        parts.join(", ")
    )))
}

/// A join the norm cannot take is NaN in the stream, the hand-on values and
/// the scale; a stream the next norm cannot take (a value whose square
/// overflows) keeps finite stream and hand-on values and hands on a NaN scale. Each raises
/// the norm site, and the scaled reader's rows are NaN; the plain hand-on over
/// that stream hands on the same values and a NaN scale and raises. Mutants:
/// `handon_nan_scale_finite` (the unusable next norm hands on its zero scale,
/// in both hand-ons' one walk),
/// `handon_nan_xg_stays` (a join the norm cannot take leaves the hand-on
/// values as they were).
pub fn handon_nan(h: &Harness<'_>) -> Clause {
    let post_gain = gain(0x4B01);
    let next_gain = gain(0x4B02);
    let out = oracle::activations(K, 0x4B10, 3.7);
    let before = oracle::activations(K, 0x4B11, 1.0);
    let quant = Quant::A4;
    let w = synthetic::matrix(quant, ROWS, K, 0x4B20);
    let j = Join {
        out: &out,
        gain: &post_gain,
        stream: &before,
        add: 1.0,
        eps: 1.0e-5,
    };
    let next = Next {
        quant,
        gain: &next_gain,
        eps: 1.0e-5,
    };
    let (clean, clean_xn) = join(h, &j, Some(&next))?;
    common::expect_clean(h, "the clean join")?;

    let mut bad = out.clone();
    bad[700] = f32::NAN;
    let (y, xn) = join(h, &Join { out: &bad, ..j }, Some(&next))?;
    common::expect_fault(h, "a NaN in the half's output", FaultSite::NormQuant)?;
    if !y.iter().all(|v| v.is_nan()) {
        return Err("a NaN in the half's output left a stream value that is not NaN".to_string());
    }
    // The buffer starts as the sentinel, a NaN of its own: each word is the
    // join's NaN, not the word it held.
    if !xn
        .iter()
        .all(|v| v.is_nan() && v.to_bits() != SENTINEL_BITS)
    {
        return Err(
            "a NaN in the half's output left a hand-on value or the scale not written NaN"
                .to_string(),
        );
    }
    let rows = seg(h, quant, &w, Input::Scaled { xn: &xn })?;
    common::expect_fault(h, "a scaled mat-vec over a NaN scale", FaultSite::NormQuant)?;
    if !rows.iter().all(|v| v.is_nan()) {
        return Err("a scaled mat-vec over a NaN scale stored a row that is not NaN".to_string());
    }

    // One stream value whose square overflows f32: the join is finite, and
    // the next norm's sum of squares is not.
    let mut huge = before.clone();
    huge[300] = 3.0e20;
    let (y, xn) = join(h, &Join { stream: &huge, ..j }, Some(&next))?;
    common::expect_fault(
        h,
        "a stream the next norm cannot take",
        FaultSite::NormQuant,
    )?;
    // Only the overflowing value moves: every other stream and hand-on word is
    // the clean join's.
    let others = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .enumerate()
            .filter(|(i, _)| *i != 300)
            .map(|(_, x)| *x)
            .collect()
    };
    if !y[300].is_finite() || !xn[300].is_finite() {
        return Err(
            "a stream the next norm cannot take stored a value that is not finite".to_string(),
        );
    }
    same_bits(
        "a stream the next norm cannot take: the stream",
        &others(&y),
        &others(&clean),
    )?;
    same_bits(
        "a stream the next norm cannot take: the hand-on values",
        &others(&xn[..K]),
        &others(&clean_xn[..K]),
    )?;
    if !xn[K].is_nan() {
        return Err(format!(
            "a stream the next norm cannot take handed on the scale {:e}, not NaN",
            xn[K]
        ));
    }
    let xn_plain = plain(h, &y, &next)?;
    common::expect_fault(
        h,
        "a stream the plain hand-on's norm cannot take",
        FaultSite::NormQuant,
    )?;
    same_bits(
        "a stream the next norm cannot take: the plain hand-on values",
        &xn_plain[..K],
        &xn[..K],
    )?;
    if !xn_plain[K].is_nan() {
        return Err(format!(
            "a stream the plain hand-on's norm cannot take handed on the scale {:e}, not NaN",
            xn_plain[K]
        ));
    }
    let rows = seg(h, quant, &w, Input::Scaled { xn: &xn })?;
    common::expect_fault(h, "a scaled mat-vec over a NaN scale", FaultSite::NormQuant)?;
    if !rows.iter().all(|v| v.is_nan()) {
        return Err("a scaled mat-vec over a NaN scale stored a row that is not NaN".to_string());
    }
    Ok(Verdict::Noted(
        "a NaN half is NaN in the stream, the values and the scale; an unusable next norm hands on a NaN scale; each raises the norm site and the scaled rows are NaN"
            .to_string(),
    ))
}

/// The buffers of the refusal table.
struct Refusals {
    x: MetalBuffer<f32>,
    gain: MetalBuffer<f32>,
    y: MetalBuffer<f32>,
    xn: MetalBuffer<f32>,
}

/// One launch of the refusal table: the well-formed one with at most one
/// argument changed.
#[derive(Clone, Copy)]
struct NextCase {
    quant: Quant,
    rows: usize,
    out_len: usize,
    eps: f32,
}

const GOOD: NextCase = NextCase {
    quant: Quant::A4,
    rows: 1,
    out_len: K + 1,
    eps: 1.0e-5,
};

/// The join that hands on over `b`, as `c` gives it.
fn refuse_next(
    h: &Harness<'_>,
    b: &mut Refusals,
    encoder: &mut ComputeEncoder<'_>,
    c: NextCase,
) -> Result<(), DeviceError> {
    let p = NormAddNextParams {
        join: NormAddRowsParams {
            norm: NormRowsParams {
                rows: c.rows,
                cols: K,
                x: b.x.window(Elems(0), Elems(2 * K))?,
                x_stride: K,
                gain: b.gain.window(Elems(0), Elems(K))?,
                y: b.y.window_mut(Elems(0), Elems(2 * K))?,
                y_stride: K,
                eps: 1.0e-5,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
            add: 1.0,
        },
        quant: c.quant,
        gain: b.gain.window(Elems(0), Elems(K))?,
        eps: c.eps,
        layer: LAYER,
        out: b.xn.window_mut(Elems(0), Elems(c.out_len))?,
    };
    h.kernels.norm_add_next(encoder, p)
}

/// One launch of the plain hand-on's refusal table: the well-formed one with
/// at most one argument changed.
#[derive(Clone, Copy)]
struct PlainCase {
    quant: Quant,
    cols: usize,
    gain_len: usize,
    out_len: usize,
    eps: f32,
}

const PLAIN_GOOD: PlainCase = PlainCase {
    quant: Quant::A4,
    cols: K,
    gain_len: K,
    out_len: K + 1,
    eps: 1.0e-5,
};

/// The plain hand-on over `b`, as `c` gives it.
fn refuse_plain(
    h: &Harness<'_>,
    b: &mut Refusals,
    encoder: &mut ComputeEncoder<'_>,
    c: PlainCase,
) -> Result<(), DeviceError> {
    let p = NormNextParams {
        quant: c.quant,
        x: b.x.window(Elems(0), Elems(c.cols))?,
        gain: b.gain.window(Elems(0), Elems(c.gain_len))?,
        eps: c.eps,
        layer: LAYER,
        out: b.xn.window_mut(Elems(0), Elems(c.out_len))?,
        fault: h.device.fault_word(),
    };
    h.kernels.norm_next(encoder, p)
}

/// The join that hands on refuses by name, writing nothing: another type than
/// one with an entry, more than one row, a hand-on window without its scale
/// word, an epsilon that is not finite; the well-formed launch writes. The
/// plain residual's hand-on refuses the same way: another type, a stream that
/// is not whole blocks of the type, a gain other than the stream's length, a
/// hand-on window without its scale word, an epsilon that is not finite.
pub fn handon_refusals(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let pad = f32::from_bits(SENTINEL_BITS);
    let mut b = Refusals {
        x: device
            .upload(stream, &oracle::activations(2 * K, 0x4C01, 1.0))
            .map_err(why)?,
        gain: device.upload(stream, &gain(0x4C02)).map_err(why)?,
        y: device.upload(stream, &vec![pad; 2 * K]).map_err(why)?,
        xn: device.upload(stream, &vec![pad; K + 1]).map_err(why)?,
    };
    let case = |what, needle, c: NextCase| -> BadArg<'_, Refusals> {
        BadArg::new(what, needle, Box::new(move |b, e| refuse_next(h, b, e, c)))
    };
    let cases = vec![
        case(
            "an F32 reader",
            "f32",
            NextCase {
                quant: Quant::F32,
                ..GOOD
            },
        ),
        case("two rows", "rows", NextCase { rows: 2, ..GOOD }),
        case(
            "a hand-on window without its scale word",
            "out",
            NextCase { out_len: K, ..GOOD },
        ),
        case(
            "a next epsilon that is not finite",
            "eps",
            NextCase {
                eps: f32::NAN,
                ..GOOD
            },
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Refusals| {
        common::still(h, &b.y, |v| v.to_bits() == SENTINEL_BITS)?;
        common::still(h, &b.xn, |v| v.to_bits() == SENTINEL_BITS)
    };
    let count = common::refuse_launches(h, "norm add next", &mut b, cases, unchanged)?;
    let good: Edit<'_, Refusals> = Box::new(|b, e| refuse_next(h, b, e, GOOD));
    common::accepted_writes(h, &mut b, good, |h, b| {
        common::moved(h, &b.xn, |v| v.to_bits() != SENTINEL_BITS)
    })?;

    b.xn = device.upload(stream, &vec![pad; K + 1]).map_err(why)?;
    let plain_case = |what, needle, c: PlainCase| -> BadArg<'_, Refusals> {
        BadArg::new(what, needle, Box::new(move |b, e| refuse_plain(h, b, e, c)))
    };
    let plain_cases = vec![
        plain_case(
            "an F32 reader",
            "f32",
            PlainCase {
                quant: Quant::F32,
                ..PLAIN_GOOD
            },
        ),
        plain_case(
            "a stream that is not whole blocks",
            "blocks",
            PlainCase {
                cols: K - 32,
                ..PLAIN_GOOD
            },
        ),
        plain_case(
            "a gain shorter than the stream",
            "gain",
            PlainCase {
                gain_len: K - 64,
                ..PLAIN_GOOD
            },
        ),
        plain_case(
            "a hand-on window without its scale word",
            "out",
            PlainCase {
                out_len: K,
                ..PLAIN_GOOD
            },
        ),
        plain_case(
            "a next epsilon that is not finite",
            "eps",
            PlainCase {
                eps: f32::NAN,
                ..PLAIN_GOOD
            },
        ),
    ];
    let plain_unchanged =
        |h: &Harness<'_>, b: &Refusals| common::still(h, &b.xn, |v| v.to_bits() == SENTINEL_BITS);
    let plain_count =
        common::refuse_launches(h, "norm next", &mut b, plain_cases, plain_unchanged)?;
    let good: Edit<'_, Refusals> = Box::new(|b, e| refuse_plain(h, b, e, PLAIN_GOOD));
    common::accepted_writes(h, &mut b, good, |h, b| {
        common::moved(h, &b.xn, |v| v.to_bits() != SENTINEL_BITS)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals of the join and {plain_count} of the plain hand-on, each by name, nothing written"
    )))
}

/// The hand-on's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        (
            "handon_readers",
            Box::new(|| handon_readers(h)) as ClauseRun,
        ),
        ("handon_nan", Box::new(|| handon_nan(h))),
        ("handon_refusals", Box::new(|| handon_refusals(h))),
    ]
}
