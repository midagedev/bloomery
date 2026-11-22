//! The residual short conv's clauses (`kernels/conv_res.metal`): the conv
//! against its f64 oracle over calls of every shape the walk makes (a step, a
//! chunk shorter than the filter, one longer than the ring), the ring's
//! bytes after each call, a position 0 that reads no ring slot, an element
//! that is not a number, and the launch's refusals.
//!
//! The oracle is the sequence's history: token `P`'s output is
//! `(x(P) + Σ_k w[c][k] x(P − (taps−1−k)) + Σ_lag dyn[P][head][lag] x(P − lag))
//! · scale`, an input before position 0 being 0. The ring is the visit's one
//! ring of 6,656 channels, the conv's range a window of it; every other
//! channel starts as large garbage and must stay it.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::MetalBuffer;
use bloomery_metal::inkling2_geom::{CONV_CHANNELS, HEAD, HIDDEN, KV_HEADS, TAPS};
use bloomery_metal::kernels::{ConvResParams, conv_scale};

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

/// A conv of the visit: its channels, dynamic heads and base in the ring.
#[derive(Clone, Copy)]
struct Conv {
    name: &'static str,
    d: usize,
    heads: usize,
    base: usize,
}

/// Three of the visit's four convs at the Mini's widths: the k conv (dynamic
/// on the windowless layers), the attention output's static one and the
/// feed-forward block's at the last range of the ring.
const CONVS: [Conv; 3] = [
    Conv {
        name: "k dynamic",
        d: KV_HEADS * HEAD,
        heads: KV_HEADS,
        base: 0,
    },
    Conv {
        name: "o static",
        d: HIDDEN,
        heads: 0,
        base: 2 * KV_HEADS * HEAD,
    },
    Conv {
        name: "ffn static",
        d: HIDDEN,
        heads: 0,
        base: 2 * KV_HEADS * HEAD + HIDDEN,
    },
];

fn garbage(len: usize, seed: u64) -> Vec<f32> {
    ink2::draw(len, seed, 50.0)
        .iter()
        .map(|v| 100.0 + v)
        .collect()
}

/// One conv's sequence so far.
struct Run {
    conv: Conv,
    rows: usize,
    w: Vec<f32>,
    /// Every token's input, token-major.
    hist: Vec<Vec<f32>>,
    /// Every token's dynamic taps, `[heads][taps]`.
    dyn_hist: Vec<Vec<f32>>,
}

impl Run {
    fn new(conv: Conv, rows: usize, seed: u64) -> Self {
        Self {
            conv,
            rows,
            w: ink2::draw(conv.d * TAPS, seed, 0.6),
            hist: Vec::new(),
            dyn_hist: Vec::new(),
        }
    }

    fn input(&self, back: usize, t: usize, c: usize) -> f64 {
        t.checked_sub(back)
            .map_or(0.0, |p| f64::from(self.hist[p][c]))
    }

    /// Token `t`'s output and band, channel `c`.
    fn oracle(&self, t: usize, c: usize) -> (f64, f64) {
        let mut acc = 0.0f64;
        let mut mag = 0.0f64;
        for k in 0..TAPS {
            let term = f64::from(self.w[c * TAPS + k]) * self.input(TAPS - 1 - k, t, c);
            acc += term;
            mag += term.abs();
        }
        if let Some(per_head) = self.conv.d.checked_div(self.conv.heads) {
            let head = c / per_head;
            for lag in 0..TAPS {
                let term = f64::from(self.dyn_hist[t][head * TAPS + lag]) * self.input(lag, t, c);
                acc += term;
                mag += term.abs();
            }
        }
        let x = self.input(0, t, c);
        let scale = f64::from(conv_scale(TAPS, self.conv.heads > 0));
        (
            (x + acc) * scale,
            (x.abs() + mag) * scale * (ink2::gamma(2 * TAPS + 3) + 2.0 * U),
        )
    }
}

/// One call of `m` tokens at `pos` over `ring`: the output rows. The run's
/// history gains the call's tokens.
fn call(
    h: &Harness<'_>,
    run: &mut Run,
    ring: &mut MetalBuffer<f32>,
    m: usize,
    pos: usize,
    seed: u64,
    poison: Option<(usize, usize)>,
) -> Result<Vec<f32>, String> {
    let conv = run.conv;
    let mut x = ink2::draw(m * conv.d, seed, 1.3);
    if let Some((t, c)) = poison {
        x[t * conv.d + c] = f32::NAN;
    }
    let dynt = ink2::draw(m * conv.heads * TAPS, seed + 1, 0.5);
    for t in 0..m {
        run.hist.push(x[t * conv.d..(t + 1) * conv.d].to_vec());
        run.dyn_hist
            .push(dynt[t * conv.heads * TAPS..(t + 1) * conv.heads * TAPS].to_vec());
    }
    let xb = ink2::up(h, &x)?;
    let wb = ink2::up(h, &run.w)?;
    let db = (conv.heads > 0).then(|| ink2::up(h, &dynt)).transpose()?;
    let mut yb = ink2::up(h, &ink2::blank(m * conv.d))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.conv_res(
            encoder,
            ConvResParams {
                d: conv.d,
                taps: TAPS,
                heads: conv.heads,
                rows: run.rows,
                ch: CONV_CHANNELS,
                base: conv.base,
                m,
                pos: pos as u32,
                x: xb.window(Elems(0), Elems(x.len()))?,
                w: wb.window(Elems(0), Elems(run.w.len()))?,
                dyn_taps: db
                    .as_ref()
                    .map(|b| b.window(Elems(0), Elems(dynt.len())))
                    .transpose()?,
                y: yb.window_mut(Elems(0), Elems(m * conv.d))?,
                ring: ring.window_mut(Elems(0), Elems(run.rows * CONV_CHANNELS))?,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let y = ink2::read(h, &yb)?;
    ink2::written("the conv", &y)?;
    Ok(y)
}

/// The ring after the calls so far: the conv's range holds the last `rows`
/// positions' inputs in their slots, every other value is the garbage it began
/// as.
fn want_ring(run: &Run, init: &[f32]) -> Vec<f32> {
    let mut ring = init.to_vec();
    let total = run.hist.len();
    for p in total.saturating_sub(run.rows)..total {
        let at = (p % run.rows) * CONV_CHANNELS + run.conv.base;
        ring[at..at + run.conv.d].copy_from_slice(&run.hist[p]);
    }
    ring
}

/// Three calls of one sequence per conv and ring: two tokens (shorter than the
/// reach), a single step, then a call longer than the ring; each output in the
/// oracle's band, the ring after each call the oracle's bytes and every
/// channel outside the conv's range untouched. The rings are the reach's
/// minimum and a longer one.
pub fn conv_res_rows(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (ci, conv) in CONVS.into_iter().enumerate() {
        for rows in [TAPS - 1, TAPS + 3] {
            let seed = 0x4300 + (ci * 64 + rows) as u64;
            let init = garbage(rows * CONV_CHANNELS, seed);
            let mut ring = ink2::up(h, &init)?;
            let mut run = Run::new(conv, rows, seed + 1);
            let mut pos = 0usize;
            let mut worst = 0.0f64;
            for (i, m) in [2usize, 1, rows + 6].into_iter().enumerate() {
                let y = call(
                    h,
                    &mut run,
                    &mut ring,
                    m,
                    pos,
                    seed + 10 + 4 * i as u64,
                    None,
                )?;
                common::expect_clean(h, "a conv call")?;
                for t in 0..m {
                    let (mut want, mut band) = (Vec::new(), Vec::new());
                    for c in 0..conv.d {
                        let (w, b) = run.oracle(pos + t, c);
                        want.push(w);
                        band.push(b);
                    }
                    worst = worst.max(ink2::fit(
                        &format!("{} rows {rows} call {i} token {t}", conv.name),
                        &y[t * conv.d..(t + 1) * conv.d],
                        &want,
                        &band,
                    )?);
                }
                ink2::same(
                    &format!("{} rows {rows}: the ring after call {i}", conv.name),
                    &ink2::read(h, &ring)?,
                    &want_ring(&run, &init),
                )?;
                pos += m;
            }
            parts.push(format!("{} ring {rows} {worst:.3}", conv.name));
        }
    }
    Ok(Verdict::Noted(format!(
        "three calls a conv in the f64 oracle's band, each ring bit for bit, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A predecessor before the sequence's first position is 0 whatever the ring
/// holds: a call at position 0 over a garbage ring is the bits of the same
/// call over a zeroed one, and a call past the start does read the ring.
pub fn conv_res_start(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (ci, conv) in [CONVS[0], CONVS[1]].into_iter().enumerate() {
        let rows = TAPS - 1;
        let seed = 0x4380 + 16 * ci as u64;
        let loud = garbage(rows * CONV_CHANNELS, seed);
        let quiet = vec![0f32; rows * CONV_CHANNELS];
        let run_at = |ring: &[f32], pos: usize| -> Result<Vec<f32>, String> {
            let mut run = Run::new(conv, rows, seed + 1);
            let mut b = ink2::up(h, ring)?;
            call(h, &mut run, &mut b, TAPS + 1, pos, seed + 2, None)
        };
        let (a, b) = (run_at(&loud, 0)?, run_at(&quiet, 0)?);
        ink2::same(&format!("{}: a call at position 0", conv.name), &a, &b)?;
        let (c, d) = (run_at(&loud, TAPS + 5)?, run_at(&quiet, TAPS + 5)?);
        if common::bits_eq(&c, &d) {
            return Err(format!(
                "{}: a call past the start did not read the ring",
                conv.name
            ));
        }
        parts.push(conv.name.to_string());
    }
    Ok(Verdict::Noted(format!(
        "position 0 reads no ring slot and a later call does: {}",
        parts.join(", ")
    )))
}

/// An input that is not a number raises the conv site, is NaN in its channel's
/// outputs over the next `taps` tokens, and moves no other channel.
pub fn conv_res_nan(h: &Harness<'_>) -> Clause {
    let conv = CONVS[0];
    let (m, bad_t, bad_c) = (8usize, 2usize, 700usize);
    let seed = 0x43C0;
    let run_with = |poison: Option<(usize, usize)>| -> Result<Vec<f32>, String> {
        let mut run = Run::new(conv, TAPS - 1, seed + 1);
        let mut ring = ink2::up(h, &garbage(run.rows * CONV_CHANNELS, seed))?;
        call(h, &mut run, &mut ring, m, 0, seed + 10, poison)
    };
    let clean = run_with(None)?;
    common::expect_clean(h, "the clean call")?;
    let bad = run_with(Some((bad_t, bad_c)))?;
    common::expect_fault(h, "a NaN input", FaultSite::LinearConv)?;
    for t in 0..m {
        for c in 0..conv.d {
            let (g, w) = (bad[t * conv.d + c], clean[t * conv.d + c]);
            if c == bad_c && (bad_t..bad_t + TAPS).contains(&t) {
                if !g.is_nan() {
                    return Err(format!(
                        "token {t} channel {c} is {g:e}, not NaN beside a NaN input"
                    ));
                }
            } else if g.to_bits() != w.to_bits() {
                return Err(format!(
                    "a NaN at token {bad_t} channel {bad_c} moved token {t} channel {c}"
                ));
            }
        }
    }
    Ok(Verdict::Noted(
        "a NaN input is NaN in its channel's next taps outputs, raises the conv site and moves no other channel"
            .to_string(),
    ))
}

/// The shapes and windows the launch refuses by name.
pub fn conv_res_bounds(h: &Harness<'_>) -> Clause {
    let conv = CONVS[1];
    let x = ink2::up(h, &vec![0.5f32; 4 * conv.d])?;
    let w = ink2::up(h, &vec![0.5f32; conv.d * TAPS])?;
    let mut y = ink2::up(h, &ink2::blank(4 * conv.d))?;
    let mut ring = ink2::up(h, &vec![0f32; (TAPS + 2) * CONV_CHANNELS])?;
    let mut run = |label: &str,
                   taps: usize,
                   rows: usize,
                   base: usize,
                   heads: usize,
                   x_len: usize,
                   needle: &str|
     -> Result<(), String> {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.conv_res(
                    encoder,
                    ConvResParams {
                        d: conv.d,
                        taps,
                        heads,
                        rows,
                        ch: CONV_CHANNELS,
                        base,
                        m: 4,
                        pos: 0,
                        x: x.window(Elems(0), Elems(x_len))?,
                        w: w.window(Elems(0), Elems(conv.d * TAPS))?,
                        dyn_taps: None,
                        y: y.window_mut(Elems(0), Elems(4 * conv.d))?,
                        ring: ring.window_mut(Elems(0), Elems(rows * CONV_CHANNELS))?,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    run("one tap", 1, 3, conv.base, 0, 4 * conv.d, "taps")?;
    run(
        "a ring under the reach",
        TAPS,
        TAPS - 2,
        conv.base,
        0,
        4 * conv.d,
        "ring rows",
    )?;
    run(
        "a base past the ring",
        TAPS,
        TAPS,
        CONV_CHANNELS - conv.d + 1,
        0,
        4 * conv.d,
        "pass the ring",
    )?;
    run(
        "dynamic heads and no taps",
        TAPS,
        TAPS,
        conv.base,
        8,
        4 * conv.d,
        "dynamic",
    )?;
    run(
        "a short input",
        TAPS,
        TAPS,
        conv.base,
        0,
        4 * conv.d - 1,
        "x window",
    )?;
    Ok(Verdict::Noted(
        "five refusals by name, each before an encode".to_string(),
    ))
}

/// The conv's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("conv_res_rows", Box::new(|| conv_res_rows(h)) as ClauseRun),
        ("conv_res_start", Box::new(|| conv_res_start(h))),
        ("conv_res_nan", Box::new(|| conv_res_nan(h))),
        ("conv_res_bounds", Box::new(|| conv_res_bounds(h))),
    ]
}
