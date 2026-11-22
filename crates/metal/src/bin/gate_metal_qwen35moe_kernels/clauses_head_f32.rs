//! The f32 head's clauses (`kernels/head_f32.metal`): the logits against their
//! f64 oracle at the Mini's width over a kept prefix that is no multiple of a
//! threadgroup's rows, with and without the soft cap, the pick's id, an input
//! row that is not a number poisoning every logit, a weight that is not a
//! number raising the logit site, and the launch's refusals. The kernel's
//! rule: `x̂ = (x · rstd) · gain`, `logit = dot(w_row, x̂)`, capped as
//! `cap · tanh(logit / cap)`.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::HIDDEN;
use bloomery_metal::kernels::HeadF32Params;

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

const EPS: f32 = 1.0e-6;
const CAP: f32 = 30.0;
/// Rows of the kept prefix: 8 · 61 + 3, so the last threadgroup is short.
const ROWS: usize = 491;

struct Case {
    w: Vec<f32>,
    x: Vec<f32>,
    gain: Vec<f32>,
    rows: usize,
}

fn case(rows: usize, seed: u64) -> Case {
    Case {
        w: ink2::draw(rows * HIDDEN, seed, 0.12),
        x: ink2::draw(HIDDEN, seed + 1, 2.5),
        gain: ink2::draw(HIDDEN, seed + 2, 0.5)
            .iter()
            .map(|g| 1.0 + g)
            .collect(),
        rows,
    }
}

struct Out {
    logits: Vec<f32>,
    pick: u32,
}

fn launch(h: &Harness<'_>, c: &Case, cap: Option<f32>, eps: f32) -> Result<Out, String> {
    let w = ink2::up(h, &c.w)?;
    let x = ink2::up(h, &c.x)?;
    let gain = ink2::up(h, &c.gain)?;
    let mut logits = ink2::up(h, &ink2::blank(c.rows))?;
    let mut step = ink2::up(h, &[u32::MAX])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.head_f32(
            encoder,
            HeadF32Params {
                rows: c.rows,
                k: HIDDEN,
                weights: w.window(Elems(0), Elems(c.w.len()))?,
                x: x.window(Elems(0), Elems(HIDDEN))?,
                gain: gain.window(Elems(0), Elems(HIDDEN))?,
                eps,
                softcap: cap,
                logits: logits.window_mut(Elems(0), Elems(c.rows))?,
                step_id: step.window_mut(Elems(0), Elems(1))?,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    Ok(Out {
        logits: ink2::read(h, &logits)?,
        pick: ink2::read(h, &step)?[0],
    })
}

/// The oracle's logits and bands.
fn oracle(c: &Case, cap: Option<f32>) -> (Vec<f64>, Vec<f64>) {
    let n = HIDDEN;
    let ms = c.x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / n as f64;
    let rstd = 1.0 / (ms + f64::from(EPS)).sqrt();
    let xhat: Vec<f64> =
        c.x.iter()
            .zip(&c.gain)
            .map(|(x, g)| f64::from(*x) * rstd * f64::from(*g))
            .collect();
    // The input's own rounding: the sum, the scale and two products, then
    // the dot's fused sums over a lane's 72 terms and the butterfly.
    let input = 8.0 * U + ink2::gamma_walk(n / 32);
    let dot_g = ink2::gamma_walk(n / 32 + 6);
    let mut want = Vec::new();
    let mut band = Vec::new();
    for r in 0..c.rows {
        let row = &c.w[r * n..(r + 1) * n];
        let dot: f64 = row.iter().zip(&xhat).map(|(w, x)| f64::from(*w) * x).sum();
        let mag: f64 = row
            .iter()
            .zip(&xhat)
            .map(|(w, x)| (f64::from(*w) * x).abs())
            .sum();
        let err = (dot_g + input) * mag;
        match cap {
            Some(cap) => {
                let cap = f64::from(cap);
                want.push(cap * (dot / cap).tanh());
                // The cap's slope is at most 1, its own rounding a few ulps.
                band.push(err + 8.0 * U * (cap * (dot / cap).tanh()).abs());
            }
            None => {
                want.push(dot);
                band.push(err);
            }
        }
    }
    (want, band)
}

/// The host's pick over the logits: the largest, the lowest row at a tie.
fn pick_of(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The logits are the oracle's in band with and without the cap, over a
/// prefix of 491 rows, and the id is the pick of the logits the kernel
/// stored.
pub fn head_f32(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (label, cap) in [("capped", Some(CAP)), ("uncapped", None)] {
        let mut c = case(ROWS, 0x4401);
        // A row far from the others so the pick is not a coin toss.
        let x_dir: Vec<f32> = c.x.iter().zip(&c.gain).map(|(x, g)| x * g).collect();
        let target = 307;
        for (j, v) in x_dir.iter().enumerate() {
            c.w[target * HIDDEN + j] = 0.02 * v;
        }
        let out = launch(h, &c, cap, EPS)?;
        common::expect_clean(h, label)?;
        ink2::written("the logits", &out.logits)?;
        let (want, band) = oracle(&c, cap);
        let worst = ink2::fit(&format!("the {label} logits"), &out.logits, &want, &band)?;
        if out.pick != 307 || out.pick != pick_of(&out.logits) {
            return Err(format!(
                "{label}: the pick is row {}, the planted winner 307 and the stored logits' {}",
                out.pick,
                pick_of(&out.logits)
            ));
        }
        parts.push(format!("{label} {worst:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "{ROWS} logits in the f64 oracle's band and the pick the planted row, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// An input row that is not a number makes every logit NaN and raises the
/// norm site; a weight that is not a number is stored as NaN on its row
/// alone and the pick raises the logit site; the other rows keep their bits.
pub fn head_f32_nan(h: &Harness<'_>) -> Clause {
    let clean = case(ROWS, 0x4411);
    let base = launch(h, &clean, Some(CAP), EPS)?;
    common::expect_clean(h, "the clean head")?;
    let mut bad_x = case(ROWS, 0x4411);
    bad_x.x[1000] = f32::NAN;
    let out = launch(h, &bad_x, Some(CAP), EPS)?;
    match h.fault()? {
        Some(f) if f.layer == LAYER && f.sites & (1 << FaultSite::NormQuant as u32) != 0 => {}
        other => return Err(format!("a NaN input raised {other:?}, not the norm site")),
    }
    if !out.logits.iter().all(|v| v.is_nan()) {
        return Err("a NaN input left a logit that is not NaN".to_string());
    }
    let mut bad_w = case(ROWS, 0x4411);
    bad_w.w[200 * HIDDEN + 5] = f32::NAN;
    let out = launch(h, &bad_w, Some(CAP), EPS)?;
    common::expect_fault(h, "a NaN weight", FaultSite::Logit)?;
    if !out.logits[200].is_nan() {
        return Err(format!("a NaN weight left logit {:e}", out.logits[200]));
    }
    for (i, (a, b)) in out.logits.iter().zip(&base.logits).enumerate() {
        if i != 200 && a.to_bits() != b.to_bits() {
            return Err(format!("a NaN weight in row 200 moved logit {i}"));
        }
    }
    Ok(Verdict::Noted(
        "a NaN input is NaN in every logit and raises the norm site; a NaN weight is NaN on its row alone and raises the logit site".to_string(),
    ))
}

/// The launch's refusals by name, nothing encoded.
pub fn head_f32_bounds(h: &Harness<'_>) -> Clause {
    let c = case(16, 0x4421);
    let w = ink2::up(h, &c.w)?;
    let x = ink2::up(h, &c.x)?;
    let gain = ink2::up(h, &c.gain)?;
    let mut logits = ink2::up(h, &ink2::blank(16))?;
    let mut step = ink2::up(h, &[0u32])?;
    let mut run = |label: &str,
                   rows: usize,
                   k: usize,
                   eps: f32,
                   cap: Option<f32>,
                   w_len: usize,
                   l_len: usize,
                   needle: &str|
     -> Result<(), String> {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.head_f32(
                    encoder,
                    HeadF32Params {
                        rows,
                        k,
                        weights: w.window(Elems(0), Elems(w_len))?,
                        x: x.window(Elems(0), Elems(HIDDEN))?,
                        gain: gain.window(Elems(0), Elems(HIDDEN))?,
                        eps,
                        softcap: cap,
                        logits: logits.window_mut(Elems(0), Elems(l_len))?,
                        step_id: step.window_mut(Elems(0), Elems(1))?,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let full = 16 * HIDDEN;
    run("no rows", 0, HIDDEN, EPS, None, full, 16, "row count")?;
    run(
        "a narrow width",
        16,
        HIDDEN - 128,
        EPS,
        None,
        full,
        16,
        "K is",
    )?;
    run("a zero epsilon", 16, HIDDEN, 0.0, None, full, 16, "eps")?;
    run(
        "a zero cap",
        16,
        HIDDEN,
        EPS,
        Some(0.0),
        full,
        16,
        "soft cap",
    )?;
    run(
        "a NaN cap",
        16,
        HIDDEN,
        EPS,
        Some(f32::NAN),
        full,
        16,
        "soft cap",
    )?;
    run(
        "short weights",
        16,
        HIDDEN,
        EPS,
        None,
        full - 1,
        16,
        "weights",
    )?;
    run("short logits", 16, HIDDEN, EPS, None, full, 15, "logits")?;
    Ok(Verdict::Noted(
        "seven refusals by name, each before an encode".to_string(),
    ))
}

/// The head's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("head_f32", Box::new(|| head_f32(h)) as ClauseRun),
        ("head_f32_nan", Box::new(|| head_f32_nan(h))),
        ("head_f32_bounds", Box::new(|| head_f32_bounds(h))),
    ]
}
