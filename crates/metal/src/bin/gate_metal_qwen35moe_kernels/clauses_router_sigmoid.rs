//! The biased sigmoid router's clauses (`kernels/router_situ.metal`): the
//! scores, the kept ids and the kept weights against their f64 oracle at the
//! Mini's 128 experts and width, the tie order at an equal selection value, a
//! top-k under eight, a score or input that is not a number, the ticket left
//! at zero, and the launch's refusals. The rule: `score_e = σ(dot(row_e, x))`;
//! the kept experts are the `k` largest `score_e + bias_e`, the lower id at
//! an equal value; `w_s = score_s / (Σ kept + 1e-20) · scale`.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::{EXPERTS, HIDDEN, TOP_K};
use bloomery_metal::kernels::RouterSigmoidParams;

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

const SCALE: f32 = 2.5;

struct Case {
    rows: Vec<f32>,
    x: Vec<f32>,
    bias: Vec<f32>,
}

fn case(seed: u64) -> Case {
    Case {
        rows: ink2::draw(EXPERTS * HIDDEN, seed, 0.06),
        x: ink2::draw(HIDDEN, seed + 1, 1.7),
        bias: ink2::draw(EXPERTS, seed + 2, 0.3),
    }
}

struct Out {
    scores: Vec<f32>,
    ids: Vec<u32>,
    weights: Vec<f32>,
    ticket: u32,
}

fn launch(h: &Harness<'_>, c: &Case, top_k: usize, scale: f32) -> Result<Out, String> {
    let rows = ink2::up(h, &c.rows)?;
    let x = ink2::up(h, &c.x)?;
    let bias = ink2::up(h, &c.bias)?;
    let mut partials = ink2::up(h, &ink2::blank(EXPERTS))?;
    let mut ticket = ink2::up(h, &[0u32])?;
    let mut logits = ink2::up(h, &ink2::blank(EXPERTS))?;
    let mut ids = ink2::up(h, &vec![u32::MAX; top_k])?;
    let mut weights = ink2::up(h, &ink2::blank(top_k))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.router_sigmoid(
            encoder,
            RouterSigmoidParams {
                k: HIDDEN,
                top_k,
                rows: rows.window(Elems(0), Elems(c.rows.len()))?,
                x: x.window(Elems(0), Elems(HIDDEN))?,
                bias: bias.window(Elems(0), Elems(EXPERTS))?,
                partials: partials.window_mut(Elems(0), Elems(EXPERTS))?,
                ticket: ticket.window_mut(Elems(0), Elems(1))?,
                logits: logits.window_mut(Elems(0), Elems(EXPERTS))?,
                ids: ids.window_mut(Elems(0), Elems(top_k))?,
                weights: weights.window_mut(Elems(0), Elems(top_k))?,
                scale,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    Ok(Out {
        scores: ink2::read(h, &logits)?,
        ids: ink2::read(h, &ids)?,
        weights: ink2::read(h, &weights)?,
        ticket: ink2::read(h, &ticket)?[0],
    })
}

/// The oracle's scores, their bands, and the kept ids with the smallest gap
/// between a kept selection value and the first one left out.
fn oracle(c: &Case, top_k: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>, f64) {
    let n = HIDDEN;
    let g = ink2::gamma_walk(n + 6);
    let mut score = Vec::new();
    let mut band = Vec::new();
    for e in 0..EXPERTS {
        let row = &c.rows[e * n..(e + 1) * n];
        let dot: f64 = row
            .iter()
            .zip(&c.x)
            .map(|(w, x)| f64::from(*w) * f64::from(*x))
            .sum();
        let mag: f64 = row
            .iter()
            .zip(&c.x)
            .map(|(w, x)| (f64::from(*w) * f64::from(*x)).abs())
            .sum();
        let s = 1.0 / (1.0 + (-dot).exp());
        score.push(s);
        // The sigmoid's slope is at most 1/4; its own roundings are a few ulps.
        band.push(0.25 * g * mag + 8.0 * U * s);
    }
    let sel: Vec<f64> = score
        .iter()
        .zip(&c.bias)
        .map(|(s, b)| s + f64::from(*b))
        .collect();
    let mut order: Vec<usize> = (0..EXPERTS).collect();
    order.sort_by(|a, b| sel[*b].total_cmp(&sel[*a]).then(a.cmp(b)));
    let gap = sel[order[top_k - 1]] - sel[order[top_k]];
    (score, band, order[..top_k].to_vec(), gap)
}

fn check_weights(
    out: &Out,
    score: &[f64],
    band: &[f64],
    ids: &[usize],
    scale: f32,
) -> Result<f64, String> {
    let kept: Vec<f64> = ids.iter().map(|i| score[*i]).collect();
    let div = kept.iter().sum::<f64>() + 1.0e-20;
    let want: Vec<f64> = kept.iter().map(|s| s / div * f64::from(scale)).collect();
    // A weight's error is its score's relative error twice (the score and the
    // sum) and the quotient's and product's own roundings.
    let rel = ids
        .iter()
        .map(|i| band[*i] / score[*i])
        .fold(0.0f64, f64::max);
    let wband: Vec<f64> = want
        .iter()
        .map(|w| w.abs() * (2.0 * rel + 8.0 * U))
        .collect();
    ink2::fit("the kept weights", &out.weights, &want, &wband)
}

/// Scores in band, the kept ids the oracle's in order, the weights in band,
/// and the ticket back at zero, at the top-k the family runs and at 1.
pub fn router_sigmoid(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (i, top_k) in [TOP_K, 1].into_iter().enumerate() {
        let c = case(0x4501 + 16 * i as u64);
        let out = launch(h, &c, top_k, SCALE)?;
        common::expect_clean(h, "a route")?;
        if out.ticket != 0 {
            return Err(format!("the ticket is {} after the launch", out.ticket));
        }
        let (score, band, ids, gap) = oracle(&c, top_k);
        if gap < 1.0e-4 {
            return Err(format!(
                "the drawn scene's cut is {gap:e} wide, inside the oracle's doubt: change the seed"
            ));
        }
        let worst = ink2::fit("the scores", &out.scores, &score, &band)?;
        let got: Vec<usize> = out.ids.iter().map(|i| *i as usize).collect();
        if got != ids {
            return Err(format!("top-{top_k}: ids {got:?}, the oracle's {ids:?}"));
        }
        let w = check_weights(&out, &score, &band, &ids, SCALE)?;
        parts.push(format!("top-{top_k} scores {worst:.3} weights {w:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "scores, ids and weights in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// Two experts with equal rows and equal biases tie on the selection value
/// exactly: the lower id is kept first, and a tie at the cut keeps the lower
/// id alone. Biases that steer the choice leave the weights the unbiased
/// scores': an expert kept only by its bias weighs its own score.
pub fn router_sigmoid_ties(h: &Harness<'_>) -> Clause {
    let mut c = case(0x4521);
    // Experts 90 and 33 are the same row and bias, and both far above the rest.
    let (a, b) = (33usize, 90usize);
    let row: Vec<f32> = c.rows[b * HIDDEN..(b + 1) * HIDDEN].to_vec();
    c.rows[a * HIDDEN..(a + 1) * HIDDEN].copy_from_slice(&row);
    c.bias[a] = 5.0;
    c.bias[b] = 5.0;
    let out = launch(h, &c, TOP_K, SCALE)?;
    common::expect_clean(h, "a tie scene")?;
    if out.ids[0] != a as u32 || out.ids[1] != b as u32 {
        return Err(format!(
            "equal values kept {} then {}, the lower id {a} then {b} expected",
            out.ids[0], out.ids[1]
        ));
    }
    // A tie at the cut: top-k 1 keeps the lower id.
    let one = launch(h, &c, 1, SCALE)?;
    if one.ids[0] != a as u32 {
        return Err(format!("a tie at the cut kept {}, not {a}", one.ids[0]));
    }
    // A bias moves the choice and not the weight.
    let mut d = case(0x4531);
    let (score, band, _, _) = oracle(&d, TOP_K);
    let low = (0..EXPERTS)
        .min_by(|x, y| score[*x].total_cmp(&score[*y]))
        .unwrap_or(0);
    d.bias[low] = 5.0;
    let out = launch(h, &d, TOP_K, SCALE)?;
    if out.ids[0] != low as u32 {
        return Err(format!(
            "the biased expert {low} was kept as {}, not first",
            out.ids[0]
        ));
    }
    let (score, band2, ids, _) = oracle(&d, TOP_K);
    let _ = band;
    check_weights(&out, &score, &band2, &ids, SCALE)?;
    Ok(Verdict::Noted(
        "equal values keep the lower id first, a cut tie keeps it alone, and a bias moves the choice and not the weight".to_string(),
    ))
}

/// An input that is not a number makes every score NaN, raises the router
/// site and leaves NaN weights over ids 0.. in order; the ticket is zero
/// again for the next launch.
pub fn router_sigmoid_nan(h: &Harness<'_>) -> Clause {
    let mut c = case(0x4541);
    c.x[77] = f32::NAN;
    let out = launch(h, &c, TOP_K, SCALE)?;
    common::expect_fault(h, "a NaN input", FaultSite::Router)?;
    if !out.weights.iter().all(|w| w.is_nan()) {
        return Err("a NaN input left a weight that is a number".to_string());
    }
    let want: Vec<u32> = (0..TOP_K as u32).collect();
    if out.ids != want {
        return Err(format!("a NaN input left ids {:?}, not {want:?}", out.ids));
    }
    if out.ticket != 0 {
        return Err(format!("the ticket is {} after a refused pick", out.ticket));
    }
    let mut d = case(0x4551);
    d.bias[5] = f32::NAN;
    launch(h, &d, TOP_K, SCALE)?;
    common::expect_fault(h, "a NaN bias", FaultSite::Router)?;
    Ok(Verdict::Noted(
        "a NaN input or bias raises the router site, the weights NaN over ids 0.. and the ticket zero".to_string(),
    ))
}

/// The launch's refusals by name, nothing encoded.
pub fn router_sigmoid_bounds(h: &Harness<'_>) -> Clause {
    let c = case(0x4561);
    let rows = ink2::up(h, &c.rows)?;
    let x = ink2::up(h, &c.x)?;
    let bias = ink2::up(h, &c.bias)?;
    let mut partials = ink2::up(h, &ink2::blank(EXPERTS))?;
    let mut ticket = ink2::up(h, &[0u32])?;
    let mut logits = ink2::up(h, &ink2::blank(EXPERTS))?;
    let mut ids = ink2::up(h, &[0u32; TOP_K + 1])?;
    let mut weights = ink2::up(h, &ink2::blank(TOP_K + 1))?;
    let mut run = |label: &str,
                   k: usize,
                   top_k: usize,
                   scale: f32,
                   rows_len: usize,
                   bias_len: usize,
                   ids_len: usize,
                   needle: &str|
     -> Result<(), String> {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.router_sigmoid(
                    encoder,
                    RouterSigmoidParams {
                        k,
                        top_k,
                        rows: rows.window(Elems(0), Elems(rows_len))?,
                        x: x.window(Elems(0), Elems(HIDDEN))?,
                        bias: bias.window(Elems(0), Elems(bias_len))?,
                        partials: partials.window_mut(Elems(0), Elems(EXPERTS))?,
                        ticket: ticket.window_mut(Elems(0), Elems(1))?,
                        logits: logits.window_mut(Elems(0), Elems(EXPERTS))?,
                        ids: ids.window_mut(Elems(0), Elems(ids_len))?,
                        weights: weights.window_mut(Elems(0), Elems(ids_len))?,
                        scale,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let full = EXPERTS * HIDDEN;
    run("top-k 0", HIDDEN, 0, SCALE, full, EXPERTS, 1, "top-k")?;
    run(
        "top-k past the cap",
        HIDDEN,
        TOP_K + 1,
        SCALE,
        full,
        EXPERTS,
        TOP_K + 1,
        "top-k",
    )?;
    run(
        "another width",
        HIDDEN - 128,
        TOP_K,
        SCALE,
        full,
        EXPERTS,
        TOP_K,
        "hidden",
    )?;
    run(
        "a NaN scale",
        HIDDEN,
        TOP_K,
        f32::NAN,
        full,
        EXPERTS,
        TOP_K,
        "scale",
    )?;
    run(
        "short rows",
        HIDDEN,
        TOP_K,
        SCALE,
        full - 1,
        EXPERTS,
        TOP_K,
        "rows",
    )?;
    run(
        "a short bias",
        HIDDEN,
        TOP_K,
        SCALE,
        full,
        EXPERTS - 1,
        TOP_K,
        "bias",
    )?;
    run(
        "short ids",
        HIDDEN,
        TOP_K,
        SCALE,
        full,
        EXPERTS,
        TOP_K - 1,
        "ids",
    )?;
    Ok(Verdict::Noted(
        "seven refusals by name, each before an encode".to_string(),
    ))
}

/// The router's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        (
            "router_sigmoid",
            Box::new(|| router_sigmoid(h)) as ClauseRun,
        ),
        ("router_sigmoid_ties", Box::new(|| router_sigmoid_ties(h))),
        ("router_sigmoid_nan", Box::new(|| router_sigmoid_nan(h))),
        (
            "router_sigmoid_bounds",
            Box::new(|| router_sigmoid_bounds(h)),
        ),
    ]
}
