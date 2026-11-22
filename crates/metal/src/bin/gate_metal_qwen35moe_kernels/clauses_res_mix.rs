//! The normed mix's clauses (`kernels/res_mix.metal`): the mix against its f64
//! oracle at the Mini's width over banks of every depth the family runs, the
//! row a vector that is not a number poisons alone, and the launch's
//! refusals. The kernel's rule: each vector is scored
//! `rstd(v) · dot(v, q) / √n`, the scores go through one softmax per query
//! row, and the output is the weighted sum of the *normed* vectors.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::{HIDDEN, MIX_NQ, MIX_VECS};
use bloomery_metal::kernels::ResMixParams;

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

const EPS: f32 = 1.0e-6;
const ROWS: usize = 3;
/// Garbage floats between one bank's rows and the next: a launch that reads
/// the stride as the row width reads them.
const PAD: usize = 64;

struct Case {
    banks: Vec<Vec<f32>>,
    cur: Vec<f32>,
    queries: Vec<f32>,
    nq: usize,
}

/// Bank `i`'s values are drawn at a different amplitude, so the softmax is
/// not uniform and the RMS scales differ a vector.
fn case(n_banks: usize, nq: usize, seed: u64) -> Case {
    let banks = (0..n_banks)
        .map(|i| ink2::draw(ROWS * HIDDEN, seed + i as u64, 0.4 + 0.45 * i as f32))
        .collect();
    Case {
        banks,
        cur: ink2::draw(ROWS * HIDDEN, seed + 100, 1.7),
        queries: ink2::draw(nq * HIDDEN, seed + 200, 3.0),
        nq,
    }
}

fn launch(h: &Harness<'_>, c: &Case) -> Result<Vec<f32>, String> {
    let stride = ROWS * HIDDEN + PAD;
    let mut flat = Vec::new();
    for bank in &c.banks {
        flat.extend_from_slice(bank);
        flat.extend(std::iter::repeat_n(1.0e6f32, PAD));
    }
    let n_banks = c.banks.len();
    let banks = (n_banks > 0).then(|| ink2::up(h, &flat)).transpose()?;
    let cur = ink2::up(h, &c.cur)?;
    let queries = ink2::up(h, &c.queries)?;
    let mut y = ink2::up(h, &ink2::blank(ROWS * c.nq * HIDDEN))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.res_mix(
            encoder,
            ResMixParams {
                n: HIDDEN,
                nq: c.nq,
                m: ROWS,
                n_banks,
                stride,
                banks: banks
                    .as_ref()
                    .map(|b| b.window(Elems(0), Elems(flat.len())))
                    .transpose()?,
                cur: cur.window(Elems(0), Elems(c.cur.len()))?,
                queries: queries.window(Elems(0), Elems(c.queries.len()))?,
                y: y.window_mut(Elems(0), Elems(ROWS * c.nq * HIDDEN))?,
                eps: EPS,
                scale: (1.0 / (HIDDEN as f64).sqrt()) as f32,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let out = ink2::read(h, &y)?;
    ink2::written("the mix", &out)?;
    Ok(out)
}

/// The oracle of row `t`, query `q`: the mix and its band.
fn oracle(c: &Case, t: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let n = HIDDEN;
    let vecs: Vec<&[f32]> = c
        .banks
        .iter()
        .map(|b| &b[t * n..(t + 1) * n])
        .chain(std::iter::once(&c.cur[t * n..(t + 1) * n]))
        .collect();
    let normed: Vec<Vec<f64>> = vecs
        .iter()
        .map(|v| {
            let ms = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / n as f64;
            let rstd = 1.0 / (ms + f64::from(EPS)).sqrt();
            v.iter().map(|x| f64::from(*x) * rstd).collect()
        })
        .collect();
    let qv = &c.queries[q * n..(q + 1) * n];
    let scale = 1.0 / (n as f64).sqrt();
    let mut score = Vec::new();
    let mut sabs = 0.0f64;
    for v in &normed {
        let dot: f64 = v.iter().zip(qv).map(|(a, b)| a * f64::from(*b)).sum();
        let mag: f64 = v
            .iter()
            .zip(qv)
            .map(|(a, b)| (a * f64::from(*b)).abs())
            .sum();
        score.push(dot * scale);
        sabs = sabs.max(mag * scale);
    }
    let top = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = score.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = exp.iter().sum();
    let p: Vec<f64> = exp.iter().map(|e| e / total).collect();
    let mut y = vec![0.0; n];
    let mut mag = vec![0.0; n];
    for (pv, v) in p.iter().zip(&normed) {
        for j in 0..n {
            y[j] += pv * v[j];
            mag[j] += pv * v[j].abs();
        }
    }
    // The score of a vector carries `dot_ops` roundings of its magnitude, the
    // softmax turns a score's error into a weight's by at most its own size,
    // and the weighted sum adds the vectors' own roundings.
    let g = ink2::gamma_walk(ink2::dot_ops(n) + 12);
    let band = mag
        .iter()
        .map(|m| 2.0 * g * m * (1.0 + sabs) + 8.0 * U * m)
        .collect();
    (y, band)
}

fn fit_case(h: &Harness<'_>, c: &Case) -> Result<f64, String> {
    let got = launch(h, c)?;
    common::expect_clean(h, "a mix")?;
    let mut worst = 0.0f64;
    for t in 0..ROWS {
        for q in 0..c.nq {
            let (want, band) = oracle(c, t, q);
            let at = (t * c.nq + q) * HIDDEN;
            worst = worst.max(ink2::fit(
                &format!("{} banks, nq {}, row {t}, query {q}", c.banks.len(), c.nq),
                &got[at..at + HIDDEN],
                &want,
                &band,
            )?);
        }
    }
    Ok(worst)
}

/// The mix is the oracle's, in band, over the bank counts the family runs:
/// the stream alone, a windowed visit's, a core visit's with the message
/// rows' four queries, and the most vectors the kernel takes. Each bank's rows
/// sit a stride apart with garbage between.
pub fn res_mix(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (i, (n_banks, nq)) in [(0, 1), (5, 1), (8, 3), (MIX_VECS - 1, MIX_NQ)]
        .into_iter()
        .enumerate()
    {
        let worst = fit_case(h, &case(n_banks, nq, 0x41A0 + 16 * i as u64))?;
        parts.push(format!("{n_banks} banks nq {nq} {worst:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "the normed mix in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A vector that is not a number makes its row's mixes NaN, raises the norm
/// site, and moves no other row: the clean launch's bits stand beside it.
pub fn res_mix_nan(h: &Harness<'_>) -> Clause {
    let clean = case(4, 2, 0x41C0);
    let want = launch(h, &clean)?;
    common::expect_clean(h, "the clean mix")?;
    let mut bad = case(4, 2, 0x41C0);
    bad.banks[2][HIDDEN + 700] = f32::NAN;
    let got = launch(h, &bad)?;
    common::expect_fault(h, "a NaN in bank 2's row 1", FaultSite::NormQuant)?;
    let per = bad.nq * HIDDEN;
    for t in 0..ROWS {
        let (g, w) = (&got[t * per..(t + 1) * per], &want[t * per..(t + 1) * per]);
        if t == 1 {
            if !g.iter().all(|v| v.is_nan()) {
                return Err("a NaN vector left a mix value that is not NaN".to_string());
            }
        } else {
            ink2::same(&format!("row {t} beside a NaN in row 1"), g, w)?;
        }
    }
    Ok(Verdict::Noted(
        "a NaN vector is NaN in its row's mixes alone and raises the norm site".to_string(),
    ))
}

/// The shapes and windows the launch refuses by name, nothing encoded.
pub fn res_mix_bounds(h: &Harness<'_>) -> Clause {
    let c = case(2, 1, 0x41E0);
    let n = HIDDEN;
    let stride = ROWS * n + PAD;
    let flat = ink2::up(h, &vec![0.5f32; stride * 2])?;
    let cur = ink2::up(h, &c.cur)?;
    let queries = ink2::up(h, &ink2::draw((MIX_NQ + 1) * n, 3, 1.0))?;
    let mut y = ink2::up(h, &ink2::blank(ROWS * (MIX_NQ + 1) * n))?;
    let mut run = |label: &str,
                   nq: usize,
                   n_banks: usize,
                   stride: usize,
                   banks_len: usize,
                   y_len: usize,
                   eps: f32,
                   needle: &str|
     -> Result<(), String> {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.res_mix(
                    encoder,
                    ResMixParams {
                        n,
                        nq,
                        m: ROWS,
                        n_banks,
                        stride,
                        banks: (n_banks > 0)
                            .then(|| flat.window(Elems(0), Elems(banks_len)))
                            .transpose()?,
                        cur: cur.window(Elems(0), Elems(ROWS * n))?,
                        queries: queries.window(Elems(0), Elems(nq.max(1) * n))?,
                        y: y.window_mut(Elems(0), Elems(y_len))?,
                        eps,
                        scale: 0.02,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let ok_len = stride + ROWS * n;
    run(
        "no query rows",
        0,
        2,
        stride,
        ok_len,
        ROWS * n,
        EPS,
        "query rows",
    )?;
    run(
        "five query rows",
        MIX_NQ + 1,
        2,
        stride,
        ok_len,
        ROWS * (MIX_NQ + 1) * n,
        EPS,
        "query rows",
    )?;
    run(
        "seventeen vectors",
        1,
        MIX_VECS,
        stride,
        ok_len,
        ROWS * n,
        EPS,
        "vectors",
    )?;
    run(
        "a stride under a row",
        1,
        2,
        ROWS * n - 1,
        ok_len,
        ROWS * n,
        EPS,
        "stride",
    )?;
    run(
        "a short bank window",
        1,
        2,
        stride,
        ok_len - 1,
        ROWS * n,
        EPS,
        "banks",
    )?;
    run(
        "a short output",
        1,
        2,
        stride,
        ok_len,
        ROWS * n - 1,
        EPS,
        "y window",
    )?;
    run("a zero epsilon", 1, 2, stride, ok_len, ROWS * n, 0.0, "eps")?;
    Ok(Verdict::Noted(
        "seven refusals by name, each before an encode".to_string(),
    ))
}

/// The mix's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("res_mix", Box::new(|| res_mix(h)) as ClauseRun),
        ("res_mix_nan", Box::new(|| res_mix_nan(h))),
        ("res_mix_bounds", Box::new(|| res_mix_bounds(h))),
    ]
}
