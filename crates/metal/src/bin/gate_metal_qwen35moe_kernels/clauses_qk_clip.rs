//! The qk clip epilogue's clauses (`kernels/qk_clip.metal`): a rel-bias visit's
//! projection rows against their f64 oracle — the query rows' scale, clip and
//! `tau`, the key rows' clip, the value rows' gainless RMS, the relative rows'
//! scale, clip and `tau` — the f16 append of the key and value rows into their
//! planes at the position and nowhere else, a position past the plane, a row
//! that is not a number, a value past the half range, and the refusals.
//!
//! The clip is the reference's: `factor = clamp(threshold / (‖x‖ + 1e-6), 0, 1)`
//! over the scaled row. The kernel sums the squares in f32 (the SIMD group's
//! tree) where the reference sums in f64: the band's one named term, a
//! `gamma(13)` relative error of the row's norm that the factor passes on.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::{D_REL, HEAD, KV_HEADS, QUERY_HEADS};
use bloomery_metal::kernels::QkClipParams;
use gguf::quant::f32_to_f16_bits;

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

const QK_THRESH: f32 = 22.627_417;
const R_THRESH: f32 = 8.0;
const V_EPS: f32 = 1.0e-6;
const CTX: usize = 7;
/// What a plane holds before the launch: no value of a row.
const PLANE_FILL: u16 = 0x5555;

struct Rows {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    r: Vec<f32>,
    gq: Vec<f32>,
    gr: Vec<f32>,
}

/// Rows whose amplitudes cycle, so some clip and some do not: a query or key
/// row of amplitude 4 has norm near 27 over the threshold 22.6, one of 2 near
/// 13.
fn rows(seed: u64) -> Rows {
    let amp = [0.2f32, 0.5, 1.0, 2.0, 4.0];
    let pick = |n: usize, width: usize, seed: u64, amps: &[f32]| -> Vec<f32> {
        (0..n)
            .flat_map(|h| ink2::draw(width, seed + h as u64, amps[h % amps.len()]))
            .collect()
    };
    Rows {
        q: pick(QUERY_HEADS, HEAD, seed, &amp),
        k: pick(KV_HEADS, HEAD, seed + 100, &amp),
        v: pick(KV_HEADS, HEAD, seed + 200, &[0.1, 1.0, 7.0]),
        r: pick(QUERY_HEADS, D_REL, seed + 300, &[0.5, 1.0, 2.0, 4.0, 6.0]),
        gq: ink2::draw(HEAD, seed + 400, 0.5)
            .iter()
            .map(|g| 1.0 + g)
            .collect(),
        gr: ink2::draw(D_REL, seed + 401, 0.5)
            .iter()
            .map(|g| 1.0 + g)
            .collect(),
    }
}

struct Knobs {
    pos: u32,
    tau: f32,
    qk_thresh: f32,
}

const KNOBS: Knobs = Knobs {
    pos: 3,
    tau: 1.0,
    qk_thresh: QK_THRESH,
};

struct Out {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    r: Vec<f32>,
    ck: Vec<u16>,
    cv: Vec<u16>,
}

fn launch(h: &Harness<'_>, rows: &Rows, knobs: &Knobs) -> Result<Out, String> {
    let plane = vec![PLANE_FILL; KV_HEADS * CTX * HEAD];
    let (mut q, mut k, mut v, mut r) = (
        ink2::up(h, &rows.q)?,
        ink2::up(h, &rows.k)?,
        ink2::up(h, &rows.v)?,
        ink2::up(h, &rows.r)?,
    );
    let (gq, gr) = (ink2::up(h, &rows.gq)?, ink2::up(h, &rows.gr)?);
    let (mut ck, mut cv) = (ink2::up(h, &plane)?, ink2::up(h, &plane)?);
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.qk_clip_append(
            encoder,
            QkClipParams {
                q: q.window_mut(Elems(0), Elems(rows.q.len()))?,
                k: k.window_mut(Elems(0), Elems(rows.k.len()))?,
                v: v.window_mut(Elems(0), Elems(rows.v.len()))?,
                r: r.window_mut(Elems(0), Elems(rows.r.len()))?,
                gq: gq.window(Elems(0), Elems(HEAD))?,
                gr: gr.window(Elems(0), Elems(D_REL))?,
                cache_k: ck.window_mut(Elems(0), Elems(plane.len()))?,
                cache_v: cv.window_mut(Elems(0), Elems(plane.len()))?,
                ctx: CTX,
                pos: knobs.pos,
                v_eps: V_EPS,
                qk_thresh: knobs.qk_thresh,
                r_thresh: R_THRESH,
                tau: knobs.tau,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    Ok(Out {
        q: ink2::read(h, &q)?,
        k: ink2::read(h, &k)?,
        v: ink2::read(h, &v)?,
        r: ink2::read(h, &r)?,
        ck: ink2::read(h, &ck)?,
        cv: ink2::read(h, &cv)?,
    })
}

/// A row clipped: `x · min(1, threshold / (‖x‖ + 1e-6)) · tau`, with its band:
/// the factor carries the f32 square sum's relative error, every value the
/// roundings of its own multiplies.
fn clip(x: &[f64], threshold: f64, tau: f64) -> (Vec<f64>, Vec<f64>) {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let f = (threshold / (norm + 1e-6)).clamp(0.0, 1.0);
    let rel = if f < 1.0 {
        ink2::gamma(13) + 8.0 * U
    } else {
        4.0 * U
    };
    let out: Vec<f64> = x.iter().map(|v| v * f * tau).collect();
    let band = out.iter().map(|o| o.abs() * rel + 1e-30).collect();
    (out, band)
}

fn scaled(row: &[f32], gain: Option<&[f32]>) -> Vec<f64> {
    row.iter()
        .enumerate()
        .map(|(i, v)| f64::from(*v) * gain.map_or(1.0, |g| f64::from(g[i % g.len()])))
        .collect()
}

fn fit_rows(
    what: &str,
    got: &[f32],
    width: usize,
    input: &[f32],
    gain: Option<&[f32]>,
    threshold: f64,
    tau: f64,
) -> Result<f64, String> {
    let mut worst = 0.0f64;
    for (h, (g, x)) in got.chunks(width).zip(input.chunks(width)).enumerate() {
        let (want, band) = clip(&scaled(x, gain), threshold, tau);
        worst = worst.max(ink2::fit(&format!("{what} row {h}"), g, &want, &band)?);
    }
    Ok(worst)
}

/// The f16 words of a key or value row at `pos` of every key head in `plane`,
/// and no other word of the plane moved.
fn fit_plane(what: &str, plane: &[u16], rows: &[f32], pos: usize) -> Result<(), String> {
    for (i, word) in plane.iter().enumerate() {
        let (kv, rest) = (i / (CTX * HEAD), i % (CTX * HEAD));
        let (p, d) = (rest / HEAD, rest % HEAD);
        let want = if p == pos {
            f32_to_f16_bits(rows[kv * HEAD + d])
        } else {
            PLANE_FILL
        };
        if *word != want {
            return Err(format!(
                "{what}: head {kv} position {p} value {d} holds {word:#06x}, the append's {want:#06x}"
            ));
        }
    }
    Ok(())
}

/// The rows are the oracle's, in band, windowed (`tau` 1) and windowless (the
/// log factor), and the planes hold the f16 words of the finished key and
/// value rows at `pos`, every other position as it was.
pub fn qk_clip_rows(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (i, tau) in [1.0f32, 0.8125].into_iter().enumerate() {
        let r = rows(0x4400 + 1000 * i as u64);
        let out = launch(h, &r, &Knobs { tau, ..KNOBS })?;
        common::expect_clean(h, "the epilogue")?;
        let tau64 = f64::from(tau);
        let q = fit_rows(
            "q",
            &out.q,
            HEAD,
            &r.q,
            Some(&r.gq),
            f64::from(QK_THRESH),
            tau64,
        )?;
        let k = fit_rows("k", &out.k, HEAD, &r.k, None, f64::from(QK_THRESH), 1.0)?;
        let rr = fit_rows(
            "r",
            &out.r,
            D_REL,
            &r.r,
            Some(&r.gr),
            f64::from(R_THRESH),
            tau64,
        )?;
        // The value rows: the gainless RMS norm.
        let mut vw = 0.0f64;
        for (hd, (g, x)) in out.v.chunks(HEAD).zip(r.v.chunks(HEAD)).enumerate() {
            let ms = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / HEAD as f64;
            let rstd = 1.0 / (ms + f64::from(V_EPS)).sqrt();
            let want: Vec<f64> = x.iter().map(|v| f64::from(*v) * rstd).collect();
            let band: Vec<f64> = want
                .iter()
                .map(|w| w.abs() * (ink2::gamma(13) + 8.0 * U) + 1e-30)
                .collect();
            vw = vw.max(ink2::fit(&format!("v row {hd}"), g, &want, &band)?);
        }
        fit_plane("the key plane", &out.ck, &out.k, KNOBS.pos as usize)?;
        fit_plane("the value plane", &out.cv, &out.v, KNOBS.pos as usize)?;
        parts.push(format!("tau {tau} q {q:.3} k {k:.3} v {vw:.3} r {rr:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "the rows in the f64 oracle's band and the planes the f16 words of the rows at the position alone, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A row the clip cannot take is NaN, raises the cache-value site and moves no
/// other row; a position at or past the plane raises the position site and
/// leaves the key and value rows NaN and the planes as they were; a key value
/// past the half range raises the quant-column site.
pub fn qk_clip_faults(h: &Harness<'_>) -> Clause {
    let clean_rows = rows(0x4500);
    let clean = launch(h, &clean_rows, &KNOBS)?;
    common::expect_clean(h, "the clean epilogue")?;

    let mut bad = rows(0x4500);
    bad.q[5 * HEAD + 9] = f32::NAN;
    bad.r[11 * D_REL + 2] = f32::INFINITY;
    let out = launch(h, &bad, &KNOBS)?;
    common::expect_fault(
        h,
        "a NaN query row and an infinite relative row",
        FaultSite::CacheValue,
    )?;
    for (name, got, want, width, hit) in [
        ("q", &out.q, &clean.q, HEAD, 5usize),
        ("r", &out.r, &clean.r, D_REL, 11usize),
    ] {
        for (row, (g, w)) in got.chunks(width).zip(want.chunks(width)).enumerate() {
            if row == hit {
                if !g.iter().all(|v| v.is_nan()) {
                    return Err(format!("{name} row {row} is not NaN beside a bad value"));
                }
            } else {
                ink2::same(&format!("{name} row {row} beside a bad row"), g, w)?;
            }
        }
    }
    ink2::same("the k rows beside a bad q row", &out.k, &clean.k)?;

    // A position past the plane: nothing appended, the key and value rows NaN.
    let out = launch(
        h,
        &clean_rows,
        &Knobs {
            pos: CTX as u32,
            ..KNOBS
        },
    )?;
    common::expect_fault(h, "a position past the plane", FaultSite::CachePos)?;
    if !out.k.iter().chain(&out.v).all(|v| v.is_nan()) {
        return Err(
            "a position past the plane left a key or value value that is not NaN".to_string(),
        );
    }
    if out.ck.iter().chain(&out.cv).any(|w| *w != PLANE_FILL) {
        return Err("a position past the plane appended a word".to_string());
    }
    ink2::same("the q rows beside a bad position", &out.q, &clean.q)?;

    // A key value past the half range: the threshold lifted so the clip
    // leaves it, the word appended the half infinity and the site raised.
    let mut wide = rows(0x4500);
    for v in &mut wide.k[..HEAD] {
        *v *= 1.0e6;
    }
    let out = launch(
        h,
        &wide,
        &Knobs {
            qk_thresh: 1.0e9,
            ..KNOBS
        },
    )?;
    common::expect_fault(h, "a key value past the half range", FaultSite::QuantColumn)?;
    let at = KNOBS.pos as usize * HEAD;
    if !out.ck[at..at + HEAD].iter().any(|w| w & 0x7fff == 0x7c00) {
        return Err("a key value past the half range was appended finite".to_string());
    }
    Ok(Verdict::Noted(
        "a bad row is NaN alone, a position past the plane appends nothing, a value past the half range raises its site"
            .to_string(),
    ))
}

/// The shapes and windows the launch refuses by name.
pub fn qk_clip_bounds(h: &Harness<'_>) -> Clause {
    let r = rows(0x4600);
    let plane = vec![PLANE_FILL; KV_HEADS * CTX * HEAD];
    let run = |label: &str,
               ctx: usize,
               q_len: usize,
               plane_len: usize,
               tau: f32,
               needle: &str|
     -> Result<(), String> {
        let (mut q, mut k, mut v, mut rb) = (
            ink2::up(h, &r.q)?,
            ink2::up(h, &r.k)?,
            ink2::up(h, &r.v)?,
            ink2::up(h, &r.r)?,
        );
        let (gq, gr) = (ink2::up(h, &r.gq)?, ink2::up(h, &r.gr)?);
        let (mut ck, mut cv) = (ink2::up(h, &plane)?, ink2::up(h, &plane)?);
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.qk_clip_append(
                    encoder,
                    QkClipParams {
                        q: q.window_mut(Elems(0), Elems(q_len))?,
                        k: k.window_mut(Elems(0), Elems(r.k.len()))?,
                        v: v.window_mut(Elems(0), Elems(r.v.len()))?,
                        r: rb.window_mut(Elems(0), Elems(r.r.len()))?,
                        gq: gq.window(Elems(0), Elems(HEAD))?,
                        gr: gr.window(Elems(0), Elems(D_REL))?,
                        cache_k: ck.window_mut(Elems(0), Elems(plane_len))?,
                        cache_v: cv.window_mut(Elems(0), Elems(plane_len))?,
                        ctx,
                        pos: 0,
                        v_eps: V_EPS,
                        qk_thresh: QK_THRESH,
                        r_thresh: R_THRESH,
                        tau,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    run(
        "a plane of no positions",
        0,
        r.q.len(),
        plane.len(),
        1.0,
        "0 positions",
    )?;
    run(
        "a short query window",
        CTX,
        r.q.len() - HEAD,
        plane.len(),
        1.0,
        "q window",
    )?;
    run(
        "a short plane",
        CTX,
        r.q.len(),
        plane.len() - HEAD,
        1.0,
        "key plane",
    )?;
    run("a zero tau", CTX, r.q.len(), plane.len(), 0.0, "tau")?;
    Ok(Verdict::Noted(
        "four refusals by name, each before an encode".to_string(),
    ))
}

/// The epilogue's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("qk_clip_rows", Box::new(|| qk_clip_rows(h)) as ClauseRun),
        ("qk_clip_faults", Box::new(|| qk_clip_faults(h))),
        ("qk_clip_bounds", Box::new(|| qk_clip_bounds(h))),
    ]
}
