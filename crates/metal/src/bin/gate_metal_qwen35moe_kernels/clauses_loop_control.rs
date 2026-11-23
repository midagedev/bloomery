//! The loop controller's and the out gate's clauses
//! (`kernels/loop_control.metal`, `kernels/out_gate.metal`).
//!
//! The controller: `out = (1 − tanh(F·h)) · cur + tanh(G·h) · msg` with
//! `h = D · [cur · rstd(cur); msg · rstd(msg)]`, the two halves gainless-RMS
//! normed on their own. The oracle is that in f64 over seeded weights in the
//! file's own memory order; the band is the f32 order's roundings: the down
//! dots a lane's `2n / 32` fused multiply-adds and the group's tree, the gain
//! and forget dots 64 in order, and the output's two products and sum, the
//! roundings added as a random walk (`ink2::gamma_walk`).

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::{CONTROLLER_RANK, HIDDEN};
use bloomery_metal::kernels::{LoopControlParams, OutGateParams};

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

const N: usize = HIDDEN;
const RANK: usize = CONTROLLER_RANK;
const EPS: f32 = 1.0e-6;
const ROWS: usize = 3;

struct Weights {
    down: Vec<f32>,
    gain: Vec<f32>,
    forget: Vec<f32>,
}

fn weights(seed: u64) -> Weights {
    Weights {
        down: ink2::draw(RANK * 2 * N, seed, 0.03),
        gain: ink2::draw(N * RANK, seed + 1, 0.35),
        forget: ink2::draw(N * RANK, seed + 2, 0.35),
    }
}

struct Rows {
    cur: Vec<f32>,
    msg: Vec<f32>,
}

fn rows(seed: u64) -> Rows {
    Rows {
        cur: ink2::draw(ROWS * N, seed, 2.0),
        msg: ink2::draw(ROWS * N, seed + 1, 0.7),
    }
}

fn launch(h: &Harness<'_>, w: &Weights, r: &Rows) -> Result<Vec<f32>, String> {
    let (cur, msg) = (ink2::up(h, &r.cur)?, ink2::up(h, &r.msg)?);
    let (down, gain, forget) = (
        ink2::up(h, &w.down)?,
        ink2::up(h, &w.gain)?,
        ink2::up(h, &w.forget)?,
    );
    let mut y = ink2::up(h, &ink2::blank(ROWS * N))?;
    let mut down_rows = ink2::up(h, &ink2::blank(ROWS * RANK))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.loop_control(
            encoder,
            LoopControlParams {
                n: N,
                rank: RANK,
                m: ROWS,
                cur: cur.window(Elems(0), Elems(r.cur.len()))?,
                msg: msg.window(Elems(0), Elems(r.msg.len()))?,
                down: down.window(Elems(0), Elems(w.down.len()))?,
                gain: gain.window(Elems(0), Elems(w.gain.len()))?,
                forget: forget.window(Elems(0), Elems(w.forget.len()))?,
                y: y.window_mut(Elems(0), Elems(ROWS * N))?,
                h: down_rows.window_mut(Elems(0), Elems(ROWS * RANK))?,
                eps: EPS,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let out = ink2::read(h, &y)?;
    ink2::written("the controller", &out)?;
    Ok(out)
}

/// Row `t`'s output and band.
fn oracle(w: &Weights, r: &Rows, t: usize) -> (Vec<f64>, Vec<f64>) {
    let half = |v: &[f32]| -> Vec<f64> {
        let ms = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / N as f64;
        let rstd = 1.0 / (ms + f64::from(EPS)).sqrt();
        v.iter().map(|x| f64::from(*x) * rstd).collect()
    };
    let (cur, msg) = (&r.cur[t * N..(t + 1) * N], &r.msg[t * N..(t + 1) * N]);
    let normed: Vec<f64> = half(cur).into_iter().chain(half(msg)).collect();
    // h and the roundings of each of its dots.
    let g_down = ink2::gamma_walk(ink2::dot_ops(2 * N) + 4);
    let mut h = vec![0.0f64; RANK];
    let mut dh = vec![0.0f64; RANK];
    for d in 0..RANK {
        let row = &w.down[d * 2 * N..(d + 1) * 2 * N];
        h[d] = row
            .iter()
            .zip(&normed)
            .map(|(a, b)| f64::from(*a) * b)
            .sum();
        let mag: f64 = row
            .iter()
            .zip(&normed)
            .map(|(a, b)| (f64::from(*a) * b).abs())
            .sum();
        dh[d] = g_down * mag;
    }
    let g_rank = ink2::gamma_walk(RANK + 4);
    let mut out = vec![0.0f64; N];
    let mut band = vec![0.0f64; N];
    for j in 0..N {
        let (mut f, mut g) = (0.0f64, 0.0f64);
        let (mut ef, mut eg) = (0.0f64, 0.0f64);
        for d in 0..RANK {
            let (fw, gw) = (
                f64::from(w.forget[j * RANK + d]),
                f64::from(w.gain[j * RANK + d]),
            );
            f += fw * h[d];
            g += gw * h[d];
            ef += g_rank * (fw * h[d]).abs() + fw.abs() * dh[d];
            eg += g_rank * (gw * h[d]).abs() + gw.abs() * dh[d];
        }
        let (c, m) = (f64::from(cur[j]), f64::from(msg[j]));
        out[j] = (1.0 - f.tanh()) * c + g.tanh() * m;
        // tanh' <= 1: the dots' errors pass on at most as they are; the
        // library's tanh and the three products and two sums add their own.
        band[j] = c.abs() * ef + m.abs() * eg + 12.0 * U * (c.abs() + m.abs()) + 1e-30;
    }
    (out, band)
}

/// The controller is the oracle's, in band, over rows of mixed magnitudes.
pub fn loop_control(h: &Harness<'_>) -> Clause {
    let (w, r) = (weights(0x4700), rows(0x4710));
    let got = launch(h, &w, &r)?;
    common::expect_clean(h, "the controller")?;
    let mut worst = 0.0f64;
    for t in 0..ROWS {
        let (want, band) = oracle(&w, &r, t);
        worst = worst.max(ink2::fit(
            &format!("row {t}"),
            &got[t * N..(t + 1) * N],
            &want,
            &band,
        )?);
    }
    Ok(Verdict::Noted(format!(
        "the controller in the f64 oracle's band, worst error over its band: {worst:.3}"
    )))
}

/// A half whose norm is not defined is NaN in its row alone and raises the
/// norm site; a weight that is not a number is NaN in its output alone and
/// raises the product site. The other rows and values are the clean launch's
/// bits.
pub fn loop_control_nan(h: &Harness<'_>) -> Clause {
    let (w, r) = (weights(0x4720), rows(0x4730));
    let clean = launch(h, &w, &r)?;
    common::expect_clean(h, "the clean controller")?;

    let mut bad = rows(0x4730);
    bad.msg[N + 40] = f32::NAN;
    let got = launch(h, &w, &bad)?;
    common::expect_fault(h, "a NaN message value", FaultSite::NormQuant)?;
    for t in 0..ROWS {
        let (g, c) = (&got[t * N..(t + 1) * N], &clean[t * N..(t + 1) * N]);
        if t == 1 {
            if !g.iter().all(|v| v.is_nan()) {
                return Err("a NaN message left an output that is not NaN".to_string());
            }
        } else {
            ink2::same(&format!("row {t} beside a NaN message"), g, c)?;
        }
    }

    let mut bw = weights(0x4720);
    bw.gain[5 * RANK + 3] = f32::NAN;
    let got = launch(h, &bw, &r)?;
    common::expect_fault(h, "a NaN gain weight", FaultSite::F32Product)?;
    for (i, (g, c)) in got.iter().zip(&clean).enumerate() {
        if i % N == 5 {
            if !g.is_nan() {
                return Err(format!(
                    "output {i} is {g:e}, not NaN beside a NaN gain weight"
                ));
            }
        } else if g.to_bits() != c.to_bits() {
            return Err(format!("a NaN gain weight of output 5 moved output {i}"));
        }
    }
    Ok(Verdict::Noted(
        "a NaN half is NaN in its row alone and a NaN weight in its output alone, each raising its site"
            .to_string(),
    ))
}

/// Row counts past one, as the prompt pass launches them: an m = 2 and an
/// m = 512 controller (the grid's height the rows), every row's merge the
/// bits of the m = 1 launch of that row alone.
pub fn loop_control_rows(h: &Harness<'_>) -> Clause {
    let w = weights(0x4760);
    let (down, gain, forget) = (
        ink2::up(h, &w.down)?,
        ink2::up(h, &w.gain)?,
        ink2::up(h, &w.forget)?,
    );
    for (i, m) in [2usize, 512].into_iter().enumerate() {
        let seed = 0x4770 + 8 * i as u64;
        let cur = ink2::up(h, &ink2::draw(m * N, seed, 2.0))?;
        let msg = ink2::up(h, &ink2::draw(m * N, seed + 1, 0.7))?;
        let mut whole = ink2::up(h, &ink2::blank(m * N))?;
        let mut whole_h = ink2::up(h, &ink2::blank(m * RANK))?;
        let mut apart = ink2::up(h, &ink2::blank(m * N))?;
        let mut apart_h = ink2::up(h, &ink2::blank(m * RANK))?;
        h.clear_fault()?;
        h.encode(|encoder| {
            h.kernels.loop_control(
                encoder,
                LoopControlParams {
                    n: N,
                    rank: RANK,
                    m,
                    cur: cur.window(Elems(0), Elems(m * N))?,
                    msg: msg.window(Elems(0), Elems(m * N))?,
                    down: down.window(Elems(0), Elems(w.down.len()))?,
                    gain: gain.window(Elems(0), Elems(w.gain.len()))?,
                    forget: forget.window(Elems(0), Elems(w.forget.len()))?,
                    y: whole.window_mut(Elems(0), Elems(m * N))?,
                    h: whole_h.window_mut(Elems(0), Elems(m * RANK))?,
                    eps: EPS,
                    faults: h.device.fault_word(),
                    layer: LAYER,
                },
            )?;
            for t in 0..m {
                h.kernels.loop_control(
                    encoder,
                    LoopControlParams {
                        n: N,
                        rank: RANK,
                        m: 1,
                        cur: cur.window(Elems(t * N), Elems(N))?,
                        msg: msg.window(Elems(t * N), Elems(N))?,
                        down: down.window(Elems(0), Elems(w.down.len()))?,
                        gain: gain.window(Elems(0), Elems(w.gain.len()))?,
                        forget: forget.window(Elems(0), Elems(w.forget.len()))?,
                        y: apart.window_mut(Elems(t * N), Elems(N))?,
                        h: apart_h.window_mut(Elems(t * RANK), Elems(RANK))?,
                        eps: EPS,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )?;
            }
            Ok(())
        })?;
        common::expect_clean(h, "the rows' controller")?;
        let (got, want) = (ink2::read(h, &whole)?, ink2::read(h, &apart)?);
        ink2::written("the rows' controller", &got)?;
        ink2::written("the rows' own controllers", &want)?;
        for t in 0..m {
            ink2::same(
                &format!("{m} rows: row {t} against its own launch"),
                &got[t * N..(t + 1) * N],
                &want[t * N..(t + 1) * N],
            )?;
        }
    }
    Ok(Verdict::Noted(
        "every row of an m = 2 and an m = 512 controller is its own m = 1 launch's bits"
            .to_string(),
    ))
}

/// The shapes and windows the launch refuses by name: the Small's hidden is
/// the staging's named refusal.
pub fn loop_control_bounds(h: &Harness<'_>) -> Clause {
    let (w, r) = (weights(0x4740), rows(0x4750));
    let (cur, msg) = (ink2::up(h, &r.cur)?, ink2::up(h, &r.msg)?);
    let (down, gain, forget) = (
        ink2::up(h, &w.down)?,
        ink2::up(h, &w.gain)?,
        ink2::up(h, &w.forget)?,
    );
    let mut y = ink2::up(h, &ink2::blank(ROWS * N))?;
    let mut down_rows = ink2::up(h, &ink2::blank(ROWS * RANK))?;
    let mut run = |label: &str,
                   n: usize,
                   rank: usize,
                   down_len: usize,
                   eps: f32,
                   needle: &str|
     -> Result<(), String> {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.loop_control(
                    encoder,
                    LoopControlParams {
                        n,
                        rank,
                        m: ROWS,
                        cur: cur.window(Elems(0), Elems(r.cur.len()))?,
                        msg: msg.window(Elems(0), Elems(r.msg.len()))?,
                        down: down.window(Elems(0), Elems(down_len))?,
                        gain: gain.window(Elems(0), Elems(w.gain.len()))?,
                        forget: forget.window(Elems(0), Elems(w.forget.len()))?,
                        y: y.window_mut(Elems(0), Elems(ROWS * N))?,
                        h: down_rows.window_mut(Elems(0), Elems(ROWS * RANK))?,
                        eps,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    run(
        "the Small's hidden",
        4608,
        RANK,
        w.down.len(),
        EPS,
        "at most 4096",
    )?;
    run("a hidden between", 2000, RANK, w.down.len(), EPS, "stages")?;
    run("another rank", N, 32, w.down.len(), EPS, "rank")?;
    run(
        "a short down window",
        N,
        RANK,
        w.down.len() - 1,
        EPS,
        "down window",
    )?;
    run("a zero epsilon", N, RANK, w.down.len(), 0.0, "eps")?;
    Ok(Verdict::Noted(
        "five refusals by name, each before an encode".to_string(),
    ))
}

/// One gate launch.
fn gate(h: &Harness<'_>, kqv: &[f32], g: &[f32]) -> Result<Vec<f32>, String> {
    let (k, gb) = (ink2::up(h, kqv)?, ink2::up(h, g)?);
    let mut y = ink2::up(h, &ink2::blank(kqv.len()))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.out_gate(
            encoder,
            OutGateParams {
                n: kqv.len(),
                kqv: k.window(Elems(0), Elems(kqv.len()))?,
                g: gb.window(Elems(0), Elems(g.len()))?,
                y: y.window_mut(Elems(0), Elems(kqv.len()))?,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let out = ink2::read(h, &y)?;
    ink2::written("the gate", &out)?;
    Ok(out)
}

/// `y = kqv ⊙ σ(g)` is the oracle's, in band, over gate values from deep in
/// the sigmoid's tails to its middle, at a width that is no multiple of a
/// threadgroup.
pub fn out_gate(h: &Harness<'_>) -> Clause {
    let n = N + 37;
    let kqv = ink2::draw(n, 0x4780, 3.0);
    let g: Vec<f32> = ink2::draw(n, 0x4781, 12.0);
    let got = gate(h, &kqv, &g)?;
    common::expect_clean(h, "the gate")?;
    let want: Vec<f64> = kqv
        .iter()
        .zip(&g)
        .map(|(k, g)| f64::from(*k) / (1.0 + (-f64::from(*g)).exp()))
        .collect();
    let band: Vec<f64> = want.iter().map(|w| w.abs() * 16.0 * U + 1e-38).collect();
    let worst = ink2::fit("the gate", &got, &want, &band)?;
    Ok(Verdict::Noted(format!(
        "the out gate in the f64 oracle's band, worst error over its band: {worst:.3}"
    )))
}

/// A value that is not a number, or whose product leaves the range, is NaN
/// alone and raises the product site.
pub fn out_gate_nan(h: &Harness<'_>) -> Clause {
    let n = 600;
    let kqv = ink2::draw(n, 0x4790, 3.0);
    let g = ink2::draw(n, 0x4791, 4.0);
    let clean = gate(h, &kqv, &g)?;
    common::expect_clean(h, "the clean gate")?;
    let mut bad = kqv;
    bad[7] = f32::NAN;
    bad[300] = f32::INFINITY;
    let got = gate(h, &bad, &g)?;
    common::expect_fault(h, "a NaN and an infinite value", FaultSite::F32Product)?;
    for (i, (a, b)) in got.iter().zip(&clean).enumerate() {
        if i == 7 || i == 300 {
            if !a.is_nan() {
                return Err(format!("value {i} is {a:e}, not NaN beside a bad input"));
            }
        } else if a.to_bits() != b.to_bits() {
            return Err(format!("a bad value moved value {i}"));
        }
    }
    Ok(Verdict::Noted(
        "a value that is not finite is NaN alone and raises the product site".to_string(),
    ))
}

/// The controller's and the gate's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("loop_control", Box::new(|| loop_control(h)) as ClauseRun),
        ("loop_control_nan", Box::new(|| loop_control_nan(h))),
        ("loop_control_rows", Box::new(|| loop_control_rows(h))),
        ("loop_control_bounds", Box::new(|| loop_control_bounds(h))),
        ("out_gate", Box::new(|| out_gate(h))),
        ("out_gate_nan", Box::new(|| out_gate_nan(h))),
    ]
}
