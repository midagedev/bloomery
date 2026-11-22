//! The situ combine's clauses (`kernels/situ.metal`): the kernel against its
//! f64 oracle, a value that is not a number raising the product site on its
//! own element, a beta refused, and the launch's shape refusals. The rule is
//! `gb·tanh(g/gb) · σ(g) · vb·tanh(u/vb)`.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::HIDDEN;
use bloomery_metal::kernels::SituParams;

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

/// Values a launch takes: three tokens of the Mini's width, so the last
/// threadgroup is a short one.
const N: usize = 3 * HIDDEN;
const GATE_BETA: f32 = 4.0;
const VALUE_BETA: f32 = 25.0;

fn situ_launch(
    h: &Harness<'_>,
    gate: &[f32],
    up: &[f32],
    betas: (f32, f32),
) -> Result<Vec<f32>, String> {
    let g_b = ink2::up(h, gate)?;
    let u_b = ink2::up(h, up)?;
    let mut y = ink2::up(h, &ink2::blank(gate.len()))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.pair_situ(
            encoder,
            SituParams {
                n: gate.len(),
                gate: g_b.window(Elems(0), Elems(gate.len()))?,
                up: u_b.window(Elems(0), Elems(up.len()))?,
                y: y.window_mut(Elems(0), Elems(gate.len()))?,
                gate_beta: betas.0,
                value_beta: betas.1,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let out = ink2::read(h, &y)?;
    ink2::written("the situ", &out)?;
    Ok(out)
}

fn situ_oracle(g: f32, u: f32) -> f64 {
    let (g, u) = (f64::from(g), f64::from(u));
    let (gb, vb) = (f64::from(GATE_BETA), f64::from(VALUE_BETA));
    gb * (g / gb).tanh() / (1.0 + (-g).exp()) * (vb * (u / vb).tanh())
}

/// The combine is the oracle's, over inputs wide enough that both tanh caps
/// and the sigmoid's saturation show: the gate term's tanh, its sigmoid, the
/// up term's quotient and tanh and the product each round once.
pub fn situ(h: &Harness<'_>) -> Clause {
    let gate = ink2::draw(N, 0x4321, 12.0);
    let up = ink2::draw(N, 0x4322, 60.0);
    let got = situ_launch(h, &gate, &up, (GATE_BETA, VALUE_BETA))?;
    common::expect_clean(h, "a situ")?;
    let want: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| situ_oracle(*g, *u))
        .collect();
    let band: Vec<f64> = want.iter().map(|w| 24.0 * U * w.abs() + 1.0e-30).collect();
    let worst = ink2::fit("the situ activation", &got, &want, &band)?;
    Ok(Verdict::Noted(format!(
        "{N} activations in the f64 oracle's band, worst error over its band: {worst:.3}"
    )))
}

/// A value that is not a number is NaN on its element alone and raises the
/// product site; a beta that is not finite and positive is refused by name.
pub fn situ_nan(h: &Harness<'_>) -> Clause {
    let gate = ink2::draw(N, 0x4331, 8.0);
    let up = ink2::draw(N, 0x4332, 30.0);
    let clean = situ_launch(h, &gate, &up, (GATE_BETA, VALUE_BETA))?;
    common::expect_clean(h, "the clean situ")?;
    for (what, at, bad_g, bad_u) in [
        ("a NaN gate", 17, f32::NAN, up[17]),
        ("a NaN up", 4242, gate[4242], f32::NAN),
    ] {
        let (mut g2, mut u2) = (gate.clone(), up.clone());
        g2[at] = bad_g;
        u2[at] = bad_u;
        let got = situ_launch(h, &g2, &u2, (GATE_BETA, VALUE_BETA))?;
        common::expect_fault(h, what, FaultSite::F32Product)?;
        if !got[at].is_nan() {
            return Err(format!("{what} left {:e} at element {at}", got[at]));
        }
        for (i, (a, b)) in got.iter().zip(&clean).enumerate() {
            if i != at && a.to_bits() != b.to_bits() {
                return Err(format!("{what} moved element {i}"));
            }
        }
    }
    for (label, betas) in [
        ("a zero gate beta", (0.0, VALUE_BETA)),
        ("a negative value beta", (GATE_BETA, -1.0)),
        ("an infinite value beta", (GATE_BETA, f32::INFINITY)),
        ("a NaN gate beta", (f32::NAN, VALUE_BETA)),
    ] {
        ink2::refused(label, situ_launch(h, &gate, &up, betas).map(|_| ()), "beta")?;
    }
    Ok(Verdict::Noted(
        "a NaN gate and a NaN up are NaN on their element alone and raise the product site; four betas refused by name".to_string(),
    ))
}

/// The situ launch's shape refusals by name, nothing encoded.
pub fn situ_bounds(h: &Harness<'_>) -> Clause {
    let a = ink2::up(h, &vec![1.0f32; N])?;
    let b = ink2::up(h, &vec![1.0f32; N])?;
    let mut y = ink2::up(h, &ink2::blank(N))?;
    let mut run = |label: &str, n: usize, gl: usize, ul: usize, yl: usize, needle: &str| {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.pair_situ(
                    encoder,
                    SituParams {
                        n,
                        gate: a.window(Elems(0), Elems(gl))?,
                        up: b.window(Elems(0), Elems(ul))?,
                        y: y.window_mut(Elems(0), Elems(yl))?,
                        gate_beta: GATE_BETA,
                        value_beta: VALUE_BETA,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    run("no values", 0, N, N, N, "0 values")?;
    run("a short gate", N, N - 1, N, N, "gate")?;
    run("a short up", N, N, N - 1, N, "up")?;
    run("a short output", N, N, N, N - 1, "y")?;
    Ok(Verdict::Noted(
        "four refusals by name, each before an encode".to_string(),
    ))
}

/// The combine's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("situ", Box::new(|| situ(h)) as ClauseRun),
        ("situ_nan", Box::new(|| situ_nan(h))),
        ("situ_bounds", Box::new(|| situ_bounds(h))),
    ]
}
