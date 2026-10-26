//! The Mamba-2 step's oracle: the kernel's arithmetic in f64 (the host twin's
//! `scan_token`, `skip_d`, `gate_z` and `grouped_rms`, and its conv, which the
//! `model` crate cannot link on this machine), and the absolute error one f32
//! evaluation of that arithmetic can carry.
//!
//! The device has no f64 in the shader. Every value is an [`Ev`], a value and
//! the radius an f32 evaluation may add to it. `U = 2^-24`. An f32 operation
//! rounds once to nearest, so a product or a sum adds `U` of its magnitude, a
//! division too, and a quotient's input radius is widened by its Lipschitz
//! factor. `exp` and `log` are taken within 4 ulp, and 1 ulp of a value `y` is
//! at most `2 U |y|`, so 4 ulp is at most `8 U |y|`. `rsqrt` is within
//! `RSQRT_ULP`. The subnormal floor is one minimum-positive f32 for each of
//! the 8 operations of the longest path, and covers the values the device may
//! flush.
//!
//! Each step is oracled from the kernel's own f32 snapshot of the lane it
//! read, so its input radius is 0. The state's band is one update
//! `s' = s dA + B (x sigma)`; the output `y` is oracled from the state the
//! kernel wrote, so the band of `y` does not carry the state's. The partial
//! sums and the norm are oracled from the `y` and the partials the kernel
//! wrote. The state carried across tokens is checked against a free-running
//! f64 recurrence with the sum of the steps' bands: with `A <= 0` the decay is
//! at most 1, so the error of a step does not grow through the next, and the
//! bound is the sum of the one-step bands.
//!
//! `sigma = log1p(exp(dt))` is the one value with no `Ev` chain: the kernel
//! computes `log(u) x / (u - 1)` with `u = 1 + x` rounded and `x = exp(dt)`,
//! which cancels `u`'s rounding, and the chain would not carry the
//! correlation. Its error is `x`'s (8 U, a 4 ulp `exp`), `u`'s rounding
//! through `log(u) / (u - 1)` (at most `U`), the `log` (8 U), the quotient
//! and the two products (3 U) and the rounding of `u - 1` (U): `SIGMA_REL`
//! takes 24 U of `sigma`. Past `dt = 20` the kernel returns `dt` itself.
//!
//! A predecessor of the conv at distance `d` is 0 while `pos < d`, and ring
//! slot `(pos - d) mod RING_ROWS` otherwise; the current token is the launch's
//! `xbc`.

use bloomery_metal::kernels::{MAMBA2_STATE, MAMBA2_TAPS, Mamba2Shape, RING_ROWS};

use super::oracle::{RSQRT_ULP, U};

/// The subnormal floor of the path the module doc counts.
const FLOOR: f64 = 8.0 * (f32::MIN_POSITIVE as f64);

/// Where `exp` stops being finite in f32. Past it the kernel's `exp` is
/// infinite and the quotient that uses it is the saturated value.
const EXP_F32: f64 = 88.722_839;

/// The relative error of the kernel's `sigma` for `dt <= 20`.
const SIGMA_REL: f64 = 24.0 * U;

/// One value and the absolute error an f32 evaluation may add.
#[derive(Clone, Copy)]
struct Ev {
    v: f64,
    e: f64,
}

impl Ev {
    fn exact(v: f32) -> Self {
        Self {
            v: f64::from(v),
            e: 0.0,
        }
    }
}

fn mul(a: Ev, b: Ev) -> Ev {
    let v = a.v * b.v;
    let cross = a.e * b.v.abs() + b.e * a.v.abs() + a.e * b.e;
    let mag = (a.v.abs() + a.e) * (b.v.abs() + b.e);
    Ev {
        v,
        e: cross + U * mag,
    }
}

fn add(a: Ev, b: Ev) -> Ev {
    let v = a.v + b.v;
    let mag = (a.v.abs() + a.e) + (b.v.abs() + b.e);
    Ev {
        v,
        e: a.e + b.e + U * mag,
    }
}

fn div(a: Ev, b: Ev) -> Result<Ev, String> {
    let denom = b.v.abs() - b.e;
    if denom.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err("band is not finite".to_string());
    }
    let v = a.v / b.v;
    let cross = a.e * b.v.abs() + b.e * a.v.abs() + a.e * b.e;
    let e_in = cross / (b.v.abs() * denom);
    let e = e_in + U * (v.abs() + e_in);
    if !v.is_finite() || !e.is_finite() {
        return Err("band is not finite".to_string());
    }
    Ok(Ev { v, e })
}

fn exp_ev(a: Ev) -> Result<Ev, String> {
    let v = a.v.exp();
    if !v.is_finite() {
        return Err("oracle is not finite".to_string());
    }
    let lip = a.e * (a.v + a.e).exp();
    if !lip.is_finite() {
        return Err("band is not finite".to_string());
    }
    Ok(Ev {
        v,
        e: lip + 8.0 * U * (v + lip),
    })
}

/// `x / (1 + exp(-x))` as the kernel evaluates it. A negative `x` past
/// `EXP_F32` makes the kernel's `exp` overflow, and the division is 0.
fn silu(a: Ev) -> Result<Ev, String> {
    if a.v + a.e < -EXP_F32 {
        return Ok(Ev { v: 0.0, e: 0.0 });
    }
    let den = add(Ev { v: 1.0, e: 0.0 }, exp_ev(Ev { v: -a.v, e: a.e })?);
    div(a, den)
}

/// `1 / sqrt(a)` through the library's `rsqrt`.
fn rsqrt_ev(a: Ev) -> Result<Ev, String> {
    let low = a.v - a.e;
    if low.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err("band is not finite".to_string());
    }
    let v = 1.0 / a.v.sqrt();
    let hi = 1.0 / low.sqrt();
    let lo = 1.0 / (a.v + a.e).sqrt();
    let lip = (hi - v).max(v - lo);
    Ok(Ev {
        v,
        e: lip + RSQRT_ULP * 2.0 * U * hi,
    })
}

/// `n U / (1 - n U)`: the relative error of a term that passes through `n`
/// rounded additions.
fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// The sum of `terms` whose longest path to the result is `depth` additions.
fn sum_ev(terms: &[Ev], depth: usize) -> Ev {
    let mut v = 0.0;
    let mut e = 0.0;
    let mut mag = 0.0;
    for t in terms {
        v += t.v;
        e += t.e;
        mag += t.v.abs() + t.e;
    }
    Ev {
        v,
        e: e + gamma(depth) * mag,
    }
}

/// The depth of the scan's `y` sum: a lane's four products in a chain (3
/// additions for the first terms), then the butterfly's five levels.
const Y_DEPTH: usize = 3 + 5;

/// `sigma` of `dtv`, a value of the f32 sum the kernel forms.
fn sigma_of(dtv: f32) -> Ev {
    let t = f64::from(dtv);
    if dtv > 20.0 {
        return Ev { v: t, e: 0.0 };
    }
    let v = t.exp().ln_1p();
    Ev {
        v,
        e: SIGMA_REL * v,
    }
}

/// The model's `sigma` in f64, for the free-running recurrence.
fn sigma_f64(dtv: f32) -> f64 {
    let t = f64::from(dtv);
    if dtv > 20.0 { t } else { t.exp().ln_1p() }
}

/// The weights of a layer.
pub struct Weights<'a> {
    pub conv_w: &'a [f32],
    pub conv_b: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub d: &'a [f32],
    pub gain: &'a [f32],
    pub eps: f32,
}

/// The conv input `back` positions before `pos` in channel `c`.
fn ring_input(ring: &[f32], pos: u32, back: u32, conv_dim: usize, c: usize) -> f32 {
    if pos < back {
        return 0.0;
    }
    let slot = (pos as usize + RING_ROWS - back as usize) % RING_ROWS;
    ring[slot * conv_dim + c]
}

/// What one step is given, besides the weights.
pub struct StepIn<'a> {
    pub shape: Mamba2Shape,
    pub w: &'a Weights<'a>,
    pub pos: u32,
    pub xbc: &'a [f32],
    pub dt: &'a [f32],
    pub z: &'a [f32],
    /// The state lane the step reads, `inner · 128` values.
    pub state: &'a [f32],
    /// The whole ring before the step.
    pub ring: &'a [f32],
}

/// What a step must leave: the new state and its bands, the ring, and how
/// narrow the state's bands are against the terms they sum.
pub struct StepOut {
    pub state: Vec<f64>,
    pub state_err: Vec<f64>,
    pub ring: Vec<f32>,
    /// The widest band of a state value over the largest magnitude of the two
    /// terms of an update in the step.
    pub tightest: f64,
    setup: Setup,
}

/// The per-head and per-group values of one step.
struct Setup {
    sigma: Vec<Ev>,
    decay: Vec<Ev>,
    x: Vec<Ev>,
    b: Vec<Ev>,
    c: Vec<Ev>,
}

fn conv_ev(inp: &StepIn<'_>, c: usize) -> Result<Ev, String> {
    let conv_dim = inp.shape.conv_dim;
    let back = |d: u32| ring_input(inp.ring, inp.pos, d, conv_dim, c);
    let taps = [back(3), back(2), back(1), inp.xbc[c]];
    let w = &inp.w.conv_w[MAMBA2_TAPS * c..MAMBA2_TAPS * (c + 1)];
    let mut acc = mul(Ev::exact(taps[0]), Ev::exact(w[0]));
    for (&x, &wt) in taps.iter().zip(w).skip(1) {
        acc = add(acc, mul(Ev::exact(x), Ev::exact(wt)));
    }
    silu(add(acc, Ev::exact(inp.w.conv_b[c])))
}

fn setup(inp: &StepIn<'_>) -> Result<Setup, String> {
    let s = inp.shape;
    let mut sigma = Vec::with_capacity(s.n_head);
    let mut decay = Vec::with_capacity(s.n_head);
    for h in 0..s.n_head {
        let dtv = inp.dt[h] + inp.w.dt_bias[h];
        let sg = sigma_of(dtv);
        decay.push(exp_ev(mul(sg, Ev::exact(inp.w.a[h])))?);
        sigma.push(sg);
    }
    let x = (0..s.inner)
        .map(|c| conv_ev(inp, c))
        .collect::<Result<Vec<_>, _>>()?;
    let bc = s.n_group * MAMBA2_STATE;
    let b = (0..bc)
        .map(|c| conv_ev(inp, s.inner + c))
        .collect::<Result<Vec<_>, _>>()?;
    let c = (0..bc)
        .map(|c| conv_ev(inp, s.inner + bc + c))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Setup {
        sigma,
        decay,
        x,
        b,
        c,
    })
}

/// The step in f64 with the bands of an f32 evaluation: the state the kernel
/// must write, the ring it must leave.
pub fn step(inp: &StepIn<'_>) -> Result<StepOut, String> {
    let s = inp.shape;
    let setup = setup(inp)?;
    let mut state = vec![0.0f64; s.inner * MAMBA2_STATE];
    let mut state_err = vec![0.0f64; s.inner * MAMBA2_STATE];
    let mut widest = 0.0f64;
    let mut largest = 0.0f64;
    for h in 0..s.n_head {
        let g = h / s.per;
        for ch in 0..s.head_dim {
            let at = h * s.head_dim + ch;
            let x_dt = mul(setup.x[at], setup.sigma[h]);
            for n in 0..MAMBA2_STATE {
                let i = at * MAMBA2_STATE + n;
                let t1 = mul(Ev::exact(inp.state[i]), setup.decay[h]);
                let t2 = mul(setup.b[g * MAMBA2_STATE + n], x_dt);
                let u = add(t1, t2);
                state[i] = u.v;
                state_err[i] = u.e;
                widest = widest.max(u.e);
                largest = largest.max(t1.v.abs() + t2.v.abs());
            }
        }
    }
    let mut ring = inp.ring.to_vec();
    let slot = inp.pos as usize % RING_ROWS;
    ring[slot * s.conv_dim..(slot + 1) * s.conv_dim].copy_from_slice(inp.xbc);
    let tightest = if largest > 0.0 { widest / largest } else { 0.0 };
    Ok(StepOut {
        state,
        state_err,
        ring,
        tightest,
        setup,
    })
}

/// The gated `y` the kernel must write, from the state it wrote: with its
/// bands.
pub fn y_gated(
    inp: &StepIn<'_>,
    out: &StepOut,
    state_out: &[f32],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let s = inp.shape;
    let mut v = vec![0.0f64; s.inner];
    let mut e = vec![0.0f64; s.inner];
    for h in 0..s.n_head {
        let g = h / s.per;
        for ch in 0..s.head_dim {
            let at = h * s.head_dim + ch;
            let terms: Vec<Ev> = (0..MAMBA2_STATE)
                .map(|n| {
                    mul(
                        Ev::exact(state_out[at * MAMBA2_STATE + n]),
                        out.setup.c[g * MAMBA2_STATE + n],
                    )
                })
                .collect();
            let acc = sum_ev(&terms, Y_DEPTH);
            let skip = mul(out.setup.x[at], Ev::exact(inp.w.d[h]));
            let gate = silu(Ev::exact(inp.z[at]))?;
            let r = mul(gate, add(acc, skip));
            v[at] = r.v;
            e[at] = r.e;
        }
    }
    Ok((v, e))
}

/// The sums of squares the kernel must write, from the `y` it wrote: one a
/// threadgroup, in the step's block order.
pub fn partials(shape: Mamba2Shape, y: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut v = Vec::with_capacity(shape.blocks());
    let mut e = Vec::with_capacity(shape.blocks());
    for h in 0..shape.n_head {
        for chunk in 0..shape.chunks {
            let live = (shape.head_dim - chunk * 16).min(16);
            let terms: Vec<Ev> = (0..live)
                .map(|k| {
                    let t = Ev::exact(y[h * shape.head_dim + chunk * 16 + k]);
                    mul(t, t)
                })
                .collect();
            let p = sum_ev(&terms, 16);
            v.push(p.v);
            e.push(p.e);
        }
    }
    (v, e)
}

/// The normalized `y` the kernel must write, from the `y` and the partials it
/// wrote: with its bands.
pub fn norm(
    shape: Mamba2Shape,
    w: &Weights<'_>,
    y: &[f32],
    part: &[f32],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let width = shape.per * shape.head_dim;
    let stat = shape.per * shape.chunks;
    let mut v = vec![0.0f64; shape.inner];
    let mut e = vec![0.0f64; shape.inner];
    for g in 0..shape.n_group {
        let terms: Vec<Ev> = part[g * stat..(g + 1) * stat]
            .iter()
            .map(|&p| Ev::exact(p))
            .collect();
        let total = sum_ev(&terms, stat);
        let mean = div(total, Ev::exact(width as f32))?;
        let scale = rsqrt_ev(add(mean, Ev::exact(w.eps)))?;
        for i in 0..width {
            let at = g * width + i;
            let r = mul(mul(scale, Ev::exact(w.gain[at])), Ev::exact(y[at]));
            v[at] = r.v;
            e[at] = r.e;
        }
    }
    Ok((v, e))
}

/// `got` within `e + floor` of `v`: the worst error over the band.
pub fn scan(got: &[f32], v: &[f64], e: &[f64], floor: f64, what: &str) -> Result<f64, String> {
    if got.len() != v.len() || v.len() != e.len() {
        return Err(format!(
            "{what}: {} values read, the oracle has {}",
            got.len(),
            v.len()
        ));
    }
    let mut ratio = 0.0f64;
    for (i, &g) in got.iter().enumerate() {
        if !v[i].is_finite() {
            return Err(format!("{what}[{i}]: oracle is not finite"));
        }
        if !e[i].is_finite() || e[i] < 0.0 {
            return Err(format!("{what}[{i}]: band is not finite"));
        }
        let width = e[i] + floor;
        let err = (f64::from(g) - v[i]).abs();
        if !g.is_finite() || err > width {
            return Err(format!(
                "{what}[{i}]: got {g:e}, want {:e}, error {err:e} over the band {width:e}",
                v[i]
            ));
        }
        ratio = ratio.max(err / width);
    }
    Ok(ratio)
}

/// The subnormal floor of one step.
pub const STEP_FLOOR: f64 = FLOOR;

/// The state of the model's recurrence in f64 over the tokens a run has fed,
/// and the sum of the steps' bands.
pub struct Carry {
    state: Vec<f64>,
    band: Vec<f64>,
    steps: usize,
}

impl Carry {
    /// A recurrence that starts from `state`, the lane the first step reads.
    pub fn new(state: &[f32]) -> Self {
        Self {
            state: state.iter().map(|&v| f64::from(v)).collect(),
            band: vec![0.0; state.len()],
            steps: 0,
        }
    }

    /// One token: the conv from the inputs of the positions before it
    /// (`history[p]` is the `xbc` of position `p`, the run's first position
    /// 0), then the scan, in f64. `step_band` is this step's state band.
    pub fn step(&mut self, inp: &StepIn<'_>, history: &[Vec<f32>], step_band: &[f64]) {
        let s = inp.shape;
        let pos = inp.pos as usize;
        let conv = |c: usize| -> f64 {
            let tap = |back: usize| -> f64 {
                if pos < back {
                    0.0
                } else {
                    f64::from(history[pos - back][c])
                }
            };
            let w = &inp.w.conv_w[MAMBA2_TAPS * c..MAMBA2_TAPS * (c + 1)];
            let acc = tap(3) * f64::from(w[0])
                + tap(2) * f64::from(w[1])
                + tap(1) * f64::from(w[2])
                + f64::from(history[pos][c]) * f64::from(w[3])
                + f64::from(inp.w.conv_b[c]);
            acc / (1.0 + (-acc).exp())
        };
        let bc = s.n_group * MAMBA2_STATE;
        let x: Vec<f64> = (0..s.inner).map(&conv).collect();
        let b: Vec<f64> = (0..bc).map(|c| conv(s.inner + c)).collect();
        for h in 0..s.n_head {
            let g = h / s.per;
            let sigma = sigma_f64(inp.dt[h] + inp.w.dt_bias[h]);
            let decay = (sigma * f64::from(inp.w.a[h])).exp();
            for ch in 0..s.head_dim {
                let at = h * s.head_dim + ch;
                let x_dt = x[at] * sigma;
                for n in 0..MAMBA2_STATE {
                    let i = at * MAMBA2_STATE + n;
                    self.state[i] = self.state[i] * decay + b[g * MAMBA2_STATE + n] * x_dt;
                }
            }
        }
        for (sum, &e) in self.band.iter_mut().zip(step_band) {
            *sum += e;
        }
        self.steps += 1;
    }

    /// The kernel's state against the recurrence, within the summed bands:
    /// the worst error over the bound.
    pub fn fit(&self, got: &[f32]) -> Result<f64, String> {
        scan(
            got,
            &self.state,
            &self.band,
            self.steps as f64 * FLOOR,
            "carried state",
        )
    }
}
