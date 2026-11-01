//! The gated delta step's oracle: the kernel's arithmetic in f64, and the
//! absolute error one f32 evaluation of that arithmetic can carry.
//!
//! The device has no f64 in the shader, so the query and key sum of squares is f32
//! `(s0 + s1) + (s2 + s3)` then a butterfly of widths 16, 8, 4, 2, 1, and the
//! delta butterfly is f32 of widths 4, 2, 1. `exp` and `log` are taken within
//! 4 ulp, and 1 ulp of a value `y` is at most `2·U·|y|`, so 4 ulp is at most
//! `8·U·|y|`. Division and square root are correctly rounded, which adds
//! `U·|y|`. `U = 2^-24`. An input's error is widened by the Lipschitz factor
//! of the operation and then by that rounding. A product pays `ea·eb` as well
//! as the two linear terms. An fma rounds `a·b + c` once. `max(sqrt, eps)`
//! covers both outcomes when the interval straddles `eps`, and softplus covers
//! both branches when the interval straddles 20. The roll factor is 1: each
//! step is re-oracled from the kernel's own f32 snapshot, so the input error
//! is 0, and the radius already includes the full image of `exp` and the
//! per-operation factor `(1 + U)^n` rather than a truncated series. The
//! subnormal floor is `8·f32::MIN_POSITIVE` and is not a widening of the factor.
//!
//! A predecessor at distance `d` is 0 while `pos < d`, and ring slot
//! `(pos − d) mod ring_rows` otherwise. Distance 0 is the launch's `x`. At
//! position 0 every predecessor is 0. At position 1 the distance-1 tap reads
//! the ring.
//!
//! The numbers of the step (value heads, columns of a head, taps, ring rows,
//! the query's scale) are the geometry the input carries ([`StepIn::delta`]),
//! so the oracle judges a launch at the instance it ran on.

use bloomery_metal::geom::{DeltaGeom, Geometry};
use bloomery_metal::instance::REGISTERED;
use bloomery_metal::kernels::{HeadMap, PARTIALS_PER_HEAD, key_of};

use super::oracle::{Ev, U, fma, mul};

/// What the kernel's lane map and the oracle's association are written for,
/// held for every registered row: a head of 128 columns (32 lanes of four,
/// sixteen keys by eight lanes a column) and a conv of four taps.
const _: () = {
    let mut i = 0;
    while i < REGISTERED.len() {
        let d = Geometry::of(&REGISTERED[i]).delta;
        assert!(d.head_cols == 128);
        assert!(d.conv_taps == 4);
        i += 1;
    }
};

/// The subnormal floor: one minimum-positive f32 per operation of the path
/// the model in this module's doc counts, which the relative bound does not
/// cover.
///
/// PIN(2026-10-22): the model in this module's doc sets the one-step band.
const FLOOR: f64 = 8.0 * (f32::MIN_POSITIVE as f64);

fn add(a: Ev, b: Ev) -> Ev {
    let v = a.v + b.v;
    let mag = (a.v.abs() + a.e) + (b.v.abs() + b.e);
    Ev {
        v,
        e: a.e + b.e + U * mag,
    }
}

fn sub(a: Ev, b: Ev) -> Ev {
    add(a, Ev { v: -b.v, e: b.e })
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

fn sqrt_ev(a: Ev) -> Result<Ev, String> {
    if a.v < 0.0 || a.v - a.e < 0.0 {
        return Err("band is not finite".to_string());
    }
    let lo = (a.v - a.e).sqrt();
    let hi = (a.v + a.e).sqrt();
    let v = a.v.sqrt();
    let e = (v - lo).max(hi - v) + U * hi;
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
    let e = lip + 8.0 * U * (v + lip);
    Ok(Ev { v, e })
}

fn log_ev(a: Ev) -> Result<Ev, String> {
    if a.v.partial_cmp(&a.e) != Some(std::cmp::Ordering::Greater) {
        return Err("band is not finite".to_string());
    }
    let v = a.v.ln();
    let lip = a.e / (a.v - a.e);
    if !v.is_finite() || !lip.is_finite() {
        return Err("band is not finite".to_string());
    }
    Ok(Ev {
        v,
        e: lip + 8.0 * U * (v.abs() + lip),
    })
}

fn neg_exp(a: Ev) -> Result<Ev, String> {
    exp_ev(Ev { v: -a.v, e: a.e })
}

fn one() -> Ev {
    Ev { v: 1.0, e: 0.0 }
}

/// Where `exp` stops being finite in f32. Past it the kernel's `exp` is
/// infinite and the quotient that uses it is the saturated value.
const EXP_F32: f64 = 88.722_839;

fn silu(a: Ev) -> Result<Ev, String> {
    // acc / (1 + exp(-acc)). A negative acc past EXP_F32 makes the kernel's
    // exp overflow, and the division is 0.
    if a.v + a.e < -EXP_F32 {
        return Ok(Ev::zero());
    }
    let den = add(one(), neg_exp(a)?);
    div(a, den)
}

fn sigmoid(a: Ev) -> Result<Ev, String> {
    if a.v + a.e < -EXP_F32 {
        return Ok(Ev::zero());
    }
    if a.v - a.e > EXP_F32 {
        return Ok(one());
    }
    div(one(), add(one(), neg_exp(a)?))
}

fn softplus(a: Ev) -> Result<Ev, String> {
    let lo = a.v - a.e;
    let hi = a.v + a.e;
    if lo > 20.0 {
        return Ok(a);
    }
    let series = log_ev(add(one(), exp_ev(a)?))?;
    if hi <= 20.0 {
        return Ok(series);
    }
    let span_lo = (a.v - a.e).min(series.v - series.e);
    let span_hi = (a.v + a.e).max(series.v + series.e);
    let v = if a.v > 20.0 { a.v } else { series.v };
    Ok(Ev {
        v,
        e: (v - span_lo).max(span_hi - v),
    })
}

/// `max(sqrt, eps)` as the kernel compares them. An interval that contains
/// `eps` covers both branches.
fn max_eps(root: Ev, eps: f64) -> Ev {
    let lo = root.v - root.e;
    let hi = root.v + root.e;
    if lo > eps {
        root
    } else if hi <= eps {
        Ev { v: eps, e: 0.0 }
    } else {
        let v = if root.v > eps { root.v } else { eps };
        let span_lo = lo.min(eps);
        let span_hi = hi.max(eps);
        Ev {
            v,
            e: (v - span_lo).max(span_hi - v),
        }
    }
}

fn butterfly<const N: usize>(mut v: [Ev; N], dists: &[usize]) -> [Ev; N] {
    for &d in dists {
        let old = v;
        for i in 0..N {
            v[i] = add(old[i], old[i ^ d]);
        }
    }
    v
}

fn lane_dot(a: &[Ev; 16], b: &[Ev; 16]) -> Ev {
    let mut p = [
        mul(a[0], b[0]),
        mul(a[1], b[1]),
        mul(a[2], b[2]),
        mul(a[3], b[3]),
    ];
    for r in 1..4 {
        for (c, slot) in p.iter_mut().enumerate() {
            let i = 4 * r + c;
            *slot = fma(a[i], b[i], *slot);
        }
    }
    add(add(p[0], p[1]), add(p[2], p[3]))
}

/// One step's inputs. The slices are the kernel's windows.
pub struct StepIn<'a> {
    /// The step's numbers: the instance the launch ran on.
    pub delta: &'a DeltaGeom,
    /// Key heads.
    pub n_k: usize,
    /// How a value head finds its key head.
    pub map: HeadMap,
    /// The lane the step reads. It is inside `lanes`.
    pub lane: u32,
    /// The lane the step writes. It is inside `lanes`; equal to `lane`, the
    /// step updates that lane in place, and apart from it the read lane keeps
    /// its state.
    pub lane_out: u32,
    /// The position.
    pub pos: u32,
    /// The L2 epsilon.
    pub eps: f32,
    /// Lanes of `state`.
    pub lanes: usize,
    /// The conv input.
    pub x: &'a [f32],
    /// The conv weights, tap innermost.
    pub conv_w: &'a [f32],
    /// `β`'s pre-sigmoid input.
    pub b: &'a [f32],
    /// The decay's pre-softplus input.
    pub a: &'a [f32],
    /// The decay bias.
    pub dt: &'a [f32],
    /// The decay slope.
    pub ssm_a: &'a [f32],
    /// The gate activation's input.
    pub z: &'a [f32],
    /// The norm gain.
    pub norm_w: &'a [f32],
    /// The state before the step.
    pub state: &'a [f32],
    /// The ring before the step.
    pub ring: &'a [f32],
}

/// One step's outputs. `g`, the partials and the state carry an absolute
/// error; the ring write is the raw conv input, so it is bit-exact.
pub struct StepOut {
    /// The gated numerator.
    pub g: Vec<f64>,
    /// The absolute error of [`StepOut::g`].
    pub g_err: Vec<f64>,
    /// The partial sums of `o²`.
    pub partials: Vec<f64>,
    /// The absolute error of [`StepOut::partials`].
    pub partials_err: Vec<f64>,
    /// The state after the step.
    pub state: Vec<f64>,
    /// The absolute error of [`StepOut::state`].
    pub state_err: Vec<f64>,
    /// The ring after the step.
    pub ring: Vec<f32>,
    /// The widest sign flip of the state update, `2 |k u|`, over the step: a
    /// state band wider than this would hide the flip.
    pub flip: f64,
}

/// The worst ratio of an error to its band, and the widest state band.
pub struct Fit {
    /// The largest `error / band` over the compared values.
    pub ratio: f64,
    /// The largest absolute state band.
    pub widest_state: f64,
}

struct Conv<'a> {
    delta: &'a DeltaGeom,
    pos: u32,
    ch: usize,
    x: &'a [f32],
    w: &'a [f32],
    ring: &'a [f32],
}

fn tap(c: &Conv<'_>, channel: usize, d: u32) -> Ev {
    if d == 0 {
        return Ev::exact(c.x[channel]);
    }
    if c.pos < d {
        return Ev::zero();
    }
    let ring_rows = c.delta.ring_rows as u32;
    let slot = (c.pos - d) % ring_rows;
    Ev::exact(c.ring[slot as usize * c.ch + channel])
}

fn conv_silu(c: &Conv<'_>, channel: usize) -> Result<Ev, String> {
    let xs = [
        tap(c, channel, 3),
        tap(c, channel, 2),
        tap(c, channel, 1),
        tap(c, channel, 0),
    ];
    let taps = c.delta.conv_taps;
    let mut acc = mul(xs[0], Ev::exact(c.w[taps * channel]));
    for (j, &x) in xs.iter().enumerate().take(taps).skip(1) {
        acc = fma(x, Ev::exact(c.w[taps * channel + j]), acc);
    }
    silu(acc)
}

fn conv_head(c: &Conv<'_>, head: usize) -> Result<Vec<Ev>, String> {
    let cols = c.delta.head_cols;
    let mut out = vec![Ev::zero(); cols];
    for (col, slot) in out.iter_mut().enumerate() {
        *slot = conv_silu(c, head * cols + col)?;
    }
    Ok(out)
}

fn l2_scale(sum: Ev, eps: f64) -> Result<Ev, String> {
    if !sum.v.is_finite() {
        return Err("oracle is not finite".to_string());
    }
    div(one(), max_eps(sqrt_ev(sum)?, eps))
}

fn normalize(
    delta: &DeltaGeom,
    q: &[Ev],
    k: &[Ev],
    eps: f64,
) -> Result<(Vec<Ev>, Vec<Ev>), String> {
    let mut sq = [Ev::zero(); 32];
    let mut sk = [Ev::zero(); 32];
    for lane in 0..32 {
        let qsum = add(
            add(mul(q[lane], q[lane]), mul(q[lane + 32], q[lane + 32])),
            add(
                mul(q[lane + 64], q[lane + 64]),
                mul(q[lane + 96], q[lane + 96]),
            ),
        );
        let ksum = add(
            add(mul(k[lane], k[lane]), mul(k[lane + 32], k[lane + 32])),
            add(
                mul(k[lane + 64], k[lane + 64]),
                mul(k[lane + 96], k[lane + 96]),
            ),
        );
        sq[lane] = qsum;
        sk[lane] = ksum;
    }
    let sq = butterfly(sq, &[16, 8, 4, 2, 1]);
    let sk = butterfly(sk, &[16, 8, 4, 2, 1]);
    let q_scale = Ev::exact(delta.q_scale);
    let mut qo = vec![Ev::zero(); delta.head_cols];
    let mut ko = vec![Ev::zero(); delta.head_cols];
    for lane in 0..32 {
        let qs = l2_scale(sq[lane], eps)?;
        let ks = l2_scale(sk[lane], eps)?;
        for off in [0, 32, 64, 96] {
            qo[lane + off] = mul(mul(q[lane + off], qs), q_scale);
            ko[lane + off] = mul(k[lane + off], ks);
        }
    }
    Ok((qo, ko))
}

fn need(len: usize, want: usize, what: &str) -> Result<(), String> {
    if len == want {
        Ok(())
    } else {
        Err(format!("{what}: {len} values, the step has {want}"))
    }
}

fn check_lens(inp: &StepIn<'_>) -> Result<usize, String> {
    let d = inp.delta;
    let ch = d
        .channels(inp.n_k)
        .ok_or_else(|| format!("n_k {} channels do not fit", inp.n_k))?;
    let state = d
        .state_elems(inp.lanes)
        .ok_or_else(|| "the state does not fit".to_string())?;
    let ring = d
        .ring_elems(inp.n_k)
        .ok_or_else(|| "the ring does not fit".to_string())?;
    need(inp.x.len(), ch, "x")?;
    need(inp.conv_w.len(), d.conv_taps * ch, "conv weight")?;
    need(inp.b.len(), d.value_heads, "b")?;
    need(inp.a.len(), d.value_heads, "a")?;
    need(inp.dt.len(), d.value_heads, "dt bias")?;
    need(inp.ssm_a.len(), d.value_heads, "ssm a")?;
    need(inp.z.len(), d.value_heads * d.head_cols, "z")?;
    need(inp.norm_w.len(), d.head_cols, "norm weight")?;
    need(inp.state.len(), state, "state")?;
    need(inp.ring.len(), ring, "ring")?;
    if inp.n_k == 0 || !d.value_heads.is_multiple_of(inp.n_k) {
        return Err(format!("n_k {} does not divide the value heads", inp.n_k));
    }
    for (what, lane) in [("read", inp.lane), ("write", inp.lane_out)] {
        if (lane as usize) >= inp.lanes {
            return Err(format!(
                "oracle {what} lane {lane} is past the {} lanes",
                inp.lanes
            ));
        }
    }
    Ok(ch)
}

struct Head<'a> {
    delta: &'a DeltaGeom,
    h: usize,
    lane: usize,
    lane_out: usize,
    decay: Ev,
    beta: Ev,
    q: &'a [Ev],
    k: &'a [Ev],
    state_in: &'a [f32],
    z: &'a [f32],
    norm_w: &'a [f32],
}

struct Out<'a> {
    g: &'a mut [f64],
    g_err: &'a mut [f64],
    state: &'a mut [f64],
    state_err: &'a mut [f64],
    /// The widest sign flip of the state update, `2 |k u|`, over the step.
    flip: &'a mut f64,
}

fn column(
    head: &Head<'_>,
    out: &mut Out<'_>,
    col: usize,
    v: Ev,
    o2: &mut [Ev],
) -> Result<(), String> {
    let mut sp = [[Ev::zero(); 16]; 8];
    let mut kk = [[Ev::zero(); 16]; 8];
    let mut qq = [[Ev::zero(); 16]; 8];
    for j in 0..8 {
        for i in 0..16 {
            let key = key_of(j, i);
            let at = head.delta.state_index(head.lane, head.h, col, key);
            let s0 = Ev::exact(head.state_in[at]);
            sp[j][i] = mul(s0, head.decay);
            kk[j][i] = head.k[key];
            qq[j][i] = head.q[key];
        }
    }
    let mut dots = [Ev::zero(); 8];
    for j in 0..8 {
        dots[j] = lane_dot(&sp[j], &kk[j]);
    }
    let kvdot = butterfly(dots, &[4, 2, 1]);
    let mut u = [Ev::zero(); 8];
    for j in 0..8 {
        u[j] = mul(sub(v, kvdot[j]), head.beta);
    }
    for j in 0..8 {
        for i in 0..16 {
            *out.flip = (*out.flip).max(2.0 * (kk[j][i].v * u[j].v).abs());
            sp[j][i] = fma(kk[j][i], u[j], sp[j][i]);
            let key = key_of(j, i);
            let at = head.delta.state_index(head.lane_out, head.h, col, key);
            out.state[at] = sp[j][i].v;
            out.state_err[at] = sp[j][i].e;
        }
    }
    let mut odots = [Ev::zero(); 8];
    for j in 0..8 {
        odots[j] = lane_dot(&sp[j], &qq[j]);
    }
    let o = butterfly(odots, &[4, 2, 1])[0];
    let at = head.h * head.delta.head_cols + col;
    let gval = mul(
        mul(o, Ev::exact(head.norm_w[col])),
        silu(Ev::exact(head.z[at]))?,
    );
    out.g[at] = gval.v;
    out.g_err[at] = gval.e;
    o2[col % 16] = mul(o, o);
    Ok(())
}

/// The step: the kernel's association, in f64, with the absolute error.
pub fn step(inp: &StepIn<'_>) -> Result<StepOut, String> {
    let ch = check_lens(inp)?;
    let d = inp.delta;
    let conv = Conv {
        delta: d,
        pos: inp.pos,
        ch,
        x: inp.x,
        w: inp.conv_w,
        ring: inp.ring,
    };
    let eps = f64::from(inp.eps);
    let mut qs = Vec::with_capacity(inp.n_k);
    let mut ks = Vec::with_capacity(inp.n_k);
    for kh in 0..inp.n_k {
        let q = conv_head(&conv, kh)?;
        let k = conv_head(&conv, inp.n_k + kh)?;
        let (q, k) = normalize(d, &q, &k, eps)?;
        qs.push(q);
        ks.push(k);
    }
    let lane = inp.lane as usize;
    let lane_out = inp.lane_out as usize;
    let g_len = d.value_heads * d.head_cols;
    let mut g = vec![0.0; g_len];
    let mut g_err = vec![0.0; g_len];
    let mut partials = vec![0.0; d.value_heads * PARTIALS_PER_HEAD];
    let mut partials_err = vec![0.0; d.value_heads * PARTIALS_PER_HEAD];
    let mut out_state: Vec<f64> = inp.state.iter().map(|&v| f64::from(v)).collect();
    let mut out_err = vec![0.0; inp.state.len()];
    let mut flip = 0.0f64;
    for h in 0..d.value_heads {
        let decay = exp_ev(mul(
            Ev::exact(inp.ssm_a[h]),
            softplus(add(Ev::exact(inp.a[h]), Ev::exact(inp.dt[h])))?,
        ))?;
        let beta = sigmoid(Ev::exact(inp.b[h]))?;
        let kh = inp.map.k_head_of(h, d.value_heads, inp.n_k);
        let head = Head {
            delta: d,
            h,
            lane,
            lane_out,
            decay,
            beta,
            q: &qs[kh],
            k: &ks[kh],
            state_in: inp.state,
            z: inp.z,
            norm_w: inp.norm_w,
        };
        let mut out = Out {
            g: &mut g,
            g_err: &mut g_err,
            state: &mut out_state,
            state_err: &mut out_err,
            flip: &mut flip,
        };
        let mut o2 = [Ev::zero(); 16];
        for col in 0..d.head_cols {
            if col % 16 == 0 {
                o2 = [Ev::zero(); 16];
            }
            let v = conv_silu(&conv, (2 * inp.n_k + h) * d.head_cols + col)?;
            column(&head, &mut out, col, v, &mut o2)?;
            if col % 16 == 15 {
                let mut p = o2[0];
                for term in o2.iter().skip(1) {
                    p = add(p, *term);
                }
                let at = h * PARTIALS_PER_HEAD + col / 16;
                partials[at] = p.v;
                partials_err[at] = p.e;
            }
        }
    }
    let mut ring = inp.ring.to_vec();
    let slot = (inp.pos as usize) % d.ring_rows;
    ring[slot * ch..(slot + 1) * ch].copy_from_slice(inp.x);
    Ok(StepOut {
        g,
        g_err,
        partials,
        partials_err,
        state: out_state,
        state_err: out_err,
        ring,
        flip,
    })
}

fn cell(got: f32, v: f64, e: f64, what: &str, index: usize) -> Result<f64, String> {
    if !v.is_finite() {
        return Err(format!("{what}[{index}]: oracle is not finite"));
    }
    if !e.is_finite() || e < 0.0 {
        return Err(format!("{what}[{index}]: band is not finite"));
    }
    let width = e + FLOOR;
    let err = (f64::from(got) - v).abs();
    if !got.is_finite() || err > width {
        return Err(format!(
            "{what}[{index}]: got {got:e}, want {v:e}, error {err:e} over the band {width:e}"
        ));
    }
    Ok(err / width)
}

fn scan(got: &[f32], v: &[f64], e: &[f64], what: &str) -> Result<(f64, f64), String> {
    if got.len() != v.len() || v.len() != e.len() {
        return Err(format!(
            "{what}: {} values read, the oracle has {}",
            got.len(),
            v.len()
        ));
    }
    let mut ratio = 0.0f64;
    let mut widest = 0.0f64;
    for (i, &g) in got.iter().enumerate() {
        ratio = ratio.max(cell(g, v[i], e[i], what, i)?);
        widest = widest.max(e[i]);
    }
    Ok((ratio, widest))
}

/// `got` against `out`. The ring is compared bit for bit. When `tripwire` is
/// set, a state band wider than this step's sign-flip width `2 |k u|` fails:
/// that width would hide the state-update mutant.
pub fn fit(
    got_g: &[f32],
    got_partials: &[f32],
    got_state: &[f32],
    got_ring: &[f32],
    out: &StepOut,
    tripwire: bool,
) -> Result<Fit, String> {
    let (g_ratio, _) = scan(got_g, &out.g, &out.g_err, "g")?;
    let (p_ratio, _) = scan(got_partials, &out.partials, &out.partials_err, "partials")?;
    let (s_ratio, widest) = scan(got_state, &out.state, &out.state_err, "state")?;
    if got_ring.len() != out.ring.len() {
        return Err(format!(
            "ring: {} values read, the oracle has {}",
            got_ring.len(),
            out.ring.len()
        ));
    }
    for (i, (&got, &want)) in got_ring.iter().zip(out.ring.iter()).enumerate() {
        if got.to_bits() != want.to_bits() {
            return Err(format!("ring[{i}]: got {got:e}, want {want:e}"));
        }
    }
    if tripwire && widest > out.flip {
        return Err(format!(
            "the widest state band is {widest:e}, over this step's sign-flip width {:e}, which would hide a sign flip of the state update",
            out.flip
        ));
    }
    Ok(Fit {
        ratio: g_ratio.max(p_ratio).max(s_ratio),
        widest_state: widest,
    })
}
