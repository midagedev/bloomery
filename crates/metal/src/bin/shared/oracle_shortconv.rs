//! The short conv's and the sandwich residual's oracle: the kernels'
//! arithmetic in f64, and the absolute error one f32 evaluation of it can
//! carry.
//!
//! # Short conv
//!
//! A sequence is held position by position, the way the contract states it,
//! with no ring: position `t`'s output is
//! `y_t[c] = C_t[c] · Σ_k w[c][k] · u_{t-(taps-1)+k}[c]`, the `u` of a position
//! before the sequence's first being 0. `u = B ⊙ x` is one f32 product, the
//! value the kernels store in the ring and recompute for an in-chunk
//! predecessor, so the oracle takes it as an exact f32 input. The ring a
//! kernel leaves is the contract's cache of that history: slot `s` holds the
//! `u` of the latest position `p` with `p mod ring_rows = s`, and a slot no
//! position has reached holds what the ring began with. The ring is compared
//! bit for bit.
//!
//! The error is the model of the delta conv's oracle (`oracle.rs` owns its arithmetic): the
//! kernel's sum is one fma chain from the oldest tap and the C product, each
//! operation rounds to nearest once, `U = 2^-24`. A product pays `ea·eb` as
//! well as the two linear terms, and an fma rounds `a·b + c` once. The
//! subnormal floor is one minimum-positive f32 per operation of the path
//! (`taps` fmas and the C product).
//!
//! # Sandwich residual
//!
//! `y += add · ((x · gain) · scale)` with `scale = rsqrt(mean(x²) + eps)`, a
//! row at a time. The kernel's sum of squares is f32: a square and three adds
//! inside a float4, one add a float4 of the thread's chain (`cols / 4 /
//! threads` of them), a SIMD-group sum (at most 31 adds whatever its shape),
//! and the groups' shares added in group order (`simd_groups` adds): the
//! longest path from a term to the total is `n_sum = 4 + chain + 31 +
//! simd_groups`. The mean divides (one rounding) and `eps` adds (one), so the
//! scale's argument is off by at most `t = G(n_sum + 2)` relative and the
//! scale by at most `(1 - t)^(-1/2) - 1` plus the library `rsqrt`'s own error,
//! `RSQRT_ULP` units in the last place (`oracle.rs`, the norm on load, whose
//! first term to question this is too). The value passes the gain product, the
//! scale product and the `add` product: three roundings, `g = G(3)`; the
//! relative factor of the added term is `theta_s + g + theta_s·g`. The final
//! add rounds the sum once, `U |y|`. The band of an element is
//! `|add · v| · factor · (1 + U) + U |y| +` the floor, one minimum-positive
//! f32 per operation of the path.

use super::oracle::{Ev, RSQRT_ULP, U, fma, gamma_bound, mul};

/// The adds of a SIMD-group sum, longest path, whatever its tree.
const SIMD_SUM_DEPTH: usize = 31;

/// One sequence's history, a position at a time: the products `B ⊙ x` the
/// ring keeps and the `C` factors, as the f32 values the kernels read.
pub struct Sequence {
    hidden: usize,
    taps: usize,
    w: Vec<f32>,
    /// `B ⊙ x`, position-major.
    u: Vec<f32>,
    /// `C`, position-major.
    gate: Vec<f32>,
}

impl Sequence {
    /// A sequence of `hidden` channels over `w` (`hidden · taps`, a channel's
    /// taps together, oldest first).
    pub fn new(hidden: usize, taps: usize, w: &[f32]) -> Result<Self, String> {
        if hidden == 0 || taps == 0 {
            return Err(format!(
                "a sequence of {hidden} channels and {taps} taps has no conv"
            ));
        }
        if w.len() != hidden * taps {
            return Err(format!(
                "{} weights, {hidden} channels of {taps} taps need {}",
                w.len(),
                hidden * taps
            ));
        }
        Ok(Self {
            hidden,
            taps,
            w: w.to_vec(),
            u: Vec::new(),
            gate: Vec::new(),
        })
    }

    /// The positions pushed so far.
    pub fn positions(&self) -> usize {
        self.u.len() / self.hidden
    }

    /// The next position's row of `B | C | x`, at least `3 · hidden` values.
    pub fn push(&mut self, row: &[f32]) -> Result<(), String> {
        let h = self.hidden;
        if row.len() < 3 * h {
            return Err(format!(
                "a row of {} values, B | C | x need {}",
                row.len(),
                3 * h
            ));
        }
        self.u
            .extend(row[..h].iter().zip(&row[2 * h..3 * h]).map(|(b, x)| b * x));
        self.gate.extend_from_slice(&row[h..2 * h]);
        Ok(())
    }

    /// The conv of position `t` for every channel, and the absolute error an
    /// f32 evaluation may add to each.
    fn y(&self, t: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
        if t >= self.positions() {
            return Err(format!(
                "position {t} of a sequence of {}",
                self.positions()
            ));
        }
        let h = self.hidden;
        let mut value = Vec::with_capacity(h);
        let mut error = Vec::with_capacity(h);
        for c in 0..h {
            let mut acc = Ev::zero();
            for k in 0..self.taps {
                let back = self.taps - 1 - k;
                let u = if back <= t {
                    Ev::exact(self.u[(t - back) * h + c])
                } else {
                    Ev::zero()
                };
                acc = fma(u, Ev::exact(self.w[c * self.taps + k]), acc);
            }
            let y = mul(Ev::exact(self.gate[t * h + c]), acc);
            value.push(y.v);
            error.push(y.e);
        }
        Ok((value, error))
    }

    /// `got`, dense rows of `hidden`, against positions `first..`: every
    /// element within its band. Returns the worst error over its band.
    pub fn fit_y(&self, first: usize, got: &[f32]) -> Result<f64, String> {
        let h = self.hidden;
        if got.is_empty() || !got.len().is_multiple_of(h) {
            return Err(format!(
                "{} values read, rows of {h} channels were expected",
                got.len()
            ));
        }
        let floor = (self.taps + 1) as f64 * f64::from(f32::MIN_POSITIVE);
        let mut worst = 0.0f64;
        for (i, row) in got.chunks_exact(h).enumerate() {
            let (value, error) = self.y(first + i)?;
            for (c, &g) in row.iter().enumerate() {
                if !value[c].is_finite() || !error[c].is_finite() {
                    return Err(format!(
                        "position {}, channel {c}: the oracle or its band is not finite",
                        first + i
                    ));
                }
                let width = error[c] + floor;
                let err = (f64::from(g) - value[c]).abs();
                if !g.is_finite() || err > width {
                    return Err(format!(
                        "position {}, channel {c}: got {g:e}, want {:e}, error {err:e} over the band {width:e}",
                        first + i,
                        value[c]
                    ));
                }
                worst = worst.max(err / width);
            }
        }
        Ok(worst)
    }

    /// The ring after the positions so far, from a ring that began as `init`
    /// (`ring_rows · hidden`).
    pub fn ring(&self, ring_rows: usize, init: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.hidden;
        if ring_rows == 0 || init.len() != ring_rows * h {
            return Err(format!(
                "a ring of {ring_rows} rows needs {} values, not {}",
                ring_rows * h,
                init.len()
            ));
        }
        let mut out = init.to_vec();
        for t in 0..self.positions() {
            let s = t % ring_rows;
            out[s * h..(s + 1) * h].copy_from_slice(&self.u[t * h..(t + 1) * h]);
        }
        Ok(out)
    }
}

/// `got` and `want` hold the same words; the first that differs is named.
pub fn same_ring(got: &[f32], want: &[f32], what: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} values read, the oracle has {}",
            got.len(),
            want.len()
        ));
    }
    match got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits())
    {
        Some(i) => Err(format!("{what}[{i}]: got {:e}, want {:e}", got[i], want[i])),
        None => Ok(()),
    }
}

/// The inputs of one sandwich launch, rows dense (`rows · cols`).
pub struct SandwichIn<'a> {
    pub cols: usize,
    pub eps: f32,
    pub add: f32,
    /// The threads of the norm's threadgroup, and the SIMD groups in it.
    pub threads: usize,
    pub simd_groups: usize,
    /// The sub-layer's output rows.
    pub out: &'a [f32],
    pub gain: &'a [f32],
    /// The residual stream's rows before the add.
    pub stream: &'a [f32],
}

/// The stream after the add, in f64, and the absolute error band of each element.
pub struct Sandwich {
    pub want: Vec<f64>,
    pub band: Vec<f64>,
}

/// The stream a sandwich launch leaves, and its band.
pub fn sandwich(inp: &SandwichIn<'_>) -> Result<Sandwich, String> {
    let cols = inp.cols;
    if cols == 0 || inp.gain.len() != cols {
        return Err(format!(
            "a gain of {} values for rows of {cols}",
            inp.gain.len()
        ));
    }
    if inp.out.is_empty()
        || !inp.out.len().is_multiple_of(cols)
        || inp.stream.len() != inp.out.len()
    {
        return Err(format!(
            "{} output values and {} stream values are not the same whole rows of {cols}",
            inp.out.len(),
            inp.stream.len()
        ));
    }
    if inp.threads == 0 || !cols.is_multiple_of(4 * inp.threads) {
        return Err(format!(
            "{cols} columns are not whole float4s of {} threads",
            inp.threads
        ));
    }
    let chain = cols / 4 / inp.threads;
    let n_sum = 4 + chain + SIMD_SUM_DEPTH + inp.simd_groups;
    let t = gamma_bound(n_sum + 2);
    let theta_s = (1.0 - t).powf(-0.5) - 1.0 + RSQRT_ULP * 2.0 * U;
    let g = gamma_bound(3);
    let factor = theta_s + g + theta_s * g;
    let floor = (n_sum + 2 + 4) as f64 * f64::from(f32::MIN_POSITIVE);
    let add = f64::from(inp.add);
    let mut want = Vec::with_capacity(inp.out.len());
    let mut band = Vec::with_capacity(inp.out.len());
    for (x, stream) in inp
        .out
        .chunks_exact(cols)
        .zip(inp.stream.chunks_exact(cols))
    {
        let mean = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / cols as f64;
        let scale = 1.0 / (mean + f64::from(inp.eps)).sqrt();
        for ((&xv, &gv), &sv) in x.iter().zip(inp.gain).zip(stream) {
            let v = f64::from(xv) * f64::from(gv) * scale;
            let added = add * v;
            let y = f64::from(sv) + added;
            want.push(y);
            band.push(added.abs() * factor * (1.0 + U) + U * y.abs() + floor);
        }
    }
    Ok(Sandwich { want, band })
}

/// `got` against the oracle, every element within its band. Returns the
/// worst error over its band.
pub fn fit_sandwich(got: &[f32], oracle: &Sandwich) -> Result<f64, String> {
    if got.len() != oracle.want.len() {
        return Err(format!(
            "{} values read, the oracle has {}",
            got.len(),
            oracle.want.len()
        ));
    }
    let mut worst = 0.0f64;
    for (i, &g) in got.iter().enumerate() {
        let err = (f64::from(g) - oracle.want[i]).abs();
        if !g.is_finite() || err > oracle.band[i] {
            return Err(format!(
                "[{i}]: got {g:e}, want {:e}, error {err:e} over the band {:e}",
                oracle.want[i], oracle.band[i]
            ));
        }
        worst = worst.max(err / oracle.band[i]);
    }
    Ok(worst)
}
