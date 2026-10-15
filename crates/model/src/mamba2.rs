//! Host Mamba-2: one selective-scan step at state 128, the causal conv in
//! front of it, and the mix that follows the scan.
//!
//! The scan is the group kernel's recurrence, not a gated delta. Per head,
//! per channel, state length 128:
//!
//! ```text
//! dt_sp = dt                 if dt > 20
//!       = log1p(exp(dt))     otherwise
//! dA    = exp(dt_sp * A)     A is one float for the head
//! x_dt  = x * dt_sp
//! s[n]  = s[n] * dA + B[n] * x_dt
//! y     = sum_n s[n] * C[n]
//! ```
//!
//! `dt <= 20` takes `log1pf(expf(dt))`. Only a value past 20 stays itself.
//! `B` and `C` are one vector of 128 per group; heads in a group share them.
//! The group is `head / (n_head / n_group)`, both divisions truncating, the
//! inner one first. `y` adds the four products of lane `lane + 32*j`
//! (`j` from 0 to 3) and then a 32-wide xor reduction at offsets 16, 8, 4,
//! 2, 1, and keeps lane 0.
//!
//! State 96 and state 256 are refused by name. Nothing is written.
//!
//! The conv is the depthwise causal dot: tap 0 is the oldest sample and tap
//! `d_conv - 1` is the current one. That is llama.cpp `ace20f679e30`
//! `ggml/src/ggml-cuda/ssm-conv.cu`: `w[j]` multiplies time `i + j`.
//! A channel's row is its `d_conv - 1` state samples followed by the new
//! tokens. The bias add and then SiLU sit
//! on that output, and the result splits into `x`, `B`, `C`. After the scan,
//! `y = y + x * D` (`D` one float per head, broadcast across the head), then
//! `silu(z) * y`, then a grouped RMSNorm of width `d_inner / n_group` when a
//! gain is present. The out projection is not this module.

use runtime::hc_gated::card::butterfly;

use crate::ops::{self, Tensor2};

/// The scan's `dt` map. `dt <= 20` takes `log1p(exp(dt))`; a greater value
/// is itself.
fn scan_dt(dt: f32) -> f32 {
    if dt <= 20.0 { dt.exp().ln_1p() } else { dt }
}

/// The one state length the scan runs.
pub const STATE: usize = 128;

/// A Mamba-2 layer's widths. `d_state` is 128 on a layer this module runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geom {
    /// Channels, `n_head * head_dim`.
    pub d_inner: usize,
    /// Heads. `ssm.time_step_rank` on a Nemotron file.
    pub n_heads: usize,
    /// Groups that share one `B` and one `C`.
    pub n_groups: usize,
    /// Recurrent state length. The running value is 128.
    pub d_state: usize,
    /// Causal conv taps. The 4B file's kernel is 4.
    pub d_conv: usize,
}

/// Widths that divide, with state 128.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checked {
    geom: Geom,
    head_dim: usize,
    conv_dim: usize,
}

/// A width or a buffer this step will not run.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Mamba2Error {
    /// A state length other than [`STATE`] (Puzzle's 96, Falcon-H1's 256,
    /// any other). The scan does not launch it.
    #[error("mamba2: state {0} is refused; the scan runs state 128 and writes nothing")]
    State(usize),
    /// A buffer size that cannot be represented.
    #[error("mamba2: {0} size overflows usize")]
    Size(&'static str),
    /// `n_head` does not divide into `n_group`, so the group index is not a
    /// function of the head.
    #[error("mamba2: n_head {n_heads} is not a multiple of n_group {n_groups}")]
    Groups { n_heads: usize, n_groups: usize },
    /// `d_inner` does not split into heads and groups.
    #[error(
        "mamba2: d_inner {d_inner} does not divide into n_head {n_heads} and n_group {n_groups}"
    )]
    Inner {
        d_inner: usize,
        n_heads: usize,
        n_groups: usize,
    },
    /// A conv of no taps.
    #[error("mamba2: conv kernel 0 is refused")]
    Conv0,
    /// A buffer whose length is not the geometry's.
    #[error("mamba2: {what} has {got} values; it needs {need}")]
    Len {
        what: &'static str,
        got: usize,
        need: usize,
    },
}

impl Checked {
    /// Head width, `d_inner / n_head`.
    #[must_use]
    pub fn head_dim(self) -> usize {
        self.head_dim
    }

    /// Conv channels: `d_inner + 2 * n_group * d_state`.
    #[must_use]
    pub fn conv_dim(self) -> usize {
        self.conv_dim
    }

    /// The in-projection's row: `z`, `xBC` and `dt` side by side,
    /// `d_inner + conv_dim + n_head` values.
    #[must_use]
    pub fn in_proj_width(self) -> usize {
        self.geom.d_inner + self.conv_dim + self.geom.n_heads
    }

    /// The widths this check accepted.
    #[must_use]
    pub fn geom(self) -> Geom {
        self.geom
    }
}

/// Accept a geometry or refuse it by name. State 96 and 256 never become a
/// [`Checked`].
pub fn check(g: Geom) -> Result<Checked, Mamba2Error> {
    match g.d_state {
        STATE => {}
        other => return Err(Mamba2Error::State(other)),
    }
    if g.d_conv == 0 {
        return Err(Mamba2Error::Conv0);
    }
    if g.n_heads == 0 || g.n_groups == 0 || !g.n_heads.is_multiple_of(g.n_groups) {
        return Err(Mamba2Error::Groups {
            n_heads: g.n_heads,
            n_groups: g.n_groups,
        });
    }
    if g.d_inner == 0
        || !g.d_inner.is_multiple_of(g.n_heads)
        || !g.d_inner.is_multiple_of(g.n_groups)
    {
        return Err(Mamba2Error::Inner {
            d_inner: g.d_inner,
            n_heads: g.n_heads,
            n_groups: g.n_groups,
        });
    }
    let bc_n = g
        .n_groups
        .checked_mul(STATE)
        .ok_or(Mamba2Error::Size("B/C"))?;
    let conv_dim = bc_n
        .checked_mul(2)
        .and_then(|n| g.d_inner.checked_add(n))
        .ok_or(Mamba2Error::Size("conv channels"))?;
    g.d_inner
        .checked_mul(STATE)
        .ok_or(Mamba2Error::Size("scan state"))?;
    g.d_conv
        .checked_mul(conv_dim)
        .ok_or(Mamba2Error::Size("conv weight"))?;
    Ok(Checked {
        geom: g,
        head_dim: g.d_inner / g.n_heads,
        conv_dim,
    })
}

fn len(what: &'static str, got: usize, need: usize) -> Result<(), Mamba2Error> {
    if got == need {
        Ok(())
    } else {
        Err(Mamba2Error::Len { what, got, need })
    }
}

/// One scan token. `state` is `d_inner * 128` floats, `n` the fastest index
/// inside `(head, channel)`. `x` and `y` are `d_inner` channels, channel
/// `c + head * head_dim`. `dt` and `a` are one float per head (`dt` already
/// includes its bias). `b` and `c` are `n_group * 128`, `n` fastest inside
/// the group. Writes `y` and the updated state.
#[allow(
    clippy::too_many_arguments,
    reason = "one scan step: geometry, state, x, dt, A, B, C, y"
)]
pub fn scan_token(
    g: &Checked,
    state: &mut [f32],
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    y: &mut [f32],
) -> Result<(), Mamba2Error> {
    let (inner, heads, groups, head_dim) =
        (g.geom.d_inner, g.geom.n_heads, g.geom.n_groups, g.head_dim);
    let state_n = inner * STATE;
    let bc_n = groups * STATE;
    len("state", state.len(), state_n)?;
    len("x", x.len(), inner)?;
    len("y", y.len(), inner)?;
    len("dt", dt.len(), heads)?;
    len("A", a.len(), heads)?;
    len("B", b.len(), bc_n)?;
    len("C", c.len(), bc_n)?;
    let per = heads / groups;
    for h in 0..heads {
        let dt_sp = scan_dt(dt[h]);
        let d_a = (dt_sp * a[h]).exp();
        let group = h / per;
        let b_base = group * STATE;
        for ch in 0..head_dim {
            let x_dt = x[h * head_dim + ch] * dt_sp;
            let s_base = (h * head_dim + ch) * STATE;
            let mut lane = [0.0f32; 32];
            for (l, slot) in lane.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for j in 0..4 {
                    let n = l + 32 * j;
                    let updated = state[s_base + n] * d_a + b[b_base + n] * x_dt;
                    state[s_base + n] = updated;
                    acc += updated * c[b_base + n];
                }
                *slot = acc;
            }
            // The card's scan sums the lanes in this order: the gate.
            y[h * head_dim + ch] = butterfly(&lane);
        }
    }
    Ok(())
}

/// `y[c] += x[c] * D[head]`, `D` one float per head.
pub fn skip_d(y: &mut [f32], x: &[f32], d: &[f32], head_dim: usize) -> Result<(), Mamba2Error> {
    len("x", x.len(), y.len())?;
    if head_dim == 0 || !y.len().is_multiple_of(head_dim) {
        return Err(Mamba2Error::Len {
            what: "D",
            got: d.len(),
            need: 0,
        });
    }
    let heads = y.len() / head_dim;
    len("D", d.len(), heads)?;
    for (h, &gain) in d.iter().enumerate() {
        for ch in 0..head_dim {
            let i = h * head_dim + ch;
            y[i] += x[i] * gain;
        }
    }
    Ok(())
}

/// `y = silu(z) * y`, the host SiLU ([`qdot::swiglu`]).
pub fn gate_z(y: &mut [f32], z: &[f32]) -> Result<(), Mamba2Error> {
    len("z", z.len(), y.len())?;
    let mut out = vec![0.0; y.len()];
    qdot::swiglu(z, y, &mut out);
    y.copy_from_slice(&out);
    Ok(())
}

/// RMSNorm over each group of `d_inner / n_group` channels. The gain is one
/// value per channel, the same reduction as [`ops::rms_norm`]: squares summed
/// in f64, the mean narrowed to f32, then `scale * gain * x`.
pub fn grouped_rms(
    y: &mut [f32],
    gain: &[f32],
    n_groups: usize,
    eps: f32,
) -> Result<(), Mamba2Error> {
    len("norm", gain.len(), y.len())?;
    if n_groups == 0 || !y.len().is_multiple_of(n_groups) {
        return Err(Mamba2Error::Groups {
            n_heads: y.len(),
            n_groups,
        });
    }
    let width = y.len() / n_groups;
    for g in 0..n_groups {
        let start = g * width;
        let col = Tensor2::from_vec(width, 1, y[start..start + width].to_vec());
        let out = ops::rms_norm(&col, &gain[start..start + width], eps);
        y[start..start + width].copy_from_slice(out.col(0));
    }
    Ok(())
}

/// Sample `time` of one channel: the conv state, then the new tokens.
fn conv_sample(
    state: &[f32],
    x_new: &[f32],
    channel: usize,
    channels: usize,
    ages: usize,
    time: usize,
) -> f32 {
    if time < ages {
        state[time + channel * ages]
    } else {
        let tok = time - ages;
        x_new[channel + tok * channels]
    }
}

/// Causal depthwise conv over `n_tok` tokens, then the bias add.
///
/// `state` is `(d_conv - 1) * channels`, age fastest inside the channel, and
/// becomes the last `d_conv - 1` samples. `x_new` and `y` are channel-fastest
/// inside the token (`channel + token * channels`). `w` is tap-fastest inside
/// the channel (`tap + channel * d_conv`); tap 0 multiplies the oldest sample.
/// `y` is the dot plus `bias`, before SiLU.
pub fn conv_tokens(
    d_conv: usize,
    state: &mut [f32],
    x_new: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
) -> Result<(), Mamba2Error> {
    if d_conv == 0 {
        return Err(Mamba2Error::Conv0);
    }
    let channels = bias.len();
    if channels == 0 || !x_new.len().is_multiple_of(channels) {
        return Err(Mamba2Error::Len {
            what: "conv x",
            got: x_new.len(),
            need: channels,
        });
    }
    let n_tok = x_new.len() / channels;
    let ages = d_conv - 1;
    let state_n = ages
        .checked_mul(channels)
        .ok_or(Mamba2Error::Size("conv state"))?;
    let weight_n = d_conv
        .checked_mul(channels)
        .ok_or(Mamba2Error::Size("conv weight"))?;
    len("conv state", state.len(), state_n)?;
    len("conv weight", w.len(), weight_n)?;
    len("conv y", y.len(), n_tok * channels)?;
    for tok in 0..n_tok {
        for channel in 0..channels {
            let mut sum = 0.0f32;
            let base = channel * d_conv;
            for tap in 0..d_conv {
                sum +=
                    conv_sample(state, x_new, channel, channels, ages, tok + tap) * w[base + tap];
            }
            y[channel + tok * channels] = sum + bias[channel];
        }
    }
    if ages > 0 {
        let mut next = vec![0.0; state.len()];
        for channel in 0..channels {
            for age in 0..ages {
                next[age + channel * ages] =
                    conv_sample(state, x_new, channel, channels, ages, n_tok + age);
            }
        }
        state.copy_from_slice(&next);
    }
    Ok(())
}

/// One token's inputs, already the projection split except `xbc`, which is
/// still one conv row (`d_inner` of `x`, then `B`, then `C`).
pub struct Token<'a> {
    /// Conv input, `conv_dim` values, channel fastest.
    pub xbc: &'a [f32],
    /// `n_head` raw `dt` values, before the bias.
    pub dt: &'a [f32],
    /// `n_head` `dt` biases.
    pub dt_bias: &'a [f32],
    /// `d_inner` gate values. Not convolved.
    pub z: &'a [f32],
    /// One `A` per head.
    pub a: &'a [f32],
    /// One `D` per head.
    pub d: &'a [f32],
    /// Conv weight, tap fastest inside the channel.
    pub conv_w: &'a [f32],
    /// Conv bias, one per conv channel. The add is unconditional.
    pub conv_bias: &'a [f32],
    /// Grouped RMSNorm gain, one per `d_inner` channel. Absent skips the norm.
    pub norm: Option<&'a [f32]>,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Conv, SiLU, scan, `D`, `z`, and the grouped norm, for one token.
/// `conv_state` is the conv window and `scan_state` the recurrent state.
/// Both update in place. `y` receives `d_inner` channels.
pub fn step_token(
    g: &Checked,
    conv_state: &mut [f32],
    scan_state: &mut [f32],
    tok: &Token<'_>,
    y: &mut [f32],
) -> Result<(), Mamba2Error> {
    let inner = g.geom.d_inner;
    let heads = g.geom.n_heads;
    let conv_dim = g.conv_dim;
    len("xbc", tok.xbc.len(), conv_dim)?;
    len("dt", tok.dt.len(), heads)?;
    len("dt bias", tok.dt_bias.len(), heads)?;
    len("z", tok.z.len(), inner)?;
    len("A", tok.a.len(), heads)?;
    len("D", tok.d.len(), heads)?;
    len("y", y.len(), inner)?;
    len("conv weight", tok.conv_w.len(), g.geom.d_conv * conv_dim)?;
    len("conv bias", tok.conv_bias.len(), conv_dim)?;
    if let Some(norm) = tok.norm {
        len("norm", norm.len(), inner)?;
    }
    len("scan state", scan_state.len(), inner * STATE)?;
    len(
        "conv state",
        conv_state.len(),
        g.geom.d_conv.saturating_sub(1) * conv_dim,
    )?;

    let mut conv_y = vec![0.0; conv_dim];
    conv_tokens(
        g.geom.d_conv,
        conv_state,
        tok.xbc,
        tok.conv_w,
        tok.conv_bias,
        &mut conv_y,
    )?;
    let ones = vec![1.0; conv_dim];
    let mut mixed = vec![0.0; conv_dim];
    qdot::swiglu(&conv_y, &ones, &mut mixed);

    let x = &mixed[..inner];
    let b = &mixed[inner..inner + g.geom.n_groups * STATE];
    let c = &mixed[inner + g.geom.n_groups * STATE..];
    let dt: Vec<f32> = tok.dt.iter().zip(tok.dt_bias).map(|(d, b)| d + b).collect();
    scan_token(g, scan_state, x, &dt, tok.a, b, c, y)?;
    skip_d(y, x, tok.d, g.head_dim)?;
    gate_z(y, tok.z)?;
    if let Some(norm) = tok.norm {
        grouped_rms(y, norm, g.geom.n_groups, tok.eps)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_adds_bias_after_the_dot() {
        let mut state = [16_777_216.0, -16_777_216.0, 0.0];
        let mut y = [0.0];
        conv_tokens(4, &mut state, &[0.0], &[1.0; 4], &[1.0], &mut y).unwrap();
        assert_eq!(y, [1.0]);
    }

    fn one_channel() -> Checked {
        check(Geom {
            d_inner: 1,
            n_heads: 1,
            n_groups: 1,
            d_state: STATE,
            d_conv: 4,
        })
        .unwrap()
    }

    /// A negative `A` decays the held state by `exp(dt_sp * A)`; a decay of 1
    /// leaves it, so the state words tell the two apart.
    #[test]
    fn a_negative_a_decays_the_state() {
        let g = one_channel();
        let mut state = vec![1.0; STATE];
        let mut y = [0.0];
        scan_token(
            &g,
            &mut state,
            &[0.0],
            &[21.0],
            &[-1.0],
            &[1.0; STATE],
            &[1.0; STATE],
            &mut y,
        )
        .unwrap();
        let d_a = (-21.0f32).exp();
        assert!(d_a < 1.0);
        assert!(state.iter().all(|&v| v == d_a));
    }

    /// Lanes 0, 1, 2 hold 2^24, -2^24 and 1. The xor reduction adds lane 2
    /// into lane 0 first, where 2^24 + 1 rounds to 2^24, so `y` is 0; a
    /// sequential sum would give 1.
    #[test]
    fn y_sums_the_lanes_in_xor_order() {
        let g = one_channel();
        let mut state = vec![0.0; STATE];
        state[0] = 16_777_216.0;
        state[1] = -16_777_216.0;
        state[2] = 1.0;
        let mut y = [0.0];
        scan_token(
            &g,
            &mut state,
            &[0.0],
            &[21.0],
            &[0.0],
            &[1.0; STATE],
            &[1.0; STATE],
            &mut y,
        )
        .unwrap();
        assert_eq!(y[0], 0.0);
    }

    /// One composed token at two channels against an f64 reading of the same
    /// stages: the conv (only the current tap), its bias, SiLU, the `dt` bias
    /// and softplus, the scan from a zero state, `+ x*D`, `silu(z)*y`, and the
    /// grouped RMSNorm with a gain.
    #[test]
    fn a_composed_token_wires_every_stage() {
        // Two heads of one channel with their own `dt`, so a scan term that
        // moved by one common factor would not cancel in the norm.
        let g = check(Geom {
            d_inner: 2,
            n_heads: 2,
            n_groups: 1,
            d_state: STATE,
            d_conv: 4,
        })
        .unwrap();
        let conv_dim = g.conv_dim;
        let mut xbc = vec![0.0f32; conv_dim];
        xbc[0] = 0.5;
        xbc[1] = -1.0;
        for v in &mut xbc[2..2 + STATE] {
            *v = 0.25;
        }
        for v in &mut xbc[2 + STATE..] {
            *v = 0.5;
        }
        let mut conv_w = vec![0.0f32; 4 * conv_dim];
        for ch in 0..conv_dim {
            conv_w[ch * 4 + 3] = 1.0;
        }
        let conv_bias = vec![0.1f32; conv_dim];
        let (z, norm) = ([0.8f32, -0.4], [1.0f32, 2.0]);
        let tok = Token {
            xbc: &xbc,
            dt: &[0.3, 0.1],
            dt_bias: &[0.2, 0.6],
            z: &z,
            a: &[-0.7, -0.7],
            d: &[1.5, 1.5],
            conv_w: &conv_w,
            conv_bias: &conv_bias,
            norm: Some(&norm),
            eps: 1e-5,
        };
        let mut conv_state = vec![0.0; 3 * conv_dim];
        let mut scan_state = vec![0.0; 2 * STATE];
        let mut y = [0.0f32; 2];
        step_token(&g, &mut conv_state, &mut scan_state, &tok, &mut y).unwrap();

        let silu = |v: f64| v / (1.0 + (-v).exp());
        let (bm, cm) = (silu(0.35), silu(0.6));
        let mut want = [0.0f64; 2];
        for (ch, w) in want.iter_mut().enumerate() {
            let x = silu(f64::from(xbc[ch]) + 0.1);
            let dt_sp = (1.0 + [0.5f64, 0.7][ch].exp()).ln();
            let scan = STATE as f64 * bm * x * dt_sp * cm;
            *w = (scan + x * 1.5) * silu(f64::from(z[ch]));
        }
        let rms = ((want[0] * want[0] + want[1] * want[1]) / 2.0 + 1e-5).sqrt();
        for (ch, w) in want.iter_mut().enumerate() {
            *w = *w / rms * f64::from(norm[ch]);
        }
        for ch in 0..2 {
            let rel = (f64::from(y[ch]) - want[ch]).abs() / want[ch].abs();
            assert!(rel < 1e-5, "channel {ch}: {} vs {}", y[ch], want[ch]);
        }
    }

    /// The update is two rounded products then an add, never a fused
    /// multiply-add on either side.
    #[test]
    fn the_recurrence_rounds_its_product_before_the_add() {
        // Each case cancels the state to 0 only when its product is rounded
        // before the add: the first fuses `B * (x * dt)`, the second the
        // decay `state * dA`.
        let g = one_channel();
        for (initial, a, b) in [
            (-21.0, 0.0, 1.0 + f32::EPSILON),
            (21.0, -f32::EPSILON / 21.0, -1.0),
        ] {
            let mut state = vec![initial; STATE];
            let mut y = [0.0];
            scan_token(
                &g,
                &mut state,
                &[1.0 - f32::EPSILON],
                &[21.0],
                &[a],
                &[b; STATE],
                &[1.0; STATE],
                &mut y,
            )
            .unwrap();
            assert!(state.iter().all(|&v| v == 0.0), "a {a}: {:?}", &state[..4]);
        }
    }

    #[test]
    fn overflowing_sizes_are_refused() {
        assert_eq!(
            conv_tokens(
                usize::MAX / 2 + 2,
                &mut [],
                &[1.0; 2],
                &[1.0; 2],
                &[0.0; 2],
                &mut [0.0; 2]
            ),
            Err(Mamba2Error::Size("conv state"))
        );
        assert_eq!(
            check(Geom {
                d_inner: usize::MAX - 255,
                n_heads: 1,
                n_groups: 1,
                d_state: STATE,
                d_conv: 4,
            }),
            Err(Mamba2Error::Size("conv channels"))
        );
    }

    fn nano() -> Checked {
        check(Geom {
            d_inner: 7680,
            n_heads: 96,
            n_groups: 8,
            d_state: STATE,
            d_conv: 4,
        })
        .unwrap()
    }

    #[test]
    fn widths_of_the_4b_file() {
        let g = nano();
        assert_eq!(g.head_dim(), 80);
        assert_eq!(g.conv_dim(), 9728);
    }

    #[test]
    fn state_96_and_256_are_refused_by_name() {
        for (n, needle) in [(96, "state 96"), (256, "state 256")] {
            let err = check(Geom {
                d_inner: 7680,
                n_heads: 96,
                n_groups: 8,
                d_state: n,
                d_conv: 4,
            })
            .unwrap_err();
            let text = err.to_string();
            assert!(text.contains(needle), "{text}");
            assert!(text.contains("writes nothing"), "{text}");
        }
    }

    #[test]
    fn a_remainder_group_is_refused() {
        let err = check(Geom {
            d_inner: 6,
            n_heads: 3,
            n_groups: 2,
            d_state: STATE,
            d_conv: 4,
        })
        .unwrap_err();
        assert!(err.to_string().contains("not a multiple"), "{err}");
    }

    /// `dt > 20` stays itself. `dt <= 20` is `log1p(exp(dt))`. In f32 that
    /// value at 20 is 20, so the arms meet there; at 1 they do not.
    // PIN(2026-10-15): dt_20_uses_the_log1p_arm removed: at dt = 20 both softplus arms give the same f32 bits;
    // this test owns the boundary and uniform_row_at_the_4b_widths the uniform recurrence.
    #[test]
    fn dt_21_is_identity_and_dt_20_is_log1p_exp() {
        assert_eq!(scan_dt(21.0), 21.0);
        assert_eq!(scan_dt(20.0), 20.0f32.exp().ln_1p());
        assert_eq!(scan_dt(20.0), 20.0);
        assert_ne!(scan_dt(1.0), 1.0);
        assert_eq!(scan_dt(1.0), 1.0f32.exp().ln_1p());
    }

    /// `A = 0`, `dt = 21`, `x = B = C = 1`, state 0. After `t` tokens every
    /// state element is `21 * t` and each channel's `y` is `128 * 21 * t`.
    /// That 2688 is the state sum, not an embedding width: `y` has 7680
    /// channels here.
    #[test]
    fn uniform_row_at_the_4b_widths() {
        let g = nano();
        let inner = 7680;
        let mut state = vec![0.0; inner * STATE];
        let x = vec![1.0; inner];
        let dt = vec![21.0; 96];
        let a = vec![0.0; 96];
        let b = vec![1.0; 8 * STATE];
        let c = vec![1.0; 8 * STATE];
        let mut y = vec![0.0; inner];
        scan_token(&g, &mut state, &x, &dt, &a, &b, &c, &mut y).unwrap();
        assert_eq!(y.len(), 7680);
        assert!(y.iter().all(|&v| v == 2688.0));
        assert!(state.iter().all(|&v| v == 21.0));
        scan_token(&g, &mut state, &x, &dt, &a, &b, &c, &mut y).unwrap();
        assert!(y.iter().all(|&v| v == 5376.0));
        assert!(state.iter().all(|&v| v == 42.0));
    }

    /// Heads 0..3 share group 0 when eight heads form two groups. Head 2
    /// would read group 1 if the index were `head / n_group`.
    #[test]
    fn group_offset_divides_heads_per_group_first() {
        let g = check(Geom {
            d_inner: 8,
            n_heads: 8,
            n_groups: 2,
            d_state: STATE,
            d_conv: 4,
        })
        .unwrap();
        let mut b = vec![0.0; 256];
        b[0] = 1.0;
        let mut c = vec![0.0; 256];
        c[0] = 1.0;
        let mut state = vec![0.0; 8 * STATE];
        let mut y = vec![0.0; 8];
        scan_token(
            &g, &mut state, &[1.0; 8], &[21.0; 8], &[0.0; 8], &b, &c, &mut y,
        )
        .unwrap();
        assert_eq!(&y[..4], &[21.0, 21.0, 21.0, 21.0]);
        assert_eq!(&y[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn conv_tap_0_is_the_oldest_sample() {
        // Two channels, one token. Channel 0's newest tap is 1, so y is the
        // token. Channel 1's oldest tap is 1, so y is the oldest state.
        let mut state = vec![1.0, 2.0, 3.0, 9.0, 9.0, 9.0];
        let w = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut y = [0.0; 2];
        conv_tokens(4, &mut state, &[4.0, 8.0], &w, &[0.0, 0.0], &mut y).unwrap();
        assert_eq!(y, [4.0, 9.0]);
        assert_eq!(state, vec![2.0, 3.0, 4.0, 9.0, 9.0, 8.0]);
    }

    #[test]
    fn conv_slides_across_two_tokens() {
        let mut state = vec![10.0, 20.0, 30.0];
        let w = vec![0.0, 0.0, 0.0, 1.0];
        let mut y = [0.0; 2];
        conv_tokens(4, &mut state, &[40.0, 50.0], &w, &[0.0], &mut y).unwrap();
        assert_eq!(y, [40.0, 50.0]);
        assert_eq!(state, vec![30.0, 40.0, 50.0]);
    }

    #[test]
    fn d_is_one_float_per_head() {
        let mut y = [0.0; 4];
        skip_d(&mut y, &[1.0, 1.0, 1.0, 1.0], &[10.0, 0.0], 2).unwrap();
        assert_eq!(y, [10.0, 10.0, 0.0, 0.0]);
    }

    #[test]
    fn grouped_rms_norms_each_group() {
        // Two groups of two. A single norm over all four would use mean 9/4,
        // not 9/2, so this pins the group width. Epsilon keeps the all-zero
        // group at 0; epsilon 0 would make that scale infinite.
        let mut y = [3.0, 0.0, 0.0, 0.0];
        let eps = 1.0e-5;
        grouped_rms(&mut y, &[1.0; 4], 2, eps).unwrap();
        let scale = 1.0 / (((9.0f64 / 2.0) as f32) + eps).sqrt();
        assert_eq!(y[0], scale * 3.0);
        assert_eq!(&y[1..], &[0.0, 0.0, 0.0]);
    }
}
