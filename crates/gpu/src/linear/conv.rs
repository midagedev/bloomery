//! The conv and prep of a linear-attention layer, one launch per call over
//! `m` tokens: the depthwise causal conv of every channel over its last
//! [`CONV_TAPS`] inputs, SiLU, the L2 norm of each query and key head (the
//! query also scaled by [`Q_SCALE`]), β and the decay of each value head, and
//! the conv state after the call's last token.
//!
//! Geometry: one warp per (token, conv head), four warps per block, no
//! shared memory and no barrier. Lane `l` owns channels `l + 32·i`
//! (`i = 0..4`) of its head. The tokens are independent: a token's window is
//! the call's input where it has one and the incoming state before it. The
//! warps of the call's last token write the new state; a value head's lane 0
//! computes that head's β and decay for its token.
//!
//! Numeric contract (the host rule [`conv_prep_host`] is this list):
//! - conv: `acc = x₀·w₀` rounded, then `acc = fma(xⱼ, wⱼ, acc)` for taps
//!   1, 2, 3 — oldest input first, tap j of channel c the weight at `4c + j`;
//! - [`silu`] of `acc`;
//! - L2 norm of a query or key head: each value squared in f32, lane `l`'s
//!   four squares summed in f64 in `i` order, the 32 lanes by the xor
//!   butterfly (16, 8, 4, 2, 1) in f64, the sum rounded to f32; `scale =
//!   1 / max(√sum, eps)` in f32 (ggml's `l2_norm`), then `value·scale`, and
//!   for a query that times [`Q_SCALE`];
//! - β = [`sigmoid`]`(b)`; decay = [`expf_ik`]`(`[`softplus`]`(a + dt)·ssm_a)`.
//!
//! No silent failure: a non-finite input, a conv sum that overflows, a norm
//! sum that is not finite as f32 (the square of a finite value can be), or a
//! β or decay that is not finite raises [`FaultSite::LinearConv`], and the
//! value written there is not finite. `max(NaN, eps)` would otherwise turn
//! an overflowed norm into a zero head.

use super::{BLOCK, CONV_ROWS, CONV_TAPS, HEAD, LinearShape, Q_SCALE};
use super::{expf_ik, sigmoid, silu, softplus};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

const _: () = assert!(HEAD == 128 && CONV_TAPS == 4 && BLOCK == 128);

#[cuda_module]
mod conv_kernels {
    use super::*;

    /// The conv and prep of `m` tokens (module doc). Warp `w = 4·block +
    /// warp` is token `w / H` and conv head `w % H`, `H = 2·n_k + n_v`: heads
    /// below `n_k` are queries, below `2·n_k` keys, the rest values.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            x.len() >= m * (2 * n_k + n_v) * 128,
            state_in.len() >= 3 * (2 * n_k + n_v) * 128,
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            b_raw.len() >= m * n_v,
            a_raw.len() >= m * n_v,
            dt_bias.len() >= n_v,
            ssm_a.len() >= n_v,
            y.len() >= m * (2 * n_k + n_v) * 128,
            state_out.len() >= 3 * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v
        )
    )]
    pub fn gdn_conv_prep(
        x: &[f32],
        state_in: &[f32],
        w: &[f32],
        b_raw: &[f32],
        a_raw: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        eps: f32,
        n_k: u32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut state_out: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
    ) {
        let heads = (2 * n_k + n_v) as usize;
        let ch = heads * HEAD;
        let tid = thread::threadIdx_x() as usize;
        let wi = thread::blockIdx_x() as usize * 4 + tid / 32;
        let m = m as usize;
        if wi >= m * heads {
            return; // warp-uniform
        }
        let lane = tid % 32;
        let t = wi / heads;
        let hp = wi - t * heads;
        let n_k = n_k as usize;
        let last = t + 1 == m;

        let mut out = [0.0f32; 4];
        let mut sumsq = 0.0f64;
        let mut ok = true;
        let mut i = 0usize;
        #[unroll]
        while i < 4 {
            let c = hp * HEAD + lane + 32 * i;
            // Tap j is input t − 3 + j: the call's row where it has one, else
            // row t + j of the incoming state.
            let mut xs = [0.0f32; 4];
            let mut j = 0usize;
            #[unroll]
            while j < CONV_TAPS {
                // SAFETY: c < ch; for t + j >= 3 the row t + j − 3 < m, inside
                // x (m·ch values by the contract); else t + j < 3, a row of
                // state_in (3·ch values).
                xs[j] = unsafe {
                    if t + j >= CONV_ROWS {
                        *x.get_unchecked((t + j - CONV_ROWS) * ch + c)
                    } else {
                        *state_in.get_unchecked((t + j) * ch + c)
                    }
                };
                j += 1;
            }
            // SAFETY: 4c + 3 < 4·ch <= w.len() by the contract.
            let wv = unsafe {
                [
                    *w.get_unchecked(4 * c),
                    *w.get_unchecked(4 * c + 1),
                    *w.get_unchecked(4 * c + 2),
                    *w.get_unchecked(4 * c + 3),
                ]
            };
            let mut acc = mul_rn_f32(xs[0], wv[0]);
            acc = fma_rn_f32(xs[1], wv[1], acc);
            acc = fma_rn_f32(xs[2], wv[2], acc);
            acc = fma_rn_f32(xs[3], wv[3], acc);
            ok &= crate::fault::quad_finite(xs) & acc.is_finite();
            let s = silu(acc);
            out[i] = s;
            sumsq += f64::from(mul_rn_f32(s, s));
            if last {
                // SAFETY: rows 0..3 of state_out hold 3·ch values; c < ch.
                // Each channel's rows are written by this one lane.
                unsafe {
                    *state_out.get_unchecked_mut(c) = xs[1];
                    *state_out.get_unchecked_mut(ch + c) = xs[2];
                    *state_out.get_unchecked_mut(2 * ch + c) = xs[3];
                }
            }
            i += 1;
        }

        if hp < 2 * n_k {
            let mut acc = sumsq;
            acc += warp::shuffle_xor_f64(acc, 16);
            acc += warp::shuffle_xor_f64(acc, 8);
            acc += warp::shuffle_xor_f64(acc, 4);
            acc += warp::shuffle_xor_f64(acc, 2);
            acc += warp::shuffle_xor_f64(acc, 1);
            let sum = acc as f32;
            let fin = sum.is_finite();
            ok &= fin;
            let scale = if fin {
                1.0 / sum.sqrt().max(eps)
            } else {
                f32::NAN
            };
            let mut i = 0usize;
            #[unroll]
            while i < 4 {
                out[i] = mul_rn_f32(out[i], scale);
                if hp < n_k {
                    out[i] = mul_rn_f32(out[i], Q_SCALE);
                }
                i += 1;
            }
        }
        let mut i = 0usize;
        #[unroll]
        while i < 4 {
            // SAFETY: t < m and c < ch, inside y (m·ch values by the
            // contract); one lane per channel.
            unsafe { *y.get_unchecked_mut(t * ch + hp * HEAD + lane + 32 * i) = out[i] };
            i += 1;
        }

        if hp >= 2 * n_k && lane == 0 {
            let h = hp - 2 * n_k;
            let at = t * n_v as usize + h;
            // SAFETY: h < n_v and t < m: `at` < m·n_v <= b_raw, a_raw, beta,
            // decay lengths; h < n_v <= dt_bias, ssm_a lengths.
            let (b, a, dt, sa) = unsafe {
                (
                    *b_raw.get_unchecked(at),
                    *a_raw.get_unchecked(at),
                    *dt_bias.get_unchecked(h),
                    *ssm_a.get_unchecked(h),
                )
            };
            let bt = sigmoid(b);
            let dc = expf_ik(mul_rn_f32(softplus(a + dt), sa));
            let fin = crate::fault::quad_finite([b, a, dt, sa]) & bt.is_finite() & dc.is_finite();
            ok &= fin;
            // SAFETY: as the reads above; one lane per (token, head).
            unsafe {
                *beta.get_unchecked_mut(at) = if fin { bt } else { f32::NAN };
                *decay.get_unchecked_mut(at) = if fin { dc } else { f32::NAN };
            }
        }
        if !ok {
            fault.raise(FaultSite::LinearConv);
        }
    }
}

/// [`ConvKernels::enqueue_conv_prep`]'s arguments: `m` tokens of the
/// projected channels `x` (`[m][C]`), `b_raw` and `a_raw` (`[m][n_v]`, the β
/// and α projections), the layer's conv taps `w` (`[C][CONV_TAPS]`, oldest
/// first), `dt_bias` and `ssm_a` (`[n_v]`), the incoming conv state, and the
/// buffers the call writes: the conv output `y` (`[m][C]`), β and decay
/// (`[m][n_v]`), the conv state after the last token.
pub struct ConvArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub b_raw: &'a DeviceBuffer<f32>,
    pub a_raw: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub dt_bias: &'a DeviceBuffer<f32>,
    pub ssm_a: &'a DeviceBuffer<f32>,
    pub state_in: &'a DeviceBuffer<f32>,
    pub shape: LinearShape,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub decay: &'a mut DeviceBuffer<f32>,
    pub state_out: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ConvKernels {
    module: conv_kernels::LoadedModule,
}

impl ConvKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ConvKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { conv_kernels::load(ctx)? };
        Ok(ConvKernels { module })
    }

    /// Enqueue the conv and prep of `args.m` tokens: `⌈m·H / 4⌉` blocks of
    /// 128 threads, `H = 2·n_k + n_v`. The incoming and outgoing conv states
    /// are distinct buffers. Asynchronous, allocation-free, capturable.
    pub fn enqueue_conv_prep(
        &self,
        stream: &CudaStream,
        args: ConvArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_conv_prep";
        let ConvArgs {
            x,
            b_raw,
            a_raw,
            w,
            dt_bias,
            ssm_a,
            state_in,
            shape,
            eps,
            m,
            fault,
            y,
            beta,
            decay,
            state_out,
        } = args;
        shape.check(what)?;
        if m == 0 {
            return Err(GpuError::shape(what, "need m >= 1, got m=0".to_owned()));
        }
        let ch = shape.channels();
        let nv = shape.n_v;
        let lens = [
            ("x", x.len(), m * ch),
            ("b_raw", b_raw.len(), m * nv),
            ("a_raw", a_raw.len(), m * nv),
            ("w", w.len(), CONV_TAPS * ch),
            ("dt_bias", dt_bias.len(), nv),
            ("ssm_a", ssm_a.len(), nv),
            ("state_in", state_in.len(), CONV_ROWS * ch),
            ("y", y.len(), m * ch),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), m * nv),
            ("state_out", state_out.len(), CONV_ROWS * ch),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", (m * shape.conv_heads()).div_ceil(4))?;
        let n_k = launch_u32(what, "n_k", shape.n_k)?;
        let n_v = launch_u32(what, "n_v", nv)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_gdn_conv_prep(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.gdn_conv_prep(
            stream, &prep, x, state_in, w, b_raw, a_raw, dt_bias, ssm_a, eps, n_k, n_v, m, fault,
            y, state_out, beta, decay,
        )?;
        Ok(())
    }
}

/// What [`conv_prep_host`] computes: the conv output, β, decay and the conv
/// state after the last token, in the kernel's layouts.
pub struct ConvOut {
    pub y: Vec<f32>,
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub state: Vec<f32>,
}

/// The host rule of [`ConvKernels::enqueue_conv_prep`]: the module doc's
/// numeric contract, op for op. Inputs as [`ConvArgs`] names them, host
/// slices of the same lengths.
///
/// # Panics
///
/// When a slice is shorter than [`ConvArgs`]'s lengths for `shape` and `m`.
#[allow(
    clippy::too_many_arguments,
    reason = "the host twin of one launch: its inputs, each named as the launch names them"
)]
#[must_use]
pub fn conv_prep_host(
    x: &[f32],
    b_raw: &[f32],
    a_raw: &[f32],
    w: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    state_in: &[f32],
    shape: LinearShape,
    eps: f32,
    m: usize,
) -> ConvOut {
    let ch = shape.channels();
    let heads = shape.conv_heads();
    let nv = shape.n_v;
    let mut y = vec![0.0f32; m * ch];
    let mut state = vec![0.0f32; CONV_ROWS * ch];
    let mut beta = vec![0.0f32; m * nv];
    let mut decay = vec![0.0f32; m * nv];
    for t in 0..m {
        for hp in 0..heads {
            let mut vals = [0.0f32; HEAD];
            let mut lanes = [0.0f64; 32];
            for (lane, sum) in lanes.iter_mut().enumerate() {
                for i in 0..4 {
                    let c = hp * HEAD + lane + 32 * i;
                    let xs: [f32; 4] = std::array::from_fn(|j| {
                        if t + j >= CONV_ROWS {
                            x[(t + j - CONV_ROWS) * ch + c]
                        } else {
                            state_in[(t + j) * ch + c]
                        }
                    });
                    let mut acc = xs[0] * w[4 * c];
                    for j in 1..CONV_TAPS {
                        acc = xs[j].mul_add(w[4 * c + j], acc);
                    }
                    let s = silu(acc);
                    vals[lane + 32 * i] = s;
                    *sum += f64::from(s * s);
                    if t + 1 == m {
                        for r in 0..CONV_ROWS {
                            state[r * ch + c] = xs[r + 1];
                        }
                    }
                }
            }
            if hp < 2 * shape.n_k {
                let sum = super::butterfly_f64(&lanes) as f32;
                let scale = if sum.is_finite() {
                    1.0 / sum.sqrt().max(eps)
                } else {
                    f32::NAN
                };
                for v in &mut vals {
                    *v *= scale;
                    if hp < shape.n_k {
                        *v *= Q_SCALE;
                    }
                }
            }
            y[t * ch + hp * HEAD..t * ch + (hp + 1) * HEAD].copy_from_slice(&vals);
        }
        for h in 0..nv {
            let at = t * nv + h;
            let (b, a, dt, sa) = (b_raw[at], a_raw[at], dt_bias[h], ssm_a[h]);
            let bt = sigmoid(b);
            let dc = expf_ik(softplus(a + dt) * sa);
            let fin = [b, a, dt, sa, bt, dc].iter().all(|v| v.is_finite());
            beta[at] = if fin { bt } else { f32::NAN };
            decay[at] = if fin { dc } else { f32::NAN };
        }
    }
    ConvOut {
        y,
        beta,
        decay,
        state,
    }
}
