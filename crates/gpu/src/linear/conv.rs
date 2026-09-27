//! The conv and prep of a linear-attention layer, one launch per call over
//! `m` tokens: the depthwise causal conv of every channel over its last
//! [`CONV_TAPS`] inputs, SiLU, the L2 norm of each query and key head (the
//! query also scaled by [`Q_SCALE`]), β and the decay of each value head, and
//! the conv ring's new slots.
//!
//! Positions: token `t` is position `pos[t]`, the words the embedding launch
//! writes, `pos[t] = pos[0] + t`. Its predecessor at distance `d` (1 to
//! [`CONV_ROWS`]) is the call's row `t − d` when `d <= t`; otherwise it lies
//! before the call: zero when `pos[t] < d` (the sequence start, whatever the
//! ring holds), else ring slot `(pos[t] − d) mod RING_ROWS`. The call writes
//! the input of each of its last `min(m, RING_ROWS)` tokens into slot
//! `pos[t] mod RING_ROWS`, so the ring holds the inputs of the last
//! [`RING_ROWS`] positions and a later call from any of the last
//! [`super::PASS_ROWS`] + 1 positions reads its predecessors without a copy.
//! A token whose position is not `pos[0] + t` has no defined predecessors: it
//! raises [`FaultSite::LinearConv`] and its output is NaN.
//!
//! Geometry: one warp per (unit, conv head), four warps per block, no shared
//! memory and no barrier. Lane `l` owns channels `l + 32·i` (`i = 0..4`) of
//! its head. Unit 0 is tokens `0 .. min(m, 3)` — every token that reads the
//! ring — and afterwards writes the ring; unit `u >= 1` is token `u + 2`,
//! whose predecessors are all the call's rows. The ring is touched by unit 0
//! alone, each channel by one lane, its reads before its writes. A value
//! head's lane 0 computes that head's β and decay for its token.
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
//! sum that is not finite as f32 (the square of a finite value can be), a β
//! or decay that is not finite, or a position that is not `pos[0] + t`
//! raises [`FaultSite::LinearConv`], and the value written there is not
//! finite. `max(NaN, eps)` would otherwise turn an overflowed norm into a
//! zero head.

use super::{BLOCK, CONV_ROWS, CONV_TAPS, HEAD, LinearShape, Q_SCALE, RING_ROWS};
use super::{expf_ik, sigmoid, silu, softplus};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

const _: () = assert!(HEAD == 128 && CONV_TAPS == 4 && BLOCK == 128 && RING_ROWS == 11);

/// The warps of a call of `m` tokens per conv head: unit 0 takes the first
/// `min(m, CONV_ROWS)` tokens, every later token a unit of its own.
#[inline(always)]
#[must_use]
pub fn units(m: usize) -> usize {
    m.max(CONV_ROWS) - (CONV_ROWS - 1)
}

/// The inputs one warp reads.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    x: &'a [f32],
    w: &'a [f32],
    b_raw: &'a [f32],
    a_raw: &'a [f32],
    dt_bias: &'a [f32],
    ssm_a: &'a [f32],
    eps: f32,
    n_k: usize,
    n_v: usize,
}

/// Where one warp writes: the call's outputs.
#[derive(Clone, Copy)]
struct Outputs {
    y: *mut f32,
    beta: *mut f32,
    decay: *mut f32,
}

/// Token `t` (position `p`, `contiguous` whether `p = pos[0] + t`) of conv
/// head `hp` on this lane's four channels: the conv, the norm, `y`, and for
/// a value head's lane 0 β and decay. Whether every value stayed finite.
///
/// # Safety
///
/// `t < m`, `hp < 2·n_k + n_v`, `lane < 32`; the slices and `out` hold the
/// lengths `gdn_conv_prep`'s launch contract names; `ring` addresses
/// `RING_ROWS·C` f32s that no other warp touches while this one runs (it is
/// read only when `t < CONV_ROWS`, unit 0's tokens); the (t, hp) outputs are
/// this warp's own. Called by all 32 lanes of the warp together.
#[allow(
    clippy::too_many_arguments,
    reason = "one token's coordinates and the warp's inputs and outputs, forwarded from the entry"
)]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn conv_token(
    t: usize,
    p: usize,
    contiguous: bool,
    hp: usize,
    lane: usize,
    inp: Inputs<'_>,
    ring: *const f32,
    out: Outputs,
) -> bool {
    let heads = 2 * inp.n_k + inp.n_v;
    let ch = heads * HEAD;
    let mut vals = [0.0f32; 4];
    let mut sumsq = 0.0f64;
    let mut ok = contiguous;
    for i in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        let c = hp * HEAD + lane + 32 * i;
        // Tap j reads the predecessor at distance d = CONV_ROWS − j.
        let mut xs = [0.0f32; 4];
        for j in 0..CONV_TAPS {
            cuda_device::thread::__unroll_config::<0>();
            let d = CONV_ROWS - j;
            // SAFETY: c < ch; for d <= t the row t − d < m, inside x (m·ch
            // values by the contract); else the slot (p − d) mod RING_ROWS
            // (p >= d) is a row of the ring (RING_ROWS·ch values).
            xs[j] = unsafe {
                if d <= t {
                    *inp.x.get_unchecked((t - d) * ch + c)
                } else if p < d {
                    0.0
                } else {
                    *ring.add((p + RING_ROWS - d) % RING_ROWS * ch + c)
                }
            };
        }
        // SAFETY: 4c + 3 < 4·ch <= w.len() by the contract.
        let wv = unsafe {
            [
                *inp.w.get_unchecked(4 * c),
                *inp.w.get_unchecked(4 * c + 1),
                *inp.w.get_unchecked(4 * c + 2),
                *inp.w.get_unchecked(4 * c + 3),
            ]
        };
        let mut acc = mul_rn_f32(xs[0], wv[0]);
        acc = fma_rn_f32(xs[1], wv[1], acc);
        acc = fma_rn_f32(xs[2], wv[2], acc);
        acc = fma_rn_f32(xs[3], wv[3], acc);
        ok &= crate::fault::quad_finite(xs) & acc.is_finite();
        let s = silu(acc);
        vals[i] = s;
        sumsq += f64::from(mul_rn_f32(s, s));
    }

    if hp < 2 * inp.n_k {
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
            1.0 / sum.sqrt().max(inp.eps)
        } else {
            f32::NAN
        };
        for i in 0usize..4 {
            cuda_device::thread::__unroll_config::<0>();
            vals[i] = mul_rn_f32(vals[i], scale);
            if hp < inp.n_k {
                vals[i] = mul_rn_f32(vals[i], Q_SCALE);
            }
        }
    }
    for i in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        let v = if contiguous { vals[i] } else { f32::NAN };
        // SAFETY: t < m and c < ch, inside y (m·ch values by the contract);
        // one lane per channel.
        unsafe { *out.y.add(t * ch + hp * HEAD + lane + 32 * i) = v };
    }

    if hp >= 2 * inp.n_k && lane == 0 {
        let h = hp - 2 * inp.n_k;
        let at = t * inp.n_v + h;
        // SAFETY: h < n_v and t < m: `at` < m·n_v <= b_raw, a_raw, beta,
        // decay lengths; h < n_v <= dt_bias, ssm_a lengths.
        let (b, a, dt, sa) = unsafe {
            (
                *inp.b_raw.get_unchecked(at),
                *inp.a_raw.get_unchecked(at),
                *inp.dt_bias.get_unchecked(h),
                *inp.ssm_a.get_unchecked(h),
            )
        };
        let bt = sigmoid(b);
        let dc = expf_ik(mul_rn_f32(softplus(a + dt), sa));
        let fin = crate::fault::quad_finite([b, a, dt, sa]) & bt.is_finite() & dc.is_finite();
        ok &= fin;
        // SAFETY: as the reads above; one lane per (token, head).
        unsafe {
            *out.beta.add(at) = if fin { bt } else { f32::NAN };
            *out.decay.add(at) = if fin { dc } else { f32::NAN };
        }
    }
    ok
}

#[cuda_module]
mod conv_kernels {
    use super::*;

    /// The conv and prep of `m` tokens (module doc). Warp `w = 4·block +
    /// warp` is unit `w / H` ([`units`]) and conv head `w % H`, `H = 2·n_k +
    /// n_v`: heads below `n_k` are queries, below `2·n_k` keys, the rest
    /// values.
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
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            b_raw.len() >= m * n_v,
            a_raw.len() >= m * n_v,
            dt_bias.len() >= n_v,
            ssm_a.len() >= n_v,
            pos.len() >= m,
            m >= 1,
            y.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v,
            ring.len() >= 11 * (2 * n_k + n_v) * 128
        )
    )]
    pub fn gdn_conv_prep(
        x: &[f32],
        w: &[f32],
        b_raw: &[f32],
        a_raw: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        pos: &[u32],
        eps: f32,
        n_k: u32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        let heads = (2 * n_k + n_v) as usize;
        let ch = heads * HEAD;
        let tid = thread::threadIdx_x() as usize;
        let wi = thread::blockIdx_x() as usize * 4 + tid / 32;
        let m = m as usize;
        if wi >= units(m) * heads {
            return; // warp-uniform
        }
        let lane = tid % 32;
        let u = wi / heads;
        let hp = wi - u * heads;
        let inp = Inputs {
            x,
            w,
            b_raw,
            a_raw,
            dt_bias,
            ssm_a,
            eps,
            n_k: n_k as usize,
            n_v: n_v as usize,
        };
        let out = Outputs {
            y: y.as_mut_ptr(),
            beta: beta.as_mut_ptr(),
            decay: decay.as_mut_ptr(),
        };
        let ring = ring.as_mut_ptr();
        let (t0, t1) = if u == 0 {
            (0, m.min(CONV_ROWS))
        } else {
            (u + CONV_ROWS - 1, u + CONV_ROWS)
        };
        // SAFETY: pos holds m >= 1 words by the contract.
        let p0 = unsafe { *pos.get_unchecked(0) } as u64;
        let mut ok = true;
        let mut t = t0;
        while t < t1 {
            // SAFETY: t < m <= pos.len().
            let p = unsafe { *pos.get_unchecked(t) };
            // SAFETY: t < m and hp < heads by the guard, lane < 32; the
            // contract's lengths; the ring is read only by unit 0, whose
            // tokens these are when t < CONV_ROWS, and written below by the
            // same lanes after these reads; (t, hp) is this warp's own.
            ok &= unsafe {
                conv_token(
                    t,
                    p as usize,
                    u64::from(p) == p0 + t as u64,
                    hp,
                    lane,
                    inp,
                    ring,
                    out,
                )
            };
            t += 1;
        }
        if u == 0 {
            // The inputs of the last min(m, RING_ROWS) tokens into their
            // slots, after every read of the ring above.
            let mut t = m.saturating_sub(RING_ROWS);
            while t < m {
                // SAFETY: t < m <= pos.len().
                let slot = unsafe { *pos.get_unchecked(t) } as usize % RING_ROWS;
                #[unroll]
                for i in 0usize..4 {
                    let c = hp * HEAD + lane + 32 * i;
                    // SAFETY: slot < RING_ROWS and c < ch: inside the ring
                    // (RING_ROWS·ch values) and row t of x (m·ch values);
                    // channel c of the ring is this lane's alone.
                    unsafe { *ring.add(slot * ch + c) = *x.get_unchecked(t * ch + c) };
                }
                t += 1;
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
/// first), `dt_bias` and `ssm_a` (`[n_v]`), the tokens' positions `pos`
/// (`[m]`, the words the rope reads), the buffers the call writes: the conv
/// output `y` (`[m][C]`), β and decay (`[m][n_v]`), and the layer's conv ring
/// (`[RING_ROWS][C]`), read and written in place.
pub struct ConvArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub b_raw: &'a DeviceBuffer<f32>,
    pub a_raw: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub dt_bias: &'a DeviceBuffer<f32>,
    pub ssm_a: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub shape: LinearShape,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub decay: &'a mut DeviceBuffer<f32>,
    pub ring: &'a mut DeviceBuffer<f32>,
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

    /// Enqueue the conv and prep of `args.m` tokens: `⌈units(m)·H / 4⌉`
    /// blocks of 128 threads, `H = 2·n_k + n_v`, the ring read and written
    /// in place. Asynchronous, allocation-free, capturable.
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
            pos,
            shape,
            eps,
            m,
            fault,
            y,
            beta,
            decay,
            ring,
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
            ("pos", pos.len(), m),
            ("y", y.len(), m * ch),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), m * nv),
            ("ring", ring.len(), shape.ring_len()),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", (units(m) * shape.conv_heads()).div_ceil(4))?;
        let n_k = launch_u32(what, "n_k", shape.n_k)?;
        let n_v = launch_u32(what, "n_v", nv)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_gdn_conv_prep(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.gdn_conv_prep(
            stream, &prep, x, w, b_raw, a_raw, dt_bias, ssm_a, pos, eps, n_k, n_v, m, fault, y,
            beta, decay, ring,
        )?;
        Ok(())
    }
}

/// What [`conv_prep_host`] computes: the conv output, β, decay and the ring
/// after the call, in the kernel's layouts.
pub struct ConvOut {
    pub y: Vec<f32>,
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub ring: Vec<f32>,
}

/// The host rule of [`ConvKernels::enqueue_conv_prep`]: the module doc's
/// numeric contract and position rule, op for op. Inputs as [`ConvArgs`]
/// names them, host slices of the same lengths; `ring` is the ring before
/// the call.
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
    pos: &[u32],
    ring: &[f32],
    shape: LinearShape,
    eps: f32,
    m: usize,
) -> ConvOut {
    let ch = shape.channels();
    let heads = shape.conv_heads();
    let nv = shape.n_v;
    let mut y = vec![0.0f32; m * ch];
    let mut beta = vec![0.0f32; m * nv];
    let mut decay = vec![0.0f32; m * nv];
    for t in 0..m {
        let p = pos[t] as usize;
        let contiguous = u64::from(pos[t]) == u64::from(pos[0]) + t as u64;
        for hp in 0..heads {
            let mut vals = [0.0f32; HEAD];
            let mut lanes = [0.0f64; 32];
            for (lane, sum) in lanes.iter_mut().enumerate() {
                for i in 0..4 {
                    let c = hp * HEAD + lane + 32 * i;
                    let xs: [f32; 4] = std::array::from_fn(|j| {
                        let d = CONV_ROWS - j;
                        if d <= t {
                            x[(t - d) * ch + c]
                        } else if p < d {
                            0.0
                        } else {
                            ring[(p + RING_ROWS - d) % RING_ROWS * ch + c]
                        }
                    });
                    let mut acc = xs[0] * w[4 * c];
                    for j in 1..CONV_TAPS {
                        acc = xs[j].mul_add(w[4 * c + j], acc);
                    }
                    let s = silu(acc);
                    vals[lane + 32 * i] = s;
                    *sum += f64::from(s * s);
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
            if !contiguous {
                vals.fill(f32::NAN);
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
    let mut ring = ring[..shape.ring_len()].to_vec();
    for t in m.saturating_sub(RING_ROWS)..m {
        let slot = pos[t] as usize % RING_ROWS;
        ring[slot * ch..(slot + 1) * ch].copy_from_slice(&x[t * ch..(t + 1) * ch]);
    }
    ConvOut {
        y,
        beta,
        decay,
        ring,
    }
}
