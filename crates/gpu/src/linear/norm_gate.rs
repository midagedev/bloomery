//! The gated output norm of a linear-attention layer: per (token, value
//! head), the RMS norm of the delta step's `o` over its [`HEAD`] values with
//! the layer's one [`HEAD`]-wide gain, times the gate activation of `z`
//! (the `W_gate` projection). Its output is the input of `W_out`.
//!
//! A kernel of its own rather than the delta kernel's store: the norm needs
//! a head's 128 columns, which the delta step spreads over eight blocks so
//! that a prompt's token loop runs on `8·n_v` blocks instead of `n_v`.
//!
//! Geometry: one warp per (token, head), four per block, no shared memory.
//! Lane `l` owns values `l + 32·i` (`i = 0..4`).
//!
//! Numeric contract (the host rule [`norm_gate_host`] is this list), ggml's
//! `rms_norm` then `mul` then the gate:
//! - each value squared in f32, lane `l`'s four squares summed in f64 in `i`
//!   order, the 32 lanes by the xor butterfly (16, 8, 4, 2, 1) in f64;
//! - `mean = (sum / HEAD)` in f64, rounded once to f32; `scale = 1 /
//!   √(mean + eps)` in f32;
//! - `y = ((o·scale)·w)·act(z)`, each product rounded, `act` [`silu`] for
//!   [`GATE_SILU`] (`gdn_norm_gate`) and [`sigmoid`] for [`GATE_SIGMOID`]
//!   (`gdn_norm_gate_sigmoid`).
//!
//! No silent failure: a non-finite `o`, `z` or gain, or a mean that is not
//! finite (squares of finite values can overflow), raises
//! [`FaultSite::LinearGate`], and the head's values are written non-finite
//! (the scale is NaN) rather than as a zero head.

use super::{BLOCK, GATE_SIGMOID, GATE_SILU, HEAD, sigmoid, silu};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::mul_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
pub use model::arch::models::GdnGate;
use std::sync::Arc;

const _: () = assert!(HEAD == 128 && BLOCK == 128);

/// The gate activation `ACT` names.
#[inline(always)]
fn act<const ACT: u32>(z: f32) -> f32 {
    if ACT == GATE_SILU {
        silu(z)
    } else {
        sigmoid(z)
    }
}

/// One warp's gated norm (module doc) with the gate `ACT`: head `wi` of
/// the `[m][n_v]` order.
///
/// # Safety
///
/// `wi < m·n_v`, `lane < 32`; `o` and `z` hold `m·n_v·HEAD` values, `w`
/// `HEAD`, `y` `m·n_v·HEAD`; head `wi` of `y` is this warp's own. Called by
/// all 32 lanes of the warp together.
#[allow(
    clippy::too_many_arguments,
    reason = "one warp's coordinates and the entry's arguments, forwarded flat"
)]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn norm_gate_body<const ACT: u32>(
    wi: usize,
    lane: usize,
    o: &[f32],
    z: &[f32],
    w: &[f32],
    eps: f32,
    fault: FaultSink,
    y: &mut DisjointSlice<f32>,
) {
    let base = wi * HEAD;
    let mut ov = [0.0f32; 4];
    let mut zv = [0.0f32; 4];
    let mut wv = [0.0f32; 4];
    for i in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        let at = lane + 32 * i;
        // SAFETY: wi is live and at < HEAD, inside o's checked span.
        ov[i] = unsafe { *o.get_unchecked(base + at) };
        // SAFETY: wi is live and at < HEAD, inside z's checked span, which is o's.
        zv[i] = unsafe { *z.get_unchecked(base + at) };
        // SAFETY: at < HEAD <= w.len().
        wv[i] = unsafe { *w.get_unchecked(at) };
    }
    let values = norm_gate_values::<ACT>(ov, zv, wv, eps, fault);
    for i in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: this lane owns these four values of the live head.
        unsafe { *y.get_unchecked_mut(base + lane + 32 * i) = values[i] };
    }
}

/// The four values of one lane of a head, called by all 32 lanes together.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "the CUDA unroller needs indexed fixed-size arrays"
)]
pub(crate) fn norm_gate_values<const ACT: u32>(
    ov: [f32; 4],
    zv: [f32; 4],
    wv: [f32; 4],
    eps: f32,
    fault: FaultSink,
) -> [f32; 4] {
    let mut acc = 0.0f64;
    for i in 0usize..4 {
        thread::__unroll_config::<0>();
        acc += f64::from(mul_rn_f32(ov[i], ov[i]));
    }
    acc += warp::shuffle_xor_f64(acc, 16);
    acc += warp::shuffle_xor_f64(acc, 8);
    acc += warp::shuffle_xor_f64(acc, 4);
    acc += warp::shuffle_xor_f64(acc, 2);
    acc += warp::shuffle_xor_f64(acc, 1);
    let mean = (acc / HEAD as f64) as f32;
    let fin = mean.is_finite();
    let scale = if fin {
        1.0 / (mean + eps).sqrt()
    } else {
        f32::NAN
    };
    let ok = fin
        & crate::fault::quad_finite(ov)
        & crate::fault::quad_finite(zv)
        & crate::fault::quad_finite(wv);
    let mut values = [0.0f32; 4];
    for i in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        values[i] = mul_rn_f32(
            mul_rn_f32(mul_rn_f32(ov[i], scale), wv[i]),
            act::<ACT>(zv[i]),
        );
    }
    if !ok {
        fault.raise(FaultSite::LinearGate);
    }
    values
}

#[cuda_module]
mod norm_gate_kernels {
    use super::*;

    /// The gated norm of `m·n_v` heads with the SiLU gate (module doc). Warp
    /// `w = 4·block + warp` is head `w` of the `[m][n_v]` order.
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
            o.len() >= m * n_v * 128,
            z.len() >= m * n_v * 128,
            w.len() >= 128,
            y.len() >= m * n_v * 128
        )
    )]
    pub fn gdn_norm_gate(
        o: &[f32],
        z: &[f32],
        w: &[f32],
        eps: f32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let wi = thread::blockIdx_x() as usize * 4 + tid / 32;
        if wi >= m as usize * n_v as usize {
            return; // warp-uniform
        }
        // SAFETY: wi < m·n_v by the guard, lane < 32; the launch contract's
        // lengths cover o, z, w and y; head wi of y is this warp's.
        unsafe { norm_gate_body::<GATE_SILU>(wi, tid % 32, o, z, w, eps, fault, &mut y) };
    }

    /// [`gdn_norm_gate`] with the sigmoid gate: Qwen3.8's Gated DeltaNet and
    /// GLM-5.3-Flash's KDA (`RMSNorm_w(o)·sigmoid(z)`, the same product in
    /// either operand order).
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
            o.len() >= m * n_v * 128,
            z.len() >= m * n_v * 128,
            w.len() >= 128,
            y.len() >= m * n_v * 128
        )
    )]
    pub fn gdn_norm_gate_sigmoid(
        o: &[f32],
        z: &[f32],
        w: &[f32],
        eps: f32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let wi = thread::blockIdx_x() as usize * 4 + tid / 32;
        if wi >= m as usize * n_v as usize {
            return; // warp-uniform
        }
        // SAFETY: wi < m·n_v by the guard, lane < 32; the contract's
        // lengths; head wi of y is this warp's.
        unsafe { norm_gate_body::<GATE_SIGMOID>(wi, tid % 32, o, z, w, eps, fault, &mut y) };
    }
}

/// [`NormGateKernels::enqueue_norm_gate`]'s arguments: `m` tokens of the
/// delta step's `o` and the gate projection's `z` (`[m][n_v][HEAD]`), the
/// layer's [`HEAD`] gains, the activation of `z` its model gates by, and the
/// output `y` of the same shape.
pub struct NormGateArgs<'a> {
    pub o: &'a DeviceBuffer<f32>,
    pub z: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub gate: GdnGate,
    pub eps: f32,
    pub n_v: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The norm and gate operands consumed by a product's final store, with the
/// activation of `z` the model gates by.
pub struct NormGateTail<'a> {
    pub o: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub gate: GdnGate,
    pub eps: f32,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

impl NormGateTail<'_> {
    /// Refused by name as `what`'s unless the tail gates by sigmoid: a
    /// product epilogue's norm, whose entries carry that gate alone.
    pub(crate) fn check_sigmoid(&self, what: &'static str) -> Result<(), GpuError> {
        match self.gate {
            GdnGate::Sigmoid => Ok(()),
            GdnGate::Silu => Err(GpuError::shape(
                what,
                "the epilogue's norm gates by sigmoid; a SiLU gate runs the norm apart",
            )),
        }
    }

    pub(crate) fn check_rows(
        &self,
        what: &'static str,
        rows: usize,
        m: usize,
        z_len: usize,
    ) -> Result<(), GpuError> {
        if !rows.is_multiple_of(HEAD) {
            return Err(GpuError::shape(
                what,
                "weight rows must be whole 128-value heads",
            ));
        }
        self.check(what, rows / HEAD, m, z_len)
    }

    pub(crate) fn check(
        &self,
        what: &'static str,
        n_v: usize,
        m: usize,
        z_len: usize,
    ) -> Result<(), GpuError> {
        if n_v == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_v, m >= 1, got n_v={n_v} m={m}"),
            ));
        }
        let values = m
            .checked_mul(n_v)
            .and_then(|v| v.checked_mul(HEAD))
            .ok_or_else(|| GpuError::shape(what, "m * n_v * HEAD overflows"))?;
        let lens = [
            ("o", self.o.len(), values),
            ("z", z_len, values),
            ("w", self.w.len(), HEAD),
            ("y", self.y.len(), values),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        Ok(())
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct NormGateKernels {
    module: norm_gate_kernels::LoadedModule,
}

impl NormGateKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<NormGateKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(norm_gate_kernels, ctx)? };
        Ok(NormGateKernels { module })
    }

    /// Enqueue the gated norm of `args.m · args.n_v` heads with the gate
    /// `args.gate` (`gdn_norm_gate` for SiLU, `gdn_norm_gate_sigmoid`):
    /// `⌈m·n_v / 4⌉` blocks of 128 threads. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_norm_gate(
        &self,
        stream: &CudaStream,
        args: NormGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "NormGateKernels::enqueue_norm_gate";
        let NormGateArgs {
            o,
            z,
            w,
            gate,
            eps,
            n_v,
            m,
            fault,
            y,
        } = args;
        NormGateTail {
            o,
            w,
            gate,
            eps,
            fault,
            y: &mut *y,
        }
        .check(what, n_v, m, z.len())?;
        let grid = launch_u32(what, "grid", (m * n_v).div_ceil(4))?;
        let n_v = launch_u32(what, "n_v", n_v)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        match gate {
            GdnGate::Silu => {
                let prep = self.module.prepare_gdn_norm_gate(cfg)?;
                self.module
                    .gdn_norm_gate(stream, &prep, o, z, w, eps, n_v, m, fault, y)?;
            }
            GdnGate::Sigmoid => {
                let prep = self.module.prepare_gdn_norm_gate_sigmoid(cfg)?;
                self.module
                    .gdn_norm_gate_sigmoid(stream, &prep, o, z, w, eps, n_v, m, fault, y)?;
            }
        }
        Ok(())
    }
}

/// The host rule of [`NormGateKernels::enqueue_norm_gate`] (`ACT`
/// [`GATE_SILU`] under [`GdnGate::Silu`], [`GATE_SIGMOID`] under
/// [`GdnGate::Sigmoid`]): the module doc's numeric contract, op for op, over `o.len() /
/// HEAD` heads.
///
/// # Panics
///
/// When `z` is shorter than `o` or `w` shorter than [`HEAD`].
#[must_use]
pub fn norm_gate_host<const ACT: u32>(o: &[f32], z: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; o.len()];
    for (hd, (oh, zh)) in o.chunks(HEAD).zip(z.chunks(HEAD)).enumerate() {
        let lanes: Vec<f64> = (0..32)
            .map(|l| {
                let mut s = 0.0f64;
                for i in 0..4 {
                    let v = oh[l + 32 * i];
                    s += f64::from(v * v);
                }
                s
            })
            .collect();
        let mean = (super::butterfly_f64(&lanes) / HEAD as f64) as f32;
        let scale = if mean.is_finite() {
            1.0 / (mean + eps).sqrt()
        } else {
            f32::NAN
        };
        for d in 0..HEAD {
            y[hd * HEAD + d] = ((oh[d] * scale) * w[d]) * act::<ACT>(zh[d]);
        }
    }
    y
}
