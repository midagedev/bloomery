//! The sigmoid output gate of a gated softmax-attention layer: `y = attn ·
//! sigmoid(g)` elementwise over `m` tokens of `n_head` heads of [`HEAD`]
//! values, `g` the gate projection's own output (its own buffer, not packed
//! into the query's rows). Its output is the input of `W_o`. Two entries by
//! the gate's shape: [`OutGate::Proj`], one `g` a value, `[m][n_head][HEAD]`
//! (`attn_out_gate_sigmoid`), and [`OutGate::ProjHead`], one `g` a head,
//! `[m][n_head]`, every value of the head gated alike
//! (`attn_out_gate_sigmoid_head`).
//!
//! Geometry: one thread per value, [`OUT_GATE_THREADS`] a block, no shared
//! memory. Value `v` of head `h` of token `t` sits at `(t·n_head + h)·HEAD +
//! v` of `attn` and `y`, and of `gate` under `Proj`; under `ProjHead` its gate
//! is `gate[t·n_head + h]`, index `i / HEAD` of the value's index `i`.
//!
//! Numeric contract (the host rule [`out_gate_host`] is this, `gate` expanded
//! to a value each under `ProjHead`): `y =` the product of `attn` and
//! [`sigmoid`]`(g)`, rounded once.
//!
//! No silent failure: a non-finite `attn` or `g`, or a product that is not
//! finite, raises [`FaultSite::F32Product`], and that value is written NaN.
//! Under `ProjHead` a non-finite `g` makes every value of its head NaN.

use super::{HEAD, sigmoid};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::mul_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use model::arch::models::OutGate;
use std::sync::Arc;

const _: () = assert!(HEAD == 128);

/// Threads per block of the entry.
pub const OUT_GATE_THREADS: u32 = 256;
// The entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(OUT_GATE_THREADS == 256);

#[cuda_module]
mod out_gate_kernels {
    use super::*;

    /// `y = attn · sigmoid(gate)` over `m` tokens of `n_head` heads of
    /// [`HEAD`] values (module doc). A non-finite input or product raises
    /// [`FaultSite::F32Product`] and the value is written NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            attn.len() >= 128 * n_head * m,
            gate.len() >= 128 * n_head * m,
            y.len() >= 128 * n_head * m
        )
    )]
    pub fn attn_out_gate_sigmoid(
        attn: &[f32],
        gate: &[f32],
        n_head: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= HEAD * n_head as usize * m as usize {
            return;
        }
        // SAFETY: i < HEAD·n_head·m <= attn.len(), gate.len() by the launch
        // contract.
        let (a, g) = unsafe { (*attn.get_unchecked(i), *gate.get_unchecked(i)) };
        let out = mul_rn_f32(a, sigmoid(g));
        let ok = a.is_finite() & g.is_finite() & out.is_finite();
        if !ok {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < HEAD·n_head·m <= y.len() by the launch contract; one
        // thread per value.
        unsafe { *y.get_unchecked_mut(i) = if ok { out } else { f32::NAN } };
    }

    /// `y = attn · sigmoid(gate[i / HEAD])` over `m` tokens of `n_head` heads
    /// of [`HEAD`] values, one gate value a head (module doc). A non-finite
    /// input or product raises [`FaultSite::F32Product`] and the value is
    /// written NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            attn.len() >= 128 * n_head * m,
            gate.len() >= n_head * m,
            y.len() >= 128 * n_head * m
        )
    )]
    pub fn attn_out_gate_sigmoid_head(
        attn: &[f32],
        gate: &[f32],
        n_head: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= HEAD * n_head as usize * m as usize {
            return;
        }
        // SAFETY: i < HEAD·n_head·m <= attn.len() and i / HEAD < n_head·m <=
        // gate.len() by the launch contract.
        let (a, g) = unsafe { (*attn.get_unchecked(i), *gate.get_unchecked(i / HEAD)) };
        let out = mul_rn_f32(a, sigmoid(g));
        let ok = a.is_finite() & g.is_finite() & out.is_finite();
        if !ok {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < HEAD·n_head·m <= y.len() by the launch contract; one
        // thread per value.
        unsafe { *y.get_unchecked_mut(i) = if ok { out } else { f32::NAN } };
    }
}

/// [`OutGateKernels::enqueue_out_gate`]'s arguments: `m` tokens of the
/// attention output `attn` (`[m][n_head][HEAD]`), the gate projection's output
/// `gate` (the same shape under [`OutGate::Proj`], `[m][n_head]` under
/// [`OutGate::ProjHead`]), and the output `y` of the attention's shape.
pub struct OutGateArgs<'a> {
    pub kind: OutGate,
    pub attn: &'a DeviceBuffer<f32>,
    pub gate: &'a DeviceBuffer<f32>,
    pub n_head: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct OutGateKernels {
    module: out_gate_kernels::LoadedModule,
}

impl OutGateKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<OutGateKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(out_gate_kernels, ctx)? };
        Ok(OutGateKernels { module })
    }

    /// Enqueue `y = attn · sigmoid(gate)` over `args.m` tokens of
    /// `args.n_head` heads under the gate `args.kind`, `Proj` or `ProjHead`
    /// (any other kind is refused by name): `⌈m·n_head·HEAD / 256⌉` blocks of
    /// 256 threads. Asynchronous, allocation-free, capturable.
    pub fn enqueue_out_gate(
        &self,
        stream: &CudaStream,
        args: OutGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_out_gate";
        let OutGateArgs {
            kind,
            attn,
            gate,
            n_head,
            m,
            fault,
            y,
        } = args;
        if n_head == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_head, m >= 1, got n_head={n_head} m={m}"),
            ));
        }
        let n = n_head * m * HEAD;
        let per_head = match kind {
            OutGate::Proj => false,
            OutGate::ProjHead => true,
            OutGate::None | OutGate::Packed => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "{kind:?} is not a gate of its own: this entry takes Proj and ProjHead"
                    ),
                ));
            }
        };
        let gates = if per_head { n_head * m } else { n };
        let lens = [
            ("attn", attn.len(), n),
            ("gate", gate.len(), gates),
            ("y", y.len(), n),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", n.div_ceil(OUT_GATE_THREADS as usize))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, OUT_GATE_THREADS, 0);
        if per_head {
            let prep = self.module.prepare_attn_out_gate_sigmoid_head(cfg)?;
            self.module
                .attn_out_gate_sigmoid_head(stream, &prep, attn, gate, n_head, m, fault, y)?;
        } else {
            let prep = self.module.prepare_attn_out_gate_sigmoid(cfg)?;
            self.module
                .attn_out_gate_sigmoid(stream, &prep, attn, gate, n_head, m, fault, y)?;
        }
        Ok(())
    }
}

/// The host rule of [`OutGateKernels::enqueue_out_gate`]: the module doc's
/// numeric contract, op for op, over `attn.len()` values; a value whose
/// input or product is not finite is NaN.
///
/// # Panics
///
/// When `gate` is shorter than `attn`.
#[must_use]
pub fn out_gate_host(attn: &[f32], gate: &[f32]) -> Vec<f32> {
    attn.iter()
        .zip(gate)
        .map(|(&a, &g)| {
            let out = a * sigmoid(g);
            if a.is_finite() && g.is_finite() && out.is_finite() {
                out
            } else {
                f32::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::out_gate_host;
    use crate::linear::sigmoid;

    /// The host rule against `a / (1 + e^-g)` in f64: the exp polynomial's
    /// 2.5 ulp at most through `1 + e`, and the sum, the division and the
    /// product each half an ulp, under 8e-7 relative.
    #[test]
    fn out_gate_is_attn_times_sigmoid() {
        let n = 1u16 << 12;
        let a: Vec<f32> = (0..n).map(|i| (f32::from(i) * 0.37).sin() * 3.0).collect();
        let g: Vec<f32> = (0..n).map(|i| (f32::from(i) * 0.11).cos() * 9.0).collect();
        let y = out_gate_host(&a, &g);
        for (i, ((&a, &g), &y)) in a.iter().zip(&g).zip(&y).enumerate() {
            let want = f64::from(a) / (1.0 + (-f64::from(g)).exp());
            let err = (f64::from(y) - want).abs();
            assert!(err <= 8e-7 * want.abs() + 1e-30, "i={i}: {y} vs {want}");
            assert_eq!(y.to_bits(), (a * sigmoid(g)).to_bits());
        }
    }

    /// A non-finite input or product is NaN, never a plausible value: an
    /// infinite gate with a finite `attn` would otherwise pass `attn`
    /// through.
    #[test]
    fn out_gate_writes_nan_for_a_non_finite_value() {
        let y = out_gate_host(
            &[1.0, f32::NAN, 1.0, f32::INFINITY, 2.0],
            &[f32::INFINITY, 1.0, f32::NAN, 1.0, 0.5],
        );
        assert!(y[..4].iter().all(|v| v.is_nan()));
        assert!(y[4].is_finite());
    }
}
