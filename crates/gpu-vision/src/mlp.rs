//! The SiLU gate between the ViT MLP's two linears (`vision.py` `MLP`): `w1` writes `gate` and
//! `up` side by side (one GEMM of `2·ff` columns, the file's `ffn_gate` rows then its `ffn_up`
//! rows), and this op makes `silu(gate) · up` for `w2`.
//!
//! Numeric rule, per value, the reference's two roundings: `s = bf16(g / (1 + exp(−g)))` —
//! torch's SiLU on a bf16 tensor, computed in f32 and rounded — then `y = bf16(s · u)`, the bf16
//! product. Fusing it into the `w1` epilogue would change no bit, only move the `2·ff` columns
//! through registers instead of memory. The device's `expf` and the host's `exp` differ in the
//! last ulps; [`silu_mul_ref`] is the rule with the host's.
//!
//! With a clamp (`enqueue`'s `Some(limit)`, the `glm5v` tower's gate and the merger's MLP) the op
//! is llama.cpp's `ggml_swiglu_clamp` node, one op of one rounding: `y = bf16(silu(min(g, limit))
//! · clamp(u, −limit, limit))` in f32 — the gate clamped from above only, the up value from both
//! sides. A NaN stays a NaN. It is its own entry, so the unclamped entry's code and rounding are
//! untouched. The host rule of the clamped op is the f32 function of the vision crate's tower
//! rules (`swiglu_clamp`) rounded once to bf16, which the gate applies; it is not repeated here.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const BLOCK: u32 = 256;

/// The host rule of one value: `g` and `u` bf16 bits.
#[must_use]
pub fn silu_mul_ref(g: u16, u: u16) -> u16 {
    let gv = crate::bf16_f32(g);
    let s = crate::f32_bf16(gv / (1.0 + (-gv).exp()));
    crate::f32_bf16(crate::bf16_f32(s) * crate::bf16_f32(u))
}

#[cuda_module]
mod mlp_kernels {
    use super::*;

    /// `y[r, j] = silu(u[r, j]) · u[r, ff + j]` over `rows` rows, `u` rows of `2·ff` bf16 and `y`
    /// rows of `ff`, the module rule. One thread per output value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (u.len() >= 2 * rows * ff, y.len() >= rows * ff)
    )]
    pub fn vis_silu_mul(u: &[u16], ff: u32, rows: u32, mut y: DisjointSlice<u16>) {
        let i = thread::index_1d().get();
        let f = ff as usize;
        if i >= rows as usize * f {
            return;
        }
        let r = i / f;
        let j = i - r * f;
        // SAFETY: r < rows and j < ff put both indices below 2·rows·ff <= u.len() (contract).
        let (g, up) = unsafe {
            (
                bf16_to_f32(*u.get_unchecked(2 * r * f + j)),
                bf16_to_f32(*u.get_unchecked(2 * r * f + f + j)),
            )
        };
        let s = f32_to_bf16_rne(div_rn_f32(g, add_rn_f32(1.0, (-g).exp())));
        let v = f32_to_bf16_rne(mul_rn_f32(bf16_to_f32(s), up));
        // SAFETY: i < rows·ff <= y.len(); one thread per value.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// `y[r, j] = silu(min(g, limit)) · clamp(u, ±limit)` with `g = u[r, j]` and `u = u[r, ff + j]`
    /// over `rows` rows, the module's clamped rule; shapes as `vis_silu_mul`. A comparison, not
    /// `min`/`max`, so a NaN passes through. One thread per output value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (u.len() >= 2 * rows * ff, y.len() >= rows * ff)
    )]
    pub fn vis_silu_mul_clamp(
        u: &[u16],
        ff: u32,
        rows: u32,
        limit: f32,
        mut y: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        let f = ff as usize;
        if i >= rows as usize * f {
            return;
        }
        let r = i / f;
        let j = i - r * f;
        // SAFETY: r < rows and j < ff put both indices below 2·rows·ff <= u.len() (contract).
        let (g, up) = unsafe {
            (
                bf16_to_f32(*u.get_unchecked(2 * r * f + j)),
                bf16_to_f32(*u.get_unchecked(2 * r * f + f + j)),
            )
        };
        let g = if g > limit { limit } else { g };
        let up = if up > limit {
            limit
        } else if up < -limit {
            -limit
        } else {
            up
        };
        let v = f32_to_bf16_rne(mul_rn_f32(div_rn_f32(g, add_rn_f32(1.0, (-g).exp())), up));
        // SAFETY: i < rows·ff <= y.len(); one thread per value.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }
}

/// The loaded MLP-gate module.
pub struct MlpKernels {
    module: mlp_kernels::LoadedModule,
}

impl MlpKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MlpKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(mlp_kernels, ctx)? };
        Ok(MlpKernels { module })
    }

    /// Enqueue `y = silu(gate) · up` over `rows` rows of `ff` outputs, `u` the `w1` output of
    /// `2·ff` columns per row; with `clamp = Some(limit)` the clamped op of the module doc (a
    /// `limit` that is not a positive finite number is refused). Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        u: &DeviceBuffer<u16>,
        ff: usize,
        rows: usize,
        clamp: Option<f32>,
        y: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "MlpKernels::enqueue";
        if clamp.is_some_and(|l| !(l.is_finite() && l > 0.0)) {
            return Err(GpuError::Shape {
                what,
                detail: format!("clamp {clamp:?}: a limit is a positive finite number"),
            });
        }
        if ff == 0 || rows == 0 || u.len() < 2 * rows * ff || y.len() < rows * ff {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "rows {rows} × ff {ff}: u.len() {} (need {}), y.len() {} (need {})",
                    u.len(),
                    2 * rows * ff,
                    y.len(),
                    rows * ff
                ),
            });
        }
        let grid = launch_u32(what, "grid", (rows * ff).div_ceil(BLOCK as usize))?;
        let config = LaunchConfig1D::new(grid, BLOCK, 0);
        let (ff, rows) = (launch_u32(what, "ff", ff)?, launch_u32(what, "rows", rows)?);
        match clamp {
            None => {
                let prep = self.module.prepare_vis_silu_mul(config)?;
                self.module.vis_silu_mul(stream, &prep, u, ff, rows, y)?;
            }
            Some(limit) => {
                let prep = self.module.prepare_vis_silu_mul_clamp(config)?;
                self.module
                    .vis_silu_mul_clamp(stream, &prep, u, ff, rows, limit, y)?;
            }
        }
        Ok(())
    }
}
