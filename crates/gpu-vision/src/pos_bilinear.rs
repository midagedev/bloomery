//! The Qwen3-VL tower's learned position rows for one image: the `side × side` f32 table
//! (`v.position_embd.weight`, `side²` rows of `dim` values) resized to the image's patch grid by
//! bilinear interpolation with align-corners, in the patch order of the tower (merge groups of
//! 2 × 2 patches), each value rounded to bf16 — the rows the patch embedding's GEMM adds as its
//! residual operand. llama.cpp's `resize_position_embeddings` (`ggml_interpolate` with
//! `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS`) and the merge reorder after it.
//!
//! Numeric rule, per output `(row, col)`: the row's patch `(y, x)` of the `n_y × n_x` grid (rows
//! in merge order); per axis the scale `s = (n − 1) / (side − 1)` in f32 (`n / side` when the grid
//! or the table is one wide), the source coordinate `u = i / s`, `i0 = floor(u)` and `i1 = i0 + 1`
//! clamped to `[0, side − 1]`, the weight `d = clamp(u − i0, 0, 1)` with the clamped `i0`; then
//! `v = a·(1−dx)·(1−dy) + b·dx·(1−dy) + c·(1−dx)·dy + d·dx·dy` for the table values `a, b, c, d` at
//! `(x0, y0)`, `(x1, y0)`, `(x0, y1)`, `(x1, y1)`, each product left to right and the four terms
//! added left to right, every op rounded on its own. [`pos_bilinear_ref`] is that rule on the
//! host, written from the loops of ggml's `ggml_compute_forward_upscale_f32`.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::f32_to_bf16_rne;
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const BLOCK: u32 = 256;

/// Patches per merge group side: the rows of the output come in groups of `MERGE²`.
pub const MERGE: usize = 2;

/// ggml's per-axis scale: `(dst − 1) / (src − 1)` when both are wider than one, else `dst / src`.
fn scale(dst: usize, src: usize) -> f32 {
    if dst > 1 && src > 1 {
        (dst - 1) as f32 / (src - 1) as f32
    } else {
        dst as f32 / src as f32
    }
}

/// One axis of the interpolation on the host, ggml's loop body: the two source indices
/// (`floor(u)` and `floor(u) + 1`, each clamped to the table) and the weight of the second.
fn axis(i: usize, dst: usize, src: usize) -> (usize, usize, f32) {
    let u = i as f32 / scale(dst, src);
    let last = src as i64 - 1;
    let f = u.floor() as i64;
    let (lo, hi) = (f.clamp(0, last), (f + 1).clamp(0, last));
    let d = (u - lo as f32).clamp(0.0, 1.0);
    (lo as usize, hi as usize, d)
}

/// The host rule over a whole image: `table` the `side²` rows of `dim` f32, the grid `n_y × n_x`
/// patches (both multiples of [`MERGE`]), the output `n_y · n_x` rows of `dim` bf16 in merge order
/// (merge groups in raster order, the patches of a group in raster order).
#[must_use]
pub fn pos_bilinear_ref(
    table: &[f32],
    side: usize,
    dim: usize,
    (n_y, n_x): (usize, usize),
) -> Vec<u16> {
    let mut out = Vec::with_capacity(n_y * n_x * dim);
    for gy in 0..n_y / MERGE {
        for gx in 0..n_x / MERGE {
            for dy in 0..MERGE {
                for dx in 0..MERGE {
                    let (y, x) = (gy * MERGE + dy, gx * MERGE + dx);
                    let (y0, y1, wy) = axis(y, n_y, side);
                    let (x0, x1, wx) = axis(x, n_x, side);
                    for c in 0..dim {
                        let at = |yy: usize, xx: usize| table[(yy * side + xx) * dim + c];
                        let (a, b) = (at(y0, x0), at(y0, x1));
                        let (cc, d) = (at(y1, x0), at(y1, x1));
                        let v = a * (1.0 - wx) * (1.0 - wy)
                            + b * wx * (1.0 - wy)
                            + cc * (1.0 - wx) * wy
                            + d * wx * wy;
                        out.push(crate::f32_bf16(v));
                    }
                }
            }
        }
    }
    out
}

#[cuda_module]
mod pos_bilinear_kernels {
    use super::*;

    /// The module doc's rows of an `n_y × n_x` grid (`n_x`, `n_y` multiples of 2) from `table`
    /// (`side²` rows of `dim` f32) into `out`, `n_y·n_x` rows of `dim` bf16. One thread per
    /// output value.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::manual_clamp,
        reason = "ggml's max(0, min(·, 1)); `clamp` carries a panic path a kernel cannot take"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (table.len() >= side * side * dim, out.len() >= n_y * n_x * dim)
    )]
    pub fn vis_pos_bilinear(
        table: &[f32],
        side: u32,
        dim: u32,
        n_y: u32,
        n_x: u32,
        mut out: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let rows = n_y as usize * n_x as usize;
        if i >= rows * d {
            return;
        }
        let m = i / d;
        let c = i - m * d;
        // The patch of row `m` in merge order: group `m / 4`, patch `m % 4` of it.
        let groups_x = n_x as usize / 2;
        let (g, q) = (m / 4, m % 4);
        let y = 2 * (g / groups_x) + q / 2;
        let x = 2 * (g % groups_x) + q % 2;
        let s = side as usize;

        let sy = if n_y > 1 && side > 1 {
            div_rn_f32((n_y - 1) as f32, (side - 1) as f32)
        } else {
            div_rn_f32(n_y as f32, side as f32)
        };
        let sx = if n_x > 1 && side > 1 {
            div_rn_f32((n_x - 1) as f32, (side - 1) as f32)
        } else {
            div_rn_f32(n_x as f32, side as f32)
        };
        // u >= 0, so the truncation is the floor.
        let (uy, ux) = (div_rn_f32(y as f32, sy), div_rn_f32(x as f32, sx));
        let (y0, x0) = (
            if (uy as usize) < s - 1 {
                uy as usize
            } else {
                s - 1
            },
            if (ux as usize) < s - 1 {
                ux as usize
            } else {
                s - 1
            },
        );
        let (y1, x1) = (
            if y0 + 1 < s - 1 { y0 + 1 } else { s - 1 },
            if x0 + 1 < s - 1 { x0 + 1 } else { s - 1 },
        );
        let (wy, wx) = (uy - y0 as f32, ux - x0 as f32);
        let (wy, wx) = (wy.max(0.0).min(1.0), wx.max(0.0).min(1.0));
        // SAFETY: y0, y1, x0, x1 < side and c < dim keep the four indices below side²·dim <=
        // table.len() (contract).
        let (a, b, cc, dd) = unsafe {
            (
                *table.get_unchecked((y0 * s + x0) * d + c),
                *table.get_unchecked((y0 * s + x1) * d + c),
                *table.get_unchecked((y1 * s + x0) * d + c),
                *table.get_unchecked((y1 * s + x1) * d + c),
            )
        };
        let (ox, oy) = (1.0 - wx, 1.0 - wy);
        let t1 = mul_rn_f32(mul_rn_f32(a, ox), oy);
        let t2 = mul_rn_f32(mul_rn_f32(b, wx), oy);
        let t3 = mul_rn_f32(mul_rn_f32(cc, ox), wy);
        let t4 = mul_rn_f32(mul_rn_f32(dd, wx), wy);
        let v = add_rn_f32(add_rn_f32(add_rn_f32(t1, t2), t3), t4);
        // SAFETY: i < n_y·n_x·dim <= out.len() (contract); one thread per value.
        unsafe {
            *out.get_unchecked_mut(i) = f32_to_bf16_rne(v);
        }
    }
}

/// The loaded position module.
pub struct PosBilinearKernels {
    module: pos_bilinear_kernels::LoadedModule,
}

impl PosBilinearKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<PosBilinearKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(pos_bilinear_kernels, ctx)? };
        Ok(PosBilinearKernels { module })
    }

    /// Enqueue the rows of an `n_y × n_x` patch grid from the `side²`-row f32 `table` of `dim`
    /// columns into `out`. Both sides even and non-zero. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        table: &DeviceBuffer<f32>,
        (side, dim): (usize, usize),
        (n_y, n_x): (usize, usize),
        out: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "PosBilinearKernels::enqueue";
        let (rows, values) = (n_y * n_x, n_y * n_x * dim);
        if side == 0
            || dim == 0
            || n_y == 0
            || n_x == 0
            || !n_y.is_multiple_of(MERGE)
            || !n_x.is_multiple_of(MERGE)
            || table.len() < side * side * dim
            || out.len() < values
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "grid {n_y}x{n_x} (even sides) of {dim} values from a table of {side}² rows: \
                     table.len() {}, out.len() {} (need {values}, {rows} rows)",
                    table.len(),
                    out.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", values.div_ceil(BLOCK as usize))?;
        let prep = self
            .module
            .prepare_vis_pos_bilinear(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_pos_bilinear(
            stream,
            &prep,
            table,
            launch_u32(what, "side", side)?,
            launch_u32(what, "dim", dim)?,
            launch_u32(what, "n_y", n_y)?,
            launch_u32(what, "n_x", n_x)?,
            out,
        )?;
        Ok(())
    }
}
