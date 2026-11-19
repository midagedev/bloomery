//! HunyuanVL's learned position rows for one image, the rule llama.cpp's mtmd computes on the host
//! for its `hunyuanvl`, `hy_v4_flash` and `hy_v4_vl` towers (the `hunyuanvl_pos_embd` input): the
//! `side × side` table (`v.position_embd.weight`, `side²` rows of `dim` values, bf16 on the card,
//! each value the file's f32 exactly) resized to the image's patch grid by bilinear sampling at a
//! source scale of `(n + 0.1) / side`, in the patch order of the tower (merge groups of 2 × 2
//! patches), each value rounded to bf16 — the rows the patch embedding's GEMM adds as its
//! residual operand.
//!
//! Numeric rule, per output `(row, col)`: the row's patch `(y, x)` of the `n_y × n_x` grid (rows in
//! merge order); per axis the scale `s = (n + 0.1) / side` in f32, the pixel-centre source
//! coordinate `f = (i + 0.5) / s − 0.5`, `i0 = floor(f)` and `i1 = i0 + 1` each clamped to
//! `[0, side − 1]`, the second index's weight `w1 = clamp(f − i0, 0, 1)` with the clamped `i0`,
//! and `w0 = 1 − w1`; then the four weights `w00 = wy0·wx0`, `w01 = wy0·wx1`, `w10 = wy1·wx0`,
//! `w11 = wy1·wx1` and `v = w00·t(y0, x0) + w01·t(y0, x1) + w10·t(y1, x0) + w11·t(y1, x1)`, the
//! four terms added left to right, every op rounded on its own. The reference's compiler may fuse
//! a product into the sum after it; that moves the f32 sum by its last bits, under the bf16
//! rounding of the output. [`pos_hunyuan_ref`] is the rule on the host, written from mtmd's loop.
//!
//! The source scale and the half-pixel coordinate are what [`crate::pos_bilinear`] (ggml's
//! align-corners interpolation) and [`crate::pos_half`] (RADIO's `n / side`) do not compute, so the
//! rule is this module's own entry.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::f32_to_bf16_rne;
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const BLOCK: u32 = 256;

/// Patches per merge group side: the rows of the output come in groups of `MERGE²`, the tower's
/// patch order.
pub const MERGE: usize = 2;

/// The term mtmd adds to a side before it divides by the table's side.
const SCALE_BIAS: f32 = 0.1;

/// One axis on the host, mtmd's loop body: the two source indices and the two weights.
fn axis(i: usize, n: usize, side: usize) -> (usize, usize, f32, f32) {
    let s = (n as f32 + SCALE_BIAS) / side as f32;
    let f = (i as f32 + 0.5) / s - 0.5;
    let last = side as i64 - 1;
    let lo = f.floor() as i64;
    let (i0, i1) = (lo.clamp(0, last), (lo + 1).clamp(0, last));
    let w1 = (f - i0 as f32).clamp(0.0, 1.0);
    (i0 as usize, i1 as usize, 1.0 - w1, w1)
}

/// The host rule over a whole image: `table` the `side²` rows of `dim` f32, the grid `n_y × n_x`
/// patches (both multiples of [`MERGE`]), the output `n_y · n_x` rows of `dim` bf16 in merge order
/// (merge groups in raster order, the patches of a group in raster order).
#[must_use]
pub fn pos_hunyuan_ref(
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
                    let (y0, y1, wy0, wy1) = axis(y, n_y, side);
                    let (x0, x1, wx0, wx1) = axis(x, n_x, side);
                    let (w00, w01, w10, w11) = (wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1);
                    for c in 0..dim {
                        let at = |yy: usize, xx: usize| table[(yy * side + xx) * dim + c];
                        let v = w00 * at(y0, x0)
                            + w01 * at(y0, x1)
                            + w10 * at(y1, x0)
                            + w11 * at(y1, x1);
                        out.push(crate::f32_bf16(v));
                    }
                }
            }
        }
    }
    out
}

#[cuda_module]
mod pos_hunyuan_kernels {
    use super::*;

    /// The module doc's rows of an `n_y × n_x` grid (`n_x`, `n_y` multiples of 2) from `table`
    /// (`side²` rows of `dim` bf16) into `out`, `n_y·n_x` rows of `dim` bf16. One thread per
    /// output value.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::manual_clamp,
        reason = "mtmd's clamp of the weight; `clamp` carries a panic path a kernel cannot take"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (table.len() >= side * side * dim, out.len() >= n_y * n_x * dim)
    )]
    pub fn vis_pos_hunyuan(
        table: &[u16],
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

        let sy = div_rn_f32(add_rn_f32(n_y as f32, SCALE_BIAS), side as f32);
        let sx = div_rn_f32(add_rn_f32(n_x as f32, SCALE_BIAS), side as f32);
        // `f − 0.5` as `f + (−0.5)`: the same correctly rounded f32 result.
        let fy = add_rn_f32(div_rn_f32(add_rn_f32(y as f32, 0.5), sy), -0.5);
        let fx = add_rn_f32(div_rn_f32(add_rn_f32(x as f32, 0.5), sx), -0.5);
        // floor(f): `as i32` truncates toward zero, one less for a negative fraction.
        let (ty, tx) = (fy as i32, fx as i32);
        let ly = if fy < ty as f32 { ty - 1 } else { ty };
        let lx = if fx < tx as f32 { tx - 1 } else { tx };
        let last = s as i32 - 1;
        let clamp = |v: i32| {
            if v < 0 {
                0
            } else if v > last {
                last
            } else {
                v
            }
        };
        let (y0, x0) = (clamp(ly), clamp(lx));
        let (y1, x1) = (clamp(ly + 1), clamp(lx + 1));
        let (wy1, wx1) = (fy - y0 as f32, fx - x0 as f32);
        let (wy1, wx1) = (wy1.max(0.0).min(1.0), wx1.max(0.0).min(1.0));
        let (wy0, wx0) = (1.0 - wy1, 1.0 - wx1);
        let (y0, x0, y1, x1) = (y0 as usize, x0 as usize, y1 as usize, x1 as usize);
        let (w00, w01) = (mul_rn_f32(wy0, wx0), mul_rn_f32(wy0, wx1));
        let (w10, w11) = (mul_rn_f32(wy1, wx0), mul_rn_f32(wy1, wx1));
        let widen = |b: u16| f32::from_bits(u32::from(b) << 16);
        // SAFETY: y0, y1, x0, x1 < side and c < dim keep the four indices below side²·dim <=
        // table.len() (contract).
        let (a, b, cc, dd) = unsafe {
            (
                widen(*table.get_unchecked((y0 * s + x0) * d + c)),
                widen(*table.get_unchecked((y0 * s + x1) * d + c)),
                widen(*table.get_unchecked((y1 * s + x0) * d + c)),
                widen(*table.get_unchecked((y1 * s + x1) * d + c)),
            )
        };
        let v = add_rn_f32(
            add_rn_f32(
                add_rn_f32(mul_rn_f32(w00, a), mul_rn_f32(w01, b)),
                mul_rn_f32(w10, cc),
            ),
            mul_rn_f32(w11, dd),
        );
        // SAFETY: i < n_y·n_x·dim <= out.len() (contract); one thread per value.
        unsafe {
            *out.get_unchecked_mut(i) = f32_to_bf16_rne(v);
        }
    }
}

/// The loaded position module.
pub struct PosHunyuanKernels {
    module: pos_hunyuan_kernels::LoadedModule,
}

impl PosHunyuanKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<PosHunyuanKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(pos_hunyuan_kernels, ctx)? };
        Ok(PosHunyuanKernels { module })
    }

    /// Enqueue the rows of an `n_y × n_x` patch grid from the `side²`-row bf16 `table` of `dim`
    /// columns into `out`. Both sides even and non-zero. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        table: &DeviceBuffer<u16>,
        (side, dim): (usize, usize),
        (n_y, n_x): (usize, usize),
        out: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "PosHunyuanKernels::enqueue";
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
            .prepare_vis_pos_hunyuan(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_pos_hunyuan(
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

#[cfg(test)]
mod tests {
    use super::axis;

    /// A grid narrower than the table samples inside it; one wider reaches past its first centre,
    /// where the floor is −1 and both indices clamp to the first row with all the weight on it.
    #[test]
    fn the_axis_is_mtmds() {
        let (i0, i1, w0, w1) = axis(0, 32, 128);
        let f = 0.5 / (32.1f32 / 128.0) - 0.5;
        assert_eq!((i0, i1), (f.floor() as usize, f.floor() as usize + 1));
        assert_eq!(w1, f - f.floor());
        assert_eq!(w0, 1.0 - w1);
        assert_eq!(axis(0, 156, 128), (0, 0, 1.0, 0.0));
        let (i0, i1, _, w1) = axis(155, 156, 128);
        assert_eq!((i0, i1), (127, 127));
        assert!((0.0..=1.0).contains(&w1));
    }

    /// At the table's own side the scale is a hair above 1, so the source coordinate of patch `i`
    /// falls just short of `i`: it reads rows `i − 1` and `i`, nearly all the weight on `i`, and
    /// patch 0 (a coordinate just under 0) reads row 0 alone.
    #[test]
    fn the_tables_own_side_is_near_identity() {
        assert_eq!(axis(0, 128, 128), (0, 0, 1.0, 0.0));
        for i in [1, 64, 127] {
            let (i0, i1, w0, w1) = axis(i, 128, 128);
            assert_eq!((i0, i1), (i - 1, i), "{i}");
            assert!(w1 > 0.89 && w0 < 0.11, "{i}: {w0} {w1}");
        }
    }
}
