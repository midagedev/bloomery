//! The aligner's input (`vision.py` `Aligner.forward`): the ViT output viewed as its `n_h × n_w`
//! patch grid, padded with zeros at the bottom and right to multiples of the downsample ratio
//! `r` (3), and cut by `F.unfold(x, r, stride=r)` into one row per `r × r` cell, cells row-major.
//! A cell's row holds `dim · r²` values in unfold's order, channel slowest: value
//! `c·r² + kh·r + kw` is channel `c` of patch `(r·oh + kh, r·ow + kw)`, zero where that patch is
//! padding. The two linears after it are [`crate::gemm_bf16`] launches (`w1` with the GELU
//! epilogue, then `w2`).
//!
//! A pure data movement: kernel and host rule ([`unfold_ref`]) agree bit for bit.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const BLOCK: u32 = 256;

/// Cells of an `n_h × n_w` grid at ratio `r`: `(ceil(n_h / r), ceil(n_w / r))`.
#[must_use]
pub fn cells(n_h: usize, n_w: usize, r: usize) -> (usize, usize) {
    (n_h.div_ceil(r), n_w.div_ceil(r))
}

/// The host rule: the unfolded rows of `x` (`n_h · n_w` rows of `dim` bf16).
#[must_use]
pub fn unfold_ref(x: &[u16], n_h: usize, n_w: usize, dim: usize, r: usize) -> Vec<u16> {
    let (ch, cw) = cells(n_h, n_w, r);
    let width = dim * r * r;
    let mut y = vec![0u16; ch * cw * width];
    for oh in 0..ch {
        for ow in 0..cw {
            let row = &mut y[(oh * cw + ow) * width..][..width];
            for c in 0..dim {
                for kh in 0..r {
                    for kw in 0..r {
                        let (ph, pw) = (r * oh + kh, r * ow + kw);
                        if ph < n_h && pw < n_w {
                            row[c * r * r + kh * r + kw] = x[(ph * n_w + pw) * dim + c];
                        }
                    }
                }
            }
        }
    }
    y
}

#[cuda_module]
mod aligner_kernels {
    use super::*;

    /// The unfold of the module doc: `x` holds `n_h · n_w` rows of `dim` bf16, `y` takes
    /// `ceil(n_h/r) · ceil(n_w/r)` rows of `dim · r²`. One thread per output value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= n_h * n_w * dim, y.len() >= cells * dim * r * r)
    )]
    pub fn vis_unfold(
        x: &[u16],
        n_h: u32,
        n_w: u32,
        dim: u32,
        r: u32,
        cells: u32,
        mut y: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        let (d, rr) = (dim as usize, r as usize);
        let width = d * rr * rr;
        if i >= cells as usize * width {
            return;
        }
        let cell = i / width;
        let v = i - cell * width;
        let cw = (n_w as usize).div_ceil(rr);
        let oh = cell / cw;
        let ow = cell - oh * cw;
        let c = v / (rr * rr);
        let k = v - c * rr * rr;
        let kh = k / rr;
        let kw = k - kh * rr;
        let (ph, pw) = (rr * oh + kh, rr * ow + kw);
        let val = if ph < n_h as usize && pw < n_w as usize {
            // SAFETY: ph < n_h, pw < n_w and c < dim put the index below n_h·n_w·dim <=
            // x.len() (contract).
            unsafe { *x.get_unchecked((ph * n_w as usize + pw) * d + c) }
        } else {
            0
        };
        // SAFETY: i < cells·dim·r² <= y.len(); one thread per value.
        unsafe {
            *y.get_unchecked_mut(i) = val;
        }
    }
}

/// [`AlignerKernels::enqueue`]'s arguments: `x` holds the `n_h × n_w` grid of `dim`-wide rows,
/// `r` the ratio, `y` takes the unfolded rows.
pub struct UnfoldArgs<'a> {
    pub x: &'a DeviceBuffer<u16>,
    pub n_h: usize,
    pub n_w: usize,
    pub dim: usize,
    pub r: usize,
    pub y: &'a mut DeviceBuffer<u16>,
}

/// The loaded aligner-input module.
pub struct AlignerKernels {
    module: aligner_kernels::LoadedModule,
}

impl AlignerKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<AlignerKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { aligner_kernels::load(ctx)? };
        Ok(AlignerKernels { module })
    }

    /// Enqueue the unfold ([`UnfoldArgs`]). Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: UnfoldArgs<'_>) -> Result<(), GpuError> {
        let what = "AlignerKernels::enqueue";
        let UnfoldArgs {
            x,
            n_h,
            n_w,
            dim,
            r,
            y,
        } = args;
        let (ch, cw) = cells(n_h, n_w, r.max(1));
        let out = ch * cw * dim * r * r;
        if n_h == 0 || n_w == 0 || dim == 0 || r == 0 || x.len() < n_h * n_w * dim || y.len() < out
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "grid {n_h}x{n_w} of {dim} at ratio {r}: x.len() {}, y.len() {} (need {out})",
                    x.len(),
                    y.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", out.div_ceil(BLOCK as usize))?;
        let prep = self
            .module
            .prepare_vis_unfold(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_unfold(
            stream,
            &prep,
            x,
            launch_u32(what, "n_h", n_h)?,
            launch_u32(what, "n_w", n_w)?,
            launch_u32(what, "dim", dim)?,
            launch_u32(what, "r", r)?,
            launch_u32(what, "cells", ch * cw)?,
            y,
        )?;
        Ok(())
    }
}
