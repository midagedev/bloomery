//! Small row operations between a tower's shared GEMMs and norms.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// A row operation, each with a separate f32 rounding at its arithmetic boundaries.
#[derive(Clone, Copy)]
pub(crate) enum Op {
    GeGlu,
    QuickGeGlu,
    Affine,
    Clamp,
    Copy,
    Scale,
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Apply one row operation, writing each destination lane once.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry uses the flat device ABI (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, out.len() >= n))]
    pub(crate) fn vis_row_op(
        x: &[u16],
        b: &[u16],
        gain: &[f32],
        bias: &[f32],
        n: u32,
        width: u32,
        stride: u32,
        col0: u32,
        op: u32,
        scale: f32,
        lo: f32,
        hi: f32,
        mut out: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let w = width as usize;
        let row = i / w;
        let d = i % w;
        // SAFETY: i < n <= x.len().
        let a = bf16_to_f32(unsafe { *x.get_unchecked(i) });
        let y = match op {
            1 | 7 => {
                // SAFETY: both gated operations require b.len() >= n; i < n.
                let other = bf16_to_f32(unsafe { *b.get_unchecked(i) });
                if op == 1 {
                    mul_rn_f32(bloomery_gpu::gelu::gelu_tanh_dev(a), other)
                } else {
                    let e = mul_rn_f32(-1.702, a).exp();
                    let gelu = mul_rn_f32(a, div_rn_f32(1.0, add_rn_f32(1.0, e)));
                    mul_rn_f32(gelu, other)
                }
            }
            2 => {
                // SAFETY: Affine checks gain at width; d < width.
                let gain = unsafe { *gain.get_unchecked(d) };
                // SAFETY: Affine checks bias at width; d < width.
                let bias = unsafe { *bias.get_unchecked(d) };
                mul_rn_f32(add_rn_f32(mul_rn_f32(a, scale), -bias), gain)
            }
            3 => {
                if a < lo {
                    lo
                } else if a > hi {
                    hi
                } else {
                    a
                }
            }
            6 => mul_rn_f32(a, scale),
            _ => a,
        };
        let dst = row * stride as usize + col0 as usize + d;
        // SAFETY: the host proves the row extent and col0 + width <= stride;
        // each source lane has one distinct destination lane.
        unsafe {
            *out.get_unchecked_mut(dst) = f32_to_bf16_rne(y);
        }
    }

    /// Clamp F32 values without narrowing the result.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, out.len() >= n))]
    pub(crate) fn vis_clamp_f32(x: &[f32], n: u32, lo: f32, hi: f32, mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= x.len().
        let value = unsafe { *x.get_unchecked(i) };
        let value = if value < lo {
            lo
        } else if value > hi {
            hi
        } else {
            value
        };
        // SAFETY: i < n <= out.len(); thread i owns this output.
        unsafe {
            *out.get_unchecked_mut(i) = value;
        }
    }

    /// Add two F32 residual streams without an intervening narrowing.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (a.len() >= n, b.len() >= n, out.len() >= n))]
    pub(crate) fn vis_add_f32(a: &[f32], b: &[f32], n: u32, mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= a.len().
        let av = unsafe { *a.get_unchecked(i) };
        // SAFETY: i < n <= b.len().
        let bv = unsafe { *b.get_unchecked(i) };
        let value = add_rn_f32(av, bv);
        // SAFETY: i < n <= out.len(); thread i owns this output.
        unsafe {
            *out.get_unchecked_mut(i) = value;
        }
    }

    /// Gather three-by-three raster cells and narrow each value for the shared group mean.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, out.len() >= n))]
    pub(crate) fn vis_pool_order_f32(
        x: &[f32],
        n: u32,
        width: u32,
        grid_w: u32,
        mut out: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let w = width as usize;
        let row = i / w;
        let group = row / 9;
        let cell = row % 9;
        let cols = grid_w as usize / 3;
        let src = ((group / cols * 3 + cell / 3) * grid_w as usize + group % cols * 3 + cell % 3)
            * w
            + i % w;
        // SAFETY: complete three-by-three groups permute n lanes, so src < n <= x.len().
        let value = unsafe { *x.get_unchecked(src) };
        let value = f32_to_bf16_rne(value);
        // SAFETY: i < n <= out.len(); thread i owns this output.
        unsafe {
            *out.get_unchecked_mut(i) = value;
        }
    }

    /// Add the learned x and y rows to the F32 residual stream.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, out.len() >= n))]
    pub(crate) fn vis_position_rows(
        x: &[f32],
        table: &[f32],
        n: u32,
        width: u32,
        grid_w: u32,
        pos_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let w = width as usize;
        let row = i / w;
        let d = i % w;
        let px = row % grid_w as usize;
        let py = row / grid_w as usize;
        // SAFETY: i < n <= x.len().
        let x = unsafe { *x.get_unchecked(i) };
        // SAFETY: px < pos_rows and d < width bound the first table plane.
        let px = unsafe { *table.get_unchecked(px * w + d) };
        // SAFETY: py < pos_rows and d < width bound the second table plane.
        let py = unsafe { *table.get_unchecked((pos_rows as usize + py) * w + d) };
        let a = add_rn_f32(x, px);
        let y = add_rn_f32(a, py);
        if let Some(dst) = out.get_mut(thread::index_1d()) {
            *dst = y;
        }
    }
}

/// Operands and extents of one row operation; `out_col` places a segment inside wider rows.
pub(crate) struct Args<'a> {
    pub op: Op,
    pub x: &'a DeviceBuffer<u16>,
    pub b: &'a DeviceBuffer<u16>,
    pub gain: &'a DeviceBuffer<f32>,
    pub bias: &'a DeviceBuffer<f32>,
    pub rows: usize,
    pub width: usize,
    pub out_width: usize,
    pub out_col: usize,
    pub scale: f32,
    pub clamp: (f32, f32),
    pub out: &'a mut DeviceBuffer<u16>,
}

/// The learned position addition's input and output.
pub(crate) struct PositionArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub rows: usize,
    pub width: usize,
    pub grid_w: usize,
    pub pos_rows: usize,
    pub out: &'a mut DeviceBuffer<f32>,
}

pub(crate) struct FloatPoolArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub rows: usize,
    pub width: usize,
    pub grid_w: usize,
    pub out: &'a mut DeviceBuffer<u16>,
}

/// Loaded row-operation kernels.
pub(crate) struct Kernels {
    module: kernels::LoadedModule,
}

impl Kernels {
    pub(crate) fn clamp_f32(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        bounds: (f32, f32),
        n: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "vision F32 clamp";
        if n == 0
            || x.len() < n
            || out.len() < n
            || !bounds.0.is_finite()
            || !bounds.1.is_finite()
            || bounds.0 > bounds.1
        {
            return Err(GpuError::shape(what, "invalid clamp bounds or buffers"));
        }
        let prep = self.module.prepare_vis_clamp_f32(LaunchConfig1D::new(
            launch_u32(what, "grid", n.div_ceil(256))?,
            256,
            0,
        ))?;
        self.module.vis_clamp_f32(
            stream,
            &prep,
            x,
            launch_u32(what, "n", n)?,
            bounds.0,
            bounds.1,
            out,
        )?;
        Ok(())
    }

    pub(crate) fn add_f32(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "vision F32 residual add";
        if n == 0 || a.len() < n || b.len() < n || out.len() < n {
            return Err(GpuError::shape(
                what,
                "residual operands do not hold the output extent",
            ));
        }
        let prep = self.module.prepare_vis_add_f32(LaunchConfig1D::new(
            launch_u32(what, "grid", n.div_ceil(256))?,
            256,
            0,
        ))?;
        self.module
            .vis_add_f32(stream, &prep, a, b, launch_u32(what, "n", n)?, out)?;
        Ok(())
    }

    pub(crate) fn pool_f32(
        &self,
        stream: &CudaStream,
        a: FloatPoolArgs<'_>,
    ) -> Result<(), GpuError> {
        let FloatPoolArgs {
            x,
            rows,
            width,
            grid_w,
            out,
        } = a;
        let what = "vision F32 pooling gather";
        let group_rows = grid_w
            .checked_mul(3)
            .ok_or_else(|| GpuError::shape(what, "pool grid overflows"))?;
        let n = rows
            .checked_mul(width)
            .ok_or_else(|| GpuError::shape(what, "element count overflows"))?;
        if n == 0
            || grid_w == 0
            || !grid_w.is_multiple_of(3)
            || !rows.is_multiple_of(group_rows)
            || x.len() < n
            || out.len() < n
        {
            return Err(GpuError::shape(
                what,
                "pooling buffers must hold complete three-by-three groups",
            ));
        }
        let u = |name, value| launch_u32(what, name, value);
        let prep = self.module.prepare_vis_pool_order_f32(LaunchConfig1D::new(
            u("grid", n.div_ceil(256))?,
            256,
            0,
        ))?;
        self.module.vis_pool_order_f32(
            stream,
            &prep,
            x,
            u("n", n)?,
            u("width", width)?,
            u("grid width", grid_w)?,
            out,
        )?;
        Ok(())
    }

    /// Load this module's device bundle into the tower's context.
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
        // SAFETY: this crate owns the module's embedded device bundle.
        let module = unsafe { bloomery_gpu::shared_module!(kernels, ctx)? };
        Ok(Self { module })
    }

    /// Apply one checked row operation asynchronously.
    pub(crate) fn enqueue(&self, stream: &CudaStream, a: Args<'_>) -> Result<(), GpuError> {
        let what = "vision row operation";
        let n = a
            .rows
            .checked_mul(a.width)
            .ok_or_else(|| GpuError::shape(what, "element count overflows"))?;
        let need = a
            .rows
            .checked_mul(a.out_width)
            .ok_or_else(|| GpuError::shape(what, "output count overflows"))?;
        if n == 0
            || a.out_col
                .checked_add(a.width)
                .is_none_or(|v| v > a.out_width)
            || a.x.len() < n
            || a.out.len() < need
            || !a.scale.is_finite()
            || !a.clamp.0.is_finite()
            || !a.clamp.1.is_finite()
            || a.clamp.0 > a.clamp.1
        {
            return Err(GpuError::shape(
                what,
                "invalid row extent, buffers, scale or clamp",
            ));
        }
        let op = match a.op {
            Op::GeGlu | Op::QuickGeGlu if a.b.len() < n => {
                return Err(GpuError::shape(what, "second operand is short"));
            }
            Op::Affine if a.gain.len() < a.width || a.bias.len() < a.width => {
                return Err(GpuError::shape(what, "affine vectors are short"));
            }
            Op::GeGlu => 1,
            Op::QuickGeGlu => 7,
            Op::Affine => 2,
            Op::Clamp => 3,
            Op::Copy => 5,
            Op::Scale => 6,
        };
        let u = |name, v| launch_u32(what, name, v);
        let prep = self.module.prepare_vis_row_op(LaunchConfig1D::new(
            u("grid", n.div_ceil(256))?,
            256,
            0,
        ))?;
        self.module.vis_row_op(
            stream,
            &prep,
            a.x,
            a.b,
            a.gain,
            a.bias,
            u("n", n)?,
            u("width", a.width)?,
            u("stride", a.out_width)?,
            u("column", a.out_col)?,
            op,
            a.scale,
            a.clamp.0,
            a.clamp.1,
            a.out,
        )?;
        Ok(())
    }

    /// Add learned x/y position rows to the patch embeddings asynchronously.
    pub(crate) fn positions(
        &self,
        stream: &CudaStream,
        a: PositionArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "vision learned position rows";
        let n = a
            .rows
            .checked_mul(a.width)
            .ok_or_else(|| GpuError::shape(what, "element count overflows"))?;
        let table = a
            .pos_rows
            .checked_mul(a.width)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| GpuError::shape(what, "position table count overflows"))?;
        if n == 0
            || a.grid_w == 0
            || !a.rows.is_multiple_of(a.grid_w)
            || a.grid_w > a.pos_rows
            || a.rows / a.grid_w > a.pos_rows
            || a.x.len() < n
            || a.out.len() < n
            || a.table.len() < table
        {
            return Err(GpuError::shape(
                what,
                "patch or position extent does not fit its buffer",
            ));
        }
        let u = |name, v| launch_u32(what, name, v);
        let prep = self.module.prepare_vis_position_rows(LaunchConfig1D::new(
            u("grid", n.div_ceil(256))?,
            256,
            0,
        ))?;
        self.module.vis_position_rows(
            stream,
            &prep,
            a.x,
            a.table,
            u("n", n)?,
            u("width", a.width)?,
            u("grid width", a.grid_w)?,
            u("position rows", a.pos_rows)?,
            a.out,
        )?;
        Ok(())
    }
}
