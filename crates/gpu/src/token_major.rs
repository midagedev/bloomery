//! The copy of a row-major `m`-column output into token-major rows, shared by
//! every architecture whose K-quant gemv writes `m > 1` columns row-major
//! (`q6k_gemv`) into a chain that reads token-major. At `m = 1` the two
//! layouts are the same.

use crate::GpuError;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod token_major_kernels {
    use super::*;

    /// A row-major `m`-column output to token-major: `y[c · rows + r] =
    /// x[r · m + c]`, one thread per value — a copy, so exact.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= rows * m, y.len() >= rows * m)
    )]
    pub fn token_major(x: &[f32], rows: u32, m: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        let (rows, m) = (rows as usize, m as usize);
        if i >= rows * m {
            return;
        }
        let c = i / rows;
        // SAFETY: i < rows·m, so c < m and i − c·rows < rows: both indices are
        // below rows·m, inside x and y by the launch contract; thread i is
        // y[i]'s only writer.
        unsafe { *y.get_unchecked_mut(i) = *x.get_unchecked((i - c * rows) * m + c) };
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct TokenMajor {
    module: token_major_kernels::LoadedModule,
}

impl TokenMajor {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<TokenMajor, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(token_major_kernels, ctx)? };
        Ok(TokenMajor { module })
    }

    /// Enqueue the copy of `x`'s `m` row-major columns of `rows` values into
    /// `y` token-major. Asynchronous, allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        rows: usize,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "token_major::enqueue";
        if rows == 0 || m == 0 || x.len() < rows * m || y.len() < rows * m {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} columns of {rows}: x.len() {}, y.len() {}",
                    x.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (rows * m).div_ceil(256))?;
        let rows = launch_u32(what, "rows", rows)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_token_major(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.token_major(stream, &prep, x, rows, m, y)?;
        Ok(())
    }
}
