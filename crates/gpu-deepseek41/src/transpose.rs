//! The layout change of a multi-token pass: a projection over `m` tokens
//! writes row-major, `y[r·m + c]` (a row's `m` outputs together — the
//! K-quant gemvs' layout, [`crate::dense::DenseKernels::enqueue_m`]), and the
//! ops after it read token-major rows, `x[c·rows + r]` (ggml's order: the
//! norms, the ropes, the latent append, the pooling, the index key,
//! HC_POST). [`TransposeKernels::enqueue`] copies one into the other: an
//! exact copy, no arithmetic, so a token's values are the one-token
//! projection's bits in the order its one-token consumer reads them. A
//! one-token pass has one layout and launches none.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block: one value per thread.
const BLOCK: u32 = 256;

#[cuda_module]
mod transpose_kernels {
    use super::*;

    /// `dst[c·rows + r] = src[r·m + c]` for every `r < rows`, `c < m`: thread
    /// `i` writes `dst[i]`, so the stores are contiguous and each value has
    /// one writer.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (src.len() >= rows * m, dst.len() >= rows * m)
    )]
    pub fn ds41_rows_to_tokens(src: &[f32], rows: u32, m: u32, mut dst: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        let (rows, m) = (rows as usize, m as usize);
        if i >= rows * m {
            return;
        }
        let (c, r) = (i / rows, i % rows);
        // SAFETY: r < rows and c < m, so r·m + c < rows·m <= src.len() and i
        // < rows·m <= dst.len() by the launch contract; thread i is dst[i]'s
        // only writer.
        unsafe {
            *dst.get_unchecked_mut(i) = *src.get_unchecked(r * m + c);
        }
    }
}

/// The loaded module. Owns no stream: every enqueue takes the caller's, so
/// the copy orders with the launches around it.
pub struct TransposeKernels {
    module: transpose_kernels::LoadedModule,
}

impl TransposeKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<TransposeKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { transpose_kernels::load(ctx)? };
        Ok(TransposeKernels { module })
    }

    /// Enqueue the token-major copy of `src`, `rows` rows of `m` tokens
    /// row-major, into `dst`. Refused before the launch: no row or no token,
    /// and either buffer shorter than `rows · m`. Asynchronous,
    /// allocation-free.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        rows: usize,
        m: usize,
        dst: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "TransposeKernels::enqueue";
        let n = rows * m;
        if n == 0 || src.len() < n || dst.len() < n {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{rows} rows of {m} tokens from {} values into {}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", n.div_ceil(BLOCK as usize))?;
        let rows = launch_u32(what, "rows", rows)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_ds41_rows_to_tokens(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .ds41_rows_to_tokens(stream, &prep, src, rows, m, dst)?;
        Ok(())
    }
}
