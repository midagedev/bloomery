//! A second `#[cuda_module]` in this crate, holding one near-empty kernel.
//! It answers two packaging questions by existing — a crate may carry more
//! than one device module, and a module may live in its own file — and it
//! gives the node-gap probe a body of one store per thread.

use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod probe_kernels {
    use super::*;

    /// `y[i] = 1.0` for `i < n`: the smallest body that is still a launch.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (y.len() >= n))]
    pub fn touch(n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = 1.0;
        }
    }
}

/// The loaded probe module.
pub struct Probe {
    module: probe_kernels::LoadedModule,
}

impl Probe {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Probe, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { probe_kernels::load(ctx)? };
        Ok(Probe { module })
    }

    /// Enqueue one `touch` over the first 32 elements of `y`.
    pub fn enqueue_touch(
        &self,
        stream: &CudaStream,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let prep = self.module.prepare_touch(LaunchConfig1D::new(1, 32, 0))?;
        self.module.touch(stream, &prep, 32, y)?;
        Ok(())
    }
}
