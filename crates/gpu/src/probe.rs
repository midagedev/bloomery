//! A second `#[cuda_module]` in this crate, holding one near-empty kernel.
//! It answers two packaging questions by existing — a crate may carry more
//! than one device module, and a module may live in its own file — and it
//! gives the node-gap probe a body of one store per thread.
//!
//! It also carries the grid-barrier trio ([`gap_plain`], [`gap_coop`],
//! [`gap_coop_sync`]). A fold that crosses a launch boundary needs a
//! grid-wide barrier, which needs a cooperative launch, and those are two
//! separate prices: the trio isolates each by holding everything else fixed
//! — the same 256-thread blocks (the block shape the candidate folds' gemv
//! producers run), the same grid, and the same two stores per thread. Phase
//! A stores into the low half and phase B into the high half so that no arm
//! loses a store to dead-store elimination; without that the barrier arm
//! would be paying for two stores and its twins for one. Subtracting across
//! the trio at one grid: `gap_coop − gap_plain` is what the cooperative
//! launch costs, `gap_coop_sync − gap_coop` is what the barrier costs, and
//! `gap_plain` itself is a node of that shape.

use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, cooperative_launch, grid, kernel, launch_bounds, launch_contract, thread,
};
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

    /// Two phases inside ONE launch, ordered by a grid-wide barrier: every
    /// block writes 1.0, all blocks meet at `grid::sync`, then every block
    /// writes 3.0. Cooperative launch is what makes the barrier legal — all
    /// blocks are co-resident — and is the ordering primitive a fused block
    /// kernel needs between its stages.
    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (y.len() >= n))]
    pub fn two_phase(n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: i < n <= y.len() by the launch contract.
            unsafe {
                *y.get_unchecked_mut(i) = 1.0;
            }
        }
        grid::sync();
        if i < n as usize {
            // SAFETY: i < n <= y.len() by the launch contract; the grid
            // barrier above published the first store.
            unsafe {
                *y.get_unchecked_mut(i) = 3.0;
            }
        }
    }

    /// The grid-barrier trio's control arm: an ordinary launch, [`GAP_THREADS`]
    /// threads per block, two stores per thread into disjoint halves of `y`
    /// and nothing between them. Its replay time is one graph node of this
    /// block shape.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() >= 2 * n))]
    pub fn gap_plain(n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n and n + i < 2n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = 1.0;
            *y.get_unchecked_mut(n as usize + i) = 3.0;
        }
    }

    /// The same two stores under a cooperative launch and no barrier: the
    /// arm that separates what the launch kind costs from what the barrier
    /// costs.
    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() >= 2 * n))]
    pub fn gap_coop(n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n and n + i < 2n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = 1.0;
            *y.get_unchecked_mut(n as usize + i) = 3.0;
        }
    }

    /// The same two stores with one `grid::sync` between them. The early
    /// return is block-uniform, so no block reaches the barrier alone.
    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() >= 2 * n))]
    pub fn gap_coop_sync(n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < n as usize {
            // SAFETY: i < n <= y.len() by the launch contract.
            unsafe {
                *y.get_unchecked_mut(i) = 1.0;
            }
        }
        grid::sync();
        if i < n as usize {
            // SAFETY: n + i < 2n <= y.len() by the launch contract.
            unsafe {
                *y.get_unchecked_mut(n as usize + i) = 3.0;
            }
        }
    }
}

/// Threads per block of the grid-barrier trio, matching the literal in each
/// of the three launch contracts. It is the block shape of the gemv
/// producers a fold would have to absorb (one warp per row, eight rows per
/// block), so a price measured here is a price in their geometry.
pub const GAP_THREADS: u32 = 256;

/// Which arm of the grid-barrier trio to enqueue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapArm {
    /// Ordinary launch, no barrier.
    Plain,
    /// Cooperative launch, no barrier.
    Coop,
    /// Cooperative launch with one `grid::sync` between the two stores.
    CoopSync,
}

impl GapArm {
    pub fn name(self) -> &'static str {
        match self {
            GapArm::Plain => "gap_plain",
            GapArm::Coop => "gap_coop",
            GapArm::CoopSync => "gap_coop_sync",
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

    /// Enqueue one cooperative `two_phase` over `blocks` 32-thread blocks
    /// (`y.len() >= blocks * 32`).
    pub fn enqueue_two_phase(
        &self,
        stream: &CudaStream,
        blocks: u32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let prep = self
            .module
            .prepare_two_phase(LaunchConfig1D::new(blocks, 32, 0))?;
        self.module.two_phase(stream, &prep, blocks * 32, y)?;
        Ok(())
    }

    /// Enqueue one arm of the grid-barrier trio over `blocks`
    /// [`GAP_THREADS`]-thread blocks. `y` holds both halves, so
    /// `y.len() >= 2 * blocks * GAP_THREADS`; every arm writes `1.0` into
    /// the low half and `3.0` into the high one, which is what a reader
    /// checks to know the barrier arm actually reached phase B.
    pub fn enqueue_gap(
        &self,
        stream: &CudaStream,
        arm: GapArm,
        blocks: u32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig1D::new(blocks, GAP_THREADS, 0);
        let n = blocks * GAP_THREADS;
        match arm {
            GapArm::Plain => {
                let prep = self.module.prepare_gap_plain(cfg)?;
                self.module.gap_plain(stream, &prep, n, y)?;
            }
            GapArm::Coop => {
                let prep = self.module.prepare_gap_coop(cfg)?;
                self.module.gap_coop(stream, &prep, n, y)?;
            }
            GapArm::CoopSync => {
                let prep = self.module.prepare_gap_coop_sync(cfg)?;
                self.module.gap_coop_sync(stream, &prep, n, y)?;
            }
        }
        Ok(())
    }
}
