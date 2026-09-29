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
//!
//! The host-side helpers at the end ([`BacklegPage`],
//! [`enqueue_wait_geq`], [`enqueue_write`], [`launch_window`]) are the gate
//! bins' reach into the stream memory operations and the host-mapped page —
//! `graph`'s own items are crate-private because only the engine should use
//! them freely. They change no engine path; a gate names them through
//! `bloomery_gpu::probe`.

use crate::GpuError;
use crate::graph::{MappedHost, mem_batch, op_wait_geq, op_write};
use crate::tensor::window;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, cooperative_launch, grid, kernel, launch_bounds, launch_contract, thread,
};
use cuda_host::cuda_module;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::Ordering;

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
    /// The kernel entry name this arm launches.
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
    /// Load this file's device bundle into `ctx`. Load-time only.
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

// --------------------------------------------------------- gate-bin helpers

/// A stream memory operation a gate's probe enqueues: one wait until the u32
/// at `word` is at least `value`, on any device-reachable address — a
/// host-mapped word or a VRAM one. `#[doc(hidden)]`: a probe helper of the
/// gate bins, not engine API.
#[doc(hidden)]
pub fn enqueue_wait_geq(
    stream: &CudaStream,
    word: cuda_core::sys::CUdeviceptr,
    value: u32,
) -> Result<(), GpuError> {
    let mut ops = [op_wait_geq(word, value)];
    mem_batch(stream, &mut ops, "cuStreamBatchMemOp_v2 (probe wait)")
}

/// A stream memory operation a gate's probe enqueues: one write of `value`
/// to the u32 at `word`, on any device-reachable address. `#[doc(hidden)]`:
/// a probe helper of the gate bins, not engine API.
#[doc(hidden)]
pub fn enqueue_write(
    stream: &CudaStream,
    word: cuda_core::sys::CUdeviceptr,
    value: u32,
) -> Result<(), GpuError> {
    let mut ops = [op_write(word, value)];
    mem_batch(stream, &mut ops, "cuStreamBatchMemOp_v2 (probe write)")
}

/// A non-owning launch window of `len` `T` at device address `ptr`: what a
/// gate's probe hands a launch for a span inside another allocation — a
/// host-mapped page, a copy's payload buffer. Never dropped, so it frees
/// nothing. `#[doc(hidden)]`: a probe helper of the gate bins, not engine
/// API. Safety: the span is device memory of `ctx`, and the owning
/// allocation outlives every launch handed the window.
#[doc(hidden)]
pub fn launch_window<T>(
    ptr: cuda_core::sys::CUdeviceptr,
    len: usize,
    ctx: &Arc<CudaContext>,
) -> ManuallyDrop<DeviceBuffer<T>> {
    // SAFETY: the caller meets the contract above.
    unsafe { window(ptr, len, ctx) }
}

/// The host-mapped page a back-leg probe runs its crossings over:
/// `values_len` f32 the host writes and the card reads through the mapping,
/// then two words of their own cache line each — a counter the stream's
/// wait holds on and an ack word the probe's kernel writes back. The
/// round-trip protocol: the host fills the values, bumps the counter
/// (Release), and spins on the ack (Acquire); the card, between a wait on
/// the counter and its ack write, reads the values exactly once. The
/// protocol, not the type, keeps the two sides off the values span — the
/// argument [`MappedHost`]'s `Sync` impl makes for its plain accesses.
/// `#[doc(hidden)]`: a probe helper of the gate bins, not engine API.
#[doc(hidden)]
pub struct BacklegPage {
    page: MappedHost,
    cnt_off: usize,
    ack_off: usize,
    values_len: usize,
}

/// Bytes each of the page's two flag words owns: a line of its own, so the
/// host's stores to one never share a line with the other.
const BACKLEG_WORD_LINE: usize = 64;

impl BacklegPage {
    /// A zeroed page of `ctx` for `values_len` f32 (at least one) and the
    /// two flag words behind them. Load-time only.
    pub fn new(ctx: &Arc<CudaContext>, values_len: usize) -> Result<BacklegPage, GpuError> {
        const WHAT: &str = "BacklegPage::new";
        if values_len == 0 {
            return Err(GpuError::shape(WHAT, "no values"));
        }
        let round_line = |bytes: usize| -> Option<usize> {
            Some(bytes.checked_add(BACKLEG_WORD_LINE - 1)? & !(BACKLEG_WORD_LINE - 1))
        };
        let values_bytes = values_len
            .checked_mul(4)
            .ok_or_else(|| GpuError::shape(WHAT, "the values span passes usize bytes"))?;
        let cnt_off = round_line(values_bytes)
            .ok_or_else(|| GpuError::shape(WHAT, "the values span passes the page"))?;
        let ack_off = cnt_off
            .checked_add(BACKLEG_WORD_LINE)
            .ok_or_else(|| GpuError::shape(WHAT, "the ack word's line passes the page"))?;
        let bytes = ack_off
            .checked_add(BACKLEG_WORD_LINE)
            .ok_or_else(|| GpuError::shape(WHAT, "the page passes usize bytes"))?;
        Ok(BacklegPage {
            page: MappedHost::new(ctx, bytes, "cuMemHostAlloc (backleg page)")?,
            cnt_off,
            ack_off,
            values_len,
        })
    }

    /// The values span's device address, byte 0 of the page.
    #[must_use]
    pub fn values_dev(&self) -> cuda_core::sys::CUdeviceptr {
        self.page.dev_at(0)
    }

    /// The counter word's device address.
    #[must_use]
    pub fn cnt_dev(&self) -> cuda_core::sys::CUdeviceptr {
        self.page.dev_at(self.cnt_off)
    }

    /// The ack word's device address.
    #[must_use]
    pub fn ack_dev(&self) -> cuda_core::sys::CUdeviceptr {
        self.page.dev_at(self.ack_off)
    }

    /// Fill the values span with `v`, between the ack of the previous round
    /// and the next bump — the owner's protocol keeps the card off the span
    /// meanwhile.
    pub fn fill_values(&self, v: f32) {
        // SAFETY: `values_len` f32 at byte 0, inside the allocation and
        // 4-aligned (a cuMemHostAlloc allocation is page-aligned); this page
        // hands its values span to no one else, so this is the only host
        // reference to it.
        unsafe {
            std::slice::from_raw_parts_mut(self.page.host_at(0).cast::<f32>(), self.values_len)
                .fill(v);
        }
    }

    /// Add one to the counter and answer its new value: the round's go.
    pub fn cnt_bump(&self) -> u32 {
        self.cnt_word().fetch_add(1, Ordering::Release) + 1
    }

    /// Set the counter far past anything a round bumps it to: every wait
    /// still pending in the stream passes, so a probe that stops driving
    /// mid-run drains instead of hanging.
    pub fn release(&self) {
        self.cnt_word().store(1 << 30, Ordering::Release);
    }

    /// The ack word as the host sees it.
    pub fn ack_value(&self) -> u32 {
        self.ack_word().load(Ordering::Acquire)
    }

    fn cnt_word(&self) -> &std::sync::atomic::AtomicU32 {
        self.page
            .atomic_u32(self.cnt_off)
            .expect("the counter word lies in the page, on its own line")
    }

    fn ack_word(&self) -> &std::sync::atomic::AtomicU32 {
        self.page
            .atomic_u32(self.ack_off)
            .expect("the ack word lies in the page, on its own line")
    }
}
