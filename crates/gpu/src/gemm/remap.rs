//! The remapped route table: [`GemmKernels::enqueue_route`]'s table built
//! from ids that first go through a per-expert map — `map[id]` the expert's
//! slot in the card's stack, or [`HOST`] for an expert the host tier
//! computes. Built on the card by `gemm_route_remap` (declared in
//! `kernels32.rs`), read by every GEMM over the same slots and stack, of
//! either family.
//!
//! [`GemmKernels::enqueue_route`]: super::GemmKernels::enqueue_route

use super::GemmRoute;
use super::kernels32::Gemm32Kernels;
use super::route::{NO_EXPERT, ROUTE_THREADS, gemm_max_tiles};
use crate::fault::FaultSink;
use crate::hybrid::HOST;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// [`ROUTE_THREADS`] as the block width a launch takes.
const REMAP_THREADS_U32: u32 = ROUTE_THREADS as u32;
const _: () = assert!(REMAP_THREADS_U32 as usize == ROUTE_THREADS);

/// Slot `s`'s expert in the remapped route's walks over a chunk that ends at
/// `hi`: [`NO_EXPERT`] with `false` for a lane past the chunk and for a slot
/// whose id maps to [`HOST`] (the host serves it: no tile, not refused);
/// the mapped slot `map[id]` of the card's stack when it is below the
/// stack's first `e_n` experts; and [`NO_EXPERT`] with `true` — the slot the
/// count walk raises and the table lists as refused — for an id at or past
/// the map (`n_map`) or a map value at or past the stack that is not
/// [`HOST`].
///
/// # Safety
///
/// `hi <= ids.len()` and `n_map <= map.len()`.
#[inline(always)]
pub(super) unsafe fn remap_id(
    ids: &[u32],
    map: &[u32],
    s: usize,
    hi: usize,
    n_map: u32,
    n_experts: u32,
    e_n: usize,
) -> (u32, bool) {
    if s >= hi {
        return (NO_EXPERT, false);
    }
    // SAFETY: s < hi <= ids.len() by this fn's contract.
    let id = unsafe { *ids.get_unchecked(s) };
    if id >= n_map {
        return (NO_EXPERT, true);
    }
    // SAFETY: id < n_map <= map.len() by this fn's contract.
    let v = unsafe { *map.get_unchecked(id as usize) };
    if v == HOST {
        (NO_EXPERT, false)
    } else if v < n_experts && (v as usize) < e_n {
        (v, false)
    } else {
        (NO_EXPERT, true)
    }
}

impl Gemm32Kernels {
    /// Enqueue the route table for `n_slots` slots whose expert ids are
    /// `ids[0..n_slots]` (slot `token · top_k + k`), each id mapped through
    /// `map` (`map.len()` experts, entry `e` the expert's slot in the stack
    /// the table routes over, or [`HOST`]), read on the card: no host
    /// synchronisation, capturable. The table is
    /// [`GemmKernels::enqueue_route`]'s over the mapped ids, less the slots
    /// that map to [`HOST`]: those are neither in a tile nor refused, so
    /// every GEMM over the table leaves their output rows as they were. An id
    /// at or past `map.len()`, or a map value at or past the table's expert
    /// count that is not [`HOST`], raises [`FaultSite::ExpertId`] on `fault`
    /// and lists its slot as refused (NaN outputs).
    ///
    /// The slot list then holds the listed slots at its start and the
    /// refused ones at its end (`GemmRoute::refused_back`); the entries
    /// between them — as many as there are host slots — are not written by
    /// this fill and hold nothing a reader may use.
    ///
    /// [`GemmKernels::enqueue_route`]: super::GemmKernels::enqueue_route
    /// [`FaultSite::ExpertId`]: crate::FaultSite::ExpertId
    pub fn enqueue_route_remap(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        map: &DeviceBuffer<u32>,
        n_slots: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_route_remap";
        if n_slots == 0 || n_slots > route.max_slots || ids.len() < n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "1 <= n_slots <= the table's {} slots and ids.len() {} >= n_slots, got \
                     {n_slots}",
                    route.max_slots,
                    ids.len()
                ),
            ));
        }
        if map.is_empty() || map.len() >= HOST as usize {
            return Err(GpuError::shape(
                what,
                format!("the map holds 1 to {} experts, got {}", HOST - 1, map.len()),
            ));
        }
        let max_tiles = gemm_max_tiles(n_slots, route.n_experts);
        let n_slots_u = launch_u32(what, "n_slots", n_slots)?;
        let n_map = launch_u32(what, "n_map", map.len())?;
        let n_experts = launch_u32(what, "n_experts", route.n_experts)?;
        let max_tiles = launch_u32(what, "max_tiles", max_tiles)?;
        let prep =
            self.module
                .prepare_gemm_route_remap(LaunchConfig1D::new(1, REMAP_THREADS_U32, 0))?;
        self.module.gemm_route_remap(
            stream,
            &prep,
            ids,
            map,
            n_slots_u,
            n_map,
            n_experts,
            max_tiles,
            &mut route.cols,
            &mut route.tiles,
            &mut route.n_tiles,
            fault,
        )?;
        route.filled = Some(n_slots);
        Ok(())
    }
}
