//! The remapped route table: [`GemmKernels::enqueue_route`]'s table built
//! from ids that first go through a per-expert map — `map[id]` the expert's
//! slot in the card's stack, or [`HOST`] for an expert the host tier
//! computes. Built on the card by `gemm_route_remap` (declared in
//! `kernels32.rs`), read by every GEMM over the same slots and stack, of
//! either family.
//!
//! Slots that carry their place in the stack instead of an expert id (an
//! expert tier's block) take the same table through [`place_map`]: their
//! ids are the places, a [`HOST`] place the one id past the stack
//! (`gemm_place_ids`, [`Gemm32Kernels::enqueue_place_ids`]), which that map
//! sends back to the place, or to [`HOST`].
//!
//! [`GemmKernels::enqueue_route`]: super::GemmKernels::enqueue_route

use super::GemmRoute;
use super::kernels32::Gemm32Kernels;
use super::route::{GEMM_MAX_EXPERTS, NO_EXPERT, ROUTE_THREADS, gemm_max_tiles};
use crate::fault::FaultSink;
use crate::hybrid::HOST;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// [`ROUTE_THREADS`] as the block width a launch takes.
const REMAP_THREADS_U32: u32 = ROUTE_THREADS as u32;
const _: () = assert!(REMAP_THREADS_U32 as usize == ROUTE_THREADS);
/// The block width of `gemm_place_ids`, which its launch contract spells.
const PLACE_THREADS: u32 = 256;

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

/// The map that routes a stack of `n_experts` experts by place: entries
/// `0 .. n_experts` the places themselves, then [`HOST`] — so the ids
/// [`Gemm32Kernels::enqueue_place_ids`] writes reach the stack slot each
/// place names, a [`HOST`] place (id `n_experts`) reaches the host, and a
/// place in `[n_experts, HOST)` is an id past the map, which the route
/// refuses and raises. A stack past a route table's `GEMM_MAX_EXPERTS` is
/// refused by name. Load-time only.
pub fn place_map(n_experts: usize) -> Result<Vec<u32>, GpuError> {
    if !(1..=GEMM_MAX_EXPERTS).contains(&n_experts) {
        return Err(GpuError::shape(
            "gemm::place_map",
            format!("a stack of {n_experts} experts: a route table takes 1 to {GEMM_MAX_EXPERTS}"),
        ));
    }
    let n = launch_u32("gemm::place_map", "n_experts", n_experts)?;
    Ok((0..n).chain(std::iter::once(HOST)).collect())
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

    /// Enqueue the remapped route's ids of `n_slots` slots that carry a
    /// place in a stack of `n_experts` experts, not an expert id
    /// (`gemm_place_ids`): a place as it is, [`HOST`] as `n_experts`, the ids
    /// [`place_map`]'s map of that stack sends back to each place. No slot,
    /// a stack of no expert, and buffers short of `n_slots` are refused by
    /// name. One launch; asynchronous, allocation-free, capturable.
    pub fn enqueue_place_ids(
        &self,
        stream: &CudaStream,
        sel: &DeviceBuffer<u32>,
        (n_slots, n_experts): (usize, usize),
        ids: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_place_ids";
        if n_slots == 0 || n_experts == 0 || sel.len() < n_slots || ids.len() < n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots of a stack of {n_experts} experts from {} places into {} \
                     ids: at least one of each, both buffers long enough",
                    sel.len(),
                    ids.len()
                ),
            ));
        }
        let n = launch_u32(what, "n_slots", n_slots)?;
        let n_experts = launch_u32(what, "n_experts", n_experts)?;
        let grid = n.div_ceil(PLACE_THREADS);
        let prep =
            self.module
                .prepare_gemm_place_ids(LaunchConfig1D::new(grid, PLACE_THREADS, 0))?;
        self.module
            .gemm_place_ids(stream, &prep, sel, n, n_experts, ids)?;
        Ok(())
    }
}
