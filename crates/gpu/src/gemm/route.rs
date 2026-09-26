//! The route table: the slots grouped by expert, ascending slot order within
//! an expert, cut into tiles of at most [`GEMM_BN`] columns, built on the
//! card by `gemm_route` (declared in `kernels.rs`) and read by every GEMM
//! over the same slots and stack.

use super::{GEMM_BN, GEMM_MAX_SLOTS, GemmKernels};
use crate::fault::FaultSink;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// Experts one stack may have — the route kernel's counters live in shared
/// memory, one per (chunk, expert), at least one chunk's.
pub(super) const GEMM_MAX_EXPERTS: usize = 1024;
/// Threads of the route kernel.
pub(super) const ROUTE_THREADS: usize = 1024;
/// [`ROUTE_THREADS`] as the block width a launch takes.
const ROUTE_THREADS_U32: u32 = ROUTE_THREADS as u32;
/// Warps of the route kernel: the most chunks the slots are cut into.
pub(super) const ROUTE_WARPS: usize = ROUTE_THREADS / 32;
/// u32 words of the route kernel's per-chunk counts, one per (chunk,
/// expert): a stack of `e` experts is cut into `min(ROUTE_WARPS,
/// ROUTE_HIST / e)` chunks.
pub(super) const ROUTE_HIST: usize = 8192;
/// A staged id the route refused (past the stack) or a lane past the slots.
pub(super) const NO_EXPERT: u32 = u32::MAX;

const _: () = assert!(ROUTE_THREADS_U32 as usize == ROUTE_THREADS);
// `gemm_route`'s launch bounds and launch contract spell the block out.
const _: () = assert!(ROUTE_THREADS == 1024);
const _: () = assert!(GEMM_MAX_EXPERTS <= ROUTE_THREADS);
// At the expert cap the counts still hold one chunk; at the chunk cap one
// warp per chunk.
const _: () = assert!(ROUTE_HIST >= GEMM_MAX_EXPERTS && ROUTE_WARPS * 32 == ROUTE_THREADS);
// The route packs a count and a tile count into one scanned u32, 16 bits
// each, and a tile's start and length into another (`tile_word`).
const _: () = assert!(GEMM_MAX_SLOTS < 1 << 16 && GEMM_MAX_EXPERTS + GEMM_MAX_SLOTS < 1 << 16);

/// A tile's word in the route table: its first index into the slot list in
/// the low 16 bits, its length in the high 16.
#[inline(always)]
pub(super) const fn tile_word(start: u32, len: u32) -> u32 {
    start | (len << 16)
}

/// A [`tile_word`]'s two halves, `(start, len)`.
#[inline(always)]
pub(super) const fn tile_parts(word: u32) -> (u32, u32) {
    (word & 0xffff, word >> 16)
}

/// The most tiles `n_slots` slots over `n_experts` experts can make: an
/// expert with `c > 0` slots makes `ceil(c / GEMM_BN)` tiles, which is at
/// most `(c - 1) / GEMM_BN` plus one, so the sum is at most
/// `n_slots / GEMM_BN` plus the count of experts that got a slot.
#[must_use]
pub(super) fn gemm_max_tiles(n_slots: usize, n_experts: usize) -> usize {
    n_slots / GEMM_BN + n_experts.min(n_slots)
}

/// Slot `s`'s expert in the route's walks over a chunk that ends at `hi`:
/// [`NO_EXPERT`] for a lane past the chunk, and for an id with no expert in
/// the stack's first `e_n` — then with `true`, the slot the count walk
/// raises.
///
/// # Safety
///
/// `hi <= ids.len()`.
#[inline(always)]
pub(super) unsafe fn route_id(
    ids: &[u32],
    s: usize,
    hi: usize,
    n_experts: u32,
    e_n: usize,
) -> (u32, bool) {
    if s >= hi {
        return (NO_EXPERT, false);
    }
    // SAFETY: s < hi <= ids.len() by this fn's contract.
    let id = unsafe { *ids.get_unchecked(s) };
    if id < n_experts && (id as usize) < e_n {
        (id, false)
    } else {
        (NO_EXPERT, true)
    }
}

/// One tile of a route table as the host reads it back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmTile {
    /// The tile's expert.
    pub expert: u32,
    /// Its first index into the slot list.
    pub start: u32,
    /// Its slot count, 1..=[`GEMM_BN`].
    pub len: u32,
}

/// The device route table one [`GemmKernels::enqueue_route`] fills and any
/// number of GEMMs over the same slots and stack read: the slot list grouped
/// by expert with the refused slots at its end, the tiles, the tile count
/// and the refused count. Remembers the slot count and
/// stack it was last filled for, so a GEMM over another count, another
/// stack or an unfilled table is a named error rather than a launch that
/// computes nothing.
pub struct GemmRoute {
    pub(super) cols: DeviceBuffer<u32>,
    pub(super) tiles: DeviceBuffer<u32>,
    pub(super) n_tiles: DeviceBuffer<u32>,
    /// All-zero ids for the dense table (a one-expert route).
    zeros: Option<DeviceBuffer<u32>>,
    max_slots: usize,
    pub(super) n_experts: usize,
    /// The slot count of the last enqueued fill.
    pub(super) filled: Option<usize>,
}

impl GemmRoute {
    /// A table for up to `max_slots` (1..=[`GEMM_MAX_SLOTS`]) slots over a
    /// stack of `n_experts` (1..=[`GEMM_MAX_EXPERTS`]) experts. Load-time
    /// only.
    pub fn new(
        stream: &CudaStream,
        max_slots: usize,
        n_experts: usize,
    ) -> Result<GemmRoute, GpuError> {
        let what = "GemmRoute::new";
        if !(1..=GEMM_MAX_SLOTS).contains(&max_slots) {
            return Err(GpuError::shape(
                what,
                format!("1 <= max_slots <= {GEMM_MAX_SLOTS}, got {max_slots}"),
            ));
        }
        if !(1..=GEMM_MAX_EXPERTS).contains(&n_experts) {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_experts <= {GEMM_MAX_EXPERTS}, got {n_experts}"),
            ));
        }
        let zeros = if n_experts == 1 {
            Some(DeviceBuffer::zeroed(stream, max_slots)?)
        } else {
            None
        };
        Ok(GemmRoute {
            cols: DeviceBuffer::zeroed(stream, max_slots)?,
            tiles: DeviceBuffer::zeroed(stream, 2 * gemm_max_tiles(max_slots, n_experts))?,
            n_tiles: DeviceBuffer::zeroed(stream, 2)?,
            zeros,
            max_slots,
            n_experts,
            filled: None,
        })
    }

    /// The stack's expert count this table routes over.
    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Device bytes of the table.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.cols.num_bytes()
            + self.tiles.num_bytes()
            + self.n_tiles.num_bytes()
            + self.zeros.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }

    /// The table as it stands on the card: the slot list and the tiles. A
    /// blocking read on `stream`, for gates and diagnostics.
    pub fn read_back(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<GemmTile>), GpuError> {
        let n_slots = self
            .filled
            .ok_or_else(|| GpuError::state("GemmRoute::read_back", "the table was never filled"))?;
        let n = self.n_tiles.to_host_vec(stream)?[0] as usize;
        let raw = self.tiles.to_host_vec(stream)?;
        if 2 * n > raw.len() {
            return Err(GpuError::shape(
                "GemmRoute::read_back",
                format!("tile count {n} exceeds the table's {} tiles", raw.len() / 2),
            ));
        }
        let tiles = raw[..2 * n]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| {
                let (start, len) = tile_parts(p[1]);
                GemmTile {
                    expert: p[0],
                    start,
                    len,
                }
            })
            .collect();
        let mut cols = self.cols.to_host_vec(stream)?;
        cols.truncate(n_slots);
        Ok((cols, tiles))
    }

    /// The slots the last fill refused (an expert id past the stack), in
    /// ascending slot order — the end of the slot list. A blocking read on
    /// `stream`, for gates and diagnostics.
    pub fn refused_back(&self, stream: &CudaStream) -> Result<Vec<u32>, GpuError> {
        let what = "GemmRoute::refused_back";
        let n_slots = self
            .filled
            .ok_or_else(|| GpuError::state(what, "the table was never filled"))?;
        let refused = self.n_tiles.to_host_vec(stream)?[1] as usize;
        if refused > n_slots {
            return Err(GpuError::shape(
                what,
                format!("refused count {refused} exceeds the table's {n_slots} slots"),
            ));
        }
        let cols = self.cols.to_host_vec(stream)?;
        Ok(cols[n_slots - refused..n_slots].to_vec())
    }
}

impl GemmKernels {
    /// Enqueue the route table for `n_slots` slots whose expert ids are
    /// `ids[0..n_slots]` (slot `token * top_k + k`), read on the card: no
    /// host synchronisation, capturable. An id at or past the table's expert
    /// count raises [`FaultSite::ExpertId`] on `fault` and its slot is left
    /// out of the tiles and listed as refused, so every GEMM over the table
    /// writes NaN to its outputs — the step that reads the fault word back
    /// turns it into `GpuError::Fault`.
    ///
    /// [`FaultSite::ExpertId`]: crate::FaultSite::ExpertId
    pub fn enqueue_route(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        n_slots: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_route";
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
        self.route_launch(stream, ids, n_slots, route, fault)
    }

    /// Enqueue the dense table: `n_cols` slots, all on expert 0 of a
    /// one-expert table, slot `s` reading column `s` — the GEMM is then a
    /// plain `y = W · x` over `n_cols` columns.
    pub fn enqueue_route_dense(
        &self,
        stream: &CudaStream,
        n_cols: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_route_dense";
        if route.n_experts != 1 {
            return Err(GpuError::shape(
                what,
                format!("a dense table has one expert, this one {}", route.n_experts),
            ));
        }
        if n_cols == 0 || n_cols > route.max_slots {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= {}, got {n_cols}", route.max_slots),
            ));
        }
        let zeros = route
            .zeros
            .take()
            .ok_or_else(|| GpuError::state(what, "the one-expert table's zero ids"))?;
        let r = self.route_launch(stream, &zeros, n_cols, route, fault);
        route.zeros = Some(zeros);
        r
    }

    /// The one launcher of `gemm_route`.
    fn route_launch(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        n_slots: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_route";
        let max_tiles = gemm_max_tiles(n_slots, route.n_experts);
        let n_slots_u = launch_u32(what, "n_slots", n_slots)?;
        let n_experts = launch_u32(what, "n_experts", route.n_experts)?;
        let max_tiles = launch_u32(what, "max_tiles", max_tiles)?;
        let prep = self
            .module
            .prepare_gemm_route(LaunchConfig1D::new(1, ROUTE_THREADS_U32, 0))?;
        self.module.gemm_route(
            stream,
            &prep,
            ids,
            n_slots_u,
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
