//! The wide F32 product: an F32 weight of any row count against up to a
//! ubatch of f32 columns, as register tiles — `f32_tile_gemm` (declared in
//! `kernels32.rs`), for the F32 projections a wide prompt call runs (a
//! linear-attention layer's β·α rows, a selecting attention's indexer
//! projections) where the decode path runs `q8f32::f32_gemv`.
//!
//! Contract: every output is bit for bit `f32_gemv`'s for its (row,
//! column). The body is `qwen3moe_router_logits`' with the row count a
//! launch argument: a block of sixteen warps covers [`TILE_COLS`] columns
//! and [`TILE_ROWS`] rows, each warp eight rows by eight columns, the rows'
//! and columns' next 64 values staged in shared memory with `cp.async`,
//! three stages deep; each lane advances its 64 sums over the staged 32-value
//! chunks in increasing order (`q8f32::f32_tile_chunk`), which takes each
//! (row, column) pair's terms in `f32_gemv`'s row walk order (`q8f32.rs`'s
//! module doc says why that keeps its bits), and each column's eight sums go
//! through the same butterfly (`q8f32::gemv_lane_sums`). A row past the last
//! is staged as the last row and a column past the last as the last column;
//! their sums are not stored. The output is column-major, `y[c · rows + r]`
//! — the GEMM families' `y[s][n]`, not `f32_gemv`'s `y[r · m + c]`.

use super::kernels32::Gemm32Kernels;
use crate::q8f32::{TILE, f32_tile_chunk, gemv_lane_sums};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::{DisjointSlice, thread, warp};

/// Columns and rows one block covers.
pub(super) const TILE_COLS: usize = 32;
pub(super) const TILE_ROWS: usize = 32;
/// Threads of a block: sixteen warps, [`TILE_ROWS`]` / 8` row groups by
/// [`TILE_COLS`]` / 8` column groups.
pub(super) const F32T_THREADS: usize = 512;
const F32T_THREADS_U32: u32 = F32T_THREADS as u32;
const ROW_GROUPS: usize = TILE_ROWS / TILE;
const COL_GROUPS: usize = TILE_COLS / TILE;
/// 32-value chunks one stage holds of every row and column, and the values
/// of one such line: K must be a multiple of it.
const STAGE_CHUNKS: usize = 2;
pub(super) const LINE: usize = 32 * STAGE_CHUNKS;
/// Stages in shared memory: the one the warps read and two being copied in.
pub(super) const STAGES: usize = 3;
/// One stage: every row's line, then every column's.
pub(super) const STAGE_FLOATS: usize = (TILE_ROWS + TILE_COLS) * LINE;
/// 16-byte copies per line.
const LINE_COPIES: usize = LINE / 4;

const _: () = assert!(ROW_GROUPS * TILE == TILE_ROWS && COL_GROUPS * TILE == TILE_COLS);
const _: () = assert!(ROW_GROUPS * COL_GROUPS * 32 == F32T_THREADS && F32T_THREADS == 512);
// Every thread copies one piece of one row line and of one column line.
const _: () = assert!(TILE_ROWS * LINE_COPIES == F32T_THREADS && TILE_COLS == TILE_ROWS);
const _: () = assert!(4 * STAGES * STAGE_FLOATS <= 48 * 1024);

/// Lane 0's stores of the first `live` of one column's [`TILE`] row sums,
/// `y[base + i] = sums[i]`, each store guarded by a constant index.
///
/// # Safety
///
/// `live <= TILE`, `base + live <= y.len()`, and no other thread writes
/// those slots.
#[inline(always)]
unsafe fn store_live(y: &mut DisjointSlice<f32>, base: usize, sums: &[f32; TILE], live: usize) {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each store is guarded by i < live, so its slot base + i lies
    // inside y and belongs to this thread alone by this fn's contract.
    unsafe {
        if live > 0 {
            *y.get_unchecked_mut(base) = s0;
        }
        if live > 1 {
            *y.get_unchecked_mut(base + 1) = s1;
        }
        if live > 2 {
            *y.get_unchecked_mut(base + 2) = s2;
        }
        if live > 3 {
            *y.get_unchecked_mut(base + 3) = s3;
        }
        if live > 4 {
            *y.get_unchecked_mut(base + 4) = s4;
        }
        if live > 5 {
            *y.get_unchecked_mut(base + 5) = s5;
        }
        if live > 6 {
            *y.get_unchecked_mut(base + 6) = s6;
        }
        if live > 7 {
            *y.get_unchecked_mut(base + 7) = s7;
        }
    }
}

/// The product of `n` columns over `rows` weight rows (module doc): block
/// `b` covers columns `32·(b / B) ..` and rows `32·(b % B) ..`, `B =
/// ⌈rows / 32⌉`; writes `y[c · rows + r]`.
///
/// # Safety
///
/// Every thread of the block calls it, converged; the block is
/// [`F32T_THREADS`] wide and the grid `⌈n / 32⌉ · B` blocks; `sh` is the
/// block's own `STAGES · STAGE_FLOATS` shared values, 16-byte aligned;
/// `rows >= 1`, `n >= 1`, `k` a positive multiple of [`LINE`], `w.len() >=
/// rows·k`, `x.len() >= n·k`, both 16-byte aligned, `y.len() >= rows·n`.
#[inline(always)]
pub(super) unsafe fn f32_tile_body(
    w: &[f32],
    x: &[f32],
    rows: u32,
    k: u32,
    n: u32,
    y: &mut DisjointSlice<f32>,
    sh: *mut f32,
) {
    let rows = rows as usize;
    let n = n as usize;
    let row_blocks = rows.div_ceil(TILE_ROWS);
    let b = thread::blockIdx_x() as usize;
    let c0 = (b / row_blocks) * TILE_COLS;
    let row0 = (b % row_blocks) * TILE_ROWS;
    if c0 >= n {
        return; // block-uniform, before the first barrier
    }
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let (rg, cg) = ((tid / 32) / COL_GROUPS, (tid / 32) % COL_GROUPS);
    let kk = k as usize;
    let stages = kk / LINE;

    // This thread's copies of every stage: piece `piece` of row line `line`
    // and of column line `line`, a row past the last staged as the last and
    // a column past the last as the last.
    let line = tid / LINE_COPIES;
    let piece = 4 * (tid % LINE_COPIES);
    let w_off = (row0 + line).min(rows - 1) * kk + piece;
    let x_off = (c0 + line).min(n - 1) * kk + piece;
    let w_dst = line * LINE + piece;
    let x_dst = (TILE_ROWS + line) * LINE + piece;
    macro_rules! stage_copy {
        ($s:expr, $buf:expr) => {{
            let (s, buf): (usize, usize) = ($s, $buf);
            // SAFETY: s < stages, so s·LINE + piece + 4 <= k: the 16 source
            // bytes lie in a row below `rows` of w and a column below `n` of
            // x, 16-byte aligned (k a multiple of 64, both bases 16-byte
            // aligned: this fn's contract). The destination is this thread's
            // own 16 bytes of stage buf < STAGES, 16-byte aligned, read by no
            // warp until the barrier after the wait that completes this copy.
            unsafe {
                let dst = sh.add(buf * STAGE_FLOATS);
                cp_async_cg_16(
                    dst.add(w_dst).cast::<u32>(),
                    w.as_ptr().add(w_off + s * LINE).cast::<u32>(),
                );
                cp_async_cg_16(
                    dst.add(x_dst).cast::<u32>(),
                    x.as_ptr().add(x_off + s * LINE).cast::<u32>(),
                );
            }
        }};
    }

    stage_copy!(0, 0);
    // SAFETY: commits this thread's copies above as one group.
    unsafe { cp_async_commit_group() };
    if stages > 1 {
        stage_copy!(1, 1);
    }
    // SAFETY: commits this thread's copies of stage 1 as the second group,
    // empty when `stages` is 1; the loop's wait counts it as stage 1's either
    // way.
    unsafe { cp_async_commit_group() };

    let mut acc = [[0.0f32; TILE]; TILE];
    let mut s = 0usize;
    let mut buf = 0usize;
    while s < stages {
        // SAFETY: one group per stage was committed before this wait (two
        // ahead of the loop, one per earlier pass), so leaving the newest
        // pending completes stage s's. The barrier then publishes every
        // thread's copies of it, and tells that every warp is done reading
        // stage s − 1, whose buffer the copy below refills.
        unsafe { cp_async_wait_group(1) };
        thread::sync_threads();
        if s + 2 < stages {
            stage_copy!(s + 2, if buf == 0 { STAGES - 1 } else { buf - 1 });
        }
        // SAFETY: one group per pass, empty near the end.
        unsafe { cp_async_commit_group() };
        // SAFETY: stage buf's row lines rg·TILE + i and column lines
        // TILE_ROWS + cg·TILE + c (i, c < TILE) hold values 32·j + lane of
        // chunks j < STAGE_CHUNKS, published by the barrier above and not
        // refilled before the next pass's barrier.
        unsafe {
            let base = sh.add(buf * STAGE_FLOATS).cast_const();
            let wp = base.add(rg * TILE * LINE + lane);
            let xp = base.add((TILE_ROWS + cg * TILE) * LINE + lane);
            acc = f32_tile_chunk(acc, wp, xp, LINE);
            acc = f32_tile_chunk(acc, wp.add(32), xp.add(32), LINE);
        }
        s += 1;
        buf = if buf == STAGES - 1 { 0 } else { buf + 1 };
    }

    // The warp's first column, how many of its columns exist, its first row
    // and how many of its rows exist.
    let cw = c0 + cg * TILE;
    let live = if cw < n { (n - cw).min(TILE) } else { 0 };
    let r0 = row0 + rg * TILE;
    let live_rows = if r0 < rows { (rows - r0).min(TILE) } else { 0 };
    // Every lane runs each column's butterfly; lane 0 stores a live one's
    // live rows.
    macro_rules! column {
        ($c:literal) => {{
            let sums = gemv_lane_sums(acc[$c], TILE as u32);
            if lane == 0 && $c < live {
                // SAFETY: column cw + $c < n, so its slots (cw + $c)·rows +
                // r0 + i, i < live_rows, lie inside y (this fn's contract);
                // lane 0 of the warp owning rows r0 .. of that column is their
                // only writer.
                unsafe { store_live(y, (cw + $c) * rows + r0, &sums, live_rows) };
            }
        }};
    }
    column!(0);
    column!(1);
    column!(2);
    column!(3);
    column!(4);
    column!(5);
    column!(6);
    column!(7);
}

impl Gemm32Kernels {
    /// Enqueue `y[c · rows + r] = Σ_k w[r·k + k'] · x[c·k + k']` for the
    /// first `n` columns of `x`, `w` an F32 weight of `rows × k` (module
    /// doc): every output bit for bit `Q8F32Kernels::enqueue_f32_gemv`'s for
    /// its row and column. `k` must be a positive multiple of 64 (the
    /// staged line); another is a named `Shape` error. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_f32_tile(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_f32_tile";
        let (rows, k) = (w.rows(), w.cols());
        if k == 0 || !k.is_multiple_of(LINE) {
            return Err(GpuError::shape(
                what,
                format!("k must be a positive multiple of {LINE} (the staged line), got {k}"),
            ));
        }
        if rows == 0 || n == 0 {
            return Err(GpuError::shape(
                what,
                format!("rows and n must be positive, got {rows} and {n}"),
            ));
        }
        if w.buf().len() < rows * k || x.len() < n * k || y.len() < rows * n {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.len() {} >= rows*k = {rows}*{k}, x.len() {} >= n*k = {n}*{k}, y.len() {} \
                     >= rows*n = {rows}*{n}",
                    w.buf().len(),
                    x.len(),
                    y.len()
                ),
            ));
        }
        let grid = n.div_ceil(TILE_COLS) * rows.div_ceil(TILE_ROWS);
        let grid = launch_u32(what, "grid", grid)?;
        let rows = launch_u32(what, "rows", rows)?;
        let k = launch_u32(what, "k", k)?;
        let n = launch_u32(what, "n", n)?;
        let prep =
            self.module
                .prepare_f32_tile_gemm(LaunchConfig1D::new(grid, F32T_THREADS_U32, 0))?;
        self.module
            .f32_tile_gemm(stream, &prep, w.buf(), x, rows, k, n, y)?;
        Ok(())
    }
}
