//! The wide F32 product: an F32 weight of any row count against up to a
//! ubatch of f32 columns, as register tiles — `f32_tile_gemm` (declared in
//! `kernels32.rs`), for the F32 projections a wide prompt call runs (a
//! linear-attention layer's β·α rows, a selecting attention's indexer
//! projections) where the decode path runs `q8f32::f32_gemv`.
//!
//! Contract: every output is bit for bit `f32_gemv`'s for its (row,
//! column). The row walk gives lane `L` of a row the sequential sum of the
//! terms `32·it + L` and combines the 32 lane sums by the butterfly xor 16,
//! 8, 4, 2, 1 (`q8f32.rs`'s module doc). A tile keeps every one of those
//! lane sums and the tree, and only moves them between threads: a block of
//! eight warps covers [`TILE_COLS`] columns and [`TILE_ROWS`] rows, the rows'
//! and columns' next 64 values staged in shared memory with `cp.async`,
//! three stages deep. Sixteen threads of a warp cover a 32-value chunk, two
//! values each, so thread `p` keeps the lane sums `2p` and `2p + 1` of each
//! (row, column) pair it owns, eight rows by eight columns ([`pair_chunk`],
//! one 8-byte load per row and column a chunk); the warp's two
//! sixteen-thread halves take different columns. The butterfly then runs
//! xor 16, 8, 4, 2 of the lanes as xor 8, 4, 2, 1 of the threads, and xor 1
//! as the add of a thread's two sums. A row past the last is staged as the
//! last row (and a row group with no live row is neither staged nor summed),
//! a column past the last as the last column; their sums are not stored.
//! The output is column-major, `y[c · rows + r]` — the GEMM families'
//! `y[s][n]`, not `f32_gemv`'s `y[r · m + c]`. The routers' prompt logits
//! (`softmax_router_logits` and the gated router's) run this body too.
//!
//! The rows come from a [`TileRows`]: an F32 weight's values are copied into
//! the stage as they are ([`F32Rows`], `cp.async`), and an E4M3 weight's
//! planes (`q8f32::e4m3_planes`) are decoded into it ([`ByteRows`]: each
//! piece's value word and its block's scale loaded, the four values formed
//! as the planes' `ByteBlocks` gives them, exactly the host's dequant, and
//! stored): `mxfp8_tile_gemm` and `fp8_b128_tile_gemm`. Either way the warps
//! read f32 values from the stage, so an E4M3 tile is bit for bit `f32_gemv`
//! over the host's dequantized rows. BF16 words are widened into the same stage
//! ([`Bf16Rows`], `q8f32`'s `bf16_tile_gemm`), bit for bit the F32 tile over those values.
//!
//! Prompt sites of a compute capability 12 device select [`bf16_mma_body`]
//! for BF16 and E4M3 weights.
//! It rounds activations and decoded weights to BF16 and accumulates in F32.
//! A BF16 site whose input holds its columns already rounded
//! ([`bf16_cols_body`], `GemmAct32::with_form`) copies those words instead,
//! with the same bits. On a native block-scaled build an MXFP8 site whose
//! input holds its columns as E4M3 with UE8M0 scales a 32-value block runs
//! the same tile on the planes as stored, W8A8 on the block-scaled MMA, and
//! accumulates in F32 (`nvfp4mma.rs`: `mxfp8_cols_body`, `Mxfp8Rows`,
//! `Mxfp8Cols`).
//! A cell whose row or column stages an operand outside the tensor cores'
//! range is recomputed by the exact F32 owner, in its exponent range and
//! reduction order; every other cell keeps its MMA value, so a cell's bits
//! depend on its own row and column alone. Activations held only as BF16
//! words ([`Bf16WordCols`]) are checked as their copies land and replay on
//! their widened values, which are exact. The exact tile entries stay
//! available. A narrow block ([`bf16_mma_narrow`]: sixteen rows of a BF16
//! weight against up to eight columns, one fragment) runs the same
//! arithmetic, so its cells are the tile's bits; a decode launch that must
//! match a prompt launch per token uses it.

use super::act32::ColsForm;
use super::gemm32::{GEMM32_GROUP, raster32};
use super::kernels32::Gemm32Kernels;
use super::{GemmAct32, MmaForm};
use crate::fault::FaultSink;
use crate::linear::norm_gate::{NormGateTail, norm_gate_values};
use crate::linear::{GATE_SIGMOID, HEAD};
use crate::q8f32::{Bf16Words, ByteBlocks, E4m3Scales, E4m3Weight};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use core::marker::PhantomData;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::vector::{F32x2, F32x4, U32x4};
use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k16_f32_bf16};
use cuda_device::{DisjointSlice, thread, warp};
#[cfg(bloomery_nvfp4_mma)]
use cuda_device::{
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
        mbarrier_init, mbarrier_try_wait_parity,
    },
    tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s},
};

/// Columns and rows one block covers.
pub(super) const TILE_COLS: usize = 32;
pub(super) const TILE_ROWS: usize = 32;
/// Values of a chunk one thread takes ([`pair_chunk`]), the threads that
/// cover a 32-value chunk, and the column halves of a warp (one a sixteen
/// threads).
const PAIR: usize = 2;
const CHUNK_THREADS: usize = 32 / PAIR;
const COL_HALVES: usize = 32 / CHUNK_THREADS;
/// Rows and columns one thread sums in a full pass ([`f32_tile_body`]); the
/// MMA replay's passes sum [`MmaRows::REPLAY_ROWS`] rows.
const FULL_ROWS: usize = 8;
const THREAD_COLS: usize = 8;
/// Warps of a block: row groups by column groups, a column group
/// [`COL_HALVES`]` · `[`THREAD_COLS`] columns.
const ROW_GROUPS: usize = 4;
const COL_GROUPS: usize = TILE_COLS / (COL_HALVES * THREAD_COLS);
/// Threads of a block.
pub(super) const F32T_THREADS: usize = ROW_GROUPS * COL_GROUPS * 32;
pub(crate) const F32T_THREADS_U32: u32 = F32T_THREADS as u32;
/// 32-value chunks one stage holds of every row and column, and the values
/// of one such line: K must be a multiple of it.
const STAGE_CHUNKS: usize = 2;
/// Values in a staged F32 tile line; K must be a positive multiple.
pub const LINE: usize = 32 * STAGE_CHUNKS;
/// Stages in shared memory: the one the warps read and two being copied in.
pub(crate) const STAGES: usize = 3;
/// One stage: every row's line, then every column's.
pub(crate) const STAGE_FLOATS: usize = (TILE_ROWS + TILE_COLS) * LINE;
/// 16-byte copies per line.
const LINE_COPIES: usize = LINE / 4;

// A full pass's row groups cover the tile; the column groups cover the
// tile's columns; the kernel entries' launch bounds and contracts spell the
// block out.
const _: () = assert!(ROW_GROUPS * FULL_ROWS == TILE_ROWS);
const _: () = assert!(COL_GROUPS * COL_HALVES * THREAD_COLS == TILE_COLS);
const _: () = assert!(PAIR == 2 && CHUNK_THREADS == 16 && F32T_THREADS == 256);
// The staged row lines are whole row groups of a full pass.
const _: () = assert!(TILE_ROWS.is_multiple_of(FULL_ROWS));
// Every thread copies whole pieces of row lines and column lines, the same
// pieces of each.
const _: () = assert!((TILE_ROWS * LINE_COPIES).is_multiple_of(F32T_THREADS));
const _: () = assert!(TILE_COLS == TILE_ROWS);
const _: () = assert!(4 * STAGES * STAGE_FLOATS <= 48 * 1024);

/// A tile's weight rows: how one thread's piece of a row line reaches the
/// stage.
pub(crate) trait TileRows: Copy {
    /// Values `at .. at + 4` of the weight (its rows back to back, `at` a
    /// multiple of 4) into the stage's 16 bytes at `dst`, readable by every
    /// warp of the block after the next `cp_async_wait_group` that completes
    /// this thread's last committed group and the barrier after it.
    ///
    /// # Safety
    ///
    /// The four values lie inside the weight's rows; `dst` is 16 bytes of
    /// shared memory, 16-byte aligned, that this thread alone writes and no
    /// warp reads before that barrier.
    unsafe fn stage(self, dst: *mut f32, at: usize);
}

/// An F32 weight's values, copied as they are.
#[derive(Clone, Copy)]
pub(crate) struct F32Rows<'a>(pub &'a [f32]);

impl TileRows for F32Rows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: the 16 source bytes are values at .. at + 4 of the weight,
        // 16-byte aligned (at a multiple of 4, the base 16-byte aligned: the
        // body's contract), and dst is this thread's own 16 bytes, by this
        // fn's contract; the copy completes at the wait that the contract
        // names.
        unsafe { cp_async_cg_16(dst.cast::<u32>(), self.0.as_ptr().add(at).cast::<u32>()) };
    }
}

/// An E4M3 weight's two planes, decoded as they are staged by `F`, the
/// planes' [`ByteBlocks`].
pub(super) struct ByteRows<'a, F> {
    qs: &'a [u32],
    e: &'a [u32],
    blocks: PhantomData<F>,
}

impl<'a, F> ByteRows<'a, F> {
    #[inline(always)]
    pub(super) fn new(qs: &'a [u32], e: &'a [u32]) -> Self {
        ByteRows {
            qs,
            e,
            blocks: PhantomData,
        }
    }
}

impl<F> Clone for ByteRows<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for ByteRows<'_, F> {}

impl<F: ByteBlocks<Scale = u32>> TileRows for ByteRows<'_, F> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: values at .. at + 4 are the bytes of word at / 4, and their
        // block at / 32 has its scale in the `e` plane: both inside the
        // planes because the values lie inside the weight's rows (this fn's
        // contract).
        let (q, e) = unsafe {
            (
                *self.qs.get_unchecked(at >> 2),
                F::scale_at(self.e, at >> 5),
            )
        };
        let d = F::scale(e, at >> 5);
        let [v0, v1, v2, v3] = F::values(q);
        // SAFETY: dst is this thread's own 16 bytes of the stage (this fn's
        // contract); the stores are plain, published by the barrier.
        unsafe {
            *dst = v0 * d;
            *dst.add(1) = v1 * d;
            *dst.add(2) = v2 * d;
            *dst.add(3) = v3 * d;
        }
    }
}

/// A BF16 weight's words (`DevWeight::Bf16`), widened as they are staged.
#[derive(Clone, Copy)]
pub(crate) struct Bf16Rows<'a>(pub &'a [u32]);

impl TileRows for Bf16Rows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: values at .. at + 4 are the halves of words at / 2 and
        // at / 2 + 1 (at a multiple of 4), inside the words because the
        // values lie inside the weight's rows (this fn's contract).
        let (lo, hi) = unsafe {
            (
                *self.0.get_unchecked(at >> 1),
                *self.0.get_unchecked((at >> 1) + 1),
            )
        };
        // SAFETY: dst is this thread's own 16 bytes of the stage (this fn's
        // contract); the stores are plain, published by the barrier.
        unsafe {
            *dst = Bf16Words::widen(lo, false);
            *dst.add(1) = Bf16Words::widen(lo, true);
            *dst.add(2) = Bf16Words::widen(hi, false);
            *dst.add(3) = Bf16Words::widen(hi, true);
        }
    }
}

/// One thread's sums of one 32-value chunk (module doc): `w` points at the
/// thread's two staged values of its first row's chunk and `x` at those of
/// its first column's, each next row or column [`LINE`] floats on, each
/// pair one 8-byte load. Returns `acc[c][i][e] + w_i[e]·x_c[e]`: for row
/// `i`, column `c` and lane sum `2p + e` the term the row walk's lane `2p +
/// e` adds for this chunk. So sums advanced over a row's chunks in
/// increasing order hold, pair for pair, the row walk's lane sums.
///
/// # Safety
///
/// `w.add(i · LINE)` for `i < RT` and `x.add(c · LINE)` for `c <`
/// [`THREAD_COLS`] are readable, 8-byte aligned pairs of f32 that nothing
/// writes during the call.
#[inline(always)]
unsafe fn pair_chunk<const RT: usize>(
    mut acc: [[[f32; PAIR]; RT]; THREAD_COLS],
    w: *const f32,
    x: *const f32,
) -> [[[f32; PAIR]; RT]; THREAD_COLS] {
    let mut wv = [[0.0f32; PAIR]; RT];
    let mut i = 0usize;
    while i < RT {
        thread::__unroll_config::<0>();
        // SAFETY: i < RT, a readable aligned pair by this fn's contract.
        wv[i] = unsafe { (*w.add(i * LINE).cast::<F32x2>()).0 };
        i += 1;
    }
    let mut xv = [[0.0f32; PAIR]; THREAD_COLS];
    let mut c = 0usize;
    while c < THREAD_COLS {
        thread::__unroll_config::<0>();
        // SAFETY: c < THREAD_COLS, a readable aligned pair by this fn's contract.
        xv[c] = unsafe { (*x.add(c * LINE).cast::<F32x2>()).0 };
        c += 1;
    }
    for c in 0..THREAD_COLS {
        thread::__unroll_config::<0>();
        for i in 0..RT {
            thread::__unroll_config::<0>();
            acc[c][i] = [
                f32::mul_add(wv[i][0], xv[c][0], acc[c][i][0]),
                f32::mul_add(wv[i][1], xv[c][1], acc[c][i][1]),
            ];
        }
    }
    acc
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
/// `rows >= 1`, `n >= 1`, `k` a positive multiple of [`LINE`], `w` holding
/// `rows` rows of `k` values ([`TileRows::stage`]'s bound; an F32 weight
/// 16-byte aligned), `x.len() >= n·k`, 16-byte aligned, `y.len() >= rows·n`.
#[inline(always)]
pub(crate) unsafe fn f32_tile_body<W: TileRows>(
    w: W,
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
    let (x, dims) = (F32Rows(x), (rows, k as usize, n));
    // SAFETY: the wrapper supplies this block's checked tile and all its threads.
    unsafe { f32_tile_at::<W, _, FULL_ROWS, false>(w, x, dims, y, sh, (row0, c0, 0)) };
}

/// Exact tile owner (module doc): a pass of [`F32T_THREADS`] threads over
/// the tile at `(row0, c0)`, each thread `RT` rows by [`THREAD_COLS`]
/// columns: pass `pass` covers the tile's rows `pass · ROW_GROUPS · RT ..`,
/// so the one pass of [`FULL_ROWS`] covers them all. With `MARKED`, only
/// outputs that hold NaN are written. Each output is stored as its
/// [`ProductCell`].
///
/// # Safety
///
/// All threads of the block call converged, the block [`F32T_THREADS`]
/// wide. RT is [`FULL_ROWS`], 1 or 2 and `pass · ROW_GROUPS · RT
/// < TILE_ROWS`. Rows and n are positive, row0 < rows and c0 < n; kk is a
/// positive multiple of LINE. The weight holds rows*kk values, the columns
/// `x` n*kk values (each [`TileRows::stage`]'s bound), and y holds rows*n
/// outputs. The aligned shared allocation holds STAGES*STAGE_FLOATS values.
/// No other block writes this output tile.
#[inline(always)]
unsafe fn f32_tile_at<W: TileRows, X: TileRows, const RT: usize, const MARKED: bool>(
    w: W,
    x: X,
    (rows, kk, n): (usize, usize, usize),
    y: &mut DisjointSlice<impl ProductCell>,
    sh: *mut f32,
    (row0, c0, pass): (usize, usize, usize),
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let (rg, cg) = ((tid / 32) / COL_GROUPS, (tid / 32) % COL_GROUPS);
    let (p, half) = (lane % CHUNK_THREADS, lane / CHUNK_THREADS);
    // The thread's first row line and column line in a stage.
    let rline = (pass * ROW_GROUPS + rg) * RT;
    let cline = (cg * COL_HALVES + half) * THREAD_COLS;
    // Warp-uniform: whether any of the warp's rows exists.
    let live = row0 + rline < rows;
    // The row lines a live warp reads: the live rows rounded up to whole row
    // groups of a full pass. The rest are not staged.
    let staged_rows = (rows - row0).min(TILE_ROWS).next_multiple_of(FULL_ROWS);
    let stages = kk / LINE;

    // This thread's copies of every stage: piece `piece` of row line `line`
    // (a staged one) and of column line `line`, a row past the last staged as
    // the last and a column past the last as the last.
    macro_rules! stage_copy {
        ($s:expr, $buf:expr) => {{
            let (s, buf): (usize, usize) = ($s, $buf);
            // SAFETY: buf < STAGES selects this block's allocated stage.
            let dst = unsafe { sh.add(buf * STAGE_FLOATS) };
            for i in 0..TILE_ROWS * LINE_COPIES / F32T_THREADS {
                thread::__unroll_config::<0>();
                let copy = tid + i * F32T_THREADS;
                let line = copy / LINE_COPIES;
                let piece = 4 * (copy % LINE_COPIES);
                if line < staged_rows {
                    let w_off = (row0 + line).min(rows - 1) * kk + piece;
                    // SAFETY: the copies partition the checked tile's aligned
                    // weight pieces; no warp reads them until the wait/barrier.
                    unsafe { w.stage(dst.add(line * LINE + piece), w_off + s * LINE) };
                }
                let x_off = (c0 + line).min(n - 1) * kk + piece;
                // SAFETY: x_off selects a live column's piece of four values
                // and the destination is this thread's disjoint aligned piece
                // of the stage; no warp reads it until the wait/barrier.
                unsafe { x.stage(dst.add((TILE_ROWS + line) * LINE + piece), x_off + s * LINE) };
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

    let mut acc = [[[0.0f32; PAIR]; RT]; THREAD_COLS];
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
        if live {
            // SAFETY: stage buf's row lines rline + i (i < RT, below
            // staged_rows since row0 + rline < rows) and column lines
            // TILE_ROWS + cline + c (c < THREAD_COLS) hold values 32·j + 2p
            // and 2p + 1 of chunks j < STAGE_CHUNKS, 8-byte aligned,
            // published by the barrier above and not refilled before the
            // next pass's barrier.
            unsafe {
                let base = sh.add(buf * STAGE_FLOATS).cast_const();
                let wp = base.add(rline * LINE + PAIR * p);
                let xp = base.add((TILE_ROWS + cline) * LINE + PAIR * p);
                acc = pair_chunk::<RT>(acc, wp, xp);
                acc = pair_chunk::<RT>(acc, wp.add(32), xp.add(32));
            }
        }
        s += 1;
        buf = if buf == STAGES - 1 { 0 } else { buf + 1 };
    }
    if !live {
        return; // warp-uniform: no barrier follows
    }

    // Every thread runs each pair's butterfly; thread 0 of each column half
    // stores a live one.
    let mut c = 0usize;
    while c < THREAD_COLS {
        thread::__unroll_config::<0>();
        let mut i = 0usize;
        while i < RT {
            thread::__unroll_config::<0>();
            let [mut a, mut b] = acc[c][i];
            a = a + warp::shuffle_xor_f32(a, 8);
            b = b + warp::shuffle_xor_f32(b, 8);
            a = a + warp::shuffle_xor_f32(a, 4);
            b = b + warp::shuffle_xor_f32(b, 4);
            a = a + warp::shuffle_xor_f32(a, 2);
            b = b + warp::shuffle_xor_f32(b, 2);
            a = a + warp::shuffle_xor_f32(a, 1);
            b = b + warp::shuffle_xor_f32(b, 1);
            let (r, col) = (row0 + rline + i, c0 + cline + c);
            if p == 0 && r < rows && col < n {
                // SAFETY: row r < rows and column col < n, so the slot lies
                // inside y (this fn's contract); thread 0 of the column half
                // owning row r and column col is its only writer.
                let slot = unsafe { y.get_unchecked_mut(col * rows + r) };
                if !MARKED || slot.is_nan() {
                    *slot = ProductCell::of(a + b);
                }
            }
            i += 1;
        }
        c += 1;
    }
}

/// Threads in a dense BF16 MMA tile: eight warps, two row halves by four
/// column quarters, each covering 64 rows by 32 columns.
pub(crate) const MMA_THREADS: u32 = 256;
// The replay runs the exact tile's passes with the MMA tile's threads.
const _: () = assert!(MMA_THREADS as usize == F32T_THREADS);
/// Lanes that hold one fragment row's cells.
const MMA_ROW_LANES: usize = 4;
/// Rows and columns of a block's outputs.
pub(crate) const MMA_ROWS: usize = 128;
pub(crate) const MMA_COLS: usize = 128;
/// K values one stage holds of every row and column: two 16-value MMA steps.
const MMA_LINE: usize = 32;
/// 16-byte pieces of BF16 values in one staged line.
const MMA_PIECES: usize = MMA_LINE / 8;
/// 32-bit words in one staged line, two BF16 values a word.
const MMA_LINE_WORDS: usize = MMA_LINE / 2;
/// Staged words before the column lines.
const MMA_COLS_AT: usize = MMA_ROWS * MMA_LINE_WORDS;
/// Lines whose pieces one pass of the block stages.
const MMA_PASS_LINES: usize = MMA_THREADS as usize / MMA_PIECES;
const _: () = assert!(MMA_THREADS == 256 && MMA_ROWS == 128 && MMA_COLS == 128 && MMA_LINE == 32);
// Every thread stages the same piece of two row lines and of two column lines.
const _: () = assert!(2 * MMA_PASS_LINES == MMA_ROWS && MMA_ROWS == MMA_COLS);
// The BF16 rows and columns of a stage fill the exact owner's stage, which the replay reuses.
const _: () = assert!((MMA_ROWS + MMA_COLS) * MMA_LINE_WORDS == STAGE_FLOATS && STAGES == 3);

/// Word offset of piece `slot` of staged line `line`. The slot is permuted by
/// the line's pair, so the eight lines one `ldmatrix` phase reads at one slot
/// cover all 32 banks.
#[inline(always)]
fn mma_piece_at(line: usize, slot: usize) -> usize {
    line * MMA_LINE_WORDS + 4 * (slot ^ ((line >> 1) & (MMA_PIECES - 1)))
}

/// Whether an operand stays on the tensor cores: zero or normal with
/// `|v| <= 2^47`. With K below `2^32` that bounds every partial absolute sum
/// below `2^126`, so no product, sum or exact butterfly leaves the finite range.
#[inline(always)]
#[allow(
    clippy::manual_range_contains,
    reason = "the kernel's two compares on the bits; a RangeInclusive would move its PTX"
)]
pub(super) fn mma_fits(v: f32) -> bool {
    let bits = v.to_bits() & 0x7fff_ffff;
    bits == 0 || (bits >= 0x0080_0000 && bits <= 0x5700_0000)
}

/// [`mma_fits`] for both BF16 values of a word, each its f32's high half.
#[inline(always)]
fn mma_fits_bf16x2(word: u32) -> bool {
    // Every 15-bit magnitude lane stays below 2^16 after each add, so no
    // carry crosses into the other value's lane.
    let big = (word & 0x7fff_7fff) + 0x28ff_28ff; // bit 15 set above 0x5700
    let exponent = (word & 0x7f80_7f80) + 0x7f80_7f80; // bit 15 set for a nonzero exponent
    let mantissa = (word & 0x007f_007f) + 0x7fff_7fff; // bit 15 set for a nonzero mantissa
    (big | (mantissa & !exponent)) & 0x8000_8000 == 0
}

/// A weight that the MMA tile stages as BF16 pieces of eight values.
pub(crate) trait MmaRows: TileRows {
    /// What a thread holds of one piece between [`Self::fetch`] and
    /// [`Self::land`].
    type Piece: Copy + Default;

    /// Whether [`Self::fetch`] is an asynchronous copy, so a thread holds no
    /// piece in registers across the products. The tile then runs the stages
    /// that fetch as their own loop, with no branch around the fetches; a
    /// weight that holds its pieces keeps one loop, where the compiler's
    /// hoisting of the next pieces' loads would spill.
    const COPIED: bool = true;

    /// Column tiles of one raster group ([`raster32`]). A wave of resident
    /// blocks reads the weight slabs of its rows and the F32 activations of
    /// its columns; for a BF16 weight, half an activation's bytes a value,
    /// eight tiles balance the two on GB10's 96 resident blocks.
    const GROUP: usize = 8;

    /// Rows one thread sums in a pass of [`MmaTile::replay`], 1 or 2. Fewer
    /// rows hold fewer registers beside the tile's and take more passes; a
    /// kernel keeps the count its registers fit without a spill.
    const REPLAY_ROWS: usize = 1;

    /// Starts moving values `at .. at + 8` (`at` a multiple of 8) to the 16
    /// bytes at `dst`: an asynchronous copy completed by this thread's
    /// `cp_async_wait_group` for its committed group, a store it makes
    /// itself, or loads [`Self::land`] stores.
    ///
    /// # Safety
    ///
    /// The eight values lie inside the weight's rows; `dst` is 16 bytes of
    /// shared memory, 16-byte aligned, that this thread alone writes and no
    /// warp reads before the barrier after that wait or that store.
    unsafe fn fetch(self, dst: *mut u32, at: usize) -> Self::Piece;

    /// Stores a fetched piece of values `at ..` at `dst` as BF16 rounded to
    /// nearest even; false when a value fails [`mma_fits`]. The default, for
    /// a piece [`Self::fetch`] copies as BF16 words, stores nothing.
    ///
    /// # Safety
    ///
    /// `piece` and `at` are one [`Self::fetch`]'s, and `dst` is its 16 bytes.
    #[inline(always)]
    unsafe fn land(_piece: Self::Piece, _dst: *mut u32, _at: usize) -> bool {
        true
    }

    /// After this thread's wait completed its copy to `dst`, whether the
    /// piece's values pass [`mma_fits`]. The default reads the copied BF16
    /// words; a weight whose [`Self::land`] checks its values returns true.
    ///
    /// # Safety
    ///
    /// `dst` is a completed, 16-byte aligned piece of this thread's.
    #[inline(always)]
    unsafe fn landed_fits(dst: *const u32) -> bool {
        // SAFETY: this fn's contract.
        let [a, b, c, d] = unsafe { (*dst.cast::<U32x4>()).0 };
        mma_fits_bf16x2(a) & mma_fits_bf16x2(b) & mma_fits_bf16x2(c) & mma_fits_bf16x2(d)
    }

    /// Whether the tile's staged 16-bit units are E4M3 value pairs with one
    /// UE8M0 scale a 32-value block ([`Self::scale_word`], [`Self::mma`]),
    /// not BF16 values. The activations' [`MmaCols::SCALED`] must agree.
    const SCALED: bool = false;

    /// For a scaled weight, the word holding the four block scales of the
    /// 64 units (128 values) of a row that start at unit `at`, a multiple of
    /// 64; 0 otherwise.
    ///
    /// # Safety
    ///
    /// The units lie inside the weight's rows.
    #[inline(always)]
    unsafe fn scale_word(self, _at: usize) -> u32 {
        0
    }

    /// For a weight whose staged units are its stored bytes ([`tma_mma_tile`]),
    /// whether the 64 units of a row from unit `at` (a multiple of 64) pass
    /// the check [`Self::land`] makes when it lands their first piece.
    ///
    /// # Safety
    ///
    /// The units lie inside the weight's rows.
    #[cfg(bloomery_nvfp4_mma)]
    #[inline(always)]
    unsafe fn span_fits(self, _at: usize) -> bool {
        true
    }

    /// One tensor-core step on a row fragment `a` and a column fragment `b`
    /// of 16 units each: BF16 `m16n8k16`, or for a scaled weight E4M3
    /// `m16n8k32` whose block scales are the low bytes of `sa` and `sb` in
    /// the lanes of each quad that `ta` (A: lanes `2·ta` and `2·ta + 1`, rows
    /// g and g + 8) and `tb` (B: lane `tb`, column g) name.
    ///
    /// # Safety
    ///
    /// Every lane of the warp calls it converged, `a` row-major and `b`
    /// column-major in the instruction's fragment layout.
    #[inline(always)]
    unsafe fn mma(
        acc: [f32; 4],
        a: [u32; 4],
        b: [u32; 2],
        _sa: u32,
        _sb: u32,
        _ta: usize,
        _tb: usize,
    ) -> [f32; 4] {
        // SAFETY: this fn's contract.
        unsafe { mma_m16n8k16_f32_bf16(acc, a, b) }
    }
}

impl MmaRows for Bf16Rows<'_> {
    type Piece = ();

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: values at .. at + 8 are words at / 2 .. at / 2 + 4, inside
        // the weight by this fn's contract and 16-byte aligned (the launcher
        // checks the words' alignment, at is a multiple of 8); dst is this
        // thread's own 16 bytes until the wait that completes the copy.
        unsafe { cp_async_cg_16(dst, self.0.as_ptr().add(at >> 1)) };
    }
}

impl<F: ByteBlocks<Scale = u32>> MmaRows for ByteRows<'_, F> {
    /// The piece's two value words and its block's scale element.
    type Piece = (u32, u32, u32);
    const COPIED: bool = false;
    /// The F32 tiles' column span: a byte weight is a quarter of an
    /// activation's bytes a value.
    const GROUP: usize = GEMM32_GROUP * TILE_COLS / MMA_COLS;

    #[inline(always)]
    unsafe fn fetch(self, _: *mut u32, at: usize) -> (u32, u32, u32) {
        // SAFETY: values at .. at + 8 are the bytes of words at / 4 and
        // at / 4 + 1, inside `qs` by this fn's contract.
        let (lo, hi) = unsafe {
            (
                *self.qs.get_unchecked(at >> 2),
                *self.qs.get_unchecked((at >> 2) + 1),
            )
        };
        // SAFETY: their 32-value block at / 32 has its scale in the `e` plane.
        let e = unsafe { F::scale_at(self.e, at >> 5) };
        (lo, hi, e)
    }

    #[inline(always)]
    unsafe fn land((lo, hi, e): (u32, u32, u32), dst: *mut u32, at: usize) -> bool {
        let d = F::scale(e, at >> 5);
        let ([l0, l1, l2, l3], [h0, h1, h2, h3]) = (F::values(lo), F::values(hi));
        let v = [
            l0 * d,
            l1 * d,
            l2 * d,
            l3 * d,
            h0 * d,
            h1 * d,
            h2 * d,
            h3 * d,
        ];
        // SAFETY: dst is this fetch's 16 bytes (this fn's contract).
        unsafe { land_values(v, dst) }
    }

    #[inline(always)]
    unsafe fn landed_fits(_: *const u32) -> bool {
        true
    }
}

/// Eight f32 values at `dst` as BF16 rounded to nearest even, two a word in
/// increasing order; false when one fails [`mma_fits`].
///
/// # Safety
///
/// `dst` is 16 bytes, 16-byte aligned, that this thread alone writes and no
/// thread reads before the barrier after this store.
#[inline(always)]
pub(crate) unsafe fn land_values(v: [f32; 8], dst: *mut u32) -> bool {
    let fits = mma_fits(v[0])
        & mma_fits(v[1])
        & mma_fits(v[2])
        & mma_fits(v[3])
        & mma_fits(v[4])
        & mma_fits(v[5])
        & mma_fits(v[6])
        & mma_fits(v[7]);
    let words = U32x4([
        cvt_bf16x2_f32(v[0], v[1]),
        cvt_bf16x2_f32(v[2], v[3]),
        cvt_bf16x2_f32(v[4], v[5]),
        cvt_bf16x2_f32(v[6], v[7]),
    ]);
    // SAFETY: dst is this thread's own aligned 16 bytes (this fn's contract).
    unsafe { *dst.cast::<U32x4>() = words };
    fits
}

/// Row and column of the first output of this block's tile among `rows` by
/// `n` outputs, in the raster used by `bf16_mma_tile`; `None` for a block
/// past the last tile.
#[inline(always)]
#[allow(
    clippy::question_mark,
    reason = "the let-else of bf16_mma_tile's raster, where a `?` moves the PTX"
)]
pub(crate) fn mma_tile_origin<W: MmaRows>(rows: usize, n: usize) -> Option<(usize, usize)> {
    let row_blocks = rows.div_ceil(MMA_ROWS);
    let block = thread::blockIdx_x() as usize;
    let Some((col_tile, row_tile)) = raster32(block, row_blocks, n.div_ceil(MMA_COLS), W::GROUP)
    else {
        return None;
    };
    let row0 = row_tile * MMA_ROWS;
    let col0 = col_tile * MMA_COLS;
    Some((row0, col0))
}

/// Activations that the MMA tile stages as BF16 pieces of eight values,
/// column `c`'s value `i` at `c·K + i`.
pub(crate) trait MmaCols: Copy {
    /// What a thread holds of one piece between [`Self::fetch`] and
    /// [`Self::land`].
    type Piece: Copy + Default;

    /// The columns the exact owner replays, as it stages their values.
    type Exact: TileRows;

    /// The columns the exact owner replays.
    fn exact(&self) -> Self::Exact;

    /// Whether column `c` holds a value [`mma_fits`] refuses, known before
    /// any piece is staged; false when [`Self::land`] checks each piece.
    ///
    /// # Safety
    ///
    /// `c` is a live column.
    unsafe fn refused(self, c: usize) -> bool;

    /// Starts moving values `at .. at + 8` to the 16 bytes at `dst`, as
    /// [`MmaRows::fetch`] does.
    ///
    /// # Safety
    ///
    /// [`MmaRows::fetch`]'s, over the columns.
    unsafe fn fetch(self, dst: *mut u32, at: usize) -> Self::Piece;

    /// Stores a fetched piece at `dst` as BF16 rounded to nearest even; false
    /// when a value fails [`mma_fits`].
    ///
    /// # Safety
    ///
    /// `piece` is one [`Self::fetch`]'s, and `dst` is its 16 bytes.
    unsafe fn land(piece: Self::Piece, dst: *mut u32) -> bool;

    /// After this thread's wait completed its copy to `dst`, whether the
    /// piece's values pass [`mma_fits`]: true where [`Self::refused`] or
    /// [`Self::land`] already checked them.
    ///
    /// # Safety
    ///
    /// [`MmaRows::landed_fits`]'.
    #[inline(always)]
    unsafe fn landed_fits(_dst: *const u32) -> bool {
        true
    }

    /// Whether the staged units are E4M3 value pairs with one UE8M0 scale a
    /// 32-value block ([`Self::scale_word`]); the weight's
    /// [`MmaRows::SCALED`] must agree.
    const SCALED: bool = false;

    /// For scaled columns, the word holding the four block scales of the 64
    /// units (128 values) of a column that start at unit `at`, a multiple of
    /// 64; 0 otherwise.
    ///
    /// # Safety
    ///
    /// The units lie inside a live column.
    #[inline(always)]
    unsafe fn scale_word(self, _at: usize) -> u32 {
        0
    }
}

/// F32 activations, loaded a piece at a time and rounded as they are staged.
#[derive(Clone, Copy)]
pub(crate) struct F32Cols<'a>(pub &'a [f32]);

impl<'a> MmaCols for F32Cols<'a> {
    type Piece = [f32; 8];
    type Exact = F32Rows<'a>;

    #[inline(always)]
    fn exact(&self) -> F32Rows<'a> {
        F32Rows(self.0)
    }

    #[inline(always)]
    unsafe fn refused(self, _: usize) -> bool {
        false
    }

    #[inline(always)]
    unsafe fn fetch(self, _: *mut u32, at: usize) -> [f32; 8] {
        let p = self.0.as_ptr().wrapping_add(at).cast::<F32x4>();
        // SAFETY: p starts eight values of a live column inside x, 16-byte
        // aligned (x aligned, at a multiple of 8).
        let [F32x4([a0, a1, a2, a3]), F32x4([b0, b1, b2, b3])] = unsafe { [*p, *p.add(1)] };
        [a0, a1, a2, a3, b0, b1, b2, b3]
    }

    #[inline(always)]
    unsafe fn land(v: [f32; 8], dst: *mut u32) -> bool {
        // SAFETY: dst is this fetch's 16 bytes (this fn's contract).
        unsafe { land_values(v, dst) }
    }
}

/// The same activations already rounded ([`bf16_cols_body`]): BF16 words,
/// two values a word in increasing order, copied asynchronously as a BF16
/// weight's are, and a word a column that is nonzero when one of the
/// column's f32 values fails [`mma_fits`]. The f32 columns stay for the
/// replay.
#[derive(Clone, Copy)]
pub(crate) struct Bf16Cols<'a> {
    pub words: &'a [u32],
    pub refused: &'a [u32],
    pub exact: &'a [f32],
}

impl<'a> MmaCols for Bf16Cols<'a> {
    type Piece = ();
    type Exact = F32Rows<'a>;

    #[inline(always)]
    fn exact(&self) -> F32Rows<'a> {
        F32Rows(self.exact)
    }

    #[inline(always)]
    unsafe fn refused(self, c: usize) -> bool {
        // SAFETY: a live column has its word (this fn's contract).
        unsafe { *self.refused.get_unchecked(c) != 0 }
    }

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: the columns' words are laid out as a BF16 weight's rows, and
        // this fn's contract is Bf16Rows::fetch's over them.
        unsafe { Bf16Rows(self.words).fetch(dst, at) };
    }

    #[inline(always)]
    unsafe fn land(_: (), _: *mut u32) -> bool {
        true
    }
}

/// Activations held only as BF16 words, two values a word in increasing
/// order, column `c`'s value `i` at `c·K + i`: copied asynchronously as a
/// BF16 weight's are, each piece checked against [`mma_fits`] once its copy
/// lands, and replayed on their widened values.
#[derive(Clone, Copy)]
pub(crate) struct Bf16WordCols<'a>(pub &'a [u32]);

impl<'a> MmaCols for Bf16WordCols<'a> {
    type Piece = ();
    type Exact = Bf16Rows<'a>;

    #[inline(always)]
    fn exact(&self) -> Bf16Rows<'a> {
        Bf16Rows(self.0)
    }

    #[inline(always)]
    unsafe fn refused(self, _: usize) -> bool {
        false
    }

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: the columns' words are laid out as a BF16 weight's rows, and
        // this fn's contract is Bf16Rows::fetch's over them.
        unsafe { Bf16Rows(self.0).fetch(dst, at) };
    }

    #[inline(always)]
    unsafe fn land(_: (), _: *mut u32) -> bool {
        true
    }

    #[inline(always)]
    unsafe fn landed_fits(dst: *const u32) -> bool {
        // SAFETY: this fn's contract is the weight's over the same words.
        unsafe { <Bf16Rows<'_> as MmaRows>::landed_fits(dst) }
    }
}

/// Threads of a [`bf16_cols_body`] or `nvfp4mma::mxfp8_cols_body` block: a
/// warp a column.
pub(crate) const FORM_THREADS: u32 = 256;

/// The first `n` columns of `k` f32 values (`k` a multiple of 8) as
/// [`Bf16Cols`]' words, each value rounded exactly as [`land_values`] rounds
/// it, and each column's refusal word. Block `b`'s warp `w` writes column
/// `8b + w`.
///
/// # Safety
///
/// x holds n aligned K-value columns, words n·K/2 and refused n.
#[inline(always)]
pub(crate) unsafe fn bf16_cols_body(
    x: &[f32],
    k: u32,
    n: u32,
    words: &mut DisjointSlice<u32>,
    refused: &mut DisjointSlice<u32>,
) {
    let (k, n) = (k as usize, n as usize);
    let c = form_col();
    let lane = warp::lane_id() as usize;
    let mut fits = true;
    if c < n {
        let mut piece = lane;
        while piece < k / 8 {
            let at = c * k + 8 * piece;
            // SAFETY: values at .. at + 8 lie in column c of x (this fn's contract).
            let v = unsafe { F32Cols(x).fetch(core::ptr::null_mut(), at) };
            // SAFETY: words at/2 .. at/2 + 4 are this lane's own, 16-byte aligned.
            fits &= unsafe { land_values(v, words.as_mut_ptr().add(at >> 1)) };
            piece += 32;
        }
    }
    // SAFETY: c is this warp's column (this fn's contract).
    unsafe { store_refused(refused, c, n, fits) };
}

/// The column a form block's warp writes: block `b`'s warp `w`, `8b + w`.
#[inline(always)]
pub(crate) fn form_col() -> usize {
    thread::blockIdx_x() as usize * (FORM_THREADS as usize / 32)
        + thread::threadIdx_x() as usize / 32
}

/// Column `c`'s refusal word: nonzero when a lane of its warp saw a value
/// [`mma_fits`] refuses (`fits` false). Every lane of the warp calls it.
///
/// # Safety
///
/// `refused` holds `n` words, and `c` is this warp's column.
#[inline(always)]
pub(crate) unsafe fn store_refused(
    refused: &mut DisjointSlice<u32>,
    c: usize,
    n: usize,
    fits: bool,
) {
    let refuse = warp::any(!fits);
    if c < n && warp::lane_id() == 0 {
        // SAFETY: lane 0 of column c's warp is its word's only writer.
        unsafe { *refused.get_unchecked_mut(c) = u32::from(refuse) };
    }
}

/// Cells of a thread's MMA fragments: four row by four column tiles of four.
pub(crate) const MMA_CELLS: usize = 64;
/// Words of the stage left after the pipeline before the line flags
/// [`MmaTile::needs_replay`] marks cells with: one vote a warp.
const MMA_FLAGS_AT: usize = MMA_THREADS as usize / 32;
const _: () = assert!(MMA_FLAGS_AT.is_multiple_of(4) && MMA_CELLS == 4 * 4 * 4);
const _: () = assert!(MMA_FLAGS_AT + (MMA_ROWS + MMA_COLS) * MMA_PIECES <= STAGE_FLOATS);
/// Rows and columns of one warp's outputs.
pub(crate) const MMA_WARP_ROWS: usize = 64;
pub(crate) const MMA_WARP_COLS: usize = 32;
/// Shared floats [`MmaTile::warp_scratch`] gives each warp: the two stages
/// that hold no votes, divided between the eight warps.
pub(crate) const MMA_WARP_SCRATCH: usize = 2 * STAGE_FLOATS / (MMA_THREADS as usize / 32);
const _: () = assert!(MMA_ROWS == 2 * MMA_WARP_ROWS && MMA_COLS == 4 * MMA_WARP_COLS);

/// A product's output cell: its F32 sum as stored, or the BF16 bits of that
/// sum rounded to nearest even (a NaN stays NaN; a value past the BF16 range
/// rounds to an infinity, which a reader's finite check sees).
pub(crate) trait ProductCell: Copy {
    fn of(v: f32) -> Self;
    /// Whether the cell holds a NaN: the mark a replay overwrites.
    fn is_nan(self) -> bool;
}

impl ProductCell for f32 {
    #[inline(always)]
    fn of(v: f32) -> Self {
        v
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl ProductCell for u16 {
    #[inline(always)]
    fn of(v: f32) -> Self {
        cuda_device::convert::f32_to_bf16_rne(v)
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        self & 0x7fff > 0x7f80
    }
}

/// Where a product launch stores its cells: F32 sums, or their BF16 bits
/// ([`ProductCell`]). A `&mut DeviceBuffer` of either converts into it.
pub enum ProductOut<'a> {
    F32(&'a mut DeviceBuffer<f32>),
    Bf16(&'a mut DeviceBuffer<u16>),
}

impl<'a> From<&'a mut DeviceBuffer<f32>> for ProductOut<'a> {
    fn from(y: &'a mut DeviceBuffer<f32>) -> Self {
        ProductOut::F32(y)
    }
}

impl<'a> From<&'a mut DeviceBuffer<u16>> for ProductOut<'a> {
    fn from(y: &'a mut DeviceBuffer<u16>) -> Self {
        ProductOut::Bf16(y)
    }
}

/// Row and column of warp `warp`'s outputs inside the tile.
#[inline(always)]
fn mma_warp_at(warp: usize) -> (usize, usize) {
    ((warp / 4) * MMA_WARP_ROWS, (warp % 4) * MMA_WARP_COLS)
}

/// A thread's completed MMA fragment and logical output coordinates.
pub(crate) struct MmaTile {
    values: [[[f32; 4]; 4]; 4],
    row0: usize,
    col0: usize,
    warp: usize,
    lane: usize,
    /// Bit i (2 + i): a value this thread staged of its weight (activation)
    /// line i fails [`mma_fits`].
    lines: u32,
    /// The cells [`Self::needs_replay`] marks for the exact owner.
    bad: u64,
}

/// Row and column of value `i` (< 4) of an m16n8 accumulator fragment in
/// lane `lane`, at the fragment's origin `(r0, c0)`.
#[inline(always)]
fn fragment_cell((r0, c0): (usize, usize), lane: usize, i: usize) -> (usize, usize) {
    (
        r0 + lane / MMA_ROW_LANES + (i / 2) * 8,
        c0 + 2 * (lane % MMA_ROW_LANES) + i % 2,
    )
}

impl MmaTile {
    #[inline(always)]
    pub(crate) fn coordinates(&self, cell: usize) -> (usize, usize) {
        let (mi, j, i) = (cell / 16, (cell / 4) % 4, cell % 4);
        let (r0, c0) = self.warp_origin();
        fragment_cell((r0 + mi * 16, c0 + j * 8), self.lane, i)
    }

    /// The accumulated value of `cell` (< [`MMA_CELLS`]). Cells
    /// `n·MMA_CELLS/4 ..` hold the warp's output rows `16n .. 16n + 16`.
    #[inline(always)]
    pub(crate) fn value(&self, cell: usize) -> f32 {
        self.values[cell / 16][(cell / 4) % 4][cell % 4]
    }

    /// Row and column of this warp's first output.
    #[inline(always)]
    pub(crate) fn warp_origin(&self) -> (usize, usize) {
        let (wr, wc) = mma_warp_at(self.warp);
        (self.row0 + wr, self.col0 + wc)
    }

    /// This warp's [`MMA_WARP_SCRATCH`] floats of `sh`, the tile's allocation
    /// for K = `k`. No other warp uses them, and once [`Self::needs_replay`]
    /// returns no read of the pipeline or the votes is left in them; a replay
    /// reuses them.
    #[inline(always)]
    pub(crate) fn warp_scratch(&self, sh: *mut f32, k: usize) -> *mut f32 {
        let warps_a_stage = STAGE_FLOATS / MMA_WARP_SCRATCH;
        let stage = (k / MMA_LINE + 1 + self.warp / warps_a_stage) % STAGES;
        sh.wrapping_add(stage * STAGE_FLOATS + (self.warp % warps_a_stage) * MMA_WARP_SCRATCH)
    }
}

impl MmaTile {
    /// Whether any thread of the block staged an operand that fails
    /// [`mma_fits`]; if so, also marks this thread's cells whose row or column
    /// staged one, the cells [`Self::store`] leaves to [`Self::replay`].
    ///
    /// # Safety
    /// All 256 threads call with the tile's original shared allocation and K.
    /// After a true return, a barrier precedes any write to the stages.
    #[inline(always)]
    pub(crate) unsafe fn needs_replay(&mut self, sh: *mut f32, k: usize) -> bool {
        // The pipeline left this stage unread since its last pass's barrier;
        // the votes and then the line flags reuse it.
        let stage = sh.wrapping_add((k / MMA_LINE % STAGES) * STAGE_FLOATS);
        let retry = warp::any(self.lines != 0);
        if self.lane == 0 {
            // SAFETY: each warp owns one vote slot of the unused stage.
            unsafe { *stage.add(self.warp) = if retry { 1.0 } else { 0.0 } };
        }
        thread::sync_threads();
        let mut any = false;
        for i in 0..MMA_THREADS as usize / 32 {
            // SAFETY: the block barrier published all eight warp votes.
            any |= unsafe { *stage.add(i) != 0.0 };
        }
        if !any {
            return false;
        }
        // Each thread flags its four lines' pieces after the votes; a cell is
        // marked when a piece of its row or column is flagged.
        let flags = stage.cast::<u32>().wrapping_add(MMA_FLAGS_AT);
        let tid = self.warp * 32 + self.lane;
        let (line, slot) = (tid / MMA_PIECES, tid % MMA_PIECES);
        for i in 0..4 {
            thread::__unroll_config::<0>();
            let at = (MMA_ROWS * (i / 2) + line + MMA_PASS_LINES * (i % 2)) * MMA_PIECES + slot;
            // SAFETY: this thread alone writes its piece's flag, inside the stage
            // and past the votes.
            unsafe { *flags.add(at) = (self.lines >> i) & 1 };
        }
        thread::sync_threads();
        let line_bad = |line: usize| {
            // SAFETY: the barrier published the line's four aligned flags.
            let [a, b, c, d] = unsafe { (*flags.add(line * MMA_PIECES).cast::<U32x4>()).0 };
            a | b | c | d != 0
        };
        for cell in 0..MMA_CELLS {
            thread::__unroll_config::<0>();
            let (r, c) = self.coordinates(cell);
            if line_bad(r - self.row0) || line_bad(MMA_ROWS + c - self.col0) {
                self.bad |= 1 << cell;
            }
        }
        true
    }

    /// Stores this thread's live cells, each as its [`ProductCell`]; with
    /// MARKED, a marked or non-finite cell as NaN, which [`Self::replay`]
    /// recomputes.
    ///
    /// # Safety
    /// y holds rows*n cells. This tile is disjoint from every other block's.
    #[inline(always)]
    pub(crate) unsafe fn store<const MARKED: bool>(
        &self,
        y: &mut DisjointSlice<impl ProductCell>,
        rows: usize,
        n: usize,
    ) {
        for cell in 0..MMA_CELLS {
            thread::__unroll_config::<0>();
            let (r, c) = self.coordinates(cell);
            if r < rows && c < n {
                let value = self.value(cell);
                let value = if MARKED && (self.bad & (1 << cell) != 0 || !value.is_finite()) {
                    f32::NAN
                } else {
                    value
                };
                // SAFETY: this thread owns the checked live output cell.
                unsafe { *y.get_unchecked_mut(c * rows + r) = ProductCell::of(value) };
            }
        }
    }

    /// Recomputes the cells [`Self::needs_replay`] marked, which
    /// `store::<true>` left NaN, with the exact owner over the columns `x`
    /// stages; every other cell keeps its stored value.
    ///
    /// # Safety
    /// All 256 threads call converged with this tile's checked spans and shared
    /// allocation. Prior output stores and shared readers have completed.
    #[inline(always)]
    pub(crate) unsafe fn replay<W: MmaRows, X: TileRows>(
        &self,
        w: W,
        x: X,
        dims: (usize, usize, usize),
        y: &mut DisjointSlice<impl ProductCell>,
        sh: *mut f32,
    ) {
        const { assert!(W::REPLAY_ROWS == 1 || W::REPLAY_ROWS == 2) };
        for dr in 0..MMA_ROWS / TILE_ROWS {
            for dc in 0..MMA_COLS / TILE_COLS {
                let (r, c) = (self.row0 + dr * TILE_ROWS, self.col0 + dc * TILE_COLS);
                if r >= dims.0 || c >= dims.2 {
                    continue; // block-uniform: the sub-tile holds no output
                }
                for pass in 0..TILE_ROWS / (ROW_GROUPS * W::REPLAY_ROWS) {
                    let at = (r, c, pass);
                    // SAFETY: the passes cover disjoint checked row ranges and
                    // reuse the exact owner's shared allocation with its block
                    // of threads (the const assert beside MMA_THREADS).
                    unsafe {
                        if W::REPLAY_ROWS == 1 {
                            f32_tile_at::<W, X, 1, true>(w, x, dims, y, sh, at)
                        } else {
                            f32_tile_at::<W, X, 2, true>(w, x, dims, y, sh, at)
                        }
                    };
                    thread::sync_threads();
                }
            }
        }
    }
}

/// One 16-unit step of a published stage's products into a warp's 64 by 32
/// accumulators `acc` ([`bf16_mma_tile`]'s): `stage` the stage's shared
/// address, `(s, step)` the stage and step, the warp's origin `(wr, wc)` in
/// the tile, the lanes' ldmatrix pieces and a scaled step's scale words.
macro_rules! mma_step {
    ($W:ty, $acc:ident, $stage:expr, ($s:expr, $step:expr), ($wr:expr, $wc:expr), ($a_lane:expr, $b_lane:expr), ($sa:expr, $sb:expr)) => {{
        let half = 8 * $step;
        // A scaled step's block, 2s + step, is this byte of the pair's words.
        let shift = (8 * (2 * ($s % 2) + $step)) as u32;
        // Both column fragments first; then one row fragment at a time, so
        // the warp holds twelve fragment words, not twenty.
        let mut b = [[0u32; 4]; 2];
        for nj in 0..2 {
            thread::__unroll_config::<0>();
            let at = MMA_COLS_AT + ($wc + 16 * nj) * MMA_LINE_WORDS + ($b_lane ^ half);
            // SAFETY: the warp is converged; each lane's 16-byte row lies in
            // this published stage's column lines.
            b[nj] = unsafe { ldmatrix_x4_shared_u32($stage + (4 * at) as u32) };
        }
        for mi in 0..4 {
            thread::__unroll_config::<0>();
            let at = ($wr + 16 * mi) * MMA_LINE_WORDS + ($a_lane ^ half);
            // SAFETY: as above, in the stage's row lines.
            let a = unsafe { ldmatrix_x4_shared_u32($stage + (4 * at) as u32) };
            for j in 0..4 {
                thread::__unroll_config::<0>();
                let b = [b[j / 2][2 * (j % 2)], b[j / 2][2 * (j % 2) + 1]];
                let (sa, sb) = ($sa[mi / 2] >> shift, $sb >> shift);
                // SAFETY: the whole warp is converged; A is row-major and B
                // column-major in the step's fragment layout, and lanes mi % 2
                // (A) and j (B) of each quad hold their scales.
                $acc[mi][j] = unsafe { <$W>::mma($acc[mi][j], a, b, sa, sb, mi % 2, j) };
            }
        }
    }};
}

/// A scaled tile's block scales for the stage pair from stage `s`, from the
/// quad lane each MMA's selectors name ([`MmaRows::mma`]): lane q of a quad
/// holds in `sa[p]` the scale word of row g + 8(q % 2) of 16-row tile
/// 2p + q / 2 of the warp at `origin`, and in `sb` that of column g of
/// 8-column tile q. One word a row or column holds the four blocks of the
/// pair. Zero for an unscaled tile.
///
/// # Safety
///
/// Stage s starts a stage pair inside K; w and x are the tile's operands of
/// `dims` (rows, K units, n).
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn mma_scale_words<W: MmaRows, X: MmaCols>(
    w: W,
    x: X,
    origin: (usize, usize),
    (rows, kk, n): (usize, usize, usize),
    lane: usize,
    s: usize,
) -> ([u32; 2], u32) {
    let mut sa = [0u32; 2];
    let mut sb = 0u32;
    if W::SCALED {
        let (g, q) = (lane / MMA_ROW_LANES, lane % MMA_ROW_LANES);
        for p in 0..2 {
            thread::__unroll_config::<0>();
            let r = (origin.0 + 16 * (2 * p + q / 2) + g + 8 * (q % 2)).min(rows - 1);
            // SAFETY: stage s starts a stage pair of the clamped row, inside
            // the operands.
            sa[p] = unsafe { w.scale_word(r * kk + s * MMA_LINE) };
        }
        let c = (origin.1 + 8 * q + g).min(n - 1);
        // SAFETY: as the rows', of the clamped live column.
        sb = unsafe { x.scale_word(c * kk + s * MMA_LINE) };
    }
    (sa, sb)
}

/// BF16 tensor-core fragments of the stored weight's decoded rows, and
/// whether the tile replays. Activations and decoded weights round once to
/// BF16; accumulation is F32, each output's 16-value MMA steps in increasing K.
///
/// Both operands are staged as BF16 lines, three stages deep, two stages
/// ahead: the weight's pieces through [`MmaRows`] (a BF16 weight by
/// asynchronous copy) and the activations' through [`MmaCols`] (F32 values
/// loaded and converted after the current stage's products, or rounded BF16
/// words copied). Each warp reads its fragments with `ldmatrix`. The tile
/// keeps, per thread, which of its staged lines held an operand [`mma_fits`]
/// refuses; [`MmaTile::needs_replay`] marks the cells of those rows and
/// columns for the exact owner.
///
/// The tile counts K in 16-bit units: BF16 values, or for scaled operands
/// ([`MmaRows::SCALED`]) E4M3 value pairs, whose steps are block-scaled
/// `m16n8k32` products ([`MmaRows::mma`]) and whose threads hold, a stage
/// pair at a time, the scale words of the rows and columns their fragments
/// supply scales for.
///
/// # Safety
///
/// All 256 threads call converged. Rows and n are positive, K (in units) is a
/// positive multiple of 64, x holds n aligned K-unit columns and w holds
/// rows*K units ([`MmaRows::fetch`]'s bound). The grid covers 128 by 128
/// output tiles. sh holds STAGES*STAGE_FLOATS aligned shared values.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
#[allow(
    clippy::question_mark,
    reason = "the kernel's early return on the raster; a `?` would move its PTX"
)]
pub(crate) unsafe fn bf16_mma_tile<W: MmaRows, X: MmaCols>(
    w: W,
    x: X,
    rows: u32,
    k: u32,
    n: u32,
    sh: *mut f32,
) -> Option<MmaTile> {
    let (rows, kk, n) = (rows as usize, k as usize, n as usize);
    let Some((row0, col0)) = mma_tile_origin::<W>(rows, n) else {
        return None;
    };
    const { assert!(W::SCALED == X::SCALED) };
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let warp = tid / 32;
    let (wr, wc) = mma_warp_at(warp);
    let stages = kk / MMA_LINE;
    let words = sh.cast::<u32>();
    // SAFETY: sh is this block's shared allocation (this fn's contract).
    let shared = unsafe { cvta_generic_to_shared_u32(words.cast_const().cast::<u8>()) };

    // This thread's piece `slot` of lines `line` and `line + MMA_PASS_LINES`
    // of both operands; a row past the last is staged as the last row and a
    // column past the last as the last column.
    let slot = tid % MMA_PIECES;
    let line = tid / MMA_PIECES;
    let w_src = [
        (row0 + line).min(rows - 1) * kk + 8 * slot,
        (row0 + line + MMA_PASS_LINES).min(rows - 1) * kk + 8 * slot,
    ];
    let cols = [
        (col0 + line).min(n - 1),
        (col0 + line + MMA_PASS_LINES).min(n - 1),
    ];
    let x_src = [cols[0] * kk + 8 * slot, cols[1] * kk + 8 * slot];
    let w_dst = [
        mma_piece_at(line, slot),
        mma_piece_at(line + MMA_PASS_LINES, slot),
    ];
    let x_dst = [MMA_COLS_AT + w_dst[0], MMA_COLS_AT + w_dst[1]];

    // A scaled step's block scales, from the quad lane each MMA's selectors
    // name (MmaRows::mma): lane q of a quad holds in `sa[p]` the scale word
    // of row g + 8(q % 2) of 16-row tile 2p + q / 2, and in `sb` that of
    // column g of 8-column tile q. One word a row or column holds the four
    // blocks of a stage pair, loaded after the products of the pair before it.
    let origin = (row0 + wr, col0 + wc);
    // SAFETY: stage 0 starts a stage pair inside K.
    let (mut sa, mut sb) = unsafe { mma_scale_words(w, x, origin, (rows, kk, n), lane, 0) };

    // Bit i (2 + i) is set once this thread stages a value of its weight
    // (activation) line i that fails mma_fits.
    let mut bad = 0u32;
    for i in 0..2 {
        thread::__unroll_config::<0>();
        // SAFETY: cols[i] < n is a live column.
        bad |= u32::from(unsafe { x.refused(cols[i]) }) << (2 + i);
    }
    for s in 0..2 {
        thread::__unroll_config::<0>();
        let stage = words.wrapping_add(s * STAGE_FLOATS);
        for i in 0..2 {
            thread::__unroll_config::<0>();
            let at = w_src[i] + s * MMA_LINE;
            let dst = stage.wrapping_add(w_dst[i]);
            // SAFETY: K is a positive multiple of 64, so stage s < 2 lies
            // inside it; dst is this thread's own aligned 16 bytes of the stage.
            let piece = unsafe { w.fetch(dst, at) };
            // SAFETY: that fetch's piece and bytes.
            bad |= u32::from(!unsafe { W::land(piece, dst, at) }) << i;
            let dst = stage.wrapping_add(x_dst[i]);
            // SAFETY: as the weight's, in this thread's own activation piece.
            let piece = unsafe { x.fetch(dst, x_src[i] + s * MMA_LINE) };
            // SAFETY: that fetch's piece and bytes.
            bad |= u32::from(!unsafe { X::land(piece, dst) }) << (2 + i);
        }
        // SAFETY: commits this thread's copies of stage s as one group.
        unsafe { cp_async_commit_group() };
    }

    // [m tile][n tile] accumulators of the warp's 64 by 32 outputs.
    let mut acc = [[[0.0f32; 4]; 4]; 4];
    // ldmatrix rows: A row `lane % 16` at K half `lane / 16`; B column
    // `lane % 8 + 8·(lane / 16)` at K half `(lane / 8) % 2`. Bits 1..2 of the
    // line pick the permutation, and the warp's line offsets keep them.
    let a_lane = mma_piece_at(lane % 16, lane / 16);
    let b_lane = mma_piece_at(lane % 8 + 8 * (lane / 16), (lane / 8) % 2);
    // Stage s's products; with `ahead`, stage s + 2 goes to the buffer every
    // warp finished reading before this pass's barrier: piece `step` of each
    // operand is fetched before MMA step `step` and landed after it, so a
    // thread holds one piece of each in flight. A copied weight's stages that
    // fetch run as their own loop ([`MmaRows::COPIED`]).
    macro_rules! stage_pass {
        ($s:expr, $ahead:expr) => {{
            let (s, ahead) = ($s, $ahead);
            let buf = s % STAGES;
            // SAFETY: two groups precede this wait; the oldest is stage s.
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();
            let next = words.wrapping_add(((s + 2) % STAGES) * STAGE_FLOATS);
            let stage = shared + (4 * buf * STAGE_FLOATS) as u32;
            for step in 0..2 {
                thread::__unroll_config::<0>();
                let at = w_src[step] + (s + 2) * MMA_LINE;
                let mut piece = W::Piece::default();
                let mut xs = X::Piece::default();
                if ahead {
                    // SAFETY: stage s + 2 lies inside K; the piece is this thread's.
                    piece = unsafe { w.fetch(next.wrapping_add(w_dst[step]), at) };
                    let x_at = x_src[step] + (s + 2) * MMA_LINE;
                    // SAFETY: as the weight's, in the activation piece.
                    xs = unsafe { x.fetch(next.wrapping_add(x_dst[step]), x_at) };
                }
                if step == 1 {
                    // SAFETY: commits this pass's copies as one group, empty near the end.
                    unsafe { cp_async_commit_group() };
                }
                mma_step!(W, acc, stage, (s, step), (wr, wc), (a_lane, b_lane), (sa, sb));
                if step == 1 {
                    for i in 0..2 {
                        thread::__unroll_config::<0>();
                        let fits =
                            // SAFETY: the wait completed this thread's copies of stage
                            // s, which no warp overwrites before the next pass's barrier.
                            unsafe { W::landed_fits(words.add(buf * STAGE_FLOATS + w_dst[i])) };
                        bad |= u32::from(!fits) << i;
                        let fits =
                            // SAFETY: as the weight's, this thread's activation piece.
                            unsafe { X::landed_fits(words.add(buf * STAGE_FLOATS + x_dst[i])) };
                        bad |= u32::from(!fits) << (2 + i);
                    }
                }
                if ahead {
                    // SAFETY: this thread's own aligned activation piece of stage
                    // s + 2, which no warp reads before the next pass's barrier.
                    let fits = unsafe { X::land(xs, next.wrapping_add(x_dst[step])) };
                    bad |= u32::from(!fits) << (2 + step);
                    // SAFETY: this step's fetch of stage s + 2, into its piece, as above.
                    let fits = unsafe { W::land(piece, next.wrapping_add(w_dst[step]), at) };
                    bad |= u32::from(!fits) << step;
                }
            }
            if s % 2 == 1 && s + 1 < stages {
                // SAFETY: stage s + 1 < stages starts a stage pair inside K.
                (sa, sb) = unsafe { mma_scale_words(w, x, origin, (rows, kk, n), lane, s + 1) };
            }
        }};
    }
    if W::COPIED {
        for s in 0..stages - 2 {
            stage_pass!(s, true);
        }
        for s in stages - 2..stages {
            stage_pass!(s, false);
        }
    } else {
        for s in 0..stages {
            stage_pass!(s, s + 2 < stages);
        }
    }
    Some(MmaTile {
        values: acc,
        row0,
        col0,
        warp,
        lane,
        lines: bad,
        bad: 0,
    })
}

/// Bytes of one operand's TMA box: 128 lines of one stage's 64 bytes.
#[cfg(bloomery_nvfp4_mma)]
pub(crate) const MMA_BOX_BYTES: u32 = (MMA_ROWS * MMA_LINE * 2) as u32;

/// [`bf16_mma_tile`] for operands whose staged units are their stored bytes
/// (MXFP8 codes), each stage's two 128-line boxes brought by TMA from
/// `w_map` and `x_map` (UINT8 `[lines][K bytes]`, box 64 by 128 bytes,
/// 64-byte swizzle, which is [`mma_piece_at`]'s layout). Lines past the last
/// row or column arrive as zeros; their cells are never stored.
///
/// Thread 0 issues the copies: stages 0 and 1 first, stage s + 2 when stage
/// s starts, into the buffer of stage s - 1 once every warp has released it.
/// `bars` are [`STAGES`] `full` barriers (one arrival and the boxes' bytes)
/// then [`STAGES`] `empty` barriers (one arrival a warp); stage s waits the
/// parity of its round s / STAGES. No block barrier runs in the loop; one
/// after it hands the stages to the epilogue. Scale words stay global reads,
/// and each thread checks the spans of its two copy lines
/// ([`MmaRows::span_fits`]) as [`MmaRows::land`] does for a fetched piece,
/// so the tile's `lines` are [`bf16_mma_tile`]'s.
///
/// # Safety
///
/// [`bf16_mma_tile`]'s contract for scaled operands, `w_map` and `x_map`
/// encoding w's rows and x's n columns of K bytes, sh 1024-byte aligned and
/// bars 2·STAGES uninitialized barriers of this block.
#[cfg(bloomery_nvfp4_mma)]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
#[allow(
    clippy::question_mark,
    reason = "the kernel's early return on the raster; a `?` would move its PTX"
)]
pub(crate) unsafe fn tma_mma_tile<W: MmaRows, X: MmaCols>(
    (w, w_map): (W, &TmaDescriptor),
    (x, x_map): (X, &TmaDescriptor),
    (rows, k, n): (u32, u32, u32),
    sh: *mut f32,
    bars: *mut Barrier,
) -> Option<MmaTile> {
    const { assert!(W::SCALED && X::SCALED) };
    let (rows, kk, n) = (rows as usize, k as usize, n as usize);
    let Some((row0, col0)) = mma_tile_origin::<W>(rows, n) else {
        return None;
    };
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let warp = tid / 32;
    let (wr, wc) = mma_warp_at(warp);
    let stages = kk / MMA_LINE;
    let words = sh.cast::<u32>();
    // SAFETY: sh is this block's shared allocation (this fn's contract).
    let shared = unsafe { cvta_generic_to_shared_u32(words.cast_const().cast::<u8>()) };
    let (full, empty) = (bars, bars.wrapping_add(STAGES));
    if tid == 0 {
        for b in 0..STAGES {
            // SAFETY: the block's own barriers, before any other use.
            unsafe {
                mbarrier_init(full.add(b), 1);
                mbarrier_init(empty.add(b), MMA_THREADS / 32);
            }
        }
        // SAFETY: makes the initialized barriers visible to the TMA unit.
        unsafe { fence_proxy_async_shared_cta() };
    }
    thread::sync_threads();
    // Thread 0's copies of stage $s into its buffer, completing on its full
    // barrier.
    macro_rules! issue {
        ($s:expr) => {{
            let s = $s;
            let bar = full.wrapping_add(s % STAGES);
            let stage = words.wrapping_add((s % STAGES) * STAGE_FLOATS);
            let cols = stage.wrapping_add(MMA_COLS_AT);
            let at = (2 * MMA_LINE * s) as i32;
            // SAFETY: the stage's full barrier and buffer in sh, released by
            // every warp, the maps of this fn's contract, and boxes inside the
            // 2 * MMA_BOX_BYTES the arrival expects. A launch without a cluster
            // is a cluster of this one block, so its shared addresses are the
            // copies' cluster addresses.
            unsafe {
                let _ = mbarrier_arrive_expect_tx(bar, 1, 2 * MMA_BOX_BYTES);
                cp_async_bulk_tensor_2d_g2s(stage.cast(), w_map, at, row0 as i32, bar);
                cp_async_bulk_tensor_2d_g2s(cols.cast(), x_map, at, col0 as i32, bar);
            }
        }};
    }
    if tid == 0 {
        issue!(0);
        issue!(1);
    }

    let origin = (row0 + wr, col0 + wc);
    // SAFETY: stage 0 starts a stage pair inside K.
    let (mut sa, mut sb) = unsafe { mma_scale_words(w, x, origin, (rows, kk, n), lane, 0) };
    // This thread's copy lines, as bf16_mma_tile's: bit i (2 + i) is set
    // once weight (activation) line i holds a value that fails mma_fits.
    let (slot, line) = (tid % MMA_PIECES, tid / MMA_PIECES);
    let w_lines = [
        (row0 + line).min(rows - 1) * kk,
        (row0 + line + MMA_PASS_LINES).min(rows - 1) * kk,
    ];
    let mut bad = 0u32;
    for i in 0..2 {
        thread::__unroll_config::<0>();
        let c = (col0 + line + i * MMA_PASS_LINES).min(n - 1);
        // SAFETY: c < n is a live column.
        bad |= u32::from(unsafe { x.refused(c) }) << (2 + i);
    }

    let mut acc = [[[0.0f32; 4]; 4]; 4];
    let a_lane = mma_piece_at(lane % 16, lane / 16);
    let b_lane = mma_piece_at(lane % 8 + 8 * (lane / 16), (lane / 8) % 2);
    for s in 0..stages {
        let buf = s % STAGES;
        if tid == 0 && s + 2 < stages {
            let next = s + 2;
            if next >= STAGES {
                let parity = ((next / STAGES - 1) % 2) as u32;
                // SAFETY: the block's initialized barrier; every warp releases
                // stage s - 1 after its products.
                while !unsafe { mbarrier_try_wait_parity(empty.add(next % STAGES), parity) } {}
                // The warps read the buffer through the generic proxy and the
                // copies write it through the async one: the proxy fence
                // orders the reads the barrier brought here before them.
                // SAFETY: a fence, no memory named.
                unsafe { fence_proxy_async_shared_cta() };
            }
            issue!(next);
        }
        warp::sync_mask(u32::MAX);
        // SAFETY: the block's initialized barrier; thread 0 issued stage s.
        while !unsafe { mbarrier_try_wait_parity(full.add(buf), ((s / STAGES) % 2) as u32) } {}
        // Lanes leave the wait on different polls; the `.aligned` ldmatrix and
        // mma below need the whole warp converged.
        warp::sync_mask(u32::MAX);
        let stage = shared + (4 * buf * STAGE_FLOATS) as u32;
        for step in 0..2 {
            thread::__unroll_config::<0>();
            mma_step!(
                W,
                acc,
                stage,
                (s, step),
                (wr, wc),
                (a_lane, b_lane),
                (sa, sb)
            );
        }
        warp::sync_mask(u32::MAX);
        if lane == 0 {
            // SAFETY: the block's initialized barrier; the warp's reads of
            // the stage are done.
            let _ = unsafe { mbarrier_arrive(empty.add(buf)) };
        }
        if s % 2 == 0 && slot == 0 {
            for i in 0..2 {
                thread::__unroll_config::<0>();
                // SAFETY: stage s starts a 64-unit span of the clamped row.
                bad |= u32::from(!unsafe { w.span_fits(w_lines[i] + s * MMA_LINE) }) << i;
            }
        }
        if s % 2 == 1 && s + 1 < stages {
            // SAFETY: stage s + 1 < stages starts a stage pair inside K.
            (sa, sb) = unsafe { mma_scale_words(w, x, origin, (rows, kk, n), lane, s + 1) };
        }
    }
    // Every copy completed (each stage's full wait); after this no warp reads
    // a stage, so the epilogue may reuse them.
    thread::sync_threads();
    Some(MmaTile {
        values: acc,
        row0,
        col0,
        warp,
        lane,
        lines: bad,
        bad: 0,
    })
}

/// Dense prompt product whose cells of an exceptional row or column replay on
/// the exact owner. K counts values; a scaled tile's units are value pairs.
///
/// # Safety
/// The checked launch supplies 256 converged threads, aligned stages and x,
/// rows*K weight values, n*K activations and rows*n disjoint output cells.
#[inline(always)]
pub(crate) unsafe fn bf16_mma_body<W: MmaRows, X: MmaCols>(
    w: W,
    x: X,
    rows: u32,
    k: u32,
    n: u32,
    y: &mut DisjointSlice<impl ProductCell>,
    sh: *mut f32,
) {
    let units = if W::SCALED { k / 2 } else { k };
    // SAFETY: the wrapper's spans and launch satisfy the tile's contract.
    let Some(tile) = (unsafe { bf16_mma_tile(w, x, rows, units, n, sh) }) else {
        return;
    };
    // SAFETY: the tile's operands, spans and staging allocation.
    unsafe { mma_tile_store(tile, w, x, (rows, k, n), y, sh) };
}

/// [`bf16_mma_body`]'s store of a finished tile: its cells, those of an
/// exceptional row or column replayed on the exact owner, each stored as its
/// [`ProductCell`].
///
/// # Safety
/// All 256 threads call converged with the tile's operands, checked spans
/// and staging allocation, after a barrier that retired the stages' readers.
#[inline(always)]
pub(crate) unsafe fn mma_tile_store<W: MmaRows, X: MmaCols>(
    mut tile: MmaTile,
    w: W,
    x: X,
    (rows, k, n): (u32, u32, u32),
    y: &mut DisjointSlice<impl ProductCell>,
    sh: *mut f32,
) {
    let units = if W::SCALED { k / 2 } else { k };
    let (rows, k, n) = (rows as usize, k as usize, n as usize);
    // SAFETY: all threads share the original staging allocation and K.
    if unsafe { tile.needs_replay(sh, units as usize) } {
        // SAFETY: the checked output belongs to this tile.
        unsafe { tile.store::<true>(y, rows, n) };
        // Publishes the marked stores and retires the flags' readers.
        thread::sync_threads();
        // SAFETY: spans and allocation are unchanged.
        unsafe { tile.replay(w, x.exact(), (rows, k, n), y, sh) };
    } else {
        // SAFETY: the checked output belongs to this tile.
        unsafe { tile.store::<false>(y, rows, n) };
    }
}

/// Rows of a narrow MMA block ([`bf16_mma_narrow`]): one m16 fragment.
pub(crate) const NARROW_ROWS: usize = 16;
/// Columns of a narrow MMA block: one n8 fragment.
pub(crate) const NARROW_COLS: usize = 8;
/// 32-value chunks of every row and column one narrow stage holds.
const NARROW_CHUNKS: usize = 4;
/// Narrow stages in shared memory: the one warp 0 reads and three being
/// copied in.
const NARROW_STAGES: usize = 4;
/// Words of a narrow stage's rows: per chunk, [`NARROW_ROWS`] lines laid out
/// as the tile's ([`mma_piece_at`]).
const NARROW_ROW_WORDS: usize = NARROW_CHUNKS * NARROW_ROWS * MMA_LINE_WORDS;
/// f32 values from one column of a narrow stage to the next: the stage's
/// values and eight more, so the eight columns one fragment read takes fall
/// in two rounds of the 32 banks.
const NARROW_COL_PITCH: usize = NARROW_CHUNKS * 32 + 8;
/// Words of one narrow stage: the rows' BF16 lines, then the columns' f32
/// values.
const NARROW_STAGE_WORDS: usize = NARROW_ROW_WORDS + NARROW_COLS * NARROW_COL_PITCH;
/// Shared words a narrow block takes.
pub(crate) const NARROW_WORDS: usize = NARROW_STAGES * NARROW_STAGE_WORDS;
// Each thread copies one 16-byte piece of the rows and one of the columns a
// stage; the pipeline's wait leaves the two newest stages' groups pending.
const _: () = assert!(NARROW_CHUNKS * NARROW_ROWS * MMA_PIECES == MMA_THREADS as usize);
const _: () = assert!(NARROW_COLS * NARROW_CHUNKS * 8 == MMA_THREADS as usize);
const _: () = assert!(NARROW_STAGES == 4 && NARROW_ROWS == 16 && NARROW_COLS == 8);
// Stages and column lines start 16-byte aligned.
const _: () = assert!(
    NARROW_ROW_WORDS.is_multiple_of(4)
        && NARROW_COL_PITCH.is_multiple_of(4)
        && NARROW_STAGE_WORDS.is_multiple_of(4)
);

/// Warp 0's fragment of a narrow block ([`bf16_mma_narrow`]), and the cells
/// of the block the exact owner replays: those whose row or column held an
/// operand [`mma_fits`] refuses, the cells [`MmaTile::needs_replay`] marks.
pub(crate) struct NarrowTile {
    values: [f32; 4],
    lane: usize,
    /// Bit `i` (`16 + c`): row offset `i` (column `c`) held an operand
    /// [`mma_fits`] refuses. The same in every thread of the block.
    refused: u32,
}

impl NarrowTile {
    /// Whether the exact owner replays the cell at row offset `r` and column
    /// `c`.
    #[inline(always)]
    pub(crate) fn replays(&self, r: usize, c: usize) -> bool {
        self.refused & (1 << r | 1 << (16 + c)) != 0
    }

    /// Whether the exact owner replays a cell of row offset `r` among the
    /// first `n` columns.
    #[inline(always)]
    pub(crate) fn replays_row(&self, r: usize, n: usize) -> bool {
        self.refused & (1 << r | ((1 << n) - 1) << 16) != 0
    }

    /// Warp 0 stores the cells it keeps of the rows `row0 + r` below `rows`
    /// and of the first `n` columns, at `y[c · rows + row0 + r]`.
    ///
    /// # Safety
    /// Warp 0 of the block calls it; `y` holds `rows · n` outputs, and no other
    /// block writes rows `row0 .. row0 + NARROW_ROWS`.
    #[inline(always)]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    pub(crate) unsafe fn store(
        &self,
        y: &mut DisjointSlice<f32>,
        (rows, n): (usize, usize),
        row0: usize,
    ) {
        for i in 0..4 {
            thread::__unroll_config::<0>();
            let (r, c) = fragment_cell((0, 0), self.lane, i);
            if row0 + r < rows && c < n && !self.replays(r, c) {
                // SAFETY: row row0 + r < rows and column c < n lie inside y;
                // this lane holds the fragment's only copy of the cell.
                unsafe { *y.get_unchecked_mut(c * rows + row0 + r) = self.values[i] };
            }
        }
    }
}

/// The dense MMA tile's products ([`bf16_mma_tile`]) of a narrow block: rows
/// `row0 .. row0 + NARROW_ROWS` of a BF16 weight's words against the first
/// `n` f32 columns of `x`, each output the tile's bits for its row and
/// column. The operands are what the tile stages (the weight's words as they
/// are, each activation rounded to BF16 by `cvt.rn`), and the fragment runs
/// one m16n8k16 chain from zero over K's 16-value steps in increasing order,
/// as each of the tile's fragments does. Every thread of the block copies
/// one row piece and one column piece of each stage with `cp.async`,
/// [`NARROW_STAGES`] deep, a row past the last copied as the last; warp 0
/// reads the rows with `ldmatrix` and its columns from their f32 lines, a
/// column past the last read as the last. Returns warp 0's fragment and, in
/// every thread, the rows and columns that held an operand [`mma_fits`]
/// refuses.
///
/// # Safety
///
/// All 256 threads call converged. `rows >= 1`, `row0 < rows`, `1 <= n <=
/// NARROW_COLS`, `k` a positive multiple of 32; `w` holds `rows · k / 2`
/// words and `x` `n · k` values, both 16-byte aligned; `sh` is the block's
/// own [`NARROW_WORDS`] shared words, 16-byte aligned.
#[inline(always)]
pub(crate) unsafe fn bf16_mma_narrow(
    w: &[u32],
    x: &[f32],
    (rows, k, n): (usize, usize, usize),
    row0: usize,
    sh: *mut u32,
) -> NarrowTile {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let chunks = k / 32;
    let stages = chunks.div_ceil(NARROW_CHUNKS);
    // This thread's row piece: slot `rs` of line `rl` of chunk `rc` of each
    // stage. And its column piece: values `xv .. xv + 4` of column `xc`'s
    // stage values.
    let (rc, rl, rs) = (
        tid / (NARROW_ROWS * MMA_PIECES),
        (tid / MMA_PIECES) % NARROW_ROWS,
        tid % MMA_PIECES,
    );
    let r_src = (row0 + rl).min(rows - 1) * (k / 2) + rc * MMA_LINE_WORDS + 4 * rs;
    let r_dst = rc * NARROW_ROWS * MMA_LINE_WORDS + mma_piece_at(rl, rs);
    let (xc, xv) = (tid / (NARROW_CHUNKS * 8), 4 * (tid % (NARROW_CHUNKS * 8)));
    let x_src = xc * k + xv;
    let x_dst = NARROW_ROW_WORDS + xc * NARROW_COL_PITCH + xv;
    let copy = |s: usize| {
        let stage = sh.wrapping_add((s % NARROW_STAGES) * NARROW_STAGE_WORDS);
        if s * NARROW_CHUNKS + rc < chunks {
            // SAFETY: the row's chunk lies inside w (a live chunk of a row
            // below rows), 16-byte aligned (k / 2 a multiple of 16 words, w
            // aligned); the destination is this thread's own aligned piece of
            // the stage, which no warp reads before the wait and barrier that
            // complete it.
            unsafe {
                cp_async_cg_16(
                    stage.wrapping_add(r_dst),
                    w.as_ptr().add(r_src + s * NARROW_CHUNKS * MMA_LINE_WORDS),
                )
            };
        }
        if xc < n && s * NARROW_CHUNKS + xv / 32 < chunks {
            // SAFETY: the four values lie in live column xc inside x, 16-byte
            // aligned (k a multiple of 32, x aligned); the destination is this
            // thread's own 16-byte aligned piece of the stage, which no warp
            // reads before the wait and barrier that complete it.
            unsafe {
                cp_async_cg_16(
                    stage.wrapping_add(x_dst),
                    x.as_ptr().add(x_src + s * NARROW_CHUNKS * 32).cast::<u32>(),
                )
            };
        }
    };
    for s in 0..NARROW_STAGES - 1 {
        if s < stages {
            copy(s);
        }
        // SAFETY: commits this thread's copies of stage s as one group,
        // empty past the last stage.
        unsafe { cp_async_commit_group() };
    }

    let warp0 = tid < 32;
    let (g, t) = (lane / MMA_ROW_LANES, lane % MMA_ROW_LANES);
    let a_lane = mma_piece_at(lane % 16, lane / 16);
    let col = g.min(n - 1);
    // SAFETY: sh is this block's shared allocation (this fn's contract).
    let shared = unsafe { cvta_generic_to_shared_u32(sh.cast_const().cast::<u8>()) };
    let mut acc = [0.0f32; 4];
    // Bit 0 (1): a value of row g (g + 8) failed mma_fits; bit 2: a value of
    // column `col`.
    let mut bad = 0u32;
    for s in 0..stages {
        // SAFETY: NARROW_STAGES − 1 groups precede the loop and one each
        // pass, so leaving the two newest pending completes stage s's. The
        // barrier then publishes every thread's copies of it, and tells that
        // warp 0 is done reading stage s − 1, whose buffer the copy below
        // refills.
        unsafe { cp_async_wait_group(2) };
        thread::sync_threads();
        if s + NARROW_STAGES - 1 < stages {
            copy(s + NARROW_STAGES - 1);
        }
        // SAFETY: one group per pass, empty near the end.
        unsafe { cp_async_commit_group() };
        if warp0 {
            let buf = (s % NARROW_STAGES) * NARROW_STAGE_WORDS;
            let rows_at = shared + (4 * buf) as u32;
            let cols = sh
                .wrapping_add(buf + NARROW_ROW_WORDS + col * NARROW_COL_PITCH + 2 * t)
                .cast::<F32x2>();
            for c in 0..NARROW_CHUNKS {
                thread::__unroll_config::<0>();
                if s * NARROW_CHUNKS + c < chunks {
                    for step in 0..2 {
                        thread::__unroll_config::<0>();
                        let at = c * NARROW_ROWS * MMA_LINE_WORDS + (a_lane ^ (8 * step));
                        // SAFETY: the warp is converged; each lane's 16-byte
                        // row lies in this published stage's row lines.
                        let a = unsafe { ldmatrix_x4_shared_u32(rows_at + (4 * at) as u32) };
                        let xp = cols.wrapping_add(16 * c + 8 * step);
                        // SAFETY: values 2t and 2t + 1 of the step's first
                        // eight of column `col`, 8-byte aligned, published by
                        // the barrier.
                        let F32x2([x0, x1]) = unsafe { *xp };
                        // SAFETY: values 8 + 2t and 9 + 2t of the step's sixteen of column
                        // `col`, 8-byte aligned, published by the barrier.
                        let F32x2([x2, x3]) = unsafe { *xp.add(4) };
                        bad |= u32::from(!(mma_fits_bf16x2(a[0]) & mma_fits_bf16x2(a[2])));
                        bad |= u32::from(!(mma_fits_bf16x2(a[1]) & mma_fits_bf16x2(a[3]))) << 1;
                        let fits = mma_fits(x0) & mma_fits(x1) & mma_fits(x2) & mma_fits(x3);
                        bad |= u32::from(!fits) << 2;
                        let b = [cvt_bf16x2_f32(x0, x1), cvt_bf16x2_f32(x2, x3)];
                        // SAFETY: the whole warp is converged; A is row-major
                        // and B column-major in the m16n8k16 fragment layout.
                        acc = unsafe { mma_m16n8k16_f32_bf16(acc, a, b) };
                    }
                }
            }
        }
    }

    // The votes go through the stage the pipeline left unread: the copies
    // still pending are empty, and warp 0 last read another buffer.
    let flag = sh.wrapping_add((stages % NARROW_STAGES) * NARROW_STAGE_WORDS);
    if warp0 {
        let (lo, hi, cols) = (
            warp::ballot(bad & 1 != 0),
            warp::ballot(bad & 2 != 0),
            warp::ballot(bad & 4 != 0),
        );
        if lane == 0 {
            // Lanes 4g .. 4g + 3 hold row g's, row g + 8's and column g's votes.
            let quad = |v: u32, g: usize| u32::from((v >> (MMA_ROW_LANES * g)) & 0xf != 0);
            let mut refused = 0u32;
            for g in 0..8 {
                thread::__unroll_config::<0>();
                refused |= quad(lo, g) << g | quad(hi, g) << (8 + g) | quad(cols, g) << (16 + g);
            }
            // SAFETY: lane 0 of warp 0 is the vote word's only writer.
            unsafe { *flag = refused };
        }
    }
    thread::sync_threads();
    // SAFETY: the barrier published the vote word.
    let refused = unsafe { *flag };
    NarrowTile {
        values: acc,
        lane,
        refused,
    }
}

/// What an epilogue writes of its output columns beside their f32 values
/// (a norm gate's [`bf16_mma_norm_gate_body`], the HC mix's
/// `hc_gated::up_mma_mix_body`): nothing, or the MMA form the next site
/// reads ([`Bf16Out`], `nvfp4mma.rs`'s `Mxfp8Out`), each column's refusal
/// word zero before the launch and ORed by its pieces.
pub(crate) trait OutForm: Copy {
    /// Whether the epilogue writes anything.
    const WRITES: bool;

    /// Writes the form of the `N` 32-value blocks (`N` 1 or 4) of column `c`
    /// from row `row0`, `k` values a column, that one warp holds: lane `l`'s
    /// `v[i]` is row `row0 + l + 32·i`'s, `row0` a multiple of `32·N`.
    ///
    /// # Safety
    ///
    /// The whole warp calls converged on a live column the form holds room
    /// for, and the blocks lie inside its `k` values.
    unsafe fn store<const N: usize>(self, v: [f32; N], c: usize, row0: usize, k: usize);
}

impl OutForm for () {
    const WRITES: bool = false;

    #[inline(always)]
    unsafe fn store<const N: usize>(self, _: [f32; N], _: usize, _: usize, _: usize) {}
}

/// [`Bf16Cols`]' words and refusal words of an epilogue's output columns,
/// with [`bf16_cols_body`]'s bits: each value rounded to nearest even, the
/// even lane of a pair writing their word.
#[derive(Clone, Copy)]
pub(crate) struct Bf16Out {
    pub(crate) words: *mut u32,
    pub(crate) refused: *mut u32,
}

impl OutForm for Bf16Out {
    const WRITES: bool = true;

    #[inline(always)]
    #[allow(clippy::needless_range_loop, reason = "indexed CUDA-unrolled arrays")]
    unsafe fn store<const N: usize>(self, v: [f32; N], c: usize, row0: usize, k: usize) {
        let mut fits = true;
        for i in 0..N {
            thread::__unroll_config::<0>();
            // SAFETY: rows row0 + 32i .. + 32 lie in the blocks (this fn's
            // contract).
            fits &= unsafe { self.pair(v[i], c * k + row0 + 32 * i) };
        }
        // SAFETY: c is a live column the form holds (this fn's contract).
        unsafe { refuse_or(self.refused, c, fits) };
    }
}

impl Bf16Out {
    /// Writes the words of the 32 values from value `at` (a multiple of 32)
    /// that lane `l`'s `v` is value `at + l` of; whether `v` fits.
    ///
    /// # Safety
    ///
    /// The whole warp calls converged, and the 16 words lie inside the form.
    #[inline(always)]
    unsafe fn pair(self, v: f32, at: usize) -> bool {
        let lane = warp::lane_id() as usize;
        let word = cvt_bf16x2_f32(v, warp::shuffle_xor_f32(v, 1));
        if lane.is_multiple_of(2) {
            // SAFETY: word (at + lane) / 2 is this even lane's own (this fn's
            // contract).
            unsafe { *self.words.add((at + lane) / 2) = word };
        }
        mma_fits(v)
    }
}

/// ORs 1 into column `c`'s refusal word when a lane of the warp saw a value
/// [`mma_fits`] refuses (`fits` false). Every lane of the warp calls it.
///
/// # Safety
///
/// `refused` holds a live word for column `c`, which this launch only ORs.
#[inline(always)]
pub(crate) unsafe fn refuse_or(refused: *mut u32, c: usize, fits: bool) {
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
    if warp::any(!fits) && warp::lane_id() == 0 {
        // SAFETY: word c is live and 4-byte aligned (this fn's contract); the
        // launch only ORs it atomically.
        let word = unsafe { DeviceAtomicU32::from_ptr(refused.add(c)) };
        word.fetch_or(1, AtomicOrdering::Relaxed);
    }
}

/// Dense prompt product ([`bf16_mma_body`]'s, BF16 or block-scaled E4M3)
/// followed by the shared sigmoid norm, its output also in `form`. Normal
/// tiles exchange z through the retired staging allocation; exceptional
/// tiles finish through the exact product owner first. K counts values; a
/// scaled tile's units are value pairs.
///
/// # Safety
///
/// A block has 256 converged threads and the MMA raster's grid. rows is a
/// positive multiple of HEAD, n is positive and k a positive multiple of 64
/// (128 for a scaled tile). w and x meet [`bf16_mma_tile`]'s contract for
/// rows*k weight values and n*k activations; x.exact holds n*k values. o, z
/// and y hold rows*n values, gain holds HEAD. sh is this block's
/// 16-byte-aligned allocation of STAGES*STAGE_FLOATS floats. Output and
/// scratch cells belong to this block; `form` holds room for the n columns
/// of rows values.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "flat device operands and indexed CUDA-unrolled arrays"
)]
pub(crate) unsafe fn bf16_mma_norm_gate_body<W: MmaRows, X: MmaCols, F: OutForm>(
    w: W,
    x: X,
    rows: u32,
    k: u32,
    n: u32,
    o: &[f32],
    gain: &[f32],
    eps: f32,
    fault: FaultSink,
    z: &mut DisjointSlice<f32>,
    y: &mut DisjointSlice<f32>,
    form: F,
    sh: *mut f32,
) {
    const {
        assert!(MMA_ROWS == HEAD && MMA_COLS == 128);
        assert!(MMA_WARP_COLS * HEAD <= STAGES * STAGE_FLOATS);
    };
    let units = if W::SCALED { k / 2 } else { k };
    // SAFETY: the entry supplies the checked MMA launch and staging allocation.
    let Some(mut tile) = (unsafe { bf16_mma_tile(w, x, rows, units, n, sh) }) else {
        return;
    };
    let (rows, k, n) = (rows as usize, k as usize, n as usize);
    // SAFETY: the tile's original staging allocation and K are unchanged.
    let replay = unsafe { tile.needs_replay(sh, units as usize) };
    // Retire every vote and flag reader before reusing the staging allocation.
    thread::sync_threads();
    if replay {
        // SAFETY: each marked output cell belongs to this tile.
        unsafe { tile.store::<true>(z, rows, n) };
        thread::sync_threads();
        // SAFETY: all 256 threads reach this converged with the tile's checked spans and its
        // staging allocation sh; the barrier above retired the store and every shared reader.
        unsafe { tile.replay(w, x.exact(), (rows, k, n), z, sh) };
        thread::sync_threads();
    }
    let lane = warp::lane_id() as usize;
    let warp = thread::threadIdx_x() as usize / 32;
    for pass in 0..MMA_COLS / MMA_WARP_COLS {
        let start = tile.col0 + pass * MMA_WARP_COLS;
        if !replay {
            if warp % (MMA_COLS / MMA_WARP_COLS) == pass {
                for cell in 0..MMA_CELLS {
                    thread::__unroll_config::<0>();
                    let (r, c) = tile.coordinates(cell);
                    if c < n {
                        let at = (c - start) * HEAD + r - tile.row0;
                        // SAFETY: this warp owns the slab's column range, each
                        // fragment a disjoint cell in its shared 32-column head.
                        unsafe { *sh.add(at) = tile.value(cell) };
                    }
                }
            }
            thread::sync_threads();
        }
        for token in 0..MMA_WARP_COLS / (MMA_THREADS as usize / 32) {
            let c = start + warp + token * (MMA_THREADS as usize / 32);
            if c < n {
                let base = c * rows + tile.row0;
                let mut ov = [0.0f32; 4];
                let mut zv = [0.0f32; 4];
                let mut wv = [0.0f32; 4];
                for i in 0..4 {
                    thread::__unroll_config::<0>();
                    let at = lane + 32 * i;
                    // SAFETY: c is live and the tile owns one complete head.
                    ov[i] = unsafe { *o.get_unchecked(base + at) };
                    // SAFETY: at < HEAD <= gain.len().
                    wv[i] = unsafe { *gain.get_unchecked(at) };
                    zv[i] = if replay {
                        // SAFETY: the exact owner published this live z cell.
                        unsafe { *z.get_unchecked_mut(base + at) }
                    } else {
                        // SAFETY: the preceding barrier published this slab.
                        unsafe { *sh.add((c - start) * HEAD + at) }
                    };
                }
                let values = norm_gate_values::<GATE_SIGMOID>(ov, zv, wv, eps, fault);
                for i in 0..4 {
                    thread::__unroll_config::<0>();
                    // SAFETY: this warp owns c, each lane four disjoint values.
                    unsafe { *y.get_unchecked_mut(base + lane + 32 * i) = values[i] };
                }
                // SAFETY: c < n is warp-uniform, so the warp is converged on a
                // live column; this tile owns its head at row0, and form holds
                // room for the n columns of rows values (this fn's contract).
                unsafe { form.store(values, c, tile.row0, rows) };
            }
        }
        // Retire the readers before the next slab overwrites shared memory.
        thread::sync_threads();
    }
}

impl Gemm32Kernels {
    /// Enqueue `y[c · rows + r] = Σ_k w[r·k + k'] · x[c·k + k']` for the
    /// first `n` columns of `x`, `w` an F32 weight of `rows × k` (module
    /// doc): every output bit for bit `Q8F32Kernels::enqueue_f32_gemv`'s for
    /// its row and column. `k` must be a positive multiple of 64 (the
    /// staged line) and `w` and `x` 16-byte aligned (the staged copies are
    /// 16-byte pieces); another is a named `Shape` error. Asynchronous,
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
        if w.buf().len() < rows * k {
            return Err(GpuError::shape(
                what,
                format!("w.len() {} >= rows*k = {rows}*{k}", w.buf().len()),
            ));
        }
        let w_at = w.buf().cu_deviceptr();
        if !w_at.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("w at {w_at:#x} must be 16-byte aligned"),
            ));
        }
        let t = tile_launch(what, rows, k, x, n, y)?;
        let prep =
            self.module
                .prepare_f32_tile_gemm(LaunchConfig1D::new(t.grid, F32T_THREADS_U32, 0))?;
        self.module
            .f32_tile_gemm(stream, &prep, w.buf(), x, t.rows, t.k, t.n, y)?;
        Ok(())
    }

    /// Enqueue [`Self::enqueue_f32_tile`]'s product for an E4M3 weight in
    /// its two planes (`rows × k` values, `k = 4 · w.qs.cols()`), its values
    /// decoded as they are staged (module doc) by `mxfp8_tile_gemm` or
    /// `fp8_b128_tile_gemm` as its scales say: every output bit for bit
    /// `Q8F32Kernels::enqueue_f32_gemv`'s over the host's dequantized rows.
    /// `k` must be a positive multiple of 128 with `e` `k / 128` words a row
    /// over the same rows, and `x` 16-byte aligned; another is a named
    /// `Shape` error. Asynchronous, allocation-free, capturable.
    pub fn enqueue_e4m3_tile(
        &self,
        stream: &CudaStream,
        w: E4m3Weight<'_>,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_e4m3_tile";
        let (rows, k) = w.dimensions(what)?;
        let t = tile_launch(what, rows, k, x, n, y)?;
        let cfg = LaunchConfig1D::new(t.grid, F32T_THREADS_U32, 0);
        let (qs, e, module) = (w.qs.buf(), w.e.buf(), &self.module);
        match w.scales {
            E4m3Scales::Mxfp8 => {
                let prep = module.prepare_mxfp8_tile_gemm(cfg)?;
                module.mxfp8_tile_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
            E4m3Scales::Fp8B128 => {
                let prep = module.prepare_fp8_b128_tile_gemm(cfg)?;
                module.fp8_b128_tile_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
        }
        Ok(())
    }
}

impl Gemm32Kernels {
    /// Dense E4M3 prompt product, with BF16 operands and F32 accumulation.
    /// The stored planes are decoded in shared memory, never rewritten.
    pub fn enqueue_e4m3_mma(
        &self,
        stream: &CudaStream,
        w: E4m3Weight<'_>,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_e4m3_mma";
        let (rows, k) = w.dimensions(what)?;
        let t = mma_launch(what, rows, k, x, n, y)?;
        let cfg = LaunchConfig1D::new(t.grid, MMA_THREADS, 0);
        let (qs, e, module) = (w.qs.buf(), w.e.buf(), &self.module);
        match w.scales {
            E4m3Scales::Mxfp8 => {
                let prep = module.prepare_mxfp8_mma_gemm(cfg)?;
                module.mxfp8_mma_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
            E4m3Scales::Fp8B128 => {
                let prep = module.prepare_fp8_b128_mma_gemm(cfg)?;
                module.fp8_b128_mma_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
        }
        Ok(())
    }

    /// `Q8F32Kernels::enqueue_bf16_mma`'s product, bit for bit, over the BF16
    /// form `act` holds of the first `n` columns of `x`
    /// ([`Self::enqueue_bf16_cols`]): the tile copies the rounded words, and
    /// only a replay reads `x`. Each output is stored as its [`ProductOut`]'s
    /// cell. A form of other rows or of fewer columns is a named `State`
    /// error. Asynchronous, allocation-free, capturable.
    pub fn enqueue_bf16_mma_cols<'y>(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        x: &DeviceBuffer<f32>,
        act: &GemmAct32,
        n: usize,
        y: impl Into<ProductOut<'y>>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_bf16_mma_cols";
        let y = y.into();
        let t = match &y {
            ProductOut::F32(y) => bf16_mma_launch(what, w, x, n, y)?,
            ProductOut::Bf16(y) => bf16_mma_launch(what, w, x, n, y)?,
        };
        let form = held_form(what, act, MmaForm::Bf16, x, &t)?;
        let cfg = LaunchConfig1D::new(t.grid, MMA_THREADS, 0);
        let (m, words, refused) = (&self.module, &form.words, &form.refused);
        let (w, rows, k, n) = (w.buf(), t.rows, t.k, t.n);
        match y {
            ProductOut::F32(y) => {
                let prep = m.prepare_bf16_mma_cols_gemm(cfg)?;
                m.bf16_mma_cols_gemm(stream, &prep, w, x, words, refused, rows, k, n, y)?;
            }
            ProductOut::Bf16(y) => {
                let prep = m.prepare_bf16_mma_cols_gemm_bf16(cfg)?;
                m.bf16_mma_cols_gemm_bf16(stream, &prep, w, x, words, refused, rows, k, n, y)?;
            }
        }
        Ok(())
    }

    /// The BF16 prompt product and shared sigmoid norm in one launch. z is
    /// scratch for exceptional exact replay only; ordinary tiles never write it.
    #[allow(
        clippy::too_many_arguments,
        reason = "product operands and its norm tail"
    )]
    pub fn enqueue_bf16_mma_norm_gate(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        x: &DeviceBuffer<f32>,
        act: &GemmAct32,
        n: usize,
        z: &mut DeviceBuffer<f32>,
        tail: NormGateTail<'_>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_bf16_mma_norm_gate";
        let t = bf16_mma_launch(what, w, x, n, z)?;
        tail.check_rows(what, w.rows(), n, z.len())?;
        let form = held_form(what, act, MmaForm::Bf16, x, &t)?;
        let prep =
            self.module
                .prepare_bf16_mma_norm_gate(LaunchConfig1D::new(t.grid, MMA_THREADS, 0))?;
        self.module.bf16_mma_norm_gate(
            stream,
            &prep,
            w.buf(),
            x,
            &form.words,
            &form.refused,
            t.rows,
            t.k,
            t.n,
            tail.o,
            tail.w,
            tail.eps,
            tail.fault,
            z,
            tail.y,
        )?;
        Ok(())
    }

    /// The W8A8 product of an MXFP8 weight's planes as stored over the MXFP8
    /// form `act` holds of the first `n` columns of `x`
    /// ([`Self::enqueue_mxfp8_cols`]): E4M3 operands with their UE8M0 block
    /// scales on the block-scaled tensor cores, F32 accumulation, each
    /// output's 32-value steps in increasing K. The cells of a row or column
    /// that holds an operand outside the tensor cores' range replay on the
    /// exact owner over `x`, as [`Self::enqueue_e4m3_mma`]'s do. The planes
    /// must be MXFP8 and 16-byte aligned ([`mxfp8_w8a8_reads`]); another, or a
    /// form of other rows or of fewer columns, is a named error, and so is a
    /// build without the native block-scaled kernels. Each output is stored
    /// as its [`ProductOut`]'s cell. Asynchronous, allocation-free,
    /// capturable.
    #[cfg_attr(
        not(bloomery_nvfp4_mma),
        allow(unused_variables, reason = "the native build's launch reads them")
    )]
    pub fn enqueue_mxfp8_w8a8<'y>(
        &self,
        stream: &CudaStream,
        w: E4m3Weight<'_>,
        x: &DeviceBuffer<f32>,
        act: &GemmAct32,
        n: usize,
        y: impl Into<ProductOut<'y>>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_mxfp8_w8a8";
        let y = y.into();
        let t = match &y {
            ProductOut::F32(y) => mxfp8_mma_launch(what, w, x, n, y)?,
            ProductOut::Bf16(y) => mxfp8_mma_launch(what, w, x, n, y)?,
        };
        let form = held_form(what, act, MmaForm::Mxfp8, x, &t)?;
        #[cfg(bloomery_nvfp4_mma)]
        return self.native.mxfp8_w8a8(stream, w, x, form, &t, y);
        #[cfg(not(bloomery_nvfp4_mma))]
        Err(GpuError::state(what, NATIVE_BUILD))
    }

    /// [`Self::enqueue_mxfp8_w8a8`] and the shared sigmoid norm in one
    /// kernel launch, as [`Self::enqueue_bf16_mma_norm_gate`] finishes a BF16
    /// product: z is scratch for exceptional exact replay only. The kernel
    /// also writes the MXFP8 form of the norm's output `tail.y` into `out`,
    /// a scratch of `k` = the weight's rows made with room for it, bit for
    /// bit [`Self::enqueue_mxfp8_cols`]'s, so the next MXFP8 site reads it
    /// without a quantizer launch; `out`'s other form is left as another
    /// rows'. A memset of `out`'s refusal words, which the kernel's heads OR,
    /// comes first: two graph nodes.
    #[cfg_attr(
        not(bloomery_nvfp4_mma),
        allow(unused_variables, reason = "the native build's launch reads them")
    )]
    pub fn enqueue_mxfp8_w8a8_norm_gate(
        &self,
        stream: &CudaStream,
        a: Mxfp8NormGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_mxfp8_w8a8_norm_gate";
        let Mxfp8NormGateArgs {
            w,
            x,
            act,
            n_cols: n,
            z,
            tail,
            out,
        } = a;
        let t = mxfp8_mma_launch(what, w, x, n, z)?;
        tail.check_rows(what, t.rows as usize, n, z.len())?;
        let form = held_form(what, act, MmaForm::Mxfp8, x, &t)?;
        let y = tail.y.cu_deviceptr();
        let into = out.out_planes(stream, what, MmaForm::Mxfp8, (t.rows as usize, n))?;
        #[cfg(bloomery_nvfp4_mma)]
        {
            self.native.mxfp8_w8a8_norm_gate(
                stream,
                super::nvfp4mma::NativeNormGateArgs {
                    w,
                    x,
                    form,
                    t: &t,
                    z,
                    tail,
                    out: into,
                },
            )?;
            out.wrote(MmaForm::Mxfp8, y, n);
            Ok(())
        }
        #[cfg(not(bloomery_nvfp4_mma))]
        Err(GpuError::state(what, NATIVE_BUILD))
    }

    /// The wide HC up/mix launch (`hc_gated::up_mma_mix_body`) that also
    /// writes the MXFP8 form of `mixed`'s columns into `a.planes`
    /// ([`HcUpMixArgs`]). A build without the native block-scaled kernels
    /// refuses it by name.
    #[cfg_attr(
        not(bloomery_nvfp4_mma),
        allow(unused_variables, reason = "the native build's launch reads them")
    )]
    pub(crate) fn enqueue_hc_up_mix_mxfp8(
        &self,
        stream: &CudaStream,
        a: HcUpMixArgs<'_>,
    ) -> Result<(), GpuError> {
        #[cfg(bloomery_nvfp4_mma)]
        {
            self.native.hc_up_mix_mxfp8(stream, a)
        }
        #[cfg(not(bloomery_nvfp4_mma))]
        Err(GpuError::state(
            "Gemm32Kernels::enqueue_hc_up_mix_mxfp8",
            NATIVE_BUILD,
        ))
    }
}

/// What a build without the native block-scaled kernels lacks for an MXFP8
/// form or a W8A8 product.
#[cfg(not(bloomery_nvfp4_mma))]
pub(super) const NATIVE_BUILD: &str = "a native block-scaled build (sm_120a or sm_121a)";

/// Whether an MXFP8 site's product runs on the native W8A8 tile
/// ([`Gemm32Kernels::enqueue_mxfp8_w8a8`]): `w`'s planes are MXFP8 and
/// 16-byte aligned, and `act` holds the MXFP8 form of the first `n` columns
/// of `x`.
#[must_use]
pub(crate) fn mxfp8_w8a8_reads(
    w: &E4m3Weight<'_>,
    act: &GemmAct32,
    x: &DeviceBuffer<f32>,
    n: usize,
) -> bool {
    w.scales == E4m3Scales::Mxfp8
        && w.qs.buf().cu_deviceptr().is_multiple_of(16)
        && act.holds(MmaForm::Mxfp8, x, n)
}

/// `form` of `act` for a tile launch `t` over the first `t.n` columns of `x`:
/// a scratch of another K is a named `Shape` error, a form of other rows or
/// of fewer columns a named `State` error, as `what`'s.
fn held_form<'a>(
    what: &'static str,
    act: &'a GemmAct32,
    form: MmaForm,
    x: &DeviceBuffer<f32>,
    t: &TileLaunch,
) -> Result<&'a ColsForm, GpuError> {
    if act.k() != t.k as usize {
        return Err(GpuError::shape(
            what,
            format!("act holds {} values a column, the weight {}", act.k(), t.k),
        ));
    }
    act.form_of(form, x, t.n as usize).ok_or(GpuError::state(
        what,
        match form {
            MmaForm::Bf16 => "act's BF16 form of x's first n columns (enqueue_bf16_cols)",
            MmaForm::Mxfp8 => "act's MXFP8 form of x's first n columns (enqueue_mxfp8_cols)",
        },
    ))
}

/// [`mma_launch`] for an MXFP8 weight's planes as stored. The tile copies
/// the value plane in 16-byte pieces; other scales or alignment are a named
/// `Shape` error.
fn mxfp8_mma_launch<T>(
    what: &'static str,
    w: E4m3Weight<'_>,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &DeviceBuffer<T>,
) -> Result<TileLaunch, GpuError> {
    let (rows, k) = w.dimensions(what)?;
    if w.scales != E4m3Scales::Mxfp8 {
        return Err(GpuError::shape(
            what,
            format!("the weight's scales are {:?}, not MXFP8's", w.scales),
        ));
    }
    crate::tensor::check_alignment::<U32x4, _>(what, "w.qs", w.qs.buf())?;
    mma_launch(what, rows, k, x, n, y)
}

/// Checked spans shared with the exact tile, with the MMA tile's grid.
pub(crate) fn mma_launch<T>(
    what: &'static str,
    rows: usize,
    k: usize,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &DeviceBuffer<T>,
) -> Result<TileLaunch, GpuError> {
    let mut launch = tile_launch(what, rows, k, x, n, y)?;
    launch.grid = launch_u32(what, "grid", n.div_ceil(MMA_COLS) * rows.div_ceil(MMA_ROWS))?;
    Ok(launch)
}

/// [`mma_launch`] over an `x` that holds its values as BF16 words, two a
/// word: `x` holds `n·k / 2` words.
pub(crate) fn mma_launch_words(
    what: &'static str,
    rows: usize,
    k: usize,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &DeviceBuffer<f32>,
) -> Result<TileLaunch, GpuError> {
    let values = x
        .len()
        .checked_mul(2)
        .ok_or_else(|| GpuError::shape(what, "x's BF16 words to values overflows"))?;
    let mut launch = tile_spans(what, rows, k, (values, x.cu_deviceptr()), n, y)?;
    launch.grid = launch_u32(what, "grid", n.div_ceil(MMA_COLS) * rows.div_ceil(MMA_ROWS))?;
    Ok(launch)
}

/// [`mma_launch`] for a BF16 weight's `rows × 2·cols` values. The tile copies
/// its words in 16-byte pieces, so they must be 16-byte aligned; another is a
/// named `Shape` error.
pub(crate) fn bf16_mma_launch<T>(
    what: &'static str,
    w: &DeviceTensor<u32>,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &DeviceBuffer<T>,
) -> Result<TileLaunch, GpuError> {
    let k = w
        .cols()
        .checked_mul(2)
        .ok_or_else(|| GpuError::shape(what, "BF16 words to values overflows"))?;
    let launch = mma_launch(what, w.rows(), k, x, n, y)?;
    crate::tensor::check_alignment::<U32x4, _>(what, "w", w.buf())?;
    Ok(launch)
}

/// [`Gemm32Kernels::enqueue_mxfp8_w8a8_norm_gate`]'s arguments: the MXFP8
/// weight `w` over the MXFP8 form `act` holds of the first `n_cols` columns
/// of `x`, the exact replay's scratch `z` (`[n_cols][rows]`), the norm and
/// gate operands `tail`, and `out`, the scratch the output's MXFP8 form goes
/// into.
pub struct Mxfp8NormGateArgs<'a> {
    pub w: E4m3Weight<'a>,
    pub x: &'a DeviceBuffer<f32>,
    pub act: &'a GemmAct32,
    pub n_cols: usize,
    pub z: &'a mut DeviceBuffer<f32>,
    pub tail: NormGateTail<'a>,
    pub out: &'a mut GemmAct32,
}

/// [`Gemm32Kernels::enqueue_hc_up_mix_mxfp8`]'s arguments, as
/// `hc_gated::HcWideKernels::enqueue_mix` checked them: the launch `cfg`
/// over the BF16 up weight `w`, the bottleneck `lo` with its BF16 words
/// `lo_words` and refusal words `lo_refused`, the normed rows `xn`, `m`
/// columns of `hidden` values, the fault sink, the outputs `g` and `mixed`,
/// and `planes`, the words, scale words and zeroed refusal words of
/// `mixed`'s MXFP8 form ([`GemmAct32::out_planes`]).
#[cfg_attr(
    not(bloomery_nvfp4_mma),
    allow(dead_code, reason = "the native build's launch reads them")
)]
pub(crate) struct HcUpMixArgs<'a> {
    pub(crate) cfg: LaunchConfig1D,
    pub(crate) w: &'a DeviceBuffer<u32>,
    pub(crate) lo: &'a DeviceBuffer<f32>,
    pub(crate) lo_words: &'a DeviceBuffer<u32>,
    pub(crate) lo_refused: &'a DeviceBuffer<u32>,
    /// The normed streams' BF16 words.
    pub(crate) xn: &'a DeviceBuffer<u32>,
    pub(crate) hidden: u32,
    pub(crate) m: u32,
    pub(crate) fault: FaultSink,
    pub(crate) g: &'a mut DeviceBuffer<f32>,
    pub(crate) mixed: &'a mut DeviceBuffer<f32>,
    pub(crate) planes: [&'a mut DeviceBuffer<u32>; 3],
}

/// A tile launch's grid and its `u32` arguments.
pub(crate) struct TileLaunch {
    pub(crate) grid: u32,
    pub(crate) rows: u32,
    pub(crate) k: u32,
    pub(crate) n: u32,
}

/// The checks every tile launcher shares, beyond the weight's own: `k` a
/// positive multiple of [`LINE`], `rows` and `n` positive, `x` holding `n`
/// columns of `k` values 16-byte aligned and `y` `rows · n` outputs; each
/// failure a named `Shape` error as `what`'s.
pub(crate) fn tile_launch<T>(
    what: &'static str,
    rows: usize,
    k: usize,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &DeviceBuffer<T>,
) -> Result<TileLaunch, GpuError> {
    tile_spans(what, rows, k, (x.len(), x.cu_deviceptr()), n, y)
}

/// [`tile_launch`]'s checks over an `x` of `x_len` values at `x_at`.
fn tile_spans<T>(
    what: &'static str,
    rows: usize,
    k: usize,
    (x_len_values, x_at): (usize, cuda_core::sys::CUdeviceptr),
    n: usize,
    y: &DeviceBuffer<T>,
) -> Result<TileLaunch, GpuError> {
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
    let (Some(x_len), Some(y_len)) = (n.checked_mul(k), rows.checked_mul(n)) else {
        return Err(GpuError::shape(
            what,
            format!("n*k = {n}*{k} or rows*n = {rows}*{n} overflows"),
        ));
    };
    if x_len_values < x_len || y.len() < y_len {
        return Err(GpuError::shape(
            what,
            format!(
                "x holds {x_len_values} values, want n*k = {n}*{k}; y.len() {} >= rows*n = {rows}*{n}",
                y.len()
            ),
        ));
    }
    if !x_at.is_multiple_of(16) {
        return Err(GpuError::shape(
            what,
            format!("x at {x_at:#x} must be 16-byte aligned"),
        ));
    }
    let grid = n.div_ceil(TILE_COLS) * rows.div_ceil(TILE_ROWS);
    Ok(TileLaunch {
        grid: launch_u32(what, "grid", grid)?,
        rows: launch_u32(what, "rows", rows)?,
        k: launch_u32(what, "k", k)?,
        n: launch_u32(what, "n", n)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_mma_layout() {
        let mut seen = std::collections::HashSet::new();
        for warp in 0..8 {
            for lane in 0..32 {
                let tile = MmaTile {
                    values: [[[0.0; 4]; 4]; 4],
                    row0: 128,
                    col0: 256,
                    warp,
                    lane,
                    lines: 0,
                    bad: 0,
                };
                let (r0, c0) = tile.warp_origin();
                for cell in 0..MMA_CELLS {
                    let (r, c) = tile.coordinates(cell);
                    assert!((128..256).contains(&r) && (256..384).contains(&c));
                    assert!(seen.insert((r, c)));
                    assert!((r - r0) / 16 == cell / 16 && c - c0 < MMA_WARP_COLS);
                }
            }
        }
        assert_eq!(seen.len(), MMA_ROWS * MMA_COLS);
    }
}
