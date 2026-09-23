//! The op set of P1–P7 has no f32 concat or strided per-head addressing
//! (both live in the CPU chain: the `kvr`/flash-row concats and the
//! per-head `wv_b`/q_nope2 legs of `attn.rs`), so this file carries one
//! small `#[cuda_module]` of its own — decision 6's per-file module rule:
//! a pair-table gather and two per-head gemv wrappers. The wrappers give
//! the q_nope2 (derived Q8_0) and wv_b (Q3_K) sites one launch each: one
//! warp per output row, the row's head selecting both the activation
//! slice it reads (an x base or a column of the quantized activation)
//! and the output slot it writes, m = 1 through the gated row bodies
//! (`q8f32::q8_0_lane_partial_1col`, `cores::q3k_row_dot`) — so every dot
//! equals the plain gemv's on the same row and column bit for bit.
//! Everything else runs the gated kernels verbatim.

use crate::GpuError;
use crate::launch_u32;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

// ------------------------------------------------------------- step kernels

#[cuda_module]
mod step_kernels {
    use super::*;
    use crate::cores::q3k_row_dot;
    use crate::q8f32::q8_0_lane_partial_1col;
    use cuda_device::warp;

    /// `y[dst_idx[i]] = x[src_idx[i]]` for `i < n` — the f32 concat/extract
    /// the step needs and no P4 op provides. Both tables are load-time
    /// constants built by the host side from the buffer shapes it owns, so a
    /// pair is always in range; the guard keeps a wrong table a deterministic
    /// no-op rather than an out-of-bounds access (the policy `embed_rows`
    /// takes for its device-resident ids). Destinations must not overlap.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= 1,
            src_idx.len() >= n,
            dst_idx.len() >= n,
            y.len() >= 1
        )
    )]
    pub fn gather_pairs(
        x: &[f32],
        src_idx: &[u32],
        dst_idx: &[u32],
        n: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= src_idx.len() by the launch contract.
        let s = unsafe { *src_idx.get_unchecked(i) } as usize;
        // SAFETY: i < n <= dst_idx.len() by the launch contract.
        let d = unsafe { *dst_idx.get_unchecked(i) } as usize;
        if s < x.len() && d < y.len() {
            // SAFETY: both indices guarded against their buffers' lengths.
            unsafe {
                *y.get_unchecked_mut(d) = *x.get_unchecked(s);
            }
        }
    }

    /// Q8_0 gemv over per-head activation slices: launch row `r` belongs to
    /// head `h = r / rows_per_head` and output slot `r % rows_per_head` of
    /// that head; it dots weight row `r` of the Q8_0 planes against
    /// `x[h*x_head_stride .. +k]` (the gated single-column row body, the
    /// `q8_0_gemv` skeleton with one warp per row, 8 rows per 256-thread
    /// block), lane 0 storing `y[h*y_head_stride + y_off + r %
    /// rows_per_head]`. `n_rows` must be `n_heads * rows_per_head`; every
    /// store is disjoint because each (head, slot) pair belongs to exactly
    /// one row.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= (n_heads - 1) * x_head_stride + k,
            y.len() >= (n_heads - 1) * y_head_stride + y_off + rows_per_head
        )
    )]
    pub fn q8_0_gemv_heads(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        n_rows: u32,
        k: u32,
        n_heads: u32,
        rows_per_head: u32,
        x_head_stride: u32,
        y_head_stride: u32,
        y_off: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let h = row / rows_per_head as usize;
        // The launch contract cannot bind n_rows to n_heads*rows_per_head
        // (no division in its grammar); this guard keeps a violated
        // divisibility from writing outside the last head's y span.
        if h >= n_heads as usize {
            return;
        }
        let j = row % rows_per_head as usize;
        let lane = warp::lane_id() as usize;
        // SAFETY: the row body's contract: row < n_rows puts the row's words
        // and scales inside qs and d (the contract's 4·qs.len() and 32·d.len()
        // bounds), and h < n_heads bounds its x window, x0 = h*x_head_stride
        // <= (n_heads-1)*x_head_stride with x.len() >= x0 + k.
        let f =
            unsafe { q8_0_lane_partial_1col(qs, d, x, k, row, h * x_head_stride as usize, lane) };
        let s0 = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes; the slot
            // h*y_head_stride + y_off + j is inside y by the launch contract
            // and belongs to this row alone.
            unsafe {
                *y.get_unchecked_mut(h * y_head_stride as usize + y_off as usize + j) = s0;
            }
        }
    }

    /// Q3_K gemv over per-head activation columns: launch row `r` belongs
    /// to head `h = head_base + r / rows_per_head`, dots absolute weight
    /// row `h * row_stride_per_head + row_off + r % rows_per_head` against
    /// activation column `r / rows_per_head` of `q`/`d8` (m = 1 through
    /// the gated row body, one warp per row, 8 rows per 256-thread block),
    /// lane 0 storing `y[h * y_head_stride + r % rows_per_head]`.
    /// `n_rows` must be `heads * rows_per_head` and
    /// `row_off + rows_per_head <= row_stride_per_head` (host-validated),
    /// which keeps every `row_abs` below
    /// `(head_base + heads) * row_stride_per_head` — the bound the launch
    /// contract puts on `w`. Stores are disjoint: each (head, slot) pair
    /// belongs to exactly one row.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= (head_base + heads) * row_stride_per_head * 110 * n_sb,
            q.len() >= heads * 64 * iters,
            d8.len() >= heads * 2 * n_sb,
            y.len() >= (head_base + heads - 1) * y_head_stride + rows_per_head
        )
    )]
    pub fn q3k_gemv_heads(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        head_base: u32,
        heads: u32,
        rows_per_head: u32,
        row_stride_per_head: u32,
        row_off: u32,
        y_head_stride: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let hi = row / rows_per_head as usize;
        // The launch contract cannot bind n_rows to heads*rows_per_head
        // (no division in its grammar); this guard keeps a violated
        // divisibility from reading weight rows past the last head's
        // block.
        if hi >= heads as usize {
            return;
        }
        let row_in = row % rows_per_head as usize;
        let h = head_base as usize + hi;
        let row_abs = h * row_stride_per_head as usize + row_off as usize + row_in;
        let lane = warp::lane_id() as usize;
        // SAFETY: row_abs + 1 <= (head_base + heads) * row_stride_per_head
        // (the contract's bound on w, given the host-validated
        // row_off + rows_per_head <= row_stride_per_head); column hi <
        // heads keeps q/d8 inside their contract bounds.
        let f = q3k_row_dot(w, q, d8, n_sb as usize, iters, row_abs, hi, 1, lane);
        let s0 = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes; the slot
            // h*y_head_stride + row_in is inside y by the launch contract
            // and belongs to this row alone.
            unsafe {
                *y.get_unchecked_mut(h * y_head_stride as usize + row_in) = s0;
            }
        }
    }

    /// Both head halves of [`q3k_gemv_heads`] in ONE launch: the same body
    /// over the same rows, taking the activation from the half the row's
    /// head belongs to — heads below `split` read `(q_lo, d8_lo)` at column
    /// `hi`, the rest `(q_hi, d8_hi)` at column `hi - split`. The two halves
    /// read DIFFERENT weight rows, so this saves a launch, not a weight
    /// read.
    ///
    /// The branch is warp-uniform: a warp owns one row, hence one head, so
    /// all 32 lanes take the same side and the warp reduction inside
    /// `q3k_row_dot` still sees a full warp. Every row's loads, accumulation
    /// order and store are the ones the two-launch form runs, so the outputs
    /// agree bit for bit.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= (head_base + heads) * row_stride_per_head * 110 * n_sb,
            q_lo.len() >= split * 64 * iters,
            d8_lo.len() >= split * 2 * n_sb,
            q_hi.len() >= (heads - split) * 64 * iters,
            d8_hi.len() >= (heads - split) * 2 * n_sb,
            y.len() >= (head_base + heads - 1) * y_head_stride + rows_per_head
        )
    )]
    pub fn q3k_gemv_heads_pair(
        w: &[u32],
        q_lo: &[u64],
        d8_lo: &[f32],
        q_hi: &[u64],
        d8_hi: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        head_base: u32,
        heads: u32,
        split: u32,
        rows_per_head: u32,
        row_stride_per_head: u32,
        row_off: u32,
        y_head_stride: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let hi = row / rows_per_head as usize;
        // As `q3k_gemv_heads`: the launch contract cannot bind n_rows to
        // heads*rows_per_head, so a violated divisibility stops here rather
        // than read weight rows past the last head's block.
        if hi >= heads as usize {
            return;
        }
        let row_in = row % rows_per_head as usize;
        let h = head_base as usize + hi;
        let row_abs = h * row_stride_per_head as usize + row_off as usize + row_in;
        let lane = warp::lane_id() as usize;
        // SAFETY (both arms): row_abs + 1 <= (head_base + heads) *
        // row_stride_per_head (the contract's bound on w, given the
        // host-validated row_off + rows_per_head <= row_stride_per_head);
        // the column is below `split` on the lo side and below
        // `heads - split` on the hi side, which are the contract's bounds on
        // the two activation pairs.
        let s0 = if hi < split as usize {
            let f = q3k_row_dot(w, q_lo, d8_lo, n_sb as usize, iters, row_abs, hi, 1, lane);
            warp::reduce_sum_f32(f[0])
        } else {
            let col = hi - split as usize;
            let f = q3k_row_dot(w, q_hi, d8_hi, n_sb as usize, iters, row_abs, col, 1, lane);
            warp::reduce_sum_f32(f[0])
        };
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes; the slot
            // h*y_head_stride + row_in is inside y by the launch contract
            // and belongs to this row alone.
            unsafe {
                *y.get_unchecked_mut(h * y_head_stride as usize + row_in) = s0;
            }
        }
    }
}

/// The loaded step module of this file. Owns no context and no stream —
/// every enqueue takes the engine stream, so it orders with the rest of the
/// step and is capturable.
pub struct StepKernels {
    module: step_kernels::LoadedModule,
}

/// [`StepKernels::enqueue_q8_0_gemv_heads`]'s arguments. The strides and
/// the offset count f32 elements, `rows_per_head` weight rows.
pub struct Q8_0GemvHeadsArgs<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub x: &'a DeviceBuffer<f32>,
    pub rows_per_head: usize,
    pub x_head_stride: usize,
    pub y_head_stride: usize,
    pub y_off: usize,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The per-head Q3_K launch's head layout, shared by both of its entries.
/// `head_base` counts heads, `rows_per_head`/`row_stride_per_head`/`row_off`
/// weight rows, and `y_head_stride` f32 elements.
#[derive(Clone, Copy)]
pub(crate) struct HeadsGeom {
    pub head_base: usize,
    pub rows_per_head: usize,
    pub row_stride_per_head: usize,
    pub row_off: usize,
    pub y_head_stride: usize,
}

/// [`StepKernels::enqueue_q3k_gemv_heads`]'s arguments.
pub(crate) struct Q3kGemvHeadsArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub geom: HeadsGeom,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`StepKernels::enqueue_q3k_gemv_heads_pair`]'s arguments: as
/// [`Q3kGemvHeadsArgs`], the activation columns split over `lo` and `hi`.
pub(crate) struct Q3kGemvHeadsPairArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub lo: &'a Q8Act,
    pub hi: &'a Q8Act,
    pub geom: HeadsGeom,
    pub y: &'a mut DeviceBuffer<f32>,
}

impl StepKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<StepKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { step_kernels::load(ctx)? };
        Ok(StepKernels { module })
    }

    /// Enqueue the pair-table gather of `n` pairs. Asynchronous,
    /// allocation-free, capturable. The tables must be built for the shapes
    /// of `x` and `y` this call pairs (load-time construction owns that).
    pub fn enqueue_gather(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        src_idx: &DeviceBuffer<u32>,
        dst_idx: &DeviceBuffer<u32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || src_idx.len() < n || dst_idx.len() < n || x.is_empty() || y.is_empty() {
            return Err(GpuError::shape(
                "enqueue_gather",
                format!(
                    "n={n}, src_idx.len() {}, dst_idx.len() {}, x.len() {}, \
                 y.len() {}",
                    src_idx.len(),
                    dst_idx.len(),
                    x.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32("enqueue_gather", "n", n)?;
        let prep =
            self.module
                .prepare_gather_pairs(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module
            .gather_pairs(stream, &prep, x, src_idx, dst_idx, n, y)?;
        Ok(())
    }

    /// Enqueue the per-head Q8_0 gemv: `d.rows()` weight rows (a file
    /// tensor's or a derived weight's planes, `qs`/`d` as
    /// `enqueue_q8_0_gemv` takes them, k = `d.cols() * 32`), each dotted
    /// against its head's slice of `x` —
    /// head `h` reads `x[h*x_head_stride .. +k]`, m = 1 — with lane 0
    /// writing `y[h*y_head_stride + y_off + j]` for weight row
    /// `h*rows_per_head + j`. `d.rows()` must be a multiple of
    /// `rows_per_head` (then `n_heads = d.rows() / rows_per_head`), and
    /// `x_head_stride` a multiple of 4, so every head's slice starts 16-byte
    /// aligned in `x` and the row body reads it in quads. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q8_0_gemv_heads(
        &self,
        stream: &CudaStream,
        a: Q8_0GemvHeadsArgs<'_>,
    ) -> Result<(), GpuError> {
        let Q8_0GemvHeadsArgs {
            qs,
            d,
            x,
            rows_per_head,
            x_head_stride,
            y_head_stride,
            y_off,
            y,
        } = a;
        let n_rows = d.rows();
        let k = d.cols() * 32;
        if k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv_heads",
                format!(
                    "need k a positive multiple of 32, got k={k} \
                 (d is {}x{})",
                    d.rows(),
                    d.cols()
                ),
            ));
        }
        if qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv_heads",
                format!(
                    "qs is {}x{}, want {}x{} (k/4 words per row, k = \
                 d.cols()*32 = {k})",
                    qs.rows(),
                    qs.cols(),
                    n_rows,
                    d.cols() * 8
                ),
            ));
        }
        if rows_per_head == 0 || !n_rows.is_multiple_of(rows_per_head) {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv_heads",
                format!(
                    "n_rows={n_rows} is not a positive multiple of \
                 rows_per_head={rows_per_head}"
                ),
            ));
        }
        // A stride off the quad grid is not an error the kernel can see: the
        // row body would take its scalar walk wherever a head's slice starts
        // off the grid — the same sums at a quarter of the load width.
        if !x_head_stride.is_multiple_of(4) {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv_heads",
                format!(
                    "x_head_stride={x_head_stride} is not a multiple of 4: a head's slice \
                 would not start 16-byte aligned"
                ),
            ));
        }
        let n_heads = n_rows / rows_per_head;
        if x.len() < (n_heads - 1) * x_head_stride + k {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv_heads",
                format!(
                    "x.len() {} < (n_heads-1)*x_head_stride + k = \
                 {}*{x_head_stride} + {k}",
                    x.len(),
                    n_heads - 1
                ),
            ));
        }
        if y.len() < (n_heads - 1) * y_head_stride + y_off + rows_per_head {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv_heads",
                format!(
                    "y.len() {} < (n_heads-1)*y_head_stride + y_off + \
                 rows_per_head = {}*{y_head_stride} + {y_off} + {rows_per_head}",
                    y.len(),
                    n_heads - 1
                ),
            ));
        }
        let what = "enqueue_q8_0_gemv_heads";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let k = launch_u32(what, "k", k)?;
        let n_heads = launch_u32(what, "n_heads", n_heads)?;
        let rows_per_head = launch_u32(what, "rows_per_head", rows_per_head)?;
        let x_head_stride = launch_u32(what, "x_head_stride", x_head_stride)?;
        let y_head_stride = launch_u32(what, "y_head_stride", y_head_stride)?;
        let y_off = launch_u32(what, "y_off", y_off)?;
        let prep =
            self.module
                .prepare_q8_0_gemv_heads(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.q8_0_gemv_heads(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            n_rows,
            k,
            n_heads,
            rows_per_head,
            x_head_stride,
            y_head_stride,
            y_off,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the per-head Q3_K gemv: `heads` heads starting at
    /// `head_base` (of the `w.rows()`-row weight, `row_stride_per_head`
    /// rows per head), each head's `rows_per_head` output rows dotting
    /// absolute weight row `h*row_stride_per_head + row_off + j` against
    /// activation column `h - head_base` of `act` (m = 1), lane 0 writing
    /// `y[h*y_head_stride + j]`. `heads` must be `act.m()` (one column per
    /// head), `w.cols()` `110 * n_sb / 4` words with even `n_sb` (as
    /// `enqueue_gemv_q3k`), and `row_off + rows_per_head <=
    /// row_stride_per_head` so a head's rows stay on its own block.
    /// Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_q3k_gemv_heads(
        &self,
        stream: &CudaStream,
        a: Q3kGemvHeadsArgs<'_>,
    ) -> Result<(), GpuError> {
        let Q3kGemvHeadsArgs { w, act, geom, y } = a;
        let HeadsGeom {
            head_base,
            rows_per_head,
            row_stride_per_head,
            row_off,
            y_head_stride,
        } = geom;
        let n_sb = act.n_sb();
        let heads = act.m();
        check_q3k_heads(
            "enqueue_q3k_gemv_heads",
            w,
            (n_sb, act.k()),
            heads,
            geom,
            y.len(),
        )?;
        let n_rows = heads * rows_per_head;
        let what = "enqueue_q3k_gemv_heads";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let head_base = launch_u32(what, "head_base", head_base)?;
        let heads = launch_u32(what, "heads", heads)?;
        let rows_per_head = launch_u32(what, "rows_per_head", rows_per_head)?;
        let row_stride_per_head = launch_u32(what, "row_stride_per_head", row_stride_per_head)?;
        let row_off = launch_u32(what, "row_off", row_off)?;
        let y_head_stride = launch_u32(what, "y_head_stride", y_head_stride)?;
        let prep =
            self.module
                .prepare_q3k_gemv_heads(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.q3k_gemv_heads(
            stream,
            &prep,
            w.buf(),
            &act.q3,
            &act.d8,
            n_rows,
            n_sb,
            n_sb.div_ceil(2),
            head_base,
            heads,
            rows_per_head,
            row_stride_per_head,
            row_off,
            y_head_stride,
            y,
        )?;
        Ok(())
    }

    /// Enqueue [`StepKernels::enqueue_q3k_gemv_heads`] for BOTH head halves
    /// in one launch: heads `head_base .. head_base + lo.m()` take their
    /// activation column from `lo`, the `hi.m()` heads after them from `hi`.
    /// Same weight, same shapes and the same per-head contract as the single
    /// call — the two together replace the pair of launches, bit for bit.
    /// Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_q3k_gemv_heads_pair(
        &self,
        stream: &CudaStream,
        a: Q3kGemvHeadsPairArgs<'_>,
    ) -> Result<(), GpuError> {
        let Q3kGemvHeadsPairArgs { w, lo, hi, geom, y } = a;
        let HeadsGeom {
            head_base,
            rows_per_head,
            row_stride_per_head,
            row_off,
            y_head_stride,
        } = geom;
        let n_sb = lo.n_sb();
        let (split, heads) = (lo.m(), lo.m() + hi.m());
        if hi.n_sb() != n_sb || hi.k() != lo.k() {
            return Err(GpuError::shape(
                "enqueue_q3k_gemv_heads_pair",
                format!(
                    "both halves must share K, got lo k={} hi k={}",
                    lo.k(),
                    hi.k()
                ),
            ));
        }
        check_q3k_heads(
            "enqueue_q3k_gemv_heads_pair",
            w,
            (n_sb, lo.k()),
            heads,
            geom,
            y.len(),
        )?;
        let n_rows = heads * rows_per_head;
        let what = "enqueue_q3k_gemv_heads_pair";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let head_base = launch_u32(what, "head_base", head_base)?;
        let heads = launch_u32(what, "heads", heads)?;
        let split = launch_u32(what, "lo.m()", split)?;
        let rows_per_head = launch_u32(what, "rows_per_head", rows_per_head)?;
        let row_stride_per_head = launch_u32(what, "row_stride_per_head", row_stride_per_head)?;
        let row_off = launch_u32(what, "row_off", row_off)?;
        let y_head_stride = launch_u32(what, "y_head_stride", y_head_stride)?;
        let prep = self
            .module
            .prepare_q3k_gemv_heads_pair(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.q3k_gemv_heads_pair(
            stream,
            &prep,
            w.buf(),
            &lo.q3,
            &lo.d8,
            &hi.q3,
            &hi.d8,
            n_rows,
            n_sb,
            n_sb.div_ceil(2),
            head_base,
            heads,
            split,
            rows_per_head,
            row_stride_per_head,
            row_off,
            y_head_stride,
            y,
        )?;
        Ok(())
    }
}

/// The shape checks both per-head Q3_K launches share: even `n_sb`, `w`'s
/// row width, a head's rows inside its own block, the heads inside `w`, and
/// `y` covering the last head's rows. `(n_sb, k)` is the activation's
/// super-block count at `K = k`; `heads` heads start at `geom.head_base`.
fn check_q3k_heads(
    what: &'static str,
    w: &DeviceTensor<u32>,
    (n_sb, k): (usize, usize),
    heads: usize,
    geom: HeadsGeom,
    y_len: usize,
) -> Result<(), GpuError> {
    let HeadsGeom {
        head_base,
        rows_per_head,
        row_stride_per_head,
        row_off,
        y_head_stride,
    } = geom;
    if !n_sb.is_multiple_of(2) {
        return Err(GpuError::shape(
            what,
            format!(
                "odd super-block count {n_sb} (K={k}) leaves rows unaligned; repack rows at \
                 load time"
            ),
        ));
    }
    if w.cols() != 110 * n_sb / 4 {
        return Err(GpuError::shape(
            what,
            format!(
                "Q3_K rows are 110*{n_sb}/4 = {} words at K={k}, got {}",
                110 * n_sb / 4,
                w.cols()
            ),
        ));
    }
    if row_off + rows_per_head > row_stride_per_head || rows_per_head == 0 {
        return Err(GpuError::shape(
            what,
            format!(
                "row_off {row_off} + rows_per_head {rows_per_head} must lie inside \
                 row_stride_per_head {row_stride_per_head}"
            ),
        ));
    }
    if (head_base + heads) * row_stride_per_head > w.rows() {
        return Err(GpuError::shape(
            what,
            format!(
                "heads {head_base}..{} need (head_base+heads)*row_stride_per_head = {} rows, \
                 w has {}",
                head_base + heads,
                (head_base + heads) * row_stride_per_head,
                w.rows()
            ),
        ));
    }
    if y_len < (head_base + heads - 1) * y_head_stride + rows_per_head {
        return Err(GpuError::shape(
            what,
            format!(
                "y.len() {y_len} < (head_base+heads-1)*y_head_stride + rows_per_head = \
                 {}*{y_head_stride} + {rows_per_head}",
                head_base + heads - 1
            ),
        ));
    }
    Ok(())
}
