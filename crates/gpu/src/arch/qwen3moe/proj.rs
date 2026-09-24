//! A qwen3moe layer's attention projections in two launches beside the
//! flash: the query, key and value rows in one (`qwen3moe_qkv_q4k`; a Q6_K
//! value projection keeps its own `q6k_gemv` launch), and the output
//! projection with the residual add in its store (`qwen3moe_o_resid_q4k`;
//! at one token `qwen3moe_o_resid_quant_q4k`, which also quantizes its f32
//! input per block, so the q8_1 launch before it goes).
//! `q6k_gemv` writes `m > 1` columns row-major; `qwen3moe_token_major`
//! copies them into the token-major layout the rest of the chain reads.
//!
//! Every row and column is `q4k_gemv`'s one-column body:
//! `cores::q4k_row_dot_1col`, the fixed warp tree and the lane-0 store, run
//! once per activation column. So each output is bit for bit the value the
//! per-matrix `q4k_gemv` launch writes at one column, and the residual
//! output is that value plus the residual — one add, `elem::add`'s. The
//! multi-column walk `cores::q4k_row_dot` is not used: at K = 4096 its
//! column sums differ from the one-column body's in the low bits, so a
//! prompt pass through it would not leave the rows a decode step leaves.
//! Outputs are token-major: column `c` of an `m`-column launch lands at
//! `c · rows + r`, which at `m = 1` is the one layout of every consumer.

use crate::GpuError;
use crate::cores::q4k_row_dot_1col;
use crate::fault::{FaultSink, FaultSite};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{launch_u32, q8_1_quant_block};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The widest attention row `qwen3moe_o_resid_quant_q4k` quantizes into
/// shared memory: 32 query heads of 128.
pub const ATTN_K: usize = 4096;
/// Super-blocks of an [`ATTN_K`] row, and the q8_1 planes' lengths one
/// column of it takes (`Q8Act`'s layout: `64·half_it` u64, `256·quad_it` and
/// `128·half_it` u32, `8·n_sb` group sums, `2·n_sb` scales).
const ATTN_SB: usize = ATTN_K / 256;
const ATTN_Q3: usize = 64 * ATTN_SB.div_ceil(2);
const ATTN_Q4: usize = 256 * ATTN_SB.div_ceil(4);
const ATTN_Q6: usize = 128 * ATTN_SB.div_ceil(2);
const ATTN_S8: usize = 8 * ATTN_SB;
const ATTN_D8: usize = 2 * ATTN_SB;
// The launch contract spells the bound out.
const _: () = assert!(ATTN_K == 4096);

#[cuda_module]
mod qwen3moe_proj_kernels {
    use super::*;

    /// The query, key and value projections of `m_cols` activation columns
    /// in one launch (module doc): thread row `row = block·8 + warp` of
    /// `rows_q + rows_k + rows_v` is row `row` of `wq`, then of `wk`, then
    /// of `wv`, stored token-major into the blocks of `y` at 0, `off_k` and
    /// `off_v`. `rows_v` 0 leaves `wv` and the value block untouched (a Q6_K
    /// value projection).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            wq.len() >= rows_q * 36 * n_sb,
            wk.len() >= rows_k * 36 * n_sb,
            wv.len() >= rows_v * 36 * n_sb,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            off_k >= rows_q * m_cols,
            off_v >= off_k + rows_k * m_cols,
            y.len() >= off_v + rows_v * m_cols
        )
    )]
    pub fn qwen3moe_qkv_q4k(
        wq: &[u32],
        wk: &[u32],
        wv: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        rows_q: u32,
        rows_k: u32,
        rows_v: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        off_k: u32,
        off_v: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let (rq, rk, rv) = (rows_q as usize, rows_k as usize, rows_v as usize);
        if row >= rq + rk + rv {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        // The row's matrix and its output block, warp-uniform: the warp's
        // 32 lanes share `row`.
        let (w, base, r, rows) = if row < rq {
            (wq, 0, row, rq)
        } else if row < rq + rk {
            (wk, off_k as usize, row - rq, rk)
        } else {
            (wv, off_v as usize, row - rq - rk, rv)
        };
        // The core's caller contract, from the launch contract: r < rows
        // rows of `w`, m_cols columns of q/s8/d8, iters = ceil(n_sb/4) from
        // the host, and all 32 lanes of the warp are here (the return above
        // is warp-uniform). One column takes the single-column body
        // directly, as `q4k_gemv` does.
        let mut c = 0;
        while c < m {
            let v = warp::reduce_sum_f32(q4k_row_dot_1col(w, q, s8, d8, n_sb, iters, r, c, lane));
            if lane == 0 {
                // SAFETY: c < m_cols and r < rows, so base + c·rows + r <
                // base + rows·m_cols <= y.len() by the launch contract's
                // block bounds; lane 0 of the row's warp is the only writer
                // of the row's slots.
                unsafe { *y.get_unchecked_mut(base + c * rows + r) = v };
            }
            c += 1;
        }
    }

    /// The output projection of `m_cols` activation columns with the
    /// residual: `y[c · rows + r] = (w · act_c)[r] + x[c · rows + r]`, the
    /// dot as `q4k_gemv` computes it and one add (module doc).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= rows * 36 * n_sb,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            x.len() >= rows * m_cols,
            y.len() >= rows * m_cols
        )
    )]
    pub fn qwen3moe_o_resid_q4k(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        x: &[f32],
        rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let rows = rows as usize;
        if row >= rows {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        // The core's caller contract as in `qwen3moe_qkv_q4k`.
        let mut c = 0;
        while c < m {
            let v = warp::reduce_sum_f32(q4k_row_dot_1col(w, q, s8, d8, n_sb, iters, row, c, lane));
            if lane == 0 {
                let i = c * rows + row;
                // SAFETY: c < m_cols and row < rows, so i < rows·m_cols <=
                // x.len(), y.len() by the launch contract; lane 0 of the
                // row's warp is the only writer of the row's slots.
                unsafe { *y.get_unchecked_mut(i) = v + *x.get_unchecked(i) };
            }
            c += 1;
        }
    }

    /// The output projection of one token with the q8_1 of its input folded
    /// in: `y[r] = (w · q8_1(attn))[r] + x[r]`. Every block first quantizes
    /// the whole `attn` row (`256 · n_sb <= ATTN_K` values) into its own
    /// shared copy of the q8_1 planes with `q8_1_quant_block` —
    /// `q3k_quantize_q8_1`'s body, the non-finite raise
    /// ([`FaultSite::QuantColumn`]) with it — warp `w` taking the 128-value
    /// blocks `w, w + 8, …` (which warp takes a block moves no bit: the
    /// block's work is warp-local). Then each warp's row is
    /// `qwen3moe_o_resid_q4k`'s at one column, reading those planes: the
    /// same bytes the quantizer launch writes, so the same dot.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_sb <= 16,
            w.len() >= rows * 36 * n_sb,
            attn.len() >= 256 * n_sb,
            x.len() >= rows,
            y.len() >= rows
        )
    )]
    pub fn qwen3moe_o_resid_quant_q4k(
        w: &[u32],
        attn: &[f32],
        x: &[f32],
        rows: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut y: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut SQ3: SharedArray<u64, ATTN_Q3> = SharedArray::UNINIT;
        static mut SQ4: SharedArray<u32, ATTN_Q4> = SharedArray::UNINIT;
        static mut SQ6: SharedArray<u32, ATTN_Q6> = SharedArray::UNINIT;
        static mut SS8: SharedArray<i32, ATTN_S8> = SharedArray::UNINIT;
        static mut SD8: SharedArray<f32, ATTN_D8> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let wi = tid / 32;
        let lane = warp::lane_id() as usize;
        let n_sb = n_sb as usize;
        // SAFETY: the five arrays are this block's own shared allocations;
        // the raw form reaches each `static mut` without a reference.
        let (p3, p4, p6, ps8, pd8) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut SQ3),
                SharedArray::as_raw_mut_ptr(&raw mut SQ4),
                SharedArray::as_raw_mut_ptr(&raw mut SQ6),
                SharedArray::as_raw_mut_ptr(&raw mut SS8),
                SharedArray::as_raw_mut_ptr(&raw mut SD8),
            )
        };
        {
            // SAFETY: each slice spans its array's whole allocation, and
            // these are the only views of them until the barrier below; the
            // quantizer's stores are disjoint across the block's lanes (its
            // contract), as a kernel output's are across a grid.
            let (mut q3, mut q4, mut q6, mut s8, mut d8) = unsafe {
                (
                    DisjointSlice::from_raw_parts(p3, ATTN_Q3),
                    DisjointSlice::from_raw_parts(p4, ATTN_Q4),
                    DisjointSlice::from_raw_parts(p6, ATTN_Q6),
                    DisjointSlice::from_raw_parts(ps8, ATTN_S8),
                    DisjointSlice::from_raw_parts(pd8, ATTN_D8),
                )
            };
            let mut b = wi;
            while b < 2 * n_sb {
                // SAFETY: one column (col 0), b < 2·n_sb, attn holds 256·n_sb
                // values (launch contract), the planes hold one column of an
                // n_sb <= 16 row (the arrays' lengths at ATTN_SB = 16, with the
                // host's half_it = ceil(n_sb/2), quad_it = ceil(n_sb/4)), and
                // the whole warp is here with the same b.
                q8_1_quant_block(
                    attn,
                    0,
                    0,
                    b,
                    n_sb,
                    half_it,
                    quad_it,
                    lane,
                    &mut q3,
                    &mut q4,
                    &mut q6,
                    &mut s8,
                    &mut d8,
                    fault,
                    FaultSite::QuantColumn,
                );
                b += 8;
            }
        }
        thread::sync_threads();

        let row = thread::blockIdx_x() as usize * 8 + wi;
        if row >= rows as usize {
            return;
        }
        // SAFETY: the planes were written before the barrier above and
        // nothing writes them from here on; each slice spans its array.
        let (q, s8, d8) = unsafe {
            (
                core::slice::from_raw_parts(p4.cast_const(), ATTN_Q4),
                core::slice::from_raw_parts(ps8.cast_const(), ATTN_S8),
                core::slice::from_raw_parts(pd8.cast_const(), ATTN_D8),
            )
        };
        // The core's caller contract: row < rows rows of `w`, one column of
        // q/s8/d8 (q holds 256·quad_it words with quad_it = iters), and all
        // 32 lanes of the warp are here (the return above is warp-uniform).
        let v = warp::reduce_sum_f32(q4k_row_dot_1col(w, q, s8, d8, n_sb, quad_it, row, 0, lane));
        if lane == 0 {
            // SAFETY: row < rows <= x.len(), y.len() by the launch contract;
            // lane 0 of the row's warp is the slot's only writer.
            unsafe { *y.get_unchecked_mut(row) = v + *x.get_unchecked(row) };
        }
    }

    /// A row-major `m`-column output to token-major: `y[c · rows + r] =
    /// x[r · m + c]`, one thread per value — a copy, so exact.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= rows * m, y.len() >= rows * m)
    )]
    pub fn qwen3moe_token_major(x: &[f32], rows: u32, m: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        let (rows, m) = (rows as usize, m as usize);
        if i >= rows * m {
            return;
        }
        let c = i / rows;
        // SAFETY: i < rows·m, so c < m and i − c·rows < rows: both indices are
        // below rows·m, inside x and y by the launch contract; thread i is
        // y[i]'s only writer.
        unsafe { *y.get_unchecked_mut(i) = *x.get_unchecked((i - c * rows) * m + c) };
    }
}

/// [`ProjKernels::enqueue_qkv`]'s arguments: the Q4_K query and key
/// projections, the value projection when it is Q4_K too (`None` when it is
/// Q6_K and runs its own gemv), the quantized input (`act.m()` columns), and
/// the one output allocation holding the query block at element 0, the key
/// block at `off_k` and the value block at `off_v`, each token-major.
pub struct QkvArgs<'a> {
    pub wq: &'a DeviceTensor<u32>,
    pub wk: &'a DeviceTensor<u32>,
    pub wv: Option<&'a DeviceTensor<u32>>,
    pub act: &'a Q8Act,
    pub y: &'a mut DeviceBuffer<f32>,
    pub off_k: usize,
    pub off_v: usize,
}

/// [`ProjKernels::enqueue_o_resid`]'s arguments: the Q4_K output projection,
/// its quantized input (`act.m()` columns), the residual and the output,
/// both token-major `act.m() · w.rows()`.
pub struct OResidArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub x: &'a DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`ProjKernels::enqueue_o_resid_quant`]'s arguments: the Q4_K output
/// projection, its one token's f32 input `attn` (`256 · n_sb` values, at
/// most [`ATTN_K`]), the residual and the output (`w.rows()` values each).
pub struct OResidQuantArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub attn: &'a DeviceBuffer<f32>,
    pub x: &'a DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ProjKernels {
    module: qwen3moe_proj_kernels::LoadedModule,
}

/// Err unless `w` holds Q4_K rows of `act`'s K (`36 · n_sb` words) and the
/// output room `y_len` holds `rows · m` values.
fn check_q4k(
    what: &'static str,
    name: &str,
    w: &DeviceTensor<u32>,
    act: &Q8Act,
    y_len: usize,
) -> Result<(), GpuError> {
    let n_sb = act.n_sb();
    if w.cols() != 36 * n_sb {
        return Err(GpuError::shape(
            what,
            format!(
                "{name}: Q4_K rows are 36*{n_sb} words at K={}, got {}x{}",
                act.k(),
                w.rows(),
                w.cols()
            ),
        ));
    }
    if y_len < w.rows() * act.m() {
        return Err(GpuError::shape(
            what,
            format!(
                "{name}: {} values out < rows*m = {}*{}",
                y_len,
                w.rows(),
                act.m()
            ),
        ));
    }
    Ok(())
}

impl ProjKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ProjKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { qwen3moe_proj_kernels::load(ctx)? };
        Ok(ProjKernels { module })
    }

    /// Enqueue the query, key and (a Q4_K) value projections of `a.act`'s
    /// columns in one launch (module doc). Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_qkv(&self, stream: &CudaStream, a: QkvArgs<'_>) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_qkv";
        let QkvArgs {
            wq,
            wk,
            wv,
            act,
            y,
            off_k,
            off_v,
        } = a;
        let room = |at: usize, end: usize| end.saturating_sub(at);
        check_q4k(what, "q", wq, act, off_k)?;
        check_q4k(what, "k", wk, act, room(off_k, off_v))?;
        let rows_v = match wv {
            Some(v) => {
                check_q4k(what, "v", v, act, room(off_v, y.len()))?;
                v.rows()
            }
            None => 0,
        };
        if off_v < off_k || off_v > y.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "blocks at 0, {off_k}, {off_v} of a {}-value output",
                    y.len()
                ),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", (wq.rows() + wk.rows() + rows_v).div_ceil(8))?;
        let rows_q = launch_u32(what, "rows_q", wq.rows())?;
        let rows_k = launch_u32(what, "rows_k", wk.rows())?;
        let rows_v = launch_u32(what, "rows_v", rows_v)?;
        let m = launch_u32(what, "m", act.m())?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let off_k = launch_u32(what, "off_k", off_k)?;
        let off_v = launch_u32(what, "off_v", off_v)?;
        let prep = self
            .module
            .prepare_qwen3moe_qkv_q4k(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.qwen3moe_qkv_q4k(
            stream,
            &prep,
            wq.buf(),
            wk.buf(),
            wv.unwrap_or(wk).buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            rows_q,
            rows_k,
            rows_v,
            m,
            n_sb,
            n_sb.div_ceil(4),
            off_k,
            off_v,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the copy of `x`'s `m` row-major columns of `rows` values into
    /// `y` token-major (module doc). Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_token_major(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        rows: usize,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_token_major";
        if rows == 0 || m == 0 || x.len() < rows * m || y.len() < rows * m {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} columns of {rows}: x.len() {}, y.len() {}",
                    x.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (rows * m).div_ceil(256))?;
        let rows = launch_u32(what, "rows", rows)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_qwen3moe_token_major(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .qwen3moe_token_major(stream, &prep, x, rows, m, y)?;
        Ok(())
    }

    /// Enqueue `y = w · act + x` over `a.act`'s columns (module doc).
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_o_resid(&self, stream: &CudaStream, a: OResidArgs<'_>) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_o_resid";
        let OResidArgs { w, act, x, y } = a;
        check_q4k(what, "attn_output", w, act, y.len())?;
        if x.len() < w.rows() * act.m() {
            return Err(GpuError::shape(
                what,
                format!(
                    "residual: {} values < rows*m = {}*{}",
                    x.len(),
                    w.rows(),
                    act.m()
                ),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", w.rows().div_ceil(8))?;
        let rows = launch_u32(what, "rows", w.rows())?;
        let m = launch_u32(what, "m", act.m())?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_qwen3moe_o_resid_q4k(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.qwen3moe_o_resid_q4k(
            stream,
            &prep,
            w.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            x,
            rows,
            m,
            n_sb,
            n_sb.div_ceil(4),
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = w · q8_1(attn) + x` for one token in one launch
    /// (`qwen3moe_o_resid_quant_q4k`): the bytes `y` holds after
    /// `Gpu::enqueue_quantize_q8_1` on `attn` and then
    /// [`ProjKernels::enqueue_o_resid`], with `fault` taking the quantizer's
    /// raise. Asynchronous, allocation-free, capturable.
    pub fn enqueue_o_resid_quant(
        &self,
        stream: &CudaStream,
        a: OResidQuantArgs<'_>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_o_resid_quant";
        let OResidQuantArgs { w, attn, x, y } = a;
        let n_sb = w.cols() / 36;
        let rows = w.rows();
        if n_sb == 0 || w.cols() != 36 * n_sb || 256 * n_sb > ATTN_K {
            return Err(GpuError::shape(
                what,
                format!(
                    "attn_output: Q4_K rows of {} words, want 36 per super-block and at most {ATTN_K} values",
                    w.cols()
                ),
            ));
        }
        if attn.len() < 256 * n_sb || x.len() < rows || y.len() < rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "attn.len() {} for K={}, residual {} and output {} for {rows} rows",
                    attn.len(),
                    256 * n_sb,
                    x.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", rows.div_ceil(8))?;
        let rows = launch_u32(what, "rows", rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_qwen3moe_o_resid_quant_q4k(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.qwen3moe_o_resid_quant_q4k(
            stream,
            &prep,
            w.buf(),
            attn,
            x,
            rows,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            y,
            fault,
        )?;
        Ok(())
    }
}
