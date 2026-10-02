//! Q8_0 gemv over expert slots selected on the device, 32 values a block:
//! the MoE down projection of a Q8_0 expert stack resident as the file's
//! bytes, all `n_slots` routed experts in one launch — `q5_1_gemv_sel`'s
//! shape (`crate::q5_1_sel`) with Q8_0's decoder: slot `s` dots the rows of
//! expert `sel[s]` with activation column `s`. It takes any K a multiple of
//! 32, where the K-quant family's `q8_0_gemv_sel` takes whole 256-value
//! super-blocks.
//!
//! Weights. The stack is ggml's `block_q8_0` stream unchanged, 34 bytes per
//! 32 values — `d` (f16) then 32 signed codes — rows back to back as
//! little-endian words, zero-padded at the end (`DevWeight::KQuant`). Block
//! `g` of the stream (`g = row · k/32 + b`) starts at byte `34·g`: its nine
//! words start at word `17·g >> 1`, for `g` even `d` is that word's low half
//! and the codes start two bytes in, for `g` odd `d` is its high half and the
//! codes start on the next word; code word `i` is `kquant::q8_0::q8_0_code`
//! of words `i`, `i + 1` shifted by 16 or 32. A row of an odd count of blocks
//! starts half a word in on every other row, so the block's parity is its
//! index in the stream, not in its row.
//!
//! Activation. [`Q8Blocks32`] of one column a slot, the 32-value q8_1 the
//! q5 gemvs read (`crate::q5_1_sel`'s module doc).
//!
//! Arithmetic. Lane `b` of the row's warp owns block `b` (warp stride): its
//! eight code words, values `4·i ..` in word `i`, run the q5 dp4a chain
//! (`q5::q5_a_chain`, signed × signed) against the column, and the block
//! adds `(A·d)·e` to the lane's partial; the fixed warp tree reduces the
//! partials. A NaN or infinite `d` makes its row NaN.
//!
//! Ids. [`HOST`] is a slot the host tier serves: the slot's warps return
//! before their first load and raise nothing. Any other id at or past the
//! stack's expert count raises [`FaultSite::ExpertId`] first (warp-uniform);
//! the slot's rows of `y` stay as they were.

use crate::cores::half_to_f32;
use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::kquant::q8_0::q8_0_code;
use crate::q5::{Q8Blocks32, q5_a_chain};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Bytes of one `block_q8_0` (32 values).
pub const BLOCK_BYTES: usize = 34;
/// Threads per block of the entry, one warp per output row.
pub const THREADS: u32 = 256;
/// Output rows a block computes: one per warp.
pub const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

// The entry's launch attributes spell the block as a literal and its launch
// contract the stream as `17 / 2` words a block.
const _: () = assert!(THREADS == 256 && BLOCK_BYTES == 34);

/// Words of a stream of `rows` rows of `k_blocks` blocks each as the card
/// holds it: the stream's bytes in whole words, rounded up to a whole number
/// of words a row (`DevWeight::KQuant`'s layout), and the words a row.
#[must_use]
pub fn stream_cols(k_blocks: usize, rows: usize) -> Option<usize> {
    let bytes = BLOCK_BYTES.checked_mul(k_blocks)?.checked_mul(rows)?;
    (rows > 0).then(|| bytes.div_ceil(4).div_ceil(rows))
}

/// One row's dot with activation column `col`, lane `lane`'s partial (the
/// caller reduces over the warp): lane `b` takes blocks `b, b + 32, ..` of
/// row `row_abs` and adds each block's `(A·d)·e` (module doc).
///
/// # Safety
///
/// `2 · w.len() >= 17 · (row_abs + 1) · k_blocks` (the stream holds the row's
/// last block, whose nine words end inside it), `q.len() >= (col + 1) ·
/// q_stride` with `q_stride >= 256 · ceil(k_blocks / 32)`, `d8.len() >= (col
/// + 1) · k_blocks`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn q8_0_row_dot32(
    w: &[u32],
    q: &[u32],
    d8: &[f32],
    k_blocks: usize,
    q_stride: usize,
    row_abs: usize,
    col: usize,
    lane: usize,
) -> f32 {
    let g0 = row_abs * k_blocks;
    let q0 = col * q_stride;
    let d8b0 = col * k_blocks;
    let mut f0 = 0.0f32;
    let mut b = lane;
    while b < k_blocks {
        let g = g0 + b;
        let wb = (17 * g) >> 1;
        let odd = (g & 1) as u32;
        // SAFETY: block g's bytes are 34·g .. 34·g + 34, inside the stream's
        // ceil(17·(g + 1) / 2) words by this fn's contract (g < (row_abs + 1)
        // · k_blocks); the last of its nine words, wb + 8, is the word of its
        // last byte.
        let v = unsafe {
            [
                *w.get_unchecked(wb),
                *w.get_unchecked(wb + 1),
                *w.get_unchecked(wb + 2),
                *w.get_unchecked(wb + 3),
                *w.get_unchecked(wb + 4),
                *w.get_unchecked(wb + 5),
                *w.get_unchecked(wb + 6),
                *w.get_unchecked(wb + 7),
                *w.get_unchecked(wb + 8),
            ]
        };
        let d = half_to_f32((v[0] >> (16 * odd)) as u16);
        let sh = 16 + 16 * odd;
        let cw = [
            q8_0_code(v[0], v[1], sh),
            q8_0_code(v[1], v[2], sh),
            q8_0_code(v[2], v[3], sh),
            q8_0_code(v[3], v[4], sh),
            q8_0_code(v[4], v[5], sh),
            q8_0_code(v[5], v[6], sh),
            q8_0_code(v[6], v[7], sh),
            q8_0_code(v[7], v[8], sh),
        ];
        // SAFETY: d8b0 + b < (col + 1) · k_blocks <= d8.len() by this fn's
        // contract.
        let e = unsafe { *d8.get_unchecked(d8b0 + b) };
        // SAFETY: q5_a_chain's window: q0 + 256·(b >> 5) + (b & 31) + 224 <
        // q0 + 256·((b >> 5) + 1) <= q0 + q_stride <= q.len(), by this fn's
        // `# Safety` and b < k_blocks.
        let a = unsafe { q5_a_chain(&cw, q, q0 + 256 * (b >> 5) + (b & 31)) };
        f0 += (a as f32 * d) * e;
        b += 32;
    }
    f0
}

#[cuda_module]
mod q8_0_sel32_kernels {
    use super::*;

    /// Q8_0 gemv over expert slots selected on the device, 32 values a
    /// block: one warp per output row, eight rows per 256-thread block;
    /// thread row `n = slot · rows_per_expert + r` stores `y[n]` from weight
    /// row `sel[slot] · rows_per_expert + r` against column `slot` of the
    /// 32-value q8_1 activation ([`q8_0_row_dot32`]). Ids: module doc.
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
            2 * w.len() >= n_experts * rows_per_expert * 17 * k_blocks,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * k_blocks,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q8_0_gemv_sel32(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        k_blocks: u32,
        q_stride: u32,
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
        // SAFETY: row_abs < n_experts · rows_per_expert rows, so the stream
        // holds its last block by the launch contract; column slot < n_slots
        // of the activation, and the host passes the activation's own
        // q_stride (>= 256 · ceil(k_blocks/32)).
        let f0 = unsafe {
            q8_0_row_dot32(
                w,
                q,
                d8,
                k_blocks as usize,
                q_stride as usize,
                row_abs,
                slot,
                lane,
            )
        };
        let s0 = warp::reduce_sum_f32(f0);
        if lane == 0 {
            // SAFETY: row < n_slots · rows_per_expert <= y.len() by the
            // launch contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }
}

/// A Q8_0 down `_sel` launch ([`Q80SelKernels::enqueue_gemv_q8_0_sel32`]).
pub struct Q80SelDown<'a> {
    /// The resident stack: `n_experts · rows_per_expert` rows, the file's
    /// `block_q8_0` bytes as words ([`stream_cols`] words a row, `w.rows()` a
    /// positive multiple of `rows_per_expert`).
    pub w: &'a DeviceTensor<u32>,
    /// One 32-value q8_1 column per slot: `act.m() == n_slots`.
    pub act: &'a Q8Blocks32,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// The loaded Q8_0 expert-select module. Owns no stream: every enqueue takes
/// the engine stream, so launches order with the step and are capturable.
pub struct Q80SelKernels {
    module: q8_0_sel32_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

impl Q80SelKernels {
    /// Load this file's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Q80SelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Q80SelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(q8_0_sel32_kernels, ctx)? };
        Ok(Q80SelKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the Q8_0 down `_sel`: slot `s` writes `y[s · rows_per_expert
    /// ..][..rows_per_expert]` as the rows of expert `sel[s]` dotted with
    /// column `s` of `a.act` (module doc for [`HOST`] and ids past the
    /// stack, which raise on `fault`). Refused by name: a stack whose row
    /// width is not the activation's K in `block_q8_0`s (K itself a multiple
    /// of 32, as [`Q8Blocks32`] holds it), a non-dividing expert size, and
    /// slots that the activation, the ids or `y` do not cover. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_gemv_q8_0_sel32(
        &self,
        stream: &CudaStream,
        a: &Q80SelDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q8_0_sel32";
        let k = a.act.k();
        let k_blocks = k / 32;
        let (n_slots, rpe) = (a.n_slots, a.rows_per_expert);
        if rpe == 0 || a.w.rows() == 0 || !a.w.rows().is_multiple_of(rpe) {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.rows() {} must be a positive multiple of rows_per_expert {rpe}",
                    a.w.rows()
                ),
            ));
        }
        let rows = a.w.rows();
        let bytes = BLOCK_BYTES * k_blocks * rows;
        let cols = stream_cols(k_blocks, rows);
        if !k.is_multiple_of(32) || cols != Some(a.w.cols()) || 4 * a.w.buf().len() < bytes {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q8_0 rows of k = {k} (a multiple of 32) are {} blocks of 34 bytes: \
                     {rows} rows take {cols:?} words a row over {bytes} bytes, got {} words a \
                     row over {} bytes",
                    k / 32,
                    a.w.cols(),
                    4 * a.w.buf().len()
                ),
            ));
        }
        if n_slots == 0 || a.act.m() != n_slots || a.sel.len() < n_slots || y.len() < n_slots * rpe
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots (at least one): one activation column a slot (act.m() {}), \
                     an id a slot (sel.len() {}), y.len() {} >= n_slots*rows_per_expert = {}",
                    a.act.m(),
                    a.sel.len(),
                    y.len(),
                    n_slots * rpe
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?;
        let prep = self
            .module
            .prepare_q8_0_gemv_sel32(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.q8_0_gemv_sel32(
            stream,
            &prep,
            a.w.buf(),
            &a.act.q,
            &a.act.d8,
            a.sel,
            launch_u32(what, "k_blocks", k_blocks)?,
            launch_u32(what, "q_stride", a.act.q_stride())?,
            launch_u32(what, "n_experts", rows / rpe)?,
            launch_u32(what, "rows_per_expert", rpe)?,
            launch_u32(what, "n_slots", n_slots)?,
            fault,
            y,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::stream_cols;
    use runtime::words::stream_words;

    /// A stack's words a row are the card upload's (`stream_words` over the
    /// rows' file bytes): Qwen3.8's routed down (2,560 rows of 640 values,
    /// 170 words a row), a row of an odd count of blocks, one block.
    #[test]
    fn stream_cols_are_the_uploads() {
        for (kb, rows) in [
            (20, 2560),
            (20, 7 * 2560),
            (35, 64),
            (35, 1),
            (1, 64),
            (1, 1),
        ] {
            let want =
                stream_words((34 * kb * rows) as u64, rows as u64).map(|w| w as usize / rows);
            assert_eq!(stream_cols(kb, rows), want, "{kb} blocks x {rows} rows");
        }
        assert_eq!(stream_cols(20, 2560), Some(170));
        assert_eq!(stream_cols(20, 0), None);
    }
}
