//! Q5_1 gemv over expert slots selected on the device: the MoE down
//! projection of a Q5_1 expert stack resident as the file's bytes, all
//! `n_slots` routed experts in one launch — the per-slot-column shape of
//! `q5_0_gemv_sel` and the K-quant `_sel` entries: slot `s` dots the rows of
//! expert `sel[s]` with activation column `s`.
//!
//! Weights. The stack is ggml's `block_q5_1` stream unchanged, 24 bytes (six
//! words) per 32 values: word 0 holds `d` (low half) and `m` (high half) as
//! f16, word 1 the high-bit plane `qh`, words 2..6 the nibbles `qs[16]`.
//! Value `j < 16` is `d·(nib_lo(qs[j]) + 16·qh_j) + m`, value `16 + j` is
//! `d·(nib_hi(qs[j]) + 16·qh_{16+j}) + m`. A row of `k` values is `6 · k/32`
//! words, so every row and block is word-aligned and nothing is repacked.
//!
//! Activation. [`Q8Blocks32`] of one column a slot
//! ([`Q8Blocks32::with_slots`]), the straight 32-value q8_1 geometry the q5
//! gemvs read: per block an f32 scale `e`, an i32 byte sum `s`, and the
//! codes in the transposed window order (block `b`'s word `i` at `256·(b >>
//! 5) + 32·i + (b & 31)`).
//!
//! Arithmetic. Lane `b` of the row's warp owns block `b` (warp stride): its
//! eight code words ([`q5_1_codes`]: bytes 0..31, word `i` holding values
//! `4·i ..`) run the q5 dp4a chain (`q5::q5_a_chain`) against the column, and
//! the block adds `(A·d + m·s)·e` to the lane's partial; the fixed warp tree
//! reduces the partials. That is `q5_row_dot`'s column-0 term over the
//! `pack_q5_1` layout, so a slot is `q5_1_gemv` of the packed copy of expert
//! `sel[s]` against column `s`, bit for bit (`gate_kquant` pins it).
//!
//! Ids. [`HOST`] is a slot the host tier serves: the slot's warps return
//! before their first load and raise nothing. Any other id at or past the
//! stack's expert count raises [`FaultSite::ExpertId`] first (warp-uniform);
//! the slot's rows of `y` stay as they were.

use crate::cores::half_to_f32;
use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::q5::{Q8Blocks32, q5_a_chain};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// u32 words of one `block_q5_1` (24 bytes, 32 values).
pub const BLOCK_WORDS: usize = 6;
/// Threads per block of the entry, one warp per output row.
pub const THREADS: u32 = 256;
/// Output rows a block computes: one per warp.
pub const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

// The entry's launch attributes spell the block as a literal and its launch
// contract the row as `6 * k_blocks` words.
const _: () = assert!(THREADS == 256 && BLOCK_WORDS == 6);

/// Bits 0..4 of `x` as bit 4 of bytes 0..4: the high-bit plane's four bits
/// of one code word, each at the fifth bit of its value's byte.
#[inline(always)]
#[must_use]
pub fn spread4(x: u32) -> u32 {
    ((x << 4) & 0x0000_0010)
        | ((x << 11) & 0x0000_1000)
        | ((x << 18) & 0x0010_0000)
        | ((x << 25) & 0x1000_0000)
}

/// The eight code words of one Q5_1 block from its `qh` word and its four
/// `qs` words: word `i < 4` is the low nibbles of `qs[i]` with `qh` bits
/// `4·i ..` as the fifth bits (values `4·i ..`), word `4 + i` the high
/// nibbles of `qs[i]` with `qh` bits `16 + 4·i ..` (values `16 + 4·i ..`).
/// Every byte is a code 0..31, exact as a signed byte against the q8 plane.
#[inline(always)]
#[must_use]
pub fn q5_1_codes(qh: u32, qs: [u32; 4]) -> [u32; 8] {
    [
        (qs[0] & 0x0f0f_0f0f) | spread4(qh),
        (qs[1] & 0x0f0f_0f0f) | spread4(qh >> 4),
        (qs[2] & 0x0f0f_0f0f) | spread4(qh >> 8),
        (qs[3] & 0x0f0f_0f0f) | spread4(qh >> 12),
        ((qs[0] >> 4) & 0x0f0f_0f0f) | spread4(qh >> 16),
        ((qs[1] >> 4) & 0x0f0f_0f0f) | spread4(qh >> 20),
        ((qs[2] >> 4) & 0x0f0f_0f0f) | spread4(qh >> 24),
        ((qs[3] >> 4) & 0x0f0f_0f0f) | spread4(qh >> 28),
    ]
}

/// One row's dot with activation column `col`, lane `lane`'s partial (the
/// caller reduces over the warp): lane `b` takes blocks `b, b + 32, ..` of
/// row `row_abs` and adds each block's `(A·d + m·s)·e` (module doc).
///
/// # Safety
///
/// `w.len() >= (row_abs + 1) · 6 · k_blocks`, `q.len() >= (col + 1) ·
/// q_stride` with `q_stride >= 256 · ceil(k_blocks / 32)`, `d8.len()` and
/// `s8.len() >= (col + 1) · k_blocks`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn q5_1_row_dot(
    w: &[u32],
    q: &[u32],
    d8: &[f32],
    s8: &[i32],
    k_blocks: usize,
    q_stride: usize,
    row_abs: usize,
    col: usize,
    lane: usize,
) -> f32 {
    let wbase = row_abs * BLOCK_WORDS * k_blocks;
    let q0 = col * q_stride;
    let d8b0 = col * k_blocks;
    let mut f0 = 0.0f32;
    let mut b = lane;
    while b < k_blocks {
        let wb = wbase + BLOCK_WORDS * b;
        // SAFETY: b < k_blocks, so wb + 5 < (row_abs + 1) · 6 · k_blocks <=
        // w.len() by this fn's contract.
        let (dm, qh, qs) = unsafe {
            (
                *w.get_unchecked(wb),
                *w.get_unchecked(wb + 1),
                [
                    *w.get_unchecked(wb + 2),
                    *w.get_unchecked(wb + 3),
                    *w.get_unchecked(wb + 4),
                    *w.get_unchecked(wb + 5),
                ],
            )
        };
        let d = half_to_f32((dm & 0xffff) as u16);
        let m = half_to_f32((dm >> 16) as u16);
        let cw = q5_1_codes(qh, qs);
        // SAFETY: d8b0 + b < (col + 1) · k_blocks <= d8.len(), s8.len() by
        // this fn's contract.
        let (e, s) = unsafe { (*d8.get_unchecked(d8b0 + b), *s8.get_unchecked(d8b0 + b)) };
        // q5_a_chain's window: q0 + 256·(b >> 5) + (b & 31) + 224 <
        // q0 + 256·((b >> 5) + 1) <= q0 + q_stride <= q.len().
        let a = q5_a_chain(&cw, q, q0 + 256 * (b >> 5) + (b & 31));
        // q5_row_dot's term, so the packed kernel's sum is this one's.
        f0 += (a as f32 * d + m * s as f32) * e;
        b += 32;
    }
    f0
}

#[cuda_module]
mod q5_1_sel_kernels {
    use super::*;

    /// Q5_1 gemv over expert slots selected on the device: one warp per
    /// output row, eight rows per 256-thread block; thread row `n = slot ·
    /// rows_per_expert + r` stores `y[n]` from weight row `sel[slot] ·
    /// rows_per_expert + r` against column `slot` of the 32-value q8_1
    /// activation ([`q5_1_row_dot`]). Ids: module doc.
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
            w.len() >= n_experts * rows_per_expert * 6 * k_blocks,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * k_blocks,
            s8.len() >= n_slots * k_blocks,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q5_1_gemv_sel(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        s8: &[i32],
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
        // SAFETY: row_abs < n_experts · rows_per_expert rows of `w`, column
        // slot < n_slots of the activation, and the host passes the
        // activation's own q_stride (>= 256 · ceil(k_blocks/32)).
        let f0 = unsafe {
            q5_1_row_dot(
                w,
                q,
                d8,
                s8,
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

/// A Q5_1 down `_sel` launch ([`Q51SelKernels::enqueue_gemv_q5_1_sel`]).
pub struct Q51SelDown<'a> {
    /// The resident stack: `n_experts · rows_per_expert` rows of `6 ·
    /// k/32` words, the file's `block_q5_1` bytes as words (`w.rows()` a
    /// positive multiple of `rows_per_expert`).
    pub w: &'a DeviceTensor<u32>,
    /// One 32-value q8_1 column per slot: `act.m() == n_slots`.
    pub act: &'a Q8Blocks32,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// The loaded Q5_1 expert-select module. Owns no stream: every enqueue takes
/// the engine stream, so launches order with the step and are capturable.
pub struct Q51SelKernels {
    module: q5_1_sel_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

impl Q51SelKernels {
    /// Load this file's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Q51SelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Q51SelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { q5_1_sel_kernels::load(ctx)? };
        Ok(Q51SelKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the Q5_1 down `_sel`: slot `s` writes `y[s · rows_per_expert
    /// ..][..rows_per_expert]` as the rows of expert `sel[s]` dotted with
    /// column `s` of `a.act` (module doc for [`HOST`] and ids past the
    /// stack, which raise on `fault`). Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gemv_q5_1_sel(
        &self,
        stream: &CudaStream,
        a: &Q51SelDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q5_1_sel";
        let k_blocks = a.act.k() / 32;
        let (n_slots, rpe) = (a.n_slots, a.rows_per_expert);
        if a.w.cols() != BLOCK_WORDS * k_blocks {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q5_1 rows of k = {} are 6*{k_blocks} = {} words, got {}",
                    a.act.k(),
                    BLOCK_WORDS * k_blocks,
                    a.w.cols()
                ),
            ));
        }
        if rpe == 0 || a.w.rows() == 0 || !a.w.rows().is_multiple_of(rpe) {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.rows() {} must be a positive multiple of rows_per_expert {rpe}",
                    a.w.rows()
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
            .prepare_q5_1_gemv_sel(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.q5_1_gemv_sel(
            stream,
            &prep,
            a.w.buf(),
            &a.act.q,
            &a.act.d8,
            &a.act.s8,
            a.sel,
            launch_u32(what, "k_blocks", k_blocks)?,
            launch_u32(what, "q_stride", a.act.q_stride())?,
            launch_u32(what, "n_experts", a.w.rows() / rpe)?,
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
    use super::{BLOCK_WORDS, q5_1_codes};
    use crate::q5::pack_q5_1;

    /// xorshift64, the gates' synthetic source.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// The raw decode is `pack_q5_1`'s: for random blocks, the eight code
    /// words from the block's `qh` and `qs` words are the packed layout's
    /// code section (word `i` at `32·i` of a one-block row).
    #[test]
    fn codes_are_pack_q5_1s() {
        let mut s = 0x51c0_de51_u64;
        for _ in 0..4096 {
            let mut words = [0u32; BLOCK_WORDS];
            for w in &mut words {
                *w = (next(&mut s) >> 32) as u32;
            }
            // d = 1.0, m = 0.0 as f16: the codes do not read them.
            words[0] = 0x3c00;
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let packed = pack_q5_1(&bytes, 32, 1).expect("one 32-value block packs");
            let want: Vec<u32> = (0..8).map(|i| packed[32 * i]).collect();
            let got = q5_1_codes(words[1], [words[2], words[3], words[4], words[5]]);
            assert_eq!(got.as_slice(), want.as_slice(), "block words {words:08x?}");
        }
    }
}
