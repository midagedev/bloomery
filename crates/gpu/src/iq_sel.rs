//! The i-quant expert-select entries over the file's own blocks, resident as
//! the KQuant word stream ([`crate::weights`]'s `DevWeight::KQuant`: the row
//! stream as little-endian u32 words, rows back to back, zero-padded at the
//! end) — the card leg of Qwen3.8 UD-Q3_K_XL's routed experts, where the
//! gate·up stacks are IQ3_XXS and the down stack IQ4_NL:
//!
//! - `iq3_xxs_gate_up_sel`, `iq4_xs_gate_up_sel`: the [`crate::iq`] gate·up
//!   with its rule as a launch argument ([`crate::kquant::act`]), one warp a
//!   thread row, eight rows a 256-thread block; thread row `n = slot ·
//!   rows_per_expert + r` dots weight row `sel[slot] · rows_per_expert + r`
//!   of the gate stack and of the up stack with activation column `slot /
//!   slots_per_col` of a [`Q8Act`] and stores `h[n] = act::apply(act, limit,
//!   g, u)` from lane 0. Each row's dot is `iq.rs`'s rule exactly — the same
//!   lane split, grid words, integer factors and fused multiply-adds as the
//!   plane kernels — so a row's `g` equals `iq3_xxs_rows` /
//!   `iq4_xs_rows` over `IqFormat::repack` of the same row, bit for bit;
//!   only the reading differs (the raw-block windows below).
//! - `iq4_nl_gemv_sel32`: [`crate::q8_0_sel32`]'s shape with IQ4_NL's
//!   18-byte blocks — slot `s` dots the rows of expert `sel[s]` with column
//!   `s` of a [`Q8Blocks32`], lane `b` owning block `b` of the row (warp
//!   stride) and adding `(A·d)·e` (`q8_0_row_dot32`'s term with this
//!   decoder).
//!
//! Raw-block windows. A super-block of absolute row `ra` (the expert's row
//! `sel[slot] · rows_per_expert + r`) starts at byte `ra · bb · n_sb + bb ·
//! hb`, `bb` the format's block bytes (IQ3_XXS 98, IQ4_XS 136), and every
//! field is read through whole words:
//! - IQ3_XXS: `x` even, so `d` (f16 at `x`) is one half of word `x >> 2`;
//!   sub-block `ib`'s eight index bytes (`x + 2 + 8·ib`) are two windows over
//!   three words, its `aux` word (`x + 66 + 4·ib`) one window over two —
//!   each `kquant::q8_0::q8_0_code(lo, hi, 8·(byte & 3))`, the parity-aware
//!   extraction `q8_0_sel32` reads `block_q8_0`'s fields with.
//! - IQ4_XS: 136-byte super-blocks are word-aligned; `d | scales_h << 16` is
//!   word `x/4`, `scales_l` word `x/4 + 1`, sub-block `ib`'s code words
//!   `x/4 + 2 + 4·ib .. + 4`.
//! - IQ4_NL: block `g` of the stream (`g = ra · k/32 + b`) starts at byte
//!   `18·g`; five words from word `9·g >> 1` cover it, `d` the half word `g`'s
//!   parity picks and code word `i < 4` the window `2 + 4i .. 2 + 4i + 4` of
//!   the block, `iq4_word`'s low- and high-nibble value words in `[lo0..lo3,
//!   hi0..hi3]` order (values `0..16` then `16..32`).
//!
//! Ids, as the K-quant family ([`crate::kquant::sel`]). [`HOST`] is a slot
//! the host tier serves: the slot's warps return before their first load,
//! write nothing and raise nothing. Any other id at or past the stack's
//! expert count raises [`FaultSite::ExpertId`] first (warp-uniform, before
//! the first weight load); the gate·up writes NaN into its rows of `h`, the
//! down leaves its rows of `y` as they were.
//!
//! A non-finite `d` poisons its row, never a finite value: a NaN `d` makes
//! every product of its block NaN, and an inf `d` makes the row's sums ±inf —
//! NaN once the row's terms cancel (both signs in one row) or through the
//! gate·up's rule (`silu(−inf)` is NaN in both of [`Act`]'s rules). The raw
//! bytes carry what the file wrote: unlike `IqFormat::repack`, which refuses
//! a non-finite scale at load, these entries read it and give the row to the
//! arithmetic.
//!
//! Activations. The gate·up's columns are a [`Q8Act`]'s q4 permutation and
//! the down's a [`Q8Blocks32`]'s 32-value blocks, both quantized by the
//! engine's own quantizers; a column a quantizer refused holds a NaN scale
//! and NaNs every row that reads it.

use crate::cores::half_to_f32;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::half_bits_to_f32;
use crate::hybrid::HOST;
use crate::iq::{ActCols, dp4_all, iq3_group, iq4_word};
use crate::kquant::act::{self, Act};
use crate::kquant::q8_0::q8_0_code;
use crate::q5::{Q8Blocks32, q5_a_chain};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use gguf::iq_tables::KVALUES_IQ4NL;
use gguf::quant::GgmlType;
use runtime::words::stream_words;
use std::sync::Arc;

/// Threads per block of the entries, one warp per output row.
pub const THREADS: u32 = 256;
/// Output rows a block computes: one per warp.
pub const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

// The entries' launch attributes spell the block as a literal and the
// contracts the streams as whole words a block (98 B = 49 half-words, 136 B
// = 34 words, 18 B = 9 half-words).
const _: () = assert!(THREADS == 256);

/// Bytes of one `block_iq4_nl` (32 values).
pub const IQ4_NL_BLOCK_BYTES: usize = 18;

// ---------------------------------------------------------------- device

/// A gate·up format's raw super-block: its file bytes and one sub-block's
/// decode into the plane rule's inputs.
trait RawSb {
    /// File bytes per 256-value super-block.
    const BB: usize;

    /// Sub-block `ib` of the super-block starting at byte `x`: its eight
    /// weight words in value order, its integer factor `ai` and its scale
    /// `d` — the values `iq3_xxs_lane_partials` / `iq4_xs_lane_partials`
    /// compute from the planes, so the fma that consumes them is theirs.
    /// The caller keeps `x` even (IQ3_XXS) or word-aligned (IQ4_XS) and `x +
    /// Self::BB` inside `w`'s stream (rust-quality R28: a crate-internal fn
    /// over unchecked reads under a caller-kept boundary).
    fn sub(w: &[u32], x: usize, ib: usize) -> ([u32; 8], i32, f32);
}

/// IQ3_XXS's raw super-block ([`RawSb`]).
struct Iq3Xxs;

impl RawSb for Iq3Xxs {
    const BB: usize = 98;

    #[inline(always)]
    fn sub(w: &[u32], x: usize, ib: usize) -> ([u32; 8], i32, f32) {
        // x is even, so the f16 at x is one half of word x >> 2, inside the
        // stream by this fn's contract.
        // SAFETY: x even and x + 2 inside the stream.
        let dw = unsafe { *w.get_unchecked(x >> 2) };
        let d = half_bits_to_f32((dw >> (8 * (x & 2) as u32)) as u16) * 0.25;
        let at = x + 2 + 8 * ib;
        let (i0, sh) = ((at >> 2), 8 * (at & 3) as u32);
        // SAFETY: bytes at .. at + 8 (the sub-block's eight index bytes)
        // lie inside the stream, so the three words of their window do.
        let (cw0, cw1) = unsafe {
            let v0 = *w.get_unchecked(i0);
            let v1 = *w.get_unchecked(i0 + 1);
            let v2 = *w.get_unchecked(i0 + 2);
            (q8_0_code(v0, v1, sh), q8_0_code(v1, v2, sh))
        };
        let aa = x + 66 + 4 * ib;
        let aw = aa >> 2;
        // SAFETY: bytes aa .. aa + 4 (the sub-block's aux word) lie inside
        // the stream, so the two words of their window do.
        let ax = unsafe {
            q8_0_code(
                *w.get_unchecked(aw),
                *w.get_unchecked(aw + 1),
                8 * (aa & 3) as u32,
            )
        };
        let ai = (2 * (ax >> 28) + 1) as i32;
        let (w0, w1) = iq3_group(cw0, cw0 >> 8, ax);
        let (w2, w3) = iq3_group(cw0 >> 16, cw0 >> 24, ax >> 7);
        let (w4, w5) = iq3_group(cw1, cw1 >> 8, ax >> 14);
        let (w6, w7) = iq3_group(cw1 >> 16, cw1 >> 24, ax >> 21);
        ([w0, w1, w2, w3, w4, w5, w6, w7], ai, d)
    }
}

/// IQ4_XS's raw super-block ([`RawSb`]).
struct Iq4Xs;

impl RawSb for Iq4Xs {
    const BB: usize = 136;

    #[inline(always)]
    fn sub(w: &[u32], x: usize, ib: usize) -> ([u32; 8], i32, f32) {
        let w0 = x >> 2;
        // SAFETY: words w0 and w0 + 1 (d | scales_h << 16 and scales_l) lie
        // inside the stream by this fn's contract.
        let (m0, sl) = unsafe { (*w.get_unchecked(w0), *w.get_unchecked(w0 + 1)) };
        let d = half_bits_to_f32(m0 as u16);
        let s = ib as u32;
        let ls = ((sl >> (8 * (s >> 1) + 4 * (s & 1))) & 15) | (((m0 >> (16 + 2 * s)) & 3) << 4);
        let ai = ls as i32 - 32;
        let c = w0 + 2 + 4 * ib;
        // SAFETY: words c .. c + 3 (sub-block ib's sixteen code bytes) lie
        // inside the super-block, word w0 + 33 its last, by this fn's
        // contract.
        let (q0, q1, q2, q3) = unsafe {
            (
                *w.get_unchecked(c),
                *w.get_unchecked(c + 1),
                *w.get_unchecked(c + 2),
                *w.get_unchecked(c + 3),
            )
        };
        let (l0, h0) = iq4_word(q0);
        let (l1, h1) = iq4_word(q1);
        let (l2, h2) = iq4_word(q2);
        let (l3, h3) = iq4_word(q3);
        ([l0, l1, l2, l3, h0, h1, h2, h3], ai, d)
    }
}

/// The body of a gate·up entry over raw format `D` (module doc): thread row
/// `n = slot · rows_per_expert + r` dots weight row `sel[slot] ·
/// rows_per_expert + r` of both stacks with activation column `slot /
/// slots_per_col` — each dot `iq.rs`'s lane rule over the raw blocks — and
/// stores `h[n] = act::apply(act, limit, g, u)` from lane 0.
///
/// The caller keeps the entry's launch contract over its `n_slots` slots:
/// both stacks hold `D::BB`-byte super-block streams of `n_experts ·
/// rows_per_expert` rows of `n_sb` super-blocks, `slots_per_col >= 1`,
/// `n_slots <= m_cols · slots_per_col`, the activation planes `m_cols`
/// columns of `256 · n_grp` q words and `2 · n_sb` scales (`n_grp =
/// ceil(n_sb / 4)`), `sel.len() >= n_slots`, `h.len() >= n_slots ·
/// rows_per_expert`; `row < n_slots · rows_per_expert` is the calling warp's
/// row (one warp a row), all 32 lanes of the warp here (rust-quality R28: a
/// crate-internal fn over unchecked reads under a caller-kept boundary).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
fn gate_up_sel_body<D: RawSb>(
    wg: &[u32],
    wu: &[u32],
    q: &[u32],
    d8: &[f32],
    sel: &[u32],
    n_experts: u32,
    rows_per_expert: u32,
    slots_per_col: u32,
    n_sb: u32,
    n_grp: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    row: usize,
    h: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by this fn's contract; the load is
    // warp-uniform (the warp's lanes share `row`, hence `slot`), so the
    // returns below never diverge a warp.
    let id = unsafe { *sel.get_unchecked(slot) };
    let lane = warp::lane_id() as usize;
    if id >= n_experts {
        if id != HOST && lane == 0 {
            fault.raise(FaultSite::ExpertId);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
            // contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = f32::NAN };
        }
        return;
    }
    let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
    let col = slot / slots_per_col as usize;
    let a = ActCols {
        q,
        d8,
        q0: col * 256 * n_grp as usize,
        q_col: 0,
        d0: col * 2 * n_sb as usize,
        d_col: 0,
    };
    let n_sub = 8 * n_sb as usize;
    let base = row_abs * D::BB * n_sb as usize;
    let (mut fg, mut fu) = (0.0f32, 0.0f32);
    let mut b = lane;
    while b < n_sub {
        let x = base + D::BB * (b >> 3);
        // SAFETY: b < n_sub, so super-block b >> 3 of row row_abs is inside
        // both stacks by this fn's contract; column col < m_cols because
        // slot < n_slots <= m_cols · slots_per_col; all 32 lanes of the warp
        // are here (both returns above are warp-uniform).
        let ((xw, dx), (wgw, gai, gd), (wuw, uai, ud)) =
            unsafe { (a.col(0, b), D::sub(wg, x, b & 7), D::sub(wu, x, b & 7)) };
        let i = gai * dp4_all(&wgw, &xw);
        fg = (gd * dx).mul_add(i as f32, fg);
        let i = uai * dp4_all(&wuw, &xw);
        fu = (ud * dx).mul_add(i as f32, fu);
        b += 32;
    }
    let g = warp::reduce_sum_f32(fg);
    let u = warp::reduce_sum_f32(fu);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
        // contract; only lane 0 of the warp writes h[row].
        unsafe { *h.get_unchecked_mut(row) = act::apply(act, limit, g, u) };
    }
}

/// One IQ4_NL row's dot with activation column `col`, lane `lane`'s partial
/// (the caller reduces over the warp): lane `b` takes blocks `b, b + 32, ..`
/// of row `row_abs` and adds each block's `(A·d)·e` (module doc).
///
/// The caller keeps `2 · w.len() >= 9 · (row_abs + 1) · k_blocks` (the
/// stream holds the row's last block, whose five words end inside it),
/// `q.len() >= (col + 1) · q_stride` with `q_stride >= 256 ·
/// ceil(k_blocks / 32)`, `d8.len() >= (col + 1) · k_blocks` (rust-quality
/// R28: a crate-internal fn over unchecked reads under a caller-kept
/// boundary).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) fn iq4_nl_row_dot32(
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
        let wb = (9 * g) >> 1;
        let odd = (g & 1) as u32;
        // SAFETY: block g's bytes are 18·g .. 18·g + 18, inside the stream's
        // words by this fn's contract (g < (row_abs + 1) · k_blocks); the
        // last of its five words, wb + 4, is the word of its last byte.
        let v = unsafe {
            [
                *w.get_unchecked(wb),
                *w.get_unchecked(wb + 1),
                *w.get_unchecked(wb + 2),
                *w.get_unchecked(wb + 3),
                *w.get_unchecked(wb + 4),
            ]
        };
        let d = half_to_f32((v[0] >> (16 * odd)) as u16);
        let sh = 16 + 16 * odd;
        let cw = [
            q8_0_code(v[0], v[1], sh),
            q8_0_code(v[1], v[2], sh),
            q8_0_code(v[2], v[3], sh),
            q8_0_code(v[3], v[4], sh),
        ];
        let (l0, h0) = iq4_word(cw[0]);
        let (l1, h1) = iq4_word(cw[1]);
        let (l2, h2) = iq4_word(cw[2]);
        let (l3, h3) = iq4_word(cw[3]);
        let wv = [l0, l1, l2, l3, h0, h1, h2, h3];
        // SAFETY: d8b0 + b < (col + 1) · k_blocks <= d8.len() by this fn's
        // contract.
        let e = unsafe { *d8.get_unchecked(d8b0 + b) };
        // SAFETY: q5_a_chain's window: q0 + 256·(b >> 5) + (b & 31) + 224 <
        // q0 + 256·((b >> 5) + 1) <= q0 + q_stride <= q.len(), by this fn's
        // `# Safety` and b < k_blocks.
        let a = unsafe { q5_a_chain(&wv, q, q0 + 256 * (b >> 5) + (b & 31)) };
        f0 += (a as f32 * d) * e;
        b += 32;
    }
    f0
}

#[cuda_module]
mod iq_sel_kernels {
    use super::*;

    /// The IQ3_XXS gate·up over raw blocks ([`gate_up_sel_body`] over
    /// [`Iq3Xxs`]): 98-byte super-blocks at even byte offsets, read through
    /// the parity windows of the module doc. `m_cols` only bounds the
    /// activation in the launch contract.
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
            2 * wg.len() >= n_experts * rows_per_expert * 49 * n_sb,
            2 * wu.len() >= n_experts * rows_per_expert * 49 * n_sb,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn iq3_xxs_gate_up_sel(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        n_grp: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // The launch contract is the body's (49 words two a 98-byte
        // super-block); the host passes n_grp = ceil(n_sb/4) and one of
        // Act::code's codes; the return above is warp-uniform (a warp's
        // lanes share `row`) and keeps row in range.
        gate_up_sel_body::<Iq3Xxs>(
            wg,
            wu,
            q,
            d8,
            sel,
            n_experts,
            rows_per_expert,
            slots_per_col,
            n_sb,
            n_grp,
            act,
            limit,
            fault,
            row,
            &mut h,
        );
    }

    /// The IQ4_XS gate·up over raw blocks ([`gate_up_sel_body`] over
    /// [`Iq4Xs`]): word-aligned 136-byte super-blocks. `m_cols` only bounds
    /// the activation in the launch contract.
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
            wg.len() >= n_experts * rows_per_expert * 34 * n_sb,
            wu.len() >= n_experts * rows_per_expert * 34 * n_sb,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn iq4_xs_gate_up_sel(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        n_grp: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // The launch contract is the body's (34 words a 136-byte
        // super-block); the host passes n_grp = ceil(n_sb/4) and one of
        // Act::code's codes; the return above is warp-uniform and keeps row
        // in range.
        gate_up_sel_body::<Iq4Xs>(
            wg,
            wu,
            q,
            d8,
            sel,
            n_experts,
            rows_per_expert,
            slots_per_col,
            n_sb,
            n_grp,
            act,
            limit,
            fault,
            row,
            &mut h,
        );
    }

    /// The IQ4_NL down `_sel` of 32-value blocks ([`iq4_nl_row_dot32`]): one
    /// warp per output row, eight rows per 256-thread block; thread row `n =
    /// slot · rows_per_expert + r` stores `y[n]`, the dot of weight row
    /// `sel[slot] · rows_per_expert + r` with column `slot` of the 32-value
    /// q8_1 activation. Ids: module doc.
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
            2 * w.len() >= n_experts * rows_per_expert * 9 * k_blocks,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * k_blocks,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn iq4_nl_gemv_sel32(
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
        // row_abs < n_experts · rows_per_expert rows, so the stream holds
        // its last block by the launch contract; column slot < n_slots of
        // the activation, and the host passes the activation's own q_stride
        // (>= 256 · ceil(k_blocks/32)).
        let f0 = iq4_nl_row_dot32(
            w,
            q,
            d8,
            k_blocks as usize,
            q_stride as usize,
            row_abs,
            slot,
            lane,
        );
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

// ------------------------------------------------------------------ host

/// Lane `lane`'s partial of one IQ4_NL row on the host, [`iq4_nl_row_dot32`]'s
/// transcription: the same words, windows, block order and term arithmetic,
/// the nibbles read through `KVALUES_IQ4NL` instead of the register table
/// (the integer sums are exact either way, so the two agree bit for bit).
/// `xq` is the column's int8 codes in value order, `d8` its block scales.
#[allow(
    clippy::too_many_arguments,
    reason = "the device core's twin: one argument each (rust-quality R8)"
)]
#[must_use]
pub fn iq4_nl_row_dot32_host(
    w: &[u32],
    xq: &[i8],
    d8: &[f32],
    k_blocks: usize,
    row_abs: usize,
    lane: usize,
) -> f32 {
    let kv = |c: u8| i64::from(KVALUES_IQ4NL[usize::from(c & 15)]);
    let mut f0 = 0.0f32;
    let mut b = lane;
    while b < k_blocks {
        let g = row_abs * k_blocks + b;
        let wb = (9 * g) >> 1;
        let odd = (g & 1) as u32;
        let v = [w[wb], w[wb + 1], w[wb + 2], w[wb + 3], w[wb + 4]];
        let d = half_to_f32((v[0] >> (16 * odd)) as u16);
        let sh = 16 + 16 * odd;
        let mut a = 0i64;
        for i in 0..4 {
            let cwi = q8_0_code(v[i], v[i + 1], sh);
            for by in 0..4 {
                let byte = (cwi >> (8 * by)) as u8;
                a += kv(byte & 15) * i64::from(xq[32 * b + 4 * i + by]);
                a += kv(byte >> 4) * i64::from(xq[32 * b + 16 + 4 * i + by]);
            }
        }
        let e = d8[b];
        f0 += (a as f32 * d) * e;
        b += 32;
    }
    f0
}

// -------------------------------------------------------------- launcher

/// A gate·up launch ([`IqSelKernels::enqueue_gate_up`]).
pub struct IqGateUp<'a> {
    /// The format of both stacks: IQ3_XXS or IQ4_XS.
    pub ty: GgmlType,
    /// The gate stack: `n_experts · rows_per_expert` rows of the file's
    /// super-blocks as the KQuant word stream ([`stream_words`] words a
    /// row, `w.rows()` a positive multiple of `rows_per_expert`).
    pub wg: &'a DeviceTensor<u32>,
    /// The up stack, the same shape as `wg`.
    pub wu: &'a DeviceTensor<u32>,
    /// `act.m()` q8_1 columns; slot `s` reads column `s / slots_per_col`.
    pub act: &'a Q8Act,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    /// Slots a column: a token's run of slots.
    pub slots_per_col: usize,
    pub rule: Act,
}

/// An IQ4_NL down `_sel` launch ([`IqSelKernels::enqueue_down`]).
pub struct IqDown<'a> {
    /// The resident stack: `n_experts · rows_per_expert` rows of the file's
    /// `block_iq4_nl` bytes as the KQuant word stream ([`stream_words`]
    /// words a row, `w.rows()` a positive multiple of `rows_per_expert`).
    pub w: &'a DeviceTensor<u32>,
    /// One 32-value q8_1 column per slot: `act.m() == n_slots`.
    pub act: &'a Q8Blocks32,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// The loaded i-quant expert-select module and its launchers. Owns no
/// stream: every enqueue takes the engine stream, so launches order with the
/// step and are capturable.
pub struct IqSelKernels {
    module: iq_sel_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

/// The experts of a raw stack of `bb`-byte super-blocks — the words a row
/// the card upload's, `stream_words` over the rows' bytes — or the shape
/// error naming `what` and `name`.
fn stack_experts(
    what: &'static str,
    name: &'static str,
    bb: usize,
    w: &DeviceTensor<u32>,
    n_sb: usize,
    rows_per_expert: usize,
) -> Result<usize, GpuError> {
    if rows_per_expert == 0 || w.rows() == 0 || !w.rows().is_multiple_of(rows_per_expert) {
        return Err(GpuError::shape(
            what,
            format!(
                "{name} rows {} must be a positive multiple of rows_per_expert {rows_per_expert}",
                w.rows()
            ),
        ));
    }
    let rows = w.rows();
    let bytes = bb * n_sb * rows;
    let cols = stream_words(bytes as u64, rows as u64).map(|x| (x / rows as u64) as usize);
    if cols != Some(w.cols()) || 4 * w.buf().len() < bytes {
        return Err(GpuError::shape(
            what,
            format!(
                "{name} rows of k = {} (a multiple of 256) are {n_sb} super-blocks of {bb} \
                 bytes: {rows} rows take {cols:?} words a row over {bytes} bytes, got {} words a \
                 row over {} bytes",
                256 * n_sb,
                w.cols(),
                4 * w.buf().len()
            ),
        ));
    }
    Ok(rows / rows_per_expert)
}

impl IqSelKernels {
    /// Load this file's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<IqSelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "IqSelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(iq_sel_kernels, ctx)? };
        Ok(IqSelKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the i-quant gate·up of `a.ty` (IQ3_XXS or IQ4_XS): slot `s`
    /// writes `h[s · rows_per_expert ..][..rows_per_expert]` as `a.rule`
    /// over the rows of expert `sel[s]` of both stacks dotted with column `s
    /// / slots_per_col` of `a.act` (module doc for [`HOST`] and ids past the
    /// stack, which raise on `fault` and write NaN). Refused by name: a
    /// format the entries do not take, a stack whose row width is not the
    /// KQuant stream of the type, `k` not a multiple of 256, a non-dividing
    /// expert size, an up stack of another shape, more than eight
    /// activation columns, and slots the activation, the ids or `h` do not
    /// cover. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up(
        &self,
        stream: &CudaStream,
        a: &IqGateUp<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "IqSelKernels::enqueue_gate_up";
        let (name, bb) = match a.ty {
            GgmlType::IQ3_XXS => ("IQ3_XXS", Iq3Xxs::BB),
            GgmlType::IQ4_XS => ("IQ4_XS", Iq4Xs::BB),
            other => {
                return Err(GpuError::shape(
                    what,
                    format!("the gate·up takes IQ3_XXS or IQ4_XS stacks, got {other:?}"),
                ));
            }
        };
        let n_sb = a.act.n_sb();
        if !a.act.k().is_multiple_of(256) {
            return Err(GpuError::shape(
                what,
                format!("{name} needs k a multiple of 256, got {}", a.act.k()),
            ));
        }
        let n_experts = stack_experts(what, name, bb, a.wg, n_sb, a.rows_per_expert)?;
        let (n_slots, rpe, spc, m) = (a.n_slots, a.rows_per_expert, a.slots_per_col, a.act.m());
        if a.wu.rows() != a.wg.rows() || a.wu.cols() != a.wg.cols() {
            return Err(GpuError::shape(
                what,
                format!(
                    "the up stack {} x {} must be the gate stack's {} x {}",
                    a.wu.rows(),
                    a.wu.cols(),
                    a.wg.rows(),
                    a.wg.cols()
                ),
            ));
        }
        if n_slots == 0
            || spc == 0
            || n_slots > m * spc
            || a.sel.len() < n_slots
            || h.len() < n_slots * rpe
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots (at least one) at {spc} a column (at least one) over {m} \
                     columns (at most eight), sel.len() {}, h.len() {} >= n_slots*rows_per_expert \
                     = {}",
                    a.sel.len(),
                    h.len(),
                    n_slots * rpe
                ),
            ));
        }
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(
                what,
                format!("1 <= m <= 8 activation columns, got {m}"),
            ));
        }
        let n_grp = n_sb.div_ceil(4);
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?;
        let (act, limit) = a.rule.code();
        match a.ty {
            GgmlType::IQ3_XXS => {
                let prep = self
                    .module
                    .prepare_iq3_xxs_gate_up_sel(LaunchConfig1D::new(grid, THREADS, 0))?;
                self.module.iq3_xxs_gate_up_sel(
                    stream,
                    &prep,
                    a.wg.buf(),
                    a.wu.buf(),
                    &a.act.q4,
                    &a.act.d8,
                    a.sel,
                    launch_u32(what, "n_experts", n_experts)?,
                    launch_u32(what, "rows_per_expert", rpe)?,
                    launch_u32(what, "n_slots", n_slots)?,
                    launch_u32(what, "m_cols", m)?,
                    launch_u32(what, "slots_per_col", spc)?,
                    launch_u32(what, "n_sb", n_sb)?,
                    launch_u32(what, "n_grp", n_grp)?,
                    act,
                    limit,
                    fault,
                    h,
                )?;
            }
            _ => {
                let prep = self
                    .module
                    .prepare_iq4_xs_gate_up_sel(LaunchConfig1D::new(grid, THREADS, 0))?;
                self.module.iq4_xs_gate_up_sel(
                    stream,
                    &prep,
                    a.wg.buf(),
                    a.wu.buf(),
                    &a.act.q4,
                    &a.act.d8,
                    a.sel,
                    launch_u32(what, "n_experts", n_experts)?,
                    launch_u32(what, "rows_per_expert", rpe)?,
                    launch_u32(what, "n_slots", n_slots)?,
                    launch_u32(what, "m_cols", m)?,
                    launch_u32(what, "slots_per_col", spc)?,
                    launch_u32(what, "n_sb", n_sb)?,
                    launch_u32(what, "n_grp", n_grp)?,
                    act,
                    limit,
                    fault,
                    h,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue the IQ4_NL down `_sel`: slot `s` writes `y[s ·
    /// rows_per_expert ..][..rows_per_expert]` as the rows of expert
    /// `sel[s]` dotted with column `s` of `a.act` (module doc for [`HOST`]
    /// and ids past the stack, which raise on `fault`). Refused by name: a
    /// stack whose row width is not the KQuant stream of `block_iq4_nl`s
    /// (K itself a multiple of 32, as [`Q8Blocks32`] holds it), a
    /// non-dividing expert size, and slots the activation, the ids or `y` do
    /// not cover. Asynchronous, allocation-free, capturable.
    pub fn enqueue_down(
        &self,
        stream: &CudaStream,
        a: &IqDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "IqSelKernels::enqueue_down";
        let k = a.act.k();
        let k_blocks = k / 32;
        if k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!("IQ4_NL needs k a positive multiple of 32, got {k}"),
            ));
        }
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
        let bytes = IQ4_NL_BLOCK_BYTES * k_blocks * rows;
        let cols = stream_words(bytes as u64, rows as u64).map(|x| (x / rows as u64) as usize);
        if cols != Some(a.w.cols()) || 4 * a.w.buf().len() < bytes {
            return Err(GpuError::shape(
                what,
                format!(
                    "IQ4_NL rows of k = {k} (a multiple of 32) are {k_blocks} blocks of {} \
                     bytes: {rows} rows take {cols:?} words a row over {bytes} bytes, got {} \
                     words a row over {} bytes",
                    IQ4_NL_BLOCK_BYTES,
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
            .prepare_iq4_nl_gemv_sel32(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.iq4_nl_gemv_sel32(
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
    use super::IQ4_NL_BLOCK_BYTES;
    use gguf::quant::half_to_f32;

    /// Words of a byte stream as the card holds it: little-endian words,
    /// zero-padded to a whole number of words (`runtime::words::stream_words`
    /// over the whole stream).
    fn words_of(b: &[u8]) -> Vec<u32> {
        b.chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            })
            .collect()
    }

    /// The host twin's block decode at both parities: for seeded blocks, the
    /// `d` and every nibble value `iq4_nl_row_dot32_host` reads through its
    /// word windows equal the plain little-endian byte reading of the same
    /// block (so the parity window math cannot drift from the layout).
    #[test]
    fn iq4_nl_host_windows_match_the_bytes() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Two rows of three blocks: block parities 0, 1, 0, 1, 0, 1 in the
        // stream (18 bytes a block).
        let mut bytes = Vec::new();
        for _ in 0..6 {
            let d = ((1 + (next() % 9)) as u16) << 10 | (next() & 0x3ff) as u16;
            bytes.extend_from_slice(&d.to_le_bytes());
            for _ in 0..16 {
                bytes.push((next() >> 32) as u8);
            }
        }
        let w = words_of(&bytes);
        let xq = [1i8; 32];
        let d8 = [1.0f32; 3];
        let kv = gguf::iq_tables::KVALUES_IQ4NL;
        for g in 0..6 {
            let blk = &bytes[g * IQ4_NL_BLOCK_BYTES..][..IQ4_NL_BLOCK_BYTES];
            let want_d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            // The twin's per-block reading, spelled from its own windows.
            let wb = (9 * g) >> 1;
            let odd = (g & 1) as u32;
            let d = half_to_f32((w[wb] >> (16 * odd)) as u16);
            assert_eq!(d.to_bits(), want_d.to_bits(), "block {g} d");
            let sh = 16 + 16 * odd;
            for i in 0..4 {
                let cwi = (((u64::from(w[wb + i + 1]) << 32) | u64::from(w[wb + i])) >> sh) as u32;
                for by in 0..4 {
                    let byte = blk[2 + 4 * i + by];
                    assert_eq!((cwi >> (8 * by)) as u8, byte, "block {g} code byte");
                    assert_eq!(
                        kv[usize::from(byte & 15)],
                        kv[usize::from(((cwi >> (8 * by)) as u8) & 15)],
                        "block {g} low nibble"
                    );
                    assert_eq!(
                        kv[usize::from(byte >> 4)],
                        kv[usize::from(((cwi >> (8 * by)) as u8) >> 4)],
                        "block {g} high nibble"
                    );
                }
            }
        }
        // One lane's partial over a row of three blocks sums what the scalar
        // rule does, term for term.
        let row = 1;
        let mut want = 0.0f32;
        for (b, &e) in d8.iter().enumerate() {
            let g = row * 3 + b;
            let blk = &bytes[g * IQ4_NL_BLOCK_BYTES..][..IQ4_NL_BLOCK_BYTES];
            let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let mut a = 0i64;
            for j in 0..16 {
                a += i64::from(kv[usize::from(blk[2 + j] & 15)]) * i64::from(xq[j]);
                a += i64::from(kv[usize::from(blk[2 + j] >> 4)]) * i64::from(xq[16 + j]);
            }
            want += (a as f32 * d) * e;
        }
        let got = super::iq4_nl_row_dot32_host(&w, &xq, &d8, 3, row, 0);
        assert_eq!(got.to_bits(), want.to_bits(), "row {row} lane 0 partial");
    }
}
