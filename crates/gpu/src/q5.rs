//! Q5_0 / Q5_1 gemv (package P2): the legacy 32-value block types against
//! q8_1 activations quantized on the device. Cores live here outside this
//! file's `#[cuda_module]` (docs/gpu-design.md decision 6), the module holds
//! the activation quantizer and the two gemv wrappers, and the host side owns
//! the load-time weight repack, the activation scratch and the enqueue API.

use crate::GpuError;
use crate::q8_1_quant_block;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, dotprod::dp4a_s32, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use gguf::quant::half_to_f32;
use std::sync::Arc;

// ------------------------------------------------------------------- cores

/// The eight code words of 32-value block `b` of the row based at word
/// `wbase`, in the packed layout's transposed order (word i at
/// `wbase + 256*(b>>5) + 32*i + (b&31)`), so a warp's eight code loads are
/// lane-consecutive.
///
/// SAFETY: callers keep the eight words inside one row's code section —
/// `b < k_blocks` and the row based at `wbase` carries `q_stride >= 256`
/// code words with `q_stride >= 256*((b>>5)+1)`.
#[inline(always)]
pub fn q5_code_words(w: &[u32], wbase: usize, b: usize) -> [u32; 8] {
    let base = wbase + 256 * (b >> 5) + (b & 31);
    // SAFETY: base + 224 <= wbase + 256*(b>>5) + 255 < wbase + q_stride by
    // this fn's contract.
    unsafe {
        [
            *w.get_unchecked(base),
            *w.get_unchecked(base + 32),
            *w.get_unchecked(base + 64),
            *w.get_unchecked(base + 96),
            *w.get_unchecked(base + 128),
            *w.get_unchecked(base + 160),
            *w.get_unchecked(base + 192),
            *w.get_unchecked(base + 224),
        ]
    }
}

/// One block's A chain: its eight code words dp4a-accumulated against the q8
/// words at base `qb`, word i at `qb + 32*i` (the quantizer's matching
/// transposed order, so each of the eight loads is one 128 B line across the
/// warp).
///
/// SAFETY: callers keep `qb + 7*32` inside one column's q8 word span.
#[inline(always)]
pub fn q5_a_chain(cw: &[u32; 8], q: &[u32], qb: usize) -> i32 {
    // SAFETY: qb + 224 is inside the caller's column span by this fn's
    // contract (max qb within a window is 256*(b>>5) + (b&31), so + 224
    // stays under the next window boundary).
    let (w0, w1, w2, w3, w4, w5, w6, w7) = unsafe {
        (
            *q.get_unchecked(qb),
            *q.get_unchecked(qb + 32),
            *q.get_unchecked(qb + 64),
            *q.get_unchecked(qb + 96),
            *q.get_unchecked(qb + 128),
            *q.get_unchecked(qb + 160),
            *q.get_unchecked(qb + 192),
            *q.get_unchecked(qb + 224),
        )
    };
    let a = dp4a_s32(cw[0], w0, 0);
    let a = dp4a_s32(cw[1], w1, a);
    let a = dp4a_s32(cw[2], w2, a);
    let a = dp4a_s32(cw[3], w3, a);
    let a = dp4a_s32(cw[4], w4, a);
    let a = dp4a_s32(cw[5], w5, a);
    let a = dp4a_s32(cw[6], w6, a);
    dp4a_s32(cw[7], w7, a)
}

/// One (block, column) term. `mds` is the Q5_1 min-term numerator `m·s` of
/// this (block, column) pair; Q5_0 passes 0.0, for which the add is exact,
/// so both types share the body.
#[inline(always)]
fn q5_block_col(cw: &[u32; 8], q: &[u32], qb: usize, d: f32, mds: f32, e: f32) -> f32 {
    (q5_a_chain(cw, q, qb) as f32 * d + mds) * e
}

/// One row's dot products with `m_cols` (1..=8) activation columns,
/// pre-reduction: lane `lane` walks the row's 32-value blocks with warp
/// stride and keeps per-column partial sums in scalars; the wrapper reduces
/// them across the warp. Returns only the first `m_cols` accumulators.
///
/// Buffer layouts (the packers and `q5_quantize_q8` write, this reads):
/// - `w`, per row: `q_stride` code words (block b's word i at
///   `256*(b>>5) + 32*i + (b&31)`, one byte per value), then `k_blocks` f32
///   block scales d, then — Q5_1 only — `k_blocks` f32 block mins m.
///   Q5_0 code bytes hold code−16 ∈ [−16, 15], making the dp4a chain the
///   exact signed block dot; Q5_1 keeps the unsigned 0..31 code and adds the
///   `m·(block sum)` offset term instead (`q5_1` selects both).
/// - `q`, per column: `q_stride` words in the same transposed order; `d8`
///   and `s8`, per column: `k_blocks` entries, block-indexed. Column c reads
///   `(col0 + c)` of each.
///
/// SAFETY: callers guarantee `w.len() >= (row_abs + 1) * row_words` (with
/// `row_words = q_stride + k_blocks`, or `+ 2*k_blocks` when `q5_1`),
/// `q.len() >= (col0 + m_cols) * q_stride`, `d8.len()` and `s8.len()`
/// `>= (col0 + m_cols) * k_blocks`, and `1 <= m_cols <= 8`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub fn q5_row_dot(
    w: &[u32],
    q: &[u32],
    d8: &[f32],
    s8: &[i32],
    k_blocks: usize,
    q_stride: usize,
    row_abs: usize,
    col0: usize,
    m_cols: usize,
    lane: usize,
    q5_1: bool,
) -> [f32; 8] {
    let row_words = if q5_1 {
        q_stride + 2 * k_blocks
    } else {
        q_stride + k_blocks
    };
    let wbase = row_abs * row_words;
    let sbase = wbase + q_stride;
    let d8b0 = col0 * k_blocks;
    let q0 = col0 * q_stride;

    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;

    let mut b = lane;
    while b < k_blocks {
        let cw = q5_code_words(w, wbase, b);
        // SAFETY: sbase + b < sbase + k_blocks <= row end <= w.len() by this
        // fn's contract.
        let d = f32::from_bits(unsafe { *w.get_unchecked(sbase + b) });
        let mv = if q5_1 {
            // SAFETY: sbase + k_blocks + b < row end <= w.len().
            f32::from_bits(unsafe { *w.get_unchecked(sbase + k_blocks + b) })
        } else {
            0.0
        };
        let qslot = 256 * (b >> 5) + (b & 31);

        // Column 0 (always active). q0 + qslot + 224 < (col0+1)*q_stride <=
        // q.len() (m_cols >= 1) is `q5_block_col`'s contract.
        {
            // SAFETY: d8b0 + b < (col0+1)*k_blocks <= d8.len() (m_cols >= 1).
            let e0 = unsafe { *d8.get_unchecked(d8b0 + b) };
            let s0 = if q5_1 {
                // SAFETY: the same index as e0 in the parallel s8 buffer:
                // d8b0 + b < (col0+1)*k_blocks <= s8.len() when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + b) }
            } else {
                0
            };
            f0 += q5_block_col(&cw, q, q0 + qslot, d, mv * s0 as f32, e0);
        }
        // Columns 1..7: one launch-uniform guard per column so the work and
        // the buffer bounds scale with m_cols. Column c reads q words at
        // q0 + c*q_stride + qslot (+224 inside the chain), the d8/s8 block
        // at d8b0 + c*k_blocks + b.
        if m_cols > 1 {
            // SAFETY: m_cols > 1 => d8.len() >= (col0+2)*k_blocks >
            // d8b0 + k_blocks + b.
            let e1 = unsafe { *d8.get_unchecked(d8b0 + k_blocks + b) };
            let s1 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + k_blocks + b) }
            } else {
                0
            };
            f1 += q5_block_col(&cw, q, q0 + q_stride + qslot, d, mv * s1 as f32, e1);
        }
        if m_cols > 2 {
            // SAFETY: m_cols > 2 => d8.len() >= (col0+3)*k_blocks >
            // d8b0 + 2*k_blocks + b.
            let e2 = unsafe { *d8.get_unchecked(d8b0 + 2 * k_blocks + b) };
            let s2 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + 2 * k_blocks + b) }
            } else {
                0
            };
            f2 += q5_block_col(&cw, q, q0 + 2 * q_stride + qslot, d, mv * s2 as f32, e2);
        }
        if m_cols > 3 {
            // SAFETY: m_cols > 3 => d8.len() >= (col0+4)*k_blocks >
            // d8b0 + 3*k_blocks + b.
            let e3 = unsafe { *d8.get_unchecked(d8b0 + 3 * k_blocks + b) };
            let s3 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + 3 * k_blocks + b) }
            } else {
                0
            };
            f3 += q5_block_col(&cw, q, q0 + 3 * q_stride + qslot, d, mv * s3 as f32, e3);
        }
        if m_cols > 4 {
            // SAFETY: m_cols > 4 => d8.len() >= (col0+5)*k_blocks >
            // d8b0 + 4*k_blocks + b.
            let e4 = unsafe { *d8.get_unchecked(d8b0 + 4 * k_blocks + b) };
            let s4 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + 4 * k_blocks + b) }
            } else {
                0
            };
            f4 += q5_block_col(&cw, q, q0 + 4 * q_stride + qslot, d, mv * s4 as f32, e4);
        }
        if m_cols > 5 {
            // SAFETY: m_cols > 5 => d8.len() >= (col0+6)*k_blocks >
            // d8b0 + 5*k_blocks + b.
            let e5 = unsafe { *d8.get_unchecked(d8b0 + 5 * k_blocks + b) };
            let s5 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + 5 * k_blocks + b) }
            } else {
                0
            };
            f5 += q5_block_col(&cw, q, q0 + 5 * q_stride + qslot, d, mv * s5 as f32, e5);
        }
        if m_cols > 6 {
            // SAFETY: m_cols > 6 => d8.len() >= (col0+7)*k_blocks >
            // d8b0 + 6*k_blocks + b.
            let e6 = unsafe { *d8.get_unchecked(d8b0 + 6 * k_blocks + b) };
            let s6 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + 6 * k_blocks + b) }
            } else {
                0
            };
            f6 += q5_block_col(&cw, q, q0 + 6 * q_stride + qslot, d, mv * s6 as f32, e6);
        }
        if m_cols > 7 {
            // SAFETY: m_cols > 7 => d8.len() >= (col0+8)*k_blocks >
            // d8b0 + 7*k_blocks + b.
            let e7 = unsafe { *d8.get_unchecked(d8b0 + 7 * k_blocks + b) };
            let s7 = if q5_1 {
                // SAFETY: the same index in the parallel s8 buffer, which is
                // as long as d8 when q5_1.
                unsafe { *s8.get_unchecked(d8b0 + 7 * k_blocks + b) }
            } else {
                0
            };
            f7 += q5_block_col(&cw, q, q0 + 7 * q_stride + qslot, d, mv * s7 as f32, e7);
        }

        b += 32;
    }

    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// One warp's group of four 32-value quant blocks of column `col`: the body
/// both [`q5_kernels::q5_quantize_q8`] and the merged pair kernel run, so
/// neither can drift from the other. Lane ℓ owns the values of word ℓ
/// (values `128g + 4ℓ .. +3`), its octet `ℓ>>3` is one 32-value block, and
/// the block's scale and byte sum are 1-2-4 xor butterflies inside that
/// octet — masks that never cross an octet, so a partial last group is safe.
/// Lanes past the last block read the final block's values (full-warp
/// shuffles, in bounds) and store nothing.
///
/// Lives outside the `#[cuda_module]` for the reason the cores above do: a
/// device-callable body shared by two kernels. It takes the module's
/// `DisjointSlice` outputs, which no core does.
///
/// SAFETY: the caller guarantees `col < m_cols`, `g < n_groups`, the launch
/// contract's bounds on `x`, `q`, `s8` and `d8`, and that all 32 lanes of
/// one warp enter with the same `(col, g)` — the shuffles below are
/// warp-wide.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) unsafe fn q5_quant_group(
    x: &[f32],
    col: usize,
    g: usize,
    k_blocks: u32,
    q_stride: u32,
    lane: usize,
    q: &mut DisjointSlice<u32>,
    s8: &mut DisjointSlice<i32>,
    d8: &mut DisjointSlice<f32>,
) {
    let oct = lane >> 3; // block within the group (0..4)
    let i = lane & 7; // word within the block (0..8)
    let b = 4 * g + oct;
    let active = b < k_blocks as usize;
    // Inactive lanes of a partial last group read the LAST block's values
    // so every shuffle below runs on a full warp; they store nothing.
    let bread = if active { b } else { k_blocks as usize - 1 };
    let base = col * k_blocks as usize * 32 + 32 * bread + 4 * i;
    // SAFETY: base + 3 < (col+1)*32*k_blocks <= m_cols*k_blocks*32 <=
    // x.len() by the caller's contract.
    let (v0, v1, v2, v3) = unsafe {
        (
            *x.get_unchecked(base),
            *x.get_unchecked(base + 1),
            *x.get_unchecked(base + 2),
            *x.get_unchecked(base + 3),
        )
    };
    // Octet amax (masks 1, 2, 4 stay inside the octet).
    let lmax = v0.abs().max(v1.abs()).max(v2.abs()).max(v3.abs());
    let mut amax = lmax;
    amax = amax.max(warp::shuffle_xor_f32(amax, 1));
    amax = amax.max(warp::shuffle_xor_f32(amax, 2));
    amax = amax.max(warp::shuffle_xor_f32(amax, 4));
    let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let q0 = ((v0 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let q1 = ((v1 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let q2 = ((v2 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let q3 = ((v3 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let word = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
    // Octet block sum (the same masks), so lane&7 == 0 holds the total.
    let mut s = (q0 as i8 as i32) + (q1 as i8 as i32) + (q2 as i8 as i32) + (q3 as i8 as i32);
    s += warp::shuffle_xor(s as u32, 1) as i32;
    s += warp::shuffle_xor(s as u32, 2) as i32;
    s += warp::shuffle_xor(s as u32, 4) as i32;
    if active {
        // SAFETY: b < k_blocks, so 256*(b>>5) + 32*i + (b&31) + 0 (i < 8)
        // < 256*((b>>5)+1) <= q_stride, hence slot < (col+1)*q_stride <=
        // m_cols*q_stride <= q.len(); and col*k_blocks + b < m_cols*k_blocks
        // <= s8.len() and d8.len(). One lane writes each slot.
        unsafe {
            *q.get_unchecked_mut(col * q_stride as usize + 256 * (b >> 5) + 32 * i + (b & 31)) =
                word;
            if i == 0 {
                *s8.get_unchecked_mut(col * k_blocks as usize + b) = s;
                *d8.get_unchecked_mut(col * k_blocks as usize + b) = d;
            }
        }
    }
}

// ----------------------------------------------------------------- module

#[cuda_module]
mod q5_kernels {
    use super::*;

    /// Quantize f32 activations to the straight q8_1 geometry the q5 gemvs
    /// read: per 32-value block, f32 d = amax/127 (1.0 for an all-zero
    /// block) and int8 q = round(x/d) clamped to ±127, one u32 word per four
    /// consecutive values, plus the block's f32 scale and i32 byte sum s (the
    /// Q5_1 offset term's Σq). One warp per (column, group of FOUR blocks —
    /// 128 values = 32 words): lane ℓ owns the values of word ℓ (values
    /// 128g + 4ℓ .. +3), its octet ℓ>>3 is one 32-value block, and the word
    /// lands in the gemv's transposed slot `256*(b>>5) + 32*(ℓ&7) + (b&31)`,
    /// so a gemv warp's eight q8 loads are one 128 B line each. Block scale
    /// and sum are 1-2-4 xor butterflies inside the octet (the masks never
    /// cross an octet, so a partial last group is safe). Lanes past the last
    /// block of a partial group read the final block's values (full-warp
    /// shuffles, in-bounds) and store nothing. The window padding past
    /// `8*k_blocks` words per column is never written or read.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= m_cols * k_blocks * 32,
            q.len() >= m_cols * q_stride,
            s8.len() >= m_cols * k_blocks,
            d8.len() >= m_cols * k_blocks
        )
    )]
    pub fn q5_quantize_q8(
        x: &[f32],
        m_cols: u32,
        n_groups: u32,
        k_blocks: u32,
        q_stride: u32,
        mut q: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
    ) {
        // One 32-thread block per four 32-value quant blocks: grp is the
        // GROUP index (global tid / 32), not the thread id.
        let grp = thread::index_1d().get() / 32;
        let total = m_cols as usize * n_groups as usize;
        if grp >= total {
            return;
        }
        let col = grp / n_groups as usize;
        let g = grp % n_groups as usize;
        let lane = warp::lane_id() as usize;
        // SAFETY: col < m_cols and g < n_groups by the two lines above; the
        // launch contract carries the rest of `q5_quant_group`'s
        // preconditions, and the group index is warp-uniform.
        unsafe {
            q5_quant_group(
                x, col, g, k_blocks, q_stride, lane, &mut q, &mut s8, &mut d8,
            )
        }
    }

    /// The two quantizations of the MoE half in one launch: blocks below
    /// `total_a` run [`q5_quant_group`] over `xa` (the routed experts'
    /// 32-value form), the rest [`q8_1_quant_block`] over `xb` (the shared
    /// expert's q8_1). Both read only their own source and write only their
    /// own outputs, and the two sources do not alias, so this is one launch
    /// where there were two — not a change of work.
    ///
    /// The arm is a block-uniform branch and each body is the one its own
    /// kernel calls, so each output is the bytes its own launch would have
    /// written. The two geometries differ (a q5 group is four 32-value
    /// blocks, a q8_1 block is 128 values) and stay separate: nothing is
    /// reduced across the arm.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            xa.len() >= m_a * k_blocks * 32,
            qa.len() >= m_a * q_stride,
            s8a.len() >= m_a * k_blocks,
            d8a.len() >= m_a * k_blocks,
            xb.len() >= m_b * 256 * n_sb,
            q3b.len() >= m_b * 64 * half_it,
            q4b.len() >= m_b * 256 * quad_it,
            q6b.len() >= m_b * 128 * half_it,
            s8b.len() >= m_b * 8 * n_sb,
            d8b.len() >= m_b * 2 * n_sb
        )
    )]
    pub fn q5_q8_1_quantize_pair(
        xa: &[f32],
        xb: &[f32],
        m_a: u32,
        n_groups: u32,
        k_blocks: u32,
        q_stride: u32,
        m_b: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut qa: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
    ) {
        let blk = thread::index_1d().get() / 32;
        let total_a = m_a as usize * n_groups as usize;
        let n_sb = n_sb as usize;
        let total_b = m_b as usize * 2 * n_sb;
        if blk >= total_a + total_b {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: the arm's index is inside its own total, so
        // the column and group/block indices below are in range, and the
        // launch contract bounds that arm's source and outputs. The arm is
        // chosen by the block index, so a warp never splits across it.
        unsafe {
            if blk < total_a {
                let (col, g) = (blk / n_groups as usize, blk % n_groups as usize);
                q5_quant_group(
                    xa, col, g, k_blocks, q_stride, lane, &mut qa, &mut s8a, &mut d8a,
                );
            } else {
                let k = blk - total_a;
                let (col, b) = (k / (2 * n_sb), k % (2 * n_sb));
                q8_1_quant_block(
                    xb, 0, col, b, n_sb, half_it, quad_it, lane, &mut q3b, &mut q4b, &mut q6b,
                    &mut s8b, &mut d8b,
                );
            }
        }
    }

    /// Q5_0 gemv (one warp per row, eight warps per 256-thread block):
    /// `n_rows` rows of the flat stacked weight tensor starting at absolute
    /// row `row0`, against activation columns `col0..col0+m_cols` of the q8
    /// scratch, writing `y[y0 + r*m_cols + c]` (r the local row). Layouts and
    /// bounds: `q5_row_dot`, whose contract this kernel's launch contract
    /// states over its own parameters (`row_words = q_stride + k_blocks`).
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
            w.len() >= (row0 + n_rows) * (q_stride + k_blocks),
            q.len() >= (col0 + m_cols) * q_stride,
            d8.len() >= (col0 + m_cols) * k_blocks,
            s8.len() >= (col0 + m_cols) * k_blocks,
            y.len() >= y0 + n_rows * m_cols
        )
    )]
    pub fn q5_0_gemv(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        s8: &[i32],
        k_blocks: u32,
        q_stride: u32,
        row0: u32,
        n_rows: u32,
        col0: u32,
        m_cols: u32,
        y0: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let f = q5_row_dot(
            w,
            q,
            d8,
            s8,
            k_blocks as usize,
            q_stride as usize,
            row0 as usize + row,
            col0 as usize,
            m_cols as usize,
            lane,
            false,
        );
        let s = reduce_cols(f, m_cols as usize);
        if lane == 0 {
            let yb = y0 as usize + row * m_cols as usize;
            // SAFETY: yb + c < y0 + n_rows*m_cols <= y.len() by the launch
            // contract; store c is guarded by m > c and lane 0 of each warp
            // writes a disjoint m-slot segment.
            unsafe {
                *y.get_unchecked_mut(yb) = s[0];
                if m_cols > 1 {
                    *y.get_unchecked_mut(yb + 1) = s[1];
                }
                if m_cols > 2 {
                    *y.get_unchecked_mut(yb + 2) = s[2];
                }
                if m_cols > 3 {
                    *y.get_unchecked_mut(yb + 3) = s[3];
                }
                if m_cols > 4 {
                    *y.get_unchecked_mut(yb + 4) = s[4];
                }
                if m_cols > 5 {
                    *y.get_unchecked_mut(yb + 5) = s[5];
                }
                if m_cols > 6 {
                    *y.get_unchecked_mut(yb + 6) = s[6];
                }
                if m_cols > 7 {
                    *y.get_unchecked_mut(yb + 7) = s[7];
                }
            }
        }
    }

    /// Q5_1 gemv: same geometry as `q5_0_gemv` over the Q5_1 layout
    /// (`row_words = q_stride + 2*k_blocks`; unsigned codes plus the
    /// `m·(block sum)` offset term). See `q5_row_dot`.
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
            w.len() >= (row0 + n_rows) * (q_stride + 2 * k_blocks),
            q.len() >= (col0 + m_cols) * q_stride,
            d8.len() >= (col0 + m_cols) * k_blocks,
            s8.len() >= (col0 + m_cols) * k_blocks,
            y.len() >= y0 + n_rows * m_cols
        )
    )]
    pub fn q5_1_gemv(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        s8: &[i32],
        k_blocks: u32,
        q_stride: u32,
        row0: u32,
        n_rows: u32,
        col0: u32,
        m_cols: u32,
        y0: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let f = q5_row_dot(
            w,
            q,
            d8,
            s8,
            k_blocks as usize,
            q_stride as usize,
            row0 as usize + row,
            col0 as usize,
            m_cols as usize,
            lane,
            true,
        );
        let s = reduce_cols(f, m_cols as usize);
        if lane == 0 {
            let yb = y0 as usize + row * m_cols as usize;
            // SAFETY: row < n_rows, so the slots written below lie in
            // yb .. yb + m_cols <= y0 + n_rows*m_cols <= y.len(), the launch
            // contract's bound; only lane 0 of the row's warp writes.
            unsafe {
                *y.get_unchecked_mut(yb) = s[0];
                if m_cols > 1 {
                    *y.get_unchecked_mut(yb + 1) = s[1];
                }
                if m_cols > 2 {
                    *y.get_unchecked_mut(yb + 2) = s[2];
                }
                if m_cols > 3 {
                    *y.get_unchecked_mut(yb + 3) = s[3];
                }
                if m_cols > 4 {
                    *y.get_unchecked_mut(yb + 4) = s[4];
                }
                if m_cols > 5 {
                    *y.get_unchecked_mut(yb + 5) = s[5];
                }
                if m_cols > 6 {
                    *y.get_unchecked_mut(yb + 6) = s[6];
                }
                if m_cols > 7 {
                    *y.get_unchecked_mut(yb + 7) = s[7];
                }
            }
        }
    }

    /// Q5_0 gemv over expert slots selected on the device (the MoE down
    /// projection, decode shape): one launch computes `n_slots` experts of
    /// the resident flat stack — `n_experts * rows_per_expert` rows of
    /// `q_stride + k_blocks` words — where slot s reads activation column
    /// `s` of the q8 scratch (each expert's down input differs) and writes
    /// `y[s*rows_per_expert + r]`; m_cols = 1 per slot. The selected ids
    /// are read from `sel`, a device buffer, so the launch is addressable
    /// from inside a captured graph whose replay consumes whatever a
    /// router kernel last wrote there. The per-row body is `q5_row_dot`
    /// (col0 = slot, m_cols = 1), bit-identical to `q5_0_gemv` run with
    /// row0 = sel[s]*rows_per_expert, col0 = s.
    ///
    /// An id >= n_experts cannot be rejected by the host contract (it
    /// lives in device memory): the slot's warps return before their first
    /// load — warp-uniform, no divergent branch — leaving that slot of `y`
    /// untouched and every other slot unaffected.
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
            w.len() >= n_experts * rows_per_expert * (q_stride + k_blocks),
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * k_blocks,
            s8.len() >= n_slots * k_blocks,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q5_0_gemv_sel(
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
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the out-of-range return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let f = q5_row_dot(
            w,
            q,
            d8,
            s8,
            k_blocks as usize,
            q_stride as usize,
            id * rows_per_expert as usize + row % rows_per_expert as usize,
            slot, // col0: slot s reads activation column s
            1,    // m_cols: one output per slot row
            lane,
            false,
        );
        let s = reduce_cols(f, 1);
        if lane == 0 {
            // SAFETY: row < n_slots*rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s[0];
            }
        }
    }

    /// Warp-uniform reduction of the eight accumulators. m_cols is a
    /// launch-wide constant, so every lane takes the same branch and the
    /// shuffles stay warp-collective; inactive columns keep their zero
    /// accumulator. Callers must invoke this from all 32 lanes of one warp.
    fn reduce_cols(f: [f32; 8], m_cols: usize) -> [f32; 8] {
        let s0 = warp::reduce_sum_f32(f[0]);
        let s1 = if m_cols > 1 {
            warp::reduce_sum_f32(f[1])
        } else {
            0.0
        };
        let s2 = if m_cols > 2 {
            warp::reduce_sum_f32(f[2])
        } else {
            0.0
        };
        let s3 = if m_cols > 3 {
            warp::reduce_sum_f32(f[3])
        } else {
            0.0
        };
        let s4 = if m_cols > 4 {
            warp::reduce_sum_f32(f[4])
        } else {
            0.0
        };
        let s5 = if m_cols > 5 {
            warp::reduce_sum_f32(f[5])
        } else {
            0.0
        };
        let s6 = if m_cols > 6 {
            warp::reduce_sum_f32(f[6])
        } else {
            0.0
        };
        let s7 = if m_cols > 7 {
            warp::reduce_sum_f32(f[7])
        } else {
            0.0
        };
        [s0, s1, s2, s3, s4, s5, s6, s7]
    }
}

// ------------------------------------------------------------------- host

/// q8_1 activation scratch for up to `m` columns of `k` values (k a multiple
/// of 32), in the straight 32-value-block geometry the q5 gemvs read — not
/// `Q8Act`, whose layout is permuted for the K-quant lanes. One set per
/// distinct input site.
///
/// Buffer sizes (the quantizer's launch contract verbatim): per column
/// `q_stride = 256*ceil(k_blocks/32)` u32 words (256 per 32-block window; the
/// padding past `8*k_blocks` is never touched), `k_blocks = k/32` i32 block
/// sums and `k_blocks` f32 block scales.
pub struct Q8Blocks32 {
    pub(crate) q: DeviceBuffer<u32>,
    pub(crate) s8: DeviceBuffer<i32>,
    pub(crate) d8: DeviceBuffer<f32>,
    m: usize,
    k: usize,
    q_stride: usize,
}

impl Q8Blocks32 {
    /// Allocate for `m` (1..=8) columns of `k` values. Load-time only.
    pub fn new(stream: &CudaStream, k: usize, m: usize) -> Result<Self, GpuError> {
        if k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                "Q8Blocks32::new",
                format!("k must be a positive multiple of 32, got {k}"),
            ));
        }
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(
                "Q8Blocks32::new",
                format!("1 <= m <= 8, got {m}"),
            ));
        }
        let k_blocks = k / 32;
        let q_stride = 256 * k_blocks.div_ceil(32);
        Ok(Q8Blocks32 {
            q: DeviceBuffer::zeroed(stream, q_stride * m)?,
            s8: DeviceBuffer::zeroed(stream, k_blocks * m)?,
            d8: DeviceBuffer::zeroed(stream, k_blocks * m)?,
            m,
            k,
            q_stride,
        })
    }

    /// Quantized columns available (the allocation width).
    pub fn m(&self) -> usize {
        self.m
    }

    /// Values per column.
    pub fn k(&self) -> usize {
        self.k
    }

    /// u32 words per column of `q` (window-padded; see the struct doc).
    pub fn q_stride(&self) -> usize {
        self.q_stride
    }

    /// Read the three buffers back to the host — a gate's comparison between
    /// two paths (the fields are crate-private). Diagnostic readback:
    /// synchronizes `stream`, so load-time/gate use only, never inside a
    /// graph capture.
    pub fn readback(&self, stream: &CudaStream) -> Result<Q8Blocks32Host, GpuError> {
        Ok(Q8Blocks32Host {
            q: self.q.to_host_vec(stream)?,
            s8: self.s8.to_host_vec(stream)?,
            d8: self.d8.to_host_vec(stream)?,
        })
    }
}

/// The three buffers of a `Q8Blocks32` on the host, as `readback` returns
/// them.
pub struct Q8Blocks32Host {
    pub q: Vec<u32>,
    pub s8: Vec<i32>,
    pub d8: Vec<f32>,
}

/// Repack `rows` rows of raw gguf Q5_0 bytes (22 B per 32-value block: d f16,
/// qh u32 at +2, 16 qs bytes at +6) into the gemv weight layout: per row
/// `q_stride + k/32` u32 words — the code section (byte v of block b's word i
/// is the 5-bit code of value `32b + 4i + v` minus 16, so the dp4a chain is
/// the exact signed block dot; word i at `256*(b>>5) + 32*i + (b&31)`, window
/// padding zero) — then `k/32` f32 block scales d as bits. Load-time cost.
pub fn pack_q5_0(bytes: &[u8], k: usize, rows: usize) -> Result<Vec<u32>, GpuError> {
    pack_q5(bytes, k, rows, false)
}

/// Repack `rows` rows of raw gguf Q5_1 bytes (24 B per block: d f16, m f16,
/// qh u32 at +4, 16 qs bytes at +8) into the gemv weight layout: per row
/// `q_stride + 2*k/32` words — the code section as `pack_q5_0` but with the
/// unsigned 0..31 code (value = q5·d + m), then `k/32` f32 scales d, then
/// `k/32` f32 mins m. The gemv adds `m·(block sum)` as the offset term.
pub fn pack_q5_1(bytes: &[u8], k: usize, rows: usize) -> Result<Vec<u32>, GpuError> {
    pack_q5(bytes, k, rows, true)
}

fn pack_q5(bytes: &[u8], k: usize, rows: usize, q5_1: bool) -> Result<Vec<u32>, GpuError> {
    if k == 0 || !k.is_multiple_of(32) {
        return Err(GpuError::shape(
            "pack_q5",
            format!("k must be a positive multiple of 32, got {k}"),
        ));
    }
    let k_blocks = k / 32;
    let blk_bytes = if q5_1 { 24 } else { 22 };
    let need = k_blocks * blk_bytes * rows;
    if bytes.len() < need {
        return Err(GpuError::shape(
            "pack_q5",
            format!(
                "bytes.len() {len} < k/32 * block * rows = {need}",
                len = bytes.len()
            ),
        ));
    }
    let q_stride = 256 * k_blocks.div_ceil(32);
    let scale_words = if q5_1 { 2 * k_blocks } else { k_blocks };
    let mut out = vec![0u32; (q_stride + scale_words) * rows];
    for r in 0..rows {
        let row = &mut out[r * (q_stride + scale_words)..(r + 1) * (q_stride + scale_words)];
        for b in 0..k_blocks {
            let blk = &bytes[(r * k_blocks + b) * blk_bytes..][..blk_bytes];
            let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let (qh_off, qs_off) = if q5_1 { (4, 8) } else { (2, 6) };
            let qh = u32::from_le_bytes([
                blk[qh_off],
                blk[qh_off + 1],
                blk[qh_off + 2],
                blk[qh_off + 3],
            ]);
            let m = if q5_1 {
                half_to_f32(u16::from_le_bytes([blk[2], blk[3]]))
            } else {
                0.0
            };
            row[q_stride + b] = d.to_bits();
            if q5_1 {
                row[q_stride + k_blocks + b] = m.to_bits();
            }
            let wbase = 256 * (b >> 5) + (b & 31);
            for j in 0..16usize {
                let qs = blk[qs_off + j];
                // The fifth bit of value j is qh bit j, of value 16+j qh bit
                // 16+j (ggml's xh_0/xh_1 shifts, scalar form).
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let lo = (qs & 0x0f) as i32 | xh0 as i32;
                let hi = (qs >> 4) as i32 | xh1 as i32;
                // Q5_0 folds the −16 into the byte (two's complement); Q5_1
                // keeps the unsigned code.
                let (lo_b, hi_b) = if q5_1 {
                    (lo as u32, hi as u32)
                } else {
                    (((lo - 16) as u32) & 0xff, ((hi - 16) as u32) & 0xff)
                };
                let sh = 8 * (j % 4) as u32;
                row[wbase + 32 * (j / 4)] |= lo_b << sh;
                row[wbase + 32 * (j / 4 + 4)] |= hi_b << sh;
            }
        }
    }
    Ok(out)
}

/// The loaded q5 device module and its enqueue API. Every enqueue is
/// asynchronous, allocation-free and capturable (the engine stream; never the
/// null stream).
pub struct Q5Kernels {
    module: q5_kernels::LoadedModule,
}

impl Q5Kernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q5Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { q5_kernels::load(ctx)? };
        Ok(Q5Kernels { module })
    }

    /// Enqueue the q8 quantization of `x` — `act.m()` columns of `act.k()`
    /// f32, concatenated — into `act`.
    pub fn enqueue_quantize_q8(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Blocks32,
    ) -> Result<(), GpuError> {
        let (m, k_blocks, q_stride) = (act.m(), act.k() / 32, act.q_stride());
        if x.len() < m * k_blocks * 32 {
            return Err(GpuError::shape(
                "enqueue_quantize_q8",
                format!(
                    "x.len() {len} < m*k = {need}",
                    len = x.len(),
                    need = m * k_blocks * 32
                ),
            ));
        }
        let n_groups = k_blocks.div_ceil(4);
        let prep = self.module.prepare_q5_quantize_q8(LaunchConfig1D::new(
            (m * n_groups) as u32,
            32,
            0,
        ))?;
        self.module.q5_quantize_q8(
            stream,
            &prep,
            x,
            m as u32,
            n_groups as u32,
            k_blocks as u32,
            q_stride as u32,
            &mut act.q,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }

    /// Both MoE quantizations in one launch: `xa` into the routed experts'
    /// 32-value scratch `a`, `xb` into the shared expert's q8_1 scratch `b`.
    /// The bytes are the ones `enqueue_quantize_q8(xa, a)` and
    /// `Gpu::enqueue_quantize_q8_1(xb, b)` write, so the two sources must
    /// not alias and the caller must have enqueued both producers first.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_quantize_q8_pair(
        &self,
        stream: &CudaStream,
        xa: &DeviceBuffer<f32>,
        a: &mut Q8Blocks32,
        xb: &DeviceBuffer<f32>,
        b: &mut crate::tensor::Q8Act,
    ) -> Result<(), GpuError> {
        let (m_a, k_blocks, q_stride) = (a.m(), a.k() / 32, a.q_stride());
        if xa.len() < m_a * k_blocks * 32 {
            return Err(GpuError::shape(
                "enqueue_quantize_q8_pair",
                format!(
                    "xa.len() {len} < m*k = {need}",
                    len = xa.len(),
                    need = m_a * k_blocks * 32
                ),
            ));
        }
        let (m_b, n_sb) = (b.m(), b.n_sb());
        if xb.len() < m_b * b.k() {
            return Err(GpuError::shape(
                "enqueue_quantize_q8_pair",
                format!(
                    "xb.len() {len} < m*k = {need}",
                    len = xb.len(),
                    need = m_b * b.k()
                ),
            ));
        }
        let n_groups = k_blocks.div_ceil(4);
        let blocks = m_a * n_groups + m_b * 2 * n_sb;
        let prep = self
            .module
            .prepare_q5_q8_1_quantize_pair(LaunchConfig1D::new(blocks as u32, 32, 0))?;
        self.module.q5_q8_1_quantize_pair(
            stream,
            &prep,
            xa,
            xb,
            m_a as u32,
            n_groups as u32,
            k_blocks as u32,
            q_stride as u32,
            m_b as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            n_sb.div_ceil(4) as u32,
            &mut a.q,
            &mut a.s8,
            &mut a.d8,
            &mut b.q3,
            &mut b.q4,
            &mut b.q6,
            &mut b.s8,
            &mut b.d8,
        )?;
        Ok(())
    }

    /// Enqueue `y[y0 + r*m_cols + c] = w_row(row0 + r) · act column
    /// (col0 + c)` for Q5_0 weights packed by `pack_q5_0` — `w` uploaded
    /// with `cols = q_stride + k/32` over the whole flat stack, so experts
    /// are reached by `row0` without a gather copy.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_gemv_q5_0(
        &self,
        stream: &CudaStream,
        w: &crate::DeviceTensor<u32>,
        act: &Q8Blocks32,
        row0: usize,
        n_rows: usize,
        col0: usize,
        m_cols: usize,
        y: &mut DeviceBuffer<f32>,
        y0: usize,
    ) -> Result<(), GpuError> {
        let k_blocks = act.k() / 32;
        if w.cols() != act.q_stride() + k_blocks {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0",
                format!(
                    "Q5_0 row is q_stride + k/32 = {} words, got cols {}",
                    act.q_stride() + k_blocks,
                    w.cols()
                ),
            ));
        }
        if w.rows() < row0 + n_rows || n_rows == 0 {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0",
                format!(
                    "rows {rows} < row0 {row0} + n_rows {n_rows} or n_rows == 0",
                    rows = w.rows()
                ),
            ));
        }
        if !(1..=8).contains(&m_cols) || col0 + m_cols > act.m() {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0",
                format!(
                    "need 1 <= m_cols <= 8 and col0 + m_cols <= {}, got col0 {col0} m_cols {m_cols}",
                    act.m()
                ),
            ));
        }
        if y.len() < y0 + n_rows * m_cols {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0",
                format!(
                    "y.len() {len} < y0 + n_rows*m_cols = {need}",
                    len = y.len(),
                    need = y0 + n_rows * m_cols
                ),
            ));
        }
        let prep = self.module.prepare_q5_0_gemv(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.q5_0_gemv(
            stream,
            &prep,
            w.buf(),
            &act.q,
            &act.d8,
            &act.s8,
            k_blocks as u32,
            act.q_stride() as u32,
            row0 as u32,
            n_rows as u32,
            col0 as u32,
            m_cols as u32,
            y0 as u32,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the device-indirect MoE down projection for Q5_0: one launch
    /// computes `n_slots` experts, slot s writing
    /// `y[s*rows_per_expert .. (s+1)*rows_per_expert]` as
    /// `w[sel[s]*rows_per_expert .. +rows_per_expert] · act` column `s` (m
    /// per slot is 1; `act` holds at least `n_slots` quantized columns).
    /// `w` is the full resident stack packed by `pack_q5_0`,
    /// `cols = q_stride + k/32`, `rows` a positive multiple of
    /// `rows_per_expert` (`n_experts = rows/rows_per_expert`). `sel` is a
    /// device buffer of at least `n_slots` ids read by the kernel per
    /// launch, so a captured graph replay picks up new ids written between
    /// replays; an id >= n_experts leaves that slot of `y` untouched.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q5_0_sel(
        &self,
        stream: &CudaStream,
        w: &crate::DeviceTensor<u32>,
        act: &Q8Blocks32,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let k_blocks = act.k() / 32;
        if w.cols() != act.q_stride() + k_blocks {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0_sel",
                format!(
                    "Q5_0 row is q_stride + k/32 = {} words, got cols {}",
                    act.q_stride() + k_blocks,
                    w.cols()
                ),
            ));
        }
        if rows_per_expert == 0 || !w.rows().is_multiple_of(rows_per_expert) {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0_sel",
                format!(
                    "w.rows() {} must be a positive multiple of \
                 rows_per_expert {rows_per_expert}",
                    w.rows()
                ),
            ));
        }
        if n_slots == 0 || n_slots > act.m() || sel.len() < n_slots {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0_sel",
                format!(
                    "need 1 <= n_slots <= act.m() = {} and sel.len() >= \
                 n_slots, got n_slots {n_slots} sel.len() {}",
                    act.m(),
                    sel.len()
                ),
            ));
        }
        if y.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_0_sel",
                format!(
                    "y.len() {len} < n_slots*rows_per_expert = {need}",
                    len = y.len(),
                    need = n_slots * rows_per_expert
                ),
            ));
        }
        let n_experts = w.rows() / rows_per_expert;
        let prep = self.module.prepare_q5_0_gemv_sel(LaunchConfig1D::new(
            (n_slots * rows_per_expert).div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.q5_0_gemv_sel(
            stream,
            &prep,
            w.buf(),
            &act.q,
            &act.d8,
            &act.s8,
            sel,
            k_blocks as u32,
            act.q_stride() as u32,
            n_experts as u32,
            rows_per_expert as u32,
            n_slots as u32,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the Q5_1 counterpart of `enqueue_gemv_q5_0` — `w` packed by
    /// `pack_q5_1` with `cols = q_stride + 2*k/32`.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_gemv_q5_1(
        &self,
        stream: &CudaStream,
        w: &crate::DeviceTensor<u32>,
        act: &Q8Blocks32,
        row0: usize,
        n_rows: usize,
        col0: usize,
        m_cols: usize,
        y: &mut DeviceBuffer<f32>,
        y0: usize,
    ) -> Result<(), GpuError> {
        let k_blocks = act.k() / 32;
        if w.cols() != act.q_stride() + 2 * k_blocks {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_1",
                format!(
                    "Q5_1 row is q_stride + 2*k/32 = {} words, got cols {}",
                    act.q_stride() + 2 * k_blocks,
                    w.cols()
                ),
            ));
        }
        if w.rows() < row0 + n_rows || n_rows == 0 {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_1",
                format!(
                    "rows {rows} < row0 {row0} + n_rows {n_rows} or n_rows == 0",
                    rows = w.rows()
                ),
            ));
        }
        if !(1..=8).contains(&m_cols) || col0 + m_cols > act.m() {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_1",
                format!(
                    "need 1 <= m_cols <= 8 and col0 + m_cols <= {}, got col0 {col0} m_cols {m_cols}",
                    act.m()
                ),
            ));
        }
        if y.len() < y0 + n_rows * m_cols {
            return Err(GpuError::shape(
                "enqueue_gemv_q5_1",
                format!(
                    "y.len() {len} < y0 + n_rows*m_cols = {need}",
                    len = y.len(),
                    need = y0 + n_rows * m_cols
                ),
            ));
        }
        let prep = self.module.prepare_q5_1_gemv(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.q5_1_gemv(
            stream,
            &prep,
            w.buf(),
            &act.q,
            &act.d8,
            &act.s8,
            k_blocks as u32,
            act.q_stride() as u32,
            row0 as u32,
            n_rows as u32,
            col0 as u32,
            m_cols as u32,
            y0 as u32,
            y,
        )?;
        Ok(())
    }
}
