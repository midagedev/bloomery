//! bloomery-gpu — the stage-0 CUDA kernels packaged as a library.
//!
//! The kernels are the algorithms of `crates/q3k-gemv/src/main.rs` with K
//! (values per activation column) as a launch argument instead of the
//! stage-0 constant 2048; the arithmetic bodies live in `cores::` as
//! device-callable functions (docs/gpu-design.md decision 6). The crate
//! must be compiled with `cargo oxide`; the codegen backend embeds the
//! compiled device bundle in a `.oxart` member of this crate's rlib, and
//! the anchor reference emitted inside `kernels::load` pulls that member
//! into any final binary that calls this API.
//!
//! K enters the kernels as `n_sb` (super-blocks per row, K = 256·n_sb) plus
//! the iteration counts the q8 scratch layout needs (`half_it` =
//! ceil(n_sb/2) two-super-block groups, `quad_it` = ceil(n_sb/4)
//! four-super-block groups); `Q8Act` owns the pairing between them. The
//! launch contracts bind every buffer length as products of those scalars —
//! the grammar has no division, so the ceils travel as arguments.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

pub mod cores;
pub mod graph;
pub mod model;
pub mod probe;
pub mod q5;
pub mod q8f32;
pub mod tensor;

pub use graph::Graph;
pub use model::{GpuModel, mla_width};
pub use tensor::{DeviceTensor, Q8Act};

/// Host-side failure: context creation, module loading, device allocation,
/// launch, capture, or copy-back.
pub type GpuError = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;
    use crate::cores::{
        funnel16, half_to_f32, q3_slot, q3k_aux_scales, q3k_chain, q3k_dequant, q3k_sub_scale,
        q4_slot, q4k_a_chain, q4k_coeff, q4k_nibble, q4k_scale_min, q6_slot, q6k_chain,
        q6k_dequant, q6k_sub_scale, q8_quad,
    };

    /// Q3_K packing recap (decode verified against ggml's
    /// `dequantize_row_q3_K` in the round-1 kernel at 1e-7):
    /// super-block = hmask[32] @ +0, qs[64] @ +32, scales[12] @ +96, d @ +108.
    /// Weight k within the super-block (k = 128c+32j+16h+8p+o) reads its low
    /// 2 bits from qs byte 32c+16h+8p+o, field j, and its high bit from
    /// hmask byte (k mod 32), bit 4c+j. Inverting: qs word w (bytes 4w..4w+3),
    /// field j, covers the four CONSECUTIVE weights
    /// k = 128*(w/8) + 32*j + 4*(w%8) + b — one dp4a per (word, field)
    /// against u32 word (k/4) of the q8_1 activation, with the sub-block
    /// scale (16 weights = one field-group of the word) applied per dp4a.

    /// Quantize f32 activations to q8_1: per 128-value block, d = amax/127 and
    /// int8 q = round(x/d). One warp per block; each lane quantizes its four
    /// consecutive values and writes the packed u32 word once per gemv lane
    /// geometry — the same bytes in the permutation that makes its format's
    /// gemv load instruction address 32 lane-consecutive words (one 128B L1
    /// line) instead of four strided clusters (four lines, four wavefronts
    /// per load — the M>1 marginal cost MUL-8 chased; the index identity
    /// p(old_slot) == new_slot is host-verified for every load site of all
    /// three formats, so the values are bit-identical to the single linear
    /// store). Q3_K goes one step further: its gemv reads the four words of
    /// an iteration as TWO u64 loads (field pairs share a slot; the pairing
    /// partner is always quantize lane lane^8 of the same block, so one
    /// shuffle builds the u64 — the load-issue count was the residual
    /// per-column cost after the permutation), so q3 is a u64 buffer and
    /// only the bit3-clear lanes store. Also emits s8: the signed-byte sum
    /// of each 32-value group, a 1-2-4 xor butterfly over lane-local quad
    /// sums — exactly the integer the q4k B chain's dp4a(0x01010101, qv)
    /// accumulated, so that chain becomes one i32 load. A 128-value block is
    /// one Q3_K half super-block, the exact span a gemv lane's four fields
    /// cover in one step: all four dp4a share this block's scale, so they
    /// chain in int behind ONE f32 FMA instead of one scale load + FMA per
    /// field.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= m_cols * 256 * n_sb,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn q3k_quantize_q8_1(
        x: &[f32],
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
    ) {
        // One 32-thread block per 128-value quant block: blk is the BLOCK
        // index (global tid / 32), not the thread id.
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb; // 128-value blocks per column
        let total = m_cols as usize * blocks_per_col;
        if blk >= total {
            return;
        }
        let col = blk / blocks_per_col;
        let b = blk % blocks_per_col;
        let lane = warp::lane_id() as usize;

        // Lane covers the four consecutive values 4*lane .. 4*lane+3 of the
        // block; the warp max over the four per-lane maxima is the block
        // amax (no cross-lane byte packing needed, unlike the 32-value
        // geometry where one value per lane forced two shuffle_downs).
        let base = col * 256 * n_sb + 128 * b + 4 * lane;
        let v0 = unsafe { *x.get_unchecked(base) };
        let v1 = unsafe { *x.get_unchecked(base + 1) };
        let v2 = unsafe { *x.get_unchecked(base + 2) };
        let v3 = unsafe { *x.get_unchecked(base + 3) };
        let amax = warp::reduce_max_f32(v0.abs().max(v1.abs()).max(v2.abs()).max(v3.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let (word, quad) = q8_quad([v0, v1, v2, v3], d);

        // v4 = this word's index in value order within the column: the word
        // covers values 128b + 4*lane .. +3, so v4 = 32b + lane (32 words
        // per 128-value block, 2*n_sb blocks per column). Each q*_slot maps
        // it to the load slot of the named gemv geometry — permutations per
        // 2-super-block (q3/q6) or 4-super-block (q4) group, host-verified
        // including this value-span tie (an earlier draft said 16b + lane,
        // which the load-site check alone could not catch because both
        // sides then agree on a wrong bijection over half the column).
        let v4 = (32 * b + lane) as u32;
        // Q3_K u64 pairing: the two fields of a gemv load PAIR (j, j^1) are
        // always held by quantize lanes lane and lane^8 of one block (v4
        // differs only in bit 3), so q3_slot(v4) is a bijection onto the
        // column's 64*half_it u64 slots with v4's bit 3 selecting the half.
        // All lanes run the collective shuffle; the bit3-clear half stores.
        let g3 = q3_slot(v4);
        let partner = warp::shuffle_xor(word, 8);
        let cb = col * 64 * half_it as usize;
        let p4 = q4_slot(v4);
        let p6 = q6_slot(v4);
        let cu4 = col * 256 * quad_it as usize;
        let cu6 = col * 128 * half_it as usize;
        // SAFETY: q3_slot < 64*half_it, q4_slot < 256*quad_it and
        // q6_slot < 128*half_it per column (permutations of the column's
        // value words onto its group slots, host-verified bijections incl.
        // the q3 pair check); the three stores hit three distinct buffers,
        // bit3-clear lanes of a block write disjoint u64 positions, every
        // lane its own u32 position.
        unsafe {
            if lane & 8 == 0 {
                *q3.get_unchecked_mut(cb + g3 as usize) = (word as u64) | ((partner as u64) << 32);
            }
            *q4.get_unchecked_mut(cu4 + p4 as usize) = word;
            *q6.get_unchecked_mut(cu6 + p6 as usize) = word;
        }

        // 32-value-group signed sums: butterfly over the lane-local quad
        // sums (masks 1, 2, 4); afterwards every lane holds its octet's
        // total and lanes 8k write group 4b + k of the column.
        let mut g = quad;
        g += warp::shuffle_xor(g as u32, 1) as i32;
        g += warp::shuffle_xor(g as u32, 2) as i32;
        g += warp::shuffle_xor(g as u32, 4) as i32;
        if lane & 7 == 0 {
            // SAFETY: group index 4b + lane/8 < 8*n_sb per column; s8 holds
            // m_cols*8*n_sb words and one lane writes each group.
            unsafe {
                *s8.get_unchecked_mut(col * 8 * n_sb + 4 * b + (lane >> 3)) = g;
            }
        }
        if lane == 0 {
            // SAFETY: lane 0 of each warp writes its own d8 slot.
            unsafe {
                *d8.get_unchecked_mut(col * 2 * n_sb + b) = d;
            }
        }
    }

    /// Q4_K packing recap (word arithmetic verified against ggml's
    /// `dequantize_row_q4_K` on blk.0.attn_output bytes at 3.6e-8 before this
    /// kernel was written): super-block = d f16 @0, dmin f16 @2, scales[12]
    /// @4, qs[128] @16 — 144 bytes, always 4-aligned, no funnel needed.
    /// Sub-block s (32 consecutive weights, values 32s..32s+32) is nibble
    /// s&1 of qs bytes 32*(s>>1) .. +32: one nibble per byte, the sibling
    /// nibble in the same byte belongs to sub-block s^1. Scale s and min s
    /// unpack from the 12 scale bytes with ggml's get_scale_min_k4.
    /// The min offset forces a second dp4a chain: weight = d*sc*nib -
    /// dmin*mi, so the sub-block dot on quantized activations is
    /// d8*(d*sc*A + (8*d*sc - dmin*mi)*B) with A = dp4a(nib-8, q8) and
    /// B = dp4a(1s, q8) = sum q8 (the -8 offset moves 8*B between the
    /// chains). 16 dp4a cover 32 values — twice Q3_K's density, inherent to
    /// the nibble-sibling layout.

    /// Q4_K (K = 256*n_sb values, n_sb super-blocks per row) times q8_1
    /// activations, M <= 8. One warp per row, guarded scalar accumulators.
    /// The warp covers FOUR super-blocks per iteration (lane L owns the
    /// 32-value sub-block s = L&7 of super-block 4*it + L>>3), so the row
    /// takes iters = ceil(n_sb/4) iterations; when n_sb is not a multiple
    /// of 4 the last iteration's high octets have no super-block and their
    /// lanes contribute nothing (no warp-collective op lives in the loop).
    /// Each lane keeps its whole sub-block local: the min-offset B chain
    /// reduces to one s8 group-sum load per column (the quantize kernel
    /// precomputed the exact integer the per-column dp4a(0x01010101) chain
    /// accumulated), and the qs window is decoded once per iteration into
    /// vi[8] instead of once per column.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 36 * n_sb,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q4k_gemv(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        let row_words = 36 * n_sb; // 144 B per super-block
        let q_col = 256 * iters as usize; // q8 words per column
        let s8_col = 8 * n_sb; // 32-value groups per column
        let d8_col = 2 * n_sb; // 128-value blocks per column

        let s = lane & 7; // sub-block within the super-block
        let grp = lane >> 3; // super-block within this iteration (0..4)

        // Scalar accumulators (launch-uniform m keeps them in registers).
        let mut f0 = 0.0f32;
        let mut f1 = 0.0f32;
        let mut f2 = 0.0f32;
        let mut f3 = 0.0f32;
        let mut f4 = 0.0f32;
        let mut f5 = 0.0f32;
        let mut f6 = 0.0f32;
        let mut f7 = 0.0f32;

        let mut it: u32 = 0;
        while it < iters {
            let sbp = 4 * it as usize + grp;
            // The sbp guard makes a partial final iteration safe: guarded
            // lanes load nothing of w/q/s8/d8. It is always true when n_sb
            // is a multiple of 4.
            if sbp < n_sb {
                let wk = row * row_words + 36 * sbp; // super-block word base

                // d/dmin word + the 12 scale bytes in words 1..3.
                // SAFETY: sbp < n_sb, so wk+3 <= row*row_words +
                // 36*(n_sb-1) + 3 < (row+1)*row_words <= w.len() by the
                // launch contract.
                let (w0, w1, w2, w3) = unsafe {
                    (
                        *w.get_unchecked(wk),
                        *w.get_unchecked(wk + 1),
                        *w.get_unchecked(wk + 2),
                        *w.get_unchecked(wk + 3),
                    )
                };
                let d = half_to_f32((w0 & 0xffff) as u16);
                let dmin = half_to_f32((w0 >> 16) as u16);

                let (sc, mi) = q4k_scale_min(s, w1, w2, w3);
                // Column-independent chain coefficients.
                let (cda, cdb) = q4k_coeff(d, dmin, sc, mi);

                // qs word base: 8 words from super-block word 4 + 8*(s>>1);
                // nibble select is the sub-block parity, 0 (low) or 4 (high).
                // Hoisted (MUL-8): the window and its SWAR decode are
                // column-independent, so decode once per iteration and let
                // every column's A chain reuse the registers.
                let qsk = wk + 4 + 8 * (s >> 1);
                let nib_sh = (s as u32 & 1) * 4;
                // SAFETY: qsk + 7 <= wk + 35, inside the row's word span by
                // the same bound as w0..w3 above.
                let vi = unsafe {
                    [
                        q4k_nibble(*w.get_unchecked(qsk), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 1), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 2), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 3), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 4), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 5), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 6), nib_sh),
                        q4k_nibble(*w.get_unchecked(qsk + 7), nib_sh),
                    ]
                };

                // This lane's q8 words in the q4 permutation: word i of
                // column c lives at q_col*c + 256it + 32i + lane
                // (host-verified identity with the value-order slot the
                // linear layout used). The B-chain group of (it, lane) is
                // 32it + lane: 32 consecutive i32 across the warp, one 128B
                // line.
                let qb = 256 * it as usize + lane;
                let s8b = 32 * it as usize + lane;
                // This lane's fields all sit in q8_1 block 2*sbp + s/4.
                let d8b = 2 * sbp + (s >> 2);

                // Column 0 (always active).
                {
                    let a = q4k_a_chain(&vi, q, qb);
                    // SAFETY: the sbp guard keeps s8b < s8_col and
                    // d8b < d8_col; s8.len() >= m*s8_col and d8.len() >=
                    // m*d8_col by the contract (m >= 1).
                    let b = unsafe { *s8.get_unchecked(s8b) };
                    let e0 = unsafe { *d8.get_unchecked(d8b) };
                    f0 += (a as f32 * cda + b as f32 * cdb) * e0;
                }
                // Columns 1..7, one launch-uniform guard per column so the
                // work scales with m (MUL-8 amortization curve). Column c
                // reads q8 words at q_col*c + qb (q4k_a_chain adds 32i), the
                // s8 group at s8_col*c + s8b and block d8b + d8_col*c.
                // SAFETY: guard c+1 means m >= c+1 is launch-uniform, so the
                // buffer lengths (m*q_col / m*s8_col / m*d8_col) cover every
                // offset below and the branch never diverges within a warp.
                if m > 1 {
                    // SAFETY: m > 1 => q.len() >= 2*q_col > q_col + qb + 224,
                    // s8.len() >= 2*s8_col > s8_col + s8b,
                    // d8.len() >= 2*d8_col > d8b + d8_col.
                    let a = q4k_a_chain(&vi, q, q_col + qb);
                    let b = unsafe { *s8.get_unchecked(s8_col + s8b) };
                    let e1 = unsafe { *d8.get_unchecked(d8b + d8_col) };
                    f1 += (a as f32 * cda + b as f32 * cdb) * e1;
                }
                if m > 2 {
                    // SAFETY: m > 2 => q.len() >= 3*q_col > 2*q_col + qb + 224,
                    // s8.len() >= 3*s8_col > 2*s8_col + s8b,
                    // d8.len() >= 3*d8_col > d8b + 2*d8_col.
                    let a = q4k_a_chain(&vi, q, 2 * q_col + qb);
                    let b = unsafe { *s8.get_unchecked(2 * s8_col + s8b) };
                    let e2 = unsafe { *d8.get_unchecked(d8b + 2 * d8_col) };
                    f2 += (a as f32 * cda + b as f32 * cdb) * e2;
                }
                if m > 3 {
                    // SAFETY: m > 3 => q.len() >= 4*q_col > 3*q_col + qb + 224,
                    // s8.len() >= 4*s8_col > 3*s8_col + s8b,
                    // d8.len() >= 4*d8_col > d8b + 3*d8_col.
                    let a = q4k_a_chain(&vi, q, 3 * q_col + qb);
                    let b = unsafe { *s8.get_unchecked(3 * s8_col + s8b) };
                    let e3 = unsafe { *d8.get_unchecked(d8b + 3 * d8_col) };
                    f3 += (a as f32 * cda + b as f32 * cdb) * e3;
                }
                if m > 4 {
                    // SAFETY: m > 4 => q.len() >= 5*q_col > 4*q_col + qb + 224,
                    // s8.len() >= 5*s8_col > 4*s8_col + s8b,
                    // d8.len() >= 5*d8_col > d8b + 4*d8_col.
                    let a = q4k_a_chain(&vi, q, 4 * q_col + qb);
                    let b = unsafe { *s8.get_unchecked(4 * s8_col + s8b) };
                    let e4 = unsafe { *d8.get_unchecked(d8b + 4 * d8_col) };
                    f4 += (a as f32 * cda + b as f32 * cdb) * e4;
                }
                if m > 5 {
                    // SAFETY: m > 5 => q.len() >= 6*q_col > 5*q_col + qb + 224,
                    // s8.len() >= 6*s8_col > 5*s8_col + s8b,
                    // d8.len() >= 6*d8_col > d8b + 5*d8_col.
                    let a = q4k_a_chain(&vi, q, 5 * q_col + qb);
                    let b = unsafe { *s8.get_unchecked(5 * s8_col + s8b) };
                    let e5 = unsafe { *d8.get_unchecked(d8b + 5 * d8_col) };
                    f5 += (a as f32 * cda + b as f32 * cdb) * e5;
                }
                if m > 6 {
                    // SAFETY: m > 6 => q.len() >= 7*q_col > 6*q_col + qb + 224,
                    // s8.len() >= 7*s8_col > 6*s8_col + s8b,
                    // d8.len() >= 7*d8_col > d8b + 6*d8_col.
                    let a = q4k_a_chain(&vi, q, 6 * q_col + qb);
                    let b = unsafe { *s8.get_unchecked(6 * s8_col + s8b) };
                    let e6 = unsafe { *d8.get_unchecked(d8b + 6 * d8_col) };
                    f6 += (a as f32 * cda + b as f32 * cdb) * e6;
                }
                if m > 7 {
                    // SAFETY: m > 7 => q.len() >= 8*q_col > 7*q_col + qb + 224,
                    // s8.len() >= 8*s8_col > 7*s8_col + s8b,
                    // d8.len() >= 8*d8_col > d8b + 7*d8_col.
                    let a = q4k_a_chain(&vi, q, 7 * q_col + qb);
                    let b = unsafe { *s8.get_unchecked(7 * s8_col + s8b) };
                    let e7 = unsafe { *d8.get_unchecked(d8b + 7 * d8_col) };
                    f7 += (a as f32 * cda + b as f32 * cdb) * e7;
                }
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant). Column c's
        // reduction runs only when m > c; every lane takes the same branch,
        // so the shuffles stay warp-collective, and the tail scales with m
        // exactly like the column bodies above.
        let s0 = warp::reduce_sum_f32(f0);
        if m == 1 {
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
        } else {
            let s1 = warp::reduce_sum_f32(f1);
            let s2 = if m > 2 { warp::reduce_sum_f32(f2) } else { 0.0 };
            let s3 = if m > 3 { warp::reduce_sum_f32(f3) } else { 0.0 };
            let s4 = if m > 4 { warp::reduce_sum_f32(f4) } else { 0.0 };
            let s5 = if m > 5 { warp::reduce_sum_f32(f5) } else { 0.0 };
            let s6 = if m > 6 { warp::reduce_sum_f32(f6) } else { 0.0 };
            let s7 = if m > 7 { warp::reduce_sum_f32(f7) } else { 0.0 };
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m]: store c is guarded by m > c,
                // so exactly the first m slots of the row are touched.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    if m > 1 {
                        *y.get_unchecked_mut(b + 1) = s1;
                    }
                    if m > 2 {
                        *y.get_unchecked_mut(b + 2) = s2;
                    }
                    if m > 3 {
                        *y.get_unchecked_mut(b + 3) = s3;
                    }
                    if m > 4 {
                        *y.get_unchecked_mut(b + 4) = s4;
                    }
                    if m > 5 {
                        *y.get_unchecked_mut(b + 5) = s5;
                    }
                    if m > 6 {
                        *y.get_unchecked_mut(b + 6) = s6;
                    }
                    if m > 7 {
                        *y.get_unchecked_mut(b + 7) = s7;
                    }
                }
            }
        }
    }

    /// Q6_K packing recap (word arithmetic verified against ggml's
    /// `dequantize_row_q6_K` on output.weight bytes at 5.2e-8): super-block
    /// = ql[128] @0, qh[64] @128, scales int8[16] @192, d f16 @208 — 210
    /// bytes, so odd super-blocks sit 2 mod 4 and every window takes the
    /// Q3_K 16-bit funnel. Sub-block s (16 consecutive weights, values
    /// 16s..16s+16, scale byte s signed) reads its low nibble from ql byte
    /// 64*(s>>3) + 16*(s&3) + i (nibble s&4 selects high) and its two high
    /// bits from qh byte 128 + 32*(s>>3) + 16*(s&1) + i, bit pair
    /// 2*((s>>1)&3) — the pair index is per HALF (qh's 2-bit fields repeat
    /// every 32 bytes), which is easy to get wrong. q6 - 32 in [-32,31]
    /// fits a signed byte, so one SWAR subtract folds the offset and the
    /// whole sub-block is a single dp4a chain — no B chain, no min term.

    /// Q6_K (K = 256*n_sb values, n_sb super-blocks per row) times q8_1
    /// activations, M <= 8. Same lane geometry as q3k_gemv: one warp per
    /// row, iteration covers two super-blocks (lanes 0..15 even, 16..31
    /// odd; iters = ceil(n_sb/2), an odd n_sb guards the last iteration's
    /// odd half), lane owns one 16-value sub-block = one scale = four dp4a
    /// per column.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 210 * n_sb,
            q.len() >= m_cols * 128 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q6k_gemv(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        let row_bytes = 210 * n_sb;
        let q_col = 128 * iters as usize; // q8 words per column
        let d8_col = 2 * n_sb; // 128-value blocks per column

        let w16 = lane & 15; // sub-block within the super-block
        let half = lane >> 4; // 0: even super-block, 1: odd

        // Per-lane constants (see the packing comment above).
        let qlok = 16 * (w16 & 3); // ql word base, +16 per word, inside ql
        let qhok = 16 * (w16 & 1); // qh byte base, +16 per word, inside qh
        let nib_sh = ((w16 >> 2) & 1) as u32 * 4; // low nibble vs high
        let hib_sh = 2 * ((w16 >> 1) as u32 & 3); // per-half bit pair

        let mut f0 = 0.0f32;
        let mut f1 = 0.0f32;
        let mut f2 = 0.0f32;
        let mut f3 = 0.0f32;
        let mut f4 = 0.0f32;
        let mut f5 = 0.0f32;
        let mut f6 = 0.0f32;
        let mut f7 = 0.0f32;

        let mut it: u32 = 0;
        while it < iters {
            let sbp = ((it << 1) | half as u32) as usize;
            // The sbp guard makes a partial final iteration safe; always
            // true when n_sb is even.
            if sbp < n_sb {
                let base = row * row_bytes + sbp * 210; // byte offset
                // A super-block sits 0 or 2 mod 4 depending on the row's own
                // offset too (odd n_sb shifts every other row), so the
                // funnel select comes from the byte window, not from sbp.
                let par = (base >> 1) & 1;

                // ql window: 4 words at super-block byte 64*(w16>>3) + qlok.
                let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
                // SAFETY: the last window of the last row ends at byte
                // <= row*row_bytes + row_bytes, and the floored word index
                // stays inside the row's ceil(row_bytes/4) words (see the
                // scales window below for the tight bound).
                let (l0, l1, l2, l3, l4) = unsafe {
                    (
                        *w.get_unchecked(lk),
                        *w.get_unchecked(lk + 1),
                        *w.get_unchecked(lk + 2),
                        *w.get_unchecked(lk + 3),
                        *w.get_unchecked(lk + 4),
                    )
                };
                let ql0 = if par == 0 { l0 } else { funnel16(l0, l1) };
                let ql1 = if par == 0 { l1 } else { funnel16(l1, l2) };
                let ql2 = if par == 0 { l2 } else { funnel16(l2, l3) };
                let ql3 = if par == 0 { l3 } else { funnel16(l3, l4) };

                // qh window: 4 words at super-block byte 128 + 32*(w16>>3) + qhok.
                let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
                // SAFETY: same row bounds as the ql window; the qh section
                // ends at byte 192 of the super-block, before the scales.
                let (h0, h1, h2, h3, h4) = unsafe {
                    (
                        *w.get_unchecked(hk),
                        *w.get_unchecked(hk + 1),
                        *w.get_unchecked(hk + 2),
                        *w.get_unchecked(hk + 3),
                        *w.get_unchecked(hk + 4),
                    )
                };
                let qh0 = if par == 0 { h0 } else { funnel16(h0, h1) };
                let qh1 = if par == 0 { h1 } else { funnel16(h1, h2) };
                let qh2 = if par == 0 { h2 } else { funnel16(h2, h3) };
                let qh3 = if par == 0 { h3 } else { funnel16(h3, h4) };

                // scales (16 int8 at super-block byte 192) + d (f16 @208):
                // one 5-word window, k..k+4. Even: scales are words k..k+3
                // and d is the low half of word k+4. Odd: the window is
                // funneled and d is the HIGH half of word k+4 (the same
                // load f3 uses).
                let ak = (base + 192) >> 2;
                // SAFETY: ak+4 <= ceil((row+1)*row_bytes/4) - 1: for the
                // last super-block of the last row, ak+4 is at most the
                // buffer's final (possibly zero-padded) word; the launch
                // contract bounds 4*w.len() >= n_rows*210*n_sb.
                let (a0, a1, a2, a3, a4) = unsafe {
                    (
                        *w.get_unchecked(ak),
                        *w.get_unchecked(ak + 1),
                        *w.get_unchecked(ak + 2),
                        *w.get_unchecked(ak + 3),
                        *w.get_unchecked(ak + 4),
                    )
                };
                let (sw0, sw1, sw2, sw3, d_bits) = if par == 0 {
                    (a0, a1, a2, a3, (a4 & 0xffff) as u16)
                } else {
                    (
                        funnel16(a0, a1),
                        funnel16(a1, a2),
                        funnel16(a2, a3),
                        funnel16(a3, a4),
                        (a4 >> 16) as u16,
                    )
                };
                // Sub-block scale: byte w16 of the scales window, signed.
                let sc = q6k_sub_scale(&[sw0, sw1, sw2, sw3], w16);
                let drow = half_to_f32(d_bits);

                // q6 - 32 per byte: nibble | high-bits<<4, then one SWAR
                // subtract of 32 (borrow-free with the |0x80 bias).
                let vi = [
                    q6k_dequant(ql0, qh0, nib_sh, hib_sh),
                    q6k_dequant(ql1, qh1, nib_sh, hib_sh),
                    q6k_dequant(ql2, qh2, nib_sh, hib_sh),
                    q6k_dequant(ql3, qh3, nib_sh, hib_sh),
                ];

                // q8_1 words in the q6 permutation: word i of column c lives
                // at q_col*c + 128it + 32i + lane (host-verified identity
                // with the value-order slot the linear layout used); each
                // load is 32 lane-consecutive words across the warp.
                let qb = 128 * it as usize + lane;
                let d8b = 2 * sbp + (w16 >> 3);

                // Column 0 (always active): one dp4a chain, one FMA.
                {
                    // SAFETY: qb + 96 + 3 <= q_col - 1 — this lane's four
                    // q8 words are inside the column's q_col words (the
                    // permutation's group bound); d8b < d8_col by the sbp
                    // guard.
                    let (q0, q1, q2, q3, e0) = unsafe {
                        (
                            *q.get_unchecked(qb),
                            *q.get_unchecked(qb + 32),
                            *q.get_unchecked(qb + 64),
                            *q.get_unchecked(qb + 96),
                            *d8.get_unchecked(d8b),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f0 += (a as f32) * (e0 * drow * sc as f32);
                }
                // Columns 1..7, one launch-uniform guard per column so the
                // work scales with m (MUL-8 amortization curve). Column c
                // reads q8 words at q_col*c + qb (+32 per word) and block
                // d8b + d8_col*c.
                // SAFETY: guard m > c means q.len() >= (c+1)*q_col >
                // q_col*c + qb + 99 and d8.len() >= (c+1)*d8_col >
                // d8b + d8_col*c, launch-uniform.
                if m > 1 {
                    let cb = q_col + qb;
                    let d8c = d8b + d8_col;
                    let (q0, q1, q2, q3, e1) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f1 += (a as f32) * (e1 * drow * sc as f32);
                }
                if m > 2 {
                    let cb = 2 * q_col + qb;
                    let d8c = d8b + 2 * d8_col;
                    let (q0, q1, q2, q3, e2) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f2 += (a as f32) * (e2 * drow * sc as f32);
                }
                if m > 3 {
                    let cb = 3 * q_col + qb;
                    let d8c = d8b + 3 * d8_col;
                    let (q0, q1, q2, q3, e3) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f3 += (a as f32) * (e3 * drow * sc as f32);
                }
                if m > 4 {
                    let cb = 4 * q_col + qb;
                    let d8c = d8b + 4 * d8_col;
                    let (q0, q1, q2, q3, e4) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f4 += (a as f32) * (e4 * drow * sc as f32);
                }
                if m > 5 {
                    let cb = 5 * q_col + qb;
                    let d8c = d8b + 5 * d8_col;
                    let (q0, q1, q2, q3, e5) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f5 += (a as f32) * (e5 * drow * sc as f32);
                }
                if m > 6 {
                    let cb = 6 * q_col + qb;
                    let d8c = d8b + 6 * d8_col;
                    let (q0, q1, q2, q3, e6) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f6 += (a as f32) * (e6 * drow * sc as f32);
                }
                if m > 7 {
                    let cb = 7 * q_col + qb;
                    let d8c = d8b + 7 * d8_col;
                    let (q0, q1, q2, q3, e7) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f7 += (a as f32) * (e7 * drow * sc as f32);
                }
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant). Column c's
        // reduction runs only when m > c; every lane takes the same branch,
        // so the shuffles stay warp-collective, and the tail scales with m
        // exactly like the column bodies above.
        let s0 = warp::reduce_sum_f32(f0);
        if m == 1 {
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
        } else {
            let s1 = warp::reduce_sum_f32(f1);
            let s2 = if m > 2 { warp::reduce_sum_f32(f2) } else { 0.0 };
            let s3 = if m > 3 { warp::reduce_sum_f32(f3) } else { 0.0 };
            let s4 = if m > 4 { warp::reduce_sum_f32(f4) } else { 0.0 };
            let s5 = if m > 5 { warp::reduce_sum_f32(f5) } else { 0.0 };
            let s6 = if m > 6 { warp::reduce_sum_f32(f6) } else { 0.0 };
            let s7 = if m > 7 { warp::reduce_sum_f32(f7) } else { 0.0 };
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m]: store c is guarded by m > c,
                // so exactly the first m slots of the row are touched.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    if m > 1 {
                        *y.get_unchecked_mut(b + 1) = s1;
                    }
                    if m > 2 {
                        *y.get_unchecked_mut(b + 2) = s2;
                    }
                    if m > 3 {
                        *y.get_unchecked_mut(b + 3) = s3;
                    }
                    if m > 4 {
                        *y.get_unchecked_mut(b + 4) = s4;
                    }
                    if m > 5 {
                        *y.get_unchecked_mut(b + 5) = s5;
                    }
                    if m > 6 {
                        *y.get_unchecked_mut(b + 6) = s6;
                    }
                    if m > 7 {
                        *y.get_unchecked_mut(b + 7) = s7;
                    }
                }
            }
        }
    }

    /// Q3_K (K = 256*n_sb values, n_sb super-blocks per row) times q8_1
    /// activations, M <= 8. ggml-mmvq-shaped redesign (round 2): the
    /// activation is quantized once to q8_1 by `q3k_quantize_q8_1`, so the
    /// inner product is a hardware `dp4a` over packed 4xint8 words instead
    /// of per-weight f32 FMAs, and x costs 1/4 the bytes. One warp owns one
    /// row; per iteration the warp covers two super-blocks (lanes 0..15 ->
    /// even sb, 16..31 -> odd sb; iters = ceil(n_sb/2), an odd n_sb guards
    /// the last iteration's odd half) so every lane's 16-weight qs word is
    /// one u32 and the warp's weight loads are contiguous runs. Odd
    /// super-blocks sit 2 mod 4, so every word is assembled from two
    /// aligned u32 loads with a 16-bit funnel select. Each (word, field)
    /// quad is dequantized in registers with SWAR byte arithmetic
    /// (vi = vil - 4*(1-hbit) as signed bytes) and dotted with one u32 of
    /// q8_1 x via dp4a. The q8_1 block is the 128-value half super-block,
    /// the exact span of a lane's four fields (round 3): per column the
    /// four dp4a results are multiplied by their 6-bit sub-block scales and
    /// summed in int, then ONE f32 FMA applies the shared q8_1 scale and
    /// the super-block scale (round 2 loaded d8 and multiplied per field:
    /// the structural M=8 cost). Reduced with a warp shuffle sum.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= m_cols * 64 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q3k_gemv(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        let row_bytes = 110 * n_sb;
        let q_col = 64 * iters as usize; // q8 u64 slots per column
        let d8_col = 2 * n_sb; // 128-value blocks per column

        // Per-lane constants: this lane's qs word within the super-block and
        // the derived scale/d8 bases (see the packing comment above; the q8
        // word base is the permutation position 64*it + 32*p + lane).
        let w16 = lane & 15;
        let half = lane >> 4; // 0: even super-block, 1: odd
        let s0 = 8 * (w16 >> 3) + ((w16 & 7) >> 2); // first sub-block index
        let d8_base = w16 >> 3; // half super-block = one 128-value q8_1 block

        // Scalar accumulators: a runtime c would put an array in local
        // memory; m is launch-uniform so guarded scalars stay in registers.
        let mut f0 = 0.0f32;
        let mut f1 = 0.0f32;
        let mut f2 = 0.0f32;
        let mut f3 = 0.0f32;
        let mut f4 = 0.0f32;
        let mut f5 = 0.0f32;
        let mut f6 = 0.0f32;
        let mut f7 = 0.0f32;

        let mut it: u32 = 0;
        while it < iters {
            let sbp = ((it << 1) | half as u32) as usize;
            // The sbp guard makes a partial final iteration safe; always
            // true when n_sb is even.
            if sbp < n_sb {
                let base = row * row_bytes + sbp * 110; // byte offset
                // A super-block sits 0 or 2 mod 4 depending on the row's own
                // offset too (odd n_sb shifts every other row), so the
                // funnel select comes from the byte window, not from sbp.
                let par = (base >> 1) & 1;

                // qs word: bytes base+32+4*w16 .. +3. The window start is
                // 0 mod 4 or 2 mod 4 with the super-block, so the same floor
                // division addresses both; odd windows reassemble with a
                // 16-bit funnel.
                let qk = (base + 32 + 4 * w16) >> 2;
                // SAFETY: row < n_rows, sbp < n_sb, w16 < 16, so qk+1 stays
                // inside the row's ceil(row_bytes/4) words <= w.len() by the
                // launch contract (odd super-blocks only shift the window
                // by 2).
                let (lo, hi) = unsafe { (*w.get_unchecked(qk), *w.get_unchecked(qk + 1)) };
                let vl = if par == 0 { lo } else { funnel16(lo, hi) };

                // hmask word (bytes base+4*(w16%8) .. +3), one bit per
                // weight: bit 4*(w16/8)+field. Invert so a clear hmask bit
                // (subtract 4) becomes a set bit, pre-shifted to bit 0 of
                // each byte.
                let hk = (base + 4 * (w16 & 7)) >> 2;
                // SAFETY: same row/sbp/w16 bounds as the qs window above.
                let (hlo, hhi) = unsafe { (*w.get_unchecked(hk), *w.get_unchecked(hk + 1)) };
                let hm = if par == 0 { hlo } else { funnel16(hlo, hhi) };
                let vh1 = (!hm) >> (4 * (w16 >> 3)) as u32;

                // scales: 12 bytes at base+96..108, decoded with the aux[]
                // shuffle of dequantize_row_q3_K (verbatim from round 1).
                // The 16-byte window base+96..112 (even) / base+94..110
                // (odd) is covered by four aligned words.
                let ak = (base + 96) >> 2;
                // SAFETY: ak+3 < ceil((row+1)*row_bytes/4) <= w.len(): the
                // 16-byte window ends at most 2 bytes past the row end, but
                // the floored word index stays inside the row's words.
                let (aw0, aw1, aw2, aw3) = unsafe {
                    (
                        *w.get_unchecked(ak),
                        *w.get_unchecked(ak + 1),
                        *w.get_unchecked(ak + 2),
                        *w.get_unchecked(ak + 3),
                    )
                };
                let (a0w, a1w, a2w) = if par == 0 {
                    (aw0, aw1, aw2)
                } else {
                    (funnel16(aw0, aw1), funnel16(aw1, aw2), funnel16(aw2, aw3))
                };
                let ts = q3k_aux_scales(a0w, a1w, a2w);

                // Super-block scale d (f16 at bytes base+108..109): low half
                // of aw3 for an even window (word covers 108..111), high
                // half for odd (word covers 106..109).
                let d_bits = if par == 0 {
                    (aw3 & 0xffff) as u16
                } else {
                    (aw3 >> 16) as u16
                };
                let drow = half_to_f32(d_bits);

                // Sub-block scales for fields 0..3: sub-block s0+2j.
                let sc = [
                    q3k_sub_scale(&ts, s0),
                    q3k_sub_scale(&ts, s0 + 2),
                    q3k_sub_scale(&ts, s0 + 4),
                    q3k_sub_scale(&ts, s0 + 6),
                ];

                // Dequantize the four quads to signed bytes (SWAR):
                // q3k_dequant folds the hbit subtract borrow-free.
                let vi = q3k_dequant(vl, vh1);

                // q8_1 words in the q3 u64 pairing: field pair (2p, 2p+1)
                // of column c lives in u64 slot q_col*c + 64it + 32p + lane
                // — lo is field 2p, hi is 2p+1 (host-verified identity with
                // the value-order slot the linear layout used). Two u64
                // loads per iteration instead of four u32 loads: same bytes
                // and wavefronts, half the load-issue count.
                let qb = 64 * it as usize + lane;
                // d8 base for this lane (column 0): two 128-value q8_1
                // blocks per super-block, this lane's fields all sit in
                // block 2*sbp + d8_base of the column.
                let d8b = 2 * sbp + d8_base;

                // Column 0 (always active). The lane's four fields share one
                // q8_1 block: int chain (dp4a x sub-block scale), one FMA
                // with the shared block scale and the super-block scale.
                // Fields 0/1 come from u64 slot qb (lo/hi), fields 2/3
                // from qb + 32.
                {
                    // SAFETY: qb + 32 <= q_col - 1 — both u64 slots are
                    // inside the column's q_col u64s (the permutation's
                    // group bound); d8b < d8_col by the sbp guard.
                    let w01 = unsafe { *q.get_unchecked(qb) };
                    let w23 = unsafe { *q.get_unchecked(qb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f0 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);
                }
                // Columns 1..7, one launch-uniform guard per column so the
                // work scales with m (MUL-8 amortization curve). Same shape
                // as column 0 with the per-column q/d8 offsets (q stride
                // q_col u64, d8 d8_col).
                // SAFETY: guard m > c means q.len() >= (c+1)*q_col >
                // q_col*c + qb + 32 and d8.len() >= (c+1)*d8_col >
                // d8b + d8_col*c, launch-uniform.
                if m > 1 {
                    let cb = q_col + qb;
                    let d8c = d8b + d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f1 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
                if m > 2 {
                    let cb = 2 * q_col + qb;
                    let d8c = d8b + 2 * d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f2 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
                if m > 3 {
                    let cb = 3 * q_col + qb;
                    let d8c = d8b + 3 * d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f3 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
                if m > 4 {
                    let cb = 4 * q_col + qb;
                    let d8c = d8b + 4 * d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f4 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
                if m > 5 {
                    let cb = 5 * q_col + qb;
                    let d8c = d8b + 5 * d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f5 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
                if m > 6 {
                    let cb = 6 * q_col + qb;
                    let d8c = d8b + 6 * d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f6 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
                if m > 7 {
                    let cb = 7 * q_col + qb;
                    let d8c = d8b + 7 * d8_col;
                    let w01 = unsafe { *q.get_unchecked(cb) };
                    let w23 = unsafe { *q.get_unchecked(cb + 32) };
                    let a = q3k_chain(&vi, w01, w23, &sc);
                    f7 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
                }
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant). Column c's
        // reduction runs only when m > c; every lane takes the same branch,
        // so the shuffles stay warp-collective, and the tail scales with m
        // exactly like the column bodies above.
        let s0 = warp::reduce_sum_f32(f0);
        if m == 1 {
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
        } else {
            let s1 = warp::reduce_sum_f32(f1);
            let s2 = if m > 2 { warp::reduce_sum_f32(f2) } else { 0.0 };
            let s3 = if m > 3 { warp::reduce_sum_f32(f3) } else { 0.0 };
            let s4 = if m > 4 { warp::reduce_sum_f32(f4) } else { 0.0 };
            let s5 = if m > 5 { warp::reduce_sum_f32(f5) } else { 0.0 };
            let s6 = if m > 6 { warp::reduce_sum_f32(f6) } else { 0.0 };
            let s7 = if m > 7 { warp::reduce_sum_f32(f7) } else { 0.0 };
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m]: store c is guarded by m > c,
                // so exactly the first m slots of the row are touched.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    if m > 1 {
                        *y.get_unchecked_mut(b + 1) = s1;
                    }
                    if m > 2 {
                        *y.get_unchecked_mut(b + 2) = s2;
                    }
                    if m > 3 {
                        *y.get_unchecked_mut(b + 3) = s3;
                    }
                    if m > 4 {
                        *y.get_unchecked_mut(b + 4) = s4;
                    }
                    if m > 5 {
                        *y.get_unchecked_mut(b + 5) = s5;
                    }
                    if m > 6 {
                        *y.get_unchecked_mut(b + 6) = s6;
                    }
                    if m > 7 {
                        *y.get_unchecked_mut(b + 7) = s7;
                    }
                }
            }
        }
    }
}

/// A CUDA context on device 0 with this crate's device module loaded and one
/// non-blocking stream that every launch and copy of this engine goes on.
///
/// The stream is a real `cuStreamCreate` stream, not the legacy default: the
/// driver refuses graph capture on the null stream, and `Graph::capture`
/// records whatever is enqueued on `self.stream()` between begin and end.
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

impl Gpu {
    /// Create the context, the engine stream, and load the embedded device
    /// bundle.
    pub fn new() -> Result<Gpu, GpuError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the kernels module above; every launcher checks its launch
        // contract before launching.
        let module = unsafe { kernels::load(&ctx)? };
        Ok(Gpu {
            ctx,
            stream,
            module,
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The engine stream. Allocations at load time and the per-step
    /// synchronize go here too, so nothing ever orders against the null
    /// stream.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Capture `body`'s enqueues on the engine stream into a replayable
    /// graph. See `Graph::capture` for what the body may not do.
    pub fn capture<F>(&self, body: F) -> Result<Graph, GpuError>
    where
        F: FnOnce(&CudaStream) -> Result<(), GpuError>,
    {
        Graph::capture(&self.stream, body)
    }

    /// Enqueue the q8_1 quantization of `x` (`act.m()` columns of `act.k()`
    /// f32 each) into `act`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_quantize_q8_1(
        &self,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let m = act.m();
        let n_sb = act.n_sb();
        if x.len() < m * act.k() {
            return Err(format!(
                "enqueue_quantize_q8_1: x.len() {} < m*k = {}",
                x.len(),
                m * act.k()
            )
            .into());
        }
        // Prepared per call for now: the prepare step is host-only contract
        // validation, and it is exactly the kind of per-launch host cost a
        // captured graph removes. Caching per shape is P8's business.
        let prep = self.module.prepare_q3k_quantize_q8_1(LaunchConfig1D::new(
            (m * n_sb * 2) as u32,
            32,
            0,
        ))?;
        self.module.q3k_quantize_q8_1(
            &self.stream,
            &prep,
            x,
            m as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            n_sb.div_ceil(4) as u32,
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q4_K weight of `w.rows()` rows — 36 u32
    /// words per super-block, `36 * n_sb` per row — against the quantized
    /// activations in `act`, which supply K. `y` holds `rows * m` f32,
    /// row-major with `m` outputs per row. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gemv_q4k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        let n_sb = act.n_sb();
        if w.cols() != 36 * n_sb {
            return Err(format!(
                "enqueue_gemv_q4k: Q4_K rows are 36*{} = {} words at K={}, got {}",
                n_sb,
                36 * n_sb,
                act.k(),
                w.cols()
            )
            .into());
        }
        if y.len() < n_rows * m {
            return Err(format!(
                "enqueue_gemv_q4k: y.len() {} < rows*m = {}",
                y.len(),
                n_rows * m
            )
            .into());
        }
        let prep =
            self.module
                .prepare_q4k_gemv(LaunchConfig1D::new(n_rows.div_ceil(8) as u32, 256, 0))?;
        self.module.q4k_gemv(
            &self.stream,
            &prep,
            w.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            n_rows as u32,
            m as u32,
            n_sb as u32,
            n_sb.div_ceil(4) as u32,
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q3_K weight of `w.rows()` rows — 110
    /// bytes per super-block, the row stream as u32 words with the final
    /// word zero-padded, `110 * n_sb / 4` words per row (an integer only
    /// for even n_sb, which every Q3_K site of this model has; an odd n_sb
    /// leaves rows unaligned and needs load-time repacking, rejected here)
    /// — against the quantized activations in `act`, which supply K. `y`
    /// holds `rows * m` f32, row-major with `m` outputs per row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q3k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        let n_sb = act.n_sb();
        if n_sb % 2 != 0 {
            return Err(format!(
                "enqueue_gemv_q3k: odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                act.k()
            )
            .into());
        }
        if w.cols() != 110 * n_sb / 4 {
            return Err(format!(
                "enqueue_gemv_q3k: Q3_K rows are 110*{n_sb}/4 = {} words at K={}, got {}",
                110 * n_sb / 4,
                act.k(),
                w.cols()
            )
            .into());
        }
        if y.len() < n_rows * m {
            return Err(format!(
                "enqueue_gemv_q3k: y.len() {} < rows*m = {}",
                y.len(),
                n_rows * m
            )
            .into());
        }
        let prep =
            self.module
                .prepare_q3k_gemv(LaunchConfig1D::new(n_rows.div_ceil(8) as u32, 256, 0))?;
        self.module.q3k_gemv(
            &self.stream,
            &prep,
            w.buf(),
            &act.q3,
            &act.d8,
            n_rows as u32,
            m as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q6_K weight of `w.rows()` rows — 210
    /// bytes per super-block, the row stream as u32 words with the final
    /// word zero-padded, `210 * n_sb / 4` words per row (an integer only
    /// for even n_sb; an odd n_sb leaves rows unaligned and needs load-time
    /// repacking, rejected here) — against the quantized activations in
    /// `act`, which supply K. `y` holds `rows * m` f32, row-major with `m`
    /// outputs per row. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q6k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        let n_sb = act.n_sb();
        if n_sb % 2 != 0 {
            return Err(format!(
                "enqueue_gemv_q6k: odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                act.k()
            )
            .into());
        }
        if w.cols() != 210 * n_sb / 4 {
            return Err(format!(
                "enqueue_gemv_q6k: Q6_K rows are 210*{n_sb}/4 = {} words at K={}, got {}",
                210 * n_sb / 4,
                act.k(),
                w.cols()
            )
            .into());
        }
        if y.len() < n_rows * m {
            return Err(format!(
                "enqueue_gemv_q6k: y.len() {} < rows*m = {}",
                y.len(),
                n_rows * m
            )
            .into());
        }
        let prep =
            self.module
                .prepare_q6k_gemv(LaunchConfig1D::new(n_rows.div_ceil(8) as u32, 256, 0))?;
        self.module.q6k_gemv(
            &self.stream,
            &prep,
            w.buf(),
            &act.q6,
            &act.d8,
            n_rows as u32,
            m as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            y,
        )?;
        Ok(())
    }

    /// Q4_K (K=2048) gemv through the shared q8_1 activation quantizer, with
    /// upload, scratch allocation and copy-back inside the call — the
    /// stage-0 correctness shape, not the step shape.
    ///
    /// `w` is `n_rows * 288` little-endian u32 words of Q4_K rows (144 B
    /// per super-block, 8 super-blocks per row); `x` is `m` activation
    /// columns of 2048 f32 each, concatenated; the result is `n_rows * m`
    /// f32, row-major with `m` outputs per row.
    pub fn gemv_q4k(
        &self,
        w: &[u32],
        x: &[f32],
        n_rows: usize,
        m: usize,
    ) -> Result<Vec<f32>, GpuError> {
        check_q4k_geometry(w.len(), x.len(), n_rows, m)?;
        let stream = &self.stream;
        let w_dev = DeviceTensor::upload(stream, &w[..n_rows * 288], n_rows, 288)?;
        let x_dev = DeviceBuffer::from_host(stream, &x[..m * 2048])?;
        let mut act = Q8Act::new(stream, m)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_rows * m)?;
        self.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        self.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
        stream.synchronize()?;
        Ok(y_dev.to_host_vec(stream)?)
    }

    /// Rough launch-cost probe for design, NOT a benchmark of record: stages
    /// `w`/`x` on the device and quantizes once, then times `iters` bare
    /// q4k_gemv launches, each immediately followed by a stream
    /// synchronize. Returns the mean microseconds per launch+sync.
    pub fn probe_q4k_launch_us(
        &self,
        w: &[u32],
        x: &[f32],
        n_rows: usize,
        m: usize,
        iters: u32,
    ) -> Result<f64, GpuError> {
        check_q4k_geometry(w.len(), x.len(), n_rows, m)?;
        if iters == 0 {
            return Err("probe_q4k_launch_us: iters must be >= 1".into());
        }
        let stream = &self.stream;
        let w_dev = DeviceTensor::upload(stream, &w[..n_rows * 288], n_rows, 288)?;
        let x_dev = DeviceBuffer::from_host(stream, &x[..m * 2048])?;
        let mut act = Q8Act::new(stream, m)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_rows * m)?;
        self.enqueue_quantize_q8_1(&x_dev, &mut act)?;

        let mut launch_gemv = || -> Result<(), GpuError> {
            self.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
            stream.synchronize()?;
            Ok(())
        };
        for _ in 0..2 {
            launch_gemv()?;
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_gemv()?;
        }
        Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters))
    }
}

/// Reject geometry the kernels' launch contracts do not cover: the q4k gemv
/// reads 288 words per row, the quantizer consumes 2048 values per column
/// and both support 1..=8 columns.
fn check_q4k_geometry(w_len: usize, x_len: usize, n_rows: usize, m: usize) -> Result<(), GpuError> {
    if n_rows == 0 || !(1..=8).contains(&m) {
        return Err(format!(
            "gemv_q4k: need n_rows >= 1 and 1 <= m <= 8, got n_rows={n_rows} m={m}"
        )
        .into());
    }
    if w_len < n_rows * 288 {
        return Err(format!("gemv_q4k: w.len() {w_len} < n_rows*288 = {}", n_rows * 288).into());
    }
    if x_len < m * 2048 {
        return Err(format!("gemv_q4k: x.len() {x_len} < m*2048 = {}", m * 2048).into());
    }
    Ok(())
}
