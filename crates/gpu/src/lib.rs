//! bloomery-gpu — the stage-0 CUDA kernels packaged as a library.
//!
//! The device code is copied verbatim from `crates/q3k-gemv/src/main.rs`
//! (same `#[cuda_module]`, same kernels, same launch contracts) so a binary
//! in a different package can call it. The crate must be compiled with
//! `cargo oxide`; the codegen backend embeds the compiled device bundle in a
//! `.oxart` member of this crate's rlib, and the anchor reference emitted
//! inside `kernels::load` pulls that member into any final binary that calls
//! this API.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp,
};
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
    use crate::cores::{q4k_a_chain, q4k_nibble};

    /// IEEE-754 half to float, integer-only so no `f16` feature gate is
    /// needed on either side of the unified compilation.
    fn half_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) as u32) << 31;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let mant = (bits & 0x3ff) as u32;
        let mag = if exp == 0x1f {
            0x7f800000 | (mant << 13)
        } else if exp == 0 {
            if mant == 0 {
                0
            } else {
                let mut e = 127 - 14;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                (e << 23) | ((m & 0x3ff) << 13)
            }
        } else {
            ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(sign | mag)
    }

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
            x.len() >= m_cols * 2048,
            q3.len() >= m_cols * 256,
            q4.len() >= m_cols * 512,
            q6.len() >= m_cols * 512,
            s8.len() >= m_cols * 64,
            d8.len() >= m_cols * 16
        )
    )]
    pub fn q3k_quantize_q8_1(
        x: &[f32],
        m_cols: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
    ) {
        // One 32-thread block per 128-value quant block: blk is the BLOCK
        // index (global tid / 32), not the thread id.
        let blk = thread::index_1d().get() / 32;
        let total = m_cols as usize * 16;
        if blk >= total {
            return;
        }
        let col = blk / 16;
        let b = blk % 16;
        let lane = warp::lane_id() as usize;

        // Lane covers the four consecutive values 4*lane .. 4*lane+3 of the
        // block; the warp max over the four per-lane maxima is the block
        // amax (no cross-lane byte packing needed, unlike the 32-value
        // geometry where one value per lane forced two shuffle_downs).
        let base = col * 2048 + 128 * b + 4 * lane;
        let v0 = unsafe { *x.get_unchecked(base) };
        let v1 = unsafe { *x.get_unchecked(base + 1) };
        let v2 = unsafe { *x.get_unchecked(base + 2) };
        let v3 = unsafe { *x.get_unchecked(base + 3) };
        let amax = warp::reduce_max_f32(v0.abs().max(v1.abs()).max(v2.abs()).max(v3.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let q0 = ((v0 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
        let q1 = ((v1 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
        let q2 = ((v2 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
        let qw = ((v3 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
        let word = q0 | (q1 << 8) | (q2 << 16) | (qw << 24);

        // v4 = this word's index in value order within the column: the word
        // covers values 128b + 4*lane .. +3, so v4 = (128b + 4*lane)/4 =
        // 32b + lane (32 words per 128-value block, 16 blocks per column).
        // Each p_fmt maps it to the load slot of the named gemv geometry
        // (permutations of 0..512, host-verified including this value-span
        // tie — an earlier draft said 16b + lane, which the load-site check
        // alone could not catch because both sides then agree on a wrong
        // bijection over half the column).
        let v4 = (32 * b + lane) as u32;
        // Q3_K u64 pairing: the two fields of a gemv load PAIR (j, j^1) are
        // always held by quantize lanes lane and lane^8 of one block (v4
        // differs only in bit 3), so g3(v4) is a bijection onto the 256 u64
        // slots of the column with v4's bit 3 selecting the half. All lanes
        // run the collective shuffle; the bit3-clear half stores.
        let g3 = 64 * (v4 >> 7)
            + 32 * ((v4 >> 4) & 1)
            + 16 * ((v4 >> 6) & 1)
            + 8 * ((v4 >> 5) & 1)
            + (v4 & 7);
        let partner = warp::shuffle_xor(word, 8);
        let cb = col as u32 * 256;
        let p4 = 256 * (v4 >> 8) + 32 * (v4 & 7) + 8 * ((v4 >> 6) & 3) + ((v4 >> 3) & 7);
        let p6 = 128 * (v4 >> 7) + 32 * (v4 & 3) + 16 * ((v4 >> 6) & 1) + ((v4 >> 2) & 15);
        let cu = col as u32 * 512;
        // SAFETY: g3 < 256 and p4/p6 < 512 per column (host-verified
        // bijections incl. the pair check) and the three stores hit three
        // distinct buffers; bit3-clear lanes of a block write disjoint u64
        // positions, every lane its own u32 position.
        unsafe {
            if lane & 8 == 0 {
                *q3.get_unchecked_mut((cb + g3) as usize) =
                    (word as u64) | ((partner as u64) << 32);
            }
            *q4.get_unchecked_mut((cu + p4) as usize) = word;
            *q6.get_unchecked_mut((cu + p6) as usize) = word;
        }

        // 32-value-group signed sums: butterfly over the lane-local quad
        // sums (masks 1, 2, 4); afterwards every lane holds its octet's
        // total and lanes 8k write group 4b + k of the column.
        let mut g = (q0 as i8 as i32) + (q1 as i8 as i32) + (q2 as i8 as i32) + (qw as i8 as i32);
        g += warp::shuffle_xor(g as u32, 1) as i32;
        g += warp::shuffle_xor(g as u32, 2) as i32;
        g += warp::shuffle_xor(g as u32, 4) as i32;
        if lane & 7 == 0 {
            // SAFETY: group index 4b + lane/8 < 64 per column; s8 holds
            // m_cols*64 words and one lane writes each group.
            unsafe {
                *s8.get_unchecked_mut(col * 64 + 4 * b + (lane >> 3)) = g;
            }
        }
        if lane == 0 {
            // SAFETY: lane 0 of each warp writes its own d8 slot.
            unsafe {
                *d8.get_unchecked_mut(col * 16 + b) = d;
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

    /// Q4_K (K=2048, 8 super-blocks per row) times q8_1 activations, M <= 8.
    /// Same skeleton as q3k_gemv: one warp per row, guarded scalar
    /// accumulators. The warp covers FOUR super-blocks per iteration (lane L
    /// owns the 32-value sub-block s = L&7 of super-block 4*it + L>>3), so
    /// the row's 8 super-blocks take 2 iterations. Each lane keeps its whole
    /// sub-block local: the min-offset B chain reduces to one s8 group-sum
    /// load per column (the quantize kernel precomputed the exact integer
    /// the per-column dp4a(0x01010101) chain accumulated), and the qs window
    /// is decoded once per iteration into vi[8] instead of once per column.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 288,
            q.len() >= m_cols * 512,
            s8.len() >= m_cols * 64,
            d8.len() >= m_cols * 16,
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
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;

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
        while it < 2 {
            let sbp = 4 * it as usize + grp;
            let wk = row * 288 + 36 * sbp; // super-block word base (144 B)

            // d/dmin word + the 12 scale bytes in words 1..3.
            // SAFETY: sbp < 8, so wk+3 <= row*288 + 36*7 + 3 = row*288 + 255
            // < (row+1)*288 <= w.len() by the launch contract.
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

            // get_scale_min_k4(s) against the scale words: scales bytes s and
            // s+4 live in (w1,w2) for s<4 and (w2,w3) for s>=4; the j>=4
            // branch borrows bytes s-4 and s for the 6-bit high parts.
            let (sc, mi) = if s < 4 {
                let sh = 8 * s as u32;
                (((w1 >> sh) & 63) as i32, ((w2 >> sh) & 63) as i32)
            } else {
                let sh = 8 * (s - 4) as u32;
                let qs = (w2 >> sh) & 0xff; // scales byte s
                let qs4 = (w3 >> sh) & 0xff; // scales byte s+4
                let qsm = (w1 >> sh) & 0xff; // scales byte s-4
                (
                    ((qs4 & 0x0f) as i32) | (((qsm >> 6) as i32) << 4),
                    ((qs4 >> 4) as i32) | (((qs >> 6) as i32) << 4),
                )
            };
            // Column-independent chain coefficients.
            let cda = d * sc as f32;
            let cdb = 8.0 * d * sc as f32 - dmin * mi as f32;

            // qs word base: 8 words from super-block word 4 + 8*(s>>1);
            // nibble select is the sub-block parity, 0 (low) or 4 (high).
            // Hoisted (MUL-8): the window and its SWAR decode are
            // column-independent, so decode once per iteration and let every
            // column's A chain reuse the registers.
            let qsk = wk + 4 + 8 * (s >> 1);
            let nib_sh = (s as u32 & 1) * 4;
            // SAFETY: qsk + 7 <= wk + 4 + 8*3 + 7 = wk + 35, inside the
            // row's word span by the same bound as w0..w3 above.
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

            // This lane's q8 words in the q4 permutation: word i of column c
            // lives at 512c + 256it + 32i + lane (host-verified identity
            // with the value-order slot the linear layout used). The B-chain
            // group of (it, lane) is 32it + lane: 32 consecutive i32 across
            // the warp, one 128B line.
            let qb = 256 * it as usize + lane;
            let s8b = 32 * it as usize + lane;
            // This lane's fields all sit in q8_1 block 2*sbp + s/4.
            let d8b = 2 * sbp + (s >> 2);

            // Column 0 (always active).
            {
                let a = q4k_a_chain(&vi, q, qb);
                // SAFETY: s8b < 64 <= s8.len() and d8b < 16 <= d8.len() by
                // the contract (m >= 1).
                let b = unsafe { *s8.get_unchecked(s8b) };
                let e0 = unsafe { *d8.get_unchecked(d8b) };
                f0 += (a as f32 * cda + b as f32 * cdb) * e0;
            }
            // Columns 1..7, one launch-uniform guard per column so the work
            // scales with m (MUL-8 amortization curve). Column c reads q8
            // words at 512c + qb (q4k_a_chain adds 32i), the s8 group at
            // 64c + s8b and block d8b + 16c.
            // SAFETY: guard c+1 means m >= c+1 is launch-uniform, so the
            // buffer lengths (m*512 / m*64 / m*16) cover every offset below
            // and the branch never diverges within a warp.
            if m > 1 {
                // SAFETY: m > 1 => q.len() >= 1024 > 512 + qb + 224,
                // s8.len() >= 128 > 64 + s8b, d8.len() >= 32 > d8b + 16.
                let a = q4k_a_chain(&vi, q, 512 + qb);
                let b = unsafe { *s8.get_unchecked(64 + s8b) };
                let e1 = unsafe { *d8.get_unchecked(d8b + 16) };
                f1 += (a as f32 * cda + b as f32 * cdb) * e1;
            }
            if m > 2 {
                // SAFETY: m > 2 => q.len() >= 1536 > 1024 + qb + 224,
                // s8.len() >= 192 > 128 + s8b, d8.len() >= 48 > d8b + 32.
                let a = q4k_a_chain(&vi, q, 1024 + qb);
                let b = unsafe { *s8.get_unchecked(128 + s8b) };
                let e2 = unsafe { *d8.get_unchecked(d8b + 32) };
                f2 += (a as f32 * cda + b as f32 * cdb) * e2;
            }
            if m > 3 {
                // SAFETY: m > 3 => q.len() >= 2048 > 1536 + qb + 224,
                // s8.len() >= 256 > 192 + s8b, d8.len() >= 64 > d8b + 48.
                let a = q4k_a_chain(&vi, q, 1536 + qb);
                let b = unsafe { *s8.get_unchecked(192 + s8b) };
                let e3 = unsafe { *d8.get_unchecked(d8b + 48) };
                f3 += (a as f32 * cda + b as f32 * cdb) * e3;
            }
            if m > 4 {
                // SAFETY: m > 4 => q.len() >= 2560 > 2048 + qb + 224,
                // s8.len() >= 320 > 256 + s8b, d8.len() >= 80 > d8b + 64.
                let a = q4k_a_chain(&vi, q, 2048 + qb);
                let b = unsafe { *s8.get_unchecked(256 + s8b) };
                let e4 = unsafe { *d8.get_unchecked(d8b + 64) };
                f4 += (a as f32 * cda + b as f32 * cdb) * e4;
            }
            if m > 5 {
                // SAFETY: m > 5 => q.len() >= 3072 > 2560 + qb + 224,
                // s8.len() >= 384 > 320 + s8b, d8.len() >= 96 > d8b + 80.
                let a = q4k_a_chain(&vi, q, 2560 + qb);
                let b = unsafe { *s8.get_unchecked(320 + s8b) };
                let e5 = unsafe { *d8.get_unchecked(d8b + 80) };
                f5 += (a as f32 * cda + b as f32 * cdb) * e5;
            }
            if m > 6 {
                // SAFETY: m > 6 => q.len() >= 3584 > 3072 + qb + 224,
                // s8.len() >= 448 > 384 + s8b, d8.len() >= 128 > d8b + 112.
                let a = q4k_a_chain(&vi, q, 3072 + qb);
                let b = unsafe { *s8.get_unchecked(384 + s8b) };
                let e6 = unsafe { *d8.get_unchecked(d8b + 96) };
                f6 += (a as f32 * cda + b as f32 * cdb) * e6;
            }
            if m > 7 {
                // SAFETY: m > 7 => q.len() >= 4096 > 3584 + qb + 224,
                // s8.len() >= 512 > 448 + s8b, d8.len() >= 128 > d8b + 112.
                let a = q4k_a_chain(&vi, q, 3584 + qb);
                let b = unsafe { *s8.get_unchecked(448 + s8b) };
                let e7 = unsafe { *d8.get_unchecked(d8b + 112) };
                f7 += (a as f32 * cda + b as f32 * cdb) * e7;
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

    /// Enqueue the q8_1 quantization of `x` (`act.m()` columns of 2048 f32)
    /// into `act`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_quantize_q8_1(
        &self,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let m = act.m();
        if x.len() < m * 2048 {
            return Err(format!(
                "enqueue_quantize_q8_1: x.len() {} < m*2048 = {}",
                x.len(),
                m * 2048
            )
            .into());
        }
        // Prepared per call for now: the prepare step is host-only contract
        // validation, and it is exactly the kind of per-launch host cost a
        // captured graph removes. Caching per shape is P8's business.
        let prep =
            self.module
                .prepare_q3k_quantize_q8_1(LaunchConfig1D::new(16 * m as u32, 32, 0))?;
        self.module.q3k_quantize_q8_1(
            &self.stream,
            &prep,
            x,
            m as u32,
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q4_K (K=2048) weight of `w.rows()` rows
    /// (288 u32 words per row) against the quantized activations in `act`.
    /// `y` holds `rows * m` f32, row-major with `m` outputs per row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q4k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        if w.cols() != 288 {
            return Err(format!(
                "enqueue_gemv_q4k: Q4_K K=2048 rows are 288 words, got {}",
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
