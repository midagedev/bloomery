use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, dotprod::dp4a_s32, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

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
    /// consecutive values and writes them as one packed u32 word, so the gemv
    /// reads whole words. A 128-value block is one Q3_K half super-block, the
    /// exact span a gemv lane's four fields cover in one step: all four dp4a
    /// share this block's scale, so they chain in int behind ONE f32 FMA
    /// instead of one scale load + FMA per field.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (x.len() >= m_cols * 2048, q.len() >= m_cols * 512, d8.len() >= m_cols * 16)
    )]
    pub fn q3k_quantize_q8_1(
        x: &[f32],
        m_cols: u32,
        mut q: DisjointSlice<u32>,
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
        let q3 = ((v3 / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
        let word = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        // SAFETY: lane L writes word col*512 + 32*b + L; q holds
        // m_cols*512 words and a block's lanes write disjoint words.
        unsafe {
            *q.get_unchecked_mut(col * 512 + 32 * b + lane) = word;
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

    /// One lane's A/B int chains over its 8 (qs word, q8 word) pairs.
    /// nibble select `nib_sh` is the lane's sub-block parity (0 or 4).
    /// SAFETY: callers keep `qsk + 8` inside the row's word span (the qs
    /// window is 4..36 words into a 36-word super-block) and `ub + 8`
    /// inside one column's 512 q8 words.
    fn q4k_lane_chains(w: &[u32], q: &[u32], qsk: usize, ub: usize, nib_sh: u32) -> (i32, i32) {
        let mut a = 0i32;
        let mut b = 0i32;
        let mut i: u32 = 0;
        while i < 8 {
            // SAFETY: qsk+i and ub+i stay inside the spans this fn's contract
            // documents (row words / one column's 512 q8 words).
            let (qsw, qv) = unsafe {
                (
                    *w.get_unchecked(qsk + i as usize),
                    *q.get_unchecked(ub + i as usize),
                )
            };
            // nibble - 8 per byte, SWAR (see the Q3_K bias trick).
            let vi = ((((qsw >> nib_sh) & 0x0f0f0f0f) | 0x80808080).wrapping_sub(0x08080808))
                ^ 0x80808080;
            a = dp4a_s32(vi, qv, a);
            b = dp4a_s32(0x01010101, qv, b);
            i += 1;
        }
        (a, b)
    }

    /// Q4_K (K=2048, 8 super-blocks per row) times q8_1 activations, M <= 8.
    /// Same skeleton as q3k_gemv: one warp per row, guarded scalar
    /// accumulators. The warp covers FOUR super-blocks per iteration (lane L
    /// owns the 32-value sub-block s = L&7 of super-block 4*it + L>>3), so
    /// the row's 8 super-blocks take 2 iterations. Each lane keeps its whole
    /// sub-block local, which is what makes the B chain (the min offset)
    /// lane-local instead of a cross-lane reduction.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 288,
            q.len() >= m_cols * 512,
            d8.len() >= m_cols * 16,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q4k_gemv(
        w: &[u32],
        q: &[u32],
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
            let qsk = wk + 4 + 8 * (s >> 1);
            let nib_sh = (s as u32 & 1) * 4;
            // This lane's 32 values sit in q8 words 64*sbp + 8*s .. +8 of
            // the column (value 32s+4i -> word 64*sbp + 8*s + i), inside the
            // one 128-value q8_1 block 2*sbp + s/4.
            let ub = 64 * sbp + 8 * s;
            let d8b = 2 * sbp + (s >> 2);

            // Column 0 (always active).
            {
                let (a, b) = q4k_lane_chains(w, q, qsk, ub, nib_sh);
                // SAFETY: d8b < 16 <= d8.len() by the contract (m >= 1).
                let e0 = unsafe { *d8.get_unchecked(d8b) };
                f0 += (a as f32 * cda + b as f32 * cdb) * e0;
            }
            if m > 1 {
                // Columns 1..7 (launch-uniform guard): same chains against
                // each column's q8 words (column stride 512) and blocks
                // (d8 stride 16).
                // SAFETY: m > 1 is launch-uniform, so d8.len() >= 128 and
                // d8b + 112 <= 15 + 112 < 128.
                let (e1, e2, e3, e4, e5, e6, e7) = unsafe {
                    (
                        *d8.get_unchecked(d8b + 16),
                        *d8.get_unchecked(d8b + 32),
                        *d8.get_unchecked(d8b + 48),
                        *d8.get_unchecked(d8b + 64),
                        *d8.get_unchecked(d8b + 80),
                        *d8.get_unchecked(d8b + 96),
                        *d8.get_unchecked(d8b + 112),
                    )
                };
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 512, nib_sh);
                f1 += (a as f32 * cda + b as f32 * cdb) * e1;
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 1024, nib_sh);
                f2 += (a as f32 * cda + b as f32 * cdb) * e2;
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 1536, nib_sh);
                f3 += (a as f32 * cda + b as f32 * cdb) * e3;
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 2048, nib_sh);
                f4 += (a as f32 * cda + b as f32 * cdb) * e4;
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 2560, nib_sh);
                f5 += (a as f32 * cda + b as f32 * cdb) * e5;
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 3072, nib_sh);
                f6 += (a as f32 * cda + b as f32 * cdb) * e6;
                let (a, b) = q4k_lane_chains(w, q, qsk, ub + 3584, nib_sh);
                f7 += (a as f32 * cda + b as f32 * cdb) * e7;
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant).
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
            let s2 = warp::reduce_sum_f32(f2);
            let s3 = warp::reduce_sum_f32(f3);
            let s4 = warp::reduce_sum_f32(f4);
            let s5 = warp::reduce_sum_f32(f5);
            let s6 = warp::reduce_sum_f32(f6);
            let s7 = warp::reduce_sum_f32(f7);
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m].
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    *y.get_unchecked_mut(b + 1) = s1;
                    *y.get_unchecked_mut(b + 2) = s2;
                    *y.get_unchecked_mut(b + 3) = s3;
                    *y.get_unchecked_mut(b + 4) = s4;
                    *y.get_unchecked_mut(b + 5) = s5;
                    *y.get_unchecked_mut(b + 6) = s6;
                    *y.get_unchecked_mut(b + 7) = s7;
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

    /// Q6_K (K=2048, 8 super-blocks per row) times q8_1 activations, M <= 8.
    /// Same lane geometry as q3k_gemv: one warp per row, iteration covers
    /// two super-blocks (lanes 0..15 even, 16..31 odd), lane owns one
    /// 16-value sub-block = one scale = four dp4a per column.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 420,
            q.len() >= m_cols * 512,
            d8.len() >= m_cols * 16,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q6k_gemv(
        w: &[u32],
        q: &[u32],
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
        while it < 4 {
            let sbp = ((it << 1) | half as u32) as usize;
            let base = row * 1680 + sbp * 210; // byte offset, 2 mod 4 when odd
            let par = sbp & 1;

            // ql window: 4 words at super-block byte 64*(w16>>3) + qlok.
            let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
            // SAFETY: lk+4 <= row*420 + 210*7/4 + ... the last window of the
            // last row ends at byte <= 1680*row + 1680, and the floored word
            // index stays inside the row's 420 words (see Q3_K's aw window).
            let (l0, l1, l2, l3, l4) = unsafe {
                (
                    *w.get_unchecked(lk),
                    *w.get_unchecked(lk + 1),
                    *w.get_unchecked(lk + 2),
                    *w.get_unchecked(lk + 3),
                    *w.get_unchecked(lk + 4),
                )
            };
            let ql0 = if par == 0 {
                l0
            } else {
                (l0 >> 16) | (l1 << 16)
            };
            let ql1 = if par == 0 {
                l1
            } else {
                (l1 >> 16) | (l2 << 16)
            };
            let ql2 = if par == 0 {
                l2
            } else {
                (l2 >> 16) | (l3 << 16)
            };
            let ql3 = if par == 0 {
                l3
            } else {
                (l3 >> 16) | (l4 << 16)
            };

            // qh window: 4 words at super-block byte 128 + 32*(w16>>3) + qhok.
            let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
            // SAFETY: same row bounds as the ql window; the qh section ends
            // at byte 192 of the super-block, before the scales.
            let (h0, h1, h2, h3, h4) = unsafe {
                (
                    *w.get_unchecked(hk),
                    *w.get_unchecked(hk + 1),
                    *w.get_unchecked(hk + 2),
                    *w.get_unchecked(hk + 3),
                    *w.get_unchecked(hk + 4),
                )
            };
            let qh0 = if par == 0 {
                h0
            } else {
                (h0 >> 16) | (h1 << 16)
            };
            let qh1 = if par == 0 {
                h1
            } else {
                (h1 >> 16) | (h2 << 16)
            };
            let qh2 = if par == 0 {
                h2
            } else {
                (h2 >> 16) | (h3 << 16)
            };
            let qh3 = if par == 0 {
                h3
            } else {
                (h3 >> 16) | (h4 << 16)
            };

            // scales (16 int8 at super-block byte 192) + d (f16 @208): one
            // 5-word window, k..k+4. Even: scales are words k..k+3 and d is
            // the low half of word k+4. Odd: the window is funneled and d is
            // the HIGH half of word k+4 (the same load f3 uses).
            let ak = (base + 192) >> 2;
            // SAFETY: ak+4 <= (row+1)*420 - 1: for the last super-block of
            // the last row, ak+4 is exactly the row's final word (bytes
            // 1676..1680); the launch contract bounds w.len() >= n_rows*420.
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
                    (a0 >> 16) | (a1 << 16),
                    (a1 >> 16) | (a2 << 16),
                    (a2 >> 16) | (a3 << 16),
                    (a3 >> 16) | (a4 << 16),
                    (a4 >> 16) as u16,
                )
            };
            // Sub-block scale: byte w16 of the scales window, signed.
            let sw = if w16 < 4 {
                sw0
            } else if w16 < 8 {
                sw1
            } else if w16 < 12 {
                sw2
            } else {
                sw3
            };
            let sc = ((sw >> (8 * (w16 & 3) as u32)) & 0xff) as u8 as i8 as i32;
            let drow = half_to_f32(d_bits);

            // q6 - 32 per byte: nibble | high-bits<<4, then one SWAR
            // subtract of 32 (borrow-free with the |0x80 bias).
            let vi0 = (((((ql0 >> nib_sh) & 0x0f0f0f0f) | (((qh0 >> hib_sh) & 0x03030303) << 4))
                | 0x80808080)
                .wrapping_sub(0x20202020))
                ^ 0x80808080;
            let vi1 = (((((ql1 >> nib_sh) & 0x0f0f0f0f) | (((qh1 >> hib_sh) & 0x03030303) << 4))
                | 0x80808080)
                .wrapping_sub(0x20202020))
                ^ 0x80808080;
            let vi2 = (((((ql2 >> nib_sh) & 0x0f0f0f0f) | (((qh2 >> hib_sh) & 0x03030303) << 4))
                | 0x80808080)
                .wrapping_sub(0x20202020))
                ^ 0x80808080;
            let vi3 = (((((ql3 >> nib_sh) & 0x0f0f0f0f) | (((qh3 >> hib_sh) & 0x03030303) << 4))
                | 0x80808080)
                .wrapping_sub(0x20202020))
                ^ 0x80808080;

            // q8_1 word base for column 0: this lane's 16 values sit in
            // words 4*(w16&7) .. +4 of block 2*sbp + w16/8 of the column.
            let uw = 32 * (2 * sbp + (w16 >> 3)) + 4 * (w16 & 7);
            let d8b = 2 * sbp + (w16 >> 3);

            // Column 0 (always active): one dp4a chain, one FMA.
            {
                // SAFETY: uw..uw+3 are this lane's four q8 words inside the
                // column's 512 words; d8b < 16 <= d8.len() by the contract.
                let (q0, q1, q2, q3, e0) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f0 += (a as f32) * (e0 * drow * sc as f32);
            }
            if m > 1 {
                // Columns 1..7 (launch-uniform guard), same shape: column c
                // reads q8 words uw + 512c .. +3 and block d8b + 16c.
                let mut uw = uw + 512;
                let mut d8b = d8b + 16;
                // SAFETY: m > 1 is launch-uniform, so the column-7 offsets
                // uw + 3584 + 3 < 512*8 and d8b + 112 < 128 stay in bounds.
                let (q0, q1, q2, q3, e1) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f1 += (a as f32) * (e1 * drow * sc as f32);

                uw += 512;
                d8b += 16;
                // SAFETY: column 2 of the m > 1 launch: same bounds
                // as column 1 (uw+3 < 512*3, d8b < 16*3).
                let (q0, q1, q2, q3, e2) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f2 += (a as f32) * (e2 * drow * sc as f32);

                uw += 512;
                d8b += 16;
                // SAFETY: column 3 of the m > 1 launch: same bounds
                // as column 1 (uw+3 < 512*4, d8b < 16*4).
                let (q0, q1, q2, q3, e3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f3 += (a as f32) * (e3 * drow * sc as f32);

                uw += 512;
                d8b += 16;
                // SAFETY: column 4 of the m > 1 launch: same bounds
                // as column 1 (uw+3 < 512*5, d8b < 16*5).
                let (q0, q1, q2, q3, e4) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f4 += (a as f32) * (e4 * drow * sc as f32);

                uw += 512;
                d8b += 16;
                // SAFETY: column 5 of the m > 1 launch: same bounds
                // as column 1 (uw+3 < 512*6, d8b < 16*6).
                let (q0, q1, q2, q3, e5) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f5 += (a as f32) * (e5 * drow * sc as f32);

                uw += 512;
                d8b += 16;
                // SAFETY: column 6 of the m > 1 launch: same bounds
                // as column 1 (uw+3 < 512*7, d8b < 16*7).
                let (q0, q1, q2, q3, e6) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f6 += (a as f32) * (e6 * drow * sc as f32);

                uw += 512;
                d8b += 16;
                // SAFETY: column 7 of the m > 1 launch: same bounds
                // as column 1 (uw+3 < 512*8, d8b < 16*8).
                let (q0, q1, q2, q3, e7) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 1),
                        *q.get_unchecked(uw + 2),
                        *q.get_unchecked(uw + 3),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = dp4a_s32(vi0, q0, 0);
                let a = dp4a_s32(vi1, q1, a);
                let a = dp4a_s32(vi2, q2, a);
                let a = dp4a_s32(vi3, q3, a);
                f7 += (a as f32) * (e7 * drow * sc as f32);
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant).
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
            let s2 = warp::reduce_sum_f32(f2);
            let s3 = warp::reduce_sum_f32(f3);
            let s4 = warp::reduce_sum_f32(f4);
            let s5 = warp::reduce_sum_f32(f5);
            let s6 = warp::reduce_sum_f32(f6);
            let s7 = warp::reduce_sum_f32(f7);
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m].
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    *y.get_unchecked_mut(b + 1) = s1;
                    *y.get_unchecked_mut(b + 2) = s2;
                    *y.get_unchecked_mut(b + 3) = s3;
                    *y.get_unchecked_mut(b + 4) = s4;
                    *y.get_unchecked_mut(b + 5) = s5;
                    *y.get_unchecked_mut(b + 6) = s6;
                    *y.get_unchecked_mut(b + 7) = s7;
                }
            }
        }
    }

    /// Q3_K (K=2048, 8 super-blocks per row) times q8_1 activations, M <= 8.
    /// ggml-mmvq-shaped redesign (round 2): the activation is quantized once
    /// to q8_1 by `q3k_quantize_q8_1`, so the inner product is a hardware
    /// `dp4a` over packed 4xint8 words instead of per-weight f32 FMAs, and x
    /// costs 1/4 the bytes. One warp owns one row; per iteration the warp
    /// covers two super-blocks (lanes 0..15 -> even sb, 16..31 -> odd sb) so
    /// every lane's 16-weight qs word is one u32 and the warp's weight loads
    /// are contiguous runs. Odd super-blocks sit 2 mod 4, so every word is
    /// assembled from two aligned u32 loads with a 16-bit funnel select.
    /// Each (word, field) quad is dequantized in registers with SWAR byte
    /// arithmetic (vi = vil - 4*(1-hbit) as signed bytes) and dotted with one
    /// u32 of q8_1 x via dp4a. The q8_1 block is the 128-value half
    /// super-block, the exact span of a lane's four fields (round 3): per
    /// column the four dp4a results are multiplied by their 6-bit sub-block
    /// scales and summed in int, then ONE f32 FMA applies the shared q8_1
    /// scale and the super-block scale (round 2 loaded d8 and multiplied
    /// per field: the structural M=8 cost). Reduced with a warp shuffle sum.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 220,
            q.len() >= m_cols * 512,
            d8.len() >= m_cols * 16,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q3k_gemv(
        w: &[u32],
        q: &[u32],
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

        // Per-lane constants: this lane's qs word within the super-block and
        // the derived scale/u-word/d8 bases (see the packing comment above).
        let w16 = lane & 15;
        let half = lane >> 4; // 0: even super-block, 1: odd
        let s0 = 8 * (w16 >> 3) + ((w16 & 7) >> 2); // first sub-block index
        let u_base = 32 * (w16 >> 3) + (w16 & 7); // u-word base, +8 per field
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
        while it < 4 {
            let sbp = ((it << 1) | half as u32) as usize;
            let base = row * 880 + sbp * 110; // byte offset of the super-block
            let par = sbp & 1; // 1: this sb's fields sit 2 mod 4 -> funnel

            // qs word: bytes base+32+4*w16 .. +3. `base+32+4*w16` is 0 mod 4
            // for even sbp and 2 mod 4 for odd, so the same floor division
            // addresses both; odd lanes reassemble with a 16-bit funnel.
            let qk = (base + 32 + 4 * w16) >> 2;
            // SAFETY: row < n_rows, sbp < 8, w16 < 16, so qk+1 < (row+1)*220
            // <= w.len() by the launch contract (word indices stay inside the
            // row's 220 words; odd super-blocks only shift the window by 2).
            let (lo, hi) = unsafe { (*w.get_unchecked(qk), *w.get_unchecked(qk + 1)) };
            let vl = if par == 0 {
                lo
            } else {
                (lo >> 16) | (hi << 16)
            };

            // hmask word (bytes base+4*(w16%8) .. +3), one bit per weight:
            // bit 4*(w16/8)+field. Invert so a clear hmask bit (subtract 4)
            // becomes a set bit, pre-shifted to bit 0 of each byte.
            let hk = (base + 4 * (w16 & 7)) >> 2;
            // SAFETY: same row/sbp/w16 bounds as the qs window above; hk+1 <
            // (row+1)*220 <= w.len().
            let (hlo, hhi) = unsafe { (*w.get_unchecked(hk), *w.get_unchecked(hk + 1)) };
            let hm = if par == 0 {
                hlo
            } else {
                (hlo >> 16) | (hhi << 16)
            };
            let vh1 = (!hm) >> (4 * (w16 >> 3)) as u32;

            // scales: 12 bytes at base+96..108, decoded with the aux[]
            // shuffle of dequantize_row_q3_K (verbatim from round 1). The
            // 16-byte window base+96..112 (even) / base+94..110 (odd) is
            // covered by four aligned words.
            let ak = (base + 96) >> 2;
            // SAFETY: ak+3 < (row+1)*220 <= w.len(): the 16-byte window ends
            // at most 2 bytes past the row end for the last super-block, but
            // the floored word index stays inside the row's 220 words.
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
                (
                    (aw0 >> 16) | (aw1 << 16),
                    (aw1 >> 16) | (aw2 << 16),
                    (aw2 >> 16) | (aw3 << 16),
                )
            };
            let kmask1 = 0x03030303u32;
            let kmask2 = 0x0f0f0f0fu32;
            let t0 = ((a0w >> 4) & kmask2) | (((a2w >> 4) & kmask1) << 4);
            let t1 = ((a1w >> 4) & kmask2) | (((a2w >> 6) & kmask1) << 4);
            let t2 = (a0w & kmask2) | ((a2w & kmask1) << 4);
            let t3 = (a1w & kmask2) | (((a2w >> 2) & kmask1) << 4);

            // Super-block scale d (f16 at bytes base+108..109): low half of
            // aw3 for even sbp (word covers 108..111), high half for odd
            // (word covers 106..109).
            let d_bits = if par == 0 {
                (aw3 & 0xffff) as u16
            } else {
                (aw3 >> 16) as u16
            };
            let drow = half_to_f32(d_bits);

            // Sub-block scales for fields 0..3: sub-block s0+2j, byte s&3 of
            // the t-word the if-chain picks (round-1 pattern, keeps constant
            // shifts out of local memory).
            let sx = s0;
            let wx = if sx < 4 {
                t2
            } else if sx < 8 {
                t3
            } else if sx < 12 {
                t0
            } else {
                t1
            };
            let sc0 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;
            let sx = s0 + 2;
            let wx = if sx < 4 {
                t2
            } else if sx < 8 {
                t3
            } else if sx < 12 {
                t0
            } else {
                t1
            };
            let sc1 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;
            let sx = s0 + 4;
            let wx = if sx < 4 {
                t2
            } else if sx < 8 {
                t3
            } else if sx < 12 {
                t0
            } else {
                t1
            };
            let sc2 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;
            let sx = s0 + 6;
            let wx = if sx < 4 {
                t2
            } else if sx < 8 {
                t3
            } else if sx < 12 {
                t0
            } else {
                t1
            };
            let sc3 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;

            // Dequantize the four quads to signed bytes (SWAR): vi byte b =
            // vil - 4*(1 - hbit). The |0x80 / ^0x80 bias makes the per-byte
            // subtract borrow-free; dp4a_s32 reads the bytes as signed.
            let vi0 = (((vl & 0x03030303) | 0x80808080).wrapping_sub((vh1 << 2) & 0x04040404))
                ^ 0x80808080;
            let vi1 = (((vl >> 2) & 0x03030303) | 0x80808080)
                .wrapping_sub(((vh1 >> 1) << 2) & 0x04040404)
                ^ 0x80808080;
            let vi2 = (((vl >> 4) & 0x03030303) | 0x80808080)
                .wrapping_sub(((vh1 >> 2) << 2) & 0x04040404)
                ^ 0x80808080;
            let vi3 = (((vl >> 6) & 0x03030303) | 0x80808080)
                .wrapping_sub(((vh1 >> 3) << 2) & 0x04040404)
                ^ 0x80808080;

            // u-word base for this word (column 0): 64 words per super-block.
            let uw = 64 * sbp + u_base;
            // d8 base for this lane (column 0): two 128-value q8_1 blocks
            // per super-block, this lane's fields all sit in block
            // 2*sbp + d8_base of the column.
            let d8b = 2 * sbp + d8_base;

            // Column 0 (always active). The lane's four fields share one
            // q8_1 block: int chain (dp4a x sub-block scale), one FMA with
            // the shared block scale and the super-block scale.
            {
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f0 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);
            }
            if m > 1 {
                // Columns 1..7 (launch-uniform guard, no divergence). Same
                // shape as column 0 with the per-column q/d8 offsets.
                let mut uw = uw + 512;
                let mut d8b = d8b + 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f1 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);

                uw += 512;
                d8b += 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f2 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);

                uw += 512;
                d8b += 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f3 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);

                uw += 512;
                d8b += 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f4 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);

                uw += 512;
                d8b += 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f5 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);

                uw += 512;
                d8b += 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f6 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);

                uw += 512;
                d8b += 16;
                let a = dp4a_s32(vi0, unsafe { *q.get_unchecked(uw) }, 0) * sc0
                    + dp4a_s32(vi1, unsafe { *q.get_unchecked(uw + 8) }, 0) * sc1
                    + dp4a_s32(vi2, unsafe { *q.get_unchecked(uw + 16) }, 0) * sc2
                    + dp4a_s32(vi3, unsafe { *q.get_unchecked(uw + 24) }, 0) * sc3;
                f7 += (a as f32) * (unsafe { *d8.get_unchecked(d8b) } * drow);
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant).
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
            let s2 = warp::reduce_sum_f32(f2);
            let s3 = warp::reduce_sum_f32(f3);
            let s4 = warp::reduce_sum_f32(f4);
            let s5 = warp::reduce_sum_f32(f5);
            let s6 = warp::reduce_sum_f32(f6);
            let s7 = warp::reduce_sum_f32(f7);
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m] (the pattern round 1 used).
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    *y.get_unchecked_mut(b + 1) = s1;
                    *y.get_unchecked_mut(b + 2) = s2;
                    *y.get_unchecked_mut(b + 3) = s3;
                    *y.get_unchecked_mut(b + 4) = s4;
                    *y.get_unchecked_mut(b + 5) = s5;
                    *y.get_unchecked_mut(b + 6) = s6;
                    *y.get_unchecked_mut(b + 7) = s7;
                }
            }
        }
    }
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert!(b.len() % 4 == 0, "odd size for {path}");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::env::var("MULLE_DATA").unwrap_or_else(|_| "/root/mulle-data".to_string());
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // The kernel reads the weight stream as aligned u32 words.
    let w_bytes = std::fs::read(format!("{data}/gate.q3k"))?;
    assert!(w_bytes.len() % 4 == 0, "weight bytes not u32-divisible");
    let w_host: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let x_m1 = read_f32(&format!("{data}/x_m1.f32"));
    let x_m8 = read_f32(&format!("{data}/x_m8.f32"));
    assert_eq!(x_m1.len(), 2048);
    assert_eq!(x_m8.len(), 2048 * 8);

    let w_dev = DeviceBuffer::from_host(&stream, &w_host)?;
    let x1_dev = DeviceBuffer::from_host(&stream, &x_m1)?;
    let x8_dev = DeviceBuffer::from_host(&stream, &x_m8)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };

    struct Shape {
        name: &'static str,
        n: usize,
        m: usize,
        wbytes: usize,
    }
    let shapes = [
        Shape {
            name: "expert0_m1",
            n: 1408,
            m: 1,
            wbytes: 1239040,
        },
        Shape {
            name: "expert0_m8",
            n: 1408,
            m: 8,
            wbytes: 1239040,
        },
        Shape {
            name: "stack_m1",
            n: 90112,
            m: 1,
            wbytes: 79298560,
        },
        Shape {
            name: "stack_m8",
            n: 90112,
            m: 8,
            wbytes: 79298560,
        },
    ];

    let mut all_ok = true;
    for sh in &shapes {
        let x_dev = if sh.m == 1 { &x1_dev } else { &x8_dev };
        let mut q_dev = DeviceBuffer::<u32>::zeroed(&stream, sh.m * 512)?;
        let mut d8_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.m * 16)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.n * sh.m)?;
        let prep_q =
            module.prepare_q3k_quantize_q8_1(LaunchConfig1D::new(16 * sh.m as u32, 32, 0))?;
        let prep_g =
            module.prepare_q3k_gemv(LaunchConfig1D::new(sh.n.div_ceil(8) as u32, 256, 0))?;
        // One timed iteration = quantize x + gemv, matching what ggml's
        // mul_mat graph does per call.
        let mut launch = |q_dev: &mut DeviceBuffer<u32>,
                          d8_dev: &mut DeviceBuffer<f32>|
         -> Result<(), Box<dyn std::error::Error>> {
            module.q3k_quantize_q8_1(&stream, &prep_q, x_dev, sh.m as u32, q_dev, d8_dev)?;
            module.q3k_gemv(
                &stream,
                &prep_g,
                &w_dev,
                q_dev,
                d8_dev,
                sh.n as u32,
                sh.m as u32,
                &mut y_dev,
            )?;
            Ok(())
        };
        for _ in 0..20 {
            launch(&mut q_dev, &mut d8_dev)?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            launch(&mut q_dev, &mut d8_dev)?;
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;

        let y = y_dev.to_host_vec(&stream)?;
        let yref = read_f32(&format!("{data}/y_ref_{}.f32", sh.name));
        let denom = yref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let maxerr = y
            .iter()
            .zip(yref.iter())
            .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
        let rel = maxerr / denom;
        let gbs = sh.wbytes as f64 / (us * 1e-6) / 1e9;
        println!(
            "shape {:<10} N={:>6} M={} weight_bytes={:>9} us={:>9.2} GB/s={:>7.2} max_rel_err={:.3e}",
            sh.name, sh.n, sh.m, sh.wbytes, us, gbs, rel
        );
        // q8_1-activation design gate (stated in RESULTS.md); ggml's own
        // mmvq error on these shapes is 3.7-5.2e-3.
        if rel > 1e-2 {
            eprintln!("FAIL: {} rel err {rel:.3e} exceeds 1e-2", sh.name);
            all_ok = false;
        }
    }

    // ---- Q4_K: the 27 blk.*.attn_output tensors [2048x2048], concatenated
    // in blk order by the harness into attn.q4k. attn0 benches the first
    // tensor alone; attnstk benches all 27 as one stack. Same x activations
    // and the same shared q8_1 quantize as Q3_K. ----
    let w4_bytes = std::fs::read(format!("{data}/attn.q4k"))?;
    assert!(w4_bytes.len() % 4 == 0, "attn.q4k not u32-divisible");
    assert_eq!(
        w4_bytes.len(),
        55296 * 1152,
        "attn.q4k vs 27 x [2048,2048] Q4_K rows"
    );
    let w4_host: Vec<u32> = w4_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w4_dev = DeviceBuffer::from_host(&stream, &w4_host)?;
    let shapes4 = [
        Shape {
            name: "attn0_m1",
            n: 2048,
            m: 1,
            wbytes: 2359296,
        },
        Shape {
            name: "attn0_m8",
            n: 2048,
            m: 8,
            wbytes: 2359296,
        },
        Shape {
            name: "attnstk_m1",
            n: 55296,
            m: 1,
            wbytes: 63700992,
        },
        Shape {
            name: "attnstk_m8",
            n: 55296,
            m: 8,
            wbytes: 63700992,
        },
    ];

    for sh in &shapes4 {
        let x_dev = if sh.m == 1 { &x1_dev } else { &x8_dev };
        let mut q_dev = DeviceBuffer::<u32>::zeroed(&stream, sh.m * 512)?;
        let mut d8_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.m * 16)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.n * sh.m)?;
        let prep_q =
            module.prepare_q3k_quantize_q8_1(LaunchConfig1D::new(16 * sh.m as u32, 32, 0))?;
        let prep_g =
            module.prepare_q4k_gemv(LaunchConfig1D::new(sh.n.div_ceil(8) as u32, 256, 0))?;
        let mut launch = |q_dev: &mut DeviceBuffer<u32>,
                          d8_dev: &mut DeviceBuffer<f32>|
         -> Result<(), Box<dyn std::error::Error>> {
            module.q3k_quantize_q8_1(&stream, &prep_q, x_dev, sh.m as u32, q_dev, d8_dev)?;
            module.q4k_gemv(
                &stream,
                &prep_g,
                &w4_dev,
                q_dev,
                d8_dev,
                sh.n as u32,
                sh.m as u32,
                &mut y_dev,
            )?;
            Ok(())
        };
        for _ in 0..20 {
            launch(&mut q_dev, &mut d8_dev)?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            launch(&mut q_dev, &mut d8_dev)?;
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;

        let y = y_dev.to_host_vec(&stream)?;
        let yref = read_f32(&format!("{data}/y_ref_{}.f32", sh.name));
        let denom = yref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let maxerr = y
            .iter()
            .zip(yref.iter())
            .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
        let rel = maxerr / denom;
        let gbs = sh.wbytes as f64 / (us * 1e-6) / 1e9;
        println!(
            "shape {:<10} N={:>6} M={} weight_bytes={:>9} us={:>9.2} GB/s={:>7.2} max_rel_err={:.3e}",
            sh.name, sh.n, sh.m, sh.wbytes, us, gbs, rel
        );
        // Same q8_1-activation design gate as the Q3_K shapes.
        if rel > 1e-2 {
            eprintln!("FAIL: {} rel err {rel:.3e} exceeds 1e-2", sh.name);
            all_ok = false;
        }
    }

    // ---- Q6_K: output.weight [2048x102400], the lm_head. One tensor, by
    // far the largest single gemv in the dense path. ----
    let w6_bytes = std::fs::read(format!("{data}/output.q6k"))?;
    assert!(w6_bytes.len() % 4 == 0, "output.q6k not u32-divisible");
    assert_eq!(
        w6_bytes.len(),
        102400 * 1680,
        "output.q6k vs [2048,102400] Q6_K rows"
    );
    let w6_host: Vec<u32> = w6_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w6_dev = DeviceBuffer::from_host(&stream, &w6_host)?;
    let shapes6 = [
        Shape {
            name: "head_m1",
            n: 102400,
            m: 1,
            wbytes: 172032000,
        },
        Shape {
            name: "head_m8",
            n: 102400,
            m: 8,
            wbytes: 172032000,
        },
    ];

    for sh in &shapes6 {
        let x_dev = if sh.m == 1 { &x1_dev } else { &x8_dev };
        let mut q_dev = DeviceBuffer::<u32>::zeroed(&stream, sh.m * 512)?;
        let mut d8_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.m * 16)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.n * sh.m)?;
        let prep_q =
            module.prepare_q3k_quantize_q8_1(LaunchConfig1D::new(16 * sh.m as u32, 32, 0))?;
        let prep_g =
            module.prepare_q6k_gemv(LaunchConfig1D::new(sh.n.div_ceil(8) as u32, 256, 0))?;
        let mut launch = |q_dev: &mut DeviceBuffer<u32>,
                          d8_dev: &mut DeviceBuffer<f32>|
         -> Result<(), Box<dyn std::error::Error>> {
            module.q3k_quantize_q8_1(&stream, &prep_q, x_dev, sh.m as u32, q_dev, d8_dev)?;
            module.q6k_gemv(
                &stream,
                &prep_g,
                &w6_dev,
                q_dev,
                d8_dev,
                sh.n as u32,
                sh.m as u32,
                &mut y_dev,
            )?;
            Ok(())
        };
        for _ in 0..20 {
            launch(&mut q_dev, &mut d8_dev)?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            launch(&mut q_dev, &mut d8_dev)?;
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;

        let y = y_dev.to_host_vec(&stream)?;
        let yref = read_f32(&format!("{data}/y_ref_{}.f32", sh.name));
        let denom = yref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let maxerr = y
            .iter()
            .zip(yref.iter())
            .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
        let rel = maxerr / denom;
        let gbs = sh.wbytes as f64 / (us * 1e-6) / 1e9;
        println!(
            "shape {:<10} N={:>6} M={} weight_bytes={:>9} us={:>9.2} GB/s={:>7.2} max_rel_err={:.3e}",
            sh.name, sh.n, sh.m, sh.wbytes, us, gbs, rel
        );
        // Same q8_1-activation design gate as the Q3_K shapes.
        if rel > 1e-2 {
            eprintln!("FAIL: {} rel err {rel:.3e} exceeds 1e-2", sh.name);
            all_ok = false;
        }
    }

    if !all_ok {
        std::process::exit(1);
    }
    println!("PASSED: all 10 shapes (Q3_K/Q4_K/Q6_K) within 1e-2 (q8_1 activation design)");
    Ok(())
}
