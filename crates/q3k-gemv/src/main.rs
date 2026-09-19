use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    dotprod::dp4a_s32, DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp,
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

    /// Quantize f32 activations to q8_1: per 32-value block, d = amax/127 and
    /// int8 q = round(x/d). One warp per block; the 32 lanes pack their bytes
    /// into 8 u32 words with two shuffle_downs so the gemv reads whole words.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (x.len() >= m_cols * 2048, q.len() >= m_cols * 512, d8.len() >= m_cols * 64)
    )]
    pub fn q3k_quantize_q8_1(x: &[f32], m_cols: u32, mut q: DisjointSlice<u32>, mut d8: DisjointSlice<f32>) {
        // One 32-thread block per 32-value quant block: blk is the BLOCK
        // index (global tid / 32), not the thread id.
        let blk = thread::index_1d().get() / 32;
        let total = m_cols as usize * 64;
        if blk >= total {
            return;
        }
        let col = blk / 64;
        let b = blk % 64;
        let lane = warp::lane_id() as usize;

        // SAFETY: col < m_cols, b < 64, lane < 32, so the index is
        // below m_cols*2048 <= x.len() by the launch contract.
        let v = unsafe { *x.get_unchecked(col * 2048 + 32 * b + lane) };
        let amax = warp::reduce_max_f32(v.abs());
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let qv = (v / d).round().clamp(-127.0, 127.0);
        let byte = (qv as i32 as u32) & 0xff;

        // Pack lane bytes into u32 words: lane 4g ends up holding bytes
        // 4g..4g+3 of the block. Out-of-range shuffle lanes keep their own
        // value; only lanes with lane%4==0 store.
        let pair = byte | (warp::shuffle_down(byte, 1) << 8);
        let quad = pair | (warp::shuffle_down(pair, 2) << 16);
        if lane % 4 == 0 {
            // SAFETY: lanes with lane%4==0 write disjoint words
            // col*512 + 8*b + lane/4; q holds m_cols*512 words.
            unsafe {
                *q.get_unchecked_mut(col * 512 + 8 * b + lane / 4) = quad;
            }
        }
        if lane == 0 {
            // SAFETY: lane 0 of each warp writes its own d8 slot.
            unsafe {
                *d8.get_unchecked_mut(col * 64 + b) = d;
            }
        }
    }

    /// Q3_K (K=2048, 8 super-blocks per row) times q8_1 activations, M <= 8.
    ///
    /// ggml-mmvq-shaped redesign (round 2): the activation is quantized once
    /// to q8_1 by `q3k_quantize_q8_1`, so the inner product is a hardware
    /// `dp4a` over packed 4xint8 words instead of per-weight f32 FMAs, and x
    /// costs 1/4 the bytes. One warp owns one row; per iteration the warp
    /// covers two super-blocks (lanes 0..15 -> even sb, 16..31 -> odd sb) so
    /// every lane's 16-weight qs word is one u32 and the warp's weight loads
    /// are contiguous runs. Odd super-blocks sit 2 mod 4, so every word is
    /// assembled from two aligned u32 loads with a 16-bit funnel select.
    /// Each (word, field) quad is dequantized in registers with SWAR byte
    /// arithmetic (vi = vil - 4*(1-hbit) as signed bytes), dotted with one
    /// u32 of q8_1 x via dp4a, scaled by the 6-bit sub-block scale and the
    /// q8_1 block scale in f32, and reduced with a warp shuffle sum.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 220,
            q.len() >= m_cols * 512,
            d8.len() >= m_cols * 64,
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
        let d8_base = 4 * (w16 >> 3); // d8 block base, +1 per field

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
            let vl = if par == 0 { lo } else { (lo >> 16) | (hi << 16) };

            // hmask word (bytes base+4*(w16%8) .. +3), one bit per weight:
            // bit 4*(w16/8)+field. Invert so a clear hmask bit (subtract 4)
            // becomes a set bit, pre-shifted to bit 0 of each byte.
            let hk = (base + 4 * (w16 & 7)) >> 2;
            // SAFETY: same row/sbp/w16 bounds as the qs window above; hk+1 <
            // (row+1)*220 <= w.len().
            let (hlo, hhi) = unsafe { (*w.get_unchecked(hk), *w.get_unchecked(hk + 1)) };
            let hm = if par == 0 { hlo } else { (hlo >> 16) | (hhi << 16) };
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
            let d_bits = if par == 0 { (aw3 & 0xffff) as u16 } else { (aw3 >> 16) as u16 };
            let drow = half_to_f32(d_bits);

            // Sub-block scales for fields 0..3: sub-block s0+2j, byte s&3 of
            // the t-word the if-chain picks (round-1 pattern, keeps constant
            // shifts out of local memory).
            let sx = s0;
            let wx = if sx < 4 { t2 } else if sx < 8 { t3 } else if sx < 12 { t0 } else { t1 };
            let sc0 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;
            let sx = s0 + 2;
            let wx = if sx < 4 { t2 } else if sx < 8 { t3 } else if sx < 12 { t0 } else { t1 };
            let sc1 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;
            let sx = s0 + 4;
            let wx = if sx < 4 { t2 } else if sx < 8 { t3 } else if sx < 12 { t0 } else { t1 };
            let sc2 = (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32;
            let sx = s0 + 6;
            let wx = if sx < 4 { t2 } else if sx < 8 { t3 } else if sx < 12 { t0 } else { t1 };
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
            let d8b = 8 * sbp + d8_base;

            // Column 0 (always active). Each field j has its own q8_1 block
            // (u-words step by 8 = one block), so its d8 scale too. The
            // super-block scale drow is hoisted: sum the four d8-scaled
            // fields first, multiply by drow once (ggml's shape).
            {
                // SAFETY: sbp < 8 and u_base < 40, so uw+24 < 512 and
                // d8b+3 < 64; column 0 is inside q (m*512 words) and d8
                // (m*64) for any m >= 1 by the launch contract.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p0 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f0 += p0 * drow;
            }
            if m > 1 {
                // Columns 1..7 (launch-uniform guard, no divergence). Same
                // shape as column 0 with the per-column q/d8 offsets.
                let mut uw = uw + 512;
                let mut d8b = d8b + 64;
                // SAFETY: this branch runs only for m == 8 (host launches 1
                // or 8); column 1 sits at 512..1024 < 8*512 in q and 64..128
                // < 8*64 in d8, inside the launch-contract bounds.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p1 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f1 += p1 * drow;

                uw += 512;
                d8b += 64;
                // SAFETY: column 2 of m == 8; same bounds argument as column 1.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p2 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f2 += p2 * drow;

                uw += 512;
                d8b += 64;
                // SAFETY: column 3 of m == 8; same bounds argument as column 1.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p3 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f3 += p3 * drow;

                uw += 512;
                d8b += 64;
                // SAFETY: column 4 of m == 8; same bounds argument as column 1.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p4 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f4 += p4 * drow;

                uw += 512;
                d8b += 64;
                // SAFETY: column 5 of m == 8; same bounds argument as column 1.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p5 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f5 += p5 * drow;

                uw += 512;
                d8b += 64;
                // SAFETY: column 6 of m == 8; same bounds argument as column 1.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p6 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f6 += p6 * drow;

                uw += 512;
                d8b += 64;
                // SAFETY: column 7 of m == 8; same bounds argument as column 1.
                let (u0, u1, u2, u3) = unsafe {
                    (
                        *q.get_unchecked(uw),
                        *q.get_unchecked(uw + 8),
                        *q.get_unchecked(uw + 16),
                        *q.get_unchecked(uw + 24),
                    )
                };
                let (e0, e1, e2, e3) = unsafe {
                    (
                        *d8.get_unchecked(d8b),
                        *d8.get_unchecked(d8b + 1),
                        *d8.get_unchecked(d8b + 2),
                        *d8.get_unchecked(d8b + 3),
                    )
                };
                let p7 = (dp4a_s32(vi0, u0, 0) * sc0) as f32 * e0
                    + (dp4a_s32(vi1, u1, 0) * sc1) as f32 * e1
                    + (dp4a_s32(vi2, u2, 0) * sc2) as f32 * e2
                    + (dp4a_s32(vi3, u3, 0) * sc3) as f32 * e3;
                f7 += p7 * drow;
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
        Shape { name: "expert0_m1", n: 1408, m: 1, wbytes: 1239040 },
        Shape { name: "expert0_m8", n: 1408, m: 8, wbytes: 1239040 },
        Shape { name: "stack_m1", n: 90112, m: 1, wbytes: 79298560 },
        Shape { name: "stack_m8", n: 90112, m: 8, wbytes: 79298560 },
    ];

    let mut all_ok = true;
    for sh in &shapes {
        let x_dev = if sh.m == 1 { &x1_dev } else { &x8_dev };
        let mut q_dev = DeviceBuffer::<u32>::zeroed(&stream, sh.m * 512)?;
        let mut d8_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.m * 64)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.n * sh.m)?;
        let prep_q = module.prepare_q3k_quantize_q8_1(LaunchConfig1D::new(
            64 * sh.m as u32,
            32,
            0,
        ))?;
        let prep_g = module.prepare_q3k_gemv(LaunchConfig1D::new(
            sh.n.div_ceil(8) as u32,
            256,
            0,
        ))?;
        // One timed iteration = quantize x + gemv, matching what ggml's
        // mul_mat graph does per call.
        let mut launch = |q_dev: &mut DeviceBuffer<u32>, d8_dev: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn std::error::Error>> {
            module.q3k_quantize_q8_1(&stream, &prep_q, x_dev, sh.m as u32, q_dev, d8_dev)?;
            module.q3k_gemv(&stream, &prep_g, &w_dev, q_dev, d8_dev, sh.n as u32, sh.m as u32, &mut y_dev)?;
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
    if !all_ok {
        std::process::exit(1);
    }
    println!("PASSED: all 4 shapes within 1e-2 (q8_1 activation design)");
    Ok(())
}
