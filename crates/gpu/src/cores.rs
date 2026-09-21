//! Device-callable kernel cores, kept outside any `#[cuda_module]` so more
//! than one kernel — a per-op wrapper and a fused block kernel — can call the
//! same body (docs/gpu-design.md decision 6). The codegen backend compiles
//! whatever a `#[kernel]` reaches, so these are ordinary functions; they must
//! stay free of host-only constructs (allocation, panicking bounds checks on
//! the hot path, std I/O).
//!
//! Contract: cores take values and return values; wrappers read thread/lane
//! indices and bounds, perform every memory access, and store. The one
//! established exception is `q4k_a_chain`, which loads its own q8 window and
//! predates the rule (measured on the 3090 with the loads inside).

use cuda_device::dotprod::dp4a_s32;

// ---- shared ----

/// IEEE-754 half to float, integer-only so no `f16` feature gate is needed
/// on either side of the unified compilation.
#[inline(always)]
pub fn half_to_f32(bits: u16) -> f32 {
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

/// The u32 at a byte window sitting 2 mod 4, assembled from the two aligned
/// words that cover it (Q3_K and Q6_K super-blocks are 110/210 bytes).
#[inline(always)]
pub fn funnel16(lo: u32, hi: u32) -> u32 {
    (lo >> 16) | (hi << 16)
}

// ---- q8_1 activation quantizer (q3k_quantize_q8_1) ----

/// One lane's four consecutive values of a 128-value block, quantized to
/// q8_1 with the block scale d = amax/127 (the caller reduces amax over the
/// warp): the packed little-endian byte word, and the lane-local signed
/// quad sum the s8 butterfly starts from.
#[inline(always)]
pub fn q8_quad(v: [f32; 4], d: f32) -> (u32, i32) {
    let q0 = ((v[0] / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let q1 = ((v[1] / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let q2 = ((v[2] / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let qw = ((v[3] / d).round().clamp(-127.0, 127.0) as i32 as u32) & 0xff;
    let word = q0 | (q1 << 8) | (q2 << 16) | (qw << 24);
    let g = (q0 as i8 as i32) + (q1 as i8 as i32) + (q2 as i8 as i32) + (qw as i8 as i32);
    (word, g)
}

/// q3 slot of value-order word v4: the gemv reads the u64 pair at slot
/// 64*grp + 32p + lane, so the permutation is per 2-super-block group;
/// v4's bit 3 selects the lo/hi half of the slot and is dropped here (the
/// store combines this lane's word with its lane^8 partner's).
#[inline(always)]
pub fn q3_slot(v4: u32) -> u32 {
    64 * (v4 >> 7) + 32 * ((v4 >> 4) & 1) + 16 * ((v4 >> 6) & 1) + 8 * ((v4 >> 5) & 1) + (v4 & 7)
}

/// q4 slot of v4: the gemv reads word i of iteration it at 256*it + 32*i +
/// lane — a permutation per 4-super-block group.
#[inline(always)]
pub fn q4_slot(v4: u32) -> u32 {
    256 * (v4 >> 8) + 32 * (v4 & 7) + 8 * ((v4 >> 6) & 3) + ((v4 >> 3) & 7)
}

/// q6 slot of v4: the gemv reads word i of iteration it at 128*it + 32*i +
/// lane — a permutation per 2-super-block group.
#[inline(always)]
pub fn q6_slot(v4: u32) -> u32 {
    128 * (v4 >> 7) + 32 * (v4 & 3) + 16 * ((v4 >> 6) & 1) + ((v4 >> 2) & 15)
}

// ---- Q4_K ----

/// SWAR nibble decode: q4k weight nibble - 8 per byte (see the Q3_K
/// bias trick). Called eight times per iteration on the hoisted qs
/// words instead of once per (column, word) — the per-column re-decode
/// was half of q4k's per-column instruction count and with it twice
/// attnstk's M>1 marginal cost (MUL-8).
#[inline(always)]
pub fn q4k_nibble(qsw: u32, nib_sh: u32) -> u32 {
    ((((qsw >> nib_sh) & 0x0f0f0f0f) | 0x80808080).wrapping_sub(0x08080808)) ^ 0x80808080
}

/// One lane's A chain: its eight hoisted vi words against the
/// q4-permuted q8 window at base `qb`, word i at qb + 32i (so each of
/// the eight loads is 32 lane-consecutive words across the warp).
/// SAFETY: callers keep `qb + 7*32` inside one column's q8 words.
#[inline(always)]
pub fn q4k_a_chain(vi: &[u32; 8], q: &[u32], qb: usize) -> i32 {
    // SAFETY: qb + 224 stays inside the caller's column span by this fn's
    // contract: within quad group g the largest qb is 256*g + 31, so the
    // last load is at 256*g + 255 < 256*ceil(n_sb/4), the column's words.
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
    let a = dp4a_s32(vi[0], w0, 0);
    let a = dp4a_s32(vi[1], w1, a);
    let a = dp4a_s32(vi[2], w2, a);
    let a = dp4a_s32(vi[3], w3, a);
    let a = dp4a_s32(vi[4], w4, a);
    let a = dp4a_s32(vi[5], w5, a);
    let a = dp4a_s32(vi[6], w6, a);
    dp4a_s32(vi[7], w7, a)
}

/// get_scale_min_k4(s) against the three scale words of the super-block
/// (scales[12] @ +4): scales bytes s and s+4 live in (w1,w2) for s<4 and
/// (w2,w3) for s>=4; the s>=4 branch borrows bytes s-4 and s for the 6-bit
/// high parts.
#[inline(always)]
pub fn q4k_scale_min(s: usize, w1: u32, w2: u32, w3: u32) -> (i32, i32) {
    if s < 4 {
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
    }
}

/// Column-independent chain coefficients of one sub-block: the dot on
/// quantized activations is d8*(cda*A + cdb*B) with A = dp4a(nib-8, q8)
/// and B = sum q8 — the -8 offset moves 8*B between the chains.
#[inline(always)]
pub fn q4k_coeff(d: f32, dmin: f32, sc: i32, mi: i32) -> (f32, f32) {
    let cda = d * sc as f32;
    let cdb = 8.0 * d * sc as f32 - dmin * mi as f32;
    (cda, cdb)
}

// ---- Q3_K ----

/// The aux[] shuffle of ggml's dequantize_row_q3_K over the three funneled
/// scale words (12 scale bytes at super-block +96): t0..t3 hold the 16
/// sub-block scale bytes, the 2-bit high parts borrowed from a2w.
#[inline(always)]
pub fn q3k_aux_scales(a0w: u32, a1w: u32, a2w: u32) -> [u32; 4] {
    let kmask1 = 0x03030303u32;
    let kmask2 = 0x0f0f0f0fu32;
    [
        ((a0w >> 4) & kmask2) | (((a2w >> 4) & kmask1) << 4),
        ((a1w >> 4) & kmask2) | (((a2w >> 6) & kmask1) << 4),
        (a0w & kmask2) | ((a2w & kmask1) << 4),
        (a1w & kmask2) | (((a2w >> 2) & kmask1) << 4),
    ]
}

/// Sub-block scale for sub-block sx: byte sx&3 of the t-word sx selects
/// (the 4-way if-chain keeps constant shifts out of local memory), minus
/// the 32 offset folded into every Q3_K scale byte.
#[inline(always)]
pub fn q3k_sub_scale(t: &[u32; 4], sx: usize) -> i32 {
    let wx = if sx < 4 {
        t[2]
    } else if sx < 8 {
        t[3]
    } else if sx < 12 {
        t[0]
    } else {
        t[1]
    };
    (((wx >> (8 * (sx & 3))) & 0xff) as u8 as i8 as i32) - 32
}

/// SWAR dequant of the lane's four field quads to signed bytes: vi byte b =
/// vil - 4*(1 - hbit). The |0x80 / ^0x80 bias makes the per-byte subtract
/// borrow-free; dp4a_s32 reads the bytes as signed.
#[inline(always)]
pub fn q3k_dequant(vl: u32, vh1: u32) -> [u32; 4] {
    [
        (((vl & 0x03030303) | 0x80808080).wrapping_sub((vh1 << 2) & 0x04040404)) ^ 0x80808080,
        (((vl >> 2) & 0x03030303) | 0x80808080).wrapping_sub(((vh1 >> 1) << 2) & 0x04040404)
            ^ 0x80808080,
        (((vl >> 4) & 0x03030303) | 0x80808080).wrapping_sub(((vh1 >> 2) << 2) & 0x04040404)
            ^ 0x80808080,
        (((vl >> 6) & 0x03030303) | 0x80808080).wrapping_sub(((vh1 >> 3) << 2) & 0x04040404)
            ^ 0x80808080,
    ]
}

/// One lane's whole per-column integer chain: four dp4a against the two u64
/// pair slots of the q3 permutation (fields 0/1 in w01 lo/hi, 2/3 in w23),
/// each scaled in int by its sub-block scale so one f32 FMA per column
/// carries both shared scales.
#[inline(always)]
pub fn q3k_chain(vi: &[u32; 4], w01: u64, w23: u64, sc: &[i32; 4]) -> i32 {
    dp4a_s32(vi[0], w01 as u32, 0) * sc[0]
        + dp4a_s32(vi[1], (w01 >> 32) as u32, 0) * sc[1]
        + dp4a_s32(vi[2], w23 as u32, 0) * sc[2]
        + dp4a_s32(vi[3], (w23 >> 32) as u32, 0) * sc[3]
}

// ---- Q6_K ----

/// Sub-block scale: byte w16 of the four funneled scale words, signed.
#[inline(always)]
pub fn q6k_sub_scale(sw: &[u32; 4], w16: usize) -> i32 {
    let wx = if w16 < 4 {
        sw[0]
    } else if w16 < 8 {
        sw[1]
    } else if w16 < 12 {
        sw[2]
    } else {
        sw[3]
    };
    ((wx >> (8 * (w16 & 3) as u32)) & 0xff) as u8 as i8 as i32
}

/// SWAR dequant of one ql word against its qh word: q6 - 32 per byte
/// (nibble | high-bits<<4, then one borrow-free subtract with the |0x80
/// bias).
#[inline(always)]
pub fn q6k_dequant(ql: u32, qh: u32, nib_sh: u32, hib_sh: u32) -> u32 {
    (((((ql >> nib_sh) & 0x0f0f0f0f) | (((qh >> hib_sh) & 0x03030303) << 4)) | 0x80808080)
        .wrapping_sub(0x20202020))
        ^ 0x80808080
}

/// One lane's whole per-column dp4a chain over its four vi words — no B
/// chain, no min term.
#[inline(always)]
pub fn q6k_chain(vi: &[u32; 4], q: &[u32; 4]) -> i32 {
    let a = dp4a_s32(vi[0], q[0], 0);
    let a = dp4a_s32(vi[1], q[1], a);
    let a = dp4a_s32(vi[2], q[2], a);
    dp4a_s32(vi[3], q[3], a)
}
