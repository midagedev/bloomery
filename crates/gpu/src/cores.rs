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

use cuda_device::convert::cvt_f32_f16x2_lo;
use cuda_device::dotprod::dp4a_s32;

// ---- shared ----

/// IEEE-754 half to float, integer-only so no `f16` feature gate is needed
/// on either side of the unified compilation, and bit-identical to
/// `gguf::quant::half_to_f32` — the transcription gate-1-1 pins against
/// ggml's own table — on all 65,536 patterns, NaN payloads included.
/// `gate_p4`'s `half_decode` asserts that whole space. The hardware's
/// one-instruction `cvt.f32.f16` (`flash::half_bits_to_f32`) agrees on every
/// finite and infinite input but canonicalizes NaN payloads.
///
/// That makes the hardware convert a substitute exactly where the value
/// cannot be a NaN, and nowhere else. `q3k_sb_decode` takes it for the Q3_K
/// super-block scale (2026-09-21, q3kdec round): a scale read from a GGUF is
/// finite or the model's output is not a number, and ik reads the same field
/// with `__half2float`. Q4_K's `d`/`dmin`, Q6_K's inline decode and
/// `gate_p4`'s whole-space assertion still call this function, and
/// `gate_p6`'s `q3k_half_decode_shape` is what keeps the split from drifting
/// silently — no `clz` in the Q3_K entries, `clz` still present in the
/// others. A site where the payload could carry meaning keeps the software
/// path.
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

/// Everything one Q4_K super-block yields before a column is named: the
/// eight SWAR-decoded qs words every column's A chain reuses, and the two
/// chain coefficients of this lane's sub-block. `wk` is the super-block's
/// word base in `w` and `s` the lane's sub-block within it. The whole
/// decode lives here so the single-column body and the m-column body cannot
/// drift apart.
///
/// SAFETY: callers keep `wk + 35` — the super-block's last word — inside
/// `w`.
#[inline(always)]
pub fn q4k_sb_decode(w: &[u32], wk: usize, s: usize) -> ([u32; 8], f32, f32) {
    // d/dmin word + the 12 scale bytes in words 1..3.
    // SAFETY: wk + 3 <= wk + 35, inside `w` by this fn's contract.
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

    // qs word base: 8 words from super-block word 4 + 8*(s>>1); nibble
    // select is the sub-block parity, 0 (low) or 4 (high). Hoisted
    // (MUL-8): the window and its SWAR decode are column-independent, so
    // decode once per iteration and let every column's A chain reuse the
    // registers.
    let qsk = wk + 4 + 8 * (s >> 1);
    let nib_sh = (s as u32 & 1) * 4;
    // SAFETY: qsk + 7 <= wk + 35, the same bound as w0..w3 above.
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
    (vi, cda, cdb)
}

/// One iteration's term of a single-column Q4_K row walk: the super-block
/// `sbp`'s decode, this lane's A chain against column `col0`'s q8 window,
/// and the sub-block sum, as the one value the walk adds to its
/// accumulator. Every load the iteration makes is in here, and none of the
/// walk's accumulation is — which is what lets the walk issue several
/// iterations' loads before the first add.
///
/// SAFETY: callers guarantee `sbp < n_sb` and the buffer lengths of
/// [`q4k_row_dot`] at `m_cols` 1.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn q4k_iter_term(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    it: u32,
    row_abs: usize,
    qb0: usize,
    s8b0: usize,
    d8b0: usize,
    s: usize,
    grp: usize,
    lane: usize,
) -> f32 {
    let sbp = 4 * it as usize + grp;
    // SAFETY: sbp < n_sb by this fn's contract, so the super-block's words
    // wk..wk+35 are inside row `row_abs` of `w`.
    let (vi, cda, cdb) = q4k_sb_decode(w, row_abs * 36 * n_sb + 36 * sbp, s);
    let qb = 256 * it as usize + lane;
    let s8b = 32 * it as usize + lane;
    let d8b = 2 * sbp + (s >> 2);
    let a = q4k_a_chain(&vi, q, qb0 + qb);
    // SAFETY: sbp < n_sb keeps s8b < 8*n_sb, and s8.len() >=
    // (col0+1)*8*n_sb by the contract.
    let b = unsafe { *s8.get_unchecked(s8b0 + s8b) };
    // SAFETY: sbp < n_sb keeps d8b < 2*n_sb, and d8.len() >=
    // (col0+1)*2*n_sb by the contract.
    let e0 = unsafe { *d8.get_unchecked(d8b0 + d8b) };
    (a as f32 * cda + b as f32 * cdb) * e0
}

/// Iterations of the single-column Q4_K walk whose loads are issued before
/// the first multiply-add that consumes them. The walk is a dependent chain
/// — every iteration's addresses are known in advance but its term feeds
/// the next add — so with one iteration in flight a lane can pay a memory
/// round trip per iteration. Issuing this many iterations' loads first is
/// what turns that walk into a bandwidth problem; it does not touch the
/// accumulation order, which stays increasing in the iteration index.
///
/// Q3_K has no such constant on purpose: the same lever measured flat on
/// its walk, so [`q3k_row_dot_1col`] keeps one iteration per pass. The two
/// walks load alike per weight — what differs is which one was waiting.
pub const Q4K_ITER_UNROLL: u32 = 2;

/// One row's Q4_K dot product against a single activation column —
/// [`q4k_row_dot`]'s column 0, the same loads and the same accumulation
/// order, with the column guards gone at compile time. The guards are
/// runtime tests, so in the m-column walk each iteration's work sits behind
/// its own branch and a lane can have only that iteration's loads in
/// flight. This body runs the iterations whose super-blocks are live for
/// every lane — `n_sb/4` of them, no `sbp` test — [`Q4K_ITER_UNROLL`] at a
/// time, and leaves the guarded walk as the tail.
///
/// Caller contract as [`q4k_row_dot`] with `m_cols` 1.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn q4k_row_dot_1col(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    lane: usize,
) -> f32 {
    let q_col = 256 * iters as usize; // q8 words per column
    let s8_col = 8 * n_sb; // 32-value groups per column
    let d8_col = 2 * n_sb; // 128-value blocks per column

    let s = lane & 7; // sub-block within the super-block
    let grp = lane >> 3; // super-block within this iteration (0..4)

    let mut f0 = 0.0f32;

    let qb0 = col0 * q_col;
    let s8b0 = col0 * s8_col;
    let d8b0 = col0 * d8_col;

    // Iterations every lane takes: `sbp = 4*it + grp <= 4*full - 1 <= n_sb
    // - 1` for `it < full`, whatever `grp` is, so the walk's guard is dead
    // here and the backend sees one straight counted loop.
    let full = (n_sb / 4) as u32;
    let mut it: u32 = 0;
    while it + Q4K_ITER_UNROLL <= full {
        // SAFETY: it + Q4K_ITER_UNROLL <= full, so each term's `sbp < n_sb`;
        // the rest of the contract is this fn's.
        let t0 = q4k_iter_term(
            w, q, s8, d8, n_sb, it, row_abs, qb0, s8b0, d8b0, s, grp, lane,
        );
        let t1 = q4k_iter_term(
            w,
            q,
            s8,
            d8,
            n_sb,
            it + 1,
            row_abs,
            qb0,
            s8b0,
            d8b0,
            s,
            grp,
            lane,
        );
        f0 += t0;
        f0 += t1;
        it += Q4K_ITER_UNROLL;
    }
    while it < iters {
        let sbp = 4 * it as usize + grp;
        // As `q4k_row_dot`: the guard makes a partial final iteration safe
        // and is always true when n_sb is a multiple of 4.
        if sbp < n_sb {
            // SAFETY: the guard is this term's `sbp < n_sb` precondition.
            f0 += q4k_iter_term(
                w, q, s8, d8, n_sb, it, row_abs, qb0, s8b0, d8b0, s, grp, lane,
            );
        }

        it += 1;
    }

    f0
}

/// One row's Q4_K dot products with `m_cols` (1..=8) activation columns,
/// pre-reduction: lane `lane` of the row's warp walks `iters`
/// four-super-block iterations keeping per-column partial sums in scalars
/// (the qs window decoded once per iteration and shared by every column —
/// the per-op gemv's own shape), and the CALLER reduces the returned
/// partials across the warp. The body is `q4k_gemv`'s, so a fused kernel
/// that calls this one agrees with the per-op gemv bit for bit by
/// construction.
///
/// Buffer layouts as `q4k_gemv` (see the Q4_K packing recap there): `w`
/// holds rows of `36 * n_sb` u32 words (row `row_abs` based at word
/// `row_abs * 36 * n_sb`; `row_abs` may be an indirect expert row), `q`
/// holds per column `256 * iters` u32 in the quantizer's q4 permutation,
/// `s8` per column `8 * n_sb` 32-value group sums and `d8` per column
/// `2 * n_sb` block scales. Column c reads `(col0 + c)` of each.
///
/// SAFETY: callers guarantee `w.len() >= (row_abs + 1) * 36 * n_sb`,
/// `q.len() >= (col0 + m_cols) * 256 * iters`, `s8.len() >= (col0 +
/// m_cols) * 8 * n_sb`, `d8.len() >= (col0 + m_cols) * 2 * n_sb`,
/// `iters = n_sb.div_ceil(4)`, and invoke this from all 32 lanes of one
/// warp.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn q4k_row_dot(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    m_cols: usize,
    lane: usize,
) -> [f32; 8] {
    // One column is the decode shape, and it gets a body of its own: the
    // guards below are runtime tests, so in this walk each iteration's work
    // sits behind its own branch. See `q4k_row_dot_1col`.
    if m_cols == 1 {
        return [
            q4k_row_dot_1col(w, q, s8, d8, n_sb, iters, row_abs, col0, lane),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
    }
    let m = m_cols;
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

    let qb0 = col0 * q_col;
    let s8b0 = col0 * s8_col;
    let d8b0 = col0 * d8_col;

    let mut it: u32 = 0;
    while it < iters {
        let sbp = 4 * it as usize + grp;
        // The sbp guard makes a partial final iteration safe: guarded
        // lanes load nothing of w/q/s8/d8. It is always true when n_sb
        // is a multiple of 4.
        if sbp < n_sb {
            // The super-block's column-independent decode — the eight
            // hoisted qs words and the chain coefficients — is
            // `q4k_sb_decode`, shared with `q4k_row_dot_1col`.
            // SAFETY: sbp < n_sb, so the super-block's words wk..wk+35 are
            // inside row `row_abs` of `w` by this fn's contract.
            let (vi, cda, cdb) = q4k_sb_decode(w, row_abs * row_words + 36 * sbp, s);

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
                let a = q4k_a_chain(&vi, q, qb0 + qb);
                // SAFETY: the sbp guard keeps s8b < s8_col and
                // d8b < d8_col; s8.len() >= (col0+1)*s8_col and d8.len() >=
                // (col0+1)*d8_col by the contract (m >= 1).
                let b = unsafe { *s8.get_unchecked(s8b0 + s8b) };
                let e0 = unsafe { *d8.get_unchecked(d8b0 + d8b) };
                f0 += (a as f32 * cda + b as f32 * cdb) * e0;
            }
            // Columns 1..7, one launch-uniform guard per column so the
            // work scales with m (MUL-8 amortization curve). Column c
            // reads q8 words at q_col*c + qb (q4k_a_chain adds 32i), the
            // s8 group at s8_col*c + s8b and block d8b + d8_col*c.
            // SAFETY: guard c+1 means m >= c+1 is launch-uniform, so the
            // buffer lengths ((col0+m)*q_col / *s8_col / *d8_col) cover
            // every offset below and the branch never diverges within a
            // warp.
            if m > 1 {
                // SAFETY: m > 1 => q.len() >= (col0+2)*q_col >
                // qb0 + 1*q_col + qb + 224, s8.len() >= (col0+2)*s8_col >
                // s8b0 + 1*s8_col + s8b, d8.len() >= (col0+2)*d8_col >
                // d8b0 + d8b + 1*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + s8_col + s8b) };
                let e1 = unsafe { *d8.get_unchecked(d8b0 + d8b + d8_col) };
                f1 += (a as f32 * cda + b as f32 * cdb) * e1;
            }
            if m > 2 {
                // SAFETY: m > 2 => q.len() >= (col0+3)*q_col >
                // qb0 + 2*q_col + qb + 224, s8.len() >= (col0+3)*s8_col >
                // s8b0 + 2*s8_col + s8b, d8.len() >= (col0+3)*d8_col >
                // d8b0 + d8b + 2*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + 2 * q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + 2 * s8_col + s8b) };
                let e2 = unsafe { *d8.get_unchecked(d8b0 + d8b + 2 * d8_col) };
                f2 += (a as f32 * cda + b as f32 * cdb) * e2;
            }
            if m > 3 {
                // SAFETY: m > 3 => q.len() >= (col0+4)*q_col >
                // qb0 + 3*q_col + qb + 224, s8.len() >= (col0+4)*s8_col >
                // s8b0 + 3*s8_col + s8b, d8.len() >= (col0+4)*d8_col >
                // d8b0 + d8b + 3*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + 3 * q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + 3 * s8_col + s8b) };
                let e3 = unsafe { *d8.get_unchecked(d8b0 + d8b + 3 * d8_col) };
                f3 += (a as f32 * cda + b as f32 * cdb) * e3;
            }
            if m > 4 {
                // SAFETY: m > 4 => q.len() >= (col0+5)*q_col >
                // qb0 + 4*q_col + qb + 224, s8.len() >= (col0+5)*s8_col >
                // s8b0 + 4*s8_col + s8b, d8.len() >= (col0+5)*d8_col >
                // d8b0 + d8b + 4*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + 4 * q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + 4 * s8_col + s8b) };
                let e4 = unsafe { *d8.get_unchecked(d8b0 + d8b + 4 * d8_col) };
                f4 += (a as f32 * cda + b as f32 * cdb) * e4;
            }
            if m > 5 {
                // SAFETY: m > 5 => q.len() >= (col0+6)*q_col >
                // qb0 + 5*q_col + qb + 224, s8.len() >= (col0+6)*s8_col >
                // s8b0 + 5*s8_col + s8b, d8.len() >= (col0+6)*d8_col >
                // d8b0 + d8b + 5*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + 5 * q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + 5 * s8_col + s8b) };
                let e5 = unsafe { *d8.get_unchecked(d8b0 + d8b + 5 * d8_col) };
                f5 += (a as f32 * cda + b as f32 * cdb) * e5;
            }
            if m > 6 {
                // SAFETY: m > 6 => q.len() >= (col0+7)*q_col >
                // qb0 + 6*q_col + qb + 224, s8.len() >= (col0+7)*s8_col >
                // s8b0 + 6*s8_col + s8b, d8.len() >= (col0+7)*d8_col >
                // d8b0 + d8b + 6*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + 6 * q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + 6 * s8_col + s8b) };
                let e6 = unsafe { *d8.get_unchecked(d8b0 + d8b + 6 * d8_col) };
                f6 += (a as f32 * cda + b as f32 * cdb) * e6;
            }
            if m > 7 {
                // SAFETY: m > 7 => q.len() >= (col0+8)*q_col >
                // qb0 + 7*q_col + qb + 224, s8.len() >= (col0+8)*s8_col >
                // s8b0 + 7*s8_col + s8b, d8.len() >= (col0+8)*d8_col >
                // d8b0 + d8b + 7*d8_col.
                let a = q4k_a_chain(&vi, q, qb0 + 7 * q_col + qb);
                let b = unsafe { *s8.get_unchecked(s8b0 + 7 * s8_col + s8b) };
                let e7 = unsafe { *d8.get_unchecked(d8b0 + d8b + 7 * d8_col) };
                f7 += (a as f32 * cda + b as f32 * cdb) * e7;
            }
        }

        it += 1;
    }

    [f0, f1, f2, f3, f4, f5, f6, f7]
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

/// The four sub-block scales one lane of [`q3k_row_dot`] needs, read out of
/// the three funneled scale words directly instead of building all sixteen
/// first. `s0` is the lane's first sub-block index and takes one of four
/// values, which fixes both the byte pair and the nibble half for all four:
/// the low nibbles of sub-blocks `s0` and `s0+2` sit in bytes `s0&1` and
/// `(s0&1)+2` of `a0w`, those of `s0+4` and `s0+6` in the same two bytes of
/// `a1w`, and every 2-bit high part comes from those same two bytes of
/// `a2w`, the second pair two bits up. Two packed halves carry a pair at a
/// time, so the whole extraction is six masked shifts and two ORs.
///
/// Identical to `q3k_sub_scale(&q3k_aux_scales(a0w, a1w, a2w), s0 + 2j)` for
/// j in 0..4 — the same bits by a shorter route, which is what lets the
/// decode keep one owner. `q3k_aux_scales` still builds the full sixteen for
/// the dequant-to-rows kernel, which wants every sub-block.
#[inline(always)]
pub fn q3k_sub_scales4(a0w: u32, a1w: u32, a2w: u32, s0: usize) -> [i32; 4] {
    // 8*(s0&1) picks the byte pair inside the word, 4*(s0>>3) the nibble
    // half. Both come from the lane index alone, so the shift is loop
    // invariant and lifts out of the row walk.
    let sh = (((s0 & 1) << 3) | ((s0 >> 3) << 2)) as u32;
    let p0 = ((a0w >> sh) & 0x000f_000f) | (((a2w >> sh) & 0x0003_0003) << 4);
    let p1 = ((a1w >> sh) & 0x000f_000f) | (((a2w >> (sh + 2)) & 0x0003_0003) << 4);
    // Every scale byte is at most 63, so the 32 offset folded into Q3_K
    // scales subtracts without a sign extension.
    [
        (p0 & 0xff) as i32 - 32,
        (p0 >> 16) as i32 - 32,
        (p1 & 0xff) as i32 - 32,
        (p1 >> 16) as i32 - 32,
    ]
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

/// Everything one Q3_K super-block yields before a column is named: this
/// lane's four dequantized weight quads, the four sub-block scales its
/// integer chain scales by, and the super-block scale. `base` is the
/// super-block's BYTE offset in `w`, `w16`/`s0` the lane constants of
/// [`q3k_row_dot`]. The whole decode lives here so the single-column body
/// and the m-column body cannot drift apart.
///
/// SAFETY: callers keep the super-block at `base` inside a row whose
/// ceil(bytes/4) words are all in `w`.
#[inline(always)]
pub fn q3k_sb_decode(w: &[u32], base: usize, w16: usize, s0: usize) -> ([u32; 4], [i32; 4], f32) {
    // A super-block sits 0 or 2 mod 4 depending on the row's own offset
    // too (odd n_sb shifts every other row), so the funnel select comes
    // from the byte window, not from the super-block index.
    let par = (base >> 1) & 1;

    // qs word: bytes base+32+4*w16 .. +3. The window start is 0 mod 4 or
    // 2 mod 4 with the super-block, so the same floor division addresses
    // both; odd windows reassemble with a 16-bit funnel.
    let qk = (base + 32 + 4 * w16) >> 2;
    // SAFETY: w16 < 16, so qk+1 stays inside the row's ceil(bytes/4)
    // words by this fn's contract (odd super-blocks only shift the window
    // by 2).
    let (lo, hi) = unsafe { (*w.get_unchecked(qk), *w.get_unchecked(qk + 1)) };
    let vl = if par == 0 { lo } else { funnel16(lo, hi) };

    // hmask word (bytes base+4*(w16%8) .. +3), one bit per weight: bit
    // 4*(w16/8)+field. Invert so a clear hmask bit (subtract 4) becomes a
    // set bit, pre-shifted to bit 0 of each byte.
    let hk = (base + 4 * (w16 & 7)) >> 2;
    // SAFETY: same row/base/w16 bounds as the qs window above.
    let (hlo, hhi) = unsafe { (*w.get_unchecked(hk), *w.get_unchecked(hk + 1)) };
    let hm = if par == 0 { hlo } else { funnel16(hlo, hhi) };
    let vh1 = (!hm) >> (4 * (w16 >> 3)) as u32;

    // scales: 12 bytes at base+96..108, decoded with the aux[] shuffle of
    // dequantize_row_q3_K. The 16-byte window base+96..112 (even) /
    // base+94..110 (odd) is covered by four aligned words.
    let ak = (base + 96) >> 2;
    // SAFETY: the 16-byte window ends at most 2 bytes past the row end,
    // but the floored word index stays inside the row's words.
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

    // Super-block scale d (f16 at bytes base+108..109): low half of aw3
    // for an even window (word covers 108..111), high half for odd (word
    // covers 106..109).
    let d_bits = if par == 0 {
        (aw3 & 0xffff) as u16
    } else {
        (aw3 >> 16) as u16
    };
    // The hardware's widening convert, not `half_to_f32`: widening f16 to
    // f32 is exact, so the two agree on every finite and infinite pattern,
    // and a super-block scale read from a GGUF is one of those. The decode
    // is one instruction here against a seven-block branch tree whose
    // subnormal arm is a loop, and it sits on the row walk's critical path.
    let drow = cvt_f32_f16x2_lo(d_bits as u32);

    // Sub-block scales for fields 0..3: sub-block s0+2j.
    let sc = q3k_sub_scales4(a0w, a1w, a2w, s0);

    // Dequantize the four quads to signed bytes (SWAR): q3k_dequant folds
    // the hbit subtract borrow-free.
    (q3k_dequant(vl, vh1), sc, drow)
}

/// One iteration's term of a single-column Q3_K row walk: the super-block
/// `sbp`'s decode, this lane's integer chain against column `col0`'s q8
/// pair slots, and the two shared scales, as the one value the walk adds to
/// its accumulator. Every load the iteration makes is in here, and none of
/// the walk's accumulation is — which is what lets the walk issue several
/// iterations' loads before the first add.
///
/// SAFETY: callers guarantee `sbp < n_sb`, that `base` is that
/// super-block's byte offset inside row `row_abs`, and the buffer lengths of
/// [`q3k_row_dot`] at `m_cols` 1.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn q3k_iter_term(
    w: &[u32],
    q: &[u64],
    d8: &[f32],
    base: usize,
    it: u32,
    qb0: usize,
    d8b0: usize,
    w16: usize,
    half: usize,
    s0: usize,
    d8_base: usize,
    lane: usize,
) -> f32 {
    let sbp = ((it << 1) | half as u32) as usize;
    // SAFETY: `base` is this super-block's byte offset inside the row by this
    // fn's contract, so its window stays in the row.
    let (vi, sc, drow) = q3k_sb_decode(w, base, w16, s0);
    let qb = 64 * it as usize + lane;
    let d8b = 2 * sbp + d8_base;
    // SAFETY: qb0 + qb < (col0+1)*64*iters, the column's u64 slots, by the
    // permutation's group bound and this fn's contract.
    let w01 = unsafe { *q.get_unchecked(qb0 + qb) };
    // SAFETY: qb0 + qb + 32 <= (col0+1)*64*iters - 1 — the pair's second
    // slot is inside the same column.
    let w23 = unsafe { *q.get_unchecked(qb0 + qb + 32) };
    let a = q3k_chain(&vi, w01, w23, &sc);
    // SAFETY: d8b0 + d8b < (col0+1)*2*n_sb by sbp < n_sb and the contract.
    let e0 = unsafe { *d8.get_unchecked(d8b0 + d8b) };
    (a as f32) * (e0 * drow)
}

/// One row's Q3_K dot product against a single activation column —
/// [`q3k_row_dot`]'s column 0, the same loads and the same accumulation
/// order, with the column guards gone at compile time. The guards are
/// runtime tests, so in the m-column walk each iteration's work sits behind
/// its own branch and a lane can have only that iteration's loads in
/// flight; this body is one straight run of the walk instead.
///
/// It keeps one iteration per pass. Q4_K's extra lever — a guard-free
/// prefix of [`Q4K_ITER_UNROLL`] iterations whose loads all precede the
/// adds — measured flat here, so it is not carried.
///
/// Caller contract as [`q3k_row_dot`] with `m_cols` 1.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn q3k_row_dot_1col(
    w: &[u32],
    q: &[u64],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    lane: usize,
) -> f32 {
    let q_col = 64 * iters as usize; // q8 u64 slots per column
    let d8_col = 2 * n_sb; // 128-value blocks per column

    let w16 = lane & 15;
    let half = lane >> 4; // 0: even super-block, 1: odd
    let s0 = 8 * (w16 >> 3) + ((w16 & 7) >> 2); // first sub-block index
    let d8_base = w16 >> 3; // half super-block = one 128-value q8_1 block

    let mut f0 = 0.0f32;

    let qb0 = col0 * q_col;
    let d8b0 = col0 * d8_col;

    // The super-block's byte offset is affine in the iteration — the walk
    // steps two super-blocks, 220 bytes — so it is carried and bumped.
    // Rebuilt from `it` it is a 64-bit multiply and a fresh window
    // derivation every pass: the funnel's `>> 2` of a 110-byte stride is
    // what stops the backend from reducing it.
    let mut base = row_abs * 110 * n_sb + half * 110;

    let mut it: u32 = 0;
    while it < iters {
        let sbp = ((it << 1) | half as u32) as usize;
        // As `q3k_row_dot`: the guard makes a partial final iteration safe
        // and is always true when n_sb is even.
        if sbp < n_sb {
            // SAFETY: the guard is this term's `sbp < n_sb` precondition, and
            // `base` is that super-block's byte offset in row `row_abs`.
            f0 += q3k_iter_term(w, q, d8, base, it, qb0, d8b0, w16, half, s0, d8_base, lane);
        }

        base += 220;
        it += 1;
    }

    f0
}

/// One row's Q3_K dot products with `m_cols` (1..=8) activation columns,
/// pre-reduction: lane `lane` of the row's warp walks `iters`
/// two-super-block iterations keeping per-column partial sums in scalars
/// (weight decode once per iteration, shared by every column — the per-op
/// gemv's own shape), and the CALLER reduces the returned partials across
/// the warp. `q3k_gemv`, `q3k_gemv_sel` and the P0b fused kernels call this
/// one body, so their outputs agree bit for bit by construction.
///
/// Buffer layouts as `q3k_gemv`: `w` holds rows of `110 * n_sb` bytes as u32
/// words (row `row_abs` based at byte `row_abs * 110 * n_sb`; `row_abs` may
/// be an indirect expert row), `q` holds per column `64 * iters` u64 in the
/// quantizer's pair permutation, `d8` holds per column `2 * n_sb` block
/// scales. Column c reads `(col0 + c)` of each.
///
/// SAFETY: callers guarantee `4 * w.len() >= (row_abs + 1) * 110 * n_sb`
/// (every funneled window of the row stays inside its ceil(words)-per-row
/// span), `q.len() >= (col0 + m_cols) * 64 * iters`, `d8.len() >= (col0 +
/// m_cols) * 2 * n_sb`, `iters = n_sb.div_ceil(2)`, and invoke this from all
/// 32 lanes of one warp.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn q3k_row_dot(
    w: &[u32],
    q: &[u64],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    m_cols: usize,
    lane: usize,
) -> [f32; 8] {
    // One column is the decode shape, and it gets a body of its own: the
    // guards below are runtime tests, so in this walk each iteration's work
    // sits behind its own branch. See `q3k_row_dot_1col`.
    if m_cols == 1 {
        return [
            q3k_row_dot_1col(w, q, d8, n_sb, iters, row_abs, col0, lane),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
    }
    let m = m_cols;
    let row_bytes = 110 * n_sb;
    let q_col = 64 * iters as usize; // q8 u64 slots per column
    let d8_col = 2 * n_sb; // 128-value blocks per column

    // Per-lane constants: this lane's qs word within the super-block and
    // the derived scale/d8 bases (see the Q3_K packing comment at
    // `q3k_gemv`; the q8 word base is the permutation position 64*it + lane).
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

    let qb0 = col0 * q_col;
    let d8b0 = col0 * d8_col;

    let mut it: u32 = 0;
    while it < iters {
        let sbp = ((it << 1) | half as u32) as usize;
        // The sbp guard makes a partial final iteration safe; always
        // true when n_sb is even.
        if sbp < n_sb {
            // The super-block's column-independent decode — the lane's
            // four dequantized quads, its sub-block scales and the
            // super-block scale — is `q3k_sb_decode`, shared with
            // `q3k_row_dot_1col`.
            // SAFETY: row_abs is inside w by this fn's contract and sbp <
            // n_sb, so the super-block's window stays in the row.
            let (vi, sc, drow) = q3k_sb_decode(w, row_abs * row_bytes + sbp * 110, w16, s0);

            // q8_1 words in the q3 u64 pairing: field pair (2p, 2p+1)
            // of column c lives in u64 slot q_col*c + 64it + 32p + lane
            // — lo is field 2p, hi is 2p+1. Two u64 loads per iteration
            // instead of four u32 loads: same bytes and wavefronts, half
            // the load-issue count.
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
                // SAFETY: qb0 + qb + 32 <= (col0+1)*q_col - 1 — both u64
                // slots are inside the column's q_col u64s (the
                // permutation's group bound); d8b0 + d8b < (col0+1)*d8_col
                // by the sbp guard.
                let w01 = unsafe { *q.get_unchecked(qb0 + qb) };
                let w23 = unsafe { *q.get_unchecked(qb0 + qb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f0 += (a as f32) * (unsafe { *d8.get_unchecked(d8b0 + d8b) } * drow);
            }
            // Columns 1..7, one launch-uniform guard per column so the
            // work scales with m (MUL-8 amortization curve). Same shape
            // as column 0 with the per-column q/d8 offsets (q stride
            // q_col u64, d8 d8_col).
            // SAFETY: guard m > c means q.len() >= (col0+c+1)*q_col >
            // qb0 + c*q_col + qb + 32 and d8.len() >= (col0+c+1)*d8_col >
            // d8b0 + c*d8_col + d8b, launch-uniform.
            if m > 1 {
                let cb = qb0 + q_col + qb;
                let d8c = d8b0 + d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f1 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
            if m > 2 {
                let cb = qb0 + 2 * q_col + qb;
                let d8c = d8b0 + 2 * d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f2 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
            if m > 3 {
                let cb = qb0 + 3 * q_col + qb;
                let d8c = d8b0 + 3 * d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f3 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
            if m > 4 {
                let cb = qb0 + 4 * q_col + qb;
                let d8c = d8b0 + 4 * d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f4 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
            if m > 5 {
                let cb = qb0 + 5 * q_col + qb;
                let d8c = d8b0 + 5 * d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f5 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
            if m > 6 {
                let cb = qb0 + 6 * q_col + qb;
                let d8c = d8b0 + 6 * d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f6 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
            if m > 7 {
                let cb = qb0 + 7 * q_col + qb;
                let d8c = d8b0 + 7 * d8_col + d8b;
                let w01 = unsafe { *q.get_unchecked(cb) };
                let w23 = unsafe { *q.get_unchecked(cb + 32) };
                let a = q3k_chain(&vi, w01, w23, &sc);
                f7 += (a as f32) * (unsafe { *d8.get_unchecked(d8c) } * drow);
            }
        }

        it += 1;
    }

    [f0, f1, f2, f3, f4, f5, f6, f7]
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
