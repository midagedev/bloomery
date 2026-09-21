//! Device-callable kernel cores, kept outside any `#[cuda_module]` so more
//! than one kernel — a per-op wrapper and a fused block kernel — can call the
//! same body (docs/gpu-design.md decision 6). The codegen backend compiles
//! whatever a `#[kernel]` reaches, so these are ordinary functions; they must
//! stay free of host-only constructs (allocation, panicking bounds checks on
//! the hot path, std I/O).

use cuda_device::dotprod::dp4a_s32;

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
/// SAFETY: callers keep `qb + 7*32` inside one column's 512 q8 words.
#[inline(always)]
pub fn q4k_a_chain(vi: &[u32; 8], q: &[u32], qb: usize) -> i32 {
    // SAFETY: qb + 224 <= 511 inside the caller's column span by this
    // fn's contract (max qb within a column is 256 + 31).
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

