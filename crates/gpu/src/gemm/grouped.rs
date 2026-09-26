//! The grouped GEMM: one block per (tile, [`GEMM_BM`]-row slab) of a route
//! table, its four entries' device code — the per-type decode, the staging,
//! the step walk and the block — and [`GemmKernels::enqueue_gemm`], which
//! launches them. The entries are declared in `kernels.rs`.
//!
//! Numeric contract. The weights are the file's super-blocks, decoded to the
//! integers ggml's `dequantize_row_*` multiplies: Q4_K and Q5_K codes `q`
//! (0..15, 0..31) with a 6-bit scale `sc` and min `m` per 32 values, the
//! value `d·sc·q − dmin·m`; Q6_K `q − 32` with an int8 scale per 16 values,
//! `d·sc·(q − 32)`; Q3_K `q` in −4..3 with a 6-bit scale per 16 values
//! stored +32, `d·(sc − 32)·q`. The activations are the quantizer's blocks:
//! int8 codes `a`, one scale `d8` per 128 values, and the code sums `s8`
//! per 32. Inside one 128-value block every product is an integer. The
//! tensor cores multiply codes (`m16n8k32` per 32-value sub-block for Q4_K
//! and Q5_K; two `m16n8k16` per 32 values, one per 16-value scale, for Q6_K
//! and Q3_K), each i32 result is multiplied by its sub-block scale into
//! `isum = Σ sc·(q·a)`, and for Q4_K and Q5_K the min term is
//! `imin = Σ m·s8`; every one of these is exact in i32. The block then
//! enters the f32 accumulator in this order, which is the gate:
//! `acc = fma(d8, fma(−dmin, f32(imin), d·f32(isum)), acc)` for Q4_K and
//! Q5_K, `acc = fma(d8, d·f32(isum), acc)` for Q6_K and Q3_K, blocks in
//! increasing k.
//!
//! Geometry. A block is [`GEMM_THREADS`] threads, eight warps over
//! [`GEMM_BM`] rows, sixteen rows per warp, every warp over all of the
//! tile's columns in n-tiles of eight. The tile's activation codes, `s8` and
//! `d8` are staged into shared memory with `cp.async`, double-buffered one
//! 128-value block at a time, and read by every warp. A Q4_K block stages its
//! slab's weight words with them, in the same copy group — per row the
//! super-block's header at its first half and the half's sixteen qs words —
//! and each warp reads its own rows' words from there, the header once per
//! super-block; a Q5_K, Q6_K or Q3_K warp reads its own rows' weight bytes
//! from global memory straight into its `mma` fragments at the head of each
//! step. The fragment k-mapping is the instruction's own (lane
//! `4g + t` holds values `4t..4t+3` and `16+4t..16+4t+3` of each 32), and in
//! the quantizer's q6 permutation those two words of a column are one u64 —
//! the staging copies that permutation verbatim, 16 bytes at a time.

use super::route::gemm_max_tiles;
use super::{GEMM_BN, GemmAct, GemmKernels, GemmRoute};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// Rows one GEMM block covers: eight warps of sixteen.
pub(super) const GEMM_BM: usize = 128;
/// Threads per GEMM block. The entries declare two resident blocks per SM
/// (`launch_bounds(256, 2)`), which caps them at 128 registers a thread. The
/// four warps a scheduler then holds do not hide one warp's dependency chain,
/// so a step lasts as long as its instructions take to issue along it: the
/// Q4_K entry's weight words arrive one step ahead with the activations,
/// while the other entries' loads at a step's head sit on the chain.
pub(super) const GEMM_THREADS: usize = 256;
/// [`GEMM_THREADS`] as the block width a launch takes.
const GEMM_THREADS_U32: u32 = GEMM_THREADS as u32;
/// n-tiles of eight columns in a full tile.
pub(super) const GEMM_NT: usize = GEMM_BN / 8;
/// u32 words one staged column of one 128-value block takes: four runs of
/// the q6 permutation, eight words each, and four words of pad. The stride
/// is 9 mod 8 in 16-byte units, so the eight lanes of a quarter-warp —
/// two columns, four runs — hit eight distinct 16-byte bank groups.
pub(super) const B_COL_W: usize = 36;
/// u32 words one stage of the staged activation codes takes.
pub(super) const B_STAGE_W: usize = GEMM_BN * B_COL_W;
/// i32 words one stage of the staged `s8` takes: four per column.
pub(super) const S8_STAGE: usize = GEMM_BN * 4;
/// u32 words one staged Q4_K row takes: the super-block's four header words,
/// then the half's sixteen qs words. The pitch is 4 mod 8, so a warp's eight
/// row groups start in eight distinct bank quads: its four-lane qs reads hit
/// 32 distinct banks and its 16-byte header reads eight distinct bank groups.
pub(super) const WT_ROW_W: usize = 20;
/// u32 words one stage of the staged Q4_K slab takes: the block's rows.
pub(super) const WT_STAGE_W: usize = GEMM_BM * WT_ROW_W;

const _: () = assert!(GEMM_BM == 16 * (GEMM_THREADS / 32));
const _: () = assert!(GEMM_BN.is_multiple_of(8) && GEMM_BN <= 64);
const _: () = assert!(GEMM_THREADS_U32 as usize == GEMM_THREADS);
// The entries' launch bounds and launch contracts spell the block out.
const _: () = assert!(GEMM_THREADS == 256);
const _: () = assert!((B_COL_W / 4) % 8 == 1 && B_COL_W >= 32 && B_COL_W.is_multiple_of(4));
// A tile's staged codes are at most the same count of chunks for every thread.
const _: () = assert!((GEMM_BN * 8).is_multiple_of(GEMM_THREADS));
// The Q4_K staging's chunk map: thread `tid` copies qs chunk `tid % 4` of
// slab rows `tid / 4` and `tid / 4 + GEMM_BM / 2`, and a thread below
// GEMM_BM the header of row `tid` — every chunk of a step once.
const _: () = assert!(WT_ROW_W % 8 == 4 && WT_ROW_W == 4 + 16 && GEMM_THREADS == 2 * GEMM_BM);

// Bytes of one 256-value super-block of each type, as the file stores it
// (ggml's `block_q3_K` .. `block_q6_K`).
pub(super) const Q3K_SB_BYTES: usize = 110;
const Q4K_SB_BYTES: usize = 144;
const Q5K_SB_BYTES: usize = 176;
pub(super) const Q6K_SB_BYTES: usize = 210;
// A Q4_K or Q5_K super-block in words, the unit its entry addresses the
// stack in (`sb_units`); the Q6_K and Q3_K entries address bytes.
pub(super) const Q4K_SB_WORDS: usize = Q4K_SB_BYTES / 4;
pub(super) const Q5K_SB_WORDS: usize = Q5K_SB_BYTES / 4;
const _: () = assert!(4 * Q4K_SB_WORDS == Q4K_SB_BYTES && 4 * Q5K_SB_WORDS == Q5K_SB_BYTES);
// The entries' launch contracts spell these out as literals: `requires`
// takes integer literals and parameter names only.
const _: () =
    assert!(Q4K_SB_WORDS == 36 && Q5K_SB_WORDS == 44 && Q6K_SB_BYTES == 210 && Q3K_SB_BYTES == 110);

// Byte offsets inside a Q6_K super-block (ggml's `block_q6_K`): the low
// nibbles `ql` (128 bytes), the high bit pairs `qh` (64), the sixteen int8
// scales, then the f16 `d`.
const Q6K_QL: usize = 0;
pub(super) const Q6K_QH: usize = 128;
pub(super) const Q6K_SCALES: usize = 192;
pub(super) const Q6K_D: usize = 208;
const _: () = assert!(
    Q6K_QH == Q6K_QL + 128
        && Q6K_SCALES == Q6K_QH + 64
        && Q6K_D == Q6K_SCALES + 16
        && Q6K_D + 2 == Q6K_SB_BYTES
);

// Byte offsets inside a Q3_K super-block (ggml's `block_q3_K`): the high
// bits `hmask` (32 bytes), the 2-bit codes `qs` (64), the twelve packed
// 6-bit scales, then the f16 `d`.
const Q3K_HMASK: usize = 0;
pub(super) const Q3K_QS: usize = 32;
pub(super) const Q3K_SCALES: usize = 96;
pub(super) const Q3K_D: usize = 108;
const _: () = assert!(
    Q3K_QS == Q3K_HMASK + 32
        && Q3K_SCALES == Q3K_QS + 64
        && Q3K_D == Q3K_SCALES + 12
        && Q3K_D + 2 == Q3K_SB_BYTES
);

/// The K-quant weight types the GEMM decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmWeight {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GemmWeight {
    /// The GEMM's type for a file tensor of type `ty`; any other type is a
    /// `Shape` error that names it.
    pub fn from_ggml(ty: gguf::quant::GgmlType) -> Result<GemmWeight, GpuError> {
        use gguf::quant::GgmlType;
        match ty {
            GgmlType::Q3_K => Ok(GemmWeight::Q3K),
            GgmlType::Q4_K => Ok(GemmWeight::Q4K),
            GgmlType::Q5_K => Ok(GemmWeight::Q5K),
            GgmlType::Q6_K => Ok(GemmWeight::Q6K),
            other => Err(GpuError::shape(
                "GemmWeight::from_ggml",
                format!(
                    "{other:?} is not a type the grouped GEMM decodes (Q3_K, Q4_K, Q5_K, Q6_K)"
                ),
            )),
        }
    }

    /// Bytes of one 256-value super-block.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            GemmWeight::Q3K => Q3K_SB_BYTES,
            GemmWeight::Q4K => Q4K_SB_BYTES,
            GemmWeight::Q5K => Q5K_SB_BYTES,
            GemmWeight::Q6K => Q6K_SB_BYTES,
        }
    }
}

/// Which activation column a slot reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmInput {
    /// Slot `s` reads column `s / top_k`: every slot of a token shares the
    /// token's input (gate and up).
    Shared { top_k: usize },
    /// Slot `s` reads column `s` (down: each slot's own SwiGLU output).
    PerSlot,
}

/// Raw words of one row's super-block `sb`, half `hh` (one 128-value q8_1
/// block), for the Q4_K decode: the super-block's four header words (`d`
/// and `dmin`, then the twelve scale bytes), then for its two 64-value
/// pairs `p = 2·hh, 2·hh + 1` the qs words `8p + t` and `8p + 4 + t` —
/// values `4t..4t+3` and `16+4t..16+4t+3` of both sub-blocks of the pair,
/// low nibbles the even one. `@load` takes the header from `hdr` and the qs
/// words from the row's staged copy at `row` ([`WT_ROW_W`] words: the header,
/// then the half's sixteen qs words in file order).
macro_rules! q4k_dec {
    (@load $hdr:expr, $row:expr, $t:expr) => {{
        let hdr: [u32; 4] = $hdr;
        let row: *mut u32 = $row;
        let t: usize = $t;
        // SAFETY: `row` is a staged row of this step's stage, which the
        // barrier after the stage's wait has published; words `4 + t` to
        // `16 + t` are inside its twenty.
        unsafe {
            [
                hdr[0],
                hdr[1],
                hdr[2],
                hdr[3],
                *row.add(4 + t),
                *row.add(8 + t),
                *row.add(12 + t),
                *row.add(16 + t),
            ]
        }
    }};
    (@frag $r0:ident, $r1:ident, $pp:expr, $hh:expr) => {{
        let m = 0x0f0f_0f0fu32;
        let (i0, i1) = (4 + 2 * $pp, 5 + 2 * $pp);
        let even = [$r0[i0] & m, $r1[i0] & m, $r0[i1] & m, $r1[i1] & m];
        let odd = [
            ($r0[i0] >> 4) & m,
            ($r1[i0] >> 4) & m,
            ($r0[i1] >> 4) & m,
            ($r1[i1] >> 4) & m,
        ];
        let j = 4 * $hh + 2 * $pp;
        let sc = [
            q4k_scale_min(j, $r0[1], $r0[2], $r0[3]).0,
            q4k_scale_min(j, $r1[1], $r1[2], $r1[3]).0,
            q4k_scale_min(j + 1, $r0[1], $r0[2], $r0[3]).0,
            q4k_scale_min(j + 1, $r1[1], $r1[2], $r1[3]).0,
        ];
        (even, odd, sc)
    }};
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $first:expr) => {{
        // SAFETY: every lane of the warp reaches this `mma.sync` (the branch
        // around it is on the warp's rows and the block's tile, both uniform)
        // with its fragments in the instruction's layout.
        let d = unsafe { mma_m16n8k32_s32_s8([0; 4], $a, [$b0, $b1]) };
        isum_add!($isum, 0, $sc[2 * $par] * d[0], $first);
        isum_add!($isum, 1, $sc[2 * $par] * d[1], $first);
        isum_add!($isum, 2, $sc[2 * $par + 1] * d[2], $first);
        isum_add!($isum, 3, $sc[2 * $par + 1] * d[3], $first);
    }};
    (@rows $r0:ident, $r1:ident, $hh:expr) => {
        qk_min_rows!($r0, $r1, $hh)
    };
    (@epi $rw:ident, $s8st:ident, $n0:expr, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {
        qk_min_epi!($rw, $s8st, $n0, $dcol, $isum, $acc, $nt)
    };
}

/// The Q5_K decode: as Q4_K with the fifth bit from `qh` — the header, the
/// qh words `4 + t` and `8 + t` (bit `j` of each byte is sub-block `j`'s
/// fifth bit), then the qs words of the two pairs.
macro_rules! q5k_dec {
    (@load $w:ident, $wb:expr, $hh:expr, $t:expr) => {{
        let wb: usize = $wb;
        let q = wb + 12 + 16 * $hh + $t;
        // SAFETY: `wb` is a super-block's first word and every index is below
        // `wb + Q5K_SB_WORDS`, inside the stack by the launch contract.
        unsafe {
            [
                *$w.get_unchecked(wb),
                *$w.get_unchecked(wb + 1),
                *$w.get_unchecked(wb + 2),
                *$w.get_unchecked(wb + 3),
                *$w.get_unchecked(wb + 4 + $t),
                *$w.get_unchecked(wb + 8 + $t),
                *$w.get_unchecked(q),
                *$w.get_unchecked(q + 4),
                *$w.get_unchecked(q + 8),
                *$w.get_unchecked(q + 12),
            ]
        }
    }};
    (@frag $r0:ident, $r1:ident, $pp:expr, $hh:expr) => {{
        let j = 4 * $hh + 2 * $pp;
        let jb = j as u32;
        let (i0, i1) = (6 + 2 * $pp, 7 + 2 * $pp);
        let even = [
            q5k_code($r0[i0], $r0[4], 0, jb),
            q5k_code($r1[i0], $r1[4], 0, jb),
            q5k_code($r0[i1], $r0[5], 0, jb),
            q5k_code($r1[i1], $r1[5], 0, jb),
        ];
        let odd = [
            q5k_code($r0[i0], $r0[4], 4, jb + 1),
            q5k_code($r1[i0], $r1[4], 4, jb + 1),
            q5k_code($r0[i1], $r0[5], 4, jb + 1),
            q5k_code($r1[i1], $r1[5], 4, jb + 1),
        ];
        let sc = [
            q4k_scale_min(j, $r0[1], $r0[2], $r0[3]).0,
            q4k_scale_min(j, $r1[1], $r1[2], $r1[3]).0,
            q4k_scale_min(j + 1, $r0[1], $r0[2], $r0[3]).0,
            q4k_scale_min(j + 1, $r1[1], $r1[2], $r1[3]).0,
        ];
        (even, odd, sc)
    }};
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $first:expr) => {
        q4k_dec!(@mma $a, $b0, $b1, $sc, $par, $isum, $first)
    };
    (@rows $r0:ident, $r1:ident, $hh:expr) => {
        qk_min_rows!($r0, $r1, $hh)
    };
    (@epi $rw:ident, $s8st:ident, $n0:expr, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {
        qk_min_epi!($rw, $s8st, $n0, $dcol, $isum, $acc, $nt)
    };
}

/// One product into an `isum` word: its first of the n-tile's step (`first`,
/// a constant after unrolling) writes the word, every later one adds — so
/// each n-tile's `isum` starts from the exact integer 0 without a reset.
macro_rules! isum_add {
    ($isum:ident, $i:expr, $v:expr, $first:expr) => {
        if $first {
            $isum[$i] = $v;
        } else {
            $isum[$i] += $v;
        }
    };
}

/// Q4_K/Q5_K per-step row constants for the epilogue: each row's `d`,
/// `−dmin` and the mins `m_j` of the block's four sub-blocks `j = 4·hh + c`.
macro_rules! qk_min_rows {
    ($r0:ident, $r1:ident, $hh:expr) => {{
        let j = 4 * $hh;
        (
            [
                half_bits_to_f32(($r0[0] & 0xffff) as u16),
                half_bits_to_f32(($r1[0] & 0xffff) as u16),
            ],
            [
                -half_bits_to_f32(($r0[0] >> 16) as u16),
                -half_bits_to_f32(($r1[0] >> 16) as u16),
            ],
            [
                q4k_scale_min(j, $r0[1], $r0[2], $r0[3]).1,
                q4k_scale_min(j + 1, $r0[1], $r0[2], $r0[3]).1,
                q4k_scale_min(j + 2, $r0[1], $r0[2], $r0[3]).1,
                q4k_scale_min(j + 3, $r0[1], $r0[2], $r0[3]).1,
            ],
            [
                q4k_scale_min(j, $r1[1], $r1[2], $r1[3]).1,
                q4k_scale_min(j + 1, $r1[1], $r1[2], $r1[3]).1,
                q4k_scale_min(j + 2, $r1[1], $r1[2], $r1[3]).1,
                q4k_scale_min(j + 3, $r1[1], $r1[2], $r1[3]).1,
            ],
        )
    }};
}

/// Q4_K/Q5_K epilogue of n-tile `nt` for one 128-value block, `isum` its
/// four sums: `imin = Σ_c m_c · s8_c` per (row, column), then the
/// contract's order, `acc = fma(d8, fma(−dmin, f32(imin), d·f32(isum)), acc)`.
macro_rules! qk_min_epi {
    ($rw:ident, $s8st:ident, $n0:expr, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {{
        let (dr, ndr, m0, m1) = $rw;
        // SAFETY: columns `n0` and `n0 + 1` are below GEMM_BN, so both
        // 16-byte reads are inside this stage's s8, which the barrier after
        // the stage's wait has published.
        let (sa, sb) = unsafe {
            (
                *($s8st.add(4 * $n0) as *const U32x4),
                *($s8st.add(4 * ($n0 + 1)) as *const U32x4),
            )
        };
        let (sa, sb) = (sa.0, sb.0);
        let imin = [
            m0[0] * sa[0] as i32
                + m0[1] * sa[1] as i32
                + m0[2] * sa[2] as i32
                + m0[3] * sa[3] as i32,
            m0[0] * sb[0] as i32
                + m0[1] * sb[1] as i32
                + m0[2] * sb[2] as i32
                + m0[3] * sb[3] as i32,
            m1[0] * sa[0] as i32
                + m1[1] * sa[1] as i32
                + m1[2] * sa[2] as i32
                + m1[3] * sa[3] as i32,
            m1[0] * sb[0] as i32
                + m1[1] * sb[1] as i32
                + m1[2] * sb[2] as i32
                + m1[3] * sb[3] as i32,
        ];
        let mut i = 0usize;
        while i < 4 {
            ::cuda_device::thread::__unroll_config::<{ 0 }>();
            let r = i / 2;
            let tv = mul_rn_f32(dr[r], $isum[i] as f32);
            let tv = fma_rn_f32(ndr[r], imin[i] as f32, tv);
            $acc[4 * $nt + i] = fma_rn_f32($dcol[i & 1], tv, $acc[4 * $nt + i]);
            i += 1;
        }
    }};
}

/// Q6_K/Q3_K per-scale-pair `mma`: one `m16n8k16` per 16-value half of the
/// 32 values, each scaled by its own sub-block scale. `sc` holds, per
/// chunk parity, (row g low, row g high, row g+8 low, row g+8 high).
macro_rules! k16_mma {
    ($a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $first:expr) => {{
        let a: [u32; 4] = $a;
        // SAFETY: as the k32 form — the whole warp issues both `mma.sync`
        // with fragments in the layout; the low half of a k32 A fragment is
        // the k16 fragment of values 0..15, the high half of 16..31.
        let (dl, dh) = unsafe {
            (
                mma_m16n8k16_s32_s8([0; 4], [a[0], a[1]], $b0),
                mma_m16n8k16_s32_s8([0; 4], [a[2], a[3]], $b1),
            )
        };
        let s = 4 * $par;
        isum_add!($isum, 0, $sc[s] * dl[0] + $sc[s + 1] * dh[0], $first);
        isum_add!($isum, 1, $sc[s] * dl[1] + $sc[s + 1] * dh[1], $first);
        isum_add!($isum, 2, $sc[s + 2] * dl[2] + $sc[s + 3] * dh[2], $first);
        isum_add!($isum, 3, $sc[s + 2] * dl[3] + $sc[s + 3] * dh[3], $first);
    }};
}

/// Q6_K/Q3_K epilogue of n-tile `nt`, `isum` its four sums: `acc = fma(d8,
/// d·f32(isum), acc)`.
macro_rules! k16_epi {
    ($rw:ident, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {{
        let dr = $rw;
        let mut i = 0usize;
        while i < 4 {
            ::cuda_device::thread::__unroll_config::<{ 0 }>();
            let tv = mul_rn_f32(dr[i / 2], $isum[i] as f32);
            $acc[4 * $nt + i] = fma_rn_f32($dcol[i & 1], tv, $acc[4 * $nt + i]);
            i += 1;
        }
    }};
}

/// The Q6_K decode of one row's half `hh` of a super-block at byte `x`: the
/// four ql windows (`64·hh + 4t` plus 0, 16, 32, 48 — values of chunk
/// parity `c & 1` and half `e` at `32·(c & 1) + 16·e`), the two qh windows
/// (`128 + 32·hh + 4t + 16·e`), the half's eight int8 scales (two words,
/// chunk pair `pp`'s four in word `pp`), and the word holding `d`.
macro_rules! q6k_dec {
    (@load $w:ident, $x:expr, $hh:expr, $t:expr) => {{
        let x: usize = $x;
        let l = 4 * $t;
        let lq = x + 64 * $hh + l;
        let lh = x + Q6K_QH + 32 * $hh + l;
        let ls = x + Q6K_SCALES + 8 * $hh;
        let dx = x + Q6K_D;
        // SAFETY: every window starts at an even byte of the super-block at
        // or below `x + Q6K_SCALES + 12`, and the word holding `d` is the one
        // covering bytes `x + Q6K_D, x + Q6K_D + 1`; with the stack's byte
        // stream padded to whole words these reads stay inside it (launch
        // contract).
        unsafe {
            [
                win($w, lq),
                win($w, lq + 16),
                win($w, lq + 32),
                win($w, lq + 48),
                win($w, lh),
                win($w, lh + 16),
                win($w, ls),
                win($w, ls + 4),
                *$w.get_unchecked(dx / 4) >> (8 * (dx & 2) as u32),
            ]
        }
    }};
    (@frag $r0:ident, $r1:ident, $pp:expr, $hh:expr) => {{
        let nib = 4 * $pp as u32;
        let even = [
            q6k_dequant($r0[0], $r0[4], nib, nib),
            q6k_dequant($r1[0], $r1[4], nib, nib),
            q6k_dequant($r0[1], $r0[5], nib, nib),
            q6k_dequant($r1[1], $r1[5], nib, nib),
        ];
        let odd = [
            q6k_dequant($r0[2], $r0[4], nib, nib + 2),
            q6k_dequant($r1[2], $r1[4], nib, nib + 2),
            q6k_dequant($r0[3], $r0[5], nib, nib + 2),
            q6k_dequant($r1[3], $r1[5], nib, nib + 2),
        ];
        let (s0, s1) = ($r0[6 + $pp], $r1[6 + $pp]);
        let sc = [
            sbyte(s0, 0),
            sbyte(s0, 1),
            sbyte(s1, 0),
            sbyte(s1, 1),
            sbyte(s0, 2),
            sbyte(s0, 3),
            sbyte(s1, 2),
            sbyte(s1, 3),
        ];
        (even, odd, sc)
    }};
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $first:expr) => {
        k16_mma!($a, $b0, $b1, $sc, $par, $isum, $first)
    };
    (@rows $r0:ident, $r1:ident, $hh:expr) => {
        [
            half_bits_to_f32(($r0[8] & 0xffff) as u16),
            half_bits_to_f32(($r1[8] & 0xffff) as u16),
        ]
    };
    (@epi $rw:ident, $s8st:ident, $n0:expr, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {{
        let _ = ($s8st, $n0);
        k16_epi!($rw, $dcol, $isum, $acc, $nt)
    }};
}

/// The Q3_K decode of one row's half `hh` of a super-block at byte `x`: the
/// two hmask windows (`4t + 16·e`, bit `4·hh + j` of each byte is chunk
/// `j`'s high bit), the two qs windows (`32 + 32·hh + 4t + 16·e`, field `j`
/// at bit `2j`), the three scale words, and the word holding `d`.
macro_rules! q3k_dec {
    (@load $w:ident, $x:expr, $hh:expr, $t:expr) => {{
        let x: usize = $x;
        let l = 4 * $t;
        let lq = x + Q3K_QS + 32 * $hh + l;
        let dx = x + Q3K_D;
        // SAFETY: every window starts at an even byte of the super-block at
        // or below `x + Q3K_SCALES + 8`, and the word holding `d` covers bytes
        // `x + Q3K_D, x + Q3K_D + 1`; with the stack's byte stream padded to
        // whole words these reads stay inside it (launch contract).
        unsafe {
            [
                win($w, x + l),
                win($w, x + l + 16),
                win($w, lq),
                win($w, lq + 16),
                win($w, x + Q3K_SCALES),
                win($w, x + Q3K_SCALES + 4),
                win($w, x + Q3K_SCALES + 8),
                *$w.get_unchecked(dx / 4) >> (8 * (dx & 2) as u32),
            ]
        }
    }};
    (@frag $r0:ident, $r1:ident, $pp:expr, $hh:expr) => {{
        let j = 2 * $pp as u32;
        let hb = 4 * $hh as u32 + j;
        let even = [
            q3k_code($r0[2], $r0[0], j, hb),
            q3k_code($r1[2], $r1[0], j, hb),
            q3k_code($r0[3], $r0[1], j, hb),
            q3k_code($r1[3], $r1[1], j, hb),
        ];
        let odd = [
            q3k_code($r0[2], $r0[0], j + 1, hb + 1),
            q3k_code($r1[2], $r1[0], j + 1, hb + 1),
            q3k_code($r0[3], $r0[1], j + 1, hb + 1),
            q3k_code($r1[3], $r1[1], j + 1, hb + 1),
        ];
        let s0 = q3k_scale_word($r0[4], $r0[5], $r0[6], $hh, $pp);
        let s1 = q3k_scale_word($r1[4], $r1[5], $r1[6], $hh, $pp);
        let sc = [
            ubyte(s0, 0) - 32,
            ubyte(s0, 1) - 32,
            ubyte(s1, 0) - 32,
            ubyte(s1, 1) - 32,
            ubyte(s0, 2) - 32,
            ubyte(s0, 3) - 32,
            ubyte(s1, 2) - 32,
            ubyte(s1, 3) - 32,
        ];
        (even, odd, sc)
    }};
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $first:expr) => {
        k16_mma!($a, $b0, $b1, $sc, $par, $isum, $first)
    };
    (@rows $r0:ident, $r1:ident, $hh:expr) => {
        [
            half_bits_to_f32(($r0[7] & 0xffff) as u16),
            half_bits_to_f32(($r1[7] & 0xffff) as u16),
        ]
    };
    (@epi $rw:ident, $s8st:ident, $n0:expr, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {{
        let _ = ($s8st, $n0);
        k16_epi!($rw, $dcol, $isum, $acc, $nt)
    }};
}

/// Stage block `h`'s codes, s8 and d8 into stage `st` of the GEMM block's
/// shared tiles: per column the four runs of the q6 permutation at this
/// block's eight positions (two 16-byte copies each), the four s8 of its four
/// sub-blocks (one 16-byte copy) and its d8 (one 4-byte copy). Only the live
/// columns `n < ncols` are copied; a full tile (`full`, a literal) tests none.
/// The code chunks are at most two a thread, each thread's at `tid` and
/// `tid + GEMM_THREADS`.
macro_rules! gemm_stage {
    (
        $h:expr, $st:expr, $full:literal, $tid:ident, $ncols:ident, $n_sb:ident,
        $col_words:ident, $q6:ident, $s8:ident, $d8:ident, $bt:ident, $s8t:ident, $d8t:ident,
        $acol_sh:ident
    ) => {{
        let h: usize = $h;
        let st: usize = $st;
        let ncols: usize = if $full { GEMM_BN } else { $ncols };
        let sb = h / 2;
        let off = 128 * (sb / 2) + 16 * (sb & 1) + 8 * (h & 1);
        let mut k = 0usize;
        while k < GEMM_BN * 8 / GEMM_THREADS {
            ::cuda_device::thread::__unroll_config::<{ 0 }>();
            // tid < GEMM_THREADS, the launch's block width, so u < GEMM_BN · 8.
            let u = $tid + k * GEMM_THREADS;
            if $full || u < ncols * 8 {
                let n = u / 8;
                let i = (u / 2) & 3;
                let q = u & 1;
                // SAFETY: n < ncols <= GEMM_BN; the column index is an
                // activation column below act_cols, so the source's 16 bytes
                // (inside that column's 128*half_it words, 16-byte aligned:
                // every term is a multiple of four words) are inside q6; the
                // destination is inside stage `st` and 16-byte aligned.
                unsafe {
                    let c = *$acol_sh.add(n) as usize;
                    cp_async_cg_16(
                        $bt.add(st * B_STAGE_W + n * B_COL_W + 8 * i + 4 * q),
                        $q6.as_ptr().add(c * $col_words + off + 32 * i + 4 * q),
                    );
                }
            }
            k += 1;
        }
        if $tid < ncols {
            // SAFETY: tid < ncols <= GEMM_BN. The s8 source is groups
            // 4h..4h+3 of the column (16 bytes, aligned: 8*n_sb words per
            // column, 4h words in), the d8 source its block h.
            unsafe {
                let c = *$acol_sh.add($tid) as usize;
                cp_async_cg_16(
                    $s8t.add(st * S8_STAGE + 4 * $tid).cast::<u32>(),
                    $s8.as_ptr().add(c * 8 * $n_sb + 4 * h).cast::<u32>(),
                );
                cp_async_ca_4(
                    $d8t.add(st * GEMM_BN + $tid).cast::<u32>(),
                    $d8.as_ptr().add(c * 2 * $n_sb + h).cast::<u32>(),
                );
            }
        }
    }};
}

/// Weight words of the direct entries (Q5_K, Q6_K, Q3_K): each warp reads its
/// two rows' raw words from global memory at the head of every step, before
/// the step's barrier. `@decl` is the rows' first units, `@head` the loads and
/// `@after` hands them on; there is nothing to stage.
macro_rules! wst_direct {
    (@decl [] $ws:ident, $hdr:ident; $w:ident, $e:ident, $rows_n:ident, $rt:ident,
     $row0:ident, $g:ident, $tid:ident, $wid:ident, $n_sb:ident, $rows32:ident, $n_sb32:ident,
     $sb_units:expr) => {
        let r0 = $e * $rows_n + $row0 + $g;
        let $ws = (r0 * $n_sb * $sb_units, (r0 + 8) * $n_sb * $sb_units);
        let $hdr = ();
    };
    (@stage [] $ws:ident; $w:ident, $sb:expr, $hh:expr, $st:expr, $tid:ident, $sb_units:expr) => {};
    (@head [] $dec:ident, $w:ident, $ws:ident, $active:ident, $sb:ident, $hh:ident, $t:ident,
     $sb_units:expr) => {
        if $active {
            (
                $dec!(@load $w, $ws.0 + $sb * $sb_units, $hh, $t),
                $dec!(@load $w, $ws.1 + $sb * $sb_units, $hh, $t),
            )
        } else {
            Default::default()
        }
    };
    (@after [] $ws:ident, $hdr:ident; $st:ident, $hh:ident, $t:ident, $raw:ident) => {{
        let () = $hdr;
        $raw
    }};
}

/// Weight words of the Q4_K entry, staged: step `h`'s slab — per row the
/// super-block's header at its first half, and the half's sixteen qs words —
/// is copied into stage `h & 1` of `wt` one step ahead, in the activations'
/// copy group (`@stage`), and after the barrier each active warp reads its two
/// rows from there (`@after`), the header at a super-block's first half,
/// carried in registers into its second. `@tile` declares `wt`'s two stages
/// and is its pointer.
///
/// A thread's chunks are the same at every step (the map at [`WT_ROW_W`]), so
/// `@decl` fixes their offsets into the stack at step 0 — bytes, in 32 bits,
/// so each is one register the step's 64-bit add extends with zero — and into
/// stage 0 once, and a step adds its own offset in the row. A row past the
/// slab's live rows copies the slab's last live row instead: every source
/// stays inside the stack, and those rows belong to warps that stage and wait
/// but read nothing.
macro_rules! wst_q4k {
    (@tile $wt:ident) => {{
        static mut WT: SharedArray<u32, { 2 * WT_STAGE_W }, 16> = SharedArray::UNINIT;
        SharedArray::as_raw_mut_ptr(&raw mut WT)
    }};
    (@decl [$wt:ident] $ws:ident, $hdr:ident; $w:ident, $e:ident, $rows_n:ident, $rt:ident,
     $row0:ident, $g:ident, $tid:ident, $wid:ident, $n_sb:ident, $rows32:ident, $n_sb32:ident,
     $sb_units:expr) => {
        // The host refuses a stack of 2^32 bytes or more, so no offset wraps.
        let $ws = {
            let slab = $rt as u32 * GEMM_BM as u32;
            let last = ($rows32 - slab).min(GEMM_BM as u32) - 1;
            let row_bytes = 4 * $n_sb32 * ($sb_units as u32);
            let wrow0 = $e as u32 * $rows32 + slab;
            let tid = $tid as u32;
            let (ra, c) = (tid / 4, tid % 4);
            let rb = ra + GEMM_BM as u32 / 2;
            let rh = tid.min(GEMM_BM as u32 - 1);
            (
                (wrow0 + ra.min(last)) * row_bytes + 16 + 16 * c,
                (wrow0 + rb.min(last)) * row_bytes + 16 + 16 * c,
                (wrow0 + rh.min(last)) * row_bytes,
                ra as usize * WT_ROW_W + 4 + 4 * c as usize,
                rh as usize * WT_ROW_W,
                (16 * $wid + $g) * WT_ROW_W,
            )
        };
        let mut $hdr = [[0u32; 4]; 2];
    };
    (@stage [$wt:ident] $ws:ident; $w:ident, $sb:expr, $hh:expr, $st:expr, $tid:ident,
     $sb_units:expr) => {{
        let hh: usize = $hh;
        // SAFETY: each source is 16 bytes at a chunk's offset in its row at
        // step 0, moved `sb · sb_units + 16 · hh` words on: inside that row's
        // super-block `sb` (the launch contract holds the rows), 16-byte
        // aligned — every term is a multiple of sixteen bytes over a stack
        // enqueue_gemm checked 16-byte aligned. Each destination is that
        // chunk's place in stage `st`, 16-byte aligned.
        unsafe {
            let src = $w.as_ptr().add($sb * $sb_units + 16 * hh);
            let dst = $wt.add($st * WT_STAGE_W);
            cp_async_cg_16(dst.add($ws.3), src.byte_add($ws.0 as usize));
            cp_async_cg_16(dst.add($ws.3 + GEMM_BM / 2 * WT_ROW_W), src.byte_add($ws.1 as usize));
            if hh == 0 && $tid < GEMM_BM {
                cp_async_cg_16(dst.add($ws.4), src.byte_add($ws.2 as usize));
            }
        }
    }};
    // Nothing is read at a step's head: the step's words arrived with its
    // copy group.
    (@head [$wt:ident] $dec:ident, $w:ident, $ws:ident, $active:ident, $sb:ident, $hh:ident,
     $t:ident, $sb_units:expr) => {{
        let _ = $sb;
    }};
    (@after [$wt:ident] $ws:ident, $hdr:ident; $st:ident, $hh:ident, $t:ident, $raw:ident) => {{
        let () = $raw;
        // SAFETY: rows 16·wid + g and 16·wid + g + 8 of the slab are this
        // active warp's, so both are rows of stage `st`.
        let (p0, p1) = unsafe {
            let p0 = $wt.add($st * WT_STAGE_W).add($ws.5);
            (p0, p0.add(8 * WT_ROW_W))
        };
        if $hh == 0 {
            // SAFETY: a row's first four words are its super-block's header,
            // staged at the super-block's first half, 16-byte aligned, and
            // published by the barrier after the stage's wait.
            unsafe {
                $hdr = [(*(p0 as *const U32x4)).0, (*(p1 as *const U32x4)).0];
            }
        }
        (q4k_dec!(@load $hdr[0], p0, $t), q4k_dec!(@load $hdr[1], p1, $t))
    }};
}

/// The walks an entry takes: `split` runs a full tile and a partial one
/// through their own [`gemm_walk`] (the choice is block-uniform), `plain`
/// through [`gemm_walk_plain`].
macro_rules! gemm_walks {
    (split, $nt_live:ident, $n_sb:ident, $args:tt) => {
        if $nt_live == GEMM_NT {
            gemm_walk!(true, $n_sb, $args);
        } else {
            gemm_walk!(false, $n_sb, $args);
        }
    };
    (plain, $nt_live:ident, $n_sb:ident, $args:tt) => {
        gemm_walk_plain!($n_sb, $args)
    };
}

/// The step walk of [`gemm_block`]: super-block by super-block, each one's
/// two 128-value halves as two expansions of [`gemm_step`], so a half's stage,
/// its weight offsets and its decode's scale arm are constants. `full` (a
/// literal) is whether the tile is full.
macro_rules! gemm_walk {
    ($full:literal, $n_sb:ident, $args:tt) => {{
        let mut sb = 0usize;
        while sb < $n_sb {
            gemm_step!(0usize, $full, sb, $n_sb, $args);
            gemm_step!(1usize, $full, sb, $n_sb, $args);
            sb += 1;
        }
    }};
}

/// The step walk of Q5_K and Q3_K: one expansion of [`gemm_step`], the half
/// `hh` a runtime value and every n-tile tested against the tile's length.
/// Only `gate_gemm` launches them, and the split walk doubles a loop's code.
macro_rules! gemm_walk_plain {
    ($n_sb:ident, $args:tt) => {{
        let mut h = 0usize;
        while h < 2 * $n_sb {
            let sb = h / 2;
            gemm_step!(h & 1, false, sb, $n_sb, $args);
            h += 1;
        }
    }};
}

/// One step of [`gemm_block`]: half `hh` of super-block `sb` (128-value block
/// `h = 2·sb + hh`). It loads a direct entry's weight words, stages step
/// `h + 1` into the other stage (this super-block's second half, or the next
/// super-block's first while there is one), waits for step `h`'s copies,
/// decodes both 64-value pairs, then per n-tile multiplies and folds — so
/// an n-tile's four `isum` words are all the sums a warp holds. A full tile
/// (`full`) runs every n-tile with no test against the tile's length.
macro_rules! gemm_step {
    ($hh:expr, $full:literal, $sb:ident, $n_sb:ident, (
        $dec:ident, $wst:ident($($wsa:ident),*), $sb_units:expr, $w:ident, $ws:ident, $hdr:ident,
        $q6:ident, $s8:ident, $d8:ident, $bt:ident, $s8t:ident, $d8t:ident, $acol_sh:ident,
        $tid:ident, $ncols:ident, $col_words:ident,
        $active:ident, $t:ident, $g:ident, $nt_live:ident, $acc:ident
    )) => {{
        let hh: usize = $hh;
        let st = hh;
        let raw = $wst!(@head [$($wsa),*] $dec, $w, $ws, $active, $sb, hh, $t, $sb_units);
        let (nsb, nhh) = ($sb + hh, 1 - hh);
        if hh == 0 || nsb < $n_sb {
            gemm_stage!(2 * nsb + nhh, nhh, $full, $tid, $ncols, $n_sb, $col_words,
                $q6, $s8, $d8, $bt, $s8t, $d8t, $acol_sh);
            $wst!(@stage [$($wsa),*] $ws; $w, nsb, nhh, nhh, $tid, $sb_units);
        }
        // SAFETY: one group per step (possibly empty), so waiting for all
        // but the newest leaves step h's copies complete; the barrier then
        // publishes every thread's copies of it.
        unsafe {
            cp_async_commit_group();
            cp_async_wait_group(1);
        }
        thread::sync_threads();

        if $active {
            let (raw0, raw1) = $wst!(@after [$($wsa),*] $ws, $hdr; st, hh, $t, raw);
            // SAFETY: stage `st` is inside each shared array.
            let (bst, s8st, d8st) = unsafe {
                (
                    $bt.add(st * B_STAGE_W),
                    $s8t.add(st * S8_STAGE),
                    $d8t.add(st * GEMM_BN),
                )
            };
            let (even0, odd0, sc0) = $dec!(@frag raw0, raw1, 0usize, hh);
            let (even1, odd1, sc1) = $dec!(@frag raw0, raw1, 1usize, hh);
            let rw = $dec!(@rows raw0, raw1, hh);
            let mut isum = [0i32; 4];
            let mut nt = 0usize;
            while nt < GEMM_NT {
                ::cuda_device::thread::__unroll_config::<{ 0 }>();
                if $full || nt < $nt_live {
                    // SAFETY: column 8*nt + g < ncols and words 8t .. 8t + 7
                    // < 32 of its block: inside this stage, 16-byte aligned,
                    // published above.
                    let (bw0, bw1) = unsafe {
                        let p = bst.add((8 * nt + $g) * B_COL_W + 8 * $t);
                        ((*(p as *const U32x4)).0, (*(p.add(4) as *const U32x4)).0)
                    };
                    $dec!(@mma even0, bw0[0], bw0[1], sc0, 0, isum, true);
                    $dec!(@mma odd0, bw0[2], bw0[3], sc0, 1, isum, false);
                    $dec!(@mma even1, bw1[0], bw1[1], sc1, 0, isum, false);
                    $dec!(@mma odd1, bw1[2], bw1[3], sc1, 1, isum, false);
                    let n0 = 8 * nt + 2 * $t;
                    // SAFETY: n0 is even and n0 + 1 < GEMM_BN: one 8-byte
                    // read inside this stage's d8.
                    let dcol = unsafe { *(d8st.add(n0) as *const F32x2) }.0;
                    $dec!(@epi rw, s8st, n0, dcol, isum, $acc, nt);
                }
                nt += 1;
            }
        }
        // Every warp is done with stage `st` before step h + 1 stages step
        // h + 2 into it.
        thread::sync_threads();
    }};
}

/// The GEMM block: one tile's columns against one 128-row slab of that
/// tile's expert. A macro, not a function, for the reason
/// `mma_segment_walk` is one: the four entries share the walk and differ
/// only in the decode `dec` names, and a function boundary would move the
/// accumulators out of registers.
///
/// `sb_units` is one super-block in the unit the weight words are addressed
/// in: words for Q4_K and Q5_K ([`Q4K_SB_WORDS`], [`Q5K_SB_WORDS`]), bytes
/// for Q6_K and Q3_K ([`Q6K_SB_BYTES`], [`Q3K_SB_BYTES`]). `wst` names how the
/// weight words reach the warps ([`wst_direct`] or the staged [`wst_q4k`])
/// with the names of its own shared tiles, `walk` the step walk
/// ([`gemm_walks`]). `params` is the entry's parameter list in its order —
/// the four entries share one signature, and each is this expansion alone.
/// The expansion's `unsafe` blocks rest on the entry's launch contract and on
/// the route table the host guarantees was built for this stack and slot
/// count.
///
/// The expansion declares the block's shared tiles — the staged activation
/// codes, `s8` and `d8` in two stages, the tile's slots and their activation
/// columns, and the weight path's own (`@tile`) — then the walk.
///
/// The blocks of tile index 0 — there is one per slab whatever the table
/// holds — first write NaN over their slab's rows of every slot the route
/// refused (`n_tiles[1]` of them, the last entries of `cols`).
macro_rules! gemm_block {
    (
        dec: $dec:ident,
        wst: $wst:ident($($wsa:ident),*),
        walk: $walk:ident,
        sb_units: $sb_units:expr,
        params: (
            $w:ident, $q6:ident, $s8:ident, $d8:ident, $cols:ident, $tiles:ident, $n_tiles:ident,
            $n_experts:ident, $rows:ident, $n_sb:ident, $half_it:ident, $act_cols:ident,
            $n_slots:ident, $max_tiles:ident, $slot_div:ident, $row_tiles:ident, $y:ident $(,)?
        ) $(,)?
    ) => {{
        static mut BT: SharedArray<u32, { 2 * B_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut S8T: SharedArray<i32, { 2 * S8_STAGE }, 16> = SharedArray::UNINIT;
        static mut D8T: SharedArray<f32, { 2 * GEMM_BN }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // SAFETY: each `static mut` is this block's own shared allocation,
        // reached raw; the block walk bounds every index and orders every
        // cross-thread read behind a barrier.
        let (bt, s8t, d8t, slot_sh, acol_sh, $($wsa,)*) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut S8T),
                SharedArray::as_raw_mut_ptr(&raw mut D8T),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
                $($wst!(@tile $wsa),)*
            )
        };
        // Read by the launch contract only.
        let _ = ($n_experts, $act_cols, $max_tiles);

        let b = thread::blockIdx_x() as usize;
        let rt_n = $row_tiles as usize;
        let tile = b / rt_n;
        let rt = b - tile * rt_n;
        if tile == 0 {
            // SAFETY: n_tiles.len() >= 2 by the launch contract.
            let refused = unsafe { *$n_tiles.get_unchecked(1) } as usize;
            // Block-uniform: every thread read the same word.
            if refused > 0 {
                let rows_n = $rows as usize;
                let first = $n_slots as usize - refused;
                let r0 = rt * GEMM_BM;
                let mut u = thread::threadIdx_x() as usize;
                while u < refused * GEMM_BM {
                    let r = r0 + u % GEMM_BM;
                    if r < rows_n {
                        // SAFETY: the route wrote `refused` slots below
                        // n_slots at cols[first..n_slots], inside cols; so
                        // slot · rows + r < n_slots · rows <= y.len(). No
                        // tile lists a refused slot, so no other block
                        // writes these outputs, and each (slot, row) is one
                        // thread's.
                        unsafe {
                            let slot = *$cols.get_unchecked(first + u / GEMM_BM) as usize;
                            *$y.get_unchecked_mut(slot * rows_n + r) = f32::NAN;
                        }
                    }
                    u += GEMM_THREADS;
                }
            }
        }
        // SAFETY: n_tiles.len() >= 2 by the launch contract.
        if tile >= unsafe { *$n_tiles.get_unchecked(0) } as usize {
            return; // block-uniform: the grid's bound is the table's worst case
        }
        // SAFETY: tile < n_tiles <= max_tiles, and tiles.len() >= 2*max_tiles.
        let (e, packed) = unsafe {
            (
                *$tiles.get_unchecked(2 * tile) as usize,
                *$tiles.get_unchecked(2 * tile + 1),
            )
        };
        let (start, len) = tile_parts(packed);
        let (start, len) = (start as usize, len as usize);
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let g = lane / 4;
        let t = lane & 3;

        // The tile's slots and their activation columns; a column past the
        // tile's length repeats column 0, so the last n-tile's padding
        // stages real codes and its results are simply not stored.
        if tid < GEMM_BN {
            let n = if tid < len { tid } else { 0 };
            // SAFETY: start + n < start + len <= n_slots <= cols.len() (the
            // route wrote every tile inside the slot list).
            let slot = unsafe { *$cols.get_unchecked(start + n) };
            // SAFETY: tid < GEMM_BN bounds both shared slots.
            unsafe {
                *slot_sh.add(tid) = slot;
                *acol_sh.add(tid) = slot / $slot_div;
            }
        }
        thread::sync_threads();

        let nt_live = len.div_ceil(8);
        let ncols = 8 * nt_live;
        let rows_n = $rows as usize;
        let row0 = rt * GEMM_BM + 16 * wid;
        // Warp-uniform: a warp past the expert's rows stages and waits with
        // the block but loads, multiplies and stores nothing.
        let active = row0 < rows_n;
        let n_sb = $n_sb as usize;
        let col_words = 128 * $half_it as usize;
        $wst!(@decl [$($wsa),*] ws, hdr; $w, e, rows_n, rt, row0, g, tid, wid, n_sb, $rows, $n_sb,
            $sb_units);

        let mut acc = [0.0f32; 4 * GEMM_NT];
        gemm_stage!(0usize, 0usize, false, tid, ncols, n_sb, col_words,
            $q6, $s8, $d8, bt, s8t, d8t, acol_sh);
        $wst!(@stage [$($wsa),*] ws; $w, 0usize, 0usize, 0usize, tid, $sb_units);
        // SAFETY: commits this thread's copies above as one group.
        unsafe { cp_async_commit_group() };
        gemm_walks!($walk, nt_live, n_sb, ($dec, $wst($($wsa),*), $sb_units, $w, ws, hdr,
            $q6, $s8, $d8, bt, s8t, d8t, acol_sh, tid, ncols, col_words,
            active, t, g, nt_live, acc));

        if active {
            let mut nt = 0usize;
            while nt < GEMM_NT {
                ::cuda_device::thread::__unroll_config::<{ 0 }>();
                if nt < nt_live {
                    let mut i = 0usize;
                    while i < 4 {
                        ::cuda_device::thread::__unroll_config::<{ 0 }>();
                        let n = 8 * nt + 2 * t + (i & 1);
                        if n < len {
                            let r = row0 + g + 8 * (i / 2);
                            // SAFETY: n < GEMM_BN bounds the shared read; the
                            // slot is below n_slots and r below rows, so the
                            // store is inside y (launch contract), and one
                            // lane of one block writes each (slot, row).
                            unsafe {
                                let slot = *slot_sh.add(n) as usize;
                                *$y.get_unchecked_mut(slot * rows_n + r) = acc[4 * nt + i];
                            }
                        }
                        i += 1;
                    }
                }
                nt += 1;
            }
        }
    }};
}

/// The u32 at even byte `byte` of the word stream `w`, assembled from the two
/// aligned words that cover it.
///
/// # Safety
///
/// Words `byte / 4` and `byte / 4 + 1` are inside `w`.
#[inline(always)]
pub(super) unsafe fn win(w: &[u32], byte: usize) -> u32 {
    let a = byte / 4;
    // SAFETY: both words are inside `w` by this fn's contract.
    let (lo, hi) = unsafe { (*w.get_unchecked(a), *w.get_unchecked(a + 1)) };
    (((u64::from(hi) << 32u64) | u64::from(lo)) >> (8 * (byte & 3))) as u32
}

/// Byte `b` of `w`, sign-extended.
#[inline(always)]
pub(super) fn sbyte(w: u32, b: u32) -> i32 {
    i32::from((w >> (8 * b)) as u8 as i8)
}

/// Byte `b` of `w`, zero-extended.
#[inline(always)]
pub(super) fn ubyte(w: u32, b: u32) -> i32 {
    ((w >> (8 * b)) & 0xff) as i32
}

/// Four Q5_K codes: the nibbles at `nib` of the qs word with bit `j` of
/// each qh byte as the fifth bit.
#[inline(always)]
pub(super) fn q5k_code(qs: u32, qh: u32, nib: u32, j: u32) -> u32 {
    ((qs >> nib) & 0x0f0f_0f0f) | (((qh >> j) & 0x0101_0101) << 4)
}

/// Four Q3_K values `q − 4·(1 − hbit)` as signed bytes: field `j` of the qs
/// word, high bit `hb` of each hmask byte; the `| 0x80`/`^ 0x80` bias makes
/// the per-byte subtract borrow-free.
#[inline(always)]
pub(super) fn q3k_code(qs: u32, hm: u32, j: u32, hb: u32) -> u32 {
    ((((qs >> (2 * j)) & 0x0303_0303) | 0x8080_8080).wrapping_sub(((!hm >> hb) & 0x0101_0101) << 2))
        ^ 0x8080_8080
}

/// The four +32 scale bytes of Q3_K sub-blocks `8·hh + 4·pp .. + 3` — the
/// chunk pair `pp` of half `hh` — from the three scale words, ggml's aux
/// shuffle ([`crate::cores::q3k_aux_scales`]) with the word picked by select.
#[inline(always)]
pub(super) fn q3k_scale_word(a0: u32, a1: u32, a2: u32, hh: usize, pp: usize) -> u32 {
    let t = crate::cores::q3k_aux_scales(a0, a1, a2);
    match (hh, pp) {
        (0, 0) => t[2],
        (0, _) => t[3],
        (_, 0) => t[0],
        _ => t[1],
    }
}

/// [`GemmKernels::enqueue_gemm`]'s arguments: `w` is a `ty` stack of
/// `route.n_experts() · rows_per_expert` rows in the card's K-quant format
/// (the file's byte stream as u32 words, zero-padded at its end to a whole
/// number of words per row), K = `act.k()`; `route` is the table its last
/// fill left; `input` picks each slot's activation column; `y` holds
/// `n_slots · rows_per_expert` f32, slot-major.
pub struct GemmArgs<'a> {
    pub ty: GemmWeight,
    pub w: &'a DeviceTensor<u32>,
    pub rows_per_expert: usize,
    pub act: &'a GemmAct,
    pub route: &'a GemmRoute,
    pub input: GemmInput,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// A GEMM launch's checked shape, as the entries take it.
struct GemmLaunch {
    grid: u32,
    n_experts: u32,
    rows: u32,
    n_sb: u32,
    half_it: u32,
    act_cols: u32,
    n_slots: u32,
    max_tiles: u32,
    slot_div: u32,
    row_tiles: u32,
}

impl GemmLaunch {
    /// Err unless `a` is a launchable GEMM ([`GemmArgs`]), in this order: a
    /// filled route table, rows a positive multiple of 16 that make the
    /// table's experts, `ty` rows at the activations' K, a Q4_K stack 16-byte
    /// aligned and under 2^30 words (its staging copies), whole tokens of
    /// `top_k`, and room in `act` and `y` for every slot; then every launch
    /// argument in `u32`.
    fn check(what: &'static str, a: &GemmArgs<'_>) -> Result<GemmLaunch, GpuError> {
        let GemmArgs {
            ty,
            w,
            rows_per_expert,
            act,
            route,
            input,
            y,
        } = a;
        let (ty, rows_per_expert, input) = (*ty, *rows_per_expert, *input);
        let n_slots = route
            .filled
            .ok_or_else(|| GpuError::state(what, "a filled route table (enqueue_route first)"))?;
        let n_sb = act.n_sb();
        if rows_per_expert == 0 || !rows_per_expert.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("rows_per_expert must be a positive multiple of 16, got {rows_per_expert}"),
            ));
        }
        if w.rows() != route.n_experts * rows_per_expert {
            return Err(GpuError::shape(
                what,
                format!(
                    "the stack has {} rows, the route {} experts of {rows_per_expert} rows",
                    w.rows(),
                    route.n_experts
                ),
            ));
        }
        let bpb = ty.block_bytes();
        let words = (w.rows() * bpb * n_sb).div_ceil(4).div_ceil(w.rows());
        if w.cols() != words || 4 * w.buf().len() < w.rows() * bpb * n_sb {
            return Err(GpuError::shape(
                what,
                format!(
                    "{ty:?} rows at K = {} are {words} words, got {} words over {} bytes",
                    act.k(),
                    w.cols(),
                    4 * w.buf().len()
                ),
            ));
        }
        if ty == GemmWeight::Q4K
            && (!w.buf().cu_deviceptr().is_multiple_of(16)
                || w.buf().len() > (u32::MAX / 4) as usize)
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "a Q4_K stack is staged in 16-byte copies from 32-bit byte offsets: it must \
                     start 16-byte aligned and hold fewer than 2^30 words; this one starts at \
                     device address {:#x} and holds {} words",
                    w.buf().cu_deviceptr(),
                    w.buf().len()
                ),
            ));
        }
        let slot_div = match input {
            GemmInput::Shared { top_k } => {
                if top_k == 0 || !n_slots.is_multiple_of(top_k) {
                    return Err(GpuError::shape(
                        what,
                        format!("{n_slots} slots are not whole tokens of top_k {top_k}"),
                    ));
                }
                top_k
            }
            GemmInput::PerSlot => 1,
        };
        if act.cols() < n_slots / slot_div {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots read {} activation columns, act holds {}",
                    n_slots / slot_div,
                    act.cols()
                ),
            ));
        }
        if y.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                what,
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {n_slots}*{rows_per_expert}",
                    y.len()
                ),
            ));
        }
        let max_tiles = gemm_max_tiles(n_slots, route.n_experts);
        let row_tiles = rows_per_expert.div_ceil(GEMM_BM);
        Ok(GemmLaunch {
            grid: launch_u32(what, "grid", row_tiles * max_tiles)?,
            n_experts: launch_u32(what, "n_experts", route.n_experts)?,
            rows: launch_u32(what, "rows_per_expert", rows_per_expert)?,
            n_sb: launch_u32(what, "n_sb", n_sb)?,
            half_it: launch_u32(what, "half_it", n_sb.div_ceil(2))?,
            act_cols: launch_u32(what, "act_cols", act.cols())?,
            n_slots: launch_u32(what, "n_slots", n_slots)?,
            max_tiles: launch_u32(what, "max_tiles", max_tiles)?,
            slot_div: launch_u32(what, "slot_div", slot_div)?,
            row_tiles: launch_u32(what, "row_tiles", row_tiles)?,
        })
    }
}

impl GemmKernels {
    /// Enqueue `y[s][r] = Σ_k w[e(s)][r][k] · x[c(s)][k]` for every slot of
    /// the last fill of the route table ([`GemmArgs`] names each operand).
    /// Asynchronous, allocation-free, capturable. A slot the route refused
    /// (an expert id past the stack) gets NaN in every row, and the fault
    /// word says so.
    pub fn enqueue_gemm(&self, stream: &CudaStream, a: GemmArgs<'_>) -> Result<(), GpuError> {
        let g = GemmLaunch::check("GemmKernels::enqueue_gemm", &a)?;
        let GemmArgs {
            ty,
            w,
            act,
            route,
            y,
            ..
        } = a;
        let cfg = LaunchConfig1D::new(g.grid, GEMM_THREADS_U32, 0);
        macro_rules! launch {
            ($prepare:ident, $entry:ident) => {{
                let prep = self.module.$prepare(cfg)?;
                self.module.$entry(
                    stream,
                    &prep,
                    w.buf(),
                    &act.q6,
                    &act.s8,
                    &act.d8,
                    &route.cols,
                    &route.tiles,
                    &route.n_tiles,
                    g.n_experts,
                    g.rows,
                    g.n_sb,
                    g.half_it,
                    g.act_cols,
                    g.n_slots,
                    g.max_tiles,
                    g.slot_div,
                    g.row_tiles,
                    y,
                )?;
            }};
        }
        match ty {
            GemmWeight::Q3K => launch!(prepare_gemm_q3k, gemm_q3k),
            GemmWeight::Q4K => launch!(prepare_gemm_q4k, gemm_q4k),
            GemmWeight::Q5K => launch!(prepare_gemm_q5k, gemm_q5k),
            GemmWeight::Q6K => launch!(prepare_gemm_q6k, gemm_q6k),
        }
        Ok(())
    }
}
