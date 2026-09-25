//! Grouped int8 tensor-core GEMM: K-quant weights times q8_1 activations for
//! a batch of token columns, each column routed to one expert of a stack —
//! the prefill shape of a MoE layer, and with a one-expert stack the dense
//! projection. For slots `s < n_slots`,
//! `y[s][n] = Σ_k W[e(s)][n][k] · x[c(s)][k]`, where `e(s)` is the slot's
//! routed expert and `c(s)` its activation column: `s / top_k` when every
//! slot of a token reads the token's input ([`GemmInput::Shared`], gate and
//! up), `s` when each slot has its own ([`GemmInput::PerSlot`], down).
//!
//! Between a gate·up pair and its down, [`GemmKernels::enqueue_swiglu_quant`]
//! writes the down's activations straight from the two GEMMs' rows:
//! `silu(g)·u` per slot column, quantized in the same launch
//! (`gemm_swiglu_quant`), the bytes SwiGLU followed by the quantizer write.
//!
//! Three launches and no host synchronisation between them. The q8_1
//! quantizer every gemv reads ([`Gpu::enqueue_quantize_gemm`] launches the
//! same `q3k_quantize_q8_1` into a [`GemmAct`], which holds more columns
//! than a `Q8Act` may) writes the activations. [`GemmKernels::enqueue_route`]
//! turns the router's ids into a table on the card: the slots grouped by
//! expert, ascending slot order within an expert, cut into tiles of at most
//! [`GEMM_BN`] columns, and the tile count. [`GemmKernels::enqueue_gemm`]
//! launches one block per (tile, [`GEMM_BM`]-row slab) over a grid sized for
//! the most tiles `n_slots` slots can make; a block past the table's count
//! returns at once, and an expert no slot picked has no tile, so its weights
//! are never read.
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
//! tile's columns in n-tiles of eight. A warp reads its own rows' weight
//! bytes from global memory straight into its `mma` fragments; the tile's
//! activation codes, `s8` and `d8` are staged into shared memory with
//! `cp.async`, double-buffered one 128-value block at a time, and read by
//! every warp. The fragment k-mapping is the instruction's own (lane
//! `4g + t` holds values `4t..4t+3` and `16+4t..16+4t+3` of each 32), and in
//! the quantizer's q6 permutation those two words of a column are one u64 —
//! the staging copies that permutation verbatim, 16 bytes at a time.

use crate::fault::{FaultSink, FaultSite};
use crate::tensor::{DeviceTensor, Q8ACT_MAX_K};
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Slots one route and one GEMM take: a 512-token ubatch at top-8.
pub const GEMM_MAX_SLOTS: usize = 4096;
/// Experts one stack may have — the route kernel's counters live in shared
/// memory, one per expert.
pub const GEMM_MAX_EXPERTS: usize = 1024;
/// Rows one GEMM block covers: eight warps of sixteen.
pub const GEMM_BM: usize = 128;
/// Columns (slots) one tile holds: eight n-tiles of eight.
pub const GEMM_BN: usize = 64;
/// Threads per GEMM block. The entries declare two resident blocks per SM
/// (`launch_bounds(256, 2)`), which caps them at 128 registers a thread: the
/// sixteen warps an SM then holds are what hides a step's weight loads.
pub const GEMM_THREADS: usize = 256;
/// [`GEMM_THREADS`] as the block width a launch takes.
const GEMM_THREADS_U32: u32 = GEMM_THREADS as u32;
/// n-tiles of eight columns in a full tile.
const GEMM_NT: usize = GEMM_BN / 8;
/// Threads of the route kernel.
const ROUTE_THREADS: usize = 1024;
/// [`ROUTE_THREADS`] as the block width a launch takes.
const ROUTE_THREADS_U32: u32 = ROUTE_THREADS as u32;
/// A staged id the route refused (past the stack) or a lane past the slots.
const NO_EXPERT: u32 = u32::MAX;
/// u32 words one staged column of one 128-value block takes: four runs of
/// the q6 permutation, eight words each, and four words of pad. The stride
/// is 9 mod 8 in 16-byte units, so the eight lanes of a quarter-warp —
/// two columns, four runs — hit eight distinct 16-byte bank groups.
const B_COL_W: usize = 36;
/// u32 words one stage of the staged activation codes takes.
const B_STAGE_W: usize = GEMM_BN * B_COL_W;
/// i32 words one stage of the staged `s8` takes: four per column.
const S8_STAGE: usize = GEMM_BN * 4;

const _: () = assert!(GEMM_BM == 16 * (GEMM_THREADS / 32));
const _: () = assert!(GEMM_BN.is_multiple_of(8) && GEMM_BN <= 64);
const _: () = assert!(GEMM_THREADS_U32 as usize == GEMM_THREADS);
const _: () = assert!(ROUTE_THREADS_U32 as usize == ROUTE_THREADS);
const _: () = assert!(GEMM_MAX_EXPERTS <= ROUTE_THREADS);
const _: () = assert!((B_COL_W / 4) % 8 == 1 && B_COL_W >= 32 && B_COL_W.is_multiple_of(4));
// The route packs a count and a tile count into one scanned u32, 16 bits
// each, and a tile's start and length into another.
const _: () = assert!(GEMM_MAX_SLOTS < 1 << 16 && GEMM_MAX_EXPERTS + GEMM_MAX_SLOTS < 1 << 16);

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
            GemmWeight::Q3K => 110,
            GemmWeight::Q4K => 144,
            GemmWeight::Q5K => 176,
            GemmWeight::Q6K => 210,
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

/// The most tiles `n_slots` slots over `n_experts` experts can make: an
/// expert with `c > 0` slots makes `ceil(c / GEMM_BN)` tiles, which is at
/// most `(c - 1) / GEMM_BN` plus one, so the sum is at most
/// `n_slots / GEMM_BN` plus the count of experts that got a slot.
#[must_use]
pub fn gemm_max_tiles(n_slots: usize, n_experts: usize) -> usize {
    n_slots / GEMM_BN + n_experts.min(n_slots)
}

/// Raw words of one row's super-block `sb`, half `hh` (one 128-value q8_1
/// block), for the Q4_K decode: the super-block's four header words (`d`
/// and `dmin`, then the twelve scale bytes), then for its two 64-value
/// pairs `p = 2·hh, 2·hh + 1` the qs words `8p + t` and `8p + 4 + t` —
/// values `4t..4t+3` and `16+4t..16+4t+3` of both sub-blocks of the pair,
/// low nibbles the even one.
macro_rules! q4k_dec {
    (@load $w:ident, $wb:expr, $hh:expr, $t:expr) => {{
        let wb: usize = $wb;
        let q = wb + 4 + 16 * $hh + $t;
        // SAFETY: `wb` is a super-block's first word and every index is below
        // `wb + 36`, inside the stack by the launch contract.
        unsafe {
            [
                *$w.get_unchecked(wb),
                *$w.get_unchecked(wb + 1),
                *$w.get_unchecked(wb + 2),
                *$w.get_unchecked(wb + 3),
                *$w.get_unchecked(q),
                *$w.get_unchecked(q + 4),
                *$w.get_unchecked(q + 8),
                *$w.get_unchecked(q + 12),
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
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $nt:expr) => {{
        // SAFETY: every lane of the warp reaches this `mma.sync` (the branch
        // around it is on the warp's rows and the block's tile, both uniform)
        // with its fragments in the instruction's layout.
        let d = unsafe { mma_m16n8k32_s32_s8([0; 4], $a, [$b0, $b1]) };
        $isum[4 * $nt] += $sc[2 * $par] * d[0];
        $isum[4 * $nt + 1] += $sc[2 * $par] * d[1];
        $isum[4 * $nt + 2] += $sc[2 * $par + 1] * d[2];
        $isum[4 * $nt + 3] += $sc[2 * $par + 1] * d[3];
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
        // `wb + 44`, inside the stack by the launch contract.
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
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $nt:expr) => {
        q4k_dec!(@mma $a, $b0, $b1, $sc, $par, $isum, $nt)
    };
    (@rows $r0:ident, $r1:ident, $hh:expr) => {
        qk_min_rows!($r0, $r1, $hh)
    };
    (@epi $rw:ident, $s8st:ident, $n0:expr, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {
        qk_min_epi!($rw, $s8st, $n0, $dcol, $isum, $acc, $nt)
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

/// Q4_K/Q5_K epilogue of one n-tile for one 128-value block: `imin = Σ_c
/// m_c · s8_c` per (row, column), then the contract's order, `acc =
/// fma(d8, fma(−dmin, f32(imin), d·f32(isum)), acc)`.
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
            let tv = mul_rn_f32(dr[r], $isum[4 * $nt + i] as f32);
            let tv = fma_rn_f32(ndr[r], imin[i] as f32, tv);
            $acc[4 * $nt + i] = fma_rn_f32($dcol[i & 1], tv, $acc[4 * $nt + i]);
            $isum[4 * $nt + i] = 0;
            i += 1;
        }
    }};
}

/// Q6_K/Q3_K per-scale-pair `mma`: one `m16n8k16` per 16-value half of the
/// 32 values, each scaled by its own sub-block scale. `sc` holds, per
/// chunk parity, (row g low, row g high, row g+8 low, row g+8 high).
macro_rules! k16_mma {
    ($a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $nt:expr) => {{
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
        $isum[4 * $nt] += $sc[s] * dl[0] + $sc[s + 1] * dh[0];
        $isum[4 * $nt + 1] += $sc[s] * dl[1] + $sc[s + 1] * dh[1];
        $isum[4 * $nt + 2] += $sc[s + 2] * dl[2] + $sc[s + 3] * dh[2];
        $isum[4 * $nt + 3] += $sc[s + 2] * dl[3] + $sc[s + 3] * dh[3];
    }};
}

/// Q6_K/Q3_K epilogue of one n-tile: `acc = fma(d8, d·f32(isum), acc)`.
macro_rules! k16_epi {
    ($rw:ident, $dcol:ident, $isum:ident, $acc:ident, $nt:expr) => {{
        let dr = $rw;
        let mut i = 0usize;
        while i < 4 {
            ::cuda_device::thread::__unroll_config::<{ 0 }>();
            let tv = mul_rn_f32(dr[i / 2], $isum[4 * $nt + i] as f32);
            $acc[4 * $nt + i] = fma_rn_f32($dcol[i & 1], tv, $acc[4 * $nt + i]);
            $isum[4 * $nt + i] = 0;
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
        let lh = x + 128 + 32 * $hh + l;
        let ls = x + 192 + 8 * $hh;
        let dx = x + 208;
        // SAFETY: every window starts at an even byte of the super-block at
        // or below `x + 204`, and the word holding `d` is the one covering
        // bytes `x + 208, x + 209`; with the stack's byte stream padded to
        // whole words these reads stay inside it (launch contract).
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
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $nt:expr) => {
        k16_mma!($a, $b0, $b1, $sc, $par, $isum, $nt)
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
        let lq = x + 32 + 32 * $hh + l;
        let dx = x + 108;
        // SAFETY: every window starts at an even byte of the super-block at
        // or below `x + 104`, and the word holding `d` covers bytes
        // `x + 108, x + 109`; with the stack's byte stream padded to whole
        // words these reads stay inside it (launch contract).
        unsafe {
            [
                win($w, x + l),
                win($w, x + l + 16),
                win($w, lq),
                win($w, lq + 16),
                win($w, x + 96),
                win($w, x + 100),
                win($w, x + 104),
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
    (@mma $a:expr, $b0:expr, $b1:expr, $sc:ident, $par:expr, $isum:ident, $nt:expr) => {
        k16_mma!($a, $b0, $b1, $sc, $par, $isum, $nt)
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
/// columns `n < ncols` are copied.
macro_rules! gemm_stage {
    (
        $h:expr, $st:expr, $tid:ident, $ncols:ident, $n_sb:ident, $col_words:ident,
        $q6:ident, $s8:ident, $d8:ident, $bt:ident, $s8t:ident, $d8t:ident, $acol_sh:ident
    ) => {{
        let h: usize = $h;
        let st: usize = $st;
        let sb = h / 2;
        let off = 128 * (sb / 2) + 16 * (sb & 1) + 8 * (h & 1);
        let mut u = $tid;
        while u < $ncols * 8 {
            let n = u / 8;
            let i = (u / 2) & 3;
            let q = u & 1;
            // SAFETY: n < ncols <= GEMM_BN; the column index is an activation
            // column below act_cols, so the source's 16 bytes (inside that
            // column's 128*half_it words, 16-byte aligned: every term is a
            // multiple of four words) are inside q6; the destination is inside
            // stage `st` and 16-byte aligned.
            unsafe {
                let c = *$acol_sh.add(n) as usize;
                cp_async_cg_16(
                    $bt.add(st * B_STAGE_W + n * B_COL_W + 8 * i + 4 * q),
                    $q6.as_ptr().add(c * $col_words + off + 32 * i + 4 * q),
                );
            }
            u += GEMM_THREADS;
        }
        if $tid < $ncols {
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

/// The GEMM block: one tile's columns against one 128-row slab of that
/// tile's expert. A macro, not a function, for the reason
/// `mma_segment_walk` is one: the four entries share the walk and differ
/// only in the decode `dec` names, and a function boundary would move the
/// accumulators out of registers.
///
/// `sb_base` is the unit (words for Q4_K and Q5_K, bytes for Q6_K and Q3_K)
/// the decode's `@load` takes, `row_units` those units per row and
/// `sb_units` per super-block. The expansion's `unsafe` blocks rest on the
/// entry's launch contract and on the route table the host guarantees was
/// built for this stack and slot count.
macro_rules! gemm_block {
    (
        dec: $dec:ident,
        sb_units: $sb_units:expr,
        w: $w:ident, q6: $q6:ident, s8: $s8:ident, d8: $d8:ident,
        cols: $cols:ident, tiles: $tiles:ident, n_tiles: $n_tiles:ident,
        rows: $rows:ident, n_sb: $n_sb:ident, half_it: $half_it:ident,
        slot_div: $slot_div:ident, row_tiles: $row_tiles:ident, y: $y:ident,
        scratch: ($bt:ident, $s8t:ident, $d8t:ident, $slot_sh:ident, $acol_sh:ident) $(,)?
    ) => {{
        let b = thread::blockIdx_x() as usize;
        let rt_n = $row_tiles as usize;
        let tile = b / rt_n;
        let rt = b - tile * rt_n;
        // SAFETY: n_tiles.len() >= 1 by the launch contract.
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
        let start = (packed & 0xffff) as usize;
        let len = (packed >> 16) as usize;
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
                *$slot_sh.add(tid) = slot;
                *$acol_sh.add(tid) = slot / $slot_div;
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
        let steps = 2 * n_sb;
        let col_words = 128 * $half_it as usize;
        let r0 = e * rows_n + row0 + g;
        let base0 = r0 * n_sb * $sb_units;
        let base1 = (r0 + 8) * n_sb * $sb_units;

        let mut acc = [0.0f32; 4 * GEMM_NT];
        let mut isum = [0i32; 4 * GEMM_NT];
        gemm_stage!(0usize, 0usize, tid, ncols, n_sb, col_words,
            $q6, $s8, $d8, $bt, $s8t, $d8t, $acol_sh);
        // SAFETY: commits this thread's copies above as one group.
        unsafe { cp_async_commit_group() };
        let mut h = 0usize;
        while h < steps {
            let sb = h / 2;
            let hh = h & 1;
            let st = hh;
            let (raw0, raw1) = if active {
                (
                    $dec!(@load $w, base0 + sb * $sb_units, hh, t),
                    $dec!(@load $w, base1 + sb * $sb_units, hh, t),
                )
            } else {
                Default::default()
            };
            if h + 1 < steps {
                gemm_stage!(h + 1, st ^ 1, tid, ncols, n_sb, col_words,
                    $q6, $s8, $d8, $bt, $s8t, $d8t, $acol_sh);
            }
            // SAFETY: one group per step (possibly empty), so waiting for all
            // but the newest leaves block h's copies complete; the barrier
            // then publishes every thread's copies of it.
            unsafe {
                cp_async_commit_group();
                cp_async_wait_group(1);
            }
            thread::sync_threads();

            if active {
                // SAFETY: stage `st` is inside each shared array.
                let (bst, s8st, d8st) = unsafe {
                    (
                        $bt.add(st * B_STAGE_W),
                        $s8t.add(st * S8_STAGE),
                        $d8t.add(st * GEMM_BN),
                    )
                };
                let mut pp = 0usize;
                while pp < 2 {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    let (even, odd, sc) = $dec!(@frag raw0, raw1, pp, hh);
                    let mut nt = 0usize;
                    while nt < GEMM_NT {
                        ::cuda_device::thread::__unroll_config::<{ 0 }>();
                        if nt < nt_live {
                            // SAFETY: column 8*nt + g < ncols and word
                            // 8t + 4pp + 3 < 32 of its block: inside this
                            // stage, 16-byte aligned, published above.
                            let bw = unsafe {
                                *(bst.add((8 * nt + g) * B_COL_W + 8 * t + 4 * pp)
                                    as *const U32x4)
                            }
                            .0;
                            $dec!(@mma even, bw[0], bw[1], sc, 0, isum, nt);
                            $dec!(@mma odd, bw[2], bw[3], sc, 1, isum, nt);
                        }
                        nt += 1;
                    }
                    pp += 1;
                }
                let rw = $dec!(@rows raw0, raw1, hh);
                let mut nt = 0usize;
                while nt < GEMM_NT {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    if nt < nt_live {
                        let n0 = 8 * nt + 2 * t;
                        // SAFETY: n0 is even and n0 + 1 < GEMM_BN: one 8-byte
                        // read inside this stage's d8.
                        let dcol = unsafe { *(d8st.add(n0) as *const F32x2) }.0;
                        $dec!(@epi rw, s8st, n0, dcol, isum, acc, nt);
                    }
                    nt += 1;
                }
            }
            // Every warp is done with stage `st` before step h + 1 stages
            // block h + 2 into it.
            thread::sync_threads();
            h += 1;
        }

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
                                let slot = *$slot_sh.add(n) as usize;
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
unsafe fn win(w: &[u32], byte: usize) -> u32 {
    let a = byte / 4;
    // SAFETY: both words are inside `w` by this fn's contract.
    let (lo, hi) = unsafe { (*w.get_unchecked(a), *w.get_unchecked(a + 1)) };
    (((u64::from(hi) << 32u64) | u64::from(lo)) >> (8 * (byte & 3))) as u32
}

/// Byte `b` of `w`, sign-extended.
#[inline(always)]
fn sbyte(w: u32, b: u32) -> i32 {
    i32::from((w >> (8 * b)) as u8 as i8)
}

/// Byte `b` of `w`, zero-extended.
#[inline(always)]
fn ubyte(w: u32, b: u32) -> i32 {
    ((w >> (8 * b)) & 0xff) as i32
}

/// Four Q5_K codes: the nibbles at `nib` of the qs word with bit `j` of
/// each qh byte as the fifth bit.
#[inline(always)]
fn q5k_code(qs: u32, qh: u32, nib: u32, j: u32) -> u32 {
    ((qs >> nib) & 0x0f0f_0f0f) | (((qh >> j) & 0x0101_0101) << 4)
}

/// Four Q3_K values `q − 4·(1 − hbit)` as signed bytes: field `j` of the qs
/// word, high bit `hb` of each hmask byte; the `| 0x80`/`^ 0x80` bias makes
/// the per-byte subtract borrow-free.
#[inline(always)]
fn q3k_code(qs: u32, hm: u32, j: u32, hb: u32) -> u32 {
    ((((qs >> (2 * j)) & 0x0303_0303) | 0x8080_8080).wrapping_sub(((!hm >> hb) & 0x0101_0101) << 2))
        ^ 0x8080_8080
}

/// The four +32 scale bytes of Q3_K sub-blocks `8·hh + 4·pp .. + 3` — the
/// chunk pair `pp` of half `hh` — from the three scale words, ggml's aux
/// shuffle ([`crate::cores::q3k_aux_scales`]) with the word picked by select.
#[inline(always)]
fn q3k_scale_word(a0: u32, a1: u32, a2: u32, hh: usize, pp: usize) -> u32 {
    let t = crate::cores::q3k_aux_scales(a0, a1, a2);
    match (hh, pp) {
        (0, 0) => t[2],
        (0, _) => t[3],
        (_, 0) => t[0],
        _ => t[1],
    }
}

#[cuda_module]
mod gemm_kernels {
    use super::*;
    use crate::cores::{q4k_scale_min, q6k_dequant};
    use crate::elem::silu_mul;
    use crate::fault::quad_finite;
    use crate::flash::half_bits_to_f32;
    use crate::q8_1_quant_vals;
    use cuda_device::async_copy::{
        cp_async_ca_4, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
    };
    use cuda_device::float::{fma_rn_f32, mul_rn_f32};
    use cuda_device::vector::{F32x2, U32x4};
    use cuda_device::wmma::{mma_m16n8k16_s32_s8, mma_m16n8k32_s32_s8};

    /// The route table: `ids` (`n_slots` expert ids, slot `s = token·top_k +
    /// k`) grouped by expert into `cols` — expert 0's slots first, each
    /// expert's in ascending slot order — and cut into tiles of at most
    /// GEMM_BN slots, tile i at `tiles[2i]` (its expert) and `tiles[2i + 1]`
    /// (its first index into `cols`, and its length in the upper 16 bits),
    /// experts ascending; `n_tiles[0]` is the tile count.
    ///
    /// An id at or past `n_experts` has no expert to go to: it raises
    /// [`FaultSite::ExpertId`] on `fault` and its slot is left out of the
    /// table, so no GEMM writes its outputs.
    ///
    /// One block. Every thread stages ids; warp 0 alone counts and fills, in
    /// slot order, 32 slots a pass: the lanes of one pass that share an id
    /// find each other with `match.any`, so the fill needs no atomics and
    /// the table is the same bits on every run.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (
            ids.len() >= n_slots,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 1
        )
    )]
    pub fn gemm_route(
        ids: &[u32],
        n_slots: u32,
        n_experts: u32,
        max_tiles: u32,
        mut cols: DisjointSlice<u32>,
        mut tiles: DisjointSlice<u32>,
        mut n_tiles: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut IDS: SharedArray<u32, GEMM_MAX_SLOTS> = SharedArray::UNINIT;
        static mut CNT: SharedArray<u32, GEMM_MAX_EXPERTS> = SharedArray::UNINIT;
        static mut WT: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let s_n = (n_slots as usize).min(GEMM_MAX_SLOTS);
        let e_n = (n_experts as usize).min(GEMM_MAX_EXPERTS);
        // SAFETY: each `static mut` is this block's own shared allocation,
        // reached raw; every index below is inside its array and every
        // cross-thread read follows a block or warp barrier.
        let (ids_sh, cnt, wt) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut IDS),
                SharedArray::as_raw_mut_ptr(&raw mut CNT),
                SharedArray::as_raw_mut_ptr(&raw mut WT),
            )
        };

        let mut s = tid;
        while s < s_n {
            // SAFETY: s < n_slots <= ids.len().
            let id = unsafe { *ids.get_unchecked(s) };
            let keep = if id < n_experts && (id as usize) < e_n {
                id
            } else {
                fault.raise(FaultSite::ExpertId);
                NO_EXPERT
            };
            // SAFETY: s < GEMM_MAX_SLOTS.
            unsafe { *ids_sh.add(s) = keep };
            s += ROUTE_THREADS;
        }
        if tid < e_n {
            // SAFETY: tid < GEMM_MAX_EXPERTS.
            unsafe { *cnt.add(tid) = 0 };
        }
        thread::sync_threads();

        // Counts, in slot order: per pass, the lowest lane of each group of
        // equal ids adds the group's size.
        if wid == 0 {
            let mut base = 0usize;
            while base < s_n {
                let s = base + lane;
                let id = if s < s_n {
                    // SAFETY: s < s_n <= GEMM_MAX_SLOTS.
                    unsafe { *ids_sh.add(s) }
                } else {
                    NO_EXPERT
                };
                let peers = warp::match_any_sync(u32::MAX, id);
                if id != NO_EXPERT && peers.trailing_zeros() as usize == lane {
                    // SAFETY: id < e_n <= GEMM_MAX_EXPERTS; one lane per id.
                    unsafe { *cnt.add(id as usize) += peers.count_ones() };
                }
                warp::sync_mask(u32::MAX);
                base += 32;
            }
        }
        thread::sync_threads();

        // Exclusive scan over experts of (count | tiles << 16): the low half
        // is the expert's first index into `cols`, the high half its first
        // tile. Neither half's sum reaches 1 << 16 (GEMM_MAX_SLOTS, and
        // gemm_max_tiles's bound), so no carry crosses.
        let c = if tid < e_n {
            // SAFETY: tid < GEMM_MAX_EXPERTS.
            unsafe { *cnt.add(tid) }
        } else {
            0
        };
        let tc = c.div_ceil(GEMM_BN as u32);
        let v = c | (tc << 16);
        let mut incl = v;
        let mut o = 1u32;
        while o < 32 {
            let up = warp::shuffle_up(incl, o);
            if lane as u32 >= o {
                incl += up;
            }
            o <<= 1;
        }
        if lane == 31 {
            // SAFETY: wid < 32.
            unsafe { *wt.add(wid) = incl };
        }
        thread::sync_threads();
        if wid == 0 {
            // SAFETY: lane < 32.
            let x = unsafe { *wt.add(lane) };
            let mut ix = x;
            let mut o = 1u32;
            while o < 32 {
                let up = warp::shuffle_up(ix, o);
                if lane as u32 >= o {
                    ix += up;
                }
                o <<= 1;
            }
            warp::sync_mask(u32::MAX);
            // SAFETY: lane < 32; every lane read its slot before the sync.
            unsafe { *wt.add(lane) = ix - x };
        }
        thread::sync_threads();
        // SAFETY: wid < 32, published by the barrier above.
        let excl = incl - v + unsafe { *wt.add(wid) };
        let off = excl & 0xffff;
        let toff = excl >> 16;
        if tid == ROUTE_THREADS - 1 {
            // SAFETY: n_tiles.len() >= 1; one thread writes it.
            unsafe { *n_tiles.get_unchecked_mut(0) = (excl + v) >> 16 };
        }
        if tid < e_n {
            let mut i = 0u32;
            while i < tc {
                let tix = (toff + i) as usize;
                if tix < max_tiles as usize {
                    let len = (c - i * GEMM_BN as u32).min(GEMM_BN as u32);
                    // SAFETY: tix < max_tiles, so both words are inside
                    // tiles; expert tid owns tiles toff..toff + tc.
                    unsafe {
                        *tiles.get_unchecked_mut(2 * tix) = tid as u32;
                        *tiles.get_unchecked_mut(2 * tix + 1) =
                            (off + i * GEMM_BN as u32) | (len << 16);
                    }
                } else {
                    // Unreachable by gemm_max_tiles's bound; kept loud.
                    fault.raise(FaultSite::ExpertId);
                }
                i += 1;
            }
            // SAFETY: tid < GEMM_MAX_EXPERTS; the count is consumed above.
            unsafe { *cnt.add(tid) = off };
        }
        thread::sync_threads();

        // The stable fill: slot s goes to its expert's next free index; the
        // lanes of one pass that share an id take consecutive indices in
        // lane (= slot) order.
        if wid == 0 {
            let mut base = 0usize;
            while base < s_n {
                let s = base + lane;
                let id = if s < s_n {
                    // SAFETY: s < s_n <= GEMM_MAX_SLOTS.
                    unsafe { *ids_sh.add(s) }
                } else {
                    NO_EXPERT
                };
                let peers = warp::match_any_sync(u32::MAX, id);
                let rank = (peers & warp::lanemask_lt()).count_ones();
                if id != NO_EXPERT {
                    // SAFETY: id < GEMM_MAX_EXPERTS; the index is below the
                    // expert's first index plus its count, so below n_slots
                    // <= cols.len(), and each slot writes its own index.
                    unsafe {
                        let pos = *cnt.add(id as usize) + rank;
                        *cols.get_unchecked_mut(pos as usize) = s as u32;
                    }
                }
                warp::sync_mask(u32::MAX);
                if id != NO_EXPERT && peers.trailing_zeros() as usize == lane {
                    // SAFETY: as above; one lane per id, after every lane of
                    // the pass has read the old index.
                    unsafe { *cnt.add(id as usize) += peers.count_ones() };
                }
                warp::sync_mask(u32::MAX);
                base += 32;
            }
        }
    }

    /// The grouped GEMM for a Q4_K stack: `w` is `n_experts · rows` rows of
    /// `36 · n_sb` words (the file's super-blocks); the activations are the
    /// q6 permutation, `s8` and `d8` of `act_cols` quantized columns; the
    /// route table (`cols`, `tiles`, `n_tiles`) was built for `n_slots`
    /// slots of this stack, and slot `s` reads column `s / slot_div`. Block
    /// `b` is tile `b / row_tiles`, row slab `b % row_tiles`; it writes
    /// `y[s · rows + r]` for its tile's slots and its slab's rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_experts * rows * 36 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 1,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q4k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut BT: SharedArray<u32, { 2 * B_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut S8T: SharedArray<i32, { 2 * S8_STAGE }, 16> = SharedArray::UNINIT;
        static mut D8T: SharedArray<f32, { 2 * GEMM_BN }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // SAFETY: each `static mut` is this block's own shared allocation,
        // reached raw; the block walk bounds every index and orders every
        // cross-thread read behind a barrier.
        let (bt, s8t, d8t, slot_sh, acol_sh) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut S8T),
                SharedArray::as_raw_mut_ptr(&raw mut D8T),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
            )
        };
        let _ = (n_experts, act_cols, n_slots, max_tiles);
        gemm_block!(
            dec: q4k_dec,
            sb_units: 36,
            w: w, q6: q6, s8: s8, d8: d8,
            cols: cols, tiles: tiles, n_tiles: n_tiles,
            rows: rows, n_sb: n_sb, half_it: half_it,
            slot_div: slot_div, row_tiles: row_tiles, y: y,
            scratch: (bt, s8t, d8t, slot_sh, acol_sh),
        );
    }

    /// [`gemm_q4k`] for a Q5_K stack: rows of `44 · n_sb` words.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_experts * rows * 44 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 1,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q5k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut BT: SharedArray<u32, { 2 * B_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut S8T: SharedArray<i32, { 2 * S8_STAGE }, 16> = SharedArray::UNINIT;
        static mut D8T: SharedArray<f32, { 2 * GEMM_BN }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // SAFETY: as in gemm_q4k.
        let (bt, s8t, d8t, slot_sh, acol_sh) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut S8T),
                SharedArray::as_raw_mut_ptr(&raw mut D8T),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
            )
        };
        let _ = (n_experts, act_cols, n_slots, max_tiles);
        gemm_block!(
            dec: q5k_dec,
            sb_units: 44,
            w: w, q6: q6, s8: s8, d8: d8,
            cols: cols, tiles: tiles, n_tiles: n_tiles,
            rows: rows, n_sb: n_sb, half_it: half_it,
            slot_div: slot_div, row_tiles: row_tiles, y: y,
            scratch: (bt, s8t, d8t, slot_sh, acol_sh),
        );
    }

    /// [`gemm_q4k`] for a Q6_K stack: the stack is the file's byte stream,
    /// `210 · n_sb` bytes a row, as u32 words padded at the end to a whole
    /// word; a super-block may start at 2 mod 4.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_experts * rows * 210 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 1,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q6k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut BT: SharedArray<u32, { 2 * B_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut S8T: SharedArray<i32, { 2 * S8_STAGE }, 16> = SharedArray::UNINIT;
        static mut D8T: SharedArray<f32, { 2 * GEMM_BN }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // SAFETY: as in gemm_q4k.
        let (bt, s8t, d8t, slot_sh, acol_sh) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut S8T),
                SharedArray::as_raw_mut_ptr(&raw mut D8T),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
            )
        };
        let _ = (n_experts, act_cols, n_slots, max_tiles);
        gemm_block!(
            dec: q6k_dec,
            sb_units: 210,
            w: w, q6: q6, s8: s8, d8: d8,
            cols: cols, tiles: tiles, n_tiles: n_tiles,
            rows: rows, n_sb: n_sb, half_it: half_it,
            slot_div: slot_div, row_tiles: row_tiles, y: y,
            scratch: (bt, s8t, d8t, slot_sh, acol_sh),
        );
    }

    /// [`gemm_q6k`] for a Q3_K stack: `110 · n_sb` bytes a row.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_experts * rows * 110 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 1,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q3k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut BT: SharedArray<u32, { 2 * B_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut S8T: SharedArray<i32, { 2 * S8_STAGE }, 16> = SharedArray::UNINIT;
        static mut D8T: SharedArray<f32, { 2 * GEMM_BN }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // SAFETY: as in gemm_q4k.
        let (bt, s8t, d8t, slot_sh, acol_sh) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut S8T),
                SharedArray::as_raw_mut_ptr(&raw mut D8T),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
            )
        };
        let _ = (n_experts, act_cols, n_slots, max_tiles);
        gemm_block!(
            dec: q3k_dec,
            sb_units: 110,
            w: w, q6: q6, s8: s8, d8: d8,
            cols: cols, tiles: tiles, n_tiles: n_tiles,
            rows: rows, n_sb: n_sb, half_it: half_it,
            slot_div: slot_div, row_tiles: row_tiles, y: y,
            scratch: (bt, s8t, d8t, slot_sh, acol_sh),
        );
    }

    /// SwiGLU of `n_cols` slot columns of `256 · n_sb` values quantized to
    /// q8_1 in one launch, one 32-thread block per 128-value block: lane `l`
    /// of block `(col, b)` takes values `128·b + 4·l .. +3` of column `col` of
    /// `g` and `u`, forms `elem::silu_mul` of each pair, and hands the four
    /// to `q8_1_quant_vals` — the values `elem::swiglu` stores and the bytes
    /// `q3k_quantize_q8_1` writes from them, since both run the same bodies
    /// on the same geometry. A non-finite value raises
    /// [`FaultSite::QuantColumn`], as the quantizer does, and the block is
    /// still stored.
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
            g.len() >= n_cols * 256 * n_sb,
            u.len() >= n_cols * 256 * n_sb,
            q3.len() >= n_cols * 64 * half_it,
            q4.len() >= n_cols * 256 * quad_it,
            q6.len() >= n_cols * 128 * half_it,
            s8.len() >= n_cols * 8 * n_sb,
            d8.len() >= n_cols * 2 * n_sb
        )
    )]
    pub fn gemm_swiglu_quant(
        g: &[f32],
        u: &[f32],
        n_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let per_col = 2 * n_sb;
        if blk >= n_cols as usize * per_col {
            return; // warp-uniform: one warp per block
        }
        let col = blk / per_col;
        let b = blk - col * per_col;
        let lane = warp::lane_id() as usize;
        let base = col * 256 * n_sb + 128 * b + 4 * lane;
        // SAFETY: base + 3 < (col + 1)·256·n_sb <= n_cols·256·n_sb, inside g
        // and u by the launch contract.
        let (gv, uv) = unsafe {
            (
                [
                    *g.get_unchecked(base),
                    *g.get_unchecked(base + 1),
                    *g.get_unchecked(base + 2),
                    *g.get_unchecked(base + 3),
                ],
                [
                    *u.get_unchecked(base),
                    *u.get_unchecked(base + 1),
                    *u.get_unchecked(base + 2),
                    *u.get_unchecked(base + 3),
                ],
            )
        };
        let v = [
            silu_mul(gv[0], uv[0]),
            silu_mul(gv[1], uv[1]),
            silu_mul(gv[2], uv[2]),
            silu_mul(gv[3], uv[3]),
        ];
        if !quad_finite(v) {
            fault.raise(FaultSite::QuantColumn);
        }
        // SAFETY: col < n_cols and b < 2·n_sb by the lines above, the output
        // bounds are the launch contract's, the block is one warp with one
        // `(col, b)`, and `v` holds values 128·b + 4·lane .. +3 of column col.
        unsafe {
            q8_1_quant_vals(
                v, col, b, n_sb, half_it, quad_it, lane, &mut q3, &mut q4, &mut q6, &mut s8,
                &mut d8,
            );
        }
    }
}

/// q8_1 activations for up to `cols` columns of `k` values each, in the
/// layout the quantizer writes for a `Q8Act` (the q3/q4/q6 permutations, the
/// 32-value code sums and the 128-value scales), for more columns than a
/// `Q8Act` holds. The GEMM reads the q6 permutation, `s8` and `d8`.
pub struct GemmAct {
    q3: DeviceBuffer<u64>,
    q4: DeviceBuffer<u32>,
    q6: DeviceBuffer<u32>,
    s8: DeviceBuffer<i32>,
    d8: DeviceBuffer<f32>,
    cols: usize,
    k: usize,
}

impl GemmAct {
    /// Scratch for `cols` (1..=[`GEMM_MAX_SLOTS`]) columns of `k` values, `k`
    /// a multiple of 256 up to the quantizer's cap. Load-time only.
    pub fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<GemmAct, GpuError> {
        let what = "GemmAct::new";
        if !(1..=GEMM_MAX_SLOTS).contains(&cols) {
            return Err(GpuError::shape(
                what,
                format!("1 <= cols <= {GEMM_MAX_SLOTS}, got {cols}"),
            ));
        }
        if !k.is_multiple_of(256) || !(256..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                what,
                format!("k must be a multiple of 256 in 256..={Q8ACT_MAX_K}, got {k}"),
            ));
        }
        let n_sb = k / 256;
        Ok(GemmAct {
            q3: DeviceBuffer::zeroed(stream, cols * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, cols * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, cols * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, cols * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, cols * 2 * n_sb)?,
            cols,
            k,
        })
    }

    /// Columns this scratch holds.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Values per column.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Super-blocks per column (`k / 256`).
    #[must_use]
    pub fn n_sb(&self) -> usize {
        self.k / 256
    }

    /// The 128-value block scales: `2 * n_sb` f32 per column.
    #[must_use]
    pub fn d8(&self) -> &DeviceBuffer<f32> {
        &self.d8
    }

    /// The 32-value code sums: `8 * n_sb` i32 per column.
    #[must_use]
    pub fn s8(&self) -> &DeviceBuffer<i32> {
        &self.s8
    }

    /// The codes in the q3 pair permutation: `64 * ceil(n_sb/2)` u64 per
    /// column.
    #[must_use]
    pub fn q3(&self) -> &DeviceBuffer<u64> {
        &self.q3
    }

    /// The codes in the q4 permutation: `256 * ceil(n_sb/4)` u32 per column.
    #[must_use]
    pub fn q4(&self) -> &DeviceBuffer<u32> {
        &self.q4
    }

    /// The codes in the q6 permutation the GEMM stages: `128 * ceil(n_sb/2)`
    /// u32 per column.
    #[must_use]
    pub fn q6(&self) -> &DeviceBuffer<u32> {
        &self.q6
    }

    /// Device bytes of the five planes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.q3.num_bytes()
            + self.q4.num_bytes()
            + self.q6.num_bytes()
            + self.s8.num_bytes()
            + self.d8.num_bytes()
    }
}

impl Gpu {
    /// Enqueue the q8_1 quantization of the first `n_cols` columns of `x`
    /// (`act.k()` f32 each) into `act`, raising into `fault` on a
    /// non-finite value — `q3k_quantize_q8_1`, the kernel every gemv's
    /// activations come from, so column `c` holds the bytes a `Q8Act`
    /// quantized from the same values holds. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_quantize_gemm(
        &self,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_quantize_gemm";
        if n_cols == 0 || n_cols > act.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", act.cols),
            ));
        }
        if x.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!("x.len() {} < n_cols*k = {n_cols}*{}", x.len(), act.k),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let m = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q3k_quantize_q8_1(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q3k_quantize_q8_1(
            &self.stream,
            &prep,
            x,
            0,
            m,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            fault,
        )?;
        Ok(())
    }
}

/// One tile of a route table as the host reads it back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmTile {
    /// The tile's expert.
    pub expert: u32,
    /// Its first index into the slot list.
    pub start: u32,
    /// Its slot count, 1..=[`GEMM_BN`].
    pub len: u32,
}

/// The device route table one [`GemmKernels::enqueue_route`] fills and any
/// number of GEMMs over the same slots and stack read: the slot list grouped
/// by expert, the tiles, and the tile count. Remembers the slot count and
/// stack it was last filled for, so a GEMM over another count, another
/// stack or an unfilled table is a named error rather than a launch that
/// computes nothing.
pub struct GemmRoute {
    cols: DeviceBuffer<u32>,
    tiles: DeviceBuffer<u32>,
    n_tiles: DeviceBuffer<u32>,
    /// All-zero ids for the dense table (a one-expert route).
    zeros: Option<DeviceBuffer<u32>>,
    max_slots: usize,
    n_experts: usize,
    /// The slot count of the last enqueued fill.
    filled: Option<usize>,
}

impl GemmRoute {
    /// A table for up to `max_slots` (1..=[`GEMM_MAX_SLOTS`]) slots over a
    /// stack of `n_experts` (1..=[`GEMM_MAX_EXPERTS`]) experts. Load-time
    /// only.
    pub fn new(
        stream: &CudaStream,
        max_slots: usize,
        n_experts: usize,
    ) -> Result<GemmRoute, GpuError> {
        let what = "GemmRoute::new";
        if !(1..=GEMM_MAX_SLOTS).contains(&max_slots) {
            return Err(GpuError::shape(
                what,
                format!("1 <= max_slots <= {GEMM_MAX_SLOTS}, got {max_slots}"),
            ));
        }
        if !(1..=GEMM_MAX_EXPERTS).contains(&n_experts) {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_experts <= {GEMM_MAX_EXPERTS}, got {n_experts}"),
            ));
        }
        let zeros = if n_experts == 1 {
            Some(DeviceBuffer::zeroed(stream, max_slots)?)
        } else {
            None
        };
        Ok(GemmRoute {
            cols: DeviceBuffer::zeroed(stream, max_slots)?,
            tiles: DeviceBuffer::zeroed(stream, 2 * gemm_max_tiles(max_slots, n_experts))?,
            n_tiles: DeviceBuffer::zeroed(stream, 1)?,
            zeros,
            max_slots,
            n_experts,
            filled: None,
        })
    }

    /// The stack's expert count this table routes over.
    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Device bytes of the table.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.cols.num_bytes()
            + self.tiles.num_bytes()
            + self.n_tiles.num_bytes()
            + self.zeros.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }

    /// The table as it stands on the card: the slot list and the tiles. A
    /// blocking read on `stream`, for gates and diagnostics.
    pub fn read_back(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<GemmTile>), GpuError> {
        let n_slots = self
            .filled
            .ok_or_else(|| GpuError::state("GemmRoute::read_back", "the table was never filled"))?;
        let n = self.n_tiles.to_host_vec(stream)?[0] as usize;
        let raw = self.tiles.to_host_vec(stream)?;
        if 2 * n > raw.len() {
            return Err(GpuError::shape(
                "GemmRoute::read_back",
                format!("tile count {n} exceeds the table's {} tiles", raw.len() / 2),
            ));
        }
        let tiles = raw[..2 * n]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| GemmTile {
                expert: p[0],
                start: p[1] & 0xffff,
                len: p[1] >> 16,
            })
            .collect();
        let mut cols = self.cols.to_host_vec(stream)?;
        cols.truncate(n_slots);
        Ok((cols, tiles))
    }
}

/// The loaded GEMM module and its enqueue API. Owns no context and no
/// stream — every enqueue takes the engine stream, so launches order with
/// the rest of the step and are capturable.
pub struct GemmKernels {
    module: gemm_kernels::LoadedModule,
}

impl GemmKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<GemmKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { gemm_kernels::load(ctx)? };
        Ok(GemmKernels { module })
    }

    /// Enqueue the route table for `n_slots` slots whose expert ids are
    /// `ids[0..n_slots]` (slot `token * top_k + k`), read on the card: no
    /// host synchronisation, capturable. An id at or past the table's expert
    /// count raises [`FaultSite::ExpertId`] on `fault` and its slot is left
    /// out, so its outputs are never written — the step that reads the fault
    /// word back turns it into `GpuError::Fault`.
    pub fn enqueue_route(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        n_slots: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_route";
        if n_slots == 0 || n_slots > route.max_slots || ids.len() < n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "1 <= n_slots <= the table's {} slots and ids.len() {} >= n_slots, got \
                     {n_slots}",
                    route.max_slots,
                    ids.len()
                ),
            ));
        }
        self.route_launch(stream, ids, n_slots, route, fault)
    }

    /// Enqueue the dense table: `n_cols` slots, all on expert 0 of a
    /// one-expert table, slot `s` reading column `s` — the GEMM is then a
    /// plain `y = W · x` over `n_cols` columns.
    pub fn enqueue_route_dense(
        &self,
        stream: &CudaStream,
        n_cols: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_route_dense";
        if route.n_experts != 1 {
            return Err(GpuError::shape(
                what,
                format!("a dense table has one expert, this one {}", route.n_experts),
            ));
        }
        if n_cols == 0 || n_cols > route.max_slots {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= {}, got {n_cols}", route.max_slots),
            ));
        }
        let zeros = route
            .zeros
            .take()
            .ok_or_else(|| GpuError::state(what, "the one-expert table's zero ids"))?;
        let r = self.route_launch(stream, &zeros, n_cols, route, fault);
        route.zeros = Some(zeros);
        r
    }

    /// Enqueue `act = q8_1(silu(g) · u)` over the first `n_cols` slot
    /// columns (`act.k()` values each, slot-major as the gate and up GEMMs
    /// write them): one launch, the bytes `ElemKernels::enqueue_swiglu` then
    /// [`Gpu::enqueue_quantize_gemm`] leave, a non-finite value raised on
    /// `fault` as [`FaultSite::QuantColumn`]. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_swiglu_quant(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_swiglu_quant";
        if n_cols == 0 || n_cols > act.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", act.cols),
            ));
        }
        if g.len() < n_cols * act.k || u.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!(
                    "g.len() {} and u.len() {} need n_cols*k = {n_cols}*{}",
                    g.len(),
                    u.len(),
                    act.k
                ),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let n_cols = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_gemm_swiglu_quant(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.gemm_swiglu_quant(
            stream,
            &prep,
            g,
            u,
            n_cols,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            fault,
        )?;
        Ok(())
    }

    /// The one launcher of `gemm_route`.
    fn route_launch(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        n_slots: usize,
        route: &mut GemmRoute,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_route";
        let max_tiles = gemm_max_tiles(n_slots, route.n_experts);
        let n_slots_u = launch_u32(what, "n_slots", n_slots)?;
        let n_experts = launch_u32(what, "n_experts", route.n_experts)?;
        let max_tiles = launch_u32(what, "max_tiles", max_tiles)?;
        let prep = self
            .module
            .prepare_gemm_route(LaunchConfig1D::new(1, ROUTE_THREADS_U32, 0))?;
        self.module.gemm_route(
            stream,
            &prep,
            ids,
            n_slots_u,
            n_experts,
            max_tiles,
            &mut route.cols,
            &mut route.tiles,
            &mut route.n_tiles,
            fault,
        )?;
        route.filled = Some(n_slots);
        Ok(())
    }

    /// Enqueue `y[s][r] = Σ_k w[e(s)][r][k] · x[c(s)][k]` for every slot of
    /// the last fill of `route`: `w` is a `ty` stack of
    /// `route.n_experts() · rows_per_expert` rows in the card's K-quant
    /// format (the file's byte stream as u32 words, zero-padded at its end
    /// to a whole number of words per row), K = `act.k()`; `input` picks each
    /// slot's activation column; `y` holds `n_slots · rows_per_expert` f32,
    /// slot-major. Asynchronous, allocation-free, capturable — but every
    /// expert id the table was built from must be valid, or the slot's rows
    /// are left as they were and the fault word says so.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_gemm(
        &self,
        stream: &CudaStream,
        ty: GemmWeight,
        w: &DeviceTensor<u32>,
        rows_per_expert: usize,
        act: &GemmAct,
        route: &GemmRoute,
        input: GemmInput,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_gemm";
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
        let grid = launch_u32(what, "grid", row_tiles * max_tiles)?;
        let n_experts = launch_u32(what, "n_experts", route.n_experts)?;
        let rows = launch_u32(what, "rows_per_expert", rows_per_expert)?;
        let n_sb_u = launch_u32(what, "n_sb", n_sb)?;
        let half_it = launch_u32(what, "half_it", n_sb.div_ceil(2))?;
        let act_cols = launch_u32(what, "act_cols", act.cols())?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let max_tiles = launch_u32(what, "max_tiles", max_tiles)?;
        let slot_div = launch_u32(what, "slot_div", slot_div)?;
        let row_tiles = launch_u32(what, "row_tiles", row_tiles)?;
        let cfg = LaunchConfig1D::new(grid, GEMM_THREADS_U32, 0);
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
                    n_experts,
                    rows,
                    n_sb_u,
                    half_it,
                    act_cols,
                    n_slots,
                    max_tiles,
                    slot_div,
                    row_tiles,
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
