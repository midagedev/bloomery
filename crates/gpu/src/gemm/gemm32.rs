//! The 32-value-block GEMM: Q8_0, IQ4_NL, Q5_1 and Q5_0 weights times
//! [`GemmAct32`] activations over a route table — one block per (tile,
//! [`GEMM_BM`]-row slab), as the K-quant family runs, for the weights whose
//! scales come per 32 values. Seven entries, declared in `kernels32.rs`,
//! differ only in how the weight bytes reach shared memory, and an eighth
//! only in its store:
//! - `gemm_q8_0p`: Q8_0 in the q8f32 planes (`qs` u32 words, eight per
//!   block, and `d` the blocks' f16 bits, `q8f32.rs`'s layout) — the dense
//!   projections, through a one-expert table; `gemm_q8_0p_slab` reads the
//!   same planes for a launch of a wave or more whose tiles share its slabs
//!   ([`Gemm32Kernels::q8_0p_slab`]):
//!   a slab row's step in one request, its scales eight at a time, and the
//!   blocks of a large stack in groups of tiles ([`raster32`]);
//! - `gemm_q8_0f`: the file's 34-byte `block_q8_0` stream (f16 `d`, then 32
//!   int8 codes) as u32 words;
//! - `gemm_iq4nl`: the file's 18-byte `block_iq4_nl` stream (f16 `d`, then
//!   16 code bytes whose nibbles index `kvalues_iq4nl`), its codes decoded
//!   to the table's int8 values and its `d` staged as f32 — the same staged
//!   row `gemm_q8_0f` fills;
//! - `gemm_q5_1`: the file's 24-byte `block_q5_1` stream (f16 `d` and `m`,
//!   the high bits `qh`, the nibbles `qs[16]`) as u32 words;
//! - `gemm_q5_0p`: the resident packed Q5_0 the gemv reads (`q5.rs`,
//!   `pack_q5_0`): per row the file's `qs` words in four planes, its `qh`
//!   plane and its f16 `d` two to a word, each block read and decoded to the
//!   codes `q − 16` by the gemv's own reader (`q5::q5_0_block`) — no repack
//!   and no second copy;
//! - `gemm_nvfp4p`: NVFP4 in the resident planes (`mxfp4.rs`'s `repack`:
//!   `qs` u32 words, four a 32-value unit, `e` one word a 64-value block)
//!   with one F32 tensor scale an expert of the stack — the step's code
//!   vectors and scale words copied raw into one of [`NV_STAGES`] weight
//!   stages two steps ahead ([`nv_copy`]), each warp decoding its own rows'
//!   nibbles to the E2M1 table's int8 values and their sub-block scales to
//!   f32 as it reads them ([`nv_frags`]), the store applying the tile's
//!   expert's scale. No Q8_0 twin is derived beside them.
//!   `gemm_nvfp4p_swiglu` is its walk as the up GEMM of a SwiGLU pair, as
//!   `gemm_q8_0p_swiglu_all` and its slab twin are the Q8_0 planes': the
//!   store is the SwiGLU epilogue (`swiglu32.rs`), which takes
//!   `silu(g) · u` of each output and the gate GEMM's row and quantizes a
//!   slab's rows into the down GEMM's activations instead
//!   ([`Gemm32Kernels::enqueue_up_swiglu32`]).
//!
//! Numeric contract, every entry. Per 32-value block `b` of an output
//! `(slot, row)`: `isum` the exact i32 dot of the weight codes (Q8_0 int8;
//! IQ4_NL the signed `kvalues_iq4nl` entry of each nibble, |kv| ≤ 127; Q5_1
//! the unsigned 5-bit code, 0..31; Q5_0 the signed `q − 16`, −16..15) with
//! the column's int8 codes, one `mma.m16n8k32` (mainline MMQ's Q8_0 way);
//! `d_w` (and `m_w`) the block's f16 values, exact in f32; `d_a`, `s_a` the
//! column's block scale and code sum.
//! The block enters the f32 accumulator as
//! `acc = fma(d_a, d_w·f32(isum), acc)` for Q8_0, IQ4_NL and Q5_0 and
//! `acc = fma(d_a, fma(m_w, f32(s_a), d_w·f32(isum)), acc)` for Q5_1 —
//! the value `d_w·d_a·isum + m_w·d_a·s_a` — from `acc = 0`, blocks in
//! increasing k. That order is a function of K alone: never of the grid,
//! the tile a slot lands in, its neighbours or the slot count, so an
//! output's bits depend on its weight row and its column only.
//!
//! NVFP4's contract, per 32-value unit `b` in increasing k instead: `i0`,
//! `i1` the exact i32 dots of the unit's two 16-value sub-blocks with the
//! column's int8 codes (`|i_s| ≤ 16 · 12 · 127`), each one `mma.m16n8k16`
//! over the sub-block's sixteen codes (the A words `a[2s]`, `a[2s + 1]` of
//! the unit's `ldmatrix` fragment, the B word `s` of the column's), `d0`, `d1`
//! the sub-blocks' staged scales (`ue4m3_half`, the halving of the doubled
//! codes) and `d_a` the column's 32-block scale:
//! `acc = fma(d_a, fma(d1, f32(i1), d0·f32(i0)), acc)` from `acc = 0`, the
//! stored value `scale_e·acc`, one `mul_rn` at the store by the scale of the
//! slot's expert `e`. Each `d_s·f32(i_s)`
//! is exact in f32 — a 4-bit significand times a 15-bit integer — so a unit
//! rounds twice and the store once; the same K-only order rule holds.
//! These F32 roundings apply while the intermediates stay in range. An
//! exceptional magnitude stays in F64 until the expert scale is applied,
//! so an intermediate overflow or underflow cannot erase a finite output:
//! the block walks in F32 and walks again exactly only when that walk saw
//! such a value (`grouped.rs` module doc, "Range").
//!
//! Geometry. A block is [`GEMM_THREADS`] threads, eight warps of sixteen
//! rows over the tile's columns in n-tiles of eight (the K-quant family's
//! shape). The K axis is walked in [`GEMM32_STEP`]-value steps, two blocks
//! a step, the last step holding one block when K/32 is odd. Each step's
//! weights, activation codes and activation scales sit in one of two shared
//! stages: the activations and the plane's codes arrive by `cp.async`; the
//! plane's scales and every other layout's bytes are read into registers one
//! step ahead and written into the stage after the step's compute, decoded
//! to the staged form — per row sixteen code words then the two blocks' `d`
//! (and Q5_1's `m`) as f32 ([`W32_ROW`] words). NVFP4 is the exception: its
//! raw weights arrive by `cp.async` two steps ahead into three stages of
//! their own, and each warp decodes them at its fragment reads, so no weight
//! load waits in registers through a step's compute. One barrier a step: it
//! publishes step `h`'s stage and tells that every warp is done with step
//! `h − 1`'s, which the copies and stores for step `h + 1` then refill. Each
//! warp loads its `mma` A fragments with `ldmatrix.x4` from the staged rows
//! and each n-tile's B fragments of both blocks with one more.

use super::act32::GemmAct32;
use super::grouped::{GEMM_BM, GEMM_THREADS};
use super::kernels32::Gemm32Kernels;
use super::route::gemm_max_tiles;
use super::{GEMM_BN, GEMM32_STEP, GemmInput, GemmRoute, SwigluSlots};
use crate::fault::FaultSink;
use crate::iq::iq4_word;
use crate::mxfp4_sel::Nvfp4Stack;
use crate::q5::q5_0_row_words;
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::ptx_asm;

/// u32 words of one staged weight row of a step: the two blocks' sixteen
/// code words, then `d` of blocks 0 and 1 and `m` of blocks 0 and 1 as f32
/// bits. The pitch is 5 in 16-byte units, odd, so the eight rows one
/// `ldmatrix` phase reads land in eight distinct bank quads.
pub(super) const W32_ROW: usize = 20;
/// u32 words of one weight stage: the slab's rows.
pub(super) const W32_STAGE_W: usize = GEMM_BM * W32_ROW;
/// u32 words of one staged activation column of a step: sixteen code words
/// and four of pad, the weight row's pitch for the same reason.
pub(super) const B32_ROW: usize = 20;
/// u32 words of one activation stage.
pub(super) const B32_STAGE_W: usize = GEMM_BN * B32_ROW;
/// u32 words of one staged column's scales and sums: `d` of blocks 0 and 1,
/// then `s` of blocks 0 and 1 (Q5_1 only).
pub(super) const D32_ROW: usize = 4;
/// u32 words of one scale stage.
pub(super) const D32_STAGE_W: usize = GEMM_BN * D32_ROW;
/// u32 words of one NVFP4 weight stage's codes ([`nv_copy`]): a step's two
/// 16-byte code vectors a slab row.
pub(super) const NV_CODE_W: usize = GEMM_BM * 8;
/// u32 words of one NVFP4 weight stage: the codes, then the rows' scale
/// words of the step, one a row.
pub(super) const NV_STAGE_W: usize = NV_CODE_W + GEMM_BM;
/// NVFP4 weight stages: a step's weights are copied two steps ahead.
pub(super) const NV_STAGES: usize = 3;
const _: () = assert!(NV_STAGES * NV_STAGE_W <= 2 * W32_STAGE_W);
// The SwiGLU epilogue stages its rounds in WT (`swiglu32.rs`).
const _: () = assert!(super::swiglu32::SW_STAGE <= 2 * W32_STAGE_W);
/// [`GEMM_THREADS`] as the block width a launch takes.
pub(super) const GEMM32_THREADS_U32: u32 = GEMM_THREADS as u32;
/// Tiles one group of the Q8_0 planes' block raster takes ([`raster32`]):
/// at K 6144 a group's activation columns are 6.3 MB, a quarter of a 24 MiB
/// L2.
pub(super) const GEMM32_GROUP: usize = 16;
/// An expert's Q8_0 plane bytes from which its blocks go in groups of
/// [`GEMM32_GROUP`] tiles; a smaller stack stays in L2 across tiles, and its
/// blocks go tile by tile, the order whose stores land together.
pub(super) const GEMM32_RASTER_BYTES: usize = 8 << 20;

/// Block `b`'s `(tile, slab)` over `tiles` tiles of `row_tiles` slabs each,
/// or `None` past them. The tiles go in groups of `group`, a group's blocks
/// slab by slab with its tiles in order inside each slab, so the blocks
/// resident at once share the weight slabs they read (one fetch from memory
/// for the whole group, however large the stack) while the group's
/// activation columns stay in L2 across its slabs. A group of one is tile
/// by tile, slab `b % row_tiles` of tile `b / row_tiles`.
#[inline(always)]
pub(super) fn raster32(
    b: usize,
    row_tiles: usize,
    tiles: usize,
    group: usize,
) -> Option<(usize, usize)> {
    let first = b / (group * row_tiles) * group;
    if first >= tiles {
        return None;
    }
    let n = (tiles - first).min(group);
    let r = b - first * row_tiles;
    if r >= n * row_tiles {
        return None;
    }
    Some((first + r % n, r / n))
}

// Byte sizes of one 32-value block as the file stores it (ggml's
// `block_q8_0`, `block_iq4_nl` and `block_q5_1`), and a Q5_1 block in words.
pub(super) const Q8_0_BLOCK_BYTES: usize = 34;
pub(super) const Q5_1_BLOCK_WORDS: usize = 6;
pub(super) const IQ4_NL_BLOCK_BYTES: usize = 18;

const _: () = assert!(GEMM32_STEP == 64 && W32_ROW == 16 + 4 && B32_ROW >= 16);
const _: () = assert!((W32_ROW / 4) % 2 == 1 && (B32_ROW / 4) % 2 == 1);
const _: () = assert!(W32_ROW.is_multiple_of(4) && B32_ROW.is_multiple_of(4));
// The relay map: thread `tid` owns slab row `tid / 2`, block `tid % 2` of
// every step; the activation map: column `tid / 4`, 16-byte chunk (and
// scale word) `tid % 4`.
const _: () = assert!(GEMM_THREADS == 2 * GEMM_BM && GEMM_THREADS == 4 * GEMM_BN);
// The entries' launch contracts spell the block sizes out as literals.
const _: () = assert!(
    Q8_0_BLOCK_BYTES == 34
        && Q5_1_BLOCK_WORDS == 6
        && IQ4_NL_BLOCK_BYTES == 18
        && GEMM_THREADS == 256
);
// Every staged tile fits a block's static shared memory with room for two
// blocks an SM (48 KiB static cap, 100 KiB an SM on sm_86).
const _: () =
    assert!(4 * (2 * (W32_STAGE_W + B32_STAGE_W + D32_STAGE_W) + 2 * GEMM_BN) <= 48 * 1024);

/// A Q8_0 file block's `d` bits and its eight code words from the nine
/// words `w` that cover it, the block starting at byte `x` of the stream
/// (`w[0]` the word holding byte `x`; `x` is even, a block being 34 bytes).
#[inline(always)]
pub(super) fn q8_0_file_block(w: [u32; 9], x: usize) -> (u16, [u32; 8]) {
    let o = 8 * (x & 3) as u64;
    let c = 8 * ((x & 3) + 2) as u64;
    let win = |i: usize| (((u64::from(w[i + 1]) << 32) | u64::from(w[i])) >> c) as u32;
    (
        (w[0] as u64 >> o) as u16,
        [
            win(0),
            win(1),
            win(2),
            win(3),
            win(4),
            win(5),
            win(6),
            win(7),
        ],
    )
}

/// An IQ4_NL file block's `d` bits and its eight staged code words from the
/// five words `w` that cover it, the block starting at byte `x` of the
/// stream (`x` is even, a block being 18 bytes): the low nibbles of its 16
/// code bytes are values `0..16` and the high nibbles `16..32`, each decoded
/// through `iq.rs`'s `kvalues_iq4nl` register table ([`iq4_word`]).
#[inline(always)]
pub(super) fn iq4_nl_file_block(w: [u32; 5], x: usize) -> (u16, [u32; 8]) {
    let o = 8 * (x & 3) as u64;
    let c = 8 * ((x & 3) + 2) as u64;
    let win = |i: usize| (((u64::from(w[i + 1]) << 32) | u64::from(w[i])) >> c) as u32;
    let (l0, h0) = iq4_word(win(0));
    let (l1, h1) = iq4_word(win(1));
    let (l2, h2) = iq4_word(win(2));
    let (l3, h3) = iq4_word(win(3));
    ((w[0] as u64 >> o) as u16, [l0, l1, l2, l3, h0, h1, h2, h3])
}

/// One lane's A word of an NVFP4 sub-block: the low (`hi` false) or high
/// nibbles of code word `v` through the E2M1 table (`mxfp4`'s card decode).
#[inline(always)]
pub(super) fn nv_half(v: u32, hi: bool) -> u32 {
    let (l, h) = crate::cores::table16_word(v, crate::mxfp4::E2M1);
    if hi { h } else { l }
}

/// Sub-block scale byte `b` of an NVFP4 scale word as f32 (unit `b / 2`,
/// sub-block `b % 2` of the 64-value block).
#[inline(always)]
pub(super) fn nv_scale(w: u32, b: u32) -> f32 {
    crate::mxfp4::ue4m3_half((w >> (8 * b)) & 0xff)
}

/// `mma.sync.m16n8k16` of s8 × s8 into s32 from a zero accumulator: rows `g`
/// and `g + 8` of the A fragment's two words (k = 4·tig .. 4·tig + 3 of the
/// 16) against column `g` of B's one word, the exact i32 dot of those sixteen
/// values. The pinned cuda-oxide has no wrapper for the s8 k16 shape, so the
/// instruction is written out.
///
/// # Safety
///
/// Every lane of the warp executes the call (`mma.sync.aligned`) with its
/// fragments in the instruction's layout.
#[inline(always)]
pub(super) unsafe fn mma_m16n8k16_s32_s8_zero(a: [u32; 2], b: u32) -> [i32; 4] {
    let (d0, d1, d2, d3): (i32, i32, i32, i32);
    // SAFETY: the caller's contract; the instruction reads and writes registers only.
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, \
             {%7, %8, %9, %10};",
            out("=r") d0,
            out("=r") d1,
            out("=r") d2,
            out("=r") d3,
            in("r") a[0],
            in("r") a[1],
            in("r") b,
            in("r") 0i32,
            in("r") 0i32,
            in("r") 0i32,
            in("r") 0i32,
        );
    }
    [d0, d1, d2, d3]
}

/// How the weight bytes of one (slab row, block) pair reach a stage, per
/// entry: `@copy` the `cp.async` part (the plane's codes), `@load` the
/// register part one step ahead, `@store` it decoded into the stage after
/// the step's compute; `@prep` binds `$pv`, at the head of each walk, to
/// what a layout that addresses a row rather than the block stream needs of
/// the pair's row (packed Q5_0 only; nothing for the others), `@stage_scale`
/// stages the tile's expert's tensor scale in shared memory before the K
/// walk, `@expert_scale` reads it back for the stores and `@scale` folds it
/// into a stored value (NVFP4 only); `@acc` is the exact walk's accumulators
/// of one n-tile, F64 for NVFP4 (module doc). `$live` is whether
/// the pair exists (its row inside the slab's live rows, its block below
/// K/32); `$n` is its block's index in the stack's block stream (`row · K/32
/// + block`), `$dst` its staged row's first word.
///
/// The `slab` entry's codes come by `@stage` instead, for the whole slab:
/// step `$h`'s codes into stage `$st`, thread `tid` the 16-byte chunk
/// `tid % 4` of slab rows `tid / 4` and `tid / 4 + 64`, so each of a warp's
/// eight rows is one 64-byte read. Its `@prep` binds the pair's window of
/// eight scales (`@window`, one 16-byte read every four steps), and `@group`
/// is its block raster's group ([`raster32`]; 1 for every other entry).
macro_rules! w32 {
    (@copy plane($qs:ident, $wd:ident), $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
        if $live {
            // SAFETY: block `n` is inside the stack (the launch contract
            // holds `8 · n_experts · rows · K/32` words), its eight words
            // start at word `8n`, 16-byte aligned by the host's base check;
            // the destination is the pair's eight words of its staged row,
            // 16-byte aligned.
            unsafe {
                let src = $qs.as_ptr().add(8 * $n);
                let dst = $dst.add(8 * $pj);
                cp_async_cg_16(dst, src);
                cp_async_cg_16(dst.add(4), src.add(4));
            }
        }
    }};
    (@copy $kind:ident($($w:ident),+), $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
        let _ = ($live, $n, $dst, $pj);
    }};
    (
        @stage slab($qs:ident, $wd:ident), $wt:expr, $st:expr, $slab_n:expr, $kb:expr, $h:expr,
        $live_rows:expr, $tid:expr
    ) => {{
        // SAFETY: stage `st` of WT.
        let stage = unsafe { $wt.add($st * W32_STAGE_W) };
        let (r0, c): (usize, usize) = ($tid / 4, $tid % 4);
        if 2 * $h + c / 2 < $kb {
            for i in 0..2usize {
                ::cuda_device::thread::__unroll_config::<0>();
                let r = r0 + GEMM_BM / 2 * i;
                if r < $live_rows {
                    // SAFETY: block `2h + c/2` of slab row `r` is inside the
                    // stack (the launch contract holds `8 · n_experts · rows ·
                    // K/32` words); its chunk `c % 2` starts at a 16-byte
                    // aligned word, as does chunk `c` of the row's staged
                    // codes in this stage.
                    unsafe {
                        cp_async_cg_16(
                            stage.add(r * W32_ROW + 4 * c),
                            $qs.as_ptr().add(8 * ($slab_n + r * $kb + 2 * $h) + 4 * c),
                        );
                    }
                }
            }
        }
    }};
    (
        @stage $kind:ident($($w:ident),+), $wt:expr, $st:expr, $slab_n:expr, $kb:expr, $h:expr,
        $live_rows:expr, $tid:expr
    ) => {{
        let _ = ($wt, $st, $slab_n, $kb, $h, $live_rows, $tid);
    }};
    (@prep q50p($w:ident, $sw:ident), $pv:ident, $row:expr, $first:expr, $kb:expr) => {
        // The pair's row as the packed layout addresses it: its first word
        // (rows `5 · kb + s_words` words apart), its first block's index in
        // the stack's block stream, and its block count.
        let $pv = ($row * (5 * $kb + $sw as usize), $first, $kb);
    };
    (@group slab($qs:ident, $wd:ident), $blocks:expr) => {
        if $blocks * Q8_0_BLOCK_BYTES >= GEMM32_RASTER_BYTES {
            GEMM32_GROUP
        } else {
            1
        }
    };
    (@group $kind:ident($($w:ident),+), $blocks:expr) => {
        1
    };
    (@window slab($qs:ident, $wd:ident), $n:expr) => {{
        // Block `n`'s group of eight scales, one 16-byte read.
        let n: usize = $n;
        // SAFETY: @prep admits full windows; each live @load index belongs
        // to a stack of 16-row multiples. The host checks 16-byte alignment.
        unsafe { (*($wd.as_ptr().add(n / 8 * 8) as *const U32x4)).0 }
    }};
    (@prep slab($qs:ident, $wd:ident), $pv:ident, $row:expr, $first:expr, $kb:expr) => {
        // The thread's window of the scale plane, from its row's first block
        // (zeros for a window past the plane: that pair is never live).
        let mut $pv = if 8 * ($first / 8) + 8 <= $wd.len() {
            w32!(@window slab($qs, $wd), $first)
        } else {
            [0u32; 4]
        };
    };
    (@prep $kind:ident($($w:ident),+), $pv:ident, $row:expr, $first:expr, $kb:expr) => {};
    (@load plane($qs:ident, $wd:ident), $live:expr, $n:expr, $pv:ident) => {{
        if $live {
            // SAFETY: block `n` is inside the stack, whose scale plane holds
            // `n_experts · rows · K/32` entries (launch contract).
            unsafe { *$wd.get_unchecked($n) }
        } else {
            0u16
        }
    }};
    (@load slab($qs:ident, $wd:ident), $live:expr, $n:expr, $pv:ident) => {{
        // `n` grows by two a step, so it enters a new group of eight when
        // `n % 8 < 2`; the window moves there. The window is the relayed
        // value: the store reads the scale out of it, so the load's latency
        // hides behind the step's compute.
        if $live {
            // A live pair's block `n` is below the plane's `n_experts · rows ·
            // K/32` entries (launch contract), a multiple of eight, so its
            // group of eight lies inside the plane.
            let n: usize = $n;
            if n % 8 < 2 {
                $pv = w32!(@window slab($qs, $wd), n);
            }
        }
        $pv
    }};
    (@load q8f($w:ident), $live:expr, $n:expr, $pv:ident) => {{
        if $live {
            let a = Q8_0_BLOCK_BYTES * $n / 4;
            // SAFETY: the block's 34 bytes end at byte `34n + 33`, inside the
            // stream (launch contract), so its nine covering words `a ..= a +
            // 8` end at word `(34n + 33) / 4`, inside `w`.
            unsafe {
                [
                    *$w.get_unchecked(a),
                    *$w.get_unchecked(a + 1),
                    *$w.get_unchecked(a + 2),
                    *$w.get_unchecked(a + 3),
                    *$w.get_unchecked(a + 4),
                    *$w.get_unchecked(a + 5),
                    *$w.get_unchecked(a + 6),
                    *$w.get_unchecked(a + 7),
                    *$w.get_unchecked(a + 8),
                ]
            }
        } else {
            [0u32; 9]
        }
    }};
    (@load q51($w:ident), $live:expr, $n:expr, $pv:ident) => {{
        if $live {
            let a = Q5_1_BLOCK_WORDS * $n;
            // SAFETY: the block's six words are inside the stack (launch
            // contract: `6 · n_experts · rows · K/32` words).
            unsafe {
                [
                    *$w.get_unchecked(a),
                    *$w.get_unchecked(a + 1),
                    *$w.get_unchecked(a + 2),
                    *$w.get_unchecked(a + 3),
                    *$w.get_unchecked(a + 4),
                    *$w.get_unchecked(a + 5),
                ]
            }
        } else {
            [0u32; 6]
        }
    }};
    (@load iq4nl($w:ident), $live:expr, $n:expr, $pv:ident) => {{
        if $live {
            let a = IQ4_NL_BLOCK_BYTES * $n / 4;
            // SAFETY: the block's 18 bytes end at byte `18n + 17`, inside the
            // stream (launch contract), so its five covering words `a ..= a +
            // 4` are inside `w`.
            unsafe {
                [
                    *$w.get_unchecked(a),
                    *$w.get_unchecked(a + 1),
                    *$w.get_unchecked(a + 2),
                    *$w.get_unchecked(a + 3),
                    *$w.get_unchecked(a + 4),
                ]
            }
        } else {
            [0u32; 5]
        }
    }};
    (@load q50p($w:ident, $sw:ident), $live:expr, $n:expr, $pv:ident) => {{
        if $live {
            let (wrow, first, kb) = $pv;
            // SAFETY: the pair's row is inside the stack and its block `n −
            // first` is below `K/32 = kb`; the row's `q5_0_row_words(kb)`
            // words fit its `5 · kb + s_words` (`2 · s_words >= kb`), and the
            // stack holds `n_experts · rows` such rows (launch contract).
            unsafe { crate::q5::q5_0_block($w, wrow, kb, $n - first) }
        } else {
            ([0u32; 8], 0.0f32)
        }
    }};
    (@store plane($qs:ident, $wd:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr, $pv:ident) => {{
        if $live {
            // SAFETY: word 16 + pj of the pair's staged row, which no warp
            // reads before the next barrier.
            unsafe { *$dst.add(16 + $pj) = half_bits_to_f32($raw).to_bits() };
        }
    }};
    (@store slab($qs:ident, $wd:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr, $pv:ident) => {{
        if $live {
            let (n, v): (usize, [u32; 4]) = ($n, $raw);
            let p = if n & 4 == 0 {
                [v[0], v[1]]
            } else {
                [v[2], v[3]]
            };
            let word = if n & 2 == 0 { p[0] } else { p[1] };
            let d = half_bits_to_f32((word >> (16 * (n & 1))) as u16);
            // SAFETY: word 16 + pj of the pair's staged row, which no warp
            // reads before the next barrier.
            unsafe { *$dst.add(16 + $pj) = d.to_bits() };
        }
    }};
    (@store q8f($w:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr, $pv:ident) => {{
        if $live {
            let (d, c) = q8_0_file_block($raw, Q8_0_BLOCK_BYTES * $n);
            // SAFETY: the pair's eight code words and its `d` word of its
            // staged row, which no warp reads before the next barrier.
            unsafe {
                let p = $dst.add(8 * $pj);
                *p = c[0];
                *p.add(1) = c[1];
                *p.add(2) = c[2];
                *p.add(3) = c[3];
                *p.add(4) = c[4];
                *p.add(5) = c[5];
                *p.add(6) = c[6];
                *p.add(7) = c[7];
                *$dst.add(16 + $pj) = half_bits_to_f32(d).to_bits();
            }
        }
    }};
    (@store q51($w:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr, $pv:ident) => {{
        if $live {
            let c = q5_1_codes($raw[1], [$raw[2], $raw[3], $raw[4], $raw[5]]);
            // SAFETY: the pair's eight code words and its `d` and `m` words
            // of its staged row, which no warp reads before the next barrier.
            unsafe {
                let p = $dst.add(8 * $pj);
                *p = c[0];
                *p.add(1) = c[1];
                *p.add(2) = c[2];
                *p.add(3) = c[3];
                *p.add(4) = c[4];
                *p.add(5) = c[5];
                *p.add(6) = c[6];
                *p.add(7) = c[7];
                *$dst.add(16 + $pj) = half_bits_to_f32(($raw[0] & 0xffff) as u16).to_bits();
                *$dst.add(18 + $pj) = half_bits_to_f32(($raw[0] >> 16) as u16).to_bits();
            }
        }
    }};
    (@store iq4nl($w:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr, $pv:ident) => {{
        if $live {
            let (d, c) = iq4_nl_file_block($raw, IQ4_NL_BLOCK_BYTES * $n);
            // SAFETY: the pair's eight code words and its `d` word of its
            // staged row, which no warp reads before the next barrier.
            unsafe {
                let p = $dst.add(8 * $pj);
                *p = c[0];
                *p.add(1) = c[1];
                *p.add(2) = c[2];
                *p.add(3) = c[3];
                *p.add(4) = c[4];
                *p.add(5) = c[5];
                *p.add(6) = c[6];
                *p.add(7) = c[7];
                *$dst.add(16 + $pj) = half_bits_to_f32(d).to_bits();
            }
        }
    }};
    (
        @store q50p($w:ident, $sw:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr,
        $pv:ident
    ) => {{
        if $live {
            let (c, d) = $raw;
            // SAFETY: the pair's eight code words and its `d` word of its
            // staged row, which no warp reads before the next barrier.
            unsafe {
                let p = $dst.add(8 * $pj);
                *p = c[0];
                *p.add(1) = c[1];
                *p.add(2) = c[2];
                *p.add(3) = c[3];
                *p.add(4) = c[4];
                *p.add(5) = c[5];
                *p.add(6) = c[6];
                *p.add(7) = c[7];
                *$dst.add(16 + $pj) = d.to_bits();
            }
        }
    }};
    (@stage_scale nvfp4p($qs:ident, $e:ident, $scales:ident), $ex:expr, $xs:expr) => {{
        // SAFETY: the tile's expert is below the route table's experts, the
        // stack's, and `scales` holds one an expert (launch contract); `xs`
        // is the block's one shared scale word, written by one thread before
        // the barrier that publishes the tile's slots.
        unsafe { *$xs = *$scales.get_unchecked($ex) }
    }};
    (@stage_scale $kind:ident($($w:ident),+), $ex:expr, $xs:expr) => {{
        let _ = ($ex, $xs);
    }};
    (@expert_scale nvfp4p($qs:ident, $e:ident, $scales:ident), $xs:expr) => {{
        // SAFETY: the shared scale word `@stage_scale` wrote, published by
        // the barriers since and not written again.
        unsafe { *$xs }
    }};
    (@expert_scale $kind:ident($($w:ident),+), $xs:expr) => {{
        let _ = $xs;
        1.0f32
    }};
    (@acc nvfp4p($($w:ident),+)) => { [0.0f64; 4] };
    (@acc $kind:ident($($w:ident),+)) => { [0.0f32; 4] };
    (@scale nvfp4p($qs:ident, $e:ident, $scales:ident), $s:expr, $v:expr) => {
        $crate::gemm::grouped::scaled_sum([f64::from($v), 0.0], [$s, 0.0], 1.0, 0.0) as f32
    };
    (@scale $kind:ident($($w:ident),+), $s:expr, $v:expr) => {{
        let _ = $s;
        $v
    }};
}

/// Stage step `h`'s activations into stage `st`: thread `tid`'s 16-byte
/// chunk `tid % 4` of column `tid / 4`'s step codes, and its scale word `tid
/// % 4` (`d` of blocks 0 and 1, then `s` of blocks 0 and 1 when `mins`, a
/// literal) — live columns only; a column past the tile's length reads the
/// tile's first slot's column, so every source is a real column.
macro_rules! stage32_act {
    ($h:expr, $st:expr, $mins:literal, $tid:ident, $ncols:ident, $col_q:ident, $col_d:ident,
     $q:ident, $d:ident, $s:ident, $bt:ident, $dt:ident, $acol_sh:ident) => {{
        let (h, st): (usize, usize) = ($h, $st);
        let (an, ac) = ($tid / 4, $tid % 4);
        if an < $ncols {
            // SAFETY: an < ncols <= GEMM_BN bounds the shared read; the
            // column is an activation column below act_cols, so its step-h
            // codes (words 16h .. 16h + 15 of its 16·steps) and scales (2h,
            // 2h + 1 of its 2·steps) are inside the planes (launch contract),
            // the codes 16-byte aligned; each destination is this thread's own
            // chunk or word of stage `st`, read by no warp before the barrier
            // after the wait that completes it.
            unsafe {
                let c = *$acol_sh.add(an) as usize;
                cp_async_cg_16(
                    $bt.add(st * B32_STAGE_W + an * B32_ROW + 4 * ac),
                    $q.as_ptr().add(c * $col_q + 16 * h + 4 * ac),
                );
                let dst = $dt.add(st * D32_STAGE_W + an * D32_ROW + ac);
                if ac < 2 {
                    cp_async_ca_4(dst, $d.as_ptr().add(c * $col_d + 2 * h + ac).cast::<u32>());
                } else if $mins {
                    cp_async_ca_4(
                        dst,
                        $s.as_ptr().add(c * $col_d + 2 * h + ac - 2).cast::<u32>(),
                    );
                }
            }
        }
    }};
}

/// One n-tile's product of one block into the accumulators: `isum` the
/// `mma` result (rows `g`, `g + 8` by columns `2t`, `2t + 1`), `dw`/`mw`
/// the two rows' block scale and min, `da`/`sa` the two columns' block scale
/// bits and code sum bits — the contract's order per output, by `grouped.rs`'s
/// `fold4` `fold`.
macro_rules! epi32 {
    ($mins:literal, $isum:ident, $dw:ident, $mw:ident, $da:expr, $sa:expr, $fold:tt) => {{
        let da: [u32; 2] = $da;
        let sa: [f32; 2] = if $mins {
            let sa: [u32; 2] = $sa;
            [sa[0] as i32 as f32, sa[1] as i32 as f32]
        } else {
            [0.0; 2]
        };
        let mw: [f32; 2] = if $mins { $mw } else { [0.0; 2] };
        fold4!(
            $fold,
            [[$dw[0], mw[0]], [$dw[1], mw[1]]],
            [
                [$isum[0] as f32, sa[0]],
                [$isum[1] as f32, sa[1]],
                [$isum[2] as f32, sa[0]],
                [$isum[3] as f32, sa[1]],
            ],
            [f32::from_bits(da[0]), f32::from_bits(da[1])]
        )
    }};
}

/// One n-tile's product of one NVFP4 32-value unit into the accumulators:
/// `i0`, `i1` the unit's two 16-value sub-blocks' exact i32 `mma` sums (rows
/// `g`, `g + 8` by columns `2t`, `2t + 1`), `d0`/`d1` the two rows' staged
/// sub-block scales, `da` the two columns' block-scale bits — the contract's
/// order per output: `d0·f32(i0)` exact, one fma for `d1`, one for the
/// accumulator.
macro_rules! epi32_nv {
    ($d0:ident, $d1:ident, $i0:ident, $i1:ident, $da:expr, $fold:tt) => {{
        let da: [u32; 2] = $da;
        fold4!(
            $fold,
            [[$d0[0], $d1[0]], [$d0[1], $d1[1]]],
            [
                [$i0[0] as f32, $i1[0] as f32],
                [$i0[1] as f32, $i1[1] as f32],
                [$i0[2] as f32, $i1[2] as f32],
                [$i0[3] as f32, $i1[3] as f32],
            ],
            [f32::from_bits(da[0]), f32::from_bits(da[1])]
        )
    }};
}

/// One n-tile's product of one 32-value block into the accumulators, per
/// family: `plain` the one `mma.m16n8k32` and [`epi32!`] of the five
/// one-scale layouts; `nvfp4p` the block's two 16-value sub-block sums, each
/// one [`mma_m16n8k16_s32_s8_zero`] over its half of the A fragment and its
/// word of the B fragment, and [`epi32_nv!`]'s nested scales.
macro_rules! mma_epi32 {
    (
        plain, $mins:literal, $a:ident, [$($b:expr),+], $dw:ident, $mw:ident, $da:expr, $sa:expr,
        $fold:tt
    ) => {{
        // SAFETY: every lane of the warp reaches this `mma.sync` (the
        // branches around it are uniform) with its fragments in the
        // instruction's layout.
        let i0 = unsafe { mma_m16n8k32_s32_s8([0; 4], $a, [$($b),+]) };
        epi32!($mins, i0, $dw, $mw, $da, $sa, $fold);
    }};
    (
        nvfp4p, $mins:literal, $a:ident, [$b0:expr, $b1:expr], $d0:ident, $d1:ident, $da:expr,
        $sa:expr, $fold:tt
    ) => {{
        // SAFETY: every lane of the warp reaches both `mma.sync`s (the
        // branches around them are uniform) with its fragments in the
        // instruction's layout.
        let (i0, i1) = unsafe {
            (
                mma_m16n8k16_s32_s8_zero([$a[0], $a[1]], $b0),
                mma_m16n8k16_s32_s8_zero([$a[2], $a[3]], $b1),
            )
        };
        epi32_nv!($d0, $d1, i0, i1, $da, $fold);
    }};
}

/// One n-tile's operands of a step out of the stage: `bf` the B fragments
/// of both blocks (`ldmatrix.x4` at the lane's column and chunk) and `x0`,
/// `x1` the activation scale words of the lane's two columns.
macro_rules! nt_frags {
    ($bst:ident, $dst:ident, $nt:expr, $t:ident, $b_off:ident) => {{
        let n0 = 8 * $nt + 2 * $t;
        // SAFETY: column 8·nt + lane % 8 is below ncols of this stage (the
        // whole warp issues it: the n-tile is live for every lane); columns
        // n0, n0 + 1 < GEMM_BN: two 16-byte reads of this stage's scales. All
        // published by the step's barrier.
        unsafe {
            let bf = ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                $bst.add(8 * $nt * B32_ROW + $b_off)
                    .cast_const()
                    .cast::<u8>(),
            ));
            let x0 = (*($dst.add(D32_ROW * n0) as *const U32x4)).0;
            let x1 = (*($dst.add(D32_ROW * (n0 + 1)) as *const U32x4)).0;
            (bf, x0, x1)
        }
    }};
}

/// The n-tile walk of one step of one warp: for every live n-tile (the
/// exact walk's one, `grouped.rs`'s `gemm_nts`), the products of the step's
/// blocks into the accumulators. `each` is the walk that tests the tile
/// (`nt < nt_live`) and the second block (`two`) around every product; the
/// families but `nvfp4p`, and the exact walk, run it as it stands. `nvfp4p`
/// first tests the two once, warp-uniformly: with every n-tile live and both
/// blocks present (every step of a full tile, K a multiple of 64) it runs the
/// same products with no branch between them, one straight stretch of
/// `mma.sync` and epilogues that the assembler may overlap across n-tiles and
/// blocks — the order of every output's operations is the walk's, so the
/// bits are the same.
macro_rules! nt_walk {
    (
        @each $epi:ident, $mode:ident, $nx:expr, $mins:literal, $nt_live:ident, $two:ident,
        $t:ident, $bst:ident, $dst:ident, $b_off:ident, $a0:ident, $a1:ident, $dw0:ident,
        $dw1:ident, $mw0:ident, $mw1:ident, $acc:ident, $off:ident
    ) => {
        gemm_nts!($mode, $nx, false, $nt_live, |nt, j0| {
            let (bf, x0, x1) = nt_frags!($bst, $dst, nt, $t, $b_off);
            // SAFETY: every lane of the warp reaches the `mma.sync`
            // inside (the branches around it are uniform) with its
            // fragments in the instruction's layout.
            mma_epi32!($epi, $mins, $a0, [bf[0], bf[1]], $dw0, $mw0, [x0[0], x1[0]],
                [x0[2], x1[2]], ($mode, $acc, j0, $off));
            if $two {
                // SAFETY: `two` and the branches around it are
                // warp-uniform, so every lane reaches the `mma.sync`
                // inside as well.
                mma_epi32!($epi, $mins, $a1, [bf[2], bf[3]], $dw1, $mw1, [x0[1], x1[1]],
                    [x0[3], x1[3]], ($mode, $acc, j0, $off));
            }
        })
    };
    (nvfp4p, exact, $($rest:tt)*) => {
        nt_walk!(@each nvfp4p, exact, $($rest)*)
    };
    (
        nvfp4p, fast, $nx:expr, $mins:literal, $nt_live:ident, $two:ident, $t:ident,
        $bst:ident, $dst:ident, $b_off:ident, $a0:ident, $a1:ident, $dw0:ident, $dw1:ident,
        $mw0:ident, $mw1:ident, $acc:ident, $off:ident
    ) => {
        if $nt_live == GEMM_NT && $two {
            gemm_nts!(fast, $nx, true, $nt_live, |nt, j0| {
                let (bf, x0, x1) = nt_frags!($bst, $dst, nt, $t, $b_off);
                // SAFETY: every lane of the warp reaches both `mma.sync`s
                // (the test above is warp-uniform and nothing branches in
                // between) with its fragments in the instruction's layout.
                mma_epi32!(nvfp4p, $mins, $a0, [bf[0], bf[1]], $dw0, $mw0, [x0[0], x1[0]],
                    [x0[2], x1[2]], (fast, $acc, j0, $off));
                mma_epi32!(nvfp4p, $mins, $a1, [bf[2], bf[3]], $dw1, $mw1, [x0[1], x1[1]],
                    [x0[3], x1[3]], (fast, $acc, j0, $off));
            })
        } else {
            nt_walk!(@each nvfp4p, fast, $nx, $mins, $nt_live, $two, $t, $bst, $dst, $b_off,
                $a0, $a1, $dw0, $dw1, $mw0, $mw1, $acc, $off);
        }
    };
    (plain, $($rest:tt)*) => {
        nt_walk!(@each plain, $($rest)*)
    };
}

/// Copy step `hh`'s NVFP4 weights of the relay pair's row into weight stage
/// `hh % NV_STAGES` (the `nvfp4p` walk of [`gemm32_block`]): the pair's
/// 16-byte code vector, unit `pj` of the step, at half `pj` of the row's
/// eight code words (the halves swapped on rows with bit 2 set, so a warp's
/// reads in [`nv_frags`] fall in sixteen banks) and, from the row's first
/// thread, the step's scale word after the stage's codes. A step past the
/// row's blocks copies nothing.
macro_rules! nv_copy {
    (
        $qs:ident, $e:ident, $wt:ident, $hh:expr, $pr:ident, $pj:ident, $prow_live:ident,
        $pn0:ident, $kb:ident
    ) => {{
        let hh: usize = $hh;
        if $prow_live && 2 * hh + $pj < $kb {
            let n = $pn0 + 2 * hh;
            let half = $pj ^ (($pr >> 2) & 1);
            // SAFETY: stage `hh % NV_STAGES` is inside WT (NV_STAGES ·
            // NV_STAGE_W words fit).
            let wst = unsafe { $wt.add((hh % NV_STAGES) * NV_STAGE_W) };
            // SAFETY: unit n's code words 4n ..= 4n + 3 are inside the plane
            // (launch contract), 16-byte aligned (the host checks the plane);
            // the destination, the slab row's half of the stage's codes, is
            // this thread's own chunk, read by no warp before the barrier after
            // the wait that completes it.
            unsafe { cp_async_cg_16(wst.add($pr * 8 + 4 * half), $qs.as_ptr().add(4 * n)) };
            if $pj == 0 {
                // SAFETY: the block's scale word n >> 1 (n is even for pj = 0:
                // K a multiple of 64) is inside the plane (launch contract);
                // the destination is the slab row's scale word of the stage,
                // this thread's own, read by no warp before the barrier after
                // the wait that completes it.
                unsafe { cp_async_ca_4(wst.add(NV_CODE_W + $pr), $e.as_ptr().add(n >> 1)) };
            }
        }
    }};
}

/// One warp's operands of a step from NVFP4 weight stage `wst`: the A
/// fragments of both units in the staged walk's `ldmatrix.x4` order (rows
/// `lr`, `lr + 8` of the first sub-block, then of the second), each lane's
/// word decoded by [`nv_half`] (a sub-block's 16 values are the low nibbles
/// of its two code words, then their high nibbles, so lane `t` takes word
/// `t & 1`, low for `t < 2` and high after), and the rows' sub-block scales:
/// `dw` the first sub-blocks', `mw` the second's, of units 0 and 1.
macro_rules! nv_frags {
    ($wst:ident, $lr:ident, $t:ident) => {{
        // Unit 0's half of the rows' code words ([`nv_copy`]'s swap: bit 2
        // of the row, the same for `lr` and `lr + 8`) and unit 1's.
        let h0 = 4 * (($lr >> 2) & 1) + ($t & 1);
        let h1 = (4 + h0) & 7;
        let hi = $t >= 2;
        let (r0, r1) = (8 * $lr, 8 * ($lr + 8));
        // SAFETY: rows lr and lr + 8 are below GEMM_BM, so a code word `r + h`
        // (h < 8) of either row and the rows' scale words `NV_CODE_W + lr`
        // and `+ 8` stay inside the stage, published by the step's barrier.
        let word = |at: usize| unsafe { *$wst.add(at) };
        let a0 = [
            nv_half(word(r0 + h0), hi),
            nv_half(word(r1 + h0), hi),
            nv_half(word(r0 + h0 + 2), hi),
            nv_half(word(r1 + h0 + 2), hi),
        ];
        let a1 = [
            nv_half(word(r0 + h1), hi),
            nv_half(word(r1 + h1), hi),
            nv_half(word(r0 + h1 + 2), hi),
            nv_half(word(r1 + h1 + 2), hi),
        ];
        let (e0, e1) = (word(NV_CODE_W + $lr), word(NV_CODE_W + $lr + 8));
        (
            a0,
            a1,
            [nv_scale(e0, 0), nv_scale(e1, 0)],
            [nv_scale(e0, 2), nv_scale(e1, 2)],
            [nv_scale(e0, 1), nv_scale(e1, 1)],
            [nv_scale(e0, 3), nv_scale(e1, 3)],
        )
    }};
}

/// The 32-value GEMM block: one tile's columns against one 128-row slab of
/// that tile's expert (module doc). A macro, not a function, for the
/// K-quant block's reason (`grouped.rs`, `gemm_block`): the entries share
/// the walk and differ only in the weight relay `w32` names, and a
/// function boundary would move the accumulators out of registers.
///
/// `wt` names the relay and the entry's weight parameters, `mins` (a
/// literal) whether the type has a min term (Q5_1), `epi` the per-block
/// product's family (`plain`, or `nvfp4p` for the two-sub-block planes).
/// `params` is the rest of
/// the entry's parameters in order. The expansion's `unsafe` blocks rest on
/// the entry's launch contract and on the route table the host guarantees
/// was built for this stack and slot count.
///
/// The blocks of tile index 0 — there is one per slab whatever the table
/// holds — first write NaN over their slab's rows of every slot the route
/// refused, as the K-quant block does.
macro_rules! gemm32_block {
    (
        wt: $kind:ident($($w:ident),+),
        mins: $mins:literal,
        epi: $epi:ident,
        params: (
            $q:ident, $d:ident, $s:ident, $cols:ident, $tiles:ident, $n_tiles:ident,
            $n_experts:ident, $rows:ident, $kb:ident, $steps:ident, $act_cols:ident,
            $n_slots:ident, $max_tiles:ident, $slot_div:ident, $row_tiles:ident, $y:tt $(,)?
        ) $(,)?
    ) => {{
        static mut WT: SharedArray<u32, { 2 * W32_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut BT: SharedArray<u32, { 2 * B32_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut DT: SharedArray<u32, { 2 * D32_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // The tile's expert's tensor scale (NVFP4), staged here so no register
        // holds the expert across the K walk.
        static mut XSCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut VOTES: SharedArray<u32, VOTE_WORDS> = SharedArray::UNINIT;
        // SAFETY: each `static mut` is this block's own shared allocation,
        // reached raw; the block walk bounds every index and orders every
        // cross-thread read behind a barrier.
        let (wt, bt, dt, slot_sh, acol_sh, xs, votes) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut WT),
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut DT),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
                SharedArray::as_raw_mut_ptr(&raw mut XSCALE),
                SharedArray::as_raw_mut_ptr(&raw mut VOTES),
            )
        };
        // Read by the launch contract only.
        let _ = ($n_experts, $act_cols, $max_tiles);

        let b = thread::blockIdx_x() as usize;
        let rt_n = $row_tiles as usize;
        let (tile, rt) = match w32!(@group $kind($($w),+), $rows as usize * $kb as usize) {
            1 => {
                let tile = b / rt_n;
                (tile, b - tile * rt_n)
            }
            group => {
                // SAFETY: n_tiles.len() >= 2 by the launch contract.
                let live = unsafe { *$n_tiles.get_unchecked(0) } as usize;
                // Block-uniform: a block past the live tiles' (tile 0's when
                // the table holds none) returns at once.
                let Some(p) = raster32(b, rt_n, live.max(1), group) else {
                    return;
                };
                p
            }
        };
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
                        // n_slots at cols[first..n_slots], inside cols.
                        let slot = unsafe { *$cols.get_unchecked(first + u / GEMM_BM) } as usize;
                        gemm32_block!(@refused $y, slot, r, rows_n);
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
        // tile's length repeats column 0, so the last n-tile's padding stages
        // real codes and its results are simply not stored.
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
        // The first thread past the slot stagers (GEMM_THREADS is 4 · GEMM_BN),
        // so its scale load overlaps their column loads.
        if tid == GEMM_BN {
            w32!(@stage_scale $kind($($w),+), e, xs);
        }
        thread::sync_threads();

        let nt_live = len.div_ceil(8);
        let ncols = 8 * nt_live;
        let rows_n = $rows as usize;
        let slab0 = rt * GEMM_BM;
        // Rows are a multiple of 16, so a warp's sixteen rows are all live or
        // all past the expert's rows.
        let live_rows = (rows_n - slab0).min(GEMM_BM);
        let row0 = slab0 + 16 * wid;
        // Warp-uniform: a warp past the expert's rows stages and waits with
        // the block but loads, multiplies and stores nothing.
        let active = row0 < rows_n;
        let kb = $kb as usize;
        let steps = $steps as usize;
        let (col_q, col_d) = (16 * steps, 2 * steps);
        // This thread's relay pair: slab row `pr`, block `pj` of each step;
        // `pn0` its row's first block in the stack's block stream.
        let (pr, pj) = (tid / 2, tid % 2);
        let prow_live = pr < live_rows;
        let pn0 = (e * rows_n + slab0 + pr) * kb + pj;
        // The slab's first block in the stack's block stream.
        let slab_n = (e * rows_n + slab0) * kb;
        // Each lane's `ldmatrix` rows: A row `lane % 16` of the warp's slab
        // rows at 16-byte half `lane / 16`; B column `lane % 8` of an n-tile
        // at 16-byte chunk `lane / 8` (blocks 0 and 1, low and high half).
        let a_off = (16 * wid + (lane % 16)) * W32_ROW + 4 * (lane / 16);
        let b_off = (lane % 8) * B32_ROW + 4 * (lane / 8);
        let r_off = (16 * wid + g) * W32_ROW + 16;

        range_walks!(
            gemm32_block!($kind($($w),+), $mins, $epi, $q, $d, $s, (tid, t, active, ncols, nt_live,
                steps, kb, col_q, col_d, bt, dt, wt, acol_sh, pr, pj, prow: e * rows_n + slab0 + pr,
                prow_live, pn0, slab_n, live_rows, a_off, r_off, b_off)),
            gemm32_block!($kind($($w),+), $y, (t, active, nt_live, len, row0, g, rows_n, slot_sh,
                xs, wt, rt)),
            w32!(@acc $kind($($w),+)),
            active,
            nt_live,
            votes
        );
    }};
    (
        @walk $mode:ident, $nx:expr, $acc:ident, $off:ident, nvfp4p($qs:ident, $e:ident, $sc:ident),
        $mins:literal, $epi:ident, $q:ident, $d:ident, $s:ident, (
            $tid:ident, $t:ident, $active:ident, $ncols:ident, $nt_live:ident, $steps:ident,
            $kb:ident, $col_q:ident, $col_d:ident, $bt:ident, $dt:ident, $wt:ident,
            $acol_sh:ident, $pr:ident, $pj:ident, $prow:ident: $prow_row:expr,
            $prow_live:ident, $pn0:ident, $slab_n:ident, $live_rows:ident, $a_off:ident,
            $r_off:ident, $b_off:ident
        )
    ) => {{
        let _ = ($slab_n, $live_rows, $a_off, $r_off);
        // The lane's two slab rows (`g`, `g + 8` of its warp's sixteen).
        let lr = 16 * ($tid / 32) + ($tid % 32) / 4;
        // Step 0's activations, then steps 0 and 1's weights: three groups.
        stage32_act!(0usize, 0usize, $mins, $tid, $ncols, $col_q, $col_d, $q, $d, $s, $bt, $dt,
            $acol_sh);
        // SAFETY: commits only this thread's pending step-0 activation copies as one group.
        unsafe { cp_async_commit_group() };
        nv_copy!($qs, $e, $wt, 0usize, $pr, $pj, $prow_live, $pn0, $kb);
        // SAFETY: commits only this thread's pending step-0 weight copies as one group.
        unsafe { cp_async_commit_group() };
        nv_copy!($qs, $e, $wt, 1usize, $pr, $pj, $prow_live, $pn0, $kb);
        // SAFETY: commits only this thread's pending step-1 weight copies as one group.
        unsafe { cp_async_commit_group() };

        let mut h = 0usize;
        while h < $steps {
            let st = h & 1;
            // SAFETY: each step commits its activation group and then its
            // weight group, so the newest group is step h + 1's weights and
            // waiting for all but one completes step h's activations and
            // weights; the barrier publishes them and tells that every warp
            // is done with step h − 1, whose stages step h's copies refill.
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();
            if h + 1 < $steps {
                stage32_act!(h + 1, st ^ 1, $mins, $tid, $ncols, $col_q, $col_d, $q, $d, $s, $bt,
                    $dt, $acol_sh);
            }
            // SAFETY: one activation group a step, empty past the last step.
            unsafe { cp_async_commit_group() };
            nv_copy!($qs, $e, $wt, h + 2, $pr, $pj, $prow_live, $pn0, $kb);
            // SAFETY: one weight group a step, empty past the last step.
            unsafe { cp_async_commit_group() };

            if $active {
                // Block 1 of this step exists (warp-uniform).
                let two = 2 * h + 1 < $kb;
                // SAFETY: stage `st` of BT and DT and stage `h % NV_STAGES`
                // of WT are inside the shared arrays (NV_STAGES · NV_STAGE_W
                // ≤ WT's words), published by the barrier above.
                let (wst, bst, dst) = unsafe {
                    (
                        $wt.add((h % NV_STAGES) * NV_STAGE_W),
                        $bt.add(st * B32_STAGE_W),
                        $dt.add(st * D32_STAGE_W),
                    )
                };
                let (a0, a1, dw0, dw1, mw0, mw1) = nv_frags!(wst, lr, $t);
                nt_walk!($epi, $mode, $nx, $mins, $nt_live, two, $t, bst, dst, $b_off, a0, a1,
                    dw0, dw1, mw0, mw1, $acc, $off);
            }
            h += 1;
        }
    }};
    (
        @walk $mode:ident, $nx:expr, $acc:ident, $off:ident, $kind:ident($($w:ident),+),
        $mins:literal, $epi:ident, $q:ident, $d:ident, $s:ident, (
            $tid:ident, $t:ident, $active:ident, $ncols:ident, $nt_live:ident, $steps:ident,
            $kb:ident, $col_q:ident, $col_d:ident, $bt:ident, $dt:ident, $wt:ident,
            $acol_sh:ident, $pr:ident, $pj:ident, $prow:ident: $prow_row:expr,
            $prow_live:ident, $pn0:ident, $slab_n:ident, $live_rows:ident, $a_off:ident,
            $r_off:ident, $b_off:ident
        )
    ) => {{
        w32!(@prep $kind($($w),+), $prow, $prow_row, $pn0 - $pj, $kb);
        // Step 0 into stage 0.
        stage32_act!(0usize, 0usize, $mins, $tid, $ncols, $col_q, $col_d, $q, $d, $s, $bt, $dt,
            $acol_sh);
        {
            let live = $prow_live && $pj < $kb;
            // SAFETY: the pair's staged row of stage 0, inside WT.
            let dst = unsafe { $wt.add($pr * W32_ROW) };
            w32!(@copy $kind($($w),+), live, $pn0, dst, $pj);
            w32!(@stage $kind($($w),+), $wt, 0usize, $slab_n, $kb, 0usize, $live_rows, $tid);
            let raw = w32!(@load $kind($($w),+), live, $pn0, $prow);
            w32!(@store $kind($($w),+), raw, live, $pn0, dst, $pj, $prow);
        }
        // SAFETY: commits this thread's copies above as one group.
        unsafe { cp_async_commit_group() };

        let mut h = 0usize;
        while h < $steps {
            let st = h & 1;
            let nx = h + 1 < $steps;
            // Step h + 1's pair, loaded now, stored after this step's compute.
            let nlive = nx && $prow_live && 2 * (h + 1) + $pj < $kb;
            let nn = $pn0 + 2 * (h + 1);
            let raw = w32!(@load $kind($($w),+), nlive, nn, $prow);
            // SAFETY: one group per step (possibly empty) and none newer, so
            // waiting for all leaves step h's copies complete; the barrier
            // then publishes them and every relay store of step h, and tells
            // that every warp is done with stage st ^ 1.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
            // SAFETY: the pair's staged row of stage st ^ 1, inside WT.
            let ndst = unsafe { $wt.add((st ^ 1) * W32_STAGE_W + $pr * W32_ROW) };
            if nx {
                stage32_act!(h + 1, st ^ 1, $mins, $tid, $ncols, $col_q, $col_d, $q, $d, $s, $bt,
                    $dt, $acol_sh);
                w32!(@copy $kind($($w),+), nlive, nn, ndst, $pj);
                w32!(@stage $kind($($w),+), $wt, st ^ 1, $slab_n, $kb, h + 1, $live_rows, $tid);
            }
            // SAFETY: one group per step, empty on the last.
            unsafe { cp_async_commit_group() };

            if $active {
                // Block 1 of this step exists (warp-uniform).
                let two = 2 * h + 1 < $kb;
                // SAFETY: stage `st` is inside each shared array, and every
                // offset below stays inside its stage: rows below GEMM_BM of
                // W32_ROW words, columns below GEMM_BN of B32_ROW and D32_ROW.
                let (wst, bst, dst) = unsafe {
                    (
                        $wt.add(st * W32_STAGE_W),
                        $bt.add(st * B32_STAGE_W),
                        $dt.add(st * D32_STAGE_W),
                    )
                };
                // SAFETY: the whole warp issues each `ldmatrix` (the branches
                // around them are warp-uniform) with 16-byte-aligned rows of
                // this stage, published by the barrier above.
                let (a0, a1) = unsafe {
                    let p = wst.add($a_off);
                    let a0 = ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                        p.cast_const().cast::<u8>(),
                    ));
                    let a1 = if two {
                        ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                            p.add(8).cast_const().cast::<u8>(),
                        ))
                    } else {
                        [0u32; 4]
                    };
                    (a0, a1)
                };
                // SAFETY: words 16 .. 19 of rows g and g + 8 of the warp's
                // rows in this stage, published above.
                let (dw0, dw1, mw0, mw1) = unsafe {
                    let p0 = wst.add($r_off);
                    let p1 = p0.add(8 * W32_ROW);
                    (
                        [f32::from_bits(*p0), f32::from_bits(*p1)],
                        [f32::from_bits(*p0.add(1)), f32::from_bits(*p1.add(1))],
                        [f32::from_bits(*p0.add(2)), f32::from_bits(*p1.add(2))],
                        [f32::from_bits(*p0.add(3)), f32::from_bits(*p1.add(3))],
                    )
                };
                nt_walk!($epi, $mode, $nx, $mins, $nt_live, two, $t, bst, dst, $b_off, a0, a1,
                    dw0, dw1, mw0, mw1, $acc, $off);
            }

            if nx {
                w32!(@store $kind($($w),+), raw, nlive, nn, ndst, $pj, $prow);
            }
            h += 1;
        }
    }};
    (
        @store $mode:ident, $nx:expr, $acc:ident, $kind:ident($($w:ident),+),
        [swiglu($($sw:tt)+)], (
            $t:ident, $active:ident, $nt_live:ident, $len:ident, $row0:ident, $g:ident,
            $rows_n:ident, $slot_sh:ident, $xs:ident, $wt:ident, $rt:ident
        )
    ) => {{
        let ws = w32!(@expert_scale $kind($($w),+), $xs);
        swiglu32!(@store $mode, $nx, $acc, |v| w32!(@scale $kind($($w),+), ws, v), [$($sw)+],
            ($t, $active, $nt_live, $len, $row0, $g, $rows_n, $slot_sh, $wt, $rt));
    }};
    (
        @store $mode:ident, $nx:expr, $acc:ident, $kind:ident($($w:ident),+), $y:ident, (
            $t:ident, $active:ident, $nt_live:ident, $len:ident, $row0:ident, $g:ident,
            $rows_n:ident, $slot_sh:ident, $xs:ident, $wt:ident, $rt:ident
        )
    ) => {{
        let _ = ($wt, $rt);
        if $active {
            let ws = w32!(@expert_scale $kind($($w),+), $xs);
            gemm_nts!($mode, $nx, false, $nt_live, |nt, j0| {
                for i in 0..4usize {
                    ::cuda_device::thread::__unroll_config::<0>();
                    let n = 8 * nt + 2 * $t + (i & 1);
                    if n < $len {
                        let r = $row0 + $g + 8 * (i / 2);
                        // SAFETY: n < GEMM_BN bounds the shared read; the
                        // slot is below n_slots and r below rows, so the
                        // store is inside y (launch contract), and one
                        // lane of one block writes each (slot, row).
                        unsafe {
                            let slot = *$slot_sh.add(n) as usize;
                            *$y.get_unchecked_mut(slot * $rows_n + r) =
                                $crate::gemm::f32tile::ProductCell::of(
                                    w32!(@scale $kind($($w),+), ws, $acc[j0 + i]),
                                );
                        }
                    }
                }
            });
        }
    }};
    (@refused [swiglu($($sw:tt)+)], $slot:expr, $r:expr, $rows_n:expr) => {{
        let _ = $rows_n;
        swiglu32!(@refused [$($sw)+], $slot, $r);
    }};
    (@refused $y:ident, $slot:expr, $r:expr, $rows_n:expr) => {{
        // SAFETY: slot < n_slots and r < rows put the output below
        // n_slots · rows <= y.len() (launch contract). No tile lists a
        // refused slot, so no other block writes it, and each (slot, row) is
        // one thread's.
        unsafe {
            *$y.get_unchecked_mut($slot * $rows_n + $r) =
                $crate::gemm::f32tile::ProductCell::of(f32::NAN)
        };
    }};
}

/// The weights of a 32-value GEMM: a stack of `n_experts · rows_per_expert`
/// rows of K = `act.k()` values in one of the six layouts (module doc).
#[derive(Clone, Copy)]
pub enum Gemm32Weight<'a> {
    /// Q8_0 in the q8f32 planes: `qs` rows × K/4 words, `d` rows × K/32
    /// f16 scale bits. `qs` must be 16-byte aligned; slab launches also
    /// require `d` to be 16-byte aligned.
    Q8_0Plane {
        qs: &'a DeviceTensor<u32>,
        d: &'a DeviceTensor<u16>,
    },
    /// The file's `block_q8_0` stream (34 bytes per 32 values, rows
    /// back to back) as u32 words, zero-padded at its end to a whole number
    /// of words per row, as the K-quant stacks are.
    Q8_0File(&'a DeviceTensor<u32>),
    /// The file's `block_q5_1` stream (24 bytes per 32 values): six words
    /// per block, `6 · K/32` per row.
    Q5_1File(&'a DeviceTensor<u32>),
    /// The file's `block_iq4_nl` stream (18 bytes per 32 values: f16 `d`,
    /// then 16 code bytes whose nibbles index `kvalues_iq4nl`) as u32 words,
    /// zero-padded at its end to a whole number of words per row, as the
    /// K-quant stacks are.
    Iq4NlFile(&'a DeviceTensor<u32>),
    /// The resident packed Q5_0 (`q5.rs`, `pack_q5_0`, the tensor
    /// `enqueue_gemv_q5_0` reads): per row `q5_0_row_words(K/32)` words, the
    /// file's `qs` and `qh` in planes and its f16 `d` two to a word.
    Q5_0Packed(&'a DeviceTensor<u32>),
    /// NVFP4 in the resident planes of `mxfp4.rs`'s `repack` (the planes
    /// the dense gemv and the routed `_sel` pair read): `qs` rows × K/8
    /// words, `e` rows × K/64, and one F32 tensor scale an expert of the
    /// route's stack (a dense tensor through a one-expert table holds one),
    /// applied once at the store. K must be a multiple of 64.
    Nvfp4Planes(Nvfp4Stack<'a>),
}

/// [`Gemm32Kernels::enqueue_gemm32`]'s arguments: `w` the stack, `route`
/// the table its last fill left over the stack's experts, `input` each
/// slot's activation column, `y` `n_slots · rows_per_expert` cells (f32, or
/// BF16 bits for [`Gemm32Kernels::enqueue_gemm32_bf16`]), slot-major
/// (`y[s · rows + r]`).
pub struct Gemm32Args<'a, Y = f32> {
    pub w: Gemm32Weight<'a>,
    pub rows_per_expert: usize,
    pub act: &'a GemmAct32,
    pub route: &'a GemmRoute,
    pub input: GemmInput,
    pub y: &'a mut DeviceBuffer<Y>,
}

/// [`Gemm32Kernels::enqueue_gate_up_sel32`]'s arguments: a routed block's
/// gate and up stacks `w` of `rows_per_expert` rows each over the activations
/// `x` (read per `input`) through `route`'s last fill, the slots' places
/// `sel` (those below `n_card` are the card's), the products' scratch `g` and
/// `u` (`Gemm32Args::y`'s layout, empty for native fused gate/up), and the
/// down's activations `act`.
pub struct GateUp32Args<'a> {
    pub w: [Gemm32Weight<'a>; 2],
    pub rows_per_expert: usize,
    pub x: &'a GemmAct32,
    pub route: &'a GemmRoute,
    pub input: GemmInput,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
    pub g: &'a mut DeviceBuffer<f32>,
    pub u: &'a mut DeviceBuffer<f32>,
    pub act: &'a mut GemmAct32,
}

impl GemmRoute {
    /// The slot count of the last enqueued fill; `None` before one. A caller
    /// that owns a table for a known count holds its launches to it.
    #[must_use]
    pub fn filled(&self) -> Option<usize> {
        self.filled
    }
}

/// A launch's checked shape, as the entries take it.
struct Gemm32Launch {
    grid: u32,
    n_experts: u32,
    rows: u32,
    kb: u32,
    steps: u32,
    act_cols: u32,
    n_slots: u32,
    max_tiles: u32,
    slot_div: u32,
    row_tiles: u32,
}

impl Gemm32Launch {
    /// Check the route, weight, activation and launch arguments, then room
    /// in `y` for every slot's rows.
    fn check<Y>(what: &'static str, a: &Gemm32Args<'_, Y>) -> Result<Gemm32Launch, GpuError> {
        let l = Self::operands(what, a.w, a.rows_per_expert, a.act, a.route, a.input)?;
        let n_slots = l.n_slots as usize;
        let rows = a.rows_per_expert;
        if a.y.len() < n_slots * rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {n_slots}*{rows}",
                    a.y.len()
                ),
            ));
        }
        Ok(l)
    }

    /// Check the weight, activations and route independently of an output buffer.
    fn operands(
        what: &'static str,
        w: Gemm32Weight<'_>,
        rows: usize,
        act: &GemmAct32,
        route: &GemmRoute,
        input: GemmInput,
    ) -> Result<Gemm32Launch, GpuError> {
        let n_slots = route
            .filled
            .ok_or_else(|| GpuError::state(what, "a filled route table (enqueue_route first)"))?;
        if rows == 0 || !rows.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("rows_per_expert must be a positive multiple of 16, got {rows}"),
            ));
        }
        let n_rows = route.n_experts * rows;
        let (k, kb) = (act.k(), act.blocks());
        let stack_rows = match w {
            Gemm32Weight::Q8_0Plane { qs, .. }
            | Gemm32Weight::Q8_0File(qs)
            | Gemm32Weight::Q5_1File(qs)
            | Gemm32Weight::Iq4NlFile(qs)
            | Gemm32Weight::Q5_0Packed(qs)
            | Gemm32Weight::Nvfp4Planes(Nvfp4Stack { qs, .. }) => qs.rows(),
        };
        if stack_rows != n_rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "the stack has {stack_rows} rows, the route {} experts of {rows} rows",
                    route.n_experts
                ),
            ));
        }
        match w {
            Gemm32Weight::Q8_0Plane { qs, d } => {
                if d.rows() != n_rows
                    || qs.cols() != k / 4
                    || d.cols() != kb
                    || qs.buf().len() < n_rows * k / 4
                    || d.buf().len() < n_rows * kb
                {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "Q8_0 planes at K = {k}: qs {n_rows} × {} words and d {n_rows} × \
                             {kb} scales, got qs {} × {} and d {} × {}",
                            k / 4,
                            qs.rows(),
                            qs.cols(),
                            d.rows(),
                            d.cols()
                        ),
                    ));
                }
            }
            Gemm32Weight::Q8_0File(w) => {
                let bytes = n_rows * Q8_0_BLOCK_BYTES * kb;
                let words = bytes.div_ceil(4).div_ceil(n_rows);
                if w.cols() != words || 4 * w.buf().len() < bytes {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "Q8_0 rows at K = {k} are {words} words, got {} words over {} bytes",
                            w.cols(),
                            4 * w.buf().len()
                        ),
                    ));
                }
            }
            Gemm32Weight::Q5_1File(w) => {
                if w.cols() != Q5_1_BLOCK_WORDS * kb
                    || w.buf().len() < n_rows * Q5_1_BLOCK_WORDS * kb
                {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "Q5_1 rows at K = {k} are {} words, got {} words over {} words",
                            Q5_1_BLOCK_WORDS * kb,
                            w.cols(),
                            w.buf().len()
                        ),
                    ));
                }
            }
            Gemm32Weight::Iq4NlFile(w) => {
                let bytes = n_rows * IQ4_NL_BLOCK_BYTES * kb;
                let words = bytes.div_ceil(4).div_ceil(n_rows);
                if w.cols() != words || 4 * w.buf().len() < bytes {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "IQ4_NL rows at K = {k} are {words} words, got {} words over {} \
                             bytes",
                            w.cols(),
                            4 * w.buf().len()
                        ),
                    ));
                }
            }
            Gemm32Weight::Q5_0Packed(w) => {
                let words = q5_0_row_words(kb);
                if w.cols() != words || w.buf().len() < n_rows * words {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "packed Q5_0 rows at K = {k} are {words} words, got {} words over {} \
                             words",
                            w.cols(),
                            w.buf().len()
                        ),
                    ));
                }
            }
            Gemm32Weight::Nvfp4Planes(planes) => {
                planes.experts(what, k, rows)?;
            }
        }
        if !act.q8_required && !matches!(w, Gemm32Weight::Nvfp4Planes(_)) {
            return Err(GpuError::shape(
                what,
                "a Q8 consumer requires Q8 activation planes",
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
        if act.filled() < n_slots / slot_div {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots read {} activation columns, the last quantizer launch \
                     wrote {}",
                    n_slots / slot_div,
                    act.filled()
                ),
            ));
        }
        let max_tiles = gemm_max_tiles(n_slots, route.n_experts);
        let row_tiles = rows.div_ceil(GEMM_BM);
        Ok(Gemm32Launch {
            grid: launch_u32(what, "grid", row_tiles * max_tiles)?,
            n_experts: launch_u32(what, "n_experts", route.n_experts)?,
            rows: launch_u32(what, "rows_per_expert", rows)?,
            kb: launch_u32(what, "blocks", kb)?,
            steps: launch_u32(what, "steps", act.steps())?,
            act_cols: launch_u32(what, "act_cols", act.cols())?,
            n_slots: launch_u32(what, "n_slots", n_slots)?,
            max_tiles: launch_u32(what, "max_tiles", max_tiles)?,
            slot_div: launch_u32(what, "slot_div", slot_div)?,
            row_tiles: launch_u32(what, "row_tiles", row_tiles)?,
        })
    }
}

/// One 32-value GEMM entry's launch over the weight buffers `wbuf`, the
/// activations, the route and the output, at the checked shape `g`.
macro_rules! launch_gemm32 {
    (
        $k:ident, $stream:ident, $cfg:ident, $g:ident, ($act:ident, $route:ident, $y:expr),
        $prepare:ident, $entry:ident, $($wbuf:expr),+
    ) => {{
        let prep = $k.module.$prepare($cfg)?;
        $k.module.$entry(
            $stream,
            &prep,
            $($wbuf,)+
            &$act.q,
            &$act.d,
            &$act.s,
            &$route.cols,
            &$route.tiles,
            &$route.n_tiles,
            $g.n_experts,
            $g.rows,
            $g.kb,
            $g.steps,
            $g.act_cols,
            $g.n_slots,
            $g.max_tiles,
            $g.slot_div,
            $g.row_tiles,
            $y,
        )?;
    }};
}

impl Gemm32Kernels {
    /// The slab entry's resident-block wave on this card, read at load.
    pub fn q8_0p_wave(&self) -> u32 {
        self.q8_0p_wave
    }

    /// Whether a Q8_0-plane launch of `grid` blocks over `n_slots` slots of
    /// `n_experts` experts takes `gemm_q8_0p_slab`: a grid of one wave of it
    /// on this card or more (read at load) whose slots pass one tile an
    /// expert. Its reads cost a few instructions a step, which pay where
    /// several tiles share each weight slab across a full card; a grid
    /// inside one wave, or one tile an expert (each slab read once, from
    /// memory), takes `gemm_q8_0p`.
    pub fn q8_0p_slab(&self, grid: u32, n_slots: u32, n_experts: u32) -> bool {
        grid >= self.q8_0p_wave() && u64::from(n_slots) > GEMM_BN as u64 * u64::from(n_experts)
    }

    /// Whether a Q8_0-plane launch of `g` takes the slab entry
    /// ([`Gemm32Kernels::q8_0p_slab`]), its code plane and, for the slab, its
    /// scale plane checked 16-byte aligned as `what`'s.
    fn q8_0p_entry(
        &self,
        what: &'static str,
        (qs, d): (&DeviceTensor<u32>, &DeviceTensor<u16>),
        g: &Gemm32Launch,
    ) -> Result<bool, GpuError> {
        use cuda_device::vector::U32x4;
        crate::tensor::check_alignment::<U32x4, _>(what, "Q8_0 code plane", qs.buf())?;
        let slab = self.q8_0p_slab(g.grid, g.n_slots, g.n_experts);
        if slab {
            crate::tensor::check_alignment::<U32x4, _>(what, "Q8_0 scale plane", d.buf())?;
        }
        Ok(slab)
    }

    /// Enqueue `y[s][r] = Σ_k w[e(s)][r][k] · x[c(s)][k]` for every slot of
    /// the last fill of the route table, in the module doc's numeric
    /// contract ([`Gemm32Args`] names each operand). Asynchronous,
    /// allocation-free, capturable. A slot the route refused gets NaN in
    /// every row; a slot a remapped route left to the host
    /// ([`Gemm32Kernels::enqueue_route_remap`]) is not written.
    pub fn enqueue_gemm32(&self, stream: &CudaStream, a: Gemm32Args<'_>) -> Result<(), GpuError> {
        let g = Gemm32Launch::check("Gemm32Kernels::enqueue_gemm32", &a)?;
        let Gemm32Args {
            w, act, route, y, ..
        } = a;
        let cfg = LaunchConfig1D::new(g.grid, GEMM32_THREADS_U32, 0);
        macro_rules! launch {
            ($prepare:ident, $entry:ident, $($wbuf:expr),+) => {
                launch_gemm32!(self, stream, cfg, g, (act, route, y), $prepare, $entry, $($wbuf),+)
            };
        }
        match w {
            Gemm32Weight::Q8_0Plane { qs, d } => {
                if self.q8_0p_entry("Gemm32Kernels::enqueue_gemm32", (qs, d), &g)? {
                    launch!(prepare_gemm_q8_0p_slab, gemm_q8_0p_slab, qs.buf(), d.buf());
                } else {
                    launch!(prepare_gemm_q8_0p, gemm_q8_0p, qs.buf(), d.buf());
                }
            }
            Gemm32Weight::Q8_0File(w) => launch!(prepare_gemm_q8_0f, gemm_q8_0f, w.buf()),
            Gemm32Weight::Q5_1File(w) => launch!(prepare_gemm_q5_1, gemm_q5_1, w.buf()),
            Gemm32Weight::Iq4NlFile(w) => launch!(prepare_gemm_iq4nl, gemm_iq4nl, w.buf()),
            Gemm32Weight::Q5_0Packed(w) => {
                let s_words = launch_u32(
                    "Gemm32Kernels::enqueue_gemm32",
                    "s_words",
                    (g.kb as usize).div_ceil(2),
                )?;
                launch!(prepare_gemm_q5_0p, gemm_q5_0p, w.buf(), s_words);
            }
            Gemm32Weight::Nvfp4Planes(Nvfp4Stack { qs, e, scales }) => {
                #[cfg(bloomery_nvfp4_mma)]
                if act.native.is_some() {
                    return self.native.product(
                        stream,
                        super::nvfp4mma::ProductArgs {
                            w: Nvfp4Stack { qs, e, scales },
                            act,
                            route,
                            rows: g.rows,
                            slots: g.n_slots,
                            max_tiles: g.max_tiles,
                            slot_div: g.slot_div,
                            y: super::ProductOut::F32(y),
                        },
                    );
                }
                launch!(
                    prepare_gemm_nvfp4p,
                    gemm_nvfp4p,
                    qs.buf(),
                    e.buf(),
                    scales.buf()
                );
            }
        }
        Ok(())
    }

    /// [`Gemm32Kernels::enqueue_gemm32`]'s products as BF16 bits, each value
    /// its F32 one rounded to nearest even (a value past the BF16 range an
    /// infinity), for every stack it reads: NVFP4 planes over the native FP4
    /// activation form or over the q8 blocks of 32, Q8_0 planes, and the
    /// file's Q8_0, Q5_1 and IQ4_NL blocks and the packed Q5_0 (a routed
    /// down's). Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemm32_bf16(
        &self,
        stream: &CudaStream,
        a: Gemm32Args<'_, u16>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_gemm32_bf16";
        let g = Gemm32Launch::check(what, &a)?;
        let Gemm32Args {
            w, act, route, y, ..
        } = a;
        let cfg = LaunchConfig1D::new(g.grid, GEMM32_THREADS_U32, 0);
        macro_rules! launch {
            ($prepare:ident, $entry:ident, $($wbuf:expr),+) => {
                launch_gemm32!(self, stream, cfg, g, (act, route, y), $prepare, $entry, $($wbuf),+)
            };
        }
        match w {
            Gemm32Weight::Q8_0Plane { qs, d } => {
                if self.q8_0p_entry(what, (qs, d), &g)? {
                    launch!(
                        prepare_gemm_q8_0p_slab_bf16,
                        gemm_q8_0p_slab_bf16,
                        qs.buf(),
                        d.buf()
                    );
                } else {
                    launch!(prepare_gemm_q8_0p_bf16, gemm_q8_0p_bf16, qs.buf(), d.buf());
                }
            }
            Gemm32Weight::Nvfp4Planes(Nvfp4Stack { qs, e, scales }) => {
                #[cfg(bloomery_nvfp4_mma)]
                if act.native.is_some() {
                    return self.native.product(
                        stream,
                        super::nvfp4mma::ProductArgs {
                            w: Nvfp4Stack { qs, e, scales },
                            act,
                            route,
                            rows: g.rows,
                            slots: g.n_slots,
                            max_tiles: g.max_tiles,
                            slot_div: g.slot_div,
                            y: super::ProductOut::Bf16(y),
                        },
                    );
                }
                launch!(
                    prepare_gemm_nvfp4p_bf16,
                    gemm_nvfp4p_bf16,
                    qs.buf(),
                    e.buf(),
                    scales.buf()
                );
            }
            Gemm32Weight::Q8_0File(w) => launch!(prepare_gemm_q8_0f_bf16, gemm_q8_0f_bf16, w.buf()),
            Gemm32Weight::Q5_1File(w) => launch!(prepare_gemm_q5_1_bf16, gemm_q5_1_bf16, w.buf()),
            Gemm32Weight::Iq4NlFile(w) => {
                launch!(prepare_gemm_iq4nl_bf16, gemm_iq4nl_bf16, w.buf())
            }
            Gemm32Weight::Q5_0Packed(w) => {
                let s_words = launch_u32(what, "s_words", (g.kb as usize).div_ceil(2))?;
                launch!(prepare_gemm_q5_0p_bf16, gemm_q5_0p_bf16, w.buf(), s_words);
            }
        }
        Ok(())
    }

    /// Enqueue `act = q8_1_32(silu(g) · u)` over `slots` of the last fill of
    /// `up.route`, `u` the up GEMM `up` (its `y` the up rows' scratch,
    /// `rows_per_expert` the activations' K): the bytes and faults of
    /// [`Gemm32Kernels::enqueue_gemm32`] into `up.y`, then
    /// [`Gemm32Kernels::enqueue_swiglu_quant32`] ([`SwigluSlots::All`]) or
    /// [`Gemm32Kernels::enqueue_swiglu_quant32_sel`] ([`SwigluSlots::Card`])
    /// over the route's slots. A stack and slot set with an entry of the
    /// SwiGLU epilogue (`swiglu32.rs`) runs as one launch that never writes
    /// `up.y` and never reads `u` back: NVFP4 planes on the 32-value walk
    /// over card slots (`gemm_nvfp4p_swiglu`) and Q8_0 planes over all
    /// (`gemm_q8_0p_swiglu_all`, or its slab walk where
    /// [`Gemm32Kernels::q8_0p_slab`] takes it); every other runs the two, as
    /// does an `act` in the native FP4 form. Asynchronous, allocation-free,
    /// capturable; `act`'s filled count is the route's slot count.
    pub fn enqueue_up_swiglu32(
        &self,
        stream: &CudaStream,
        up: Gemm32Args<'_>,
        g: &DeviceBuffer<f32>,
        slots: SwigluSlots<'_>,
        act: &mut GemmAct32,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_up_swiglu32";
        let l = Gemm32Launch::check(what, &up)?;
        let n_slots = l.n_slots as usize;
        let (oblocks, osteps) = act.swiglu_launch(what, n_slots, up.rows_per_expert, g)?;
        #[cfg(bloomery_nvfp4_mma)]
        let (x_native, act_native) = (up.act.native.is_some(), act.native.is_some());
        #[cfg(not(bloomery_nvfp4_mma))]
        let (x_native, act_native) = (false, false);
        let Gemm32Args {
            w,
            rows_per_expert,
            act: x,
            route,
            input,
            y: u,
        } = up;
        let cfg = LaunchConfig1D::new(l.grid, GEMM32_THREADS_U32, 0);
        // The epilogue entry's launch: the up GEMM's operands, the gate rows,
        // the card places where the entry reads them, the output planes and
        // the sink.
        macro_rules! launch {
            ($prepare:ident, $entry:ident, ($($wbuf:expr),+), ($($places:expr),*)) => {{
                let prep = self.module.$prepare(cfg)?;
                self.module.$entry(
                    stream, &prep, $($wbuf,)+ &x.q, &x.d, &x.s, &route.cols, &route.tiles,
                    &route.n_tiles, l.n_experts, l.rows, l.kb, l.steps, l.act_cols, l.n_slots,
                    l.max_tiles, l.slot_div, l.row_tiles, g, $($places,)* oblocks, osteps,
                    &mut act.q, &mut act.d, &mut act.s, fault,
                )?;
            }};
        }
        match (w, slots) {
            (Gemm32Weight::Nvfp4Planes(planes), SwigluSlots::Card { sel, n_card })
                if !x_native && !act_native =>
            {
                if sel.len() < n_slots {
                    return Err(GpuError::shape(
                        what,
                        format!("sel.len() {} < {n_slots} slots", sel.len()),
                    ));
                }
                let n_card = launch_u32(what, "n_card", n_card)?;
                launch!(
                    prepare_gemm_nvfp4p_swiglu,
                    gemm_nvfp4p_swiglu,
                    (planes.qs.buf(), planes.e.buf(), planes.scales.buf()),
                    (sel, n_card)
                );
            }
            (Gemm32Weight::Q8_0Plane { qs, d }, SwigluSlots::All) if !act_native => {
                if self.q8_0p_entry(what, (qs, d), &l)? {
                    launch!(
                        prepare_gemm_q8_0p_slab_swiglu_all,
                        gemm_q8_0p_slab_swiglu_all,
                        (qs.buf(), d.buf()),
                        ()
                    );
                } else {
                    launch!(
                        prepare_gemm_q8_0p_swiglu_all,
                        gemm_q8_0p_swiglu_all,
                        (qs.buf(), d.buf()),
                        ()
                    );
                }
            }
            _ => {
                self.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w,
                        rows_per_expert,
                        act: x,
                        route,
                        input,
                        y: &mut *u,
                    },
                )?;
                return match slots {
                    SwigluSlots::All => {
                        self.enqueue_swiglu_quant32(stream, g, u, n_slots, act, fault)
                    }
                    SwigluSlots::Card { sel, n_card } => self
                        .enqueue_swiglu_quant32_sel(stream, g, u, sel, n_card, n_slots, act, fault),
                };
            }
        }
        act.fill(n_slots);
        Ok(())
    }

    /// Enqueue `act = q8_1_32(silu(gate · x) · (up · x))` over the slots of
    /// the last fill of `route` ([`GateUp32Args`]): the gate GEMM
    /// ([`Gemm32Kernels::enqueue_gemm32`]) into `g`, then
    /// [`Gemm32Kernels::enqueue_up_swiglu32`] — or, on a native FP4 build with
    /// both stacks in NVFP4 planes, `x` in the native form and `act` made for
    /// it ([`GemmAct32::new_gate_up_output`]), one launch that holds each
    /// tile's gate and up rows in the block and stores, for every slot the
    /// route lists, the FP4 of the Q8 codes the split pair quantizes
    /// (`g`, `u` and `sel` untouched), each column scaled by a power of two
    /// where the pair's FP4 scale is its blocks' largest Q8 scale · 127/6.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up_sel32(
        &self,
        stream: &CudaStream,
        a: GateUp32Args<'_>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let GateUp32Args {
            w: [gate, up],
            rows_per_expert,
            x,
            route,
            input,
            sel,
            n_card,
            g,
            u,
            act,
        } = a;
        #[cfg(bloomery_nvfp4_mma)]
        if let (Gemm32Weight::Nvfp4Planes(gp), Gemm32Weight::Nvfp4Planes(upp)) = (gate, up)
            && x.native.is_some()
            && act
                .native
                .as_ref()
                .is_some_and(super::nvfp4mma::Act::takes_gate_up)
        {
            const WHAT: &str = "Gemm32Kernels::enqueue_gate_up_sel32";
            let l = Gemm32Launch::operands(WHAT, gate, rows_per_expert, x, route, input)?;
            Gemm32Launch::operands(WHAT, up, rows_per_expert, x, route, input)?;
            return self.native.gate_up(
                stream,
                super::nvfp4mma::GateUpArgs {
                    w: [gp, upp],
                    x,
                    route,
                    rows: l.rows,
                    max_tiles: l.max_tiles,
                    slot_div: l.slot_div,
                    n_cols: l.n_slots as usize,
                    act,
                },
                fault,
            );
        }
        self.enqueue_gemm32(
            stream,
            Gemm32Args {
                w: gate,
                rows_per_expert,
                act: x,
                route,
                input,
                y: &mut *g,
            },
        )?;
        self.enqueue_up_swiglu32(
            stream,
            Gemm32Args {
                w: up,
                rows_per_expert,
                act: x,
                route,
                input,
                y: u,
            },
            g,
            SwigluSlots::Card { sel, n_card },
            act,
            fault,
        )
    }
}
