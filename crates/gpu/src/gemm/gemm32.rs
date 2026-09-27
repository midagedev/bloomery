//! The 32-value-block GEMM: Q8_0 and Q5_1 weights times [`GemmAct32`]
//! activations over a route table — one block per (tile, [`GEMM_BM`]-row
//! slab), as the K-quant family runs, for the weights whose scales come per
//! 32 values. Three entries, declared in `kernels32.rs`, differ only in how
//! the weight bytes reach shared memory:
//! - `gemm_q8_0p`: Q8_0 in the q8f32 planes (`qs` u32 words, eight per
//!   block, and `d` the blocks' f16 bits, `q8f32.rs`'s layout) — the dense
//!   projections, through a one-expert table;
//! - `gemm_q8_0f`: the file's 34-byte `block_q8_0` stream (f16 `d`, then 32
//!   int8 codes) as u32 words;
//! - `gemm_q5_1`: the file's 24-byte `block_q5_1` stream (f16 `d` and `m`,
//!   the high bits `qh`, the nibbles `qs[16]`) as u32 words.
//!
//! Numeric contract, every entry. Per 32-value block `b` of an output
//! `(slot, row)`: `isum` the exact i32 dot of the weight codes (Q8_0 int8;
//! Q5_1 the unsigned 5-bit code, 0..31) with the column's int8 codes, one
//! `mma.m16n8k32` (mainline MMQ's Q8_0 way); `d_w` (and `m_w`) the block's
//! f16 values, exact in f32; `d_a`, `s_a` the column's block scale and code
//! sum. The block enters the f32 accumulator as
//! `acc = fma(d_a, d_w·f32(isum), acc)` for Q8_0 and
//! `acc = fma(d_a, fma(m_w, f32(s_a), d_w·f32(isum)), acc)` for Q5_1 —
//! the value `d_w·d_a·isum + m_w·d_a·s_a` — from `acc = 0`, blocks in
//! increasing k. That order is a function of K alone: never of the grid,
//! the tile a slot lands in, its neighbours or the slot count, so an
//! output's bits depend on its weight row and its column only.
//!
//! Geometry. A block is [`GEMM_THREADS`] threads, eight warps of sixteen
//! rows over the tile's columns in n-tiles of eight (the K-quant family's
//! shape). The K axis is walked in [`GEMM32_STEP`]-value steps, two blocks
//! a step, the last step holding one block when K/32 is odd. Each step's
//! weights, activation codes and activation scales sit in one of two shared
//! stages: the activations and the plane's codes arrive by `cp.async`; the
//! plane's scales and every file layout's bytes are read into registers one
//! step ahead and written into the stage after the step's compute, decoded
//! to the staged form — per row sixteen code words then the two blocks' `d`
//! (and Q5_1's `m`) as f32 ([`W32_ROW`] words). One barrier a step: it
//! publishes step `h`'s stage and tells that every warp is done with step
//! `h − 1`'s, which the copies and stores for step `h + 1` then refill. Each
//! warp loads its `mma` A fragments with `ldmatrix.x4` from the staged rows
//! and each n-tile's B fragments of both blocks with one more.

use super::act32::GemmAct32;
use super::grouped::{GEMM_BM, GEMM_THREADS};
use super::kernels32::Gemm32Kernels;
use super::route::gemm_max_tiles;
use super::{GEMM_BN, GEMM32_STEP, GemmInput, GemmRoute};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

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
/// [`GEMM_THREADS`] as the block width a launch takes.
const GEMM32_THREADS_U32: u32 = GEMM_THREADS as u32;

// Byte sizes of one 32-value block as the file stores it (ggml's
// `block_q8_0` and `block_q5_1`), and a Q5_1 block in words.
pub(super) const Q8_0_BLOCK_BYTES: usize = 34;
pub(super) const Q5_1_BLOCK_WORDS: usize = 6;

const _: () = assert!(GEMM32_STEP == 64 && W32_ROW == 16 + 4 && B32_ROW >= 16);
const _: () = assert!((W32_ROW / 4) % 2 == 1 && (B32_ROW / 4) % 2 == 1);
const _: () = assert!(W32_ROW.is_multiple_of(4) && B32_ROW.is_multiple_of(4));
// The relay map: thread `tid` owns slab row `tid / 2`, block `tid % 2` of
// every step; the activation map: column `tid / 4`, 16-byte chunk (and
// scale word) `tid % 4`.
const _: () = assert!(GEMM_THREADS == 2 * GEMM_BM && GEMM_THREADS == 4 * GEMM_BN);
// The entries' launch contracts spell the block sizes out as literals.
const _: () = assert!(Q8_0_BLOCK_BYTES == 34 && Q5_1_BLOCK_WORDS == 6 && GEMM_THREADS == 256);
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

/// How the weight bytes of one (slab row, block) pair reach a stage, per
/// entry: `@copy` the `cp.async` part (the plane's codes), `@load` the
/// register part one step ahead, `@store` it decoded into the stage after
/// the step's compute. `$live` is whether the pair exists (its row inside
/// the slab's live rows, its block below K/32); `$n` is its block's index
/// in the stack's block stream (`row · K/32 + block`), `$dst` its staged
/// row's first word.
macro_rules! w32 {
    (@copy plane($qs:ident, $wd:ident), $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
        if $live {
            // SAFETY: block `n` is inside the stack (the launch contract
            // holds `8 · n_experts · rows · K/32` words), its eight words
            // start at word `8n`, 32-byte aligned; the destination is the
            // pair's eight words of its staged row, 16-byte aligned.
            unsafe {
                let src = $qs.as_ptr().add(8 * $n);
                let dst = $dst.add(8 * $pj);
                cp_async_cg_16(dst, src);
                cp_async_cg_16(dst.add(4), src.add(4));
            }
        }
    }};
    (@copy $kind:ident($w:ident), $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
        let _ = ($live, $n, $dst, $pj);
    }};
    (@load plane($qs:ident, $wd:ident), $live:expr, $n:expr) => {{
        if $live {
            // SAFETY: block `n` is inside the stack, whose scale plane holds
            // `n_experts · rows · K/32` entries (launch contract).
            unsafe { *$wd.get_unchecked($n) }
        } else {
            0u16
        }
    }};
    (@load q8f($w:ident), $live:expr, $n:expr) => {{
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
    (@load q51($w:ident), $live:expr, $n:expr) => {{
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
    (@store plane($qs:ident, $wd:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
        if $live {
            // SAFETY: word 16 + pj of the pair's staged row, which no warp
            // reads before the next barrier.
            unsafe { *$dst.add(16 + $pj) = half_bits_to_f32($raw).to_bits() };
        }
    }};
    (@store q8f($w:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
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
    (@store q51($w:ident), $raw:ident, $live:expr, $n:expr, $dst:expr, $pj:expr) => {{
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
/// bits and code sum bits — the contract's order per output.
macro_rules! epi32 {
    ($mins:literal, $isum:ident, $dw:ident, $mw:ident, $da:expr, $sa:expr, $acc:ident, $nt:expr) => {{
        let da: [u32; 2] = $da;
        let sa: [u32; 2] = $sa;
        for i in 0..4usize {
            ::cuda_device::thread::__unroll_config::<0>();
            let tv = mul_rn_f32($dw[i / 2], $isum[i] as f32);
            let tv = if $mins {
                fma_rn_f32($mw[i / 2], sa[i & 1] as i32 as f32, tv)
            } else {
                tv
            };
            $acc[4 * $nt + i] = fma_rn_f32(f32::from_bits(da[i & 1]), tv, $acc[4 * $nt + i]);
        }
    }};
}

/// The 32-value GEMM block: one tile's columns against one 128-row slab of
/// that tile's expert (module doc). A macro, not a function, for the
/// K-quant block's reason (`grouped.rs`, `gemm_block`): the three entries
/// share the walk and differ only in the weight relay `w32` names, and a
/// function boundary would move the accumulators out of registers.
///
/// `wt` names the relay and the entry's weight parameters, `mins` (a
/// literal) whether the type has a min term (Q5_1). `params` is the rest of
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
        params: (
            $q:ident, $d:ident, $s:ident, $cols:ident, $tiles:ident, $n_tiles:ident,
            $n_experts:ident, $rows:ident, $kb:ident, $steps:ident, $act_cols:ident,
            $n_slots:ident, $max_tiles:ident, $slot_div:ident, $row_tiles:ident, $y:ident $(,)?
        ) $(,)?
    ) => {{
        static mut WT: SharedArray<u32, { 2 * W32_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut BT: SharedArray<u32, { 2 * B32_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut DT: SharedArray<u32, { 2 * D32_STAGE_W }, 16> = SharedArray::UNINIT;
        static mut SLOT: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        static mut ACOL: SharedArray<u32, GEMM_BN> = SharedArray::UNINIT;
        // SAFETY: each `static mut` is this block's own shared allocation,
        // reached raw; the block walk bounds every index and orders every
        // cross-thread read behind a barrier.
        let (wt, bt, dt, slot_sh, acol_sh) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut WT),
                SharedArray::as_raw_mut_ptr(&raw mut BT),
                SharedArray::as_raw_mut_ptr(&raw mut DT),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT),
                SharedArray::as_raw_mut_ptr(&raw mut ACOL),
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
        // Each lane's `ldmatrix` rows: A row `lane % 16` of the warp's slab
        // rows at 16-byte half `lane / 16`; B column `lane % 8` of an n-tile
        // at 16-byte chunk `lane / 8` (blocks 0 and 1, low and high half).
        let a_off = (16 * wid + (lane % 16)) * W32_ROW + 4 * (lane / 16);
        let b_off = (lane % 8) * B32_ROW + 4 * (lane / 8);
        let r_off = (16 * wid + g) * W32_ROW + 16;

        // Step 0 into stage 0.
        stage32_act!(0usize, 0usize, $mins, tid, ncols, col_q, col_d, $q, $d, $s, bt, dt, acol_sh);
        {
            let live = prow_live && pj < kb;
            // SAFETY: the pair's staged row of stage 0, inside WT.
            let dst = unsafe { wt.add(pr * W32_ROW) };
            w32!(@copy $kind($($w),+), live, pn0, dst, pj);
            let raw = w32!(@load $kind($($w),+), live, pn0);
            w32!(@store $kind($($w),+), raw, live, pn0, dst, pj);
        }
        // SAFETY: commits this thread's copies above as one group.
        unsafe { cp_async_commit_group() };

        let mut acc = [0.0f32; 4 * GEMM_NT];
        let mut h = 0usize;
        while h < steps {
            let st = h & 1;
            let nx = h + 1 < steps;
            // Step h + 1's pair, loaded now, stored after this step's compute.
            let nlive = nx && prow_live && 2 * (h + 1) + pj < kb;
            let nn = pn0 + 2 * (h + 1);
            let raw = w32!(@load $kind($($w),+), nlive, nn);
            // SAFETY: one group per step (possibly empty) and none newer, so
            // waiting for all leaves step h's copies complete; the barrier
            // then publishes them and every relay store of step h, and tells
            // that every warp is done with stage st ^ 1.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
            // SAFETY: the pair's staged row of stage st ^ 1, inside WT.
            let ndst = unsafe { wt.add((st ^ 1) * W32_STAGE_W + pr * W32_ROW) };
            if nx {
                stage32_act!(h + 1, st ^ 1, $mins, tid, ncols, col_q, col_d, $q, $d, $s, bt, dt,
                    acol_sh);
                w32!(@copy $kind($($w),+), nlive, nn, ndst, pj);
            }
            // SAFETY: one group per step, empty on the last.
            unsafe { cp_async_commit_group() };

            if active {
                // Block 1 of this step exists (warp-uniform).
                let two = 2 * h + 1 < kb;
                // SAFETY: stage `st` is inside each shared array, and every
                // offset below stays inside its stage: rows below GEMM_BM of
                // W32_ROW words, columns below GEMM_BN of B32_ROW and D32_ROW.
                let (wst, bst, dst) = unsafe {
                    (
                        wt.add(st * W32_STAGE_W),
                        bt.add(st * B32_STAGE_W),
                        dt.add(st * D32_STAGE_W),
                    )
                };
                // SAFETY: the whole warp issues each `ldmatrix` (the branches
                // around them are warp-uniform) with 16-byte-aligned rows of
                // this stage, published by the barrier above.
                let (a0, a1) = unsafe {
                    let p = wst.add(a_off);
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
                    let p0 = wst.add(r_off);
                    let p1 = p0.add(8 * W32_ROW);
                    (
                        [f32::from_bits(*p0), f32::from_bits(*p1)],
                        [f32::from_bits(*p0.add(1)), f32::from_bits(*p1.add(1))],
                        [f32::from_bits(*p0.add(2)), f32::from_bits(*p1.add(2))],
                        [f32::from_bits(*p0.add(3)), f32::from_bits(*p1.add(3))],
                    )
                };
                for nt in 0..GEMM_NT {
                    ::cuda_device::thread::__unroll_config::<0>();
                    if nt < nt_live {
                        let n0 = 8 * nt + 2 * t;
                        // SAFETY: column 8·nt + lane % 8 < ncols of this stage
                        // (the whole warp issues it: nt < nt_live is uniform);
                        // columns n0, n0 + 1 < GEMM_BN: two 16-byte reads of
                        // this stage's scales. All published above.
                        let (bf, x0, x1) = unsafe {
                            let bf = ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                                bst.add(8 * nt * B32_ROW + b_off).cast_const().cast::<u8>(),
                            ));
                            let x0 = (*(dst.add(D32_ROW * n0) as *const U32x4)).0;
                            let x1 = (*(dst.add(D32_ROW * (n0 + 1)) as *const U32x4)).0;
                            (bf, x0, x1)
                        };
                        // SAFETY: every lane of the warp reaches this `mma.sync`
                        // (the branches around it are uniform) with its
                        // fragments in the instruction's layout.
                        let i0 = unsafe { mma_m16n8k32_s32_s8([0; 4], a0, [bf[0], bf[1]]) };
                        epi32!($mins, i0, dw0, mw0, [x0[0], x1[0]], [x0[2], x1[2]], acc, nt);
                        if two {
                            // SAFETY: as above; `two` is warp-uniform.
                            let i1 = unsafe { mma_m16n8k32_s32_s8([0; 4], a1, [bf[2], bf[3]]) };
                            epi32!($mins, i1, dw1, mw1, [x0[1], x1[1]], [x0[3], x1[3]], acc, nt);
                        }
                    }
                }
            }

            if nx {
                w32!(@store $kind($($w),+), raw, nlive, nn, ndst, pj);
            }
            h += 1;
        }

        if active {
            for nt in 0..GEMM_NT {
                ::cuda_device::thread::__unroll_config::<0>();
                if nt < nt_live {
                    for i in 0..4usize {
                        ::cuda_device::thread::__unroll_config::<0>();
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
                    }
                }
            }
        }
    }};
}

/// The weights of a 32-value GEMM: a stack of `n_experts · rows_per_expert`
/// rows of K = `act.k()` values in one of the three layouts (module doc).
#[derive(Clone, Copy)]
pub enum Gemm32Weight<'a> {
    /// Q8_0 in the q8f32 planes: `qs` rows × K/4 words, `d` rows × K/32
    /// f16 scale bits.
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
}

/// [`Gemm32Kernels::enqueue_gemm32`]'s arguments: `w` the stack, `route`
/// the table its last fill left over the stack's experts, `input` each
/// slot's activation column, `y` `n_slots · rows_per_expert` f32,
/// slot-major (`y[s · rows + r]`).
pub struct Gemm32Args<'a> {
    pub w: Gemm32Weight<'a>,
    pub rows_per_expert: usize,
    pub act: &'a GemmAct32,
    pub route: &'a GemmRoute,
    pub input: GemmInput,
    pub y: &'a mut DeviceBuffer<f32>,
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
    /// Err unless `a` is launchable, in this order: a filled route table,
    /// rows a positive multiple of 16 that make the table's experts, the
    /// stack's layout at the activations' K, whole tokens of `top_k`, room in
    /// `act` and `y` for every slot; then every launch argument in `u32`.
    fn check(what: &'static str, a: &Gemm32Args<'_>) -> Result<Gemm32Launch, GpuError> {
        let n_slots = a
            .route
            .filled
            .ok_or_else(|| GpuError::state(what, "a filled route table (enqueue_route first)"))?;
        let rows = a.rows_per_expert;
        if rows == 0 || !rows.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("rows_per_expert must be a positive multiple of 16, got {rows}"),
            ));
        }
        let n_rows = a.route.n_experts * rows;
        let (k, kb) = (a.act.k(), a.act.blocks());
        let stack_rows = match a.w {
            Gemm32Weight::Q8_0Plane { qs, .. }
            | Gemm32Weight::Q8_0File(qs)
            | Gemm32Weight::Q5_1File(qs) => qs.rows(),
        };
        if stack_rows != n_rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "the stack has {stack_rows} rows, the route {} experts of {rows} rows",
                    a.route.n_experts
                ),
            ));
        }
        match a.w {
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
        }
        let slot_div = match a.input {
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
        if a.act.cols() < n_slots / slot_div {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots read {} activation columns, act holds {}",
                    n_slots / slot_div,
                    a.act.cols()
                ),
            ));
        }
        if a.y.len() < n_slots * rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {n_slots}*{rows}",
                    a.y.len()
                ),
            ));
        }
        let max_tiles = gemm_max_tiles(n_slots, a.route.n_experts);
        let row_tiles = rows.div_ceil(GEMM_BM);
        Ok(Gemm32Launch {
            grid: launch_u32(what, "grid", row_tiles * max_tiles)?,
            n_experts: launch_u32(what, "n_experts", a.route.n_experts)?,
            rows: launch_u32(what, "rows_per_expert", rows)?,
            kb: launch_u32(what, "blocks", kb)?,
            steps: launch_u32(what, "steps", a.act.steps())?,
            act_cols: launch_u32(what, "act_cols", a.act.cols())?,
            n_slots: launch_u32(what, "n_slots", n_slots)?,
            max_tiles: launch_u32(what, "max_tiles", max_tiles)?,
            slot_div: launch_u32(what, "slot_div", slot_div)?,
            row_tiles: launch_u32(what, "row_tiles", row_tiles)?,
        })
    }
}

impl Gemm32Kernels {
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
            ($prepare:ident, $entry:ident, $($wbuf:expr),+) => {{
                let prep = self.module.$prepare(cfg)?;
                self.module.$entry(
                    stream,
                    &prep,
                    $($wbuf,)+
                    &act.q,
                    &act.d,
                    &act.s,
                    &route.cols,
                    &route.tiles,
                    &route.n_tiles,
                    g.n_experts,
                    g.rows,
                    g.kb,
                    g.steps,
                    g.act_cols,
                    g.n_slots,
                    g.max_tiles,
                    g.slot_div,
                    g.row_tiles,
                    y,
                )?;
            }};
        }
        match w {
            Gemm32Weight::Q8_0Plane { qs, d } => {
                launch!(prepare_gemm_q8_0p, gemm_q8_0p, qs.buf(), d.buf());
            }
            Gemm32Weight::Q8_0File(w) => launch!(prepare_gemm_q8_0f, gemm_q8_0f, w.buf()),
            Gemm32Weight::Q5_1File(w) => launch!(prepare_gemm_q5_1, gemm_q5_1, w.buf()),
        }
        Ok(())
    }
}
