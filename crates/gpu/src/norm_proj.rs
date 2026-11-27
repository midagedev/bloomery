//! The input norms and the projections of one attention input in one launch: up to three
//! input rows, each RMS-normed with its gain, and eight one-row projections of them — five over
//! Q8_0 planes and three over F32 rows — each output bit for bit what `rms_norm` followed by the
//! site's own gemv (`q8_0_gemv`, `f32_gemv`) writes.
//!
//! The sites, in block order: the Q8_0 `q`, `k`, `v`, `rel` and `gate` and the F32 `sink`, `kt`
//! and `vt` (a site of no row is absent). `q`, `rel`, `gate` and `sink` read input 0, `k` and
//! `kt` input `ik`, `v` and `vt` input `iv` (`ik`, `iv` below 3). Every site's rows are a multiple
//! of [`ROWS`], so a block's rows are one site's.
//!
//! A block of [`THREADS`] threads first norms its site's input into shared memory as `rms_norm`
//! does: thread `tid` sums the squares of values `tid, tid + 256, …` (`acc += v · v` in that
//! order), the warp butterfly, the fixed eight-warp tree, `rms_scale_checked`, and each value
//! `(scale · gain) · x`. Then each warp walks its rows with the site's own gemv lane walk over the
//! normed row and the warp butterfly, and lane 0 writes the sum: one row an F32 site's block
//! ([`f32_lane_partial_1col`], [`ROWS`] rows), two a Q8_0 site's ([`q8_0_pair_partials`] at one
//! token, rows `r` and `r + ROWS`, [`STEP_Q8_ROWS`] rows, the last block of a site the rows its
//! count leaves). Every block of a site norms the same input in the same order, so every block's
//! normed row is `rms_norm`'s bits, and every output the site's gemv's.
//!
//! Two entries:
//!
//! - `norm_proj_q8_f32`: one token, the block above, four blocks an SM: the launch's blocks (about
//!   550 on the Mini) are one wave of the card.
//! - `norm_proj_q8_f32_cols`: `m` (1..=[`MAX_COLS`]) tokens of each input, token after token, in
//!   groups of `cols` tokens (`cols · k` at most [`GROUP_VALUES`], `cols` at most
//!   [`GROUP_COLS`]): a block is one group of [`COLS_ROWS`] rows of a Q8_0 site or [`ROWS`] of an
//!   F32 site (a site's last block the rows its count leaves), the groups of a row block
//!   neighbours in the grid, so a row block's weights come from memory once and from L2 for its
//!   other groups. The block copies its group's inputs into a shared block of `cols` rows, norms
//!   each token there as the one-token entry norms its row (the same sums over the same values,
//!   then each value in place), and each warp walks its rows (four of a Q8_0 site, two at a time;
//!   one of an F32 site) once for the group's tokens with the gemv's lane walk on each row and token
//!   ([`q8_0_lane_partials_mcol`], [`f32_lane_partials`]: each token's multiply-adds the one-token
//!   walk's, term for term and in order) and the butterfly a token ([`gemv_lane_sums`]); token
//!   `c`'s output of row `r` lands at `y[c · rows + r]`. So each token's outputs are the one-token
//!   entry's on that token, bit for bit.
//!
//! No silent failure: a sum of squares that is not finite and positive raises
//! [`crate::fault::FaultSite`]'s norm site through `rms_scale_checked` and the normed row, so
//! every output of that input, is NaN.

use crate::elem::{RMS_THREADS, RMS_WARPS, rms_scale_checked, rms_warp_tree};
use crate::fault::FaultSink;
use crate::q8f32::{
    Q8_0Bytes, Q8_STEP, block_code, f32_lane_partial_1col, f32_lane_partials, gemv_lane_sums,
    q8_0_lane_partials_mcol, word_dot,
};
use crate::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::vector::{F32x4, as_vectors};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block: `rms_norm`'s block, eight warps.
pub const THREADS: usize = 256;
const WARPS: usize = THREADS / 32;
/// Rows a block of an F32 site, and the rows of one pass of a Q8_0 block: one a warp.
pub const ROWS: usize = WARPS;
/// Rows a `norm_proj_q8_f32` block of a Q8_0 site walks: two a warp, rows `r` and `r + ROWS` of
/// its warp's pair ([`q8_0_pair_partials`]), so a block's norm is paid once for sixteen rows.
pub const STEP_Q8_ROWS: usize = 2 * ROWS;
/// The widest input row a launch norms (its shared row).
pub const MAX_K: usize = 3072;
/// The sites a launch projects: five Q8_0, three F32.
pub const SITES: usize = 8;
/// The most tokens the `_cols` entry norms and projects in one launch: a verify's rows.
pub const MAX_COLS: usize = 8;
/// The most tokens a `_cols` block norms and projects: its group.
pub const GROUP_COLS: usize = 4;
/// A `_cols` block's shared rows: its group's `cols · k` normed values at most.
pub const GROUP_VALUES: usize = 11264;
/// Rows a `_cols` block of a Q8_0 site walks: four a warp, two at a time, so a group's norms are
/// paid once for 32 rows.
pub const COLS_ROWS: usize = 4 * ROWS;
const _: () = assert!(THREADS == RMS_THREADS && WARPS == RMS_WARPS && WARPS == 8);
// The entries' `launch_bounds` and `launch_contract` blocks are literals.
const _: () = assert!(
    THREADS == 256 && MAX_K == 3072 && MAX_COLS == 8 && GROUP_COLS == 4 && GROUP_VALUES == 11264
);
// A group of the widest row holds at least one token.
const _: () = assert!(GROUP_VALUES >= MAX_K);
// `stage_group`'s twelve vectors a thread cover a group; the norm's twelve gains a thread a row.
const _: () = assert!(GROUP_VALUES <= 12 * 4 * THREADS && MAX_K <= 12 * THREADS);

/// One thread's sum of squares over `x[..k]`'s values `tid, tid + 256, …`, `acc += v · v` in
/// that order — `rms_partial_sq`'s sum — nine strides' loads at a time.
///
/// # Safety
///
/// `x.len() >= k`.
#[inline(always)]
unsafe fn sum_sq(x: &[f32], k: usize, tid: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut it = tid;
    while it + 8 * THREADS < k {
        // SAFETY: it + 8 · THREADS < k <= x.len() (the loop condition, this fn's contract).
        let v = unsafe {
            [
                *x.get_unchecked(it),
                *x.get_unchecked(it + THREADS),
                *x.get_unchecked(it + 2 * THREADS),
                *x.get_unchecked(it + 3 * THREADS),
                *x.get_unchecked(it + 4 * THREADS),
                *x.get_unchecked(it + 5 * THREADS),
                *x.get_unchecked(it + 6 * THREADS),
                *x.get_unchecked(it + 7 * THREADS),
                *x.get_unchecked(it + 8 * THREADS),
            ]
        };
        acc += v[0] * v[0];
        acc += v[1] * v[1];
        acc += v[2] * v[2];
        acc += v[3] * v[3];
        acc += v[4] * v[4];
        acc += v[5] * v[5];
        acc += v[6] * v[6];
        acc += v[7] * v[7];
        acc += v[8] * v[8];
        it += 9 * THREADS;
    }
    while it < k {
        // SAFETY: it < k <= x.len().
        let v = unsafe { *x.get_unchecked(it) };
        acc += v * v;
        it += THREADS;
    }
    acc
}

/// The block's copy of `src` (a token group's `len` values, `len <= GROUP_VALUES`) into `xn`:
/// thread `tid` copies vectors `tid, tid + 256, …` of four values, every load issued before its
/// stores, or values one at a time when `src` is not 16-byte aligned.
///
/// # Safety
///
/// `src.len() >= len`, `len <= GROUP_VALUES`, `len` a multiple of 4, and `xn` valid for writes
/// of `len` values (16-byte aligned) that no other thread of the block writes before a barrier.
#[inline(always)]
unsafe fn stage_group(xn: *mut f32, src: &[f32], len: usize, tid: usize) {
    // SAFETY: len <= src.len() (this fn's contract).
    let src = unsafe { src.get_unchecked(..len) };
    let Some(v) = as_vectors::<F32x4>(src) else {
        let mut j = tid;
        while j < len {
            // SAFETY: j < len <= src.len() and xn's len values; this thread's index.
            unsafe { *xn.add(j) = *src.get_unchecked(j) };
            j += THREADS;
        }
        return;
    };
    let n = v.len();
    let xv = xn.cast::<F32x4>();
    // Twelve vectors a thread cover GROUP_VALUES / 4 <= 12 · THREADS; vector i of the thread is
    // tid + i · THREADS, loaded when below n = len / 4 = v.len().
    macro_rules! load {
        ($i:literal) => {
            if tid + $i * THREADS < n {
                // SAFETY: tid + i · THREADS < n = v.len() (the guard).
                unsafe { *v.get_unchecked(tid + $i * THREADS) }
            } else {
                F32x4([0.0; 4])
            }
        };
    }
    macro_rules! store {
        ($vals:ident, $($i:literal),*) => {
            $(
                if tid + $i * THREADS < n {
                    // SAFETY: tid + i · THREADS < len / 4, inside xn's len values (16-byte
                    // aligned); this thread's vector.
                    unsafe { *xv.add(tid + $i * THREADS) = $vals[$i] };
                }
            )*
        };
    }
    let vals = [
        load!(0),
        load!(1),
        load!(2),
        load!(3),
        load!(4),
        load!(5),
        load!(6),
        load!(7),
        load!(8),
        load!(9),
        load!(10),
        load!(11),
    ];
    store!(vals, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
}

/// Rows a block of site `site` walks when a block of a Q8_0 site walks `q8`: `q8` for a Q8_0 site
/// ([`COLS_ROWS`] in the `_cols` entry, four a warp, a pair at a time by [`q8_0_pair_partials`];
/// [`STEP_Q8_ROWS`] in the one-token entry, two a warp), [`ROWS`] for an F32 site (one a warp).
#[inline(always)]
const fn block_rows(site: usize, q8: usize) -> usize {
    if site < 5 { q8 } else { ROWS }
}

/// Rows a `_cols` block of site `site` walks ([`block_rows`] at [`COLS_ROWS`]).
#[inline(always)]
const fn cols_rows(site: usize) -> usize {
    block_rows(site, COLS_ROWS)
}

/// The site whose blocks hold block `blk` ([`SITES`] past the last), its first block, and the
/// input it reads: the block-to-site rule at [`block_rows`] rows a block (`q8` a Q8_0 site's), a
/// site's last block holding the rows its count leaves.
#[inline(always)]
fn site_of(blk: usize, rows: [u32; SITES], (ik, iv): (u32, u32), q8: usize) -> (usize, usize, u32) {
    let mut site = 0usize;
    let mut first = 0usize;
    while site < SITES && blk >= first + (rows[site] as usize).div_ceil(block_rows(site, q8)) {
        first += (rows[site] as usize).div_ceil(block_rows(site, q8));
        site += 1;
    }
    let input = match site {
        1 | 6 => ik,
        2 | 7 => iv,
        _ => 0,
    };
    (site, first, input)
}

/// Lane `lane`'s partial sums of the Q8_0 rows `row0` and `row0 + 8` (the first `nr`, `nr` 1 or
/// 2) against `M` tokens of `xs` (token `c`'s `k` values at `c · k`): `p[i][c]`. Each row and
/// token is [`q8_0_lane_partials_mcol`]'s walk — the lane's code words `lane, lane + 32, …` in
/// increasing order, each word's four values in byte order, one multiply-add a term — so `p[i][c]`
/// is that walk's partial bit for bit; both rows' code words and scales of four steps are loaded
/// together, eight words and eight scales a lane in flight, before the multiply-adds that use
/// them. A row past `nr` and tokens `M..4` stay 0.0. Activations not 16-byte aligned take the
/// one-row walk per row, the same sums.
///
/// # Safety
///
/// Rows below `row0 + 8 · nr` exist in `qs` and `d` (`qs.len() >= (row + 1) · k/4`,
/// `d.len() >= (row + 1) · k/32`), `xs.len() >= M · k`, `k` a positive multiple of 32, `M` in
/// 1..=4, `nr` in 1..=2, `lane < 32`.
#[inline(always)]
#[allow(
    clippy::too_many_lines,
    reason = "device core: two rows' and four tokens' sums as named scalars, kept in registers"
)]
unsafe fn q8_0_pair_partials<const M: usize>(
    qs: &[u32],
    d: &[u16],
    xs: &[f32],
    k: usize,
    row0: usize,
    nr: usize,
    lane: usize,
) -> [[f32; 4]; 2] {
    const {
        assert!(M > 0 && M <= 4);
    };
    // SAFETY: M · k <= xs.len() by this fn's contract.
    let xr = unsafe { xs.get_unchecked(..M * k) };
    let Some(xq) = as_vectors::<F32x4>(xr) else {
        let mut p = [[0.0f32; 4]; 2];
        // Row i by a constant index, so `p` stays in registers.
        macro_rules! one_row {
            ($i:literal) => {
                if $i < nr {
                    // SAFETY: row row0 + 8i exists ($i < nr), xs holds M tokens of k (this fn's
                    // contract).
                    let f = unsafe {
                        q8_0_lane_partials_mcol::<M>(qs, d, xs, k as u32, row0 + 8 * $i, 0, k, lane)
                    };
                    p[$i] = [f[0], f[1], f[2], f[3]];
                }
            };
        }
        one_row!(0);
        one_row!(1);
        return p;
    };
    let words = k >> 2;
    let steps = k / Q8_STEP;
    let blocks = k >> 5;
    // Quads between one token's base and the next.
    let cq = k >> 2;
    // Row i's step-0 word and scale of this lane, as the one-row walk builds them.
    macro_rules! wq {
        ($i:literal) => {
            (row0 + 8 * $i) * words + lane
        };
    }
    macro_rules! wd {
        ($i:literal) => {
            (row0 + 8 * $i) * blocks + (lane >> 3)
        };
    }
    let (mut p00, mut p01, mut p02, mut p03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut p10, mut p11, mut p12, mut p13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    // Row i's code word and scale of step s, or zeros past the pair's rows.
    macro_rules! code {
        ($i:literal, $s:expr) => {
            if $i < nr {
                // SAFETY: row row0 + 8i exists ($i < nr) and the caller's step bound puts the
                // lane's word of step s inside it.
                unsafe { block_code::<Q8_0Bytes>(qs, d, wq!($i), wd!($i), $s) }
            } else {
                (0u32, 0.0f32)
            }
        };
    }
    let mut s = 0usize;
    while s + 4 <= steps {
        // Steps s .. s + 3 are whole (s + 4 <= steps): every lane's word in them exists.
        let c0 = (
            code!(0, s),
            code!(0, s + 1),
            code!(0, s + 2),
            code!(0, s + 3),
        );
        let c1 = (
            code!(1, s),
            code!(1, s + 1),
            code!(1, s + 2),
            code!(1, s + 3),
        );
        let a = lane + 32 * s;
        macro_rules! tok {
            ($c:literal, $f0:ident, $f1:ident) => {
                if M > $c {
                    let b = $c * cq + a;
                    // SAFETY: quad a + 96 = word 32(s + 3) + lane < k/4 of token $c < M, whose
                    // quads start at $c · cq: inside xq (M · k/4 quads).
                    let v = unsafe {
                        (
                            *xq.get_unchecked(b),
                            *xq.get_unchecked(b + 32),
                            *xq.get_unchecked(b + 64),
                            *xq.get_unchecked(b + 96),
                        )
                    };
                    $f0 = word_dot::<Q8_0Bytes>($f0, c0.0.0, c0.0.1, v.0);
                    $f0 = word_dot::<Q8_0Bytes>($f0, c0.1.0, c0.1.1, v.1);
                    $f0 = word_dot::<Q8_0Bytes>($f0, c0.2.0, c0.2.1, v.2);
                    $f0 = word_dot::<Q8_0Bytes>($f0, c0.3.0, c0.3.1, v.3);
                    if 1 < nr {
                        $f1 = word_dot::<Q8_0Bytes>($f1, c1.0.0, c1.0.1, v.0);
                        $f1 = word_dot::<Q8_0Bytes>($f1, c1.1.0, c1.1.1, v.1);
                        $f1 = word_dot::<Q8_0Bytes>($f1, c1.2.0, c1.2.1, v.2);
                        $f1 = word_dot::<Q8_0Bytes>($f1, c1.3.0, c1.3.1, v.3);
                    }
                }
            };
        }
        tok!(0, p00, p10);
        tok!(1, p01, p11);
        tok!(2, p02, p12);
        tok!(3, p03, p13);
        s += 4;
    }
    // Whole steps left over, then the partial word: a lane's word of step `steps` exists only
    // when 32 · steps + lane < k/4.
    while s <= steps {
        if s == steps && 32 * steps + lane >= words {
            break;
        }
        let c0 = code!(0, s);
        let c1 = code!(1, s);
        let a = lane + 32 * s;
        macro_rules! one {
            ($c:literal, $f0:ident, $f1:ident) => {
                if M > $c {
                    // SAFETY: quad a = word 32s + lane < k/4 of token $c < M, inside xq.
                    let v = unsafe { *xq.get_unchecked($c * cq + a) };
                    $f0 = word_dot::<Q8_0Bytes>($f0, c0.0, c0.1, v);
                    if 1 < nr {
                        $f1 = word_dot::<Q8_0Bytes>($f1, c1.0, c1.1, v);
                    }
                }
            };
        }
        one!(0, p00, p10);
        one!(1, p01, p11);
        one!(2, p02, p12);
        one!(3, p03, p13);
        s += 1;
    }
    [[p00, p01, p02, p03], [p10, p11, p12, p13]]
}

/// A Q8_0 site's output at `row` and, when `nr` is 2, at `row + ROWS`: the warp's pair of sums.
///
/// # Safety
///
/// `y` holds a value at `row` and, when `nr` is 2, at `row + ROWS`, and the caller alone writes
/// them.
#[inline(always)]
unsafe fn store_pair(y: &mut DisjointSlice<f32>, row: usize, (s0, s1): (f32, f32), nr: usize) {
    // SAFETY: row < y.len() by this fn's contract.
    unsafe { *y.get_unchecked_mut(row) = s0 };
    if nr == 2 {
        // SAFETY: row + ROWS < y.len() by this fn's contract.
        unsafe { *y.get_unchecked_mut(row + ROWS) = s1 };
    }
}

#[cuda_module]
mod norm_proj_kernels {
    use super::*;

    /// The norms and the eight projections (module doc): block `b` is rows
    /// `STEP_Q8_ROWS · (b − first) ..` of the Q8_0 site, `ROWS · (b − first) ..` of the F32 site,
    /// whose blocks hold `b`, `first` that site's first block. Four blocks an SM: the register cap
    /// the fourth sets (64) is what keeps the launch's blocks one wave of the card.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "kernel entry: the device ABI takes the eight sites' planes and outputs flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 4)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            k >= 32,
            k <= 3072,
            x0.len() >= k,
            x1.len() >= k,
            x2.len() >= k,
            g0.len() >= k,
            g1.len() >= k,
            g2.len() >= k,
            ik < 3,
            iv < 3,
            4 * qs_q.len() >= rq * k,
            32 * d_q.len() >= rq * k,
            4 * qs_k.len() >= rk * k,
            32 * d_k.len() >= rk * k,
            4 * qs_v.len() >= rv * k,
            32 * d_v.len() >= rv * k,
            4 * qs_r.len() >= rr * k,
            32 * d_r.len() >= rr * k,
            4 * qs_g.len() >= rg * k,
            32 * d_g.len() >= rg * k,
            w_s.len() >= rs * k,
            w_kt.len() >= rkt * k,
            w_vt.len() >= rvt * k,
            y_q.len() >= rq,
            y_k.len() >= rk,
            y_v.len() >= rv,
            y_r.len() >= rr,
            y_g.len() >= rg,
            y_s.len() >= rs,
            y_kt.len() >= rkt,
            y_vt.len() >= rvt
        )
    )]
    pub fn norm_proj_q8_f32(
        x0: &[f32],
        x1: &[f32],
        x2: &[f32],
        g0: &[f32],
        g1: &[f32],
        g2: &[f32],
        ik: u32,
        iv: u32,
        eps: f32,
        k: u32,
        fault: FaultSink,
        qs_q: &[u32],
        d_q: &[u16],
        qs_k: &[u32],
        d_k: &[u16],
        qs_v: &[u32],
        d_v: &[u16],
        qs_r: &[u32],
        d_r: &[u16],
        qs_g: &[u32],
        d_g: &[u16],
        w_s: &[f32],
        w_kt: &[f32],
        w_vt: &[f32],
        rq: u32,
        rk: u32,
        rv: u32,
        rr: u32,
        rg: u32,
        rs: u32,
        rkt: u32,
        rvt: u32,
        mut y_q: DisjointSlice<f32>,
        mut y_k: DisjointSlice<f32>,
        mut y_v: DisjointSlice<f32>,
        mut y_r: DisjointSlice<f32>,
        mut y_g: DisjointSlice<f32>,
        mut y_s: DisjointSlice<f32>,
        mut y_kt: DisjointSlice<f32>,
        mut y_vt: DisjointSlice<f32>,
    ) {
        static mut XN: SharedArray<f32, MAX_K, 16> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let k_us = k as usize;
        // The site whose blocks hold this block, and the block's first block in it.
        let rows = [rq, rk, rv, rr, rg, rs, rkt, rvt];
        let (site, first, input) = site_of(blk, rows, (ik, iv), STEP_Q8_ROWS);
        if site >= SITES {
            return; // block-uniform: past the last site's blocks
        }
        let (x, g) = match input {
            0 => (x0, g0),
            1 => (x1, g1),
            _ => (x2, g2),
        };
        // SAFETY: XN and WSUM are this block's own shared allocations; the raw form is the only
        // way to reach them without a reference to a `static mut`. Every access is below its
        // length (k <= MAX_K by the contract, tid / 32 < RMS_WARPS) and ordered by
        // `sync_threads`.
        let xn = unsafe { SharedArray::as_raw_mut_ptr(&raw mut XN) };
        // SAFETY: as above.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // SAFETY: x.len() >= k by the contract.
        let part = warp::reduce_sum_f32(unsafe { sum_sq(x, k_us, tid) });
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < RMS_WARPS; one lane a warp writes its slot.
            unsafe { *ws.add(tid / 32) = part };
        }
        thread::sync_threads();
        // SAFETY: every slot was written before the barrier above.
        let sums = unsafe {
            [
                *ws.add(0),
                *ws.add(1),
                *ws.add(2),
                *ws.add(3),
                *ws.add(4),
                *ws.add(5),
                *ws.add(6),
                *ws.add(7),
            ]
        };
        let scale = rms_scale_checked(rms_warp_tree(sums), k, eps, fault, tid == 0);
        let mut j = tid;
        while j < k_us {
            // SAFETY: j < k <= x.len(), g.len() and MAX_K; this thread alone writes value j.
            unsafe { *xn.add(j) = (scale * *g.get_unchecked(j)) * *x.get_unchecked(j) };
            j += THREADS;
        }
        thread::sync_threads();
        // SAFETY: the k values of XN were written before the barrier above; nothing writes them
        // past it.
        let xs: &[f32] = unsafe { core::slice::from_raw_parts(xn.cast_const(), k_us) };
        let lane = warp::lane_id() as usize;
        let wi = tid / 32;
        if site < 5 {
            // A Q8_0 site: this warp's rows `row0` and `row0 + ROWS` (the second only when the
            // site's count leaves it), walked together by the `_cols` entry's pair walk at one
            // token, whose sums are the one-row walk's bit for bit.
            // The site's planes and row count by a select, then one call: `rows` is indexed only
            // by the loop that finds the site, so it stays out of local memory.
            let (qs, d, site_rows) = match site {
                0 => (qs_q, d_q, rq),
                1 => (qs_k, d_k, rk),
                2 => (qs_v, d_v, rv),
                3 => (qs_r, d_r, rr),
                _ => (qs_g, d_g, rg),
            };
            let row0 = (blk - first) * STEP_Q8_ROWS + wi;
            let nr = 1 + usize::from(row0 + ROWS < site_rows as usize);
            // SAFETY: the rows row0 + ROWS · i below the site's count (i < nr) lie inside its
            // planes by the contract, xs holds k values, k a multiple of 32 (host-checked) and
            // lane < 32.
            let p = unsafe { q8_0_pair_partials::<1>(qs, d, xs, k_us, row0, nr, lane) };
            let (s0, s1) = (warp::reduce_sum_f32(p[0][0]), warp::reduce_sum_f32(p[1][0]));
            if lane == 0 {
                // SAFETY: row0 and, when nr is 2, row0 + ROWS are below the site's rows, which the
                // site's output holds by the contract; only lane 0 of the warp writes them.
                unsafe {
                    match site {
                        0 => store_pair(&mut y_q, row0, (s0, s1), nr),
                        1 => store_pair(&mut y_k, row0, (s0, s1), nr),
                        2 => store_pair(&mut y_v, row0, (s0, s1), nr),
                        3 => store_pair(&mut y_r, row0, (s0, s1), nr),
                        _ => store_pair(&mut y_g, row0, (s0, s1), nr),
                    }
                }
            }
            return;
        }
        // An F32 site: each warp walks one row, ROWS | the site's rows.
        let row = (blk - first) * ROWS + wi;
        // SAFETY (every arm): row < the site's rows puts the row's values inside its plane by the
        // contract, xs holds k values, k a multiple of 32 (host-checked) and lane < 32; y has the
        // site's rows by the contract, and only lane 0 of the row's warp writes slot row.
        let s = match site {
            // SAFETY: as above.
            5 => warp::reduce_sum_f32(unsafe { f32_lane_partial_1col(w_s, xs, k, row, lane) }),
            // SAFETY: as above.
            6 => warp::reduce_sum_f32(unsafe { f32_lane_partial_1col(w_kt, xs, k, row, lane) }),
            // SAFETY: as above.
            _ => warp::reduce_sum_f32(unsafe { f32_lane_partial_1col(w_vt, xs, k, row, lane) }),
        };
        if lane == 0 {
            // SAFETY: as above.
            unsafe {
                match site {
                    5 => *y_s.get_unchecked_mut(row) = s,
                    6 => *y_kt.get_unchecked_mut(row) = s,
                    _ => *y_vt.get_unchecked_mut(row) = s,
                }
            }
        }
    }

    /// The norms and the eight projections of `m` tokens in groups of `cols` (module doc): block
    /// `b` is token group `b % groups` of rows `cols_rows · (b / groups − first) ..` of the site
    /// whose blocks hold `b / groups` ([`site_of`] at [`COLS_ROWS`]), `groups = ⌈m / cols⌉`.
    /// Two blocks an SM: two blocks' shared rows fit an SM's shared memory, and the register
    /// cap the second block sets is what keeps both resident.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "kernel entry: the device ABI takes the eight sites' planes and outputs flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            k >= 32,
            k <= 3072,
            m >= 1,
            m <= 8,
            cols >= 1,
            cols <= 4,
            cols * k <= 11264,
            x0.len() >= m * k,
            x1.len() >= m * k,
            x2.len() >= m * k,
            g0.len() >= k,
            g1.len() >= k,
            g2.len() >= k,
            ik < 3,
            iv < 3,
            4 * qs_q.len() >= rq * k,
            32 * d_q.len() >= rq * k,
            4 * qs_k.len() >= rk * k,
            32 * d_k.len() >= rk * k,
            4 * qs_v.len() >= rv * k,
            32 * d_v.len() >= rv * k,
            4 * qs_r.len() >= rr * k,
            32 * d_r.len() >= rr * k,
            4 * qs_g.len() >= rg * k,
            32 * d_g.len() >= rg * k,
            w_s.len() >= rs * k,
            w_kt.len() >= rkt * k,
            w_vt.len() >= rvt * k,
            y_q.len() >= m * rq,
            y_k.len() >= m * rk,
            y_v.len() >= m * rv,
            y_r.len() >= m * rr,
            y_g.len() >= m * rg,
            y_s.len() >= m * rs,
            y_kt.len() >= m * rkt,
            y_vt.len() >= m * rvt
        )
    )]
    pub fn norm_proj_q8_f32_cols(
        x0: &[f32],
        x1: &[f32],
        x2: &[f32],
        g0: &[f32],
        g1: &[f32],
        g2: &[f32],
        ik: u32,
        iv: u32,
        eps: f32,
        k: u32,
        m: u32,
        cols: u32,
        fault: FaultSink,
        qs_q: &[u32],
        d_q: &[u16],
        qs_k: &[u32],
        d_k: &[u16],
        qs_v: &[u32],
        d_v: &[u16],
        qs_r: &[u32],
        d_r: &[u16],
        qs_g: &[u32],
        d_g: &[u16],
        w_s: &[f32],
        w_kt: &[f32],
        w_vt: &[f32],
        rq: u32,
        rk: u32,
        rv: u32,
        rr: u32,
        rg: u32,
        rs: u32,
        rkt: u32,
        rvt: u32,
        mut y_q: DisjointSlice<f32>,
        mut y_k: DisjointSlice<f32>,
        mut y_v: DisjointSlice<f32>,
        mut y_r: DisjointSlice<f32>,
        mut y_g: DisjointSlice<f32>,
        mut y_s: DisjointSlice<f32>,
        mut y_kt: DisjointSlice<f32>,
        mut y_vt: DisjointSlice<f32>,
    ) {
        static mut XN: SharedArray<f32, GROUP_VALUES, 16> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<f32, { GROUP_COLS * RMS_WARPS }> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let (k_us, m_us, cols_us) = (k as usize, m as usize, cols as usize);
        let groups = m_us.div_ceil(cols_us);
        let (row_block, group) = (blk / groups, blk % groups);
        let rows = [rq, rk, rv, rr, rg, rs, rkt, rvt];
        let (site, first, input) = site_of(row_block, rows, (ik, iv), COLS_ROWS);
        if site >= SITES {
            return; // block-uniform: past the last site's blocks
        }
        // The group's first token and its tokens: group < ⌈m / cols⌉, so c0 < m and n_tok >= 1.
        let c0 = group * cols_us;
        let n_tok = if m_us - c0 < cols_us {
            m_us - c0
        } else {
            cols_us
        };
        let (x, g) = match input {
            0 => (x0, g0),
            1 => (x1, g1),
            _ => (x2, g2),
        };
        // SAFETY: XN and WSUM are this block's own shared allocations, reached raw; every access
        // is below its length (n_tok · k <= cols · k <= GROUP_VALUES and n_tok <= GROUP_COLS by
        // the contract, tid / 32 < RMS_WARPS) and ordered by `sync_threads`.
        let (xn, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut XN),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
            )
        };
        // The group's tokens are the n_tok · k values of x from token c0's.
        // SAFETY: (c0 + n_tok) · k <= m · k <= x.len() by the contract; n_tok · k <= GROUP_VALUES
        // and a multiple of 4 (k is a multiple of 32, host-checked); XN is 16-byte aligned, and
        // each thread writes its own vectors, read past the barrier below.
        unsafe { stage_group(xn, x.get_unchecked(c0 * k_us..), n_tok * k_us, tid) };
        // The thread's gains, gain i = g[tid + i · THREADS] (0 past k), loaded beside the copy.
        macro_rules! gain {
            ($i:literal) => {
                if tid + $i * THREADS < k_us {
                    // SAFETY: tid + i · THREADS < k <= g.len() (the guard, the contract).
                    unsafe { *g.get_unchecked(tid + $i * THREADS) }
                } else {
                    0.0
                }
            };
        }
        let gv = [
            gain!(0),
            gain!(1),
            gain!(2),
            gain!(3),
            gain!(4),
            gain!(5),
            gain!(6),
            gain!(7),
            gain!(8),
            gain!(9),
            gain!(10),
            gain!(11),
        ];
        thread::sync_threads();
        let mut c = 0usize;
        while c < n_tok {
            // SAFETY: token c's k values at c · k, c · k + k <= n_tok · k, were written before the
            // barrier above; this thread reads only its own indices of them (`sum_sq`).
            let part = warp::reduce_sum_f32(unsafe {
                sum_sq(
                    core::slice::from_raw_parts(xn.add(c * k_us).cast_const(), k_us),
                    k_us,
                    tid,
                )
            });
            if warp::lane_id() == 0 {
                // SAFETY: c · RMS_WARPS + tid / 32 < GROUP_COLS · RMS_WARPS; one lane a warp
                // writes its slot.
                unsafe { *ws.add(c * RMS_WARPS + tid / 32) = part };
            }
            c += 1;
        }
        thread::sync_threads();
        let mut c = 0usize;
        while c < n_tok {
            // SAFETY: every slot of token c was written before the barrier above.
            let sums = unsafe {
                let w = ws.add(c * RMS_WARPS);
                [
                    *w.add(0),
                    *w.add(1),
                    *w.add(2),
                    *w.add(3),
                    *w.add(4),
                    *w.add(5),
                    *w.add(6),
                    *w.add(7),
                ]
            };
            let scale = rms_scale_checked(rms_warp_tree(sums), k, eps, fault, tid == 0);
            // Value j of token c normed in place, `(scale · g[j]) · x[j]` as the one-token entry
            // norms it, j = tid + i · THREADS: the indices `sum_sq` read on this thread.
            let xc = xn.wrapping_add(c * k_us);
            macro_rules! norm {
                ($($i:literal),*) => {
                    $(
                        if tid + $i * THREADS < k_us {
                            // SAFETY: c · k + j < n_tok · k <= GROUP_VALUES; this thread alone
                            // reads and writes value j of token c.
                            unsafe {
                                *xc.add(tid + $i * THREADS) =
                                    (scale * gv[$i]) * *xc.add(tid + $i * THREADS);
                            }
                        }
                    )*
                };
            }
            norm!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            c += 1;
        }
        thread::sync_threads();
        // SAFETY: the n_tok · k values of XN were written before the barrier above; nothing
        // writes them past it.
        let xs: &[f32] = unsafe { core::slice::from_raw_parts(xn.cast_const(), n_tok * k_us) };
        let lane = warp::lane_id() as usize;
        // The site's rows by a constant index, so `rows` stays in registers.
        let n = match site {
            0 => rq,
            1 => rk,
            2 => rv,
            3 => rr,
            4 => rg,
            5 => rs,
            6 => rkt,
            _ => rvt,
        } as usize;
        // The warp's rows: base + 8i for i < nr, the block's rows below the site's, so a site's
        // last block walks the rows its count leaves (nr is warp-uniform).
        let per = cols_rows(site);
        let base = (row_block - first) * per + tid / 32;
        if base >= n {
            return; // warp-uniform, and no barrier follows
        }
        let nr = (n - base).div_ceil(ROWS).min(per / ROWS);
        // Row r's butterfly a token from its partials f and lane 0's stores.
        macro_rules! finish {
            ($f:expr, $row:expr) => {{
                let f: [f32; 4] = $f;
                let sums =
                    gemv_lane_sums([f[0], f[1], f[2], f[3], 0.0, 0.0, 0.0, 0.0], n_tok as u32);
                if lane == 0 {
                    let row = $row;
                    let mut c = 0usize;
                    while c < n_tok {
                        let at = (c0 + c) * n + row;
                        // Token c's sum by a constant index, so `sums` stays in registers.
                        let v = match c {
                            0 => sums[0],
                            1 => sums[1],
                            2 => sums[2],
                            _ => sums[3],
                        };
                        // SAFETY: at < m · rows <= y.len() by the contract (c0 + c < m,
                        // row < rows); only lane 0 of the row's warp of the token's group writes
                        // token c0 + c's slot of row.
                        unsafe {
                            match site {
                                0 => *y_q.get_unchecked_mut(at) = v,
                                1 => *y_k.get_unchecked_mut(at) = v,
                                2 => *y_v.get_unchecked_mut(at) = v,
                                3 => *y_r.get_unchecked_mut(at) = v,
                                4 => *y_g.get_unchecked_mut(at) = v,
                                5 => *y_s.get_unchecked_mut(at) = v,
                                6 => *y_kt.get_unchecked_mut(at) = v,
                                _ => *y_vt.get_unchecked_mut(at) = v,
                            }
                        }
                        c += 1;
                    }
                }
            }};
        }
        // The walk, a pair of the warp's rows at a time: pair j is rows base + 16j and
        // base + 16j + 8, the first np of them; a Q8_0 site's pair together, an F32 site's one row.
        // SAFETY (both arms): those rows are below the site's rows (nr's bound), which puts their
        // words and scales, or values, inside the planes by the contract; xs holds n_tok · k
        // values, token c at c · k (16-byte aligned: XN is, and k is a multiple of 32,
        // host-checked); n_tok in 1..=GROUP_COLS, np in 1..=2 and lane < 32.
        macro_rules! pair {
            ($j:literal) => {
                if nr > 2 * $j {
                    let r0 = base + 16 * $j;
                    let np = if nr - 2 * $j < 2 { nr - 2 * $j } else { 2 };
                    let p = if site < 5 {
                        let (qs, d) = match site {
                            0 => (qs_q, d_q),
                            1 => (qs_k, d_k),
                            2 => (qs_v, d_v),
                            3 => (qs_r, d_r),
                            _ => (qs_g, d_g),
                        };
                        // SAFETY: as above; M is n_tok.
                        unsafe {
                            match n_tok {
                                1 => q8_0_pair_partials::<1>(qs, d, xs, k_us, r0, np, lane),
                                2 => q8_0_pair_partials::<2>(qs, d, xs, k_us, r0, np, lane),
                                3 => q8_0_pair_partials::<3>(qs, d, xs, k_us, r0, np, lane),
                                _ => q8_0_pair_partials::<4>(qs, d, xs, k_us, r0, np, lane),
                            }
                        }
                    } else {
                        let w = match site {
                            5 => w_s,
                            6 => w_kt,
                            _ => w_vt,
                        };
                        // SAFETY: as above (np is 1: an F32 block's warp walks one row).
                        let f = unsafe { f32_lane_partials(w, xs, k, r0, n_tok as u32, lane) };
                        [[f[0], f[1], f[2], f[3]], [0.0; 4]]
                    };
                    finish!(p[0], r0);
                    if np > 1 {
                        finish!(p[1], r0 + ROWS);
                    }
                }
            };
        }
        pair!(0);
        pair!(1);
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct NormProjKernels {
    module: norm_proj_kernels::LoadedModule,
}

/// A Q8_0 site's planes and its output.
pub struct Q8Site<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// An F32 site's rows and its output.
pub struct F32Site<'a> {
    pub w: &'a DeviceTensor<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// One launch's inputs and sites (module doc).
pub struct NormProjArgs<'a> {
    /// The input rows and their gains: one, or three (`ik`, `iv` then name rows 1 and 2).
    pub x: [&'a DeviceBuffer<f32>; 3],
    pub gain: [&'a DeviceBuffer<f32>; 3],
    /// The inputs `k`, `kt` and `v`, `vt` read.
    pub ik: usize,
    pub iv: usize,
    pub eps: f32,
    /// The input width.
    pub k: usize,
    pub fault: FaultSink,
    /// `q`, `k`, `v`, `rel`, `gate`.
    pub q8: [Q8Site<'a>; 5],
    /// `sink`, then the two conv-tap sites: run where `taps` holds, else of no row (their
    /// rows are not read and their outputs not written).
    pub sink: F32Site<'a>,
    pub kt: F32Site<'a>,
    pub vt: F32Site<'a>,
    pub taps: bool,
}

impl NormProjKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<NormProjKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above;
        // the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(norm_proj_kernels, ctx)? };
        Ok(NormProjKernels { module })
    }

    /// Enqueue the norms and the projections of `a` at one token ([`NormProjKernels::enqueue_rows`]
    /// at `m` 1: `norm_proj_q8_f32`).
    pub fn enqueue(&self, stream: &CudaStream, a: NormProjArgs<'_>) -> Result<(), GpuError> {
        self.enqueue_rows(stream, a, 1)
    }

    /// Enqueue the norms and the projections of `a` over `m` tokens (module doc): at `m` 1
    /// `norm_proj_q8_f32`, `Σ ⌈q8 rows / 16⌉ + Σ f32 rows / 8` blocks of 256 threads; past it
    /// `norm_proj_q8_f32_cols`,
    /// `Σ ⌈rows / cols_rows⌉` blocks for each group of `cols = min(GROUP_COLS, GROUP_VALUES / k)`
    /// tokens. An input is `m · k` values, token after token, and a site's output `m` times its
    /// rows, token-major. A width that is not a positive multiple of 32 or past [`MAX_K`], `m`
    /// outside `1..=`[`MAX_COLS`], an input shorter than `m · k`, a site whose planes are not `k`
    /// wide or whose rows are not a multiple of [`ROWS`], an output shorter than `m` times its rows
    /// and an input index past 2 are refused by name. Asynchronous, allocation-free, capturable.
    pub fn enqueue_rows(
        &self,
        stream: &CudaStream,
        a: NormProjArgs<'_>,
        m: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "norm_proj::enqueue";
        let bad = |detail: String| Err(GpuError::shape(WHAT, detail));
        let k = a.k;
        if k == 0 || !k.is_multiple_of(32) || k > MAX_K {
            return bad(format!(
                "an input of {k} values: a positive multiple of 32 up to {MAX_K}"
            ));
        }
        if m == 0 || m > MAX_COLS {
            return bad(format!("{m} tokens: 1..={MAX_COLS}"));
        }
        if a.ik > 2 || a.iv > 2 {
            return bad(format!("inputs {} and {}: three rows at most", a.ik, a.iv));
        }
        for (i, (x, g)) in a.x.iter().zip(&a.gain).enumerate() {
            if x.len() < m * k || g.len() < k {
                return bad(format!(
                    "input {i}: {} values and a gain of {}, for {m} token(s) of {k}",
                    x.len(),
                    g.len()
                ));
            }
        }
        let rows_of = |name: &str, rows: usize, y: usize| -> Result<usize, GpuError> {
            if !rows.is_multiple_of(ROWS) || y < m * rows {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name}: {rows} rows (a multiple of {ROWS}) of {m} token(s) into an \
                         output of {y}"
                    ),
                ));
            }
            Ok(rows)
        };
        const Q8_NAMES: [&str; 5] = ["q", "k", "v", "rel", "gate"];
        let mut q8_rows = [0usize; 5];
        for (i, s) in a.q8.iter().enumerate() {
            if s.qs.cols() * 4 != k || s.d.cols() * 32 != k || s.qs.rows() != s.d.rows() {
                return bad(format!(
                    "{}: Q8_0 planes of {} words and {} scales a row over {} rows, for {k} values",
                    Q8_NAMES[i],
                    s.qs.cols(),
                    s.d.cols(),
                    s.qs.rows()
                ));
            }
            q8_rows[i] = rows_of(Q8_NAMES[i], s.qs.rows(), s.y.len())?;
        }
        let f32_rows = |name: &str, s: &F32Site<'_>| -> Result<usize, GpuError> {
            if s.w.cols() != k {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}: F32 rows of {} values, for {k}", s.w.cols()),
                ));
            }
            rows_of(name, s.w.rows(), s.y.len())
        };
        let rs = f32_rows("sink", &a.sink)?;
        let (rkt, rvt) = if a.taps {
            (f32_rows("kt", &a.kt)?, f32_rows("vt", &a.vt)?)
        } else {
            (0, 0)
        };
        // The one-token entry's blocks: a Q8_0 site's rows in blocks of `STEP_Q8_ROWS`, its last
        // block the rows its count leaves, an F32 site's in blocks of `ROWS`.
        let blocks: usize = q8_rows
            .iter()
            .map(|r| r.div_ceil(STEP_Q8_ROWS))
            .sum::<usize>()
            + (rs + rkt + rvt) / ROWS;
        // The `_cols` entry's blocks: each site's rows in blocks of `cols_rows`, its last block the
        // rows its count leaves.
        let cols_blocks: usize = q8_rows
            .iter()
            .chain([&rs, &rkt, &rvt])
            .enumerate()
            .map(|(site, r)| r.div_ceil(cols_rows(site)))
            .sum();
        // A group of the widest row holds a token (GROUP_VALUES >= MAX_K), so `cols` >= 1.
        let cols = GROUP_COLS.min(GROUP_VALUES / k).min(m);
        let u = |name: &'static str, v: usize| launch_u32(WHAT, name, v);
        let NormProjArgs {
            x,
            gain,
            ik,
            iv,
            eps,
            fault,
            q8,
            sink,
            kt,
            vt,
            ..
        } = a;
        let [sq, sk, sv, sr, sg] = q8;
        let (
            F32Site { w: w_s, y: y_s },
            F32Site { w: w_kt, y: y_kt },
            F32Site { w: w_vt, y: y_vt },
        ) = (sink, kt, vt);
        macro_rules! launch {
            ($prepare:ident, $entry:ident, $grid:expr, $($tok:expr),*) => {{
                let prep = self.module.$prepare(LaunchConfig1D::new(
                    u("grid", $grid)?,
                    THREADS as u32,
                    0,
                ))?;
                self.module.$entry(
                    stream,
                    &prep,
                    x[0],
                    x[1],
                    x[2],
                    gain[0],
                    gain[1],
                    gain[2],
                    u("ik", ik)?,
                    u("iv", iv)?,
                    eps,
                    u("k", k)?,
                    $($tok,)*
                    fault,
                    sq.qs.buf(),
                    sq.d.buf(),
                    sk.qs.buf(),
                    sk.d.buf(),
                    sv.qs.buf(),
                    sv.d.buf(),
                    sr.qs.buf(),
                    sr.d.buf(),
                    sg.qs.buf(),
                    sg.d.buf(),
                    w_s.buf(),
                    w_kt.buf(),
                    w_vt.buf(),
                    u("rows", q8_rows[0])?,
                    u("rows", q8_rows[1])?,
                    u("rows", q8_rows[2])?,
                    u("rows", q8_rows[3])?,
                    u("rows", q8_rows[4])?,
                    u("rows", rs)?,
                    u("rows", rkt)?,
                    u("rows", rvt)?,
                    sq.y,
                    sk.y,
                    sv.y,
                    sr.y,
                    sg.y,
                    y_s,
                    y_kt,
                    y_vt,
                )?;
            }};
        }
        if m == 1 {
            launch!(prepare_norm_proj_q8_f32, norm_proj_q8_f32, blocks,);
        } else {
            launch!(
                prepare_norm_proj_q8_f32_cols,
                norm_proj_q8_f32_cols,
                cols_blocks * m.div_ceil(cols),
                u("m", m)?,
                u("cols", cols)?
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A value in [-1, 1) of the stream.
    fn unit(state: &mut u64) -> f32 {
        ((next(state) >> 33) % 2001) as f32 / 1000.0 - 1.0
    }

    /// The rows entry against the one-token entry: three inputs (`ik` 1, `iv` 2), five Q8_0
    /// sites and three F32 sites of 8 to 40 rows, at K 2304 (groups of four tokens) and K 3072
    /// (groups of three), m = 2 to [`MAX_COLS`]. Every token's outputs are the one-token
    /// launch's on that token's rows, bit for bit, and the fault word stays clear.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it"]
    fn hw_norm_proj_rows_are_the_one_token_launch() {
        const EPS: f32 = 1e-6;
        const ROWS_OF: [usize; SITES] = [40, 16, 24, 32, 16, 8, 16, 24];
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let kernels = NormProjKernels::load(gpu.context()).expect("the module");
        let mut state = 0x5151u64;
        let mut failures = Vec::new();
        for k in [2304usize, 3072] {
            let gains: Vec<DeviceBuffer<f32>> = (0..3)
                .map(|_| {
                    let g: Vec<f32> = (0..k).map(|_| 0.5 + unit(&mut state).abs()).collect();
                    DeviceBuffer::from_host(stream, &g).expect("gain")
                })
                .collect();
            // Q8_0 planes: four signed bytes a word, an f16 scale in [2^-7, 2^-4) a block.
            let planes: Vec<(DeviceTensor<u32>, DeviceTensor<u16>)> = (0..5)
                .map(|s| {
                    let words: Vec<u32> = (0..ROWS_OF[s] * k / 4)
                        .map(|_| (next(&mut state) >> 16) as u32)
                        .collect();
                    let scales: Vec<u16> = (0..ROWS_OF[s] * k / 32)
                        .map(|_| 0x2000 + (next(&mut state) % 0x0c00) as u16)
                        .collect();
                    (
                        DeviceTensor::upload(stream, &words, ROWS_OF[s], k / 4).expect("words"),
                        DeviceTensor::upload(stream, &scales, ROWS_OF[s], k / 32).expect("scales"),
                    )
                })
                .collect();
            let f32w: Vec<DeviceTensor<f32>> = (5..SITES)
                .map(|s| {
                    let w: Vec<f32> = (0..ROWS_OF[s] * k).map(|_| unit(&mut state)).collect();
                    DeviceTensor::upload(stream, &w, ROWS_OF[s], k).expect("rows")
                })
                .collect();
            for m in 2..=MAX_COLS {
                let xs: Vec<Vec<f32>> = (0..3)
                    .map(|_| (0..m * k).map(|_| 3.0 * unit(&mut state)).collect())
                    .collect();
                let launch =
                    |x: [&DeviceBuffer<f32>; 3], m: usize, ys: &mut [DeviceBuffer<f32>]| {
                        let [y0, y1, y2, y3, y4, y5, y6, y7] = ys else {
                            unreachable!("eight sites")
                        };
                        let q8 = |s: usize, y| Q8Site {
                            qs: &planes[s].0,
                            d: &planes[s].1,
                            y,
                        };
                        kernels
                            .enqueue_rows(
                                stream,
                                NormProjArgs {
                                    x,
                                    gain: [&gains[0], &gains[1], &gains[2]],
                                    ik: 1,
                                    iv: 2,
                                    eps: EPS,
                                    k,
                                    fault: gpu.unlabelled_sink(),
                                    q8: [q8(0, y0), q8(1, y1), q8(2, y2), q8(3, y3), q8(4, y4)],
                                    sink: F32Site { w: &f32w[0], y: y5 },
                                    kt: F32Site { w: &f32w[1], y: y6 },
                                    vt: F32Site { w: &f32w[2], y: y7 },
                                    taps: true,
                                },
                                m,
                            )
                            .expect("the launch");
                    };
                let outs = |m: usize| -> Vec<DeviceBuffer<f32>> {
                    (0..SITES)
                        .map(|s| {
                            DeviceBuffer::from_host(stream, &vec![f32::NAN; m * ROWS_OF[s]])
                                .expect("y")
                        })
                        .collect()
                };
                let x_dev: Vec<DeviceBuffer<f32>> = xs
                    .iter()
                    .map(|x| DeviceBuffer::from_host(stream, x).expect("x"))
                    .collect();
                let mut ys = outs(m);
                launch([&x_dev[0], &x_dev[1], &x_dev[2]], m, &mut ys);
                let got: Vec<Vec<f32>> = ys
                    .iter()
                    .map(|y| y.to_host_vec(stream).expect("got"))
                    .collect();
                for c in 0..m {
                    let tok: Vec<DeviceBuffer<f32>> = xs
                        .iter()
                        .map(|x| DeviceBuffer::from_host(stream, &x[c * k..][..k]).expect("x"))
                        .collect();
                    let mut one = outs(1);
                    launch([&tok[0], &tok[1], &tok[2]], 1, &mut one);
                    for (s, y) in one.iter().enumerate() {
                        let want = y.to_host_vec(stream).expect("want");
                        for (r, w) in want.iter().enumerate() {
                            let g = got[s][c * ROWS_OF[s] + r];
                            if g.to_bits() != w.to_bits() {
                                failures.push(format!(
                                    "k {k} m {m} site {s} token {c} row {r}: {g} against the \
                                     one-token launch's {w}"
                                ));
                            }
                        }
                    }
                }
                assert_eq!(gpu.take_fault().expect("fault word"), None);
            }
        }
        println!(
            "norm_proj rows: {} failures; first: {:?}",
            failures.len(),
            failures.first()
        );
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }
}
