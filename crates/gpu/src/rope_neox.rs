//! One launch per layer for the query and key heads of a grouped-query
//! attention layer with a per-head norm and a NEOX rope over the whole head:
//! each head's RMS norm by its own gain vector (`attn_q_norm` /
//! `attn_k_norm`), the NEOX turn of pair `(i, i + HEAD/2)` by pair `i` of
//! the rope table's row at the token's position, and for the key heads the
//! append of the turned key and of the value head to the layer's f16 cache
//! planes at that position. The table holds a row for every cache position,
//! so the one check a position gets — below the planes' rows — covers the
//! table read and the append.
//!
//! The planes are head-major, `[n_kv][ctx][HEAD]` u16: key head `h` of the
//! token at position `p` is row `h·ctx + p`, so a flash block that walks one
//! key head reads its rows contiguously.
//!
//! Numeric contract, per head of [`HEAD`] values (thread `t` owns values `t`
//! and `t + HEAD/2`, which are also NEOX pair `t`):
//! - each value squared in f32, the two squares added in f64, the 32 lanes
//!   of a warp by the xor butterfly (16, 8, 4, 2, 1) in f64, the two warps'
//!   sums added in f64 (warp 0 first);
//! - the mean `(sum / HEAD) as f32` (an f64 divide, rounded once to f32),
//!   the scale `1 / sqrt(mean + eps)` in f32, each op rounded on its own;
//! - the normalized value `(scale · gain) · x`;
//! - the turn through [`neox_pair`]: `y0 = fma(x0, c, −(x1·s))`,
//!   `y1 = fma(x0, s, x1·c)`, the inner products rounded on their own —
//!   the rounding ik's compiled CPU NEOX loop produces (its source writes
//!   `x0·c − x1·s` and `x0·s + x1·c`), so on the same normalized values the
//!   turn is ik's bit for bit;
//! - the cache rows rounded once to f16, to nearest even.
//!
//! ik's CPU norm sums the same f32 squares serially in f64; the two f64 sums
//! of 128 terms differ by far less than one f32 ulp of the mean, so the
//! rounded means agree unless the exact mean sits within that distance of an
//! f32 rounding boundary. The gates print how many values agree bit for bit.
//!
//! [`RopeNeoxKernels::enqueue_head_norm_neox_append_256`] is the same launch
//! for heads of [`HEAD_256`] values whose first [`ROT_256`] turn (a partial
//! NEOX rope, pairs `(i, i + ROT_256/2)`), 128 threads a head, the four warps'
//! sums added in warp order; it reads the query from the q+gate rows at head
//! stride `2·HEAD_256` and writes it to its own rows.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::f32_to_f16_bits;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values per head, and twice the block width: thread `t` owns a head's
/// values `t` and `t + HEAD/2`.
pub const HEAD: usize = 128;

/// Threads per block, one per NEOX pair.
const THREADS: usize = HEAD / 2;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && THREADS == 64);

/// NEOX pair `(x0, x1)` turned by `(c, s)` in the module's form: each inner
/// product rounded on its own (`mul.rn`, which the compiler never contracts),
/// then one fused multiply-add.
#[inline(always)]
pub fn neox_pair(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    (
        fma_rn_f32(x0, c, -mul_rn_f32(x1, s)),
        fma_rn_f32(x0, s, mul_rn_f32(x1, c)),
    )
}

/// Values per head of the partial-rope instance (Qwen3.6's full-attention
/// layers), and the values at the head of each row that turn: NEOX pairs
/// `(i, i + ROT_256/2)` for `i < ROT_256/2`, every other value passed through
/// normalized. The rope table's row is `ROT_256` f32, `ROT_256/2` pairs.
pub const HEAD_256: usize = 256;
pub const ROT_256: usize = 64;
/// Threads of its block, two values each, and its warps.
const THREADS_256: usize = HEAD_256 / 2;
const THREADS_256_U32: u32 = THREADS_256 as u32;
const WARPS_256: usize = THREADS_256 / 32;
const _: () = assert!(THREADS_256_U32 as usize == THREADS_256 && THREADS_256 == 128);
// The entry's launch contract spells HEAD_256 and ROT_256 out as 256 and 64.
const _: () = assert!(HEAD_256 == 256 && ROT_256 == 64);

/// The values thread `t` owns in a head of `HEAD` values whose first `ROT`
/// turn: NEOX pair `(t, t + ROT/2)` for `t < ROT/2`, else the pass-through
/// pair `(t + ROT/2, t + HEAD/2)` — every value once over `HEAD/2` threads.
#[inline(always)]
pub const fn owned_pair(head: usize, rot: usize, t: usize) -> (usize, usize) {
    if t < rot / 2 {
        (t, t + rot / 2)
    } else {
        (t + rot / 2, t + head / 2)
    }
}

/// The norm, partial turn and append of [`rope_neox_kernels::head_norm_neox_append_256`]
/// over heads of `HEAD` values, the first `ROT` turned, `WARPS = HEAD/64`
/// warps a block. Block `b = t·(n_head + n_kv) + h`. The query head `h` of
/// token `t` is read from the q+gate rows at `qg[t·n_head·2·HEAD + h·2·HEAD
/// ..]` (each head's `2·HEAD` values are the query, then its gate) and
/// written to `q[(t·n_head + h)·HEAD ..]`; key head `j = h − n_head` is
/// read and written in place in `k[(t·n_kv + j)·HEAD ..]` and appended with
/// value head `j` of `v` at plane row `j·ctx + pos[t]`. The numeric contract
/// is the module doc's with the sum over `WARPS` warps in warp order, the
/// mean `(sum / HEAD) as f32`, and the turn on pairs `(i, i + ROT/2)` by
/// table pair `i` of row `pos[t]` (`table[pos[t]·ROT ..]`); a pass-through
/// value is `(scale · gain) · x`. A position at or past `ctx` raises
/// [`FaultSite::CachePos`], writes NaN over the head's output and appends
/// nothing.
///
/// SAFETY: the entry's launch contract at `(HEAD, ROT)`, a block of
/// `HEAD/2` threads, and `wsum` this block's `WARPS` f64 of shared memory.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn norm_partial_append<const HEAD: usize, const ROT: usize, const WARPS: usize>(
    gq: &[f32],
    gk: &[f32],
    table: &[f32],
    pos: &[u32],
    qg: &[f32],
    v: &[f32],
    eps: f32,
    n_head: u32,
    n_kv: u32,
    ctx: u32,
    m: u32,
    fault: FaultSink,
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut cache_k: DisjointSlice<u16>,
    mut cache_v: DisjointSlice<u16>,
    wsum: *mut f64,
) {
    const { assert!(WARPS * 64 == HEAD && ROT.is_multiple_of(2) && ROT <= HEAD && ROT / 2 <= 32) };
    let heads = (n_head + n_kv) as usize;
    let b = thread::blockIdx_x() as usize;
    if b >= m as usize * heads {
        return; // block-uniform
    }
    let t = b / heads;
    let h = b - t * heads;
    let tid = thread::threadIdx_x() as usize;
    let is_q = h < n_head as usize;
    let kh = h.wrapping_sub(n_head as usize);
    let (i0, i1) = owned_pair(HEAD, ROT, tid);
    // The head's first value in its source and in its output.
    let (src, dst) = if is_q {
        (
            (t * n_head as usize + h) * 2 * HEAD,
            (t * n_head as usize + h) * HEAD,
        )
    } else {
        let at = (t * n_kv as usize + kh) * HEAD;
        (at, at)
    };
    // SAFETY: t < m <= pos.len() by the launch contract.
    let p = unsafe { *pos.get_unchecked(t) } as usize;

    // SAFETY: i0, i1 < HEAD; a query source is inside q+gate row t (m·n_head
    // ·2·HEAD values by the launch contract), a key inside k (m·n_kv·HEAD).
    let (x0, x1) = unsafe {
        if is_q {
            (*qg.get_unchecked(src + i0), *qg.get_unchecked(src + i1))
        } else {
            (
                *k.get_unchecked_mut(src + i0),
                *k.get_unchecked_mut(src + i1),
            )
        }
    };
    let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
    acc += warp::shuffle_xor_f64(acc, 16);
    acc += warp::shuffle_xor_f64(acc, 8);
    acc += warp::shuffle_xor_f64(acc, 4);
    acc += warp::shuffle_xor_f64(acc, 2);
    acc += warp::shuffle_xor_f64(acc, 1);
    if warp::lane_id() == 0 {
        // SAFETY: tid / 32 < WARPS; one lane per warp writes its slot.
        unsafe { *wsum.add(tid / 32) = acc };
    }
    thread::sync_threads();
    // SAFETY: every slot was written before the barrier above.
    let mut sum = unsafe { *wsum };
    for w in 1..WARPS {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: w < WARPS.
        sum += unsafe { *wsum.add(w) };
    }
    let mean = (sum / HEAD as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();

    if p >= ctx as usize {
        if tid == 0 {
            fault.raise(FaultSite::CachePos);
        }
        // SAFETY: the output positions of this thread's two values.
        unsafe {
            if is_q {
                *q.get_unchecked_mut(dst + i0) = f32::NAN;
                *q.get_unchecked_mut(dst + i1) = f32::NAN;
            } else {
                *k.get_unchecked_mut(dst + i0) = f32::NAN;
                *k.get_unchecked_mut(dst + i1) = f32::NAN;
            }
        }
        return; // block-uniform: p is the token's
    }
    // SAFETY: i0, i1 < HEAD <= the gain's length.
    let (g0, g1) = unsafe {
        let g = if is_q { gq } else { gk };
        (*g.get_unchecked(i0), *g.get_unchecked(i1))
    };
    let n0 = (scale * g0) * x0;
    let n1 = (scale * g1) * x1;
    let (y0, y1) = if tid < ROT / 2 {
        // SAFETY: p < ctx and 2·tid + 1 < ROT: the pair is inside row p of
        // the table's ctx rows of ROT.
        let (c, s) = unsafe {
            (
                *table.get_unchecked(p * ROT + 2 * tid),
                *table.get_unchecked(p * ROT + 2 * tid + 1),
            )
        };
        neox_pair(n0, n1, c, s)
    } else {
        (n0, n1)
    };

    if is_q {
        // SAFETY: the output positions of this thread's two values.
        unsafe {
            *q.get_unchecked_mut(dst + i0) = y0;
            *q.get_unchecked_mut(dst + i1) = y1;
        }
        return;
    }
    // SAFETY: the positions read above; this thread owns them.
    unsafe {
        *k.get_unchecked_mut(dst + i0) = y0;
        *k.get_unchecked_mut(dst + i1) = y1;
    }
    let row = (kh * ctx as usize + p) * HEAD;
    // SAFETY: kh < n_kv and p < ctx, so row + HEAD − 1 < n_kv·ctx·HEAD <= the
    // planes' lengths; dst + i < m·n_kv·HEAD <= v.len(). One thread per two
    // values of the row; the tokens of one launch hold distinct positions,
    // so no two blocks write one row.
    unsafe {
        *cache_k.get_unchecked_mut(row + i0) = f32_to_f16_bits(y0);
        *cache_k.get_unchecked_mut(row + i1) = f32_to_f16_bits(y1);
        *cache_v.get_unchecked_mut(row + i0) = f32_to_f16_bits(*v.get_unchecked(dst + i0));
        *cache_v.get_unchecked_mut(row + i1) = f32_to_f16_bits(*v.get_unchecked(dst + i1));
    }
}

#[cuda_module]
mod rope_neox_kernels {
    use super::*;

    /// The per-head norm, NEOX turn and cache append of `m` tokens. Block
    /// `b = t·(n_head + n_kv) + h`: for `h < n_head` query head `h` of token
    /// `t` is normalized by `gq`, turned, and written back in place in `q`;
    /// for `h = n_head + j` key head `j` is normalized by `gk`, turned,
    /// written back in place in `k` and rounded to f16 into `cache_k` row
    /// `j·ctx + pos[t]`, and value head `j` of `v` is rounded into `cache_v`
    /// at the same row. Token `t` turns by the table's row `pos[t]`,
    /// `table[pos[t]·HEAD ..]`, `[cos_0, sin_0, …]` for the 64 pairs. The
    /// branch is on the block index, so no warp splits across it. `pos[t] <
    /// ctx` is host-checked where the host can see it and bounds-checked here
    /// where it cannot: a position read from device memory at or past the
    /// planes has no row in the planes or the table, so every block of the
    /// token raises [`FaultSite::CachePos`] on `fault`, writes NaN over its
    /// head in place and appends nothing — no read or write outside the
    /// buffers, and no stale row passes for the token's. The position is
    /// loaded first and checked after the norm, so its load overlaps the
    /// head's.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            gq.len() >= 128,
            gk.len() >= 128,
            table.len() >= ctx * 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * ctx * 128,
            cache_v.len() >= n_kv * ctx * 128
        )
    )]
    pub fn head_norm_neox_append(
        gq: &[f32],
        gk: &[f32],
        table: &[f32],
        pos: &[u32],
        v: &[f32],
        eps: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;

        let heads = (n_head + n_kv) as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * heads {
            return; // block-uniform
        }
        let t = b / heads;
        let h = b - t * heads;
        let tid = thread::threadIdx_x() as usize;
        let is_q = h < n_head as usize;
        let kh = h.wrapping_sub(n_head as usize);
        let base = if is_q {
            (t * n_head as usize + h) * HEAD
        } else {
            (t * n_kv as usize + kh) * HEAD
        };
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;

        // SAFETY: base + tid + 64 < (t·n + h + 1)·128 <= m·n·128, inside the
        // head's buffer by the launch contract; one thread per value pair.
        let (x0, x1) = unsafe {
            if is_q {
                (
                    *q.get_unchecked_mut(base + tid),
                    *q.get_unchecked_mut(base + tid + THREADS),
                )
            } else {
                (
                    *k.get_unchecked_mut(base + tid),
                    *k.get_unchecked_mut(base + tid + THREADS),
                )
            }
        };
        let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
        acc += warp::shuffle_xor_f64(acc, 16);
        acc += warp::shuffle_xor_f64(acc, 8);
        acc += warp::shuffle_xor_f64(acc, 4);
        acc += warp::shuffle_xor_f64(acc, 2);
        acc += warp::shuffle_xor_f64(acc, 1);
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Two slots, one per warp, written before the barrier that publishes
        // them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 2; one lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: both slots were written before the barrier above.
        let sum = unsafe { *ws.add(0) + *ws.add(1) };
        let mean = (sum / HEAD as f64) as f32;
        let scale = 1.0 / (mean + eps).sqrt();

        if p >= ctx as usize {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            // SAFETY: the positions read above; this thread owns them.
            unsafe {
                if is_q {
                    *q.get_unchecked_mut(base + tid) = f32::NAN;
                    *q.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                } else {
                    *k.get_unchecked_mut(base + tid) = f32::NAN;
                    *k.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                }
            }
            return; // block-uniform: p is the token's
        }
        // SAFETY: tid + 64 < 128 <= the gain's length; p < ctx and 2·tid + 1
        // < 128, so the table pair is inside row p of the table's ctx rows.
        let (g0, g1, c, s) = unsafe {
            let g = if is_q { gq } else { gk };
            (
                *g.get_unchecked(tid),
                *g.get_unchecked(tid + THREADS),
                *table.get_unchecked(p * HEAD + 2 * tid),
                *table.get_unchecked(p * HEAD + 2 * tid + 1),
            )
        };
        let n0 = (scale * g0) * x0;
        let n1 = (scale * g1) * x1;
        let (y0, y1) = neox_pair(n0, n1, c, s);

        if is_q {
            // SAFETY: the positions read above; this thread owns them.
            unsafe {
                *q.get_unchecked_mut(base + tid) = y0;
                *q.get_unchecked_mut(base + tid + THREADS) = y1;
            }
            return;
        }
        // SAFETY: the positions read above; this thread owns them.
        unsafe {
            *k.get_unchecked_mut(base + tid) = y0;
            *k.get_unchecked_mut(base + tid + THREADS) = y1;
        }
        let row = (kh * ctx as usize + p) * HEAD;
        // SAFETY: kh < n_kv and p < ctx, so row + 127 < n_kv·ctx·128 <= the
        // planes' lengths; base + tid + 64 < m·n_kv·128 <= v.len(). One
        // thread per pair of the row; the tokens of one launch hold distinct
        // positions (the launch contract; the engine's embedding launch
        // writes consecutive ones), so no two blocks write one row.
        unsafe {
            *cache_k.get_unchecked_mut(row + tid) = f32_to_f16_bits(y0);
            *cache_k.get_unchecked_mut(row + tid + THREADS) = f32_to_f16_bits(y1);
            *cache_v.get_unchecked_mut(row + tid) = f32_to_f16_bits(*v.get_unchecked(base + tid));
            *cache_v.get_unchecked_mut(row + tid + THREADS) =
                f32_to_f16_bits(*v.get_unchecked(base + tid + THREADS));
        }
    }

    /// The norm, partial NEOX turn and append of `m` tokens with heads of
    /// [`HEAD_256`] values, the first [`ROT_256`] turned
    /// (`norm_partial_append`): the query heads read from the q+gate rows
    /// `qg` (`2·256` values a head, the query first) and written to `q`,
    /// the key heads normalized and turned in place in `k` and appended with
    /// `v`. The rope table holds [`ROT_256`] f32 per position.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            qg.len() >= m * n_head * 512,
            q.len() >= m * n_head * 256,
            k.len() >= m * n_kv * 256,
            v.len() >= m * n_kv * 256,
            gq.len() >= 256,
            gk.len() >= 256,
            table.len() >= ctx * 64,
            pos.len() >= m,
            cache_k.len() >= n_kv * ctx * 256,
            cache_v.len() >= n_kv * ctx * 256
        )
    )]
    pub fn head_norm_neox_append_256(
        gq: &[f32],
        gk: &[f32],
        table: &[f32],
        pos: &[u32],
        qg: &[f32],
        v: &[f32],
        eps: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        cache_k: DisjointSlice<u16>,
        cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM4: SharedArray<f64, WARPS_256> = SharedArray::UNINIT;

        // SAFETY: WSUM4 is this block's own shared allocation of WARPS_256
        // slots; the raw form reaches it without a reference. The launch
        // contract is `norm_partial_append`'s at (HEAD_256, ROT_256).
        unsafe {
            norm_partial_append::<HEAD_256, ROT_256, WARPS_256>(
                gq,
                gk,
                table,
                pos,
                qg,
                v,
                eps,
                n_head,
                n_kv,
                ctx,
                m,
                fault,
                q,
                k,
                cache_k,
                cache_v,
                SharedArray::as_raw_mut_ptr(&raw mut WSUM4),
            )
        };
    }
}

/// [`RopeNeoxKernels::enqueue_head_norm_neox_append`]'s arguments: `m`
/// tokens of `n_head` query heads in `q` and `n_kv` key and value heads in
/// `k` and `v` (token-major, head after head, [`HEAD`] values each), the
/// gains, the rope table on the device — a row of [`HEAD`] f32 for each of
/// at least `ctx` positions, `RopeTable::push` over a [`HEAD`]-wide spec —
/// and the `m` tokens' positions, the sink a position past the planes raises
/// on, and the layer's two cache planes of `n_kv · ctx` rows of [`HEAD`] f16.
pub struct NeoxArgs<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a DeviceBuffer<f32>,
    pub gq: &'a DeviceBuffer<f32>,
    pub gk: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub cache_k: &'a mut DeviceBuffer<u16>,
    pub cache_v: &'a mut DeviceBuffer<u16>,
}

/// [`RopeNeoxKernels::enqueue_head_norm_neox_append_256`]'s arguments: `m`
/// tokens' q+gate rows `qg` (`n_head` heads of `2·`[`HEAD_256`] values, the
/// query first), the query output `q` (`n_head` heads of [`HEAD_256`]), the
/// `n_kv` key heads in `k` (turned in place) and value heads in `v`, the
/// gains, the rope table ([`ROT_256`] f32 per position, at least `ctx`
/// rows), the positions, the sink, and the layer's two cache planes of
/// `n_kv · ctx` rows of [`HEAD_256`] f16.
pub struct PartialNeoxArgs<'a> {
    pub qg: &'a DeviceBuffer<f32>,
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a DeviceBuffer<f32>,
    pub gq: &'a DeviceBuffer<f32>,
    pub gk: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub cache_k: &'a mut DeviceBuffer<u16>,
    pub cache_v: &'a mut DeviceBuffer<u16>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct RopeNeoxKernels {
    module: rope_neox_kernels::LoadedModule,
}

impl RopeNeoxKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RopeNeoxKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { rope_neox_kernels::load(ctx)? };
        Ok(RopeNeoxKernels { module })
    }

    /// Enqueue the norm, turn and append of `args.m` tokens: one block per
    /// (token, head), `m·(n_head + n_kv)` blocks of 64 threads, token `t`
    /// turned by the table's row `pos[t]`. The tokens of one launch must hold
    /// distinct positions below `ctx` (the caller's positions are
    /// consecutive); a position `>= ctx` raises [`FaultSite::CachePos`] on
    /// `args.fault`, leaves the token's heads NaN and is not appended.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_head_norm_neox_append(
        &self,
        stream: &CudaStream,
        args: NeoxArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append";
        let NeoxArgs {
            q,
            k,
            v,
            gq,
            gk,
            table,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            cache_k,
            cache_v,
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                     n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let plane = n_kv * ctx * HEAD;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("gq", gq.len(), HEAD),
            ("gk", gk.len(), HEAD),
            ("table", table.len(), ctx * HEAD),
            ("pos", pos.len(), m),
            ("cache_k", cache_k.len(), plane),
            ("cache_v", cache_v.len(), plane),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_head_norm_neox_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.head_norm_neox_append(
            stream, &prep, gq, gk, table, pos, v, eps, n_head, n_kv, ctx, m, fault, q, k, cache_k,
            cache_v,
        )?;
        Ok(())
    }

    /// Enqueue the norm, partial turn and append of `args.m` tokens with
    /// heads of [`HEAD_256`] values, the first [`ROT_256`] turned: one block
    /// per (token, head), `m·(n_head + n_kv)` blocks of 128 threads. The
    /// query heads come from the q+gate rows `args.qg` (`m·n_head·512` f32,
    /// head `h` of token `t` at `(t·n_head + h)·512`, its gate the next 256)
    /// and go to `args.q` (`m·n_head·256`, token-major); `args.table` holds
    /// [`ROT_256`] f32 for each of at least `ctx` positions (`RopeTable::push`
    /// over a [`ROT_256`]-wide spec). Positions as in
    /// [`RopeNeoxKernels::enqueue_head_norm_neox_append`]: distinct, below
    /// `ctx`; one at or past it raises [`FaultSite::CachePos`] on
    /// `args.fault`, leaves the token's heads NaN and is not appended.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_head_norm_neox_append_256(
        &self,
        stream: &CudaStream,
        args: PartialNeoxArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append_256";
        let PartialNeoxArgs {
            qg,
            q,
            k,
            v,
            gq,
            gk,
            table,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            cache_k,
            cache_v,
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                     n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let plane = n_kv * ctx * HEAD_256;
        let lens = [
            ("qg", qg.len(), m * n_head * 2 * HEAD_256),
            ("q", q.len(), m * n_head * HEAD_256),
            ("k", k.len(), m * n_kv * HEAD_256),
            ("v", v.len(), m * n_kv * HEAD_256),
            ("gq", gq.len(), HEAD_256),
            ("gk", gk.len(), HEAD_256),
            ("table", table.len(), ctx * ROT_256),
            ("pos", pos.len(), m),
            ("cache_k", cache_k.len(), plane),
            ("cache_v", cache_v.len(), plane),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_head_norm_neox_append_256(LaunchConfig1D::new(grid, THREADS_256_U32, 0))?;
        self.module.head_norm_neox_append_256(
            stream, &prep, gq, gk, table, pos, qg, v, eps, n_head, n_kv, ctx, m, fault, q, k,
            cache_k, cache_v,
        )?;
        Ok(())
    }
}
