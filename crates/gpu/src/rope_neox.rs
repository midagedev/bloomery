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
//!
//! The `_q8` entries ([`RopeNeoxKernels::enqueue_head_norm_neox_append_q8`],
//! [`RopeNeoxKernels::enqueue_head_norm_neox_append_256_q8`]) are the same two
//! launches with the cache append quantized: the norm, the turn and the
//! in-place query and key writes are their f16 twins' bit for bit — the
//! `_256` pair is one body, `norm_partial_append`, over the cache format
//! ([`KvSink`]), and the head-128 q8 entry holds its twin's body with the
//! append replaced, because moving that f16 entry's kernel body into a
//! helper changes the entry's code at the toolchain pin — and
//! each key and value row is stored as Q8_0 blocks in the two-plane layout
//! the weights side owns (`weights::q8_0_planes` over `gguf::quant::Q8Block`) —
//! a codes plane of `head/4` u32 a row (code `j` of a 32-value block in word
//! `j/4`, byte `j%4`) and a scales plane of `head/32` u16 a row, each the
//! block's f16 scale bits — 17/16 bytes a value against the f16 plane's 2.
//! Each warp of a key head's block holds whole 32-value blocks of both rows
//! (thread `t`'s values sit at `base + t` over its two bases, each a multiple
//! of 32), so the block quantization closes inside the warp
//! ([`q8_block_warp`]). A block holding a non-finite value is refused: a NaN
//! scale, zero codes and [`FaultSite::KvQuant`], never a plausible block.
//!
//! [`RopeNeoxKernels::enqueue_head_norm_neox_append_rows`] is the head-128
//! launch over several sequences' f16 caches: token `t` appends into the
//! planes of row `t` of a per-row table ([`RowPlanes`], the launch's
//! grid-constant parameter). Its body, `norm_neox_append`, is the head-128
//! entry's written once over the cache format ([`KvSink`]); that entry keeps
//! its own copy for the reason the q8 entry does.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::f32_to_f16_bits;
use crate::flash_gqa::{ROW_PLANES, RowPlanes, RowTable};
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
/// warps a block, into a cache in the format `S` (that entry's body over
/// [`F16Sink`], `head_norm_neox_append_256_q8`'s over [`Q8Sink`]). Block `b =
/// t·(n_head + n_kv) + h`. The query head `h` of
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
/// SAFETY: the entry's launch contract at `(HEAD, ROT)` over `sink`'s planes,
/// a block of `HEAD/2` threads, and `wsum` this block's `WARPS` f64 of shared
/// memory.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn norm_partial_append<
    const HEAD: usize,
    const ROT: usize,
    const WARPS: usize,
    S: KvSink,
>(
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
    mut sink: S,
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
    // SAFETY: kh < n_kv and p < ctx, so the row is inside the planes; dst +
    // i < m·n_kv·HEAD <= v.len(). One thread per two values of the row, its
    // values its warp's two bases plus its lane, both bases multiples of 32,
    // so each warp holds whole 32-value blocks; the tokens of one launch hold
    // distinct positions, so no two blocks write one row. The value indices
    // are summed here, where the key stores share them: summed alone inside
    // the sink, each would be split into two address adds before inlining,
    // and the f16 entry's code would move.
    unsafe {
        sink.append::<HEAD>(
            kh * ctx as usize + p,
            (i0, i1),
            (y0, y1),
            v,
            (dst + i0, dst + i1),
            fault,
        )
    };
}

/// The norm, NEOX turn and append of [`rope_neox_kernels::head_norm_neox_append`]
/// into a cache in the format `S`: that entry's block geometry (block
/// `b = t·(n_head + n_kv) + h` of `HEAD/2` threads), numeric contract,
/// in-place query and key writes and refusal, with the key and value rows
/// appended through `sink`. The instance over [`F16Sink`] is
/// `head_norm_neox_append_rows`', whose block builds the sink over its
/// token's planes.
///
/// SAFETY: the entry's launch contract at [`HEAD`], `sink` over planes of
/// `n_kv · ctx` rows of [`HEAD`] that no other block writes at this block's
/// rows, a block of `HEAD/2` threads, and `wsum` this block's two f64 of
/// shared memory.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn norm_neox_append<S: KvSink>(
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
    mut sink: S,
    wsum: *mut f64,
) {
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
    if warp::lane_id() == 0 {
        // SAFETY: tid / 32 < 2; one lane per warp writes its slot.
        unsafe { *wsum.add(tid / 32) = acc };
    }
    thread::sync_threads();
    // SAFETY: both slots were written before the barrier above.
    let sum = unsafe { *wsum.add(0) + *wsum.add(1) };
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
    // SAFETY: tid + 64 < 128 <= the gain's length; p < ctx and 2·tid + 1 <
    // 128, so the table pair is inside row p of the table's ctx rows.
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
    // SAFETY: kh < n_kv and p < ctx, so the row is inside the planes; base +
    // tid + 64 < m·n_kv·128 <= v.len(). One thread per pair of the row, each
    // warp's values `32·w + lane` and `32·w + 64 + lane`, whole 32-value
    // blocks; no other block writes the row (this fn's contract).
    unsafe {
        sink.append::<HEAD>(
            kh * ctx as usize + p,
            (tid, tid + THREADS),
            (y0, y1),
            v,
            (base + tid, base + tid + THREADS),
            fault,
        )
    };
}

/// `qdot::nearest_int` on device — ggml's magic-number round to the nearest
/// integer, bit for bit the host's for every finite input in its range (the
/// q8_0 codes' rounding, [`q8_block_warp`]).
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    ((f32::to_bits(val) & 0x007f_ffff) as i32) - 0x0040_0000
}

/// One warp's 32-value Q8_0 block of cache row `row` of `head` values,
/// quantized and written to the row's code and scale planes: lane `lane`
/// holds the block's value `b + lane` as `x`, the base `b` a multiple of 32,
/// so the warp's lanes hold the block whole and every 4-lane group one code
/// word. The rule is the engine's one Q8_0 quantizer's
/// (`model::arch::deepseek2::attn::quantize_q8_0`, the weights side's requant
/// of `wk_b`): the block amax by the warp butterfly, the scale `amax/127`
/// stored as its f16 bits, the reciprocal `127/amax` when the amax is not
/// zero else 0, each code `nearest_int(x·id)` in byte `lane % 4` of the code
/// word `row·head/4 + b/4 + lane/4`, the four lanes of a word combining by
/// two xor shuffles. A block holding a non-finite value has no q8_0 form
/// (`f32::max` drops a NaN, so the amax cannot tell): one ballot refuses it,
/// and it is stored with a NaN scale and zero codes — every value it
/// dequantizes to is NaN, never a plausible number — with `fault` raised.
/// Returns whether the block was refused (warp-uniform).
///
/// SAFETY: `lane < 32`; the code word `row·head/4 + b/4 + lane/4` and the
/// scale `row·head/32 + b/32` lie inside `q` and `d`; all 32 lanes call it
/// together on the same `(row, b)`, and no other warp writes the block.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a device core's flat state, handed on (rust-quality R8)"
)]
unsafe fn q8_block_warp(
    x: f32,
    b: usize,
    row: usize,
    head: usize,
    lane: usize,
    q: &mut DisjointSlice<u32>,
    d: &mut DisjointSlice<u16>,
    fault: FaultSink,
) -> bool {
    let refused = warp::ballot(!x.is_finite()) != 0;
    let amax = warp::reduce_max_f32(x.abs());
    let dv = if refused { f32::NAN } else { amax / 127.0 };
    let id = if refused || amax == 0.0 {
        0.0
    } else {
        127.0 / amax
    };
    let byte = (if refused {
        0u32
    } else {
        // The two's-complement byte of the code: the low byte of the i32,
        // which a negative code already holds.
        (nearest_int(x * id).clamp(-128, 127) as u32) & 0xff
    }) << (8 * (lane % 4));
    // The four lanes of a word group combine their bytes by the two-step
    // butterfly: the first exchange pairs `lane ^ 1`, the second exchanges
    // the pairs' partial words over `lane ^ 2` — both stay inside the group
    // (its lanes share `lane / 4`), and each step shuffles the accumulated
    // word, so every lane of the group ends with all four bytes.
    let mut word = byte | warp::shuffle_xor(byte, 1);
    word |= warp::shuffle_xor(word, 2);
    // SAFETY: `row·head/4 + b/4 + lane/4` and `row·head/32 + b/32` are this
    // lane's alone inside `q` and `d` by this fn's contract (the word among
    // the lanes with this `lane / 4`, the scale lane 0's alone), and both
    // planes were written by no warp of this launch but this one.
    unsafe {
        if lane.is_multiple_of(4) {
            *q.get_unchecked_mut(row * (head / 4) + b / 4 + lane / 4) = word;
        }
        if lane == 0 {
            *d.get_unchecked_mut(row * (head / 32) + b / 32) = f32_to_f16_bits(dv);
        }
    }
    if refused && lane == 0 {
        fault.raise(FaultSite::KvQuant);
    }
    refused
}

/// A KV cache format as the appends write it: the f16 planes ([`F16Sink`])
/// or the Q8_0 planes ([`Q8Sink`]), `n_kv · ctx` rows of `HEAD` values (key
/// head `kh`'s token at position `p` is row `kh·ctx + p`). The append body
/// [`norm_partial_append`] is written once over it; its instances differ
/// only in this one write.
pub(crate) trait KvSink {
    /// Values `i0` and `i1` of key row `row` are `y0` and `y1`, and of value
    /// row `row` `v[x0]` and `v[x1]`.
    ///
    /// # Safety
    /// `row` is a row of the planes, `i0, i1 < HEAD` this thread's alone in
    /// the launch, `x0, x1 < v.len()`; all 32 lanes of the warp call it
    /// together, lane `l` with values `b0 + l` and `b1 + l` of two 32-value
    /// blocks `b0`, `b1` of the row (a Q8_0 block closes inside the warp),
    /// and no other warp writes those blocks.
    unsafe fn append<const HEAD: usize>(
        &mut self,
        row: usize,
        i: (usize, usize),
        y: (f32, f32),
        v: &[f32],
        x: (usize, usize),
        fault: FaultSink,
    );
}

/// The f16 planes: each value rounded once to f16, to nearest even.
pub(crate) struct F16Sink<'a> {
    pub(crate) k: DisjointSlice<'a, u16>,
    pub(crate) v: DisjointSlice<'a, u16>,
}

impl KvSink for F16Sink<'_> {
    #[inline(always)]
    unsafe fn append<const HEAD: usize>(
        &mut self,
        row: usize,
        (i0, i1): (usize, usize),
        (y0, y1): (f32, f32),
        v: &[f32],
        (x0, x1): (usize, usize),
        _fault: FaultSink,
    ) {
        let row = row * HEAD;
        // SAFETY: row + HEAD − 1 < n_kv·ctx·HEAD <= the planes' lengths, and
        // the values are this thread's (the fn's contract).
        unsafe {
            *self.k.get_unchecked_mut(row + i0) = f32_to_f16_bits(y0);
            *self.k.get_unchecked_mut(row + i1) = f32_to_f16_bits(y1);
            *self.v.get_unchecked_mut(row + i0) = f32_to_f16_bits(*v.get_unchecked(x0));
            *self.v.get_unchecked_mut(row + i1) = f32_to_f16_bits(*v.get_unchecked(x1));
        }
    }
}

/// The Q8_0 planes, a codes and a scales plane a side: each warp quantizes
/// its 32-value blocks through [`q8_block_warp`].
pub(crate) struct Q8Sink<'a> {
    pub(crate) kq: DisjointSlice<'a, u32>,
    pub(crate) kd: DisjointSlice<'a, u16>,
    pub(crate) vq: DisjointSlice<'a, u32>,
    pub(crate) vd: DisjointSlice<'a, u16>,
}

impl KvSink for Q8Sink<'_> {
    #[inline(always)]
    unsafe fn append<const HEAD: usize>(
        &mut self,
        row: usize,
        (i0, i1): (usize, usize),
        (y0, y1): (f32, f32),
        v: &[f32],
        (x0, x1): (usize, usize),
        fault: FaultSink,
    ) {
        let lane = warp::lane_id() as usize;
        // SAFETY: the blocks `i0 − lane` and `i1 − lane` of the row are this
        // warp's, their code words and scales inside the planes, and the value
        // reads inside `v` (the fn's contract).
        unsafe {
            q8_block_warp(
                y0,
                i0 - lane,
                row,
                HEAD,
                lane,
                &mut self.kq,
                &mut self.kd,
                fault,
            );
            q8_block_warp(
                y1,
                i1 - lane,
                row,
                HEAD,
                lane,
                &mut self.kq,
                &mut self.kd,
                fault,
            );
            let (w0, w1) = (*v.get_unchecked(x0), *v.get_unchecked(x1));
            q8_block_warp(
                w0,
                i0 - lane,
                row,
                HEAD,
                lane,
                &mut self.vq,
                &mut self.vd,
                fault,
            );
            q8_block_warp(
                w1,
                i1 - lane,
                row,
                HEAD,
                lane,
                &mut self.vq,
                &mut self.vd,
                fault,
            );
        }
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
        // contract is `norm_partial_append`'s at (HEAD_256, ROT_256) over
        // the f16 planes.
        unsafe {
            norm_partial_append::<HEAD_256, ROT_256, WARPS_256, _>(
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
                F16Sink {
                    k: cache_k,
                    v: cache_v,
                },
                SharedArray::as_raw_mut_ptr(&raw mut WSUM4),
            )
        };
    }

    /// [`head_norm_neox_append`] with the append quantized to Q8_0 (the
    /// module doc's `_q8` paragraph): the norm, the turn and the in-place
    /// query and key writes are that entry's bit for bit, and each key and
    /// value row lands as four 32-value blocks in the two planes per side —
    /// `kq`/`kd` and `vq`/`vd`, `n_kv·ctx·32` code words and `n_kv·ctx·4`
    /// scales each ([`q8_block_warp`]). A block holding a non-finite value
    /// raises [`FaultSite::KvQuant`] and is stored with a NaN scale and zero
    /// codes; a position at or past `ctx` is [`head_norm_neox_append`]'s
    /// refusal, appended nothing.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
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
            kq.len() >= n_kv * ctx * 32,
            kd.len() >= n_kv * ctx * 4,
            vq.len() >= n_kv * ctx * 32,
            vd.len() >= n_kv * ctx * 4
        )
    )]
    pub fn head_norm_neox_append_q8(
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
        mut kq: DisjointSlice<u32>,
        mut kd: DisjointSlice<u16>,
        mut vq: DisjointSlice<u32>,
        mut vd: DisjointSlice<u16>,
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
        let row = kh * ctx as usize + p;
        let lane = warp::lane_id() as usize;
        // SAFETY: kh < n_kv and p < ctx, so the row is inside the planes;
        // base + tid + 64 < m·n_kv·128 <= v.len(). The tokens of one launch
        // hold distinct positions, so no two blocks write one row. Each
        // warp's values sit at `32·w` and `32·w + 64` over its lane (a
        // thread's values `tid` and `tid + 64`), whole 32-value blocks.
        unsafe {
            q8_block_warp(y0, tid - lane, row, HEAD, lane, &mut kq, &mut kd, fault);
            q8_block_warp(
                y1,
                tid + THREADS - lane,
                row,
                HEAD,
                lane,
                &mut kq,
                &mut kd,
                fault,
            );
            let (w0, w1) = (
                *v.get_unchecked(base + tid),
                *v.get_unchecked(base + tid + THREADS),
            );
            q8_block_warp(w0, tid - lane, row, HEAD, lane, &mut vq, &mut vd, fault);
            q8_block_warp(
                w1,
                tid + THREADS - lane,
                row,
                HEAD,
                lane,
                &mut vq,
                &mut vd,
                fault,
            );
        }
    }

    /// [`head_norm_neox_append_256`] with the append quantized to Q8_0 (the
    /// module doc's `_q8` paragraph): `norm_partial_append` at `(HEAD_256,
    /// ROT_256)` over [`Q8Sink`], the planes `n_kv·ctx·64` code words and
    /// `n_kv·ctx·8` scales each.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
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
            kq.len() >= n_kv * ctx * 64,
            kd.len() >= n_kv * ctx * 8,
            vq.len() >= n_kv * ctx * 64,
            vd.len() >= n_kv * ctx * 8
        )
    )]
    pub fn head_norm_neox_append_256_q8(
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
        kq: DisjointSlice<u32>,
        kd: DisjointSlice<u16>,
        vq: DisjointSlice<u32>,
        vd: DisjointSlice<u16>,
    ) {
        static mut WSUM4: SharedArray<f64, WARPS_256> = SharedArray::UNINIT;

        // SAFETY: WSUM4 is this block's own shared allocation of WARPS_256
        // slots; the raw form reaches it without a reference. The launch
        // contract is `norm_partial_append`'s at (HEAD_256, ROT_256) over
        // the four Q8_0 planes.
        unsafe {
            norm_partial_append::<HEAD_256, ROT_256, WARPS_256, _>(
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
                Q8Sink { kq, kd, vq, vd },
                SharedArray::as_raw_mut_ptr(&raw mut WSUM4),
            )
        };
    }

    /// [`head_norm_neox_append`] over several sequences' f16 caches: token
    /// `t` appends into the planes of row `t` of `planes` ([`RowPlanes`],
    /// each `n_kv · ctx` rows of 128). `norm_neox_append` over [`F16Sink`]:
    /// that entry's norm, turn, in-place writes, append and refusal, token by
    /// token. The tokens of one sequence hold distinct positions and tokens
    /// of different sequences write different planes, so no two blocks write
    /// one row.
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
            m <= 8,
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            gq.len() >= 128,
            gk.len() >= 128,
            table.len() >= ctx * 128,
            pos.len() >= m
        )
    )]
    pub fn head_norm_neox_append_rows(
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
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        #[grid_constant] planes: &RowPlanes,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;

        let heads = (n_head + n_kv) as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * heads {
            return; // block-uniform
        }
        let plane = n_kv as usize * ctx as usize * HEAD;
        // SAFETY: b / heads < m <= ROW_PLANES (the launch contract), and the
        // launcher built the table over f16 planes of n_kv·ctx·HEAD values
        // that this launch alone writes.
        let (pk, pv) = unsafe { planes.f16_mut(b / heads) };
        // SAFETY: each block writes only its token's rows of its planes
        // (the tokens of one sequence hold distinct positions, of two
        // sequences different planes), so the slices' writes are disjoint.
        // WSUM is this block's own shared allocation; the raw form reaches
        // it without a reference. The launch contract is
        // `norm_neox_append`'s over the token's planes.
        unsafe {
            norm_neox_append(
                gq,
                gk,
                table,
                pos,
                v,
                eps,
                n_head,
                n_kv,
                ctx,
                m,
                fault,
                q,
                k,
                F16Sink {
                    k: DisjointSlice::from_raw_parts(pk, plane),
                    v: DisjointSlice::from_raw_parts(pv, plane),
                },
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
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

/// [`RopeNeoxKernels::enqueue_head_norm_neox_append_rows`]' arguments: those
/// of [`NeoxArgs`] without the cache planes, which the per-row table gives
/// each token.
pub(crate) struct NeoxRowsArgs<'a> {
    pub(crate) q: &'a mut DeviceBuffer<f32>,
    pub(crate) k: &'a mut DeviceBuffer<f32>,
    pub(crate) v: &'a DeviceBuffer<f32>,
    pub(crate) gq: &'a DeviceBuffer<f32>,
    pub(crate) gk: &'a DeviceBuffer<f32>,
    pub(crate) table: &'a DeviceBuffer<f32>,
    pub(crate) pos: &'a DeviceBuffer<u32>,
    pub(crate) eps: f32,
    pub(crate) n_head: usize,
    pub(crate) n_kv: usize,
    pub(crate) ctx: usize,
    pub(crate) m: usize,
    pub(crate) fault: FaultSink,
}

/// The plane lengths of one side of a q8_0 cache of `n_kv` key heads over
/// `ctx` rows of `head` values, in the two-plane layout the weights side
/// owns (`weights::q8_0_planes` over `gguf::quant::Q8Block`): the codes
/// plane's u32 words, `n_kv · ctx · head/4`, and the scales plane's u16
/// scales, `n_kv · ctx · head/32` — the lengths
/// [`RopeNeoxKernels::enqueue_head_norm_neox_append_q8`] and
/// [`RopeNeoxKernels::enqueue_head_norm_neox_append_256_q8`] check their
/// planes against, and the one owner of the layout a body allocates by.
#[must_use]
pub fn q8_plane_lens(head: usize, n_kv: usize, ctx: usize) -> (usize, usize) {
    assert!(
        head.is_multiple_of(32),
        "rope_neox::q8_plane_lens: a q8_0 plane of a {head}-value head (32 a block)"
    );
    (n_kv * ctx * head / 4, n_kv * ctx * head / 32)
}

/// [`NeoxArgs`]'s q8_0 form: the same inputs, the cache the four planes of
/// [`enqueue_head_norm_neox_append_q8`]'s append — per side the codes and
/// the scales of `n_kv · ctx` rows of [`HEAD`] values
/// ([`q8_plane_lens`]).
pub struct NeoxQ8Args<'a> {
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
    pub kq: &'a mut DeviceBuffer<u32>,
    pub kd: &'a mut DeviceBuffer<u16>,
    pub vq: &'a mut DeviceBuffer<u32>,
    pub vd: &'a mut DeviceBuffer<u16>,
}

/// [`PartialNeoxArgs`]'s q8_0 form: the same inputs, the cache the four
/// planes of [`enqueue_head_norm_neox_append_256_q8`]'s append — per side
/// the codes and the scales of `n_kv · ctx` rows of [`HEAD_256`] values
/// ([`q8_plane_lens`]).
pub struct PartialNeoxQ8Args<'a> {
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
    pub kq: &'a mut DeviceBuffer<u32>,
    pub kd: &'a mut DeviceBuffer<u16>,
    pub vq: &'a mut DeviceBuffer<u32>,
    pub vd: &'a mut DeviceBuffer<u16>,
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
        let module = unsafe { crate::shared_module!(rope_neox_kernels, ctx)? };
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

    /// [`RopeNeoxKernels::enqueue_head_norm_neox_append`] with the append
    /// quantized to Q8_0 (the module doc's `_q8` paragraph): the same norm,
    /// turn and in-place writes, and each key and value row stored as four
    /// 32-value blocks in `args`' four planes — per side the codes and the
    /// scales ([`q8_plane_lens`]), the quantization rule and refusal
    /// [`q8_block_warp`]'s. Asynchronous, allocation-free, capturable.
    pub fn enqueue_head_norm_neox_append_q8(
        &self,
        stream: &CudaStream,
        args: NeoxQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append_q8";
        let NeoxQ8Args {
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
            kq,
            kd,
            vq,
            vd,
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
        let (words, scales) = q8_plane_lens(HEAD, n_kv, ctx);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("gq", gq.len(), HEAD),
            ("gk", gk.len(), HEAD),
            ("table", table.len(), ctx * HEAD),
            ("pos", pos.len(), m),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
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
            .prepare_head_norm_neox_append_q8(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.head_norm_neox_append_q8(
            stream, &prep, gq, gk, table, pos, v, eps, n_head, n_kv, ctx, m, fault, q, k, kq, kd,
            vq, vd,
        )?;
        Ok(())
    }

    /// [`RopeNeoxKernels::enqueue_head_norm_neox_append_256`] with the append
    /// quantized to Q8_0 (the module doc's `_q8` paragraph): the same norm,
    /// partial turn and in-place writes, and each key and value row stored
    /// as eight 32-value blocks in `args`' four planes — per side the codes
    /// and the scales ([`q8_plane_lens`] at [`HEAD_256`]), the quantization
    /// rule and refusal [`q8_block_warp`]'s. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_head_norm_neox_append_256_q8(
        &self,
        stream: &CudaStream,
        args: PartialNeoxQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append_256_q8";
        let PartialNeoxQ8Args {
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
            kq,
            kd,
            vq,
            vd,
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
        let (words, scales) = q8_plane_lens(HEAD_256, n_kv, ctx);
        let lens = [
            ("qg", qg.len(), m * n_head * 2 * HEAD_256),
            ("q", q.len(), m * n_head * HEAD_256),
            ("k", k.len(), m * n_kv * HEAD_256),
            ("v", v.len(), m * n_kv * HEAD_256),
            ("gq", gq.len(), HEAD_256),
            ("gk", gk.len(), HEAD_256),
            ("table", table.len(), ctx * ROT_256),
            ("pos", pos.len(), m),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
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
            .prepare_head_norm_neox_append_256_q8(LaunchConfig1D::new(grid, THREADS_256_U32, 0))?;
        self.module.head_norm_neox_append_256_q8(
            stream, &prep, gq, gk, table, pos, qg, v, eps, n_head, n_kv, ctx, m, fault, q, k, kq,
            kd, vq, vd,
        )?;
        Ok(())
    }

    /// [`RopeNeoxKernels::enqueue_head_norm_neox_append`] over several
    /// sequences' f16 caches: token `t` appends into the planes `planes`
    /// gives it ([`RowTable`]), its other inputs `args`' rows as that
    /// launch's. One block per (token, head), `m·(n_head + n_kv)` blocks of
    /// 64 threads; each token's arithmetic and refusal are that launch's.
    /// The tokens of one sequence must hold distinct positions below `ctx`;
    /// a position `>= ctx` raises [`FaultSite::CachePos`] on `args.fault`,
    /// leaves the token's heads NaN and is not appended. A table of other
    /// than `m` rows, more than [`ROW_PLANES`], or a plane shorter than
    /// `n_kv · ctx · HEAD` is refused by name. Asynchronous,
    /// allocation-free, capturable.
    pub(crate) fn enqueue_head_norm_neox_append_rows(
        &self,
        stream: &CudaStream,
        planes: &RowTable<'_>,
        args: NeoxRowsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append_rows";
        let NeoxRowsArgs {
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
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ROW_PLANES {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= {ROW_PLANES}, got \
                     n_head={n_head} n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let rows = planes.planes(what, m, n_kv * ctx * HEAD)?;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("gq", gq.len(), HEAD),
            ("gk", gk.len(), HEAD),
            ("table", table.len(), ctx * HEAD),
            ("pos", pos.len(), m),
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
            .prepare_head_norm_neox_append_rows(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        // SAFETY: the table's rows are `planes`' puts, each an f16 plane pair
        // of at least n_kv·ctx·HEAD values (checked above), borrowed by the
        // table while this enqueue runs; a graph that captures the launch
        // replays it over the same planes, which the caller keeps alive and
        // in place while the graph lives, as for any buffer it captured.
        unsafe {
            self.module.head_norm_neox_append_rows(
                stream, &prep, gq, gk, table, pos, v, eps, n_head, n_kv, ctx, m, fault, q, k, rows,
            )?;
        }
        Ok(())
    }
}
