//! The two cache appends of an absorbed MLA layer with no rope and a k-pool
//! indexer (GLM-5.3-Flash's MLA layers). Both read one token's row of the
//! layer's stacked projection output (`[m][stride]` f32, each part at its
//! own column offset) and write one f16 row at the token's position of a
//! dense cache (position `p` is row `p`):
//! - [`LatentKernels::enqueue_latent_append`]: the RMS norm of the
//!   [`LATENT`]-value latent with its gain, rounded to f16 — the row the
//!   attention reads as both key and value;
//! - [`LatentKernels::enqueue_index_key_append`]: the LayerNorm (mean,
//!   variance, weight, bias) of the [`INDEX_HEAD`]-value indexer key, then
//!   the raw [`INDEX_HEAD`]-value pool gate beside it, `[key; gate]` rounded
//!   to f16 — the row the pool selector reads.
//!
//! A third append serves an absorbed MLA layer whose keys carry a rope-free
//! extra part beside the latent (Raon-1's NoPE MLA):
//! [`LatentKernels::enqueue_latent_extra_append`] writes one f16 row of
//! `extra + LATENT` values, the token's raw `extra` values first, then its
//! latent normed as [`LatentKernels::enqueue_latent_append`] norms it — the
//! `[rope | latent]` row order `crate::flash`'s latent attention reads, with
//! no rotation.
//!
//! A fourth entry reads the index cache back: [`LatentKernels::enqueue_index_pool`]
//! writes the key of every pool of [`POOL`] positions a token completes,
//! once, into the layer's pool plane (`[pools][INDEX_HEAD]` f16, pool `j` the
//! positions `4j .. 4j + 3`): per channel `d`, `Σ_i softmax_i(gate_i[d] +
//! ape[i][d]) · key_i[d]` over the four members `i`, from their f16 rows —
//! the pooled key the scores read (`kpool`).
//!
//! A fifth serves a selected attention over a cache the dense flash reads
//! whole: [`LatentKernels::enqueue_kv_gather_sel`] copies the rows a
//! selection list picks into a contiguous f16 scratch, each token's count
//! read on the device, so the flash runs unchanged over the picked rows
//! (Mistral Large 4's roped layers).
//!
//! Two more serve a k-pool selector whose pooled key is a gated pool of a
//! projection of the normed token key, normed again and turned (Hy4-flash's
//! DSA layers, llama.cpp's `build_kpool_select`):
//! - [`LatentKernels::enqueue_index_key_rms_append`]: the RMS norm of the
//!   [`INDEX_HEAD`]-value raw key with its gain, then its value and gate
//!   projections (`indexer_compressor_value` and `_gate`, q8_0), `[value;
//!   gate]` rounded to f16 into one index row — slot `p mod ring` of a ring
//!   of index rows, a dense cache when the ring has a row per position;
//! - [`LatentKernels::enqueue_index_pool_norm_rope`]: the pool key of every
//!   pool a token completes, the gated pool of its four rows' values as
//!   `enqueue_index_pool` forms it, then its RMS norm with the pool norm's
//!   gain and the NORM turn of its last [`ROT`] values at the pool's first
//!   position, rounded to f16 into the pool plane.
//!
//! Numeric contract (the host rules [`latent_append_host`],
//! [`index_key_append_host`], [`index_pool_host`],
//! [`index_key_rms_append_host`] and [`index_pool_norm_rope_host`] are these
//! lists, op for op). Latent, one block
//! of [`BLOCK`] threads per token, thread `t` owning values `t + 128·i`:
//! - each value squared in f32, thread `t`'s four squares added in f64 in
//!   `i` order, the 32 lanes of a warp by the xor butterfly (16, 8, 4, 2, 1)
//!   in f64, the four warps' sums added in f64 in warp order;
//! - the mean `(sum / LATENT) as f32`, the scale `1 / sqrt(mean + eps)` in
//!   f32; the value `(scale · gain) · x`, each product rounded.
//!
//! Index key, one warp per token, lane `l` owning values `l + 32·j`:
//! - the values added in f64 (lane sums in `j` order, then the butterfly),
//!   the mean `(sum / INDEX_HEAD) as f32`; each value's deviation `x − mean`
//!   in f32, squared in f32, added in f64 the same way, the variance
//!   `(sum / INDEX_HEAD) as f32`, the scale `1 / sqrt(variance + eps)` in
//!   f32; the value `((x − mean) · scale) · w + b`, each op rounded.
//!
//! Pool, one warp per completing token, lane `l` owning channels `l + 32·j`:
//! each member's key and gate widened from f16; `z_i = gate_i + ape_i`; `mx =
//! max(max(max(z0, z1), z2), z3)`; `e_i = expf_ik(z_i − mx)`
//! ([`crate::linear::expf_ik`]); `s = ((e0 + e1) + e2) + e3`; `w_i = e_i ·
//! (1 / s)`; the key `((w0·k0 + w1·k1) + w2·k2) + w3·k3`, each op rounded,
//! then rounded to f16 once.
//!
//! Forced-tail key, one block of [`KEY_BLOCK`] per token, every warp
//! norming the key with lane `l` owning values `4l .. 4l + 3`:
//! - each value squared in f32, the lane's four squares added in f64 in
//!   order, the lanes by the xor butterfly in f64; the mean `(sum /
//!   INDEX_HEAD) as f32`, the scale `elem::rms_scale_of_mean` (NaN where it
//!   is not finite and positive); the normed value `(x · scale) · gain`, each
//!   product rounded — ggml's `rms_norm` then `mul`;
//! - each projection row q8f32's word walk at k = 128: lane `l` the four
//!   terms of code word `l`, `fma(d·v, x, f)` in byte order from 0, the lane
//!   sums by the xor butterfly (16, 8, 4, 2, 1) — `q8_0_gemv`'s bits over
//!   the normed key; then rounded to f16.
//!
//! Forced-tail pool, one warp per completing token, lane `l` owning channels
//! `4l .. 4l + 3`: the pool's value per channel as the pool above (the value
//! in the key's place); the norm as the key's; then for the lanes past
//! `INDEX_HEAD − ROT` the pairs `(4l, 4l + 1)` and `(4l + 2, 4l + 3)` turned
//! by `rope_table::rope_pair_rn` with table pairs `2l − 32` and `2l − 31` of
//! row `4j`; then rounded to f16 once.
//!
//! This is ik's CPU order (`fused_rms_norm`, `norm`, then `mul` and `add`)
//! but for the f64 sums, which ik adds serially: two f64 sums of the same
//! f32 terms differ far below one f32 ulp of the mean, so the rounded mean
//! (and variance) agree unless the exact value sits that close to an f32
//! rounding boundary. The f16 rounding is `f32_to_f16_bits`, to nearest
//! even. The pool is ik's op order (`soft_max` over the members, `mul`,
//! `sum_rows`) with our exponential, so it matches ik within a band, and its
//! host rule [`index_pool_host`] bit for bit.
//!
//! No silent failure: a position at or past the cache's rows raises
//! [`FaultSite::CachePos`] and the token writes nothing; a mean square (or
//! variance) that is not finite makes the scale NaN, and a stored value that
//! is not finite — a non-finite input, a NaN scale, a value past f16's range
//! — raises [`FaultSite::CacheValue`] with the row written as computed,
//! never as a plausible one. The tokens of one launch hold distinct
//! positions (the caller's contract), so no two blocks write one row. A pool
//! key that is not finite raises [`FaultSite::PoolSelect`] with the key
//! written as computed; a count past the cache's rows raises
//! [`FaultSite::CachePos`] and writes nothing. The forced-tail entries bound
//! a position and a count by the context, not by the ring's rows: a
//! position at or past `ctx` and a count past it raise `CachePos` and write
//! nothing, and the ring's slot is the position's remainder, so the tokens
//! of one launch hold positions no two of which share a slot.

use crate::GpuError;
use crate::elem::rms_scale_of_mean;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{f16_finite, f32_to_f16_bits, half_bits_to_f32};
use crate::launch_u32;
use crate::linear::expf_ik;
use crate::rope_table::rope_pair_rn;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values of a latent row: the attention's key and value width.
pub const LATENT: usize = 512;
/// Values of the indexer key, and of the pool gate beside it.
pub const INDEX_HEAD: usize = 128;
/// Values of an index cache row: the key, then the gate.
pub const INDEX_ROW: usize = 2 * INDEX_HEAD;
/// Positions per pool (`attention.indexer.kpool`).
pub const POOL: usize = 4;
/// Values at the tail of a forced-tail selector's pool key (and query head)
/// that turn, NORM mode (`rope.dimension_count`): a rope table row's f32.
pub const ROT: usize = 64;

/// Pool rows a cache of `rows` positions has room for: one per [`POOL`]
/// positions, the last one incomplete when `rows` is not a multiple of it.
#[must_use]
pub const fn pools_for(rows: usize) -> usize {
    rows.div_ceil(POOL)
}

/// Threads per block of the entries but the forced-tail key append: four
/// warps.
const BLOCK: u32 = 128;
/// Threads per block of the forced-tail key append: a block per token, four
/// warps a projection.
const KEY_BLOCK: u32 = 256;
/// Latent values each thread owns.
const PER_THREAD: usize = LATENT / BLOCK as usize;
/// Index-key values each lane owns.
const PER_LANE: usize = INDEX_HEAD / 32;
const _: () = assert!(PER_THREAD == 4 && PER_LANE == 4 && BLOCK == 128 && POOL == 4);
const _: () = assert!(KEY_BLOCK == 256 && ROT == 64 && ROT <= INDEX_HEAD);

/// A warp's sum of one f64 a lane, the xor butterfly (16, 8, 4, 2, 1): every
/// lane ends with the same sum. Every lane of the warp calls it (a warp
/// collective).
#[inline(always)]
pub fn warp_sum_f64(v: f64) -> f64 {
    let mut acc = v;
    acc += warp::shuffle_xor_f64(acc, 16);
    acc += warp::shuffle_xor_f64(acc, 8);
    acc += warp::shuffle_xor_f64(acc, 4);
    acc += warp::shuffle_xor_f64(acc, 2);
    acc += warp::shuffle_xor_f64(acc, 1);
    acc
}

#[cuda_module]
mod latent_kernels {
    use super::*;

    /// The latent rows of `m` tokens (module doc), one block per token:
    /// token `t`'s latent is `x[t·stride + off ..][..LATENT]`, its f16 row
    /// goes to row `pos[t]` of `cache` (`rows` rows of [`LATENT`]). `off +
    /// LATENT <= stride` (host-checked).
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
            x.len() >= m * stride,
            gain.len() >= 512,
            pos.len() >= m,
            cache.len() >= rows * 512
        )
    )]
    pub fn latent_rms_append(
        x: &[f32],
        gain: &[f32],
        pos: &[u32],
        eps: f32,
        stride: u32,
        off: u32,
        rows: u32,
        m: u32,
        fault: FaultSink,
        mut cache: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 4> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let base = t * stride as usize + off as usize;
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;
        let mut v = [0.0f32; PER_THREAD];
        let mut acc = 0.0f64;
        let mut i = 0usize;
        #[unroll]
        while i < PER_THREAD {
            // SAFETY: tid + 128·i < LATENT and off + LATENT <= stride
            // (host-checked), so base + tid + 128·i < (t + 1)·stride <=
            // m·stride <= x.len() by the launch contract.
            v[i] = unsafe { *x.get_unchecked(base + tid + 128 * i) };
            acc += f64::from(mul_rn_f32(v[i], v[i]));
            i += 1;
        }
        let acc = warp_sum_f64(acc);
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Four slots, one per warp, written before the barrier that
        // publishes them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 4; one lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: all four slots were written before the barrier above.
        let sum = unsafe { ((*ws.add(0) + *ws.add(1)) + *ws.add(2)) + *ws.add(3) };
        let mean = (sum / LATENT as f64) as f32;
        let scale = if mean.is_finite() {
            1.0 / (mean + eps).sqrt()
        } else {
            f32::NAN
        };

        if p >= rows as usize {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // block-uniform: p is the token's
        }
        let row = p * LATENT;
        let mut ok = true;
        let mut i = 0usize;
        #[unroll]
        while i < PER_THREAD {
            let at = tid + 128 * i;
            // SAFETY: at < LATENT <= gain.len() by the launch contract.
            let g = unsafe { *gain.get_unchecked(at) };
            let h = f32_to_f16_bits(mul_rn_f32(mul_rn_f32(scale, g), v[i]));
            ok &= f16_finite(h);
            // SAFETY: p < rows, so row + at < (p + 1)·LATENT <= rows·LATENT <=
            // cache.len() by the launch contract; one thread per value, and
            // one block per position.
            unsafe { *cache.get_unchecked_mut(row + at) = h };
            i += 1;
        }
        if !ok {
            fault.raise(FaultSite::CacheValue);
        }
    }

    /// The `[extra | latent]` rows of `m` tokens (module doc), one block per
    /// token: token `t`'s latent is `x[t·stride + off ..][..LATENT]`, its
    /// extra part `x[t·stride + e_off ..][..extra]`; the f16 row goes to row
    /// `pos[t]` of `cache` (`rows` rows of `extra + LATENT`), the extra part
    /// first, raw, then the latent normed as `latent_rms_append` norms it.
    /// `off + LATENT <= stride`, `e_off + extra <= stride` and `extra <= 128`
    /// (host-checked).
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
            x.len() >= m * stride,
            gain.len() >= 512,
            pos.len() >= m,
            cache.len() >= rows * (512 + extra)
        )
    )]
    pub fn latent_extra_rms_append(
        x: &[f32],
        gain: &[f32],
        pos: &[u32],
        eps: f32,
        stride: u32,
        off: u32,
        e_off: u32,
        extra: u32,
        rows: u32,
        m: u32,
        fault: FaultSink,
        mut cache: DisjointSlice<u16>,
    ) {
        static mut XSUM: SharedArray<f64, 4> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let base = t * stride as usize + off as usize;
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;
        let mut v = [0.0f32; PER_THREAD];
        let mut acc = 0.0f64;
        let mut i = 0usize;
        #[unroll]
        while i < PER_THREAD {
            // SAFETY: tid + 128·i < LATENT and off + LATENT <= stride
            // (host-checked), so base + tid + 128·i < (t + 1)·stride <=
            // m·stride <= x.len() by the launch contract.
            v[i] = unsafe { *x.get_unchecked(base + tid + 128 * i) };
            acc += f64::from(mul_rn_f32(v[i], v[i]));
            i += 1;
        }
        let acc = warp_sum_f64(acc);
        // SAFETY: XSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Four slots, one per warp, written before the barrier that
        // publishes them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut XSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 4; one lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: all four slots were written before the barrier above.
        let sum = unsafe { ((*ws.add(0) + *ws.add(1)) + *ws.add(2)) + *ws.add(3) };
        let mean = (sum / LATENT as f64) as f32;
        let scale = if mean.is_finite() {
            1.0 / (mean + eps).sqrt()
        } else {
            f32::NAN
        };

        if p >= rows as usize {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // block-uniform: p is the token's
        }
        let width = extra as usize + LATENT;
        let row = p * width;
        let mut ok = true;
        if tid < extra as usize {
            // SAFETY: tid < extra and e_off + extra <= stride (host-checked),
            // so the read lies in token t's row of x; row + tid < (p + 1)·width
            // <= rows·width <= cache.len() by the launch contract, one thread
            // per value and one block per position.
            unsafe {
                let e = *x.get_unchecked(t * stride as usize + e_off as usize + tid);
                let h = f32_to_f16_bits(e);
                ok &= f16_finite(h);
                *cache.get_unchecked_mut(row + tid) = h;
            }
        }
        let mut i = 0usize;
        #[unroll]
        while i < PER_THREAD {
            let at = tid + 128 * i;
            // SAFETY: at < LATENT <= gain.len() by the launch contract.
            let g = unsafe { *gain.get_unchecked(at) };
            let h = f32_to_f16_bits(mul_rn_f32(mul_rn_f32(scale, g), v[i]));
            ok &= f16_finite(h);
            // SAFETY: p < rows, so row + extra + at < (p + 1)·width <=
            // rows·width <= cache.len() by the launch contract; one thread per
            // value, and one block per position.
            unsafe { *cache.get_unchecked_mut(row + extra as usize + at) = h };
            i += 1;
        }
        if !ok {
            fault.raise(FaultSite::CacheValue);
        }
    }

    /// The index cache rows of `m` tokens (module doc), one warp per token,
    /// warp `w = 4·block + warp`: token `w`'s key is `x[w·stride + k_off
    /// ..][..INDEX_HEAD]` and its gate `x[w·stride + g_off ..][..INDEX_HEAD]`;
    /// the f16 row `[key; gate]` goes to row `pos[w]` of `cache` (`rows` rows
    /// of [`INDEX_ROW`]). `k_off + INDEX_HEAD <= stride` and `g_off +
    /// INDEX_HEAD <= stride` (host-checked).
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
            x.len() >= m * stride,
            w.len() >= 128,
            b.len() >= 128,
            pos.len() >= m,
            cache.len() >= rows * 256
        )
    )]
    pub fn index_key_ln_append(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        pos: &[u32],
        eps: f32,
        stride: u32,
        k_off: u32,
        g_off: u32,
        rows: u32,
        m: u32,
        fault: FaultSink,
        mut cache: DisjointSlice<u16>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let t = thread::blockIdx_x() as usize * 4 + tid / 32;
        if t >= m as usize {
            return; // warp-uniform
        }
        let lane = tid % 32;
        let kbase = t * stride as usize + k_off as usize;
        let gbase = t * stride as usize + g_off as usize;
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;
        let mut k = [0.0f32; PER_LANE];
        let mut s1 = 0.0f64;
        let mut j = 0usize;
        #[unroll]
        while j < PER_LANE {
            // SAFETY: lane + 32·j < INDEX_HEAD and k_off + INDEX_HEAD <=
            // stride (host-checked), so the read is inside token t's row of
            // m·stride <= x.len() values (launch contract).
            k[j] = unsafe { *x.get_unchecked(kbase + lane + 32 * j) };
            s1 += f64::from(k[j]);
            j += 1;
        }
        let mean = (warp_sum_f64(s1) / INDEX_HEAD as f64) as f32;
        let mut s2 = 0.0f64;
        let mut j = 0usize;
        #[unroll]
        while j < PER_LANE {
            k[j] -= mean;
            s2 += f64::from(mul_rn_f32(k[j], k[j]));
            j += 1;
        }
        let var = (warp_sum_f64(s2) / INDEX_HEAD as f64) as f32;
        let scale = if var.is_finite() {
            1.0 / (var + eps).sqrt()
        } else {
            f32::NAN
        };

        if p >= rows as usize {
            if lane == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // warp-uniform: p is the token's
        }
        let row = p * INDEX_ROW;
        let mut ok = true;
        let mut j = 0usize;
        #[unroll]
        while j < PER_LANE {
            let at = lane + 32 * j;
            // SAFETY: at < INDEX_HEAD <= w.len(), b.len() by the launch
            // contract; the gate read is inside token t's row as the key's
            // (g_off + INDEX_HEAD <= stride, host-checked).
            let (wv, bv, g) = unsafe {
                (
                    *w.get_unchecked(at),
                    *b.get_unchecked(at),
                    *x.get_unchecked(gbase + at),
                )
            };
            let hk = f32_to_f16_bits(add_rn_f32(mul_rn_f32(mul_rn_f32(k[j], scale), wv), bv));
            let hg = f32_to_f16_bits(g);
            ok &= f16_finite(hk) & f16_finite(hg);
            // SAFETY: p < rows, so row + INDEX_HEAD + at < (p + 1)·INDEX_ROW
            // <= rows·INDEX_ROW <= cache.len() by the launch contract; one
            // lane per value pair, and one warp per position.
            unsafe {
                *cache.get_unchecked_mut(row + at) = hk;
                *cache.get_unchecked_mut(row + INDEX_HEAD + at) = hg;
            }
            j += 1;
        }
        if !ok {
            fault.raise(FaultSite::CacheValue);
        }
    }

    /// The pool keys of `m` tokens (module doc), one warp per token, warp `w
    /// = 4·block + warp`: token `w` of live count `c = n_keys[w]` (its
    /// position plus one) completes pool `j = c/4 − 1` when `c` is a positive
    /// multiple of [`POOL`], and writes that pool's [`INDEX_HEAD`] f16 values
    /// to row `j` of `pooled` from rows `4j .. 4j + 3` of `cache` (`rows`
    /// rows of [`INDEX_ROW`]) and the four [`INDEX_HEAD`]-value rows of
    /// `ape`. Any other count writes nothing; a count past `rows` raises
    /// `CachePos`.
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
            cache.len() >= rows * 256,
            ape.len() >= 512,
            n_keys.len() >= m,
            pools * 4 >= rows,
            pooled.len() >= pools * 128
        )
    )]
    pub fn index_pool(
        cache: &[u16],
        ape: &[f32],
        n_keys: &[u32],
        rows: u32,
        pools: u32,
        m: u32,
        fault: FaultSink,
        mut pooled: DisjointSlice<u16>,
    ) {
        let _ = pools;
        let tid = thread::threadIdx_x() as usize;
        let t = thread::blockIdx_x() as usize * 4 + tid / 32;
        if t >= m as usize {
            return; // warp-uniform
        }
        let lane = tid % 32;
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if c > rows {
            if lane == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // warp-uniform: c is the token's
        }
        if c == 0 || !c.is_multiple_of(POOL as u32) {
            return; // warp-uniform
        }
        let j = (c / POOL as u32 - 1) as usize;
        let base = POOL * j * INDEX_ROW;
        let mut ok = true;
        let mut d = 0usize;
        #[unroll]
        while d < PER_LANE {
            let at = lane + 32 * d;
            let mut k = [0.0f32; POOL];
            let mut z = [0.0f32; POOL];
            let mut i = 0usize;
            #[unroll]
            while i < POOL {
                // SAFETY: the rows 4j .. 4j + 3 are below c <= rows, so both
                // reads are inside rows·INDEX_ROW <= cache.len(); at <
                // INDEX_HEAD and i < POOL keep the ape read inside its
                // POOL·INDEX_HEAD values (launch contract).
                let (kb, gb, a) = unsafe {
                    (
                        *cache.get_unchecked(base + i * INDEX_ROW + at),
                        *cache.get_unchecked(base + i * INDEX_ROW + INDEX_HEAD + at),
                        *ape.get_unchecked(i * INDEX_HEAD + at),
                    )
                };
                k[i] = half_bits_to_f32(kb);
                z[i] = add_rn_f32(half_bits_to_f32(gb), a);
                i += 1;
            }
            let mx = z[0].max(z[1]).max(z[2]).max(z[3]);
            let e = [
                expf_ik(z[0] - mx),
                expf_ik(z[1] - mx),
                expf_ik(z[2] - mx),
                expf_ik(z[3] - mx),
            ];
            let inv = 1.0 / add_rn_f32(add_rn_f32(add_rn_f32(e[0], e[1]), e[2]), e[3]);
            let p = add_rn_f32(
                add_rn_f32(
                    add_rn_f32(
                        mul_rn_f32(mul_rn_f32(e[0], inv), k[0]),
                        mul_rn_f32(mul_rn_f32(e[1], inv), k[1]),
                    ),
                    mul_rn_f32(mul_rn_f32(e[2], inv), k[2]),
                ),
                mul_rn_f32(mul_rn_f32(e[3], inv), k[3]),
            );
            let h = f32_to_f16_bits(p);
            ok &= f16_finite(h);
            // SAFETY: j < c/4 <= rows/4 <= pools, so j·INDEX_HEAD + at <
            // pools·INDEX_HEAD <= pooled.len() (launch contract); the counts
            // of one launch are distinct, so one lane writes each value.
            unsafe { *pooled.get_unchecked_mut(j * INDEX_HEAD + at) = h };
            d += 1;
        }
        if !ok {
            fault.raise(FaultSite::PoolSelect);
        }
    }

    /// The cache rows a selection list picks, gathered into a contiguous
    /// scratch the dense latent attention runs over (attention is order-free
    /// over its keys): block `b` serves token `t = b / stride`'s list entry
    /// `i = b % stride`, copying row `list[t·stride + i]` of `src`
    /// (`[rows × width]` f16) to row `t·stride + i` of `dst` (`[tokens·stride
    /// × width]` f16). Token `t`'s count `n[t]` (the launch's device word,
    /// `min` of its visible rows and its list's width, the caller's own
    /// image) bounds the copy: a block at or past its token's count leaves,
    /// so the scratch's rows below each token's count are its list's rows in
    /// list order and the rest keep what they held. A list entry at or past
    /// `rows` raises [`FaultSite::CachePos`] from lane 0 and copies nothing
    /// — no row a later kernel could mistake for a picked one.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            list.len() >= tokens * stride,
            n.len() >= tokens,
            src.len() >= rows * width,
            dst.len() >= tokens * stride * width
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn kv_gather_sel(
        list: &[u32],
        n: &[u32],
        stride: u32,
        rows: u32,
        width: u32,
        tokens: u32,
        fault: FaultSink,
        src: &[u16],
        mut dst: DisjointSlice<u16>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let t = b / stride as usize;
        if t >= tokens as usize {
            return;
        }
        // SAFETY: t < tokens <= n.len() by the launch contract; the guard
        // reads only the block's index, so every thread of the block leaves
        // together before any collective.
        if b % stride as usize >= unsafe { *n.get_unchecked(t) } as usize {
            return;
        }
        // SAFETY: b < tokens·stride <= list.len() (the launch contract).
        let row = unsafe { *list.get_unchecked(b) } as usize;
        if row >= rows as usize {
            if thread::threadIdx_x() == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // block-uniform
        }
        let (src_base, dst_base) = (row * width as usize, b * width as usize);
        let mut i = thread::threadIdx_x() as usize;
        while i < width as usize {
            // SAFETY: i < width, so src_base + i < (row + 1)·width <=
            // rows·width <= src.len() and dst_base + i < (b + 1)·width <=
            // tokens·stride·width <= dst.len() (the launch contract); thread
            // i owns value i of its block's row.
            unsafe {
                *dst.get_unchecked_mut(dst_base + i) = *src.get_unchecked(src_base + i);
            }
            i += 256;
        }
    }

    /// The forced-tail selector's index rows of `m` tokens (module doc), one
    /// block of [`KEY_BLOCK`] per token `t = block`: its raw key is
    /// `x[t·stride + k_off ..][..INDEX_HEAD]`; every warp norms it, lane `l`
    /// owning values `4l .. 4l + 3`, and warp `w` writes the f16 outputs of
    /// rows `32·(w % 4) ..` of the value projection (`w < 4`) or of the gate
    /// projection, lane `l` row `32·(w % 4) + l`, into slot `pos[t] mod ring`
    /// of `cache` (`ring` rows of [`INDEX_ROW`], the value first). Each
    /// projection is `[INDEX_HEAD × INDEX_HEAD]` q8_0 in the q8f32 planes
    /// (`qs` 32 words a row, `d` 4 scales a row). A position at or past
    /// `ctx` raises `CachePos` and the token writes nothing. `k_off +
    /// INDEX_HEAD <= stride` (host-checked).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= m * stride,
            gain.len() >= 128,
            v_qs.len() >= 4096,
            v_d.len() >= 512,
            g_qs.len() >= 4096,
            g_d.len() >= 512,
            pos.len() >= m,
            ring >= 1,
            cache.len() >= ring * 256
        )
    )]
    pub fn index_key_rms_append(
        x: &[f32],
        gain: &[f32],
        v_qs: &[u32],
        v_d: &[u16],
        g_qs: &[u32],
        g_d: &[u16],
        pos: &[u32],
        eps: f32,
        stride: u32,
        k_off: u32,
        ring: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        mut cache: DisjointSlice<u16>,
    ) {
        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let wid = tid / 32;
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) };
        if p >= ctx {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // block-uniform: p is the token's
        }
        // ---- the norm, in every warp: lane `lane` values `4·lane ..`.
        let base = t * stride as usize + k_off as usize + PER_LANE * lane;
        let mut k = [0.0f32; PER_LANE];
        let mut s = 0.0f64;
        let mut i = 0usize;
        #[unroll]
        while i < PER_LANE {
            // SAFETY: 4·lane + i < INDEX_HEAD and k_off + INDEX_HEAD <= stride
            // (host-checked) keep the read inside token t's row of m·stride
            // <= x.len() values (launch contract).
            k[i] = unsafe { *x.get_unchecked(base + i) };
            s += f64::from(mul_rn_f32(k[i], k[i]));
            i += 1;
        }
        let mean = (warp_sum_f64(s) / INDEX_HEAD as f64) as f32;
        let scale = rms_scale_of_mean(mean, eps);
        let mut i = 0usize;
        #[unroll]
        while i < PER_LANE {
            // SAFETY: 4·lane + i < INDEX_HEAD <= gain.len() (launch contract).
            let gv = unsafe { *gain.get_unchecked(PER_LANE * lane + i) };
            k[i] = mul_rn_f32(mul_rn_f32(k[i], scale), gv);
            i += 1;
        }
        // ---- the projection rows: q8f32's word walk at k = 128, one code
        // word a lane, then the warp's butterfly; lane r keeps row r's sum.
        let (qs, d) = if wid < 4 { (v_qs, v_d) } else { (g_qs, g_d) };
        let row0 = 32 * (wid % 4);
        let mut mine = 0.0f32;
        let mut r = 0usize;
        while r < 32 {
            let row = row0 + r;
            // SAFETY: row < INDEX_HEAD and lane < 32 put word row·32 + lane
            // below 4096 <= qs.len() and scale row·4 + lane/8 below 512 <=
            // d.len() (launch contract), for either projection.
            let (q, db) = unsafe {
                (
                    *qs.get_unchecked(row * 32 + lane),
                    *d.get_unchecked(row * 4 + lane / 8),
                )
            };
            let dv = half_bits_to_f32(db);
            let f = f32::mul_add(dv * (q as u8 as i8 as f32), k[0], 0.0);
            let f = f32::mul_add(dv * ((q >> 8) as u8 as i8 as f32), k[1], f);
            let f = f32::mul_add(dv * ((q >> 16) as u8 as i8 as f32), k[2], f);
            let f = f32::mul_add(dv * ((q >> 24) as u8 as i8 as f32), k[3], f);
            let sum = warp::reduce_sum_f32(f);
            if lane == r {
                mine = sum;
            }
            r += 1;
        }
        let h = f32_to_f16_bits(mine);
        let col = (if wid < 4 { 0 } else { INDEX_HEAD }) + row0 + lane;
        let slot = (p % ring) as usize;
        // SAFETY: slot < ring and col < INDEX_ROW, so the index is below
        // ring·INDEX_ROW <= cache.len() (launch contract); one thread per
        // value of the row, and the tokens of one launch hold slots no two
        // share (the caller's contract).
        unsafe { *cache.get_unchecked_mut(slot * INDEX_ROW + col) = h };
        if !f16_finite(h) {
            fault.raise(FaultSite::CacheValue);
        }
    }

    /// The forced-tail selector's pool keys of `m` tokens (module doc), one
    /// warp per token, warp `w = 4·block + warp`: token `w` of live count `c
    /// = n_keys[w]` completes pool `j = c/4 − 1` when `c` is a positive
    /// multiple of [`POOL`], and writes its [`INDEX_HEAD`] f16 values to row
    /// `j` of `pooled` from the slots `(4j + i) mod ring` of `cache` (`ring`
    /// rows of [`INDEX_ROW`], value then gate), the four [`INDEX_HEAD`]-value
    /// rows of `ape`, the norm's `gain`, and row `4j` of `table` (`ctx` rows
    /// of [`ROT`] f32). Lane `l` owns channels `4l .. 4l + 3`, so a NORM pair
    /// of the turned tail is one lane's. Any other count writes nothing; a
    /// count past `ctx` raises `CachePos`.
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
            ring >= 1,
            cache.len() >= ring * 256,
            ape.len() >= 512,
            gain.len() >= 128,
            table.len() >= ctx * 64,
            n_keys.len() >= m,
            pools * 4 >= ctx,
            pooled.len() >= pools * 128
        )
    )]
    pub fn index_pool_norm_rope(
        cache: &[u16],
        ape: &[f32],
        gain: &[f32],
        table: &[f32],
        n_keys: &[u32],
        eps: f32,
        ring: u32,
        ctx: u32,
        pools: u32,
        m: u32,
        fault: FaultSink,
        mut pooled: DisjointSlice<u16>,
    ) {
        let _ = pools;
        let tid = thread::threadIdx_x() as usize;
        let t = thread::blockIdx_x() as usize * 4 + tid / 32;
        if t >= m as usize {
            return; // warp-uniform
        }
        let lane = tid % 32;
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if c > ctx {
            if lane == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return; // warp-uniform: c is the token's
        }
        if c == 0 || !c.is_multiple_of(POOL as u32) {
            return; // warp-uniform
        }
        let j = (c / POOL as u32 - 1) as usize;
        let ring = ring as usize;
        let mut slot = [0usize; POOL];
        let mut i = 0usize;
        #[unroll]
        while i < POOL {
            slot[i] = (POOL * j + i) % ring;
            i += 1;
        }
        // ---- the pool, per channel, as `index_pool` forms it.
        let mut pv = [0.0f32; PER_LANE];
        let mut sq = 0.0f64;
        let mut d = 0usize;
        #[unroll]
        while d < PER_LANE {
            let at = PER_LANE * lane + d;
            let mut k = [0.0f32; POOL];
            let mut z = [0.0f32; POOL];
            let mut i = 0usize;
            #[unroll]
            while i < POOL {
                // SAFETY: slot[i] < ring, so both reads are inside
                // ring·INDEX_ROW <= cache.len(); at < INDEX_HEAD and i < POOL
                // keep the ape read inside its POOL·INDEX_HEAD values (launch
                // contract).
                let (kb, gb, a) = unsafe {
                    (
                        *cache.get_unchecked(slot[i] * INDEX_ROW + at),
                        *cache.get_unchecked(slot[i] * INDEX_ROW + INDEX_HEAD + at),
                        *ape.get_unchecked(i * INDEX_HEAD + at),
                    )
                };
                k[i] = half_bits_to_f32(kb);
                z[i] = add_rn_f32(half_bits_to_f32(gb), a);
                i += 1;
            }
            let mx = z[0].max(z[1]).max(z[2]).max(z[3]);
            let e = [
                expf_ik(z[0] - mx),
                expf_ik(z[1] - mx),
                expf_ik(z[2] - mx),
                expf_ik(z[3] - mx),
            ];
            let inv = 1.0 / add_rn_f32(add_rn_f32(add_rn_f32(e[0], e[1]), e[2]), e[3]);
            pv[d] = add_rn_f32(
                add_rn_f32(
                    add_rn_f32(
                        mul_rn_f32(mul_rn_f32(e[0], inv), k[0]),
                        mul_rn_f32(mul_rn_f32(e[1], inv), k[1]),
                    ),
                    mul_rn_f32(mul_rn_f32(e[2], inv), k[2]),
                ),
                mul_rn_f32(mul_rn_f32(e[3], inv), k[3]),
            );
            sq += f64::from(mul_rn_f32(pv[d], pv[d]));
            d += 1;
        }
        // ---- the norm, then the turn of the tail at the pool's first
        // position.
        let mean = (warp_sum_f64(sq) / INDEX_HEAD as f64) as f32;
        let scale = rms_scale_of_mean(mean, eps);
        let mut y = [0.0f32; PER_LANE];
        let mut d = 0usize;
        #[unroll]
        while d < PER_LANE {
            // SAFETY: 4·lane + d < INDEX_HEAD <= gain.len() (launch contract).
            let gv = unsafe { *gain.get_unchecked(PER_LANE * lane + d) };
            y[d] = mul_rn_f32(mul_rn_f32(pv[d], scale), gv);
            d += 1;
        }
        let v = PER_LANE * lane;
        if v >= INDEX_HEAD - ROT {
            let at = POOL * j * ROT + v - (INDEX_HEAD - ROT);
            // SAFETY: 4j < c <= ctx and v − 64 + 3 < ROT put the four words
            // inside row 4j of the table, below ctx·ROT <= table.len()
            // (launch contract).
            let cs = unsafe {
                [
                    *table.get_unchecked(at),
                    *table.get_unchecked(at + 1),
                    *table.get_unchecked(at + 2),
                    *table.get_unchecked(at + 3),
                ]
            };
            let (a0, a1) = rope_pair_rn(y[0], y[1], cs[0], cs[1]);
            let (a2, a3) = rope_pair_rn(y[2], y[3], cs[2], cs[3]);
            y = [a0, a1, a2, a3];
        }
        let mut ok = true;
        let mut d = 0usize;
        #[unroll]
        while d < PER_LANE {
            let h = f32_to_f16_bits(y[d]);
            ok &= f16_finite(h);
            // SAFETY: j < c/4 <= ctx/4 <= pools, so j·INDEX_HEAD + v + d <
            // pools·INDEX_HEAD <= pooled.len() (launch contract); the counts
            // of one launch are distinct, so one lane writes each value.
            unsafe { *pooled.get_unchecked_mut(j * INDEX_HEAD + v + d) = h };
            d += 1;
        }
        if !ok {
            fault.raise(FaultSite::PoolSelect);
        }
    }
}

/// Where one token's parts sit in the stacked projection output: `m` rows of
/// `stride` f32, token-major.
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub stride: usize,
    pub m: usize,
}

/// [`LatentKernels::enqueue_latent_append`]'s arguments: the latent at
/// column `off` of each row of `rows`, the norm's [`LATENT`] gains, the `m`
/// positions on the device, and the dense cache (`[positions][LATENT]` f16).
pub struct LatentAppendArgs<'a> {
    pub rows: Rows<'a>,
    pub off: usize,
    pub gain: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub fault: FaultSink,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`LatentKernels::enqueue_latent_extra_append`]'s arguments: the latent at
/// column `off` and the `extra` raw values at column `e_off` of each row of
/// `rows`, the norm's [`LATENT`] gains, the `m` positions on the device, and
/// the dense cache (`[positions][extra + LATENT]` f16).
pub struct LatentExtraAppendArgs<'a> {
    pub rows: Rows<'a>,
    pub off: usize,
    pub e_off: usize,
    pub extra: usize,
    pub gain: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub fault: FaultSink,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`LatentKernels::enqueue_index_key_append`]'s arguments: the key at
/// column `k_off` and the gate at column `g_off` of each row of `rows`, the
/// LayerNorm's [`INDEX_HEAD`] weights and biases, the `m` positions on the
/// device, and the dense cache (`[positions][INDEX_ROW]` f16).
pub struct IndexKeyArgs<'a> {
    pub rows: Rows<'a>,
    pub k_off: usize,
    pub g_off: usize,
    pub w: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub fault: FaultSink,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`LatentKernels::enqueue_index_pool`]'s arguments: the layer's index
/// cache (`[rows][INDEX_ROW]` f16, the rows the call wrote included), the
/// pool gate's position bias (`indexer_compressor_ape`, [`POOL`] rows of
/// [`INDEX_HEAD`] f32), the `m` live counts on the device, and the pool
/// plane (`[pools_for(rows)][INDEX_HEAD]` f16).
pub struct IndexPoolArgs<'a> {
    pub cache: &'a DeviceTensor<u16>,
    pub ape: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub m: usize,
    pub fault: FaultSink,
    pub pooled: &'a mut DeviceTensor<u16>,
}

/// One `[INDEX_HEAD × INDEX_HEAD]` q8_0 projection in the q8f32 planes
/// (`crate::weights::DevWeight::Q8_0`'s `qs` and `d`): [`INDEX_HEAD`] rows of
/// 32 code words and of 4 f16 scales.
#[derive(Clone, Copy)]
pub struct IndexProj<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
}

/// [`LatentKernels::enqueue_index_key_rms_append`]'s arguments: the raw key
/// at column `k_off` of each row of `rows`, the key norm's [`INDEX_HEAD`]
/// gains, the value and gate projections, the `m` positions on the device
/// (each below `ctx`), and the index rows (`[ring][INDEX_ROW]` f16, position
/// `p` in slot `p mod ring`: a dense cache when `ring >= ctx`).
pub struct IndexKeyRmsArgs<'a> {
    pub rows: Rows<'a>,
    pub k_off: usize,
    pub gain: &'a DeviceBuffer<f32>,
    pub value: IndexProj<'a>,
    pub gate: IndexProj<'a>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub ctx: usize,
    pub fault: FaultSink,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`LatentKernels::enqueue_index_pool_norm_rope`]'s arguments: the index rows
/// (`[ring][INDEX_ROW]` f16, the slots the call's appends wrote included),
/// the pool gate's position bias (`indexer_compressor_ape`, [`POOL`] rows of
/// [`INDEX_HEAD`] f32), the pool norm's gains, the layer's rope table (`ctx`
/// rows of [`ROT`] f32), the `m` live counts on the device, and the pool plane
/// (`[pools_for(ctx)][INDEX_HEAD]` f16).
pub struct IndexPoolRopeArgs<'a> {
    pub cache: &'a DeviceTensor<u16>,
    pub ape: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub pooled: &'a mut DeviceTensor<u16>,
}

/// [`LatentKernels::enqueue_kv_gather_sel`]'s arguments: token `t`'s picked
/// rows at `list[t·stride ..]`, its count at `n[t]` (at most `stride`), the
/// cache `src` (`[rows × width]` f16) the list indexes, and the scratch
/// (`[tokens·stride × width]` f16) the gathered rows land in, token-major.
pub struct KvGatherSelArgs<'a> {
    pub list: &'a DeviceBuffer<u32>,
    pub n: &'a DeviceBuffer<u32>,
    pub stride: usize,
    pub src: &'a DeviceTensor<u16>,
    pub fault: FaultSink,
    pub dst: &'a mut DeviceTensor<u16>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream,
/// so its launch orders with the rest of the step and captures.
pub struct LatentKernels {
    module: latent_kernels::LoadedModule,
}

/// The checks both enqueues share: at least one token, a cache of
/// `width`-wide rows with at least one row, the parts `(name, offset,
/// values)` inside a row, the rows and the positions covering `m` tokens.
fn check_shape(
    what: &'static str,
    rows: Rows<'_>,
    parts: &[(&str, usize, usize)],
    pos: &DeviceBuffer<u32>,
    cache: &DeviceTensor<u16>,
    width: usize,
) -> Result<(), GpuError> {
    if rows.m == 0 || cache.rows() == 0 {
        return Err(GpuError::shape(
            what,
            format!(
                "need m >= 1 and a cache of at least one row, got m={} rows={}",
                rows.m,
                cache.rows()
            ),
        ));
    }
    if cache.cols() != width {
        return Err(GpuError::shape(
            what,
            format!("the cache's rows are {} wide, want {width}", cache.cols()),
        ));
    }
    for &(name, off, n) in parts {
        if off + n > rows.stride {
            return Err(GpuError::shape(
                what,
                format!(
                    "the {name} at column {off} ({n} values) passes the row stride {}",
                    rows.stride
                ),
            ));
        }
    }
    if rows.x.len() < rows.m * rows.stride || pos.len() < rows.m {
        return Err(GpuError::shape(
            what,
            format!(
                "x.len() {} (need {}), pos.len() {} (need {})",
                rows.x.len(),
                rows.m * rows.stride,
                pos.len(),
                rows.m
            ),
        ));
    }
    Ok(())
}

impl LatentKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<LatentKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(latent_kernels, ctx)? };
        Ok(LatentKernels { module })
    }

    /// Enqueue the latent rows of `m` tokens: one block per token. Refused
    /// before the launch: no token, a cache of no rows or not [`LATENT`]
    /// wide, a latent past the row stride, short rows, positions or gains.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_latent_append(
        &self,
        stream: &CudaStream,
        a: LatentAppendArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_latent_append";
        let LatentAppendArgs {
            rows,
            off,
            gain,
            pos,
            eps,
            fault,
            cache,
        } = a;
        check_shape(what, rows, &[("latent", off, LATENT)], pos, cache, LATENT)?;
        if gain.len() < LATENT {
            return Err(GpuError::shape(
                what,
                format!("gain.len() {} < {LATENT}", gain.len()),
            ));
        }
        let m = launch_u32(what, "m", rows.m)?;
        let stride = launch_u32(what, "stride", rows.stride)?;
        let off = launch_u32(what, "off", off)?;
        let n_rows = launch_u32(what, "rows", cache.rows())?;
        let prep = self
            .module
            .prepare_latent_rms_append(LaunchConfig1D::new(m, BLOCK, 0))?;
        self.module.latent_rms_append(
            stream,
            &prep,
            rows.x,
            gain,
            pos,
            eps,
            stride,
            off,
            n_rows,
            m,
            fault,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue the `[extra | latent]` rows of `m` tokens: one block per
    /// token. Refused before the launch: no token, no extra part or one wider
    /// than the block, a cache of no rows or not `extra + LATENT` wide, a
    /// part past the row stride, short rows, positions or gains.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_latent_extra_append(
        &self,
        stream: &CudaStream,
        a: LatentExtraAppendArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_latent_extra_append";
        let LatentExtraAppendArgs {
            rows,
            off,
            e_off,
            extra,
            gain,
            pos,
            eps,
            fault,
            cache,
        } = a;
        if extra == 0 || extra > BLOCK as usize {
            return Err(GpuError::shape(
                what,
                format!("the extra part is one value a thread, 1..={BLOCK}, got {extra}"),
            ));
        }
        check_shape(
            what,
            rows,
            &[("latent", off, LATENT), ("extra part", e_off, extra)],
            pos,
            cache,
            extra + LATENT,
        )?;
        if gain.len() < LATENT {
            return Err(GpuError::shape(
                what,
                format!("gain.len() {} < {LATENT}", gain.len()),
            ));
        }
        let m = launch_u32(what, "m", rows.m)?;
        let stride = launch_u32(what, "stride", rows.stride)?;
        let off = launch_u32(what, "off", off)?;
        let e_off = launch_u32(what, "e_off", e_off)?;
        let extra = launch_u32(what, "extra", extra)?;
        let n_rows = launch_u32(what, "rows", cache.rows())?;
        let prep = self
            .module
            .prepare_latent_extra_rms_append(LaunchConfig1D::new(m, BLOCK, 0))?;
        self.module.latent_extra_rms_append(
            stream,
            &prep,
            rows.x,
            gain,
            pos,
            eps,
            stride,
            off,
            e_off,
            extra,
            n_rows,
            m,
            fault,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue the index cache rows of `m` tokens: one warp per token, four
    /// a block. Refused before the launch: no token, a cache of no rows or
    /// not [`INDEX_ROW`] wide, a key or gate past the row stride, short rows,
    /// positions, weights or biases. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_index_key_append(
        &self,
        stream: &CudaStream,
        a: IndexKeyArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_index_key_append";
        let IndexKeyArgs {
            rows,
            k_off,
            g_off,
            w,
            b,
            pos,
            eps,
            fault,
            cache,
        } = a;
        check_shape(
            what,
            rows,
            &[("key", k_off, INDEX_HEAD), ("gate", g_off, INDEX_HEAD)],
            pos,
            cache,
            INDEX_ROW,
        )?;
        if w.len() < INDEX_HEAD || b.len() < INDEX_HEAD {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.len() {} and b.len() {}: both >= {INDEX_HEAD}",
                    w.len(),
                    b.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", rows.m.div_ceil(4))?;
        let m = launch_u32(what, "m", rows.m)?;
        let stride = launch_u32(what, "stride", rows.stride)?;
        let k_off = launch_u32(what, "k_off", k_off)?;
        let g_off = launch_u32(what, "g_off", g_off)?;
        let n_rows = launch_u32(what, "rows", cache.rows())?;
        let prep = self
            .module
            .prepare_index_key_ln_append(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.index_key_ln_append(
            stream,
            &prep,
            rows.x,
            w,
            b,
            pos,
            eps,
            stride,
            k_off,
            g_off,
            n_rows,
            m,
            fault,
            cache.buf_mut(),
        )?;
        Ok(())
    }
}

impl LatentKernels {
    /// Enqueue the pool keys of `m` tokens: one warp per token, four a
    /// block; a token whose count completes a pool writes it. It must follow
    /// the call's index-key appends and precede its scores, on every call
    /// from position 0 on — a pool no launch wrote is never written.
    /// Refused before the launch: no token, a cache of no rows or not
    /// [`INDEX_ROW`] wide, a plane not [`INDEX_HEAD`] wide or short of
    /// [`pools_for`] the cache's rows, short counts or ape. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_index_pool(
        &self,
        stream: &CudaStream,
        a: IndexPoolArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_index_pool";
        let IndexPoolArgs {
            cache,
            ape,
            n_keys,
            m,
            fault,
            pooled,
        } = a;
        let rows = cache.rows();
        let pools = pools_for(rows);
        if m == 0 || rows == 0 || cache.cols() != INDEX_ROW || pooled.cols() != INDEX_HEAD {
            return Err(GpuError::shape(
                what,
                format!(
                    "need m >= 1, a cache of at least one {INDEX_ROW}-wide row and a                      {INDEX_HEAD}-wide plane; got m={m}, cache {rows}x{}, plane {}x{}",
                    cache.cols(),
                    pooled.rows(),
                    pooled.cols()
                ),
            ));
        }
        let lens = [
            ("pooled rows", pooled.rows(), pools),
            ("ape", ape.len(), POOL * INDEX_HEAD),
            ("n_keys", n_keys.len(), m),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(what, format!("{name} {got} < {need}")));
        }
        let grid = launch_u32(what, "grid", m.div_ceil(4))?;
        let prep = self
            .module
            .prepare_index_pool(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.index_pool(
            stream,
            &prep,
            cache.buf(),
            ape,
            n_keys,
            launch_u32(what, "rows", rows)?,
            launch_u32(what, "pools", pools)?,
            launch_u32(what, "m", m)?,
            fault,
            pooled.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue the gather of the picked cache rows (`kv_gather_sel`): token
    /// `t`'s first `n[t]` list entries' rows of `src` into its rows of `dst`,
    /// in list order, for `a.n.len()` tokens. The grid comes from the list's
    /// width and the token count, not from the counts, so a captured graph
    /// replays at any count. Refused before the launch: no token or width, a
    /// scratch of another width or fewer than `tokens·stride` rows, and
    /// short `list` or `n`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_kv_gather_sel(
        &self,
        stream: &CudaStream,
        a: KvGatherSelArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_kv_gather_sel";
        let KvGatherSelArgs {
            list,
            n,
            stride,
            src,
            fault,
            dst,
        } = a;
        let tokens = n.len();
        let (rows, width) = (src.rows(), src.cols());
        if tokens == 0 || stride == 0 || width == 0 || rows == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "need tokens, stride, width and rows >= 1, got tokens={tokens} stride={stride} \
                     width={width} rows={rows}"
                ),
            ));
        }
        if dst.cols() != width || dst.rows() < tokens * stride {
            return Err(GpuError::shape(
                what,
                format!(
                    "the scratch is {}x{}, want at least {} rows of {width}",
                    dst.rows(),
                    dst.cols(),
                    tokens * stride
                ),
            ));
        }
        if list.len() < tokens * stride {
            return Err(GpuError::shape(
                what,
                format!(
                    "list.len() {} for {tokens} tokens of {stride} entries",
                    list.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", tokens * stride)?;
        let prep = self
            .module
            .prepare_kv_gather_sel(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.kv_gather_sel(
            stream,
            &prep,
            list,
            n,
            launch_u32(what, "stride", stride)?,
            launch_u32(what, "rows", rows)?,
            launch_u32(what, "width", width)?,
            launch_u32(what, "tokens", tokens)?,
            fault,
            src.buf(),
            dst.buf_mut(),
        )?;
        Ok(())
    }
}

/// The refusal of a projection that is not `[INDEX_HEAD × INDEX_HEAD]` q8_0
/// in the q8f32 planes.
fn check_index_proj(what: &'static str, name: &str, w: IndexProj<'_>) -> Result<(), GpuError> {
    let want = (INDEX_HEAD, INDEX_HEAD / 4, INDEX_HEAD, INDEX_HEAD / 32);
    let got = (w.qs.rows(), w.qs.cols(), w.d.rows(), w.d.cols());
    if got == want {
        return Ok(());
    }
    Err(GpuError::shape(
        what,
        format!(
            "the {name} projection's planes are qs {}x{} and d {}x{}; a {INDEX_HEAD}x{INDEX_HEAD} \
             q8_0 weight is qs {}x{} and d {}x{}",
            got.0, got.1, got.2, got.3, want.0, want.1, want.2, want.3
        ),
    ))
}

impl LatentKernels {
    /// Enqueue the forced-tail selector's index rows of `m` tokens
    /// ([`latent_kernels::index_key_rms_append`]): one block of 256 per
    /// token. Refused before the launch: no token or context, an index-row
    /// buffer of no rows or not [`INDEX_ROW`] wide, a key past the row
    /// stride, short rows, positions or gains, a projection not
    /// `[INDEX_HEAD × INDEX_HEAD]` q8_0. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_index_key_rms_append(
        &self,
        stream: &CudaStream,
        a: IndexKeyRmsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_index_key_rms_append";
        let IndexKeyRmsArgs {
            rows,
            k_off,
            gain,
            value,
            gate,
            pos,
            eps,
            ctx,
            fault,
            cache,
        } = a;
        check_shape(
            what,
            rows,
            &[("key", k_off, INDEX_HEAD)],
            pos,
            cache,
            INDEX_ROW,
        )?;
        if ctx == 0 {
            return Err(GpuError::shape(
                what,
                "a context of no position".to_string(),
            ));
        }
        crate::short(what, &[("gain", gain.len(), INDEX_HEAD)])?;
        check_index_proj(what, "value", value)?;
        check_index_proj(what, "gate", gate)?;
        let grid = launch_u32(what, "grid", rows.m)?;
        let prep = self
            .module
            .prepare_index_key_rms_append(LaunchConfig1D::new(grid, KEY_BLOCK, 0))?;
        self.module.index_key_rms_append(
            stream,
            &prep,
            rows.x,
            gain,
            value.qs.buf(),
            value.d.buf(),
            gate.qs.buf(),
            gate.d.buf(),
            pos,
            eps,
            launch_u32(what, "stride", rows.stride)?,
            launch_u32(what, "k_off", k_off)?,
            launch_u32(what, "ring", cache.rows())?,
            launch_u32(what, "ctx", ctx)?,
            launch_u32(what, "m", rows.m)?,
            fault,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue the forced-tail selector's pool keys of `m` tokens
    /// ([`latent_kernels::index_pool_norm_rope`]): one warp per token, four a
    /// block; a token whose count completes a pool writes it. It must follow
    /// the appends of the pool's four positions, with no later position
    /// appended between: an index-row ring of fewer rows than `ctx` holds a
    /// launch of at most `ring − 3` consecutive positions. Refused before the
    /// launch: no token or context, index rows of none or not [`INDEX_ROW`]
    /// wide, a ring that cannot hold the launch's pools, a plane not
    /// [`INDEX_HEAD`] wide or short of [`pools_for`]`(ctx)` rows, a table
    /// short of `ctx` rows of [`ROT`], short counts, ape or gains.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_index_pool_norm_rope(
        &self,
        stream: &CudaStream,
        a: IndexPoolRopeArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_index_pool_norm_rope";
        let IndexPoolRopeArgs {
            cache,
            ape,
            gain,
            table,
            n_keys,
            eps,
            ctx,
            m,
            fault,
            pooled,
        } = a;
        let ring = cache.rows();
        let pools = pools_for(ctx);
        if m == 0
            || ctx == 0
            || ring == 0
            || cache.cols() != INDEX_ROW
            || pooled.cols() != INDEX_HEAD
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "need m and ctx >= 1, index rows of at least one {INDEX_ROW}-wide row and a \
                     {INDEX_HEAD}-wide plane; got m={m} ctx={ctx}, rows {ring}x{}, plane {}x{}",
                    cache.cols(),
                    pooled.rows(),
                    pooled.cols()
                ),
            ));
        }
        if ring < ctx && m + POOL - 1 > ring {
            return Err(GpuError::shape(
                what,
                format!(
                    "a ring of {ring} index rows under a {ctx}-position cache holds a launch of at \
                     most {} positions, got {m}",
                    ring.saturating_sub(POOL - 1)
                ),
            ));
        }
        crate::short(
            what,
            &[
                ("pooled rows", pooled.rows(), pools),
                ("ape", ape.len(), POOL * INDEX_HEAD),
                ("gain", gain.len(), INDEX_HEAD),
                ("table", table.len(), ctx * ROT),
                ("n_keys", n_keys.len(), m),
            ],
        )?;
        let grid = launch_u32(what, "grid", m.div_ceil(4))?;
        let prep = self
            .module
            .prepare_index_pool_norm_rope(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.index_pool_norm_rope(
            stream,
            &prep,
            cache.buf(),
            ape,
            gain,
            table,
            n_keys,
            eps,
            launch_u32(what, "ring", ring)?,
            launch_u32(what, "ctx", ctx)?,
            launch_u32(what, "pools", pools)?,
            launch_u32(what, "m", m)?,
            fault,
            pooled.buf_mut(),
        )?;
        Ok(())
    }
}

/// The sites a host rule raises, as the card's layer mask holds them: bit
/// `site as u32` per site.
fn bit(site: FaultSite) -> u32 {
    1 << site as u32
}

/// The host rule of [`LatentKernels::enqueue_latent_append`]: the module
/// doc's latent contract, op for op, for the `pos.len()` tokens of `x`
/// (`[m][stride]`, the latent at column `off`) into `cache`
/// (`[rows][LATENT]` f16 bits). Returns the mask of the sites the launch
/// raises.
///
/// # Panics
///
/// When `x` holds fewer than `pos.len()` rows or the latent passes a row, or
/// `gain` is shorter than [`LATENT`].
#[must_use]
pub fn latent_append_host(
    x: &[f32],
    stride: usize,
    off: usize,
    gain: &[f32],
    pos: &[u32],
    eps: f32,
    cache: &mut [u16],
) -> u32 {
    let rows = cache.len() / LATENT;
    let mut sites = 0;
    for (t, &p) in pos.iter().enumerate() {
        let v = &x[t * stride + off..t * stride + off + LATENT];
        let warps: Vec<f64> = (0..BLOCK as usize / 32)
            .map(|w| {
                let lanes: Vec<f64> = (0..32)
                    .map(|l| {
                        let tid = w * 32 + l;
                        let mut s = 0.0f64;
                        for i in 0..PER_THREAD {
                            let e = v[tid + 128 * i];
                            s += f64::from(e * e);
                        }
                        s
                    })
                    .collect();
                crate::linear::butterfly_f64(&lanes)
            })
            .collect();
        let sum = ((warps[0] + warps[1]) + warps[2]) + warps[3];
        let mean = (sum / LATENT as f64) as f32;
        let scale = if mean.is_finite() {
            1.0 / (mean + eps).sqrt()
        } else {
            f32::NAN
        };
        let p = p as usize;
        if p >= rows {
            sites |= bit(FaultSite::CachePos);
            continue;
        }
        for (d, &e) in v.iter().enumerate() {
            let h = f32_to_f16_bits((scale * gain[d]) * e);
            if !f16_finite(h) {
                sites |= bit(FaultSite::CacheValue);
            }
            cache[p * LATENT + d] = h;
        }
    }
    sites
}

/// The host rule of [`LatentKernels::enqueue_index_key_append`]: the module
/// doc's index-key contract, op for op, for the `pos.len()` tokens of `x`
/// (`[m][stride]`, the key at column `k_off`, the gate at `g_off`) into
/// `cache` (`[rows][INDEX_ROW]` f16 bits). Returns the mask of the sites the
/// launch raises.
///
/// # Panics
///
/// When `x` holds fewer than `pos.len()` rows or a part passes a row, or `w`
/// or `b` is shorter than [`INDEX_HEAD`].
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's own arguments, as the host rule of one kernel entry"
)]
#[must_use]
pub fn index_key_append_host(
    x: &[f32],
    stride: usize,
    k_off: usize,
    g_off: usize,
    w: &[f32],
    b: &[f32],
    pos: &[u32],
    eps: f32,
    cache: &mut [u16],
) -> u32 {
    let rows = cache.len() / INDEX_ROW;
    let mut sites = 0;
    for (t, &p) in pos.iter().enumerate() {
        let k = &x[t * stride + k_off..t * stride + k_off + INDEX_HEAD];
        let g = &x[t * stride + g_off..t * stride + g_off + INDEX_HEAD];
        let lane_sum = |f: &dyn Fn(usize) -> f64| -> f64 {
            let lanes: Vec<f64> = (0..32)
                .map(|l| (0..PER_LANE).map(|j| f(l + 32 * j)).fold(0.0, |s, e| s + e))
                .collect();
            crate::linear::butterfly_f64(&lanes)
        };
        let mean = (lane_sum(&|i| f64::from(k[i])) / INDEX_HEAD as f64) as f32;
        let dev: Vec<f32> = k.iter().map(|&e| e - mean).collect();
        let var = (lane_sum(&|i| f64::from(dev[i] * dev[i])) / INDEX_HEAD as f64) as f32;
        let scale = if var.is_finite() {
            1.0 / (var + eps).sqrt()
        } else {
            f32::NAN
        };
        let p = p as usize;
        if p >= rows {
            sites |= bit(FaultSite::CachePos);
            continue;
        }
        for d in 0..INDEX_HEAD {
            let hk = f32_to_f16_bits(((dev[d] * scale) * w[d]) + b[d]);
            let hg = f32_to_f16_bits(g[d]);
            if !(f16_finite(hk) && f16_finite(hg)) {
                sites |= bit(FaultSite::CacheValue);
            }
            cache[p * INDEX_ROW + d] = hk;
            cache[p * INDEX_ROW + INDEX_HEAD + d] = hg;
        }
    }
    sites
}

/// The host rule of [`LatentKernels::enqueue_index_pool`]: the module doc's
/// pool contract, op for op, for the counts `n_keys` over `cache`
/// (`[rows][INDEX_ROW]` f16 bits) and `ape` (`[POOL][INDEX_HEAD]`) into
/// `pooled` (`[pools][INDEX_HEAD]` f16 bits). Returns the mask of the sites
/// the launch raises.
///
/// # Panics
///
/// When `ape` is shorter than [`POOL`] rows, or `pooled` than the pools the
/// counts complete.
#[must_use]
pub fn index_pool_host(cache: &[u16], ape: &[f32], n_keys: &[u32], pooled: &mut [u16]) -> u32 {
    let rows = cache.len() / INDEX_ROW;
    let mut sites = 0;
    for &c in n_keys {
        let c = c as usize;
        if c > rows {
            sites |= bit(FaultSite::CachePos);
            continue;
        }
        if c == 0 || !c.is_multiple_of(POOL) {
            continue;
        }
        let j = c / POOL - 1;
        for d in 0..INDEX_HEAD {
            let row = |i: usize| &cache[(POOL * j + i) * INDEX_ROW..][..INDEX_ROW];
            let k: [f32; POOL] = std::array::from_fn(|i| gguf::quant::half_to_f32(row(i)[d]));
            let z: [f32; POOL] = std::array::from_fn(|i| {
                gguf::quant::half_to_f32(row(i)[INDEX_HEAD + d]) + ape[i * INDEX_HEAD + d]
            });
            let mx = z[0].max(z[1]).max(z[2]).max(z[3]);
            let e: [f32; POOL] = std::array::from_fn(|i| expf_ik(z[i] - mx));
            let inv = 1.0 / (((e[0] + e[1]) + e[2]) + e[3]);
            let p = (((e[0] * inv) * k[0] + (e[1] * inv) * k[1]) + (e[2] * inv) * k[2])
                + (e[3] * inv) * k[3];
            let h = f32_to_f16_bits(p);
            if !f16_finite(h) {
                sites |= bit(FaultSite::PoolSelect);
            }
            pooled[j * INDEX_HEAD + d] = h;
        }
    }
    sites
}

/// One `[INDEX_HEAD × INDEX_HEAD]` q8_0 projection's planes on the host: the
/// `qs` words and `d` scale bits [`IndexProj`] holds on the card.
#[derive(Clone, Copy)]
pub struct IndexProjHost<'a> {
    pub qs: &'a [u32],
    pub d: &'a [u16],
}

/// Row `row` of `w` dotted with the normed key `kn` as the forced-tail append
/// forms it: q8f32's word walk at k = 128 (lane `l` the four terms of word
/// `l`, `fma(d·v, x, f)` in byte order from 0), the 32 lane sums by the
/// butterfly.
fn index_proj_row(w: IndexProjHost<'_>, row: usize, kn: &[f32]) -> f32 {
    let lanes: Vec<f32> = (0..32)
        .map(|l| {
            let q = w.qs[row * 32 + l];
            let dv = gguf::quant::half_to_f32(w.d[row * 4 + l / 8]);
            (0..4).fold(0.0f32, |f, b| {
                let v = f32::from((q >> (8 * b)) as u8 as i8);
                (dv * v).mul_add(kn[4 * l + b], f)
            })
        })
        .collect();
    crate::linear::butterfly_f32(&lanes)
}

/// The host rule of [`LatentKernels::enqueue_index_key_rms_append`]: the
/// module doc's forced-tail key contract, op for op, for the `pos.len()`
/// tokens of `x` (`[m][stride]`, the raw key at column `k_off`) into `cache`
/// (`[ring][INDEX_ROW]` f16 bits, position `p` in slot `p mod ring`).
/// Returns the mask of the sites the launch raises.
///
/// # Panics
///
/// When `x` holds fewer than `pos.len()` rows or the key passes a row, `gain`
/// is shorter than [`INDEX_HEAD`], a projection's planes are short, or
/// `cache` holds no row.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's own arguments, as the host rule of one kernel entry"
)]
#[must_use]
pub fn index_key_rms_append_host(
    x: &[f32],
    stride: usize,
    k_off: usize,
    gain: &[f32],
    (value, gate): (IndexProjHost<'_>, IndexProjHost<'_>),
    pos: &[u32],
    eps: f32,
    ctx: usize,
    cache: &mut [u16],
) -> u32 {
    let ring = cache.len() / INDEX_ROW;
    let mut sites = 0;
    for (t, &p) in pos.iter().enumerate() {
        let p = p as usize;
        if p >= ctx {
            sites |= bit(FaultSite::CachePos);
            continue;
        }
        let k = &x[t * stride + k_off..t * stride + k_off + INDEX_HEAD];
        let lanes: Vec<f64> = (0..32)
            .map(|l| (0..PER_LANE).fold(0.0f64, |s, i| s + f64::from(k[4 * l + i] * k[4 * l + i])))
            .collect();
        let mean = (crate::linear::butterfly_f64(&lanes) / INDEX_HEAD as f64) as f32;
        let scale = rms_scale_of_mean(mean, eps);
        let kn: Vec<f32> = (0..INDEX_HEAD).map(|d| (k[d] * scale) * gain[d]).collect();
        let slot = p % ring;
        for (half, w) in [value, gate].into_iter().enumerate() {
            for row in 0..INDEX_HEAD {
                let h = f32_to_f16_bits(index_proj_row(w, row, &kn));
                if !f16_finite(h) {
                    sites |= bit(FaultSite::CacheValue);
                }
                cache[slot * INDEX_ROW + half * INDEX_HEAD + row] = h;
            }
        }
    }
    sites
}

/// The host rule of [`LatentKernels::enqueue_index_pool_norm_rope`]: the
/// module doc's forced-tail pool contract, op for op, for the counts `n_keys`
/// over `cache` (`[ring][INDEX_ROW]` f16 bits, position `p` in slot `p mod
/// ring`), `ape` (`[POOL][INDEX_HEAD]`), the norm's `gain` and `table` (`ctx`
/// rows of [`ROT`]) into `pooled` (`[pools][INDEX_HEAD]` f16 bits). Returns
/// the mask of the sites the launch raises.
///
/// # Panics
///
/// When `ape`, `gain` or `table` is short, `cache` holds no row, or `pooled`
/// is shorter than the pools the counts complete.
#[must_use]
pub fn index_pool_norm_rope_host(
    cache: &[u16],
    (ape, gain, table): (&[f32], &[f32], &[f32]),
    n_keys: &[u32],
    eps: f32,
    ctx: usize,
    pooled: &mut [u16],
) -> u32 {
    let ring = cache.len() / INDEX_ROW;
    let mut sites = 0;
    for &c in n_keys {
        let c = c as usize;
        if c > ctx {
            sites |= bit(FaultSite::CachePos);
            continue;
        }
        if c == 0 || !c.is_multiple_of(POOL) {
            continue;
        }
        let j = c / POOL - 1;
        let row = |i: usize| &cache[((POOL * j + i) % ring) * INDEX_ROW..][..INDEX_ROW];
        let pv: Vec<f32> = (0..INDEX_HEAD)
            .map(|d| {
                let k: [f32; POOL] = std::array::from_fn(|i| gguf::quant::half_to_f32(row(i)[d]));
                let z: [f32; POOL] = std::array::from_fn(|i| {
                    gguf::quant::half_to_f32(row(i)[INDEX_HEAD + d]) + ape[i * INDEX_HEAD + d]
                });
                let mx = z[0].max(z[1]).max(z[2]).max(z[3]);
                let e: [f32; POOL] = std::array::from_fn(|i| expf_ik(z[i] - mx));
                let inv = 1.0 / (((e[0] + e[1]) + e[2]) + e[3]);
                (((e[0] * inv) * k[0] + (e[1] * inv) * k[1]) + (e[2] * inv) * k[2])
                    + (e[3] * inv) * k[3]
            })
            .collect();
        let lanes: Vec<f64> = (0..32)
            .map(|l| {
                (0..PER_LANE).fold(0.0f64, |s, i| s + f64::from(pv[4 * l + i] * pv[4 * l + i]))
            })
            .collect();
        let mean = (crate::linear::butterfly_f64(&lanes) / INDEX_HEAD as f64) as f32;
        let scale = rms_scale_of_mean(mean, eps);
        let mut y: Vec<f32> = (0..INDEX_HEAD).map(|d| (pv[d] * scale) * gain[d]).collect();
        let tab = &table[POOL * j * ROT..][..ROT];
        for i in 0..ROT / 2 {
            let at = INDEX_HEAD - ROT + 2 * i;
            // `rope_pair_rn`'s rounding: the host rounds every op on its own.
            let (a, b, c, s) = (y[at], y[at + 1], tab[2 * i], tab[2 * i + 1]);
            y[at] = a * c - b * s;
            y[at + 1] = a * s + b * c;
        }
        for (d, &v) in y.iter().enumerate() {
            let h = f32_to_f16_bits(v);
            if !f16_finite(h) {
                sites |= bit(FaultSite::PoolSelect);
            }
            pooled[j * INDEX_HEAD + d] = h;
        }
    }
    sites
}
