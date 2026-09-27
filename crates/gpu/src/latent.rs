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
//! Numeric contract (the host rules [`latent_append_host`] and
//! [`index_key_append_host`] are these lists, op for op). Latent, one block
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
//! This is ik's CPU order (`fused_rms_norm`, `norm`, then `mul` and `add`)
//! but for the f64 sums, which ik adds serially: two f64 sums of the same
//! f32 terms differ far below one f32 ulp of the mean, so the rounded mean
//! (and variance) agree unless the exact value sits that close to an f32
//! rounding boundary. The f16 rounding is `f32_to_f16_bits`, to nearest
//! even.
//!
//! No silent failure: a position at or past the cache's rows raises
//! [`FaultSite::CachePos`] and the token writes nothing; a mean square (or
//! variance) that is not finite makes the scale NaN, and a stored value that
//! is not finite — a non-finite input, a NaN scale, a value past f16's range
//! — raises [`FaultSite::CacheValue`] with the row written as computed,
//! never as a plausible one. The tokens of one launch hold distinct
//! positions (the caller's contract), so no two blocks write one row.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::f32_to_f16_bits;
use crate::launch_u32;
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

/// Threads per block of both entries: four warps.
const BLOCK: u32 = 128;
/// Latent values each thread owns.
const PER_THREAD: usize = LATENT / BLOCK as usize;
/// Index-key values each lane owns.
const PER_LANE: usize = INDEX_HEAD / 32;
const _: () = assert!(PER_THREAD == 4 && PER_LANE == 4 && BLOCK == 128);

/// Whether an f16 bit pattern is finite.
#[inline(always)]
fn f16_finite(h: u16) -> bool {
    h & 0x7c00 != 0x7c00
}

/// The xor butterfly (16, 8, 4, 2, 1) over a warp in f64: every lane ends
/// with the same sum.
#[inline(always)]
fn warp_sum_f64(v: f64) -> f64 {
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
        let module = unsafe { latent_kernels::load(ctx)? };
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
