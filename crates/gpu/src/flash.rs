//! P5: the KV append (f32 -> f16) and the latent (MLA) flash attention over
//! the absorbed cache (docs/gpu-design.md 작업 꾸러미 P5). One cache row per
//! token per layer, `[rope_dims | latent]` u16 with the rope part first —
//! the row order the CPU engine's `kvr` columns carry (`crates/model/src/
//! kv.rs`, oracle-verified there) — and one f32 query row per (token, head),
//! `[q_rope | q_nope2]` in the same order. The attention output is the
//! softmax(scale · q·k)-weighted sum of the rows' latent tails, projected
//! later by `attn_kv_b`'s V part (not this package).
//!
//! The f16 rounding reuses `model::attn::f32_to_f16_bits` itself — the one
//! owner of the CPU oracle's conversion — so cache bits are equal by
//! construction, not by transcription; the gate asserts it element for
//! element on real `kvr-L` rows and on the IEEE edges.
//!
//! Launch geometry: one CUDA block of [`LATENT`] threads per query row, so a
//! thread owns exactly one latent dim of that row's output for the whole run
//! and needs no accumulator array — a decode step's 16 heads are 16 blocks,
//! not 16 warps of two. Everything a thread must share with the rest of the
//! block (the staged query row, a tile's logits, its weights, the rescale)
//! goes through shared memory; nothing is accumulated across blocks, so one
//! launch still produces the whole output.
//!
//! Reduction structure (the fixed, deterministic contract of this family;
//! reruns are bit-identical, CPU bit-identity is not claimed):
//! - QK dot per key: the block splits as [`KEY_TILE`] keys x [`DIM_SPLIT`]
//!   threads. Thread `d` of a key walks dims `d, d + DIM_SPLIT, …` into
//!   [`ILP`] rotating f32 partials (one fused multiply-add each per group of
//!   `ILP` steps, so that many loads are in flight at once), combines them by
//!   `(a0+a1)+(a2+a3)` with the trailing steps folded into `a0`, and the
//!   key's threads combine by the fixed four-step xor butterfly.
//! - Online softmax in `KEY_TILE`-key tiles, keys ascending — the fa4 tile
//!   size. Warp 0 owns the running `(m, s)`: tile max by the five-step
//!   butterfly max, weights `exp(s − m)` per lane, weight sum by the
//!   five-step butterfly sum, state rescale on a max bump (`s` immediately,
//!   the V partials before the tile's accumulation), and publishes the
//!   tile's 32 weights and the rescale to the block.
//! - V accumulation: every thread accumulates its own latent dim over the
//!   tile's keys into [`ILP`] rotating partials (keys `l … l + ILP − 1`), one
//!   fused multiply-add each, combined by the same fixed tree once at the
//!   end.
//! - Final row: `r · (1/s)`, one plain multiply.
//!
//! Keys at or past the causal limit (mask, cache padding) carry weight
//! exactly `0.0`. Only the last tile of a run can reach past the limit, and
//! it takes a guarded path whose `wl == 0.0` test skips those loads — so
//! padded rows holding NaN bit patterns are never read into any result. A
//! tile wholly inside the limit reads only real rows and needs no guard.

use crate::GpuError;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::cvt_f32_f16x2_lo;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The CPU oracle's f32 -> f16 conversion, reused as this module's device
/// and host rounding so the cache bits cannot drift from the engine's.
pub use model::attn::f32_to_f16_bits;

/// Latent tail this kernel family's geometry is built for, and so the block's
/// thread count: one thread per latent dim. `enqueue_flash_latent` rejects
/// any other latent width.
pub const LATENT: usize = 512;
/// Keys per online-softmax tile — one warp's worth, so a tile's max and
/// weight sum are single warp butterflies.
pub const KEY_TILE: usize = 32;
/// Threads sharing one key's QK dot; the block's `LATENT` threads cover
/// `KEY_TILE` keys at a time.
pub const DIM_SPLIT: usize = LATENT / KEY_TILE;
/// Widest `rope_dims + latent` row the shared staging buffer holds.
pub const MAX_WIDTH: usize = 640;
/// Rotating partials each hot loop carries, and so the loads a thread keeps
/// in flight. Four is measured, not derived: eight measured slower at depth
/// on this card, so the loops are not short of memory-level parallelism. The
/// partials are combined by a fixed tree, never by the loop order.
pub const ILP: usize = 4;

// --------------------------------------------------------------- cores

/// `exp(x)` on device: the hardware's `ex2.approx.f32` on `x · log2(e)` — a
/// deterministic instruction whose ~1e-7 relative error sits far inside every
/// band this package gates against, with no dependence on host libm.
#[inline(always)]
pub fn dev_exp(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x * std::f32::consts::LOG2_E)
}

/// One `f16` bit pattern widened to `f32` by the hardware's widening
/// convert. Widening `f16` to `f32` is exact, so this is
/// `cores::half_to_f32`'s value for every finite and infinite input — one
/// instruction instead of a decode whose subnormal branch is a loop.
#[inline(always)]
pub fn half_bits_to_f32(bits: u16) -> f32 {
    cvt_f32_f16x2_lo(bits as u32)
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod flash_kernels {
    use super::*;

    /// Convert `m` new KV rows (`m * width` f32, row-major, token-major) to
    /// f16 with the oracle's rounding and store them at rows
    /// `pos..pos + m` of the preallocated cache. `pos` is a launch scalar:
    /// a captured graph replays it frozen at its capture-time value — the
    /// captured step wants `kv_append_pos_buf`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (src.len() >= m * width, dst.len() >= (pos + m) * width)
    )]
    pub fn kv_append(m: u32, width: u32, pos: u32, src: &[f32], mut dst: DisjointSlice<u16>) {
        let w = width as usize;
        let total = m as usize * w;
        let i = thread::index_1d().get();
        if i >= total {
            return;
        }
        let row = i / w;
        let col = i - row * w;
        // SAFETY: i < total <= src.len(); row < m and col < width, so the
        // store index is < (pos + m) * width <= dst.len() (launch contract).
        unsafe {
            *dst.get_unchecked_mut((pos as usize + row) * w + col) =
                f32_to_f16_bits(*src.get_unchecked(i));
        }
    }

    /// `kv_append` with `pos` read from `pos_buf[0]` at run time — the
    /// variant a captured decode step replays against a new position by
    /// rewriting the buffer between launches. Rows that would land at or
    /// past `dst_rows` (the cache's allocated height) are skipped: `pos`
    /// comes from device memory, so that bound cannot be a launch contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            pos_buf.len() >= 1,
            src.len() >= m * width,
            dst.len() >= dst_rows * width
        )
    )]
    pub fn kv_append_pos_buf(
        m: u32,
        width: u32,
        dst_rows: u32,
        pos_buf: &[u32],
        src: &[f32],
        mut dst: DisjointSlice<u16>,
    ) {
        let w = width as usize;
        let total = m as usize * w;
        let i = thread::index_1d().get();
        if i >= total {
            return;
        }
        // SAFETY: pos_buf.len() >= 1 by the launch contract.
        let pos = unsafe { *pos_buf.get_unchecked(0) } as usize;
        let row = i / w;
        if pos + row >= dst_rows as usize {
            return;
        }
        // SAFETY: i < total <= src.len(); pos + row < dst_rows and col <
        // width, so the store index is < dst_rows * width <= dst.len().
        unsafe {
            let col = i - row * w;
            *dst.get_unchecked_mut((pos + row) * w + col) = f32_to_f16_bits(*src.get_unchecked(i));
        }
    }

    /// Latent flash attention over the absorbed KV cache, `m` query tokens
    /// (1..=8) with every head in one launch. `q` holds `m * n_heads` rows
    /// of `width = rope_dims + latent` f32, row `t * n_heads + h` (the
    /// `kqv_compressed` column order); `kv` is the `[dst_rows x width]` u16
    /// cache; the live key count is `n_keys_buf[0]`. Query `t` attends to
    /// keys `0..n_keys − m + t` (its own rows are already appended, so
    /// `n_keys >= m`); `y` holds `m * n_heads` rows of `latent` f32 in the
    /// same row order. Deterministic by the module doc's reduction contract.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            y.len() >= q_rows * latent
        )
    )]
    pub fn flash_latent(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // The query row, staged once: every key's dot reads all of it.
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        // The tile's scaled logits, then its weights, then the max-bump
        // rescale at [0] and the final `1/s` at [1].
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let lane = warp::lane_id() as usize;
        let rope = rope_dims as usize;
        let lat = latent as usize;
        let width = rope + lat;
        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw pointer form is the only way to reach it
        // without a reference to a `static mut`. Every access below is
        // bounded by the array's length and ordered by `sync_threads`.
        let (qs, klog, kw, st) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QROW),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut ST),
            )
        };

        let mut s = tid;
        while s < width {
            // SAFETY: row < q_rows and s < width bound the load inside q by
            // the launch contract; s < width <= MAX_WIDTH (host-validated)
            // bounds the shared store.
            unsafe {
                *qs.add(s) = *q.get_unchecked(row * width + s);
            }
            s += LATENT;
        }

        // SAFETY: n_keys_buf.len() >= 1 by the launch contract. The clamp
        // makes a lying buffer content read at most the allocated rows.
        let n_keys = (unsafe { *n_keys_buf.get_unchecked(0) } as usize).min(dst_rows as usize);
        let t = row / n_heads as usize;
        let limit = (n_keys + t + 1).saturating_sub(m as usize);
        // This thread's key inside a tile, and its slice of that key's dims.
        let key = tid / DIM_SPLIT;
        let dim0 = tid % DIM_SPLIT;

        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut r0 = 0.0f32;
        let mut r1 = 0.0f32;
        let mut r2 = 0.0f32;
        let mut r3 = 0.0f32;

        thread::sync_threads(); // the staged query row is visible

        let mut blk = 0usize;
        while blk < limit {
            // ---- QK dot: DIM_SPLIT threads per key, four rotating partials
            let mut acc = 0.0f32;
            if blk + key < limit {
                let kb = (blk + key) * width;
                let mut a0 = 0.0f32;
                let mut a1 = 0.0f32;
                let mut a2 = 0.0f32;
                let mut a3 = 0.0f32;
                let mut i = dim0;
                while i + (ILP - 1) * DIM_SPLIT < width {
                    // SAFETY: blk + key < limit <= dst_rows and every dim
                    // index below is < width by this loop's test, so each
                    // load is inside the key's row of kv (launch contract);
                    // the shared reads are below width <= MAX_WIDTH.
                    unsafe {
                        let k0 = half_bits_to_f32(*kv.get_unchecked(kb + i));
                        let k1 = half_bits_to_f32(*kv.get_unchecked(kb + i + DIM_SPLIT));
                        let k2 = half_bits_to_f32(*kv.get_unchecked(kb + i + 2 * DIM_SPLIT));
                        let k3 = half_bits_to_f32(*kv.get_unchecked(kb + i + 3 * DIM_SPLIT));
                        a0 = f32::mul_add(*qs.add(i), k0, a0);
                        a1 = f32::mul_add(*qs.add(i + DIM_SPLIT), k1, a1);
                        a2 = f32::mul_add(*qs.add(i + 2 * DIM_SPLIT), k2, a2);
                        a3 = f32::mul_add(*qs.add(i + 3 * DIM_SPLIT), k3, a3);
                    }
                    i += ILP * DIM_SPLIT;
                }
                while i < width {
                    // SAFETY: as above, for the trailing steps.
                    unsafe {
                        a0 = f32::mul_add(
                            *qs.add(i),
                            half_bits_to_f32(*kv.get_unchecked(kb + i)),
                            a0,
                        );
                    }
                    i += DIM_SPLIT;
                }
                acc = (a0 + a1) + (a2 + a3);
            }
            // The key's DIM_SPLIT threads are one aligned lane group, so the
            // four-step butterfly stays inside the key. Every thread calls
            // it: the guard above shapes the value, never the control flow.
            acc += warp::shuffle_xor_f32(acc, 1);
            acc += warp::shuffle_xor_f32(acc, 2);
            acc += warp::shuffle_xor_f32(acc, 4);
            acc += warp::shuffle_xor_f32(acc, 8);
            if dim0 == 0 {
                let sv = if blk + key < limit {
                    scale * acc
                } else {
                    f32::NEG_INFINITY
                };
                // SAFETY: key < KEY_TILE, and one thread per key writes it.
                unsafe {
                    *klog.add(key) = sv;
                }
            }
            thread::sync_threads();

            // ---- online softmax: warp 0 owns (m, s) and publishes the tile
            if tid < KEY_TILE {
                // SAFETY: lane == tid < KEY_TILE here.
                let sv = unsafe { *klog.add(lane) };
                let smax = warp::reduce_max_f32(sv);
                // FlashMS update: s is rescaled here, the V partials just
                // before this tile's accumulation (the CPU oracle's order).
                // A first bump scales by 0.0 — the partials are still zero,
                // so the reset and the scale are the same value.
                let mut vms = 1.0f32;
                if smax > mx {
                    vms = if mx > f32::NEG_INFINITY {
                        dev_exp(mx - smax)
                    } else {
                        0.0
                    };
                    s_sum *= vms;
                    mx = smax;
                }
                let w = if sv == f32::NEG_INFINITY {
                    0.0
                } else {
                    dev_exp(sv - mx)
                };
                s_sum += warp::reduce_sum_f32(w);
                // SAFETY: lane < KEY_TILE; one lane writes each slot, and
                // lane 0 alone writes the rescale.
                unsafe {
                    *kw.add(lane) = w;
                    if lane == 0 {
                        *st.add(0) = vms;
                    }
                }
            }
            thread::sync_threads();

            // ---- V accumulation: this thread's own latent dim, four keys
            // in flight. SAFETY: ST[0] and KW hold this tile's published
            // values (both barriers above).
            let vms = unsafe { *st.add(0) };
            r0 *= vms;
            r1 *= vms;
            r2 *= vms;
            r3 *= vms;
            let tail = rope + tid;
            if blk + KEY_TILE <= limit {
                let mut l = 0usize;
                while l < KEY_TILE {
                    // SAFETY: the whole tile is below limit <= dst_rows, so
                    // all ILP rows' tails are inside kv (launch contract);
                    // tail < width and l + ILP <= KEY_TILE.
                    unsafe {
                        let base = (blk + l) * width + tail;
                        let v0 = half_bits_to_f32(*kv.get_unchecked(base));
                        let v1 = half_bits_to_f32(*kv.get_unchecked(base + width));
                        let v2 = half_bits_to_f32(*kv.get_unchecked(base + 2 * width));
                        let v3 = half_bits_to_f32(*kv.get_unchecked(base + 3 * width));
                        r0 = f32::mul_add(*kw.add(l), v0, r0);
                        r1 = f32::mul_add(*kw.add(l + 1), v1, r1);
                        r2 = f32::mul_add(*kw.add(l + 2), v2, r2);
                        r3 = f32::mul_add(*kw.add(l + 3), v3, r3);
                    }
                    l += ILP;
                }
            } else {
                // The run's last tile is the only one that reaches past the
                // limit: a zero weight is exactly a row this must not load.
                let mut l = 0usize;
                while l < KEY_TILE {
                    // SAFETY: l < KEY_TILE.
                    let wl = unsafe { *kw.add(l) };
                    if wl != 0.0 {
                        // SAFETY: wl != 0 => key blk+l is live => blk+l <
                        // limit <= dst_rows, so its tail is inside kv.
                        unsafe {
                            r0 = f32::mul_add(
                                wl,
                                half_bits_to_f32(*kv.get_unchecked((blk + l) * width + tail)),
                                r0,
                            );
                        }
                    }
                    l += 1;
                }
            }
            blk += KEY_TILE;
        }

        if tid == 0 {
            // SAFETY: thread 0 alone writes ST[1]; no thread reads it before
            // the barrier below.
            unsafe {
                *st.add(1) = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            }
        }
        thread::sync_threads();
        // SAFETY: ST[1] is published above.
        let s_inv = unsafe { *st.add(1) };
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract).
        unsafe {
            *y.get_unchecked_mut(row * lat + tid) = s_inv * ((r0 + r1) + (r2 + r3));
        }
    }
}

/// The loaded P5 device module: `kv_append`, `kv_append_pos_buf`,
/// `flash_latent`. Owns no context and no stream — every enqueue takes the
/// engine stream (`Gpu::stream()`), so launches order with the rest of the
/// step and are capturable.
pub struct FlashKernels {
    module: flash_kernels::LoadedModule,
}

impl FlashKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { flash_kernels::load(ctx)? };
        Ok(FlashKernels { module })
    }

    /// Enqueue the f32 -> f16 append of `m` new rows of `cache.cols()` f32
    /// (`src.len() >= m * width`, token-major) at rows `pos..pos + m` of the
    /// `[cache.rows() x width]` u16 cache. `pos` is a launch scalar and is
    /// frozen inside a captured graph — for the captured decode step use
    /// [`FlashKernels::enqueue_kv_append_pos_buf`]. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_kv_append(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        cache: &mut DeviceTensor<u16>,
        m: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        let (rows, width) = (cache.rows(), cache.cols());
        check_append("enqueue_kv_append", src.len(), width, m)?;
        if pos as usize + m > rows {
            return Err(format!(
                "enqueue_kv_append: rows {}..{} land past the cache's {rows} rows",
                pos,
                pos as usize + m
            )
            .into());
        }
        let prep = self.module.prepare_kv_append(LaunchConfig1D::new(
            (m * width).div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module.kv_append(
            stream,
            &prep,
            m as u32,
            width as u32,
            pos,
            src,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_kv_append`] with `pos` read from
    /// `pos_buf[0]` on the device at run time — the captured-graph form: the
    /// graph replays unchanged while the position advances through the
    /// buffer. Rows past the cache's height are skipped in-kernel. `pos_buf`
    /// must stay allocated and in place for as long as a captured graph
    /// replaying this launch lives. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_kv_append_pos_buf(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        pos_buf: &DeviceBuffer<u32>,
        cache: &mut DeviceTensor<u16>,
        m: usize,
    ) -> Result<(), GpuError> {
        let (rows, width) = (cache.rows(), cache.cols());
        if pos_buf.len() < 1 {
            return Err("enqueue_kv_append_pos_buf: pos_buf must hold 1 u32".into());
        }
        check_append("enqueue_kv_append_pos_buf", src.len(), width, m)?;
        let prep = self.module.prepare_kv_append_pos_buf(LaunchConfig1D::new(
            (m * width).div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module.kv_append_pos_buf(
            stream,
            &prep,
            m as u32,
            width as u32,
            rows as u32,
            pos_buf,
            src,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue the latent flash attention: `q` holds `m * n_heads` rows of
    /// `rope_dims + latent` f32 (row `t * n_heads + h`, content
    /// `[q_rope | q_nope2]`), `kv` is the `[rows x width]` u16 cache, and
    /// `n_keys_buf[0]` names the live key count (`>= m`: the batch's own
    /// rows are appended first) — query `t` attends to keys
    /// `0..n_keys − m + t`. `y` holds `m * n_heads` rows of `latent` f32,
    /// the `kqv_compressed` order. `scale` is `MlaParams::kq_scale` (the
    /// YaRN mscale is inside it — not `1/√d`). `latent` must be [`LATENT`],
    /// this family's block width; `rope_dims` is free as long as
    /// `rope_dims + latent` is a multiple of [`DIM_SPLIT`] and at most
    /// [`MAX_WIDTH`] (the shared staging row). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_flash_latent(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer<f32>,
        kv: &DeviceTensor<u16>,
        n_keys_buf: &DeviceBuffer<u32>,
        scale: f32,
        m: usize,
        n_heads: usize,
        rope_dims: usize,
        latent: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if !(1..=8).contains(&m) {
            return Err(format!("enqueue_flash_latent: 1 <= m <= 8, got {m}").into());
        }
        if n_heads == 0 {
            return Err("enqueue_flash_latent: n_heads >= 1".into());
        }
        if latent != LATENT {
            return Err(format!(
                "enqueue_flash_latent: this family's latent tail is {LATENT} (one block \
                 thread per dim), got {latent}"
            )
            .into());
        }
        let width = rope_dims + latent;
        if width % DIM_SPLIT != 0 {
            return Err(format!(
                "enqueue_flash_latent: the QK dot splits a row across {DIM_SPLIT} threads, \
                 need rope_dims + latent = {width} a multiple of {DIM_SPLIT}"
            )
            .into());
        }
        if width > MAX_WIDTH {
            return Err(format!(
                "enqueue_flash_latent: the query row is staged in {MAX_WIDTH} shared f32, \
                 got rope_dims + latent = {width}"
            )
            .into());
        }
        if kv.cols() != width {
            return Err(format!(
                "enqueue_flash_latent: kv is {}-wide, want rope_dims + latent = {width}",
                kv.cols()
            )
            .into());
        }
        if n_keys_buf.len() < 1 {
            return Err("enqueue_flash_latent: n_keys_buf must hold 1 u32".into());
        }
        let q_rows = m * n_heads;
        if q.len() < q_rows * width {
            return Err(format!(
                "enqueue_flash_latent: q.len() {} < m*n_heads*width = {}",
                q.len(),
                q_rows * width
            )
            .into());
        }
        if y.len() < q_rows * latent {
            return Err(format!(
                "enqueue_flash_latent: y.len() {} < m*n_heads*latent = {}",
                y.len(),
                q_rows * latent
            )
            .into());
        }
        let prep = self.module.prepare_flash_latent(LaunchConfig1D::new(
            q_rows as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_latent(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            rope_dims as u32,
            latent as u32,
            kv.rows() as u32,
            y,
        )?;
        Ok(())
    }
}

/// Reject geometry the append kernels' launch contracts do not cover:
/// positive width, `m` rows, and `src` holding all of them. The landing-row
/// bound is checked only where `pos` is a host scalar.
fn check_append(what: &str, src_len: usize, width: usize, m: usize) -> Result<(), GpuError> {
    if width == 0 || m == 0 {
        return Err(format!("{what}: need width >= 1 and m >= 1, got {width}/{m}").into());
    }
    if src_len < m * width {
        return Err(format!("{what}: src.len() {src_len} < m*width = {}", m * width).into());
    }
    Ok(())
}
