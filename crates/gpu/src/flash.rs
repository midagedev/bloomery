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
//! Reduction structure (the fixed, deterministic contract of this family;
//! reruns are bit-identical, CPU bit-identity is not claimed):
//! - QK dot per key: one lane owns one key; the row walks 64-value chunks,
//!   chunk octet `o` always accumulates into partial `p[o]` (eight rotating
//!   f32 partials, 8 fused multiply-adds per octet), combined by the fixed
//!   tree `((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))`.
//! - Online softmax in 32-key blocks, keys ascending — the fa4 block size:
//!   block max by the warp's five-step butterfly max, weights `exp(s − m)`
//!   per lane, weight sum by the five-step butterfly sum, state rescale on
//!   a max bump (`S` immediately, `R` before the block's accumulation).
//! - V accumulation: lane `l` of the 32-key block broadcasts its weight with
//!   one shuffle per key; every lane accumulates the key's latent tail into
//!   its own 16 dims (`lane + 32·j`), one fused multiply-add per element.
//! - Final row: `R · (1/S)`, one plain multiply per element.
//!
//! Keys at or past the causal limit (mask, cache padding) carry weight
//! exactly `0.0`, and the V loop's `wl == 0.0` guard skips their loads — so
//! padded rows holding NaN bit patterns are never read into any result.

use crate::GpuError;
use crate::cores::half_to_f32;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The CPU oracle's f32 -> f16 conversion, reused as this module's device
/// and host rounding so the cache bits cannot drift from the engine's.
pub use model::attn::f32_to_f16_bits;

/// Latent tail this kernel family's register geometry is built for: one warp
/// spreads the 512-wide tail as 16 dims per lane. `enqueue_flash_latent`
/// rejects any other latent width.
const LATENT_LANES: usize = 16;

// --------------------------------------------------------------- cores

/// `exp(x)` on device: the hardware's `ex2.approx.f32` on `x · log2(e)` — a
/// deterministic instruction whose ~1e-7 relative error sits far inside every
/// band this package gates against, with no dependence on host libm.
#[inline(always)]
pub fn dev_exp(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x * std::f32::consts::LOG2_E)
}

/// One key row's logit numerator: `Σ_d q[d] · f32(f16 k[d])` over the full
/// row width in the module doc's fixed partial order.
///
/// Caller contract: `q.len() == k.len() == width`, `width % 8 == 0`
/// (host-validated; every octet is then wholly inside or wholly past the
/// row).
#[inline(always)]
pub fn kq_dot_row(q: &[f32], k: &[u16], width: usize) -> f32 {
    let mut p = [0.0f32; 8];
    let mut d = 0usize;
    while d < width {
        let mut o = 0usize;
        while o < 8 {
            let b = d + 8 * o;
            if b < width {
                // SAFETY: b + 8 <= width (width % 8 == 0), so the octet is
                // inside both rows' `width` values.
                unsafe {
                    p[o] =
                        f32::mul_add(*q.get_unchecked(b), half_to_f32(*k.get_unchecked(b)), p[o]);
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 1),
                        half_to_f32(*k.get_unchecked(b + 1)),
                        p[o],
                    );
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 2),
                        half_to_f32(*k.get_unchecked(b + 2)),
                        p[o],
                    );
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 3),
                        half_to_f32(*k.get_unchecked(b + 3)),
                        p[o],
                    );
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 4),
                        half_to_f32(*k.get_unchecked(b + 4)),
                        p[o],
                    );
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 5),
                        half_to_f32(*k.get_unchecked(b + 5)),
                        p[o],
                    );
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 6),
                        half_to_f32(*k.get_unchecked(b + 6)),
                        p[o],
                    );
                    p[o] = f32::mul_add(
                        *q.get_unchecked(b + 7),
                        half_to_f32(*k.get_unchecked(b + 7)),
                        p[o],
                    );
                }
            }
            o += 1;
        }
        d += 64;
    }
    ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
}

/// One 32-key block of the online softmax for one (token, head) query row.
/// Lane `l` owns key `blk + l`: it computes that key's scaled logit (a key at
/// or past `limit` stays `−inf`), the warp reduces the block max, the running
/// state `(m, s_sum, r)` is rescaled on a max bump, the block's weight sum is
/// added, and the weighted latent tails accumulate into `r` — lane-owned dims
/// `lane + 32·j` over all 32 keys of the block. A key whose weight is
/// exactly `0.0` (masked, padded, or underflowed) is never loaded: the guard
/// is what keeps NaN padding out of the result and past-`limit` rows out of
/// memory.
///
/// Caller contract (warp-collective: all 32 lanes call together, converged):
/// `q.len() == width`, `kv.len() >= rows * width` with `limit <= rows`,
/// `blk < limit`, `rope + 32 * r.len() <= width`, `lane < 32`.
#[inline(always)]
pub fn flash_block(
    q: &[f32],
    kv: &[u16],
    blk: usize,
    limit: usize,
    scale: f32,
    rope: usize,
    lane: usize,
    m: &mut f32,
    s_sum: &mut f32,
    r: &mut [f32; LATENT_LANES],
) {
    let width = q.len();
    let live = blk + lane < limit;
    let s = if live {
        // SAFETY: live => blk + lane + 1 <= limit <= rows, so the row slice
        // [blk+lane, blk+lane+1) * width is inside kv.
        let k = unsafe { kv.get_unchecked((blk + lane) * width..(blk + lane + 1) * width) };
        scale * kq_dot_row(q, k, width)
    } else {
        f32::NEG_INFINITY
    };
    let smax = warp::reduce_max_f32(s);
    if smax == f32::NEG_INFINITY {
        return; // fully masked block: every weight is exactly 0, a no-op
    }
    // FlashMS update: S is rescaled here, R just before this block's
    // accumulation (the CPU oracle's order).
    let mut vms = 1.0f32;
    let mut need = 0u8;
    if smax > *m {
        if *m > f32::NEG_INFINITY {
            vms = dev_exp(*m - smax);
            need = 1;
        } else {
            need = 2;
        }
        *m = smax;
    }
    if need == 1 {
        *s_sum *= vms;
    } else if need == 2 {
        *s_sum = 0.0;
    }
    let w = if s == f32::NEG_INFINITY {
        0.0
    } else {
        dev_exp(s - *m)
    };
    *s_sum += warp::reduce_sum_f32(w);
    if need == 1 {
        for v in r.iter_mut() {
            *v *= vms;
        }
    } else if need == 2 {
        *r = [0.0f32; LATENT_LANES];
    }
    // V accumulation: one broadcast per key, lane-owned dims inner.
    let mut l = 0usize;
    while l < 32 {
        let wl = warp::shuffle_f32(w, l as u32);
        if wl != 0.0 {
            // SAFETY: wl != 0 => key blk+l is live => blk+l+1 <= limit <=
            // rows, so the row's [rope, width) tail is inside kv.
            let tail = unsafe { kv.get_unchecked((blk + l) * width + rope..(blk + l + 1) * width) };
            let mut j = 0usize;
            while j < LATENT_LANES {
                // SAFETY: lane + 32j < 32 * LATENT_LANES <= width - rope by
                // the caller contract.
                r[j] = f32::mul_add(
                    wl,
                    half_to_f32(unsafe { *tail.get_unchecked(lane + 32 * j) }),
                    r[j],
                );
                j += 1;
            }
        }
        l += 1;
    }
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
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
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
        let t256 = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t256 / 32;
        if row >= q_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let rope = rope_dims as usize;
        let lat = latent as usize;
        let width = rope + lat;
        // SAFETY: row < q_rows, so [row, row+1) * width is inside q (launch
        // contract).
        let qrow = unsafe { q.get_unchecked(row * width..(row + 1) * width) };
        // SAFETY: n_keys_buf.len() >= 1 by the launch contract. The clamp
        // makes a lying buffer content read at most the allocated rows.
        let n_keys = (unsafe { *n_keys_buf.get_unchecked(0) } as usize).min(dst_rows as usize);
        let t = row / n_heads as usize;
        let limit = (n_keys + t + 1).saturating_sub(m as usize);
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut r = [0.0f32; LATENT_LANES];
        let mut blk = 0usize;
        while blk < limit {
            flash_block(
                qrow, kv, blk, limit, scale, rope, lane, &mut mx, &mut s_sum, &mut r,
            );
            blk += 32;
        }
        let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
        let mut j = 0usize;
        while j < LATENT_LANES {
            // SAFETY: row < q_rows and lane + 32j < 32 * LATENT_LANES =
            // latent (host-validated), so the store is inside y's row
            // segment (launch contract).
            unsafe {
                *y.get_unchecked_mut(row * lat + lane + 32 * j) = s_inv * r[j];
            }
            j += 1;
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
    /// YaRN mscale is inside it — not `1/√d`). `latent` must be
    /// `32 * 16 = 512`, this family's register geometry; `rope_dims` is
    /// free. Asynchronous, allocation-free, capturable.
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
        if latent != 32 * LATENT_LANES {
            return Err(format!(
                "enqueue_flash_latent: this family's latent tail is 32*{LATENT_LANES} = {} \
                 (one warp spreads it as {LATENT_LANES} dims per lane), got {latent}",
                32 * LATENT_LANES
            )
            .into());
        }
        let width = rope_dims + latent;
        if width % 8 != 0 {
            return Err(format!(
                "enqueue_flash_latent: the QK dot walks 8-value octets, need \
                 rope_dims + latent = {width} a multiple of 8"
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
            q_rows.div_ceil(8) as u32,
            256,
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
