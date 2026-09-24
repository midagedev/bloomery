//! Rotary position embedding on the rope tail of each head, and its inverse
//! on the tail of the attention output (the same rotation at the negated
//! angle). The layer picks the base: plain rope on the window-only layers,
//! YaRN-scaled on the compressed ones.
//!
//! V4.1 rotates only the last `n_dims` values of a head — ggml's rope with
//! `op_params[15] = 1`, offset `ne0 − n_dims` — in adjacent pairs (NORM
//! mode): values 448.. of a 512-value attention head, 64.. of a 128-value
//! index head. The cos/sin table is host work, one per position
//! ([`RopeTable`]); the kernels apply it:
//! - [`RopeKernels::enqueue_rope_tail`] rotates heads in place — the query
//!   heads, the attention output (with a [`Direction::Back`] table), the
//!   pooled compressed rows, the index keys and the indexer query;
//! - [`RopeKernels::enqueue_kv_norm_rope_append`] is a token's latent K/V row
//!   in one launch: its norm, its tail rope, the row in f32, its f16 slot
//!   in the layer's raw window ring and the same f16 row at its position in
//!   the layer's shadow of that ring.
//!
//! Numeric contract: the table is ggml's recipe as ik's CPU build compiles it
//! ([`ggml_rope_cache`]), and every rotation rounds each product and each sum
//! on its own (`rope_pair_rn`) — ik's CPU rotation is unfused, so on the
//! same input the kernels reproduce its output bit for bit. The K/V norm is
//! `elem::rms_norm`'s f32 tree where ik sums in f64; that sum is the one
//! difference the K/V row carries.

use bloomery_gpu::elem::{RMS_THREADS, RMS_WARPS, rms_partial_sq, rms_scale, rms_warp_tree};
use bloomery_gpu::flash::f32_to_f16_bits;
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of both entries; `ds41_kv_norm_rope_append` is
/// `elem::rms_norm`'s geometry, one [`RMS_THREADS`] block per token.
const BLOCK: u32 = 256;
const _: () = assert!(BLOCK as usize == RMS_THREADS);

// ------------------------------------------------------------------- host

// The table and the rotation core are shared with every ggml-rope
// architecture; this module keeps their V4.1 paths.
pub(crate) use bloomery_gpu::rope_table::rope_pair_rn;
pub use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable, ggml_rope_cache};

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod rope_kernels {
    use super::*;

    /// Turn the last `n_dims` values of every head in place: head
    /// `r = t·n_vec + v` is the `width` values at `x[r·width ..]`, its tail
    /// pairs start at `o = width − n_dims`, and pair `i` turns by token `t`'s
    /// table pair `cs[t·n_dims + 2i ..]` through `rope_pair_rn`. One thread
    /// per (head, pair); the values before `o` are not touched. `n_dims` even
    /// and at most `width` (host-checked).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= m * n_vec * width, cs.len() >= m * n_dims)
    )]
    pub fn ds41_rope_tail(
        cs: &[f32],
        width: u32,
        n_dims: u32,
        n_vec: u32,
        m: u32,
        mut x: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let npairs = (n_dims >> 1) as usize;
        if i >= m as usize * n_vec as usize * npairs {
            return;
        }
        let r = i / npairs;
        let p = i - r * npairs;
        let t = r / n_vec as usize;
        let a = r * width as usize + (width - n_dims) as usize + 2 * p;
        let tb = t * n_dims as usize + 2 * p;
        // SAFETY: t < m and 2p + 1 < n_dims, so tb + 1 < m·n_dims <=
        // cs.len() by the launch contract.
        let (c, s) = unsafe { (*cs.get_unchecked(tb), *cs.get_unchecked(tb + 1)) };
        // SAFETY: r < m·n_vec and a + 1 <= r·width + width − 1 because
        // n_dims <= width (host-checked), so both reads stay below
        // m·n_vec·width <= x.len(); this thread is the only one that touches
        // positions a and a + 1.
        let (x0, x1) = unsafe { (*x.get_unchecked_mut(a), *x.get_unchecked_mut(a + 1)) };
        let (y0, y1) = rope_pair_rn(x0, x1, c, s);
        // SAFETY: the same two positions as the reads.
        unsafe {
            *x.get_unchecked_mut(a) = y0;
            *x.get_unchecked_mut(a + 1) = y1;
        }
    }

    /// The latent K/V rows of `m` tokens, one [`RMS_THREADS`] block per
    /// token: the norm of the whole `width`-value row — `elem::rms_norm`'s
    /// body (`rms_partial_sq`, the warp butterfly, `rms_warp_tree`,
    /// `rms_scale`, then `(scale · gain) · x`) — then the turn of its last
    /// `n_dims` values by token `t`'s table through `rope_pair_rn`. The row
    /// is stored in f32 at `out[t·width ..]` and rounded once to f16
    /// (`f32_to_f16_bits`, round to nearest even) into slot `pos[t] %
    /// window` of the ring `cache` (`window` rows of `width`), and the same
    /// bits into row `pos[t]` of `shadow` (`rows` rows of `width`, one per
    /// position; a position at or past `rows` writes no shadow row). `width`
    /// a positive multiple of 32, `n_dims` even and at most `width` and
    /// `2·RMS_THREADS` (host-checked); the tokens of one launch land in
    /// distinct slots (host-checked as `m <= window` — the caller's
    /// positions are consecutive). The token guard and the shadow guard are
    /// block-uniform, so no barrier and no warp collective is skipped.
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
            kv.len() >= m * width,
            gain.len() >= width,
            cs.len() >= m * n_dims,
            pos.len() >= m,
            out.len() >= m * width,
            cache.len() >= window * width,
            shadow.len() >= rows * width
        )
    )]
    pub fn ds41_kv_norm_rope_append(
        kv: &[f32],
        gain: &[f32],
        cs: &[f32],
        pos: &[u32],
        eps: f32,
        width: u32,
        n_dims: u32,
        window: u32,
        rows: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
        mut cache: DisjointSlice<u16>,
        mut shadow: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let k = width as usize;
        let nd = n_dims as usize;
        let head = k - nd;
        let base = t * k;

        // The norm's scale, `elem::rms_norm`'s phase A.
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        let part = warp::reduce_sum_f32(rms_partial_sq(kv, base, k, tid));
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < RMS_WARPS; one lane per warp writes its slot.
            unsafe {
                *ws.add(tid / 32) = part;
            }
        }
        thread::sync_threads();
        // SAFETY: block-shared, RMS_WARPS slots, written before the barrier
        // that publishes them.
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
        let scale = rms_scale(rms_warp_tree(sums), width, eps);

        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) };
        let slot = (p % window) as usize;
        let crow = slot * k;
        let shadowed = p < rows;
        let srow = p as usize * k;

        // The values before the tail: normalized, stored twice.
        let mut it = tid;
        while it < head {
            // SAFETY: it < head < width bounds the gain read by the contract
            // and the kv read and out write as base + it < m·width; the
            // cache slot is crow + it < (slot + 1)·width <= window·width <=
            // cache.len() because slot < window. One thread per value.
            let h = unsafe {
                let nv = (scale * *gain.get_unchecked(it)) * *kv.get_unchecked(base + it);
                *out.get_unchecked_mut(base + it) = nv;
                let h = f32_to_f16_bits(nv);
                *cache.get_unchecked_mut(crow + it) = h;
                h
            };
            if shadowed {
                // SAFETY: p < rows, so srow + it < (p + 1)·width <= rows·width
                // <= shadow.len() by the contract. One thread per value.
                unsafe {
                    *shadow.get_unchecked_mut(srow + it) = h;
                }
            }
            it += RMS_THREADS;
        }

        // The tail: one pair per thread, normalized, then turned.
        if 2 * tid < nd {
            let j = head + 2 * tid;
            let tb = t * nd + 2 * tid;
            // SAFETY: j + 1 < width bounds the gain reads by the contract and
            // the kv reads as base + j + 1 < m·width; tb + 1 < (t + 1)·n_dims
            // <= cs.len().
            let (n0, n1, c, s) = unsafe {
                (
                    (scale * *gain.get_unchecked(j)) * *kv.get_unchecked(base + j),
                    (scale * *gain.get_unchecked(j + 1)) * *kv.get_unchecked(base + j + 1),
                    *cs.get_unchecked(tb),
                    *cs.get_unchecked(tb + 1),
                )
            };
            let (y0, y1) = rope_pair_rn(n0, n1, c, s);
            let (h0, h1) = (f32_to_f16_bits(y0), f32_to_f16_bits(y1));
            // SAFETY: the out positions are the kv positions above, inside
            // m·width; the cache positions crow + j + 1 < (slot + 1)·width <=
            // cache.len(). This thread owns the pair.
            unsafe {
                *out.get_unchecked_mut(base + j) = y0;
                *out.get_unchecked_mut(base + j + 1) = y1;
                *cache.get_unchecked_mut(crow + j) = h0;
                *cache.get_unchecked_mut(crow + j + 1) = h1;
            }
            if shadowed {
                // SAFETY: p < rows, so srow + j + 1 < (p + 1)·width <=
                // rows·width <= shadow.len() by the contract. This thread owns
                // the pair.
                unsafe {
                    *shadow.get_unchecked_mut(srow + j) = h0;
                    *shadow.get_unchecked_mut(srow + j + 1) = h1;
                }
            }
        }
    }
}

// -------------------------------------------------------------- launchers

/// A head layout the tail rope walks: `m` tokens of `n_vec` heads of `width`
/// values each, token-major, the last `n_dims` values of each head turned.
#[derive(Clone, Copy, Debug)]
pub struct TailShape {
    pub width: usize,
    pub n_dims: usize,
    pub n_vec: usize,
    pub m: usize,
}

/// [`RopeKernels::enqueue_kv_norm_rope_append`]'s arguments. `kv` holds `m`
/// rows of the ring's width (the `kv_b` projection, token-major), `gain` the
/// row's norm weights, `cs` `m` tables of `n_dims` ([`RopeTable::push`]),
/// `pos` the `m` positions on the device; `out` takes the rows in f32,
/// `cache` is the layer's raw window ring, `window` rows of `width` f16, and
/// `shadow` its shadow, one row of `width` f16 per position.
pub struct KvAppendArgs<'a> {
    pub kv: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub cs: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub n_dims: usize,
    pub m: usize,
    pub out: &'a mut DeviceBuffer<f32>,
    pub cache: &'a mut DeviceTensor<u16>,
    pub shadow: &'a mut DeviceTensor<u16>,
}

/// The loaded rope module. Owns no stream: each enqueue takes the engine
/// stream, so its launches order with the rest of the step and capture.
pub struct RopeKernels {
    module: rope_kernels::LoadedModule,
}

impl RopeKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RopeKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { rope_kernels::load(ctx)? };
        Ok(RopeKernels { module })
    }

    /// Enqueue the tail rope of `x` in place, `shape`'s heads, token `t`'s
    /// table at `cs[t·n_dims ..]` ([`RopeTable::push`]; a
    /// [`Direction::Back`] table turns the other way). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_rope_tail(
        &self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<f32>,
        cs: &DeviceBuffer<f32>,
        shape: TailShape,
    ) -> Result<(), GpuError> {
        let what = "enqueue_rope_tail";
        let TailShape {
            width,
            n_dims,
            n_vec,
            m,
        } = shape;
        if n_dims < 2 || !n_dims.is_multiple_of(2) || n_dims > width {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n_dims must be even, at least 2 and at most width {width}, got {n_dims}"
                ),
            });
        }
        if n_vec == 0 || m == 0 {
            return Err(GpuError::Shape {
                what,
                detail: format!("need n_vec >= 1 and m >= 1, got n_vec={n_vec} m={m}"),
            });
        }
        let span = m * n_vec * width;
        if x.len() < span || cs.len() < m * n_dims {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "x.len() {} (need {span}), cs.len() {} (need {})",
                    x.len(),
                    cs.len(),
                    m * n_dims
                ),
            });
        }
        let grid = launch_u32(
            what,
            "grid",
            (m * n_vec * (n_dims / 2)).div_ceil(BLOCK as usize),
        )?;
        let width = launch_u32(what, "width", width)?;
        let n_dims = launch_u32(what, "n_dims", n_dims)?;
        let n_vec = launch_u32(what, "n_vec", n_vec)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_ds41_rope_tail(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .ds41_rope_tail(stream, &prep, cs, width, n_dims, n_vec, m, x)?;
        Ok(())
    }

    /// Enqueue the latent K/V rows of `args.m` tokens: norm, tail rope, the
    /// f32 rows into `out` and each row's f16 into ring slot `pos % window`
    /// (`window = cache.rows()`, `width = cache.cols()`) and into shadow row
    /// `pos` (`shadow` of the ring's width). `m <= window`, so consecutive
    /// positions land in distinct slots. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_kv_norm_rope_append(
        &self,
        stream: &CudaStream,
        args: KvAppendArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_kv_norm_rope_append";
        let KvAppendArgs {
            kv,
            gain,
            cs,
            pos,
            eps,
            n_dims,
            m,
            out,
            cache,
            shadow,
        } = args;
        let (window, width) = (cache.rows(), cache.cols());
        if shadow.cols() != width {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the shadow's rows are {} wide, the ring's {width}",
                    shadow.cols()
                ),
            });
        }
        if width == 0 || !width.is_multiple_of(32) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the norm needs a width that is a positive multiple of 32, got {width}"
                ),
            });
        }
        if n_dims < 2 || !n_dims.is_multiple_of(2) || n_dims > width || n_dims > 2 * RMS_THREADS {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the tail is one pair per thread of one {RMS_THREADS}-thread block: n_dims \
                     must be even, at least 2 and at most min(width {width}, {}), got {n_dims}",
                    2 * RMS_THREADS
                ),
            });
        }
        if m == 0 || m > window {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "need 1 <= m <= window {window} (one ring slot per token), got m={m}"
                ),
            });
        }
        if kv.len() < m * width
            || gain.len() < width
            || cs.len() < m * n_dims
            || pos.len() < m
            || out.len() < m * width
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "kv.len() {} (need {}), gain.len() {} (need {width}), cs.len() {} (need {}), \
                     pos.len() {} (need {m}), out.len() {} (need {})",
                    kv.len(),
                    m * width,
                    gain.len(),
                    cs.len(),
                    m * n_dims,
                    pos.len(),
                    out.len(),
                    m * width
                ),
            });
        }
        let width = launch_u32(what, "width", width)?;
        let n_dims = launch_u32(what, "n_dims", n_dims)?;
        let window = launch_u32(what, "window", window)?;
        let rows = launch_u32(what, "rows", shadow.rows())?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_ds41_kv_norm_rope_append(LaunchConfig1D::new(m, BLOCK, 0))?;
        self.module.ds41_kv_norm_rope_append(
            stream,
            &prep,
            kv,
            gain,
            cs,
            pos,
            eps,
            width,
            n_dims,
            window,
            rows,
            m,
            out,
            cache.buf_mut(),
            shadow.buf_mut(),
        )?;
        Ok(())
    }
}
