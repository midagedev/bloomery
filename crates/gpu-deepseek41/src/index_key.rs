//! Indexer keys: the pre-rope compressed latent projected 512 → 128, k-norm,
//! rope, a Hadamard transform, and an f16 write into the index-key cache.
//!
//! The projection is our q3_K gemv of the compressor's pre-rope row (the
//! reference's pre-rope latent). [`IndexKeyKernels::enqueue_index_key`] is
//! the rest, one warp per group: the norm, the turn of the last `n_dims`
//! values at the group's first position, the Hadamard transform of the whole
//! key, and the f16 row at the group's compressed row — the step buffer and
//! the rope table are the compressor's ([`crate::compress::CompGeom`]), so
//! a step that completes no group writes no key.
//!
//! Numeric contract: the norm sums the squares as an f32 warp tree
//! ([`norm_scale`]) where ik sums in f64 — the one difference; the rope is
//! `rope::rope_pair_rn`, ik's unfused rotation; the transform is ik's
//! `fast_ht` (`iqk_cpu_ops.cpp`) op for op — butterflies at
//! `h = 1, 2, 4, …, 64`, the lower value `x + y`, the upper `x − y`, then one
//! multiply by [`HT_SCALE`] — so on the same input it is bit-identical; the
//! cache value is its f16 rounding (to nearest even). The key's input is
//! token-major, `k[g·WIDTH ..]` for group `g`.

use bloomery_gpu::elem::rms_scale;
use bloomery_gpu::flash::f32_to_f16_bits;
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::f32::consts::FRAC_1_SQRT_2;
use std::sync::Arc;

use crate::compress::{CompGeom, W_GROUPS, w_row};
use crate::rope::rope_pair_rn;

/// Threads per block: one warp per group.
const LANES: u32 = 32;
/// Values one lane owns: `PER_LANE·lane ..`, contiguous — the transform's
/// first two stages stay in the lane, the other five are lane butterflies.
pub(crate) const PER_LANE: usize = 4;
/// Values of an index key (`indexer.head_size`).
pub const WIDTH: usize = PER_LANE * LANES as usize;
const _: () = assert!(WIDTH == 128);

/// `fast_ht`'s final scale for `n` values: `scale *= 0.707106781f` once per
/// butterfly stage, each product rounded to f32, from 1. That literal is
/// `FRAC_1_SQRT_2` as an f32 (both round to `0x3f3504f3`).
const fn ht_scale(n: usize) -> f32 {
    let mut s = 1.0f32;
    let mut h = 1;
    while h < n {
        s *= FRAC_1_SQRT_2;
        h <<= 1;
    }
    s
}

/// The Hadamard transform's scale for [`WIDTH`] values: seven stages.
pub const HT_SCALE: f32 = ht_scale(WIDTH);

// ------------------------------------------------------------------ cores

/// Four contiguous f32 at `base`.
///
/// # Safety
/// `base + 4 <= x.len()`.
#[inline(always)]
unsafe fn load4(x: &[f32], base: usize) -> [f32; PER_LANE] {
    // SAFETY: base + 3 < x.len() by this fn's contract.
    unsafe {
        [
            *x.get_unchecked(base),
            *x.get_unchecked(base + 1),
            *x.get_unchecked(base + 2),
            *x.get_unchecked(base + 3),
        ]
    }
}

/// The key's norm scale: the lane's four squares summed by fused
/// multiply-adds in value order from `+0`, then the warp butterfly
/// (`reduce_sum_f32`: xor 16, 8, 4, 2, 1, own + partner) and
/// `elem::rms_scale` at [`WIDTH`]. This order is the gate's host
/// transcription. Every lane of the warp calls it (a warp collective).
#[inline(always)]
fn norm_scale(x: [f32; PER_LANE], eps: f32) -> f32 {
    let acc = fma_rn_f32(x[0], x[0], 0.0);
    let acc = fma_rn_f32(x[1], x[1], acc);
    let acc = fma_rn_f32(x[2], x[2], acc);
    let acc = fma_rn_f32(x[3], x[3], acc);
    rms_scale(warp::reduce_sum_f32(acc), WIDTH as u32, eps)
}

/// The transform's stages `h = 1, 2`, inside the lane's four values.
#[inline(always)]
fn ht_lane(n: [f32; PER_LANE]) -> [f32; PER_LANE] {
    let a = [n[0] + n[1], n[0] - n[1], n[2] + n[3], n[2] - n[3]];
    [a[0] + a[2], a[1] + a[3], a[0] - a[2], a[1] - a[3]]
}

/// The transform's stage `h = PER_LANE·m` across lanes `lane` and
/// `lane ^ m`: the lower lane keeps `x + y`, the upper `x − y`, `x` the
/// lower value. Every lane of the warp calls it (a warp collective).
#[inline(always)]
fn ht_cross(b: [f32; PER_LANE], lane: u32, m: u32) -> [f32; PER_LANE] {
    let p = [
        warp::shuffle_xor_f32(b[0], m),
        warp::shuffle_xor_f32(b[1], m),
        warp::shuffle_xor_f32(b[2], m),
        warp::shuffle_xor_f32(b[3], m),
    ];
    if lane & m == 0 {
        [b[0] + p[0], b[1] + p[1], b[2] + p[2], b[3] + p[3]]
    } else {
        [p[0] - b[0], p[1] - b[1], p[2] - b[2], p[3] - b[3]]
    }
}

/// The transform's seven butterfly stages over a warp's [`WIDTH`] values,
/// lane `lane` holding `PER_LANE·lane ..`: [`ht_lane`], then [`ht_cross`]
/// at lane distances 1, 2, 4, 8, 16. The caller multiplies by
/// [`HT_SCALE`]. Every lane of the warp calls it (a warp collective).
#[inline(always)]
pub(crate) fn ht_warp(n: [f32; PER_LANE], lane: u32) -> [f32; PER_LANE] {
    let mut h = ht_lane(n);
    let mut m = 1;
    while m < LANES {
        h = ht_cross(h, lane, m);
        m <<= 1;
    }
    h
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod index_key_kernels {
    use super::*;

    /// One warp per group up to `max_groups`: group `g` (of the step
    /// buffer's count) takes its key `k[g·128 ..]`, normalizes it
    /// (`(scale · gain) · x`), turns its last `n_dims` values by the table at
    /// `cs[g·n_dims ..]` (two pairs per lane), transforms it and rounds it to
    /// f16 into cache row `row` of the step buffer. A group past the count,
    /// or whose row leaves the cache, writes nothing. The guards read only
    /// warp-uniform values, so every lane reaches the collectives.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            step.len() >= 2 + max_groups,
            k.len() >= max_groups * 128,
            gain.len() >= 128,
            cs.len() >= max_groups * n_dims,
            cache.len() >= rows * 128
        )
    )]
    pub fn ds41_index_key(
        step: &[u32],
        k: &[f32],
        gain: &[f32],
        cs: &[f32],
        eps: f32,
        max_groups: u32,
        n_dims: u32,
        rows: u32,
        mut cache: DisjointSlice<u16>,
    ) {
        let g = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x();
        // SAFETY: word 0 exists: step.len() >= 2 + … by the contract.
        let groups = unsafe { *step.get_unchecked(W_GROUPS) } as usize;
        if g >= groups || g >= max_groups as usize {
            return;
        }
        // SAFETY: w_row(g) = 2 + g < 2 + max_groups <= step.len().
        let row = unsafe { *step.get_unchecked(w_row(g)) } as usize;
        if row >= rows as usize {
            return;
        }
        let v = PER_LANE * lane as usize;
        // SAFETY: g < max_groups, so g·128 + v + 4 <= max_groups·128 <= k.len().
        let x = unsafe { load4(k, g * WIDTH + v) };
        let scale = norm_scale(x, eps);
        // SAFETY: v + 4 <= 128 <= gain.len().
        let gw = unsafe { load4(gain, v) };
        // Each product rounded on its own: a plain multiply feeding the
        // transform's first add would be contracted into a fused one.
        let mut n = [
            mul_rn_f32(mul_rn_f32(scale, gw[0]), x[0]),
            mul_rn_f32(mul_rn_f32(scale, gw[1]), x[1]),
            mul_rn_f32(mul_rn_f32(scale, gw[2]), x[2]),
            mul_rn_f32(mul_rn_f32(scale, gw[3]), x[3]),
        ];
        let nd = n_dims as usize;
        let tail0 = WIDTH - nd;
        if v >= tail0 {
            // SAFETY: v − tail0 + 4 <= n_dims, so the four table values sit
            // inside group g's table, below max_groups·n_dims <= cs.len().
            let c = unsafe { load4(cs, g * nd + v - tail0) };
            let (a0, a1) = rope_pair_rn(n[0], n[1], c[0], c[1]);
            let (a2, a3) = rope_pair_rn(n[2], n[3], c[2], c[3]);
            n = [a0, a1, a2, a3];
        }
        let h = ht_warp(n, lane);
        let base = row * WIDTH + v;
        // SAFETY: base + 4 <= (row + 1)·128 <= rows·128 <= cache.len(); block
        // g owns cache row `row` (rows ascend, host-checked) and the lane its
        // four values.
        unsafe {
            *cache.get_unchecked_mut(base) = f32_to_f16_bits(mul_rn_f32(h[0], HT_SCALE));
            *cache.get_unchecked_mut(base + 1) = f32_to_f16_bits(mul_rn_f32(h[1], HT_SCALE));
            *cache.get_unchecked_mut(base + 2) = f32_to_f16_bits(mul_rn_f32(h[2], HT_SCALE));
            *cache.get_unchecked_mut(base + 3) = f32_to_f16_bits(mul_rn_f32(h[3], HT_SCALE));
        }
    }
}

// -------------------------------------------------------------- launchers

/// [`IndexKeyKernels::enqueue_index_key`]'s arguments. `geom` and `step` are
/// the compressor's for the same step (the key lands on the group's
/// compressed row), `cs` its rope tables; `k` holds `geom.max_groups`
/// projections of [`WIDTH`], token-major; `gain` the key norm's weights;
/// `cache` is the index-key cache, `geom.rows` rows of [`WIDTH`] f16.
pub struct IndexKeyArgs<'a> {
    pub geom: CompGeom,
    pub step: &'a DeviceBuffer<u32>,
    pub k: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub cs: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub n_dims: usize,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// The loaded index-key module. Owns no stream: each enqueue takes the
/// engine stream, so its launches order with the rest of the step and
/// capture.
pub struct IndexKeyKernels {
    module: index_key_kernels::LoadedModule,
}

impl IndexKeyKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<IndexKeyKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { index_key_kernels::load(ctx)? };
        Ok(IndexKeyKernels { module })
    }

    /// Enqueue the keys of the groups the step buffer completes: norm, tail
    /// rope, transform, f16 row. `geom.max_groups` warps whatever the step
    /// completes. Asynchronous, allocation-free, capturable.
    pub fn enqueue_index_key(
        &self,
        stream: &CudaStream,
        args: IndexKeyArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_index_key";
        let IndexKeyArgs {
            geom,
            step,
            k,
            gain,
            cs,
            eps,
            n_dims,
            cache,
        } = args;
        geom.check(what)?;
        let gm = geom.max_groups;
        if n_dims == 0 || !n_dims.is_multiple_of(PER_LANE) || n_dims > WIDTH {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the tail is two pairs per lane: n_dims must be a positive multiple of \
                     {PER_LANE} at most {WIDTH}, got {n_dims}"
                ),
            });
        }
        if cache.cols() != WIDTH || cache.rows() != geom.rows {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the cache is {}x{}, want {} rows of {WIDTH}",
                    cache.rows(),
                    cache.cols(),
                    geom.rows
                ),
            });
        }
        if step.len() < geom.words()
            || k.len() < gm * WIDTH
            || gain.len() < WIDTH
            || cs.len() < gm * n_dims
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "step.len() {} (need {}), k.len() {} (need {}), gain.len() {} (need \
                     {WIDTH}), cs.len() {} (need {})",
                    step.len(),
                    geom.words(),
                    k.len(),
                    gm * WIDTH,
                    gain.len(),
                    cs.len(),
                    gm * n_dims
                ),
            });
        }
        let grid = launch_u32(what, "max_groups", gm)?;
        let prep = self
            .module
            .prepare_ds41_index_key(LaunchConfig1D::new(grid, LANES, 0))?;
        self.module.ds41_index_key(
            stream,
            &prep,
            step,
            k,
            gain,
            cs,
            eps,
            grid,
            launch_u32(what, "n_dims", n_dims)?,
            launch_u32(what, "rows", geom.rows)?,
            cache.buf_mut(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HT_SCALE, WIDTH};
    use std::f32::consts::FRAC_1_SQRT_2;
    use std::hint::black_box;

    /// The transform's scale is `fast_ht`'s loop run at runtime: seven f32
    /// products of ik's `0.707106781f` from 1, not a constant folded another
    /// way — and that literal is the f32 `FRAC_1_SQRT_2`.
    #[test]
    fn ht_scale_is_fast_ht_loop() {
        let ik: f32 = black_box("0.707106781")
            .parse()
            .expect("ik's literal parses");
        assert_eq!(ik.to_bits(), FRAC_1_SQRT_2.to_bits());
        let mut s = 1.0f32;
        let mut h = 1;
        while h < black_box(WIDTH) {
            s *= ik;
            h <<= 1;
        }
        assert_eq!(HT_SCALE.to_bits(), s.to_bits());
    }
}
