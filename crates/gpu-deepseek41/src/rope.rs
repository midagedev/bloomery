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
//!   in one launch: its norm, its tail rope, the row in f32 and its f16 slot
//!   in the layer's raw window ring.
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
use cuda_device::float::{add_rn_f32, mul_rn_f32};
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

/// Which way a table turns: `Forward` is rope, `Back` its inverse — ik's
/// `ROPE_BACK`, the same table with the sines negated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Back,
}

impl Direction {
    /// ggml's `sin_sign`.
    fn sin_sign(self) -> f32 {
        match self {
            Direction::Forward => 1.0,
            Direction::Back => -1.0,
        }
    }
}

/// The constants one rope site hands ggml's rope (`ggml_rope_ext`'s
/// arguments after the mode). ik's V4.1 graph uses two
/// (`build_deepseek4.cpp`): [`RopeSpec::window`] and [`RopeSpec::yarn`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeSpec {
    /// Values rotated at the tail of each head (`rope.dimension_count`).
    pub n_dims: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// ggml's `n_ctx_orig`, an `int`.
    pub n_ctx_orig: i32,
}

impl RopeSpec {
    /// The rope of a layer that keeps no compressed stream
    /// (`attention.compress_ratios` 0): `rope.freq_base` and no scaling —
    /// ik passes `freq_scale` 1, `ext_factor` 0, betas 0 and `n_ctx_orig` 0,
    /// and `dsv4_rope_attn_factor` is 1 at `ext_factor` 0, so the table is
    /// plain `cos`/`sin` of `p·θ_i`.
    #[must_use]
    pub fn window(freq_base: f32, n_dims: usize) -> RopeSpec {
        RopeSpec {
            n_dims,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            n_ctx_orig: 0,
        }
    }

    /// The rope of every other site — the compressed layers' heads, the
    /// pooled rows, the index keys and the indexer query:
    /// `attention.compress_rope_freq_base`, YaRN at `freq_scale =
    /// 1/rope.scaling.factor` with `ext_factor` 1 (llama.cpp's value for a
    /// `yarn` file) and the file's betas and original context, and
    /// `attn_factor = 1/(1 + 0.1·ln(1/freq_scale))` rounded op by op — ik's
    /// `dsv4_rope_attn_factor`, in libllama, which is built without FMA.
    #[must_use]
    pub fn yarn(
        freq_base: f32,
        scaling_factor: f32,
        n_ctx_orig: i32,
        beta_fast: f32,
        beta_slow: f32,
        n_dims: usize,
    ) -> RopeSpec {
        let freq_scale = 1.0 / scaling_factor;
        RopeSpec {
            n_dims,
            freq_base,
            freq_scale,
            ext_factor: 1.0,
            attn_factor: 1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln()),
            beta_fast,
            beta_slow,
            n_ctx_orig,
        }
    }
}

/// One site's cos/sin table at any position: [`ggml_rope_cache`]'s values,
/// with what does not depend on the position — `theta_scale`, the YaRN ramp
/// of each pair, the magnitude factor — computed once, so a position costs
/// per pair two multiplies, one fused multiply-add under YaRN, a `sin` and a
/// `cos`.
#[derive(Clone, Debug)]
pub struct RopeTable {
    n_dims: usize,
    freq_scale: f32,
    theta_scale: f32,
    /// Per pair, `(ramp_mix, 1 − ramp_mix)`; empty without YaRN.
    ramp: Vec<(f32, f32)>,
    /// `attn_factor`, times ggml's `1 + 0.1·ln(1/freq_scale)` under YaRN.
    mscale: f32,
}

impl RopeTable {
    /// The table of `spec`; `n_dims` must be even and at least 2.
    pub fn new(spec: &RopeSpec) -> Result<RopeTable, GpuError> {
        let nd = spec.n_dims;
        if nd < 2 || !nd.is_multiple_of(2) {
            return Err(GpuError::Shape {
                what: "RopeTable::new",
                detail: format!("n_dims must be even and at least 2, got {nd}"),
            });
        }
        let (ramp, mscale) = if spec.ext_factor == 0.0 {
            (Vec::new(), spec.attn_factor)
        } else {
            let corr = ggml_rope_yarn_corr_dims(spec);
            let ramp = (0..nd / 2)
                .map(|i| {
                    let mix = rope_yarn_ramp(corr[0], corr[1], 2 * i) * spec.ext_factor;
                    (mix, 1.0 - mix)
                })
                .collect();
            (ramp, spec.attn_factor * yarn_mscale(spec.freq_scale))
        };
        Ok(RopeTable {
            n_dims: nd,
            freq_scale: spec.freq_scale,
            theta_scale: theta_scale(spec),
            ramp,
            mscale,
        })
    }

    /// Values per position: `n_dims`, a cos and a sin per pair.
    #[must_use]
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Append position `pos`'s table to `out`: `n_dims` f32, `[cos_0, sin_0,
    /// cos_1, …]` with the sines negated for [`Direction::Back`] — the
    /// layout the kernels read at `t·n_dims` for token `t`. `pos` becomes
    /// f32 as ggml's `int64_t` does, exactly below 2^24.
    pub fn push(&self, pos: u32, dir: Direction, out: &mut Vec<f32>) {
        let sign = dir.sin_sign();
        let mut theta = pos as f32;
        for i in 0..self.n_dims / 2 {
            let interp = self.freq_scale * theta;
            let th = match self.ramp.get(i) {
                Some(&(mix, keep)) => interp.mul_add(keep, theta * mix),
                None => interp,
            };
            let (s, c) = th.sin_cos();
            out.push(c * self.mscale);
            out.push(s * self.mscale * sign);
            theta *= self.theta_scale;
        }
    }
}

/// ggml's `theta_scale = powf(freq_base, −2.0f/n_dims)`.
fn theta_scale(spec: &RopeSpec) -> f32 {
    spec.freq_base.powf(-2.0 / spec.n_dims as f32)
}

/// `1.0f + 0.1f·logf(1.0f/freq_scale)`, the factor `rope_yarn` multiplies
/// its magnitude by under YaRN — one fused multiply-add in ik's libggml.
fn yarn_mscale(freq_scale: f32) -> f32 {
    0.1f32.mul_add((1.0 / freq_scale).ln(), 1.0)
}

/// ggml's `MAX(a, b)`: `a > b ? a : b` — a NaN `b` comes back.
fn c_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// ggml's `MIN(a, b)`: `a < b ? a : b` — a NaN `b` comes back.
fn c_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `ggml_rope_yarn_corr_dim`: `n_dims·logf(n_ctx_orig/(n_rot·2·π))/(2·logf(base))`.
fn ggml_rope_yarn_corr_dim(n_dims: usize, n_ctx_orig: i32, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// `ggml_rope_yarn_corr_dims`: the pair range the YaRN ramp runs over.
fn ggml_rope_yarn_corr_dims(spec: &RopeSpec) -> [f32; 2] {
    let n = spec.n_dims;
    let start = ggml_rope_yarn_corr_dim(n, spec.n_ctx_orig, spec.beta_fast, spec.freq_base).floor();
    let end = ggml_rope_yarn_corr_dim(n, spec.n_ctx_orig, spec.beta_slow, spec.freq_base).ceil();
    [c_max(0.0, start), c_min(n as f32 - 1.0, end)]
}

/// `rope_yarn_ramp`: `1 − MIN(1, MAX(0, (i0/2 − low)/MAX(0.001, high − low)))`,
/// `i0/2` an integer division.
fn rope_yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = ((i0 / 2) as f32 - low) / c_max(0.001, high - low);
    1.0 - c_min(1.0, c_max(0.0, y))
}

/// ggml's rope cache for one position, transcribed statement for statement
/// from ik's `ggml.c` (`ggml_rope_cache_init`, `rope_yarn`, `rope_yarn_ramp`,
/// `ggml_rope_yarn_corr_dims`) as its CPU build compiles them: GCC, at
/// `-march=native` with GNU C's default contraction, fuses
/// `theta_interp·(1 − ramp_mix) + theta_extrap·ramp_mix` and
/// `1 + 0.1·ln(1/freq_scale)` into fused multiply-adds and rounds every other
/// op on its own, and `MAX`/`MIN` are the C macros — the window rope's
/// correction range, from `n_ctx_orig` 0, is NaN and unused. `ne0` is the
/// head width: the cache holds `ne0` values and the tail rope reads the
/// first `n_dims`. The verification side's reading of the recipe; the
/// engine's tables come from [`RopeTable`], which the unit test pins to it.
#[must_use]
pub fn ggml_rope_cache(spec: &RopeSpec, pos: u32, ne0: usize, dir: Direction) -> Vec<f32> {
    let theta_scale = theta_scale(spec);
    let corr_dims = ggml_rope_yarn_corr_dims(spec);
    let mut cache = Vec::with_capacity(ne0);
    let mut theta = pos as f32;
    for i in 0..ne0 / 2 {
        let theta_extrap = theta;
        let theta_interp = spec.freq_scale * theta_extrap;
        let mut th = theta_interp;
        let mut mscale = spec.attn_factor;
        if spec.ext_factor != 0.0 {
            let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], 2 * i) * spec.ext_factor;
            th = theta_interp.mul_add(1.0 - ramp_mix, theta_extrap * ramp_mix);
            mscale *= yarn_mscale(spec.freq_scale);
        }
        cache.push(th.cos() * mscale);
        cache.push(th.sin() * mscale * dir.sin_sign());
        theta *= theta_scale;
    }
    cache
}

// ------------------------------------------------------------------ cores

/// One adjacent pair of a head's rope tail turned by `(c, s)`:
/// `y0 = x0·c − x1·s`, `y1 = x0·s + x1·c`, each product and each sum rounded
/// on its own. ik's CPU rope rotates unfused; the compiler contracts a plain
/// `a*b − c*d` into an FMA, so every op here is an explicit round-to-nearest
/// intrinsic, which it never contracts. The inverse turn is this core on a
/// table with `s` negated.
#[inline(always)]
pub(crate) fn rope_pair_rn(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    let y0 = add_rn_f32(mul_rn_f32(x0, c), -mul_rn_f32(x1, s));
    let y1 = add_rn_f32(mul_rn_f32(x0, s), mul_rn_f32(x1, c));
    (y0, y1)
}

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
    /// window` of the ring `cache` (`window` rows of `width`). `width` a
    /// positive multiple of 32, `n_dims` even and at most `width` and
    /// `2·RMS_THREADS` (host-checked); the tokens of one launch land in
    /// distinct slots (host-checked as `m <= window` — the caller's
    /// positions are consecutive). The token guard is block-uniform, so no
    /// barrier and no warp collective is skipped.
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
            cache.len() >= window * width
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
        m: u32,
        mut out: DisjointSlice<f32>,
        mut cache: DisjointSlice<u16>,
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
        let slot = (unsafe { *pos.get_unchecked(t) } % window) as usize;
        let crow = slot * k;

        // The values before the tail: normalized, stored twice.
        let mut it = tid;
        while it < head {
            // SAFETY: it < head < width bounds the gain read by the contract
            // and the kv read and out write as base + it < m·width; the
            // cache slot is crow + it < (slot + 1)·width <= window·width <=
            // cache.len() because slot < window. One thread per value.
            unsafe {
                let nv = (scale * *gain.get_unchecked(it)) * *kv.get_unchecked(base + it);
                *out.get_unchecked_mut(base + it) = nv;
                *cache.get_unchecked_mut(crow + it) = f32_to_f16_bits(nv);
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
            // SAFETY: the out positions are the kv positions above, inside
            // m·width; the cache positions crow + j + 1 < (slot + 1)·width <=
            // cache.len(). This thread owns the pair.
            unsafe {
                *out.get_unchecked_mut(base + j) = y0;
                *out.get_unchecked_mut(base + j + 1) = y1;
                *cache.get_unchecked_mut(crow + j) = f32_to_f16_bits(y0);
                *cache.get_unchecked_mut(crow + j + 1) = f32_to_f16_bits(y1);
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
/// `pos` the `m` positions on the device; `out` takes the rows in f32 and
/// `cache` is the layer's raw window ring, `window` rows of `width` f16.
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
    /// (`window = cache.rows()`, `width = cache.cols()`). `m <= window`, so
    /// consecutive positions land in distinct slots. Asynchronous,
    /// allocation-free, capturable.
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
        } = args;
        let (window, width) = (cache.rows(), cache.cols());
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
            m,
            out,
            cache.buf_mut(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use std::hint::black_box;

    /// The engine's table is ggml's recipe bit for bit at positions the
    /// oracle sets do not reach — up to the model's context length — for
    /// both of V4.1's ropes, both ways. The constants are the V4.1 file's
    /// (`rope.freq_base`, `attention.compress_rope_freq_base`,
    /// `rope.scaling.{factor, original_context_length, yarn_beta_fast,
    /// yarn_beta_slow}`, `rope.dimension_count`); the recipe fills a
    /// 512-value head's cache and the tail reads its first `n_dims`. Every
    /// input goes through `black_box`, so neither side is folded at compile
    /// time by a libm other than the one the other side calls.
    #[test]
    fn table_is_the_ggml_recipe_at_large_positions() {
        let specs = [
            (
                "window",
                RopeSpec::window(black_box(10_000.0), black_box(64)),
            ),
            (
                "yarn",
                RopeSpec::yarn(
                    black_box(160_000.0),
                    black_box(16.0),
                    black_box(65_536),
                    black_box(32.0),
                    black_box(1.0),
                    black_box(64),
                ),
            ),
        ];
        for (name, spec) in specs {
            let table = RopeTable::new(&spec).expect("n_dims 64 is even");
            for pos in [0u32, 1, 1025, 65_535, 65_536, 1_048_575] {
                for dir in [Direction::Forward, Direction::Back] {
                    let mut got = Vec::new();
                    table.push(black_box(pos), dir, &mut got);
                    let want = ggml_rope_cache(&spec, black_box(pos), 512, dir);
                    let same = got.len() == 64
                        && got
                            .iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        same,
                        "{name} p={pos} {dir:?}: table {got:?} vs recipe {:?}",
                        &want[..64]
                    );
                }
            }
        }
    }
}
