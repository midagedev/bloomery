//! Hyper-connections with an identity mix: four residual streams around every
//! sub-layer, each stream keeping its own residual (Hy4-flash; the
//! checkpoint's `hc_attn_fn`/`hc_ffn_fn` have eight rows, no comb).
//!
//! Layouts, token-major: a token's streams are `4·n` f32, stream `i` at
//! `i·n`; read flat they are the `K = 4·n` values the mix rows take. A token's
//! gates are [`MIX`] f32: `pre[4]` then `post[4]`.
//!
//! The numeric rule each entry holds (the oracle is llama.cpp's
//! `build_hc_pre` / `build_hc_post` / `build_hc_head`, graph op for op):
//!
//! `hc_ident_pre` — one block of [`RMS_THREADS`] per token.
//! 1. For each of the eight mix rows `r`, thread `tid` sums
//!    `fma(w[r·K + i], x[i], acc)` over `i = tid, tid + 256, …` from 0, the
//!    warp butterfly (`warp::reduce_sum_f32`) and the warp partials in
//!    `elem::rms_warp_tree`'s order give the dot; the squares of `x` sum the
//!    same way (`fma(v, v, acc)`). The RMS scale multiplies the dot instead of
//!    the input: `mix_r = scale · dot_r`, `scale = 1 / sqrt(squares / K + eps)`
//!    (`elem::rms_scale_checked`: a scale that is not finite and positive is
//!    NaN and raises `FaultSite::F32Product`).
//! 2. `pre_i = σ(mix_i·s0 + base_i) + eps_hc` and `post_i = σ(mix_{4+i}·s1 +
//!    base_{4+i})·magnitude + eps_hc`, each op rounded on its own, σ(v) =
//!    `1/(1 + e^(−v))` with the exponential taken in f64 and rounded once.
//! 3. The fold `y = ((x_0·pre_0 + x_1·pre_1) + x_2·pre_2) + x_3·pre_3` with
//!    every product and sum rounded on its own (ggml's `mul` then `add` loop).
//!
//! `hc_ident_head` — the same steps 1 and 3 over four mix rows, one scale `s0`
//! for all of them and no post.
//!
//! `hc_ident_post` — `out_i = res_i + x·post_i`, each op rounded on its own.
//!
//! Every f32 product and sum whose rounding the contract names is an explicit
//! round-to-nearest intrinsic: the compiler contracts a plain `a·b + c` into a
//! fused multiply-add.

use crate::elem::{RMS_THREADS, RMS_WARPS, rms_scale_checked, rms_warp_tree};
use crate::{FaultSink, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, div_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Residual streams (`hc_mult`).
pub const HC: usize = 4;
/// Gates of one token: `pre[HC]` then `post[HC]`, and the rows of `hc_fn`.
pub const MIX: usize = 2 * HC;
/// The launch's block width, [`RMS_THREADS`].
const BLOCK: u32 = 256;
const _: () = assert!(BLOCK as usize == RMS_THREADS && RMS_WARPS == 8 && HC == 4);
/// Shared slots of the block reduce: a partial per warp for each of the
/// [`MIX`] dot products and the squares.
const RED: usize = (MIX + 1) * RMS_WARPS;

// ------------------------------------------------------------------ cores

/// `1/(1 + e^(−v))`, the exponential in f64 rounded once.
#[inline(always)]
fn sigmoid_rn(v: f32) -> f32 {
    let e = f64::from(-v).exp() as f32;
    div_rn_f32(1.0, add_rn_f32(1.0, e))
}

/// The row's partial for quantity `r` of one token: for `r < rows` the dot of
/// `w`'s row `r` with the token's flat streams, for `r == rows` their squares,
/// each as `fma(a, b, acc)` over `i = tid, tid + RMS_THREADS, …` from 0; the
/// warp butterfly and the warp's slot `r·RMS_WARPS + tid/32` of `red`.
///
/// # Safety
///
/// `xb + k <= x.len()`, `(rows)·k <= w.len()`, `red` points at this block's
/// [`RED`]-slot shared array, `r <= rows`, `rows < MIX + 1`, and every thread
/// of the block calls this with the same `r`.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel body's arguments and the shared array it declares (rust-quality R8)"
)]
#[inline(always)]
unsafe fn partial(
    x: &[f32],
    w: &[f32],
    xb: usize,
    k: usize,
    rows: usize,
    r: usize,
    tid: usize,
    red: *mut f32,
) {
    let mut acc = 0.0f32;
    let mut i = tid;
    if r < rows {
        let wb = r * k;
        while i < k {
            // SAFETY: i < k, so xb + i < xb + k <= x.len() and wb + i <
            // (r + 1)·k <= rows·k <= w.len() by this function's contract.
            let (a, b) = unsafe { (*w.get_unchecked(wb + i), *x.get_unchecked(xb + i)) };
            acc = fma_rn_f32(a, b, acc);
            i += RMS_THREADS;
        }
    } else {
        while i < k {
            // SAFETY: i < k, so xb + i < xb + k <= x.len().
            let v = unsafe { *x.get_unchecked(xb + i) };
            acc = fma_rn_f32(v, v, acc);
            i += RMS_THREADS;
        }
    }
    let p = warp::reduce_sum_f32(acc);
    if warp::lane_id() == 0 {
        // SAFETY: r·RMS_WARPS + tid/32 < (MIX + 1)·RMS_WARPS = RED; one lane
        // per warp writes its own slot before the caller's barrier.
        unsafe {
            *red.add(r * RMS_WARPS + tid / 32) = p;
        }
    }
}

/// The eight warp partials of quantity `r` in `rms_warp_tree`'s order.
///
/// # Safety
///
/// Every slot of `r` was written and published by a barrier.
#[inline(always)]
unsafe fn tree(red: *const f32, r: usize) -> f32 {
    // SAFETY: r·RMS_WARPS + 7 < RED by this function's contract.
    let sums = unsafe {
        [
            *red.add(r * RMS_WARPS),
            *red.add(r * RMS_WARPS + 1),
            *red.add(r * RMS_WARPS + 2),
            *red.add(r * RMS_WARPS + 3),
            *red.add(r * RMS_WARPS + 4),
            *red.add(r * RMS_WARPS + 5),
            *red.add(r * RMS_WARPS + 6),
            *red.add(r * RMS_WARPS + 7),
        ]
    };
    rms_warp_tree(sums)
}

/// Steps 1 and 2 of one token over `ROWS` mix rows (`ROWS` = [`MIX`] for the
/// pre entry, [`HC`] for the head): thread 0 leaves the gates in `gates`
/// (`pre[HC]`, then `post[HC]` when `ROWS` is [`MIX`]), published by the
/// barrier this function ends on. Every thread of the block calls this.
///
/// # Safety
///
/// [`partial`]'s contract for `rows = ROWS`; `scale.len() >= 2` when `ROWS`
/// is [`MIX`], `>= 1` otherwise; `base.len() >= ROWS`; `red` and `gates` are
/// this block's own shared arrays of [`RED`] and [`MIX`] slots.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel body's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gates_of<const ROWS: usize>(
    x: &[f32],
    w: &[f32],
    scale: &[f32],
    base: &[f32],
    (eps_norm, eps_hc, magnitude): (f32, f32, f32),
    xb: usize,
    k: usize,
    tid: usize,
    fault: FaultSink,
    red: *mut f32,
    gates: *mut f32,
) {
    let mut r = 0;
    while r <= ROWS {
        // SAFETY: partial's contract, from this function's.
        unsafe { partial(x, w, xb, k, ROWS, r, tid, red) };
        r += 1;
    }
    thread::sync_threads();
    // SAFETY: every slot of the squares was written before the barrier above.
    let squares = unsafe { tree(red, ROWS) };
    let rms = rms_scale_checked(squares, k as u32, eps_norm, fault, tid == 0);
    if tid == 0 {
        let mut r = 0;
        while r < ROWS {
            // SAFETY: r < ROWS, every slot of row r was written before the
            // barrier above; base.len() >= ROWS and scale.len() >= 1 + (r >=
            // HC) by this function's contract.
            let (dot, s, b) = unsafe {
                (
                    tree(red, r),
                    *scale.get_unchecked(usize::from(r >= HC)),
                    *base.get_unchecked(r),
                )
            };
            let v = add_rn_f32(mul_rn_f32(mul_rn_f32(rms, dot), s), b);
            let g = sigmoid_rn(v);
            let g = if r < HC { g } else { mul_rn_f32(g, magnitude) };
            // SAFETY: r < ROWS <= MIX, `gates`' length; thread 0 alone writes.
            unsafe {
                *gates.add(r) = add_rn_f32(g, eps_hc);
            }
            r += 1;
        }
    }
    thread::sync_threads();
}

/// Step 3: `y[t·n + d]` for `d = tid, tid + RMS_THREADS, …`, the fold of the
/// token's four streams by `gates[0..HC]`.
///
/// # Safety
///
/// `xb + HC·n <= x.len()`, `yb + n <= y.len()`, `gates` published by a
/// barrier, and `yb ..` is this block's alone.
#[inline(always)]
unsafe fn fold(
    x: &[f32],
    xb: usize,
    n: usize,
    tid: usize,
    gates: *const f32,
    y: &mut DisjointSlice<f32>,
    yb: usize,
) {
    // SAFETY: gates holds MIX >= HC slots, written before the caller's barrier.
    let (p0, p1, p2, p3) = unsafe { (*gates, *gates.add(1), *gates.add(2), *gates.add(3)) };
    let mut d = tid;
    while d < n {
        // SAFETY: d < n, so xb + 3n + d < xb + HC·n <= x.len().
        let (x0, x1, x2, x3) = unsafe {
            (
                *x.get_unchecked(xb + d),
                *x.get_unchecked(xb + n + d),
                *x.get_unchecked(xb + 2 * n + d),
                *x.get_unchecked(xb + 3 * n + d),
            )
        };
        let a = add_rn_f32(mul_rn_f32(x0, p0), mul_rn_f32(x1, p1));
        let a = add_rn_f32(a, mul_rn_f32(x2, p2));
        let a = add_rn_f32(a, mul_rn_f32(x3, p3));
        // SAFETY: d < n, so yb + d < yb + n <= y.len(); thread `tid` writes
        // the positions tid + RMS_THREADS·j of this block's row only.
        unsafe {
            *y.get_unchecked_mut(yb + d) = a;
        }
        d += RMS_THREADS;
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod hc_ident_kernels {
    use super::*;

    /// HC pre for token `t = blockIdx.x`: the gates (`mix[t·MIX ..]`, `pre`
    /// then `post`) and the fold of the four streams into `y[t·n ..]`, by the
    /// module's contract. `w` is `hc_fn` (eight rows of `4·n`), `scale` its
    /// two scales (`pre`, `post`), `base` its eight offsets. The block guard
    /// is uniform, so no barrier and no warp collective is skipped.
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
            x.len() >= m * 4 * n,
            w.len() >= 8 * 4 * n,
            scale.len() >= 2,
            base.len() >= 8,
            mix.len() >= m * 8,
            y.len() >= m * n
        )
    )]
    pub fn hc_ident_pre(
        x: &[f32],
        w: &[f32],
        scale: &[f32],
        base: &[f32],
        eps_norm: f32,
        eps_hc: f32,
        magnitude: f32,
        n: u32,
        m: u32,
        fault: FaultSink,
        mut mix: DisjointSlice<f32>,
        mut y: DisjointSlice<f32>,
    ) {
        static mut RED_S: SharedArray<f32, RED> = SharedArray::UNINIT;
        static mut GATES: SharedArray<f32, MIX> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let n = n as usize;
        let k = HC * n;
        // SAFETY: RED_S and GATES are this block's own shared allocations,
        // reached only through these pointers.
        let (red, gates) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut RED_S),
                SharedArray::as_raw_mut_ptr(&raw mut GATES),
            )
        };
        // SAFETY: t < m, so t·k + k <= m·4n <= x.len(), MIX·k <= w.len(),
        // scale.len() >= 2 and base.len() >= 8 by the launch contract; red
        // and gates are this block's, every thread arrives.
        unsafe {
            gates_of::<MIX>(
                x,
                w,
                scale,
                base,
                (eps_norm, eps_hc, magnitude),
                t * k,
                k,
                tid,
                fault,
                red,
                gates,
            );
        }
        if tid < MIX {
            // SAFETY: tid < MIX, t < m, so t·MIX + tid < m·8 <= mix.len(); the
            // slot is this thread's alone; gates was published by the barrier
            // `gates_of` ends on.
            unsafe {
                *mix.get_unchecked_mut(t * MIX + tid) = *gates.add(tid);
            }
        }
        // SAFETY: t < m, so t·k + k <= x.len() and t·n + n <= y.len() by the
        // launch contract; gates is published; row t of y is this block's.
        unsafe { fold(x, t * k, n, tid, gates, &mut y, t * n) };
    }

    /// HC head for token `t = blockIdx.x`: the fold of the four streams into
    /// `y[t·n ..]` by the four gates the head's mix (`w`: four rows of `4·n`,
    /// one `scale`, four `base` values) gives.
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
            x.len() >= m * 4 * n,
            w.len() >= 4 * 4 * n,
            scale.len() >= 1,
            base.len() >= 4,
            y.len() >= m * n
        )
    )]
    pub fn hc_ident_head(
        x: &[f32],
        w: &[f32],
        scale: &[f32],
        base: &[f32],
        eps_norm: f32,
        eps_hc: f32,
        n: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut RED_S: SharedArray<f32, RED> = SharedArray::UNINIT;
        static mut GATES: SharedArray<f32, MIX> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let n = n as usize;
        let k = HC * n;
        // SAFETY: RED_S and GATES are this block's own shared allocations,
        // reached only through these pointers.
        let (red, gates) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut RED_S),
                SharedArray::as_raw_mut_ptr(&raw mut GATES),
            )
        };
        // SAFETY: t < m, so t·k + k <= x.len(), HC·k <= w.len(), scale.len()
        // >= 1 and base.len() >= 4 by the launch contract; red and gates are
        // this block's, every thread arrives. The head has no post rows, so
        // the magnitude is never read.
        unsafe {
            gates_of::<HC>(
                x,
                w,
                scale,
                base,
                (eps_norm, eps_hc, 1.0),
                t * k,
                k,
                tid,
                fault,
                red,
                gates,
            );
        }
        // SAFETY: t < m, so t·k + k <= x.len() and t·n + n <= y.len() by the
        // launch contract; gates is published; row t of y is this block's.
        unsafe { fold(x, t * k, n, tid, gates, &mut y, t * n) };
    }

    /// HC post: `out[(t·4 + i)·n + d] = res[(t·4 + i)·n + d] + x[t·n + d] ·
    /// mix[t·MIX + HC + i]`, one thread a value over `m·4·n` values.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= m * n,
            res.len() >= m * 4 * n,
            mix.len() >= m * 8,
            out.len() >= m * 4 * n
        )
    )]
    pub fn hc_ident_post(
        x: &[f32],
        res: &[f32],
        mix: &[f32],
        n: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n = n as usize;
        if i >= m as usize * HC * n {
            return;
        }
        let (t, s, d) = (i / (HC * n), i % (HC * n) / n, i % n);
        // SAFETY: i < m·4n gives t < m, s < 4, d < n: t·n + d < m·n <=
        // x.len(), i < res.len() and out.len(), t·MIX + HC + s < m·8 <=
        // mix.len() by the launch contract; thread i owns out[i].
        unsafe {
            let post = *mix.get_unchecked(t * MIX + HC + s);
            let v = add_rn_f32(
                *res.get_unchecked(i),
                mul_rn_f32(*x.get_unchecked(t * n + d), post),
            );
            *out.get_unchecked_mut(i) = v;
        }
    }
}

// -------------------------------------------------------------- launchers

/// [`HcIdentKernels::enqueue_pre`]'s arguments: `m` tokens' streams `x`
/// (`4·n` values each), `hc_fn` as `w` (eight rows of `4·n`), its `scale`
/// (2) and `base` (8), the file's norm and hc epsilons and `hc.magnitude`;
/// the outputs `mix` (`m·MIX`) and the fold `y` (`m·n`).
pub struct PreArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub scale: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    pub eps_norm: f32,
    pub eps_hc: f32,
    pub magnitude: f32,
    pub n: usize,
    pub m: usize,
    pub mix: &'a mut DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
    /// Where a token whose squares leave the norm no scale is raised.
    pub fault: FaultSink,
}

/// [`HcIdentKernels::enqueue_head`]'s arguments: as [`PreArgs`] with the
/// head's `output_hc_fn` (four rows), its one `scale` and four `base` values,
/// and only the fold `y` out.
pub struct HeadArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub scale: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    pub eps_norm: f32,
    pub eps_hc: f32,
    pub n: usize,
    pub m: usize,
    pub y: &'a mut DeviceBuffer<f32>,
    /// As [`PreArgs::fault`].
    pub fault: FaultSink,
}

/// [`HcIdentKernels::enqueue_post`]'s arguments: the sub-layer output `x`
/// (`m·n`), the streams it adds into, `res` (`m·4·n`), the gates `mix` the
/// pre entry wrote, and `out` in `res`'s layout (another buffer than `res`).
pub struct PostArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub res: &'a DeviceBuffer<f32>,
    pub mix: &'a DeviceBuffer<f32>,
    pub n: usize,
    pub m: usize,
    pub out: &'a mut DeviceBuffer<f32>,
}

/// The loaded identity-HC module. Owns no stream: each enqueue takes the
/// engine stream, so its launches order with the rest of the step and capture.
pub struct HcIdentKernels {
    module: hc_ident_kernels::LoadedModule,
}

impl HcIdentKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcIdentKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(hc_ident_kernels, ctx)? };
        Ok(HcIdentKernels { module })
    }

    /// Enqueue HC pre over `a.m` tokens: the gates into `a.mix` and the fold
    /// into `a.y`, per the module contract. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_pre(&self, stream: &CudaStream, a: PreArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "hc_ident::enqueue_pre";
        let k = check_dims(WHAT, a.n, a.m)?;
        need(WHAT, "x", a.x.len(), a.m * k)?;
        need(WHAT, "w", a.w.len(), MIX * k)?;
        need(WHAT, "scale", a.scale.len(), 2)?;
        need(WHAT, "base", a.base.len(), MIX)?;
        need(WHAT, "mix", a.mix.len(), a.m * MIX)?;
        need(WHAT, "y", a.y.len(), a.m * a.n)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let n = launch_u32(WHAT, "n", a.n)?;
        let prep = self
            .module
            .prepare_hc_ident_pre(LaunchConfig1D::new(m, BLOCK, 0))?;
        self.module.hc_ident_pre(
            stream,
            &prep,
            a.x,
            a.w,
            a.scale,
            a.base,
            a.eps_norm,
            a.eps_hc,
            a.magnitude,
            n,
            m,
            a.fault,
            a.mix,
            a.y,
        )?;
        Ok(())
    }

    /// Enqueue the HC head over `a.m` tokens: the fold into `a.y`.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_head(&self, stream: &CudaStream, a: HeadArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "hc_ident::enqueue_head";
        let k = check_dims(WHAT, a.n, a.m)?;
        need(WHAT, "x", a.x.len(), a.m * k)?;
        need(WHAT, "w", a.w.len(), HC * k)?;
        need(WHAT, "scale", a.scale.len(), 1)?;
        need(WHAT, "base", a.base.len(), HC)?;
        need(WHAT, "y", a.y.len(), a.m * a.n)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let n = launch_u32(WHAT, "n", a.n)?;
        let prep = self
            .module
            .prepare_hc_ident_head(LaunchConfig1D::new(m, BLOCK, 0))?;
        self.module.hc_ident_head(
            stream, &prep, a.x, a.w, a.scale, a.base, a.eps_norm, a.eps_hc, n, m, a.fault, a.y,
        )?;
        Ok(())
    }

    /// Enqueue HC post over `a.m` tokens: `a.out = a.res + a.x·post`.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_post(&self, stream: &CudaStream, a: PostArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "hc_ident::enqueue_post";
        let k = check_dims(WHAT, a.n, a.m)?;
        need(WHAT, "x", a.x.len(), a.m * a.n)?;
        need(WHAT, "res", a.res.len(), a.m * k)?;
        need(WHAT, "mix", a.mix.len(), a.m * MIX)?;
        need(WHAT, "out", a.out.len(), a.m * k)?;
        let total = launch_u32(WHAT, "m·4·n", a.m * k)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let n = launch_u32(WHAT, "n", a.n)?;
        let prep = self.module.prepare_hc_ident_post(LaunchConfig1D::new(
            total.div_ceil(BLOCK),
            BLOCK,
            0,
        ))?;
        self.module
            .hc_ident_post(stream, &prep, a.x, a.res, a.mix, n, m, a.out)?;
        Ok(())
    }
}

/// `4·n`, the flat width of a token's streams, refusing an empty launch or a
/// width that overflows.
fn check_dims(what: &'static str, n: usize, m: usize) -> Result<usize, GpuError> {
    if n == 0 || m == 0 {
        return Err(GpuError::Shape {
            what,
            detail: format!("need n >= 1 and m >= 1, got n={n} m={m}"),
        });
    }
    HC.checked_mul(n)
        .and_then(|k| k.checked_mul(m).map(|_| k))
        .ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("m·4·n = {m}·4·{n} overflows"),
        })
}

/// Refuse a buffer `name` of `len` elements shorter than `want`.
fn need(what: &'static str, name: &str, len: usize, want: usize) -> Result<(), GpuError> {
    if len < want {
        return Err(GpuError::Shape {
            what,
            detail: format!("{name}.len() {len} < {want}"),
        });
    }
    Ok(())
}
