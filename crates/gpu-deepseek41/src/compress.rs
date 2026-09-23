//! Pooled compressed KV: DS4_COMP pools `ratio` consecutive tokens'
//! `wkv_c · x` under softmax(`wgate_c · x`) weights into one latent row,
//! then norm, rope and an f16 write into the compressed cache.
//!
//! A stream of ratio `r` keeps a ring of `r` projections (slot `p % r`) and
//! a cache of compressed rows; the token at `p` completes a group when
//! `p % r == r − 1`, and the group's row `p / r` pools positions
//! `p + 1 − r ..= p` — ring slots for the positions before the step, the
//! step's own projections for the rest — roped at `p + 1 − r`
//! (`model::arch::deepseek41::plan`). The kernels read those integers from a
//! device buffer the host fills before each step ([`CompGeom::pack`]), never
//! from launch arguments, so one captured graph serves every position: a
//! step that completes no group runs the same launches and writes no row.
//!
//! - [`CompressKernels::enqueue_pool`] (ratio at least 2): per completed
//!   group, DS4_COMP, the row norm, the pre-rope row in f32 (the index key's
//!   input), the tail rope and the f16 row; then the ring keeps the step's
//!   last projection of each residue (the persist).
//! - [`CompressKernels::enqueue_rows`] (ratio 1): DS4_COMP of one row is the
//!   identity — `max = s`, `w = expf(s − s) = expf(+0) = 1`, `sum = 1`,
//!   `res = fma(1, kv, +0) = kv`, `y = kv / 1` for every finite score, up to
//!   the sign of a zero `kv` — so a group `[p, p]` is its token's projection:
//!   no score, no ring, no persist (a ratio-1 group never reads outside the
//!   step's batch).
//!
//! Numeric contract: DS4_COMP is ik's type 1 (`ggml.c`
//! `ggml_compute_forward_ds4_comp_type1`) op for op — the C `MAX` from
//! `−∞` over the group's scores, then per row in order `w = expf(s − max)`,
//! `sum += w`, `res = fma(w, kv, res)` (ik's CPU build contracts that line),
//! and `y = res / sum` — with the device `expf` (`__nv_expf`) where ik calls
//! glibc's: the one difference in the pooled value. The norm sums the
//! squares as an f32 tree ([`norm_scale`]) where ik sums in f64. The rope is
//! `rope::rope_pair_rn` on the YaRN table of the group's first position, ik's
//! unfused rotation; the cache value is that row rounded once to f16
//! (round to nearest even).
//!
//! Buffers are token-major (ggml's order): token `t`'s projection is
//! `kv[t·WIDTH ..]`. Our q3_K gemv writes `[row·m + t]`, which is the same
//! order at one token.

use bloomery_gpu::elem::rms_scale;
use bloomery_gpu::flash::f32_to_f16_bits;
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

use crate::rope::rope_pair_rn;

/// Threads per block of both entries: one block per group.
const BLOCK: u32 = 128;
/// Values one thread owns: `PER_THREAD·t ..`, contiguous, so a tail pair
/// never straddles two threads.
const PER_THREAD: usize = 4;
/// Warps per block, the width of the norm's second stage.
const WARPS: usize = BLOCK as usize / 32;
/// Values of a compressed row (V4.1's `n_embd_head`): one block's values.
pub const WIDTH: usize = PER_THREAD * BLOCK as usize;
const _: () = assert!(WIDTH == 512 && WARPS == 4);

// ------------------------------------------------------------ step buffer

/// Word of the step buffer: the groups the step completes.
pub(crate) const W_GROUPS: usize = 0;
/// Word of the step buffer: the ring slots the step keeps.
const W_PERSISTS: usize = 1;

/// Word of group `g`'s compressed row.
pub(crate) const fn w_row(g: usize) -> usize {
    2 + g
}

/// Word of group `g`'s `k`-th pooled projection: below `ratio` a ring slot,
/// from `ratio` on the step's token `index − ratio`.
const fn w_read(max_groups: usize, ratio: usize, g: usize, k: usize) -> usize {
    2 + max_groups + g * ratio + k
}

/// Word of the `j`-th kept token (a batch index); its slot is `ratio` words on.
const fn w_persist(max_groups: usize, ratio: usize, j: usize) -> usize {
    2 + max_groups * (1 + ratio) + j
}

/// The shape one stream's launches are captured with: every step of that
/// graph packs its integers into this geometry, whatever it completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompGeom {
    /// Positions pooled into one compressed row.
    pub ratio: usize,
    /// Groups one step can complete (a step of `m` tokens completes at most
    /// `m.div_ceil(ratio)`): the block count of every launch.
    pub max_groups: usize,
    /// The step's tokens: the projection rows the pooling reads.
    pub tokens: usize,
    /// Rows of the compressed cache and of the index-key cache (both are
    /// written at the group's row).
    pub rows: usize,
}

impl CompGeom {
    /// u32 words of the step buffer: the counts, a row per group, `ratio`
    /// reads per group, and `ratio` kept tokens with their slots.
    #[must_use]
    pub fn words(&self) -> usize {
        w_persist(self.max_groups, self.ratio, 0) + 2 * self.ratio
    }

    /// Every extent at least one, and the step buffer and cache addressable
    /// by the kernels' u32 arithmetic.
    pub(crate) fn check(&self, what: &'static str) -> Result<(), GpuError> {
        if self.ratio == 0 || self.max_groups == 0 || self.tokens == 0 || self.rows == 0 {
            return Err(GpuError::Shape {
                what,
                detail: format!("a stream needs ratio, max_groups, tokens and rows >= 1: {self:?}"),
            });
        }
        launch_u32(what, "words", self.words())?;
        launch_u32(what, "rows * WIDTH", self.rows * WIDTH)?;
        Ok(())
    }

    /// Write one step's integers into `out` (at least [`words`](Self::words)
    /// u32), every word the layout does not use zeroed. The step is checked
    /// against the geometry, since a word out of range would address past a
    /// buffer the kernels then trust: at most `max_groups` groups of `ratio`
    /// reads, rows ascending below `rows`, each read below `ratio + tokens`
    /// and a ring slot (a read below `ratio`) only in the first group — the
    /// one group that can span the step's start, and the one block that
    /// keeps the ring, so its reads precede the ring's writes. A ratio-1
    /// stream keeps no ring: its reads are all tokens of the step, and its
    /// kept tokens are dropped. A ring's kept slots ascend below `ratio`,
    /// each fed by a token of the step.
    pub fn pack(&self, step: &StepInts<'_>, out: &mut [u32]) -> Result<(), GpuError> {
        let what = "CompGeom::pack";
        self.check(what)?;
        let bad = |detail: String| GpuError::Shape { what, detail };
        let (r, gm) = (self.ratio, self.max_groups);
        let groups = step.write_row.len();
        if out.len() < self.words() {
            return Err(bad(format!(
                "{} words for a layout of {}",
                out.len(),
                self.words()
            )));
        }
        if groups > gm || step.read.len() != groups * r {
            return Err(bad(format!(
                "{groups} group(s) with {} reads, want at most {gm} groups of {r}",
                step.read.len()
            )));
        }
        let ascending = step.write_row.windows(2).all(|w| w[0] < w[1]);
        let row_bad = step.write_row.iter().find(|&&w| w >= self.rows as u64);
        if !ascending || row_bad.is_some() {
            return Err(bad(format!(
                "rows {:?} must ascend below {}",
                step.write_row, self.rows
            )));
        }
        for (i, &s) in step.read.iter().enumerate() {
            let s = s as usize;
            let ring = s < r;
            if s >= r + self.tokens || (ring && (r == 1 || i >= r)) {
                return Err(bad(format!(
                    "read {i} is source {s}: want below {} and a ring slot only in the first \
                     group of a ratio above 1 ({r})",
                    r + self.tokens
                )));
            }
        }
        let kept = if r == 1 { 0 } else { step.persist_src.len() };
        if r > 1
            && (kept != step.persist_dst.len()
                || kept > r
                || step.persist_src.iter().any(|&s| s as usize >= self.tokens)
                || step.persist_dst.windows(2).any(|w| w[0] >= w[1])
                || step.persist_dst.iter().any(|&d| d as usize >= r))
        {
            return Err(bad(format!(
                "kept tokens {:?} into slots {:?}: want pairs, tokens below {}, slots ascending \
                 below {r}",
                step.persist_src, step.persist_dst, self.tokens
            )));
        }
        out.fill(0);
        out[W_GROUPS] = launch_u32(what, "groups", groups)?;
        out[W_PERSISTS] = launch_u32(what, "kept", kept)?;
        for (g, &w) in step.write_row.iter().enumerate() {
            out[w_row(g)] = u32::try_from(w).map_err(|_| bad(format!("row {w} past u32")))?;
        }
        out[w_read(gm, r, 0, 0)..][..step.read.len()].copy_from_slice(step.read);
        if kept > 0 {
            out[w_persist(gm, r, 0)..][..kept].copy_from_slice(step.persist_src);
            out[w_persist(gm, r, r)..][..kept].copy_from_slice(step.persist_dst);
        }
        Ok(())
    }
}

/// One stream's integers for one step, in the plan's terms
/// (`model::arch::deepseek41::plan::StreamStep`): per completed group its
/// compressed row and its `ratio` pooled projections (ring slots, then the
/// step's tokens offset by `ratio`), and the step's tokens the ring keeps
/// with their slots.
#[derive(Clone, Copy, Debug)]
pub struct StepInts<'a> {
    pub write_row: &'a [u64],
    pub read: &'a [u32],
    pub persist_src: &'a [u32],
    pub persist_dst: &'a [u32],
}

// ------------------------------------------------------------------ cores

/// Four contiguous f32 at `base`.
///
/// # Safety
/// `base + 4 <= x.len()`.
#[inline(always)]
unsafe fn load4(x: &[f32], base: usize) -> [f32; PER_THREAD] {
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

/// Four contiguous f32 at `base` of a slice the kernel also writes.
///
/// # Safety
/// `base + 4 <= x.len()`, and no other thread writes these four during the
/// launch.
#[inline(always)]
unsafe fn load4_mut(x: &mut DisjointSlice<f32>, base: usize) -> [f32; PER_THREAD] {
    // SAFETY: base + 3 < x.len() by this fn's contract.
    unsafe {
        [
            *x.get_unchecked_mut(base),
            *x.get_unchecked_mut(base + 1),
            *x.get_unchecked_mut(base + 2),
            *x.get_unchecked_mut(base + 3),
        ]
    }
}

/// Store four f32 at `base`.
///
/// # Safety
/// `base + 4 <= x.len()`, and this thread is the only writer of the four.
#[inline(always)]
unsafe fn store4(x: &mut DisjointSlice<f32>, base: usize, v: [f32; PER_THREAD]) {
    // SAFETY: base + 3 < x.len(), one writer, by this fn's contract.
    unsafe {
        *x.get_unchecked_mut(base) = v[0];
        *x.get_unchecked_mut(base + 1) = v[1];
        *x.get_unchecked_mut(base + 2) = v[2];
        *x.get_unchecked_mut(base + 3) = v[3];
    }
}

/// The C `MAX(a, b)` — `a > b ? a : b` — of each of four pairs: ik's
/// running max from `−∞`, which returns `b` when either is a NaN.
#[inline(always)]
fn c_max4(a: [f32; PER_THREAD], b: [f32; PER_THREAD]) -> [f32; PER_THREAD] {
    [
        if a[0] > b[0] { a[0] } else { b[0] },
        if a[1] > b[1] { a[1] } else { b[1] },
        if a[2] > b[2] { a[2] } else { b[2] },
        if a[3] > b[3] { a[3] } else { b[3] },
    ]
}

/// One pooled source's term of DS4_COMP type 1 on four values: `w =
/// expf(s − max)`, `sum += w`, `res = fma(w, kv, res)`. The sum is an
/// explicit round-to-nearest add: the build contracts every multiply that
/// feeds an add, and the device `expf` can end in one.
#[inline(always)]
fn pool_term(
    acc: ([f32; PER_THREAD], [f32; PER_THREAD]),
    s: [f32; PER_THREAD],
    mx: [f32; PER_THREAD],
    x: [f32; PER_THREAD],
) -> ([f32; PER_THREAD], [f32; PER_THREAD]) {
    let (sum, res) = acc;
    let w = [
        (s[0] - mx[0]).exp(),
        (s[1] - mx[1]).exp(),
        (s[2] - mx[2]).exp(),
        (s[3] - mx[3]).exp(),
    ];
    (
        [
            add_rn_f32(sum[0], w[0]),
            add_rn_f32(sum[1], w[1]),
            add_rn_f32(sum[2], w[2]),
            add_rn_f32(sum[3], w[3]),
        ],
        [
            fma_rn_f32(w[0], x[0], res[0]),
            fma_rn_f32(w[1], x[1], res[1]),
            fma_rn_f32(w[2], x[2], res[2]),
            fma_rn_f32(w[3], x[3], res[3]),
        ],
    )
}

/// The row scale of a block's pooled row: each thread's four squares summed
/// by fused multiply-adds in value order from `+0`, the warp butterfly
/// (`reduce_sum_f32`: xor 16, 8, 4, 2, 1, own + partner), the four warp
/// sums as `(w0 + w1) + (w2 + w3)`, then `elem::rms_scale` at [`WIDTH`].
/// This order is the gate's host transcription.
///
/// # Safety
/// `ws` is the calling block's shared array of [`WARPS`] f32, and every
/// thread of the block calls this once (it holds a block barrier).
#[inline(always)]
unsafe fn norm_scale(y: [f32; PER_THREAD], ws: *mut f32, t: usize, eps: f32) -> f32 {
    let acc = fma_rn_f32(y[0], y[0], 0.0);
    let acc = fma_rn_f32(y[1], y[1], acc);
    let acc = fma_rn_f32(y[2], y[2], acc);
    let acc = fma_rn_f32(y[3], y[3], acc);
    let part = warp::reduce_sum_f32(acc);
    if warp::lane_id() == 0 {
        // SAFETY: t / 32 < WARPS; one lane per warp writes its slot.
        unsafe {
            *ws.add(t / 32) = part;
        }
    }
    thread::sync_threads();
    // SAFETY: block-shared, WARPS slots, written before the barrier that
    // publishes them.
    let w = unsafe { [*ws.add(0), *ws.add(1), *ws.add(2), *ws.add(3)] };
    rms_scale((w[0] + w[1]) + (w[2] + w[3]), WIDTH as u32, eps)
}

/// Where one group's row goes.
struct RowOut<'a, 'p, 'c> {
    pre: &'a mut DisjointSlice<'p, f32>,
    cache: &'a mut DisjointSlice<'c, u16>,
}

/// Thread `t`'s four values of group `g`'s pooled row `y`, normalized
/// (`(scale · gain) · y`, each product rounded on its own — the explicit
/// intrinsics keep them out of any contraction), stored in f32 at
/// `pre[g·WIDTH ..]`, turned when
/// they sit in the tail (`n_dims` values, the table at `cs[g·n_dims ..]`,
/// two pairs per thread), and rounded to f16 into cache row `row`.
///
/// # Safety
/// As [`norm_scale`]; `gain.len() >= WIDTH`, `cs.len() >= (g + 1)·n_dims`,
/// `pre.len() >= (g + 1)·WIDTH`, `cache.len() >= (row + 1)·WIDTH`,
/// `n_dims` a multiple of 4 at most `WIDTH`, and no other block writes this
/// `pre` row or this cache row.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn finish_row(
    y: [f32; PER_THREAD],
    ws: *mut f32,
    gain: &[f32],
    cs: &[f32],
    eps: f32,
    n_dims: usize,
    g: usize,
    t: usize,
    row: usize,
    out: RowOut<'_, '_, '_>,
) {
    // SAFETY: forwarded from this fn's contract.
    let scale = unsafe { norm_scale(y, ws, t, eps) };
    let v = PER_THREAD * t;
    // SAFETY: v + 4 <= WIDTH <= gain.len().
    let gw = unsafe { load4(gain, v) };
    let mut n = [
        mul_rn_f32(mul_rn_f32(scale, gw[0]), y[0]),
        mul_rn_f32(mul_rn_f32(scale, gw[1]), y[1]),
        mul_rn_f32(mul_rn_f32(scale, gw[2]), y[2]),
        mul_rn_f32(mul_rn_f32(scale, gw[3]), y[3]),
    ];
    // SAFETY: g·WIDTH + v + 4 <= (g + 1)·WIDTH <= pre.len(); block g owns
    // pre row g and thread t its four values.
    unsafe { store4(out.pre, g * WIDTH + v, n) };
    let tail0 = WIDTH - n_dims;
    if v >= tail0 {
        // SAFETY: v − tail0 + 4 <= n_dims, so the four table values sit
        // inside group g's table, below cs.len().
        let c = unsafe { load4(cs, g * n_dims + v - tail0) };
        let (a0, a1) = rope_pair_rn(n[0], n[1], c[0], c[1]);
        let (a2, a3) = rope_pair_rn(n[2], n[3], c[2], c[3]);
        n = [a0, a1, a2, a3];
    }
    let base = row * WIDTH + v;
    // SAFETY: base + 4 <= (row + 1)·WIDTH <= cache.len(); block g owns cache
    // row `row` (rows ascend, host-checked) and thread t its four values.
    unsafe {
        *out.cache.get_unchecked_mut(base) = f32_to_f16_bits(n[0]);
        *out.cache.get_unchecked_mut(base + 1) = f32_to_f16_bits(n[1]);
        *out.cache.get_unchecked_mut(base + 2) = f32_to_f16_bits(n[2]);
        *out.cache.get_unchecked_mut(base + 3) = f32_to_f16_bits(n[3]);
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod comp_kernels {
    use super::*;

    /// The pooling of a stream of ratio `ratio` (at least 2), one block per
    /// group up to `max_groups`, four values per thread. The step buffer
    /// (`CompGeom::pack`) says how many groups completed, each group's row
    /// and its `ratio` sources: ring slot `s < ratio` of `ring_kv` /
    /// `ring_score`, else the step's token `s − ratio` of `kv` / `score`.
    /// Block `g` pools group `g` with ik's DS4_COMP type 1 order, then
    /// `finish_row`. A group whose integers leave their buffers, or a group
    /// past the first that names a ring slot, writes nothing. Block 0 then
    /// copies each kept token's projections into its ring slot; only block 0
    /// reads the ring, each thread its own four columns, so every read of a
    /// slot precedes its overwrite in program order. The group guard is
    /// block-uniform, so no barrier is skipped.
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
            step.len() >= 2 + max_groups + max_groups * ratio + 2 * ratio,
            kv.len() >= m * 512,
            score.len() >= m * 512,
            gain.len() >= 512,
            cs.len() >= max_groups * n_dims,
            ring_kv.len() >= ratio * 512,
            ring_score.len() >= ratio * 512,
            pre.len() >= max_groups * 512,
            cache.len() >= rows * 512
        )
    )]
    pub fn ds41_comp_pool(
        step: &[u32],
        kv: &[f32],
        score: &[f32],
        gain: &[f32],
        cs: &[f32],
        eps: f32,
        ratio: u32,
        max_groups: u32,
        m: u32,
        n_dims: u32,
        rows: u32,
        mut ring_kv: DisjointSlice<f32>,
        mut ring_score: DisjointSlice<f32>,
        mut pre: DisjointSlice<f32>,
        mut cache: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let g = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let (r, gm, m) = (ratio as usize, max_groups as usize, m as usize);
        let v = PER_THREAD * t;
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // SAFETY: words 0 and 1 exist: step.len() >= 2 + … by the contract.
        let (groups, kept) = unsafe {
            (
                *step.get_unchecked(W_GROUPS) as usize,
                *step.get_unchecked(W_PERSISTS) as usize,
            )
        };

        if g < groups && g < gm {
            // SAFETY: w_row(g) = 2 + g < 2 + max_groups <= step.len().
            let row = unsafe { *step.get_unchecked(w_row(g)) } as usize;
            let mut ok = row < rows as usize;
            let mut k = 0;
            while k < r {
                // SAFETY: w_read(gm, r, g, k) < 2 + gm + gm·r <= step.len()
                // because g < gm and k < r.
                let s = unsafe { *step.get_unchecked(w_read(gm, r, g, k)) } as usize;
                ok &= s < r + m && (s >= r || g == 0);
                k += 1;
            }
            if ok {
                // The max of the group's scores, per value, by the C `MAX`
                // from −∞ in source order.
                let mut mx = [f32::NEG_INFINITY; PER_THREAD];
                let mut k = 0;
                while k < r {
                    // SAFETY: as the check loop above.
                    let s = unsafe { *step.get_unchecked(w_read(gm, r, g, k)) } as usize;
                    let sc = if s < r {
                        // SAFETY: s < ratio, so s·512 + v + 4 <= ring_score.len().
                        unsafe { load4_mut(&mut ring_score, s * WIDTH + v) }
                    } else {
                        // SAFETY: s − r < m (checked), so the four values sit
                        // below m·512 <= score.len().
                        unsafe { load4(score, (s - r) * WIDTH + v) }
                    };
                    mx = c_max4(mx, sc);
                    k += 1;
                }
                // The weights, their sum and the weighted sum, in source order.
                let mut acc = ([0.0f32; PER_THREAD], [0.0f32; PER_THREAD]);
                let mut k = 0;
                while k < r {
                    // SAFETY: as the check loop above.
                    let s = unsafe { *step.get_unchecked(w_read(gm, r, g, k)) } as usize;
                    let (sc, x) = if s < r {
                        // SAFETY: as the max pass; ring_kv has the same extent.
                        unsafe {
                            (
                                load4_mut(&mut ring_score, s * WIDTH + v),
                                load4_mut(&mut ring_kv, s * WIDTH + v),
                            )
                        }
                    } else {
                        // SAFETY: as the max pass; kv has the same extent.
                        unsafe {
                            (
                                load4(score, (s - r) * WIDTH + v),
                                load4(kv, (s - r) * WIDTH + v),
                            )
                        }
                    };
                    acc = pool_term(acc, sc, mx, x);
                    k += 1;
                }
                let (sum, res) = acc;
                let y = [
                    res[0] / sum[0],
                    res[1] / sum[1],
                    res[2] / sum[2],
                    res[3] / sum[3],
                ];
                let out = RowOut {
                    pre: &mut pre,
                    cache: &mut cache,
                };
                // SAFETY: every thread of the block takes this branch (its
                // condition reads only block-uniform values); gain, cs, pre
                // and cache hold group g's extents by the contract; n_dims is
                // a multiple of 4 at most 512 (host-checked); rows ascend
                // (host-checked), so no other block writes row `row`.
                unsafe {
                    finish_row(y, ws, gain, cs, eps, n_dims as usize, g, t, row, out);
                }
            }
        }

        if g == 0 {
            let mut j = 0;
            while j < kept && j < r {
                // SAFETY: w_persist(gm, r, j) + r < 2 + gm + gm·r + 2r <=
                // step.len() because j < r.
                let (src, dst) = unsafe {
                    (
                        *step.get_unchecked(w_persist(gm, r, j)) as usize,
                        *step.get_unchecked(w_persist(gm, r, j) + r) as usize,
                    )
                };
                if src < m && dst < r {
                    // SAFETY: src < m bounds both reads below m·512; dst < r
                    // bounds both writes below r·512; thread t of block 0 is
                    // the only thread touching these four ring columns.
                    unsafe {
                        store4(&mut ring_kv, dst * WIDTH + v, load4(kv, src * WIDTH + v));
                        store4(
                            &mut ring_score,
                            dst * WIDTH + v,
                            load4(score, src * WIDTH + v),
                        );
                    }
                }
                j += 1;
            }
        }
    }

    /// The groups of a ratio-1 stream, one block per group up to
    /// `max_groups`: the step buffer's single read of group `g` names the
    /// step's token `s − 1`, whose projection is the pooled row (DS4_COMP
    /// of one row is the identity), then `finish_row`. A group whose
    /// integers leave their buffers writes nothing; the kept tokens of the
    /// step buffer are not read — this stream has no ring.
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
            step.len() >= 4 + 2 * max_groups,
            kv.len() >= m * 512,
            gain.len() >= 512,
            cs.len() >= max_groups * n_dims,
            pre.len() >= max_groups * 512,
            cache.len() >= rows * 512
        )
    )]
    pub fn ds41_comp_rows(
        step: &[u32],
        kv: &[f32],
        gain: &[f32],
        cs: &[f32],
        eps: f32,
        max_groups: u32,
        m: u32,
        n_dims: u32,
        rows: u32,
        mut pre: DisjointSlice<f32>,
        mut cache: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let g = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let gm = max_groups as usize;
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // SAFETY: word 0 exists: step.len() >= 4 + … by the contract.
        let groups = unsafe { *step.get_unchecked(W_GROUPS) } as usize;
        if g >= groups || g >= gm {
            return;
        }
        // SAFETY: w_row(g) = 2 + g and w_read(gm, 1, g, 0) = 2 + gm + g are
        // below 2 + 2·gm <= step.len() because g < gm.
        let (row, s) = unsafe {
            (
                *step.get_unchecked(w_row(g)) as usize,
                *step.get_unchecked(w_read(gm, 1, g, 0)) as usize,
            )
        };
        if row >= rows as usize || s == 0 || s > m as usize {
            return;
        }
        // SAFETY: s − 1 < m, so the four values sit below m·512 <= kv.len().
        let y = unsafe { load4(kv, (s - 1) * WIDTH + PER_THREAD * t) };
        let out = RowOut {
            pre: &mut pre,
            cache: &mut cache,
        };
        // SAFETY: the guards above are block-uniform, so every thread of the
        // block gets here; the extents are the contract's; n_dims is a
        // multiple of 4 at most 512 (host-checked); rows ascend
        // (host-checked).
        unsafe {
            finish_row(y, ws, gain, cs, eps, n_dims as usize, g, t, row, out);
        }
    }
}

// -------------------------------------------------------------- launchers

/// [`CompressKernels::enqueue_pool`]'s arguments. `step` holds
/// `geom.words()` u32 ([`CompGeom::pack`]); `kv` and `score` the step's
/// `geom.tokens` projections, token-major; `gain` the row norm's weights;
/// `cs` `geom.max_groups` tables of `n_dims` (group `g`'s at `g·n_dims`,
/// the YaRN table of its first position — [`crate::rope::RopeTable::push`]);
/// `ring_kv`/`ring_score` the stream's `ratio` kept projections (read, and
/// updated in place); `pre` takes `geom.max_groups` pre-rope rows in f32 and
/// `cache` is the compressed cache, `geom.rows` rows of [`WIDTH`] f16.
pub struct PoolArgs<'a> {
    pub geom: CompGeom,
    pub step: &'a DeviceBuffer<u32>,
    pub kv: &'a DeviceBuffer<f32>,
    pub score: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub cs: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub n_dims: usize,
    pub ring_kv: &'a mut DeviceBuffer<f32>,
    pub ring_score: &'a mut DeviceBuffer<f32>,
    pub pre: &'a mut DeviceBuffer<f32>,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`CompressKernels::enqueue_rows`]'s arguments: [`PoolArgs`] of a ratio-1
/// stream, which has no score and no ring.
pub struct RowsArgs<'a> {
    pub geom: CompGeom,
    pub step: &'a DeviceBuffer<u32>,
    pub kv: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub cs: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub n_dims: usize,
    pub pre: &'a mut DeviceBuffer<f32>,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// The tail and the buffers every compressor launch shares, checked.
fn check_common(
    what: &'static str,
    geom: &CompGeom,
    n_dims: usize,
    lens: [(&'static str, usize, usize); 5],
    cache: &DeviceTensor<u16>,
) -> Result<(), GpuError> {
    geom.check(what)?;
    if n_dims == 0 || !n_dims.is_multiple_of(PER_THREAD) || n_dims > WIDTH {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "the tail is two pairs per thread: n_dims must be a positive multiple of \
                 {PER_THREAD} at most {WIDTH}, got {n_dims}"
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
    for (name, have, need) in lens {
        if have < need {
            return Err(GpuError::Shape {
                what,
                detail: format!("{name}.len() {have}, need {need}"),
            });
        }
    }
    Ok(())
}

/// The loaded compressor module. Owns no stream: each enqueue takes the
/// engine stream, so its launches order with the rest of the step and
/// capture.
pub struct CompressKernels {
    module: comp_kernels::LoadedModule,
}

impl CompressKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<CompressKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { comp_kernels::load(ctx)? };
        Ok(CompressKernels { module })
    }

    /// Enqueue one step of a stream of ratio at least 2: every group the
    /// step buffer completes pooled, normalized, stored pre-rope in f32 and
    /// turned into its f16 cache row, then the kept tokens copied into their
    /// ring slots. `geom.max_groups` blocks whatever the step completes.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_pool(&self, stream: &CudaStream, args: PoolArgs<'_>) -> Result<(), GpuError> {
        let what = "enqueue_pool";
        let PoolArgs {
            geom,
            step,
            kv,
            score,
            gain,
            cs,
            eps,
            n_dims,
            ring_kv,
            ring_score,
            pre,
            cache,
        } = args;
        if geom.ratio < 2 {
            return Err(GpuError::Shape {
                what,
                detail: format!("a ratio-{} stream has no ring: enqueue_rows", geom.ratio),
            });
        }
        let (m, gm, r) = (geom.tokens, geom.max_groups, geom.ratio);
        check_common(
            what,
            &geom,
            n_dims,
            [
                ("step", step.len(), geom.words()),
                ("kv", kv.len().min(score.len()), m * WIDTH),
                ("gain", gain.len(), WIDTH),
                ("cs", cs.len(), gm * n_dims),
                ("pre", pre.len(), gm * WIDTH),
            ],
            cache,
        )?;
        if ring_kv.len().min(ring_score.len()) < r * WIDTH {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "rings of {} and {} values, need {} (ratio {r} of {WIDTH})",
                    ring_kv.len(),
                    ring_score.len(),
                    r * WIDTH
                ),
            });
        }
        let grid = launch_u32(what, "max_groups", gm)?;
        let prep = self
            .module
            .prepare_ds41_comp_pool(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_comp_pool(
            stream,
            &prep,
            step,
            kv,
            score,
            gain,
            cs,
            eps,
            launch_u32(what, "ratio", r)?,
            grid,
            launch_u32(what, "tokens", m)?,
            launch_u32(what, "n_dims", n_dims)?,
            launch_u32(what, "rows", geom.rows)?,
            ring_kv,
            ring_score,
            pre,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue one step of a ratio-1 stream: every group the step buffer
    /// completes is its token's projection, normalized, stored pre-rope in
    /// f32 and turned into its f16 cache row. `geom.max_groups` blocks.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_rows(&self, stream: &CudaStream, args: RowsArgs<'_>) -> Result<(), GpuError> {
        let what = "enqueue_rows";
        let RowsArgs {
            geom,
            step,
            kv,
            gain,
            cs,
            eps,
            n_dims,
            pre,
            cache,
        } = args;
        if geom.ratio != 1 {
            return Err(GpuError::Shape {
                what,
                detail: format!("a ratio-{} stream pools: enqueue_pool", geom.ratio),
            });
        }
        let (m, gm) = (geom.tokens, geom.max_groups);
        check_common(
            what,
            &geom,
            n_dims,
            [
                ("step", step.len(), geom.words()),
                ("kv", kv.len(), m * WIDTH),
                ("gain", gain.len(), WIDTH),
                ("cs", cs.len(), gm * n_dims),
                ("pre", pre.len(), gm * WIDTH),
            ],
            cache,
        )?;
        let grid = launch_u32(what, "max_groups", gm)?;
        let prep = self
            .module
            .prepare_ds41_comp_rows(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_comp_rows(
            stream,
            &prep,
            step,
            kv,
            gain,
            cs,
            eps,
            grid,
            launch_u32(what, "tokens", m)?,
            launch_u32(what, "n_dims", n_dims)?,
            launch_u32(what, "rows", geom.rows)?,
            pre,
            cache.buf_mut(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CompGeom, StepInts};

    /// The layout the kernels read, packed from the steps the V4.1 plan
    /// makes (`model::arch::deepseek41::plan`): a 5-token prefill from 0,
    /// and one decode step at an even and at an odd position — and the
    /// steps it must refuse.
    #[test]
    fn pack_lays_out_the_plan_and_refuses_what_the_kernels_cannot_run() {
        let csa = CompGeom {
            ratio: 2,
            max_groups: 3,
            tokens: 5,
            rows: 256,
        };
        let mut out = vec![7u32; csa.words()];
        assert_eq!(csa.words(), 2 + 3 + 6 + 4);
        let prefill = StepInts {
            write_row: &[0, 1],
            read: &[2, 3, 4, 5],
            persist_src: &[4, 3],
            persist_dst: &[0, 1],
        };
        csa.pack(&prefill, &mut out).expect("the prefill packs");
        assert_eq!(out, [2, 2, 0, 1, 0, 2, 3, 4, 5, 0, 0, 4, 3, 0, 1]);

        let decode = CompGeom {
            ratio: 2,
            max_groups: 1,
            tokens: 1,
            rows: 256,
        };
        let mut out = vec![7u32; decode.words()];
        let odd = StepInts {
            write_row: &[150],
            read: &[0, 2],
            persist_src: &[0],
            persist_dst: &[1],
        };
        decode.pack(&odd, &mut out).expect("position 301 packs");
        assert_eq!(out, [1, 1, 150, 0, 2, 0, 0, 1, 0]);
        let even = StepInts {
            write_row: &[],
            read: &[],
            persist_src: &[0],
            persist_dst: &[0],
        };
        decode.pack(&even, &mut out).expect("position 4 packs");
        assert_eq!(out, [0, 1, 0, 0, 0, 0, 0, 0, 0]);

        let hca = CompGeom {
            ratio: 1,
            max_groups: 5,
            tokens: 5,
            rows: 512,
        };
        let mut out = vec![7u32; hca.words()];
        let rows = StepInts {
            write_row: &[0, 1, 2, 3, 4],
            read: &[1, 2, 3, 4, 5],
            persist_src: &[4],
            persist_dst: &[0],
        };
        hca.pack(&rows, &mut out)
            .expect("the ratio-1 prefill packs");
        assert_eq!(out, [5, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 0, 0]);

        let refused = [
            (
                csa,
                StepInts {
                    read: &[2, 3, 0, 5],
                    ..prefill
                },
                "a later group reads the ring",
            ),
            (
                csa,
                StepInts {
                    read: &[2, 3, 4, 7],
                    ..prefill
                },
                "a read past the batch",
            ),
            (
                csa,
                StepInts {
                    write_row: &[1, 0],
                    ..prefill
                },
                "rows out of order",
            ),
            (
                csa,
                StepInts {
                    write_row: &[0, 256],
                    ..prefill
                },
                "a row past the cache",
            ),
            (
                csa,
                StepInts {
                    persist_dst: &[1, 0],
                    ..prefill
                },
                "slots out of order",
            ),
            (
                csa,
                StepInts {
                    persist_src: &[4, 5],
                    ..prefill
                },
                "a kept token past the batch",
            ),
            (
                hca,
                StepInts {
                    read: &[0, 2, 3, 4, 5],
                    ..rows
                },
                "a ratio-1 read of a ring",
            ),
            (
                decode,
                StepInts {
                    write_row: &[1, 2],
                    read: &[0, 2, 2, 2],
                    ..odd
                },
                "two groups in one",
            ),
        ];
        for (geom, step, why) in refused {
            let mut out = vec![0u32; geom.words()];
            assert!(geom.pack(&step, &mut out).is_err(), "{why}");
        }
    }
}
