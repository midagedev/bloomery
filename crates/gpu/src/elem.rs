//! P4: every kernel of a decode step that is not a matmul, attention or the
//! router — the embedding row dequant, rms_norm, rope (the YaRN cos/sin cache
//! is computed on the host and uploaded; the kernel applies it), swiglu, the
//! residual add, the routed-expert weighted sum and argmax. One
//! `#[cuda_module]` in its own file (docs/gpu-design.md decision 6); the
//! arithmetic bodies are cores above the module so a later fused block kernel
//! can call the same bodies.
//!
//! Buffer layout, every op: ggml's token-major order — token `t`'s values are
//! contiguous, the token axis slowest. `rms_norm`/`swiglu`/`add` work on flat
//! `width * m` spans; `rope` on `[n_dims, n_vec, m]` (column `c = t*n_vec +
//! v`, the layout of the reference dump's `q_rope`/`k_rope` views);
//! `weighted_sum` on `[rows, n_exp, m]` down-projections with `[n_exp, m]`
//! weights; the embedding table is Q3_K rows of 2048 values (880 bytes = 220
//! u32 words each). Extents are launch arguments, never buffer lengths:
//! scratch buffers may be larger than the shape in flight.

use crate::GpuError;
use crate::cores::{funnel16, half_to_f32, q3k_aux_scales, q3k_sub_scale};
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

// ------------------------------------------------------------------ cores
//
// Same standing as `crate::cores`: ordinary `#[inline(always)]` functions a
// per-op wrapper and a later fused kernel can both call. Slice arguments are
// the whole device buffer plus indices — the verified core shape
// (`cores::q4k_a_chain` takes a buffer and a base).

/// The f32 value of Q3_K weight `v16` (0..256) of the super-block at byte
/// `base` of `w`, in the reference dequantizer's op order: `d_all *
/// (scale - 32)`, then that times `(qv - hv)` — plain multiplies, no fused
/// op — so a correct caller is bit-identical to `gguf::quant::dequant_row`
/// on the same row bytes. The 2-bit code sits in qs byte `32·half +
/// 16·half16 + l` at field `2·field`; the high bit is hmask byte `v16 % 32`
/// bit `4·half + field` (clear subtracts 4); the sub-block scale is the
/// aux-shuffle byte `v16/16` minus 32; `d` is the f16 at super-block bytes
/// 108..109.
///
/// Caller contract: `base + 110 <= 4 * w.len()` and `base` inside one row's
/// span (rows are 880 bytes, so `base` sits 0 or 2 mod 4 with the
/// super-block; the scale window funnels the 2-mod-4 case, single bytes load
/// from their covering words), `v16 < 256`.
#[inline(always)]
pub fn q3k_embed_value(w: &[u32], base: usize, v16: usize) -> f32 {
    let field = (v16 >> 5) & 3;
    let qs_byte = 32 * (v16 >> 7) + 16 * ((v16 >> 4) & 1) + (v16 & 15);
    // Single bytes load directly from their covering word — the value's qs
    // and hmask bytes sit at arbitrary byte offsets (the gemv reads whole
    // aligned quads and funnels; a lone byte needs no funnel).
    let qx = base + 32 + qs_byte;
    // SAFETY: qx < base + 95 < base + 110 <= 4 * w.len() by the caller
    // contract, so the covering word is inside w.
    let qsw = unsafe { *w.get_unchecked(qx >> 2) };
    let qv = (qsw >> (8 * (qx & 3) + 2 * field)) & 3;

    let hx = base + (v16 & 31);
    // SAFETY: hx < base + 32, inside the super-block by the caller contract.
    let hmw = unsafe { *w.get_unchecked(hx >> 2) };
    let hv = if (hmw >> (8 * (hx & 3) + 4 * (v16 >> 7) + field)) & 1 != 0 {
        0
    } else {
        4
    };

    let par = (base >> 1) & 1;

    let ak = (base + 96) >> 2;
    // SAFETY: the 12 scale bytes end at 108 and d at 110, inside the
    // super-block by the caller contract.
    let (aw0, aw1, aw2, aw3) = unsafe {
        (
            *w.get_unchecked(ak),
            *w.get_unchecked(ak + 1),
            *w.get_unchecked(ak + 2),
            *w.get_unchecked(ak + 3),
        )
    };
    let (a0w, a1w, a2w) = if par == 0 {
        (aw0, aw1, aw2)
    } else {
        (funnel16(aw0, aw1), funnel16(aw1, aw2), funnel16(aw2, aw3))
    };
    let ts = q3k_aux_scales(a0w, a1w, a2w);
    let sc = q3k_sub_scale(&ts, v16 >> 4);
    let d_bits = if par == 0 {
        (aw3 & 0xffff) as u16
    } else {
        (aw3 >> 16) as u16
    };
    (half_to_f32(d_bits) * sc as f32) * ((qv as i32 - hv) as f32)
}

/// The norm's scale from the summed squares: the mean divided in the sum's
/// own width, `eps` added inside the sqrt — the reference's form. The
/// reference sums the squares in f64 serially; the device's fixed f32
/// lane/butterfly tree moves last ulps only, which the gate's band owns.
#[inline(always)]
pub fn rms_scale(sum_sq: f32, k: u32, eps: f32) -> f32 {
    let mean = sum_sq / k as f32;
    1.0 / (mean + eps).sqrt()
}

/// Adjacent-pair rotation of one rope pair, the reference's op order (the
/// pairs are (2i, 2i+1), not NeoX split halves): `y0 = x0·cos − x1·sin`,
/// `y1 = x0·sin + x1·cos`, plain multiplies.
#[inline(always)]
pub fn rope_pair_core(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    (x0 * c - x1 * s, x0 * s + x1 * c)
}

/// `silu(gate) * up` in the reference's scalar op order: `g / (1 + e^(−g))`
/// then one multiply. The device `expf` and the host's differ in the last
/// ulps; the gate bands that distance and prints the measured max.
#[inline(always)]
pub fn silu_mul(g: f32, u: f32) -> f32 {
    g / (1.0 + (-g).exp()) * u
}

/// Token `t`'s output value `d` of the routed-expert combine:
/// `Σ_e w[t·n_exp + e] · down[(t·n_exp + e)·rows + d]`, experts ascending in
/// the buffer's expert axis, one plain multiply then add per term (no fused
/// multiply-add), so the sequence of adds is fixed.
///
/// Caller contract: `down.len() >= rows * n_exp * m`, `w.len() >= n_exp * m`,
/// `t < m`, `d < rows`.
#[inline(always)]
pub fn weighted_expert_sum(
    down: &[f32],
    w: &[f32],
    rows: u32,
    n_exp: u32,
    t: usize,
    d: usize,
) -> f32 {
    let rows = rows as usize;
    let n_exp = n_exp as usize;
    let mut acc = 0.0f32;
    let mut e = 0usize;
    while e < n_exp {
        // SAFETY: e < n_exp and t < m bound the weight index t*n_exp + e and
        // the down index (t*n_exp + e)*rows + d inside their buffers by the
        // caller contract.
        let wv = unsafe { *w.get_unchecked(t * n_exp + e) };
        let dv = unsafe { *down.get_unchecked((t * n_exp + e) * rows + d) };
        acc += wv * dv;
        e += 1;
    }
    acc
}

/// Whether candidate `(v, i)` beats the running best: strictly greater
/// value, or an equal value at a lower index — the greedy sampler's tie
/// rule. A total order on (value, index), so any fixed reduction tree over
/// it is deterministic.
#[inline(always)]
pub fn argmax_take(v: f32, i: u32, best_v: f32, best_i: u32) -> bool {
    v > best_v || (v == best_v && i < best_i)
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod elem_kernels {
    use super::*;

    /// Dequantize `ids.len()` rows of the Q3_K embedding table `w` (220 u32
    /// words per row, 2048 values) into `y`, token-major. Ids live on the
    /// device, so their validity cannot be host-checked: an id past the
    /// table's `n_rows` rows reads row 0 — deterministic and in-bounds, never
    /// an out-of-bounds read — and the step's id source is the argmax, below
    /// the vocabulary by construction.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (4 * w.len() >= 880 * n_rows, y.len() >= 2048 * ids.len())
    )]
    pub fn embed_rows(w: &[u32], ids: &[u32], n_rows: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= ids.len() * 2048 {
            return;
        }
        let t = i >> 11;
        let k = i & 2047;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let id = (if id < n_rows { id } else { 0 }) as usize;
        // Row spans are whole 880-byte blocks, so only the super-block offset
        // can sit 2 mod 4; the core funnels both alignments.
        let v = q3k_embed_value(w, id * 880 + ((k >> 8) * 110), k & 255);
        // SAFETY: i < ids.len() * 2048 <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// RMS norm, one warp per token: lane partial sums of squares over
    /// values `32·it + lane` (it ascending), the fixed five-step butterfly,
    /// then `(scale · gain) · x` per value in the reference's order. `k` a
    /// positive multiple of 32 (host-checked); the row guard is
    /// warp-uniform, so the butterfly always sees a full warp.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= k * m, gain.len() >= k, y.len() >= k * m)
    )]
    pub fn rms_norm(x: &[f32], gain: &[f32], eps: f32, k: u32, m: u32, mut y: DisjointSlice<f32>) {
        let t = thread::index_1d().get() / 32;
        if t >= m as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let k = k as usize;
        let base = t * k;
        let mut acc = 0.0f32;
        let mut it = lane;
        while it < k {
            // SAFETY: it < k <= x.len() - base by the launch contract.
            let v = unsafe { *x.get_unchecked(base + it) };
            acc += v * v;
            it += 32;
        }
        let scale = rms_scale(warp::reduce_sum_f32(acc), k as u32, eps);
        let mut it = lane;
        while it < k {
            // SAFETY: it < k bounds the gain read by the contract and the x
            // read as above; base + it < k*m <= y.len() by the contract.
            let g = unsafe { *gain.get_unchecked(it) };
            let v = unsafe { *x.get_unchecked(base + it) };
            // SAFETY: the store index equals the load index, inside y by the
            // contract.
            unsafe {
                *y.get_unchecked_mut(base + it) = (scale * g) * v;
            }
            it += 32;
        }
    }

    /// Apply the host-computed rope cos/sin cache: one thread per (column,
    /// pair). Column `c = t*n_vec + v` covers `nd` values; token `t`'s cache
    /// is `nd` f32 at `cs[t*nd ..]`, interleaved `[cos0, sin0, …]`. `nd`
    /// even (host-checked).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            src.len() >= m * n_vec * nd,
            cs.len() >= m * nd,
            dst.len() >= m * n_vec * nd
        )
    )]
    pub fn rope(src: &[f32], cs: &[f32], nd: u32, n_vec: u32, m: u32, mut dst: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        let npairs = (nd >> 1) as usize;
        let total = m as usize * n_vec as usize * npairs;
        if i >= total {
            return;
        }
        let col = i / npairs;
        let d = (i % npairs) * 2;
        let nd = nd as usize;
        let t = col / n_vec as usize;
        // SAFETY: d + 1 <= nd - 1, so both src indices stay below
        // m*n_vec*nd <= src.len() and both cache indices below m*nd <=
        // cs.len() by the launch contract.
        let (x0, x1, c, s) = unsafe {
            (
                *src.get_unchecked(col * nd + d),
                *src.get_unchecked(col * nd + d + 1),
                *cs.get_unchecked(t * nd + d),
                *cs.get_unchecked(t * nd + d + 1),
            )
        };
        let (y0, y1) = rope_pair_core(x0, x1, c, s);
        // SAFETY: the same indices as the loads, inside dst by the contract.
        unsafe {
            *dst.get_unchecked_mut(col * nd + d) = y0;
            *dst.get_unchecked_mut(col * nd + d + 1) = y1;
        }
    }

    /// `y[i] = silu(gate[i]) * up[i]`, elementwise over `n` values.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (gate.len() >= n, up.len() >= n, y.len() >= n)
    )]
    pub fn swiglu(gate: &[f32], up: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n bounds both loads by the launch contract.
        let (g, u) = unsafe { (*gate.get_unchecked(i), *up.get_unchecked(i)) };
        // SAFETY: i < n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = silu_mul(g, u);
        }
    }

    /// `y[i] = a[i] + b[i]`, elementwise over `n` values — exact; the body
    /// is one add, so nothing is extracted into a core.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= n, b.len() >= n, y.len() >= n)
    )]
    pub fn add(a: &[f32], b: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n bounds both loads by the launch contract.
        let (av, bv) = unsafe { (*a.get_unchecked(i), *b.get_unchecked(i)) };
        // SAFETY: i < n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = av + bv;
        }
    }

    /// The routed-expert combine, one thread per (token, output value):
    /// `y[t*rows + d] = Σ_e w[t*n_exp + e] · down[(t*n_exp + e)*rows + d]`,
    /// token-major `[rows, n_exp, m]` down-projections with `[n_exp, m]`
    /// weights.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= rows * n_exp * m,
            w.len() >= n_exp * m,
            y.len() >= rows * m
        )
    )]
    pub fn weighted_sum(
        down: &[f32],
        w: &[f32],
        rows: u32,
        n_exp: u32,
        m: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= rows as usize * m as usize {
            return;
        }
        let t = i / rows as usize;
        let d = i % rows as usize;
        let v = weighted_expert_sum(down, w, rows, n_exp, t, d);
        // SAFETY: i < rows*m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// Index of the maximum of `n` f32, ties to the lower index — the greedy
    /// sampler's rule. One warp over the whole row: each lane's best over
    /// its strided values (indices ascending within a lane), then the fixed
    /// xor butterfly merges by (value desc, index asc). Both stages are a
    /// total order, so the result is a function of the input alone. The
    /// result stays on the device (`out[0]`, u32); the step reads it back
    /// once. Inputs are finite — the gate's loader rejects anything else.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (x.len() >= n, out.len() >= 1))]
    pub fn argmax(x: &[f32], n: u32, mut out: DisjointSlice<u32>) {
        let lane = warp::lane_id();
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0u32;
        let mut i = lane;
        while i < n {
            // SAFETY: i < n <= x.len() by the launch contract.
            let v = unsafe { *x.get_unchecked(i as usize) };
            if argmax_take(v, i, best_v, best_i) {
                best_v = v;
                best_i = i;
            }
            i += 32;
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (
                warp::shuffle_xor_f32(best_v, off),
                warp::shuffle_xor(best_i, off),
            );
            if argmax_take(ov, oi, best_v, best_i) {
                best_v = ov;
                best_i = oi;
            }
            off >>= 1;
        }
        if lane == 0 {
            // SAFETY: out.len() >= 1 by the launch contract; only lane 0
            // writes.
            unsafe {
                *out.get_unchecked_mut(0) = best_i;
            }
        }
    }
}

/// The loaded P4 device module. Owns no context and no stream — the caller
/// passes the engine stream (`Gpu::stream()`) per enqueue, so launches order
/// with the rest of the step and are capturable.
pub struct ElemKernels {
    module: elem_kernels::LoadedModule,
}

impl ElemKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ElemKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { elem_kernels::load(ctx)? };
        Ok(ElemKernels { module })
    }

    /// Enqueue the embedding lookup: `ids` (device-resident token ids, every
    /// id below the table's row count) each select one Q3_K row of `w` (220
    /// u32 words per row), dequantized to 2048 f32. `y` holds `2048 *
    /// ids.len()` f32, token-major. Bit-identical to
    /// `gguf::quant::dequant_row` on the same row bytes. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_embed_rows(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        ids: &DeviceBuffer<u32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if w.cols() != 220 || w.rows() == 0 {
            return Err(format!(
                "enqueue_embed_rows: Q3_K table is 220 words (880 bytes) per row, got {}x{}",
                w.rows(),
                w.cols()
            )
            .into());
        }
        if ids.len() == 0 {
            return Err("enqueue_embed_rows: empty ids".into());
        }
        if y.len() < 2048 * ids.len() {
            return Err(format!(
                "enqueue_embed_rows: y.len() {} < 2048*{}",
                y.len(),
                ids.len()
            )
            .into());
        }
        let prep = self.module.prepare_embed_rows(LaunchConfig1D::new(
            (ids.len() * 2048).div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module
            .embed_rows(stream, &prep, w.buf(), ids, w.rows() as u32, y)?;
        Ok(())
    }

    /// Enqueue `y = rms_norm(x, gain, eps)` over `m` tokens of `k` values
    /// (token-major). `k` a positive multiple of 32; `x`, `y` hold `k * m`
    /// f32, `gain` holds `k`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_rms_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        k: usize,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if k == 0 || k % 32 != 0 {
            return Err(
                format!("enqueue_rms_norm: k must be a positive multiple of 32, got {k}").into(),
            );
        }
        if m == 0 || x.len() < k * m || gain.len() < k || y.len() < k * m {
            return Err(format!(
                "enqueue_rms_norm: m={m}, x.len() {} (need {}), gain.len() {} (need {k}), y.len() {} (need {})",
                x.len(),
                k * m,
                gain.len(),
                y.len(),
                k * m
            )
            .into());
        }
        let prep =
            self.module
                .prepare_rms_norm(LaunchConfig1D::new(m.div_ceil(8) as u32, 256, 0))?;
        self.module
            .rms_norm(stream, &prep, x, gain, eps, k as u32, m as u32, y)?;
        Ok(())
    }

    /// Enqueue the rope rotation of `src` (`m * n_vec * nd` f32, column
    /// `c = t*n_vec + v` covering `nd` values) by the cos/sin caches in `cs`
    /// (`m * nd` f32, token `t`'s interleaved `[cos0, sin0, …]` at
    /// `t*nd` — the host YaRN cache, uploaded per step). `nd` even and at
    /// least 2. `dst` holds `m * n_vec * nd` f32 in `src`'s layout.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_rope(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        cs: &DeviceBuffer<f32>,
        n_dims: usize,
        n_vec: u32,
        m: usize,
        dst: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n_dims < 2 || n_dims % 2 != 0 {
            return Err(format!("enqueue_rope: n_dims must be even and >= 2, got {n_dims}").into());
        }
        if n_vec == 0 || m == 0 {
            return Err(format!(
                "enqueue_rope: need n_vec >= 1 and m >= 1, got n_vec={n_vec} m={m}"
            )
            .into());
        }
        let span = m * n_vec as usize * n_dims;
        if src.len() < span || cs.len() < m * n_dims || dst.len() < span {
            return Err(format!(
                "enqueue_rope: src.len() {} / cs.len() {} / dst.len() {} vs span {span}, cache {}",
                src.len(),
                cs.len(),
                dst.len(),
                m * n_dims
            )
            .into());
        }
        let threads = m * n_vec as usize * (n_dims / 2);
        let prep =
            self.module
                .prepare_rope(LaunchConfig1D::new(threads.div_ceil(256) as u32, 256, 0))?;
        self.module
            .rope(stream, &prep, src, cs, n_dims as u32, n_vec, m as u32, dst)?;
        Ok(())
    }

    /// Enqueue `y = silu(gate) * up` over `n` values (flat; token-major
    /// spans of any width). Asynchronous, allocation-free, capturable.
    pub fn enqueue_swiglu(
        &self,
        stream: &CudaStream,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || gate.len() < n || up.len() < n || y.len() < n {
            return Err(format!(
                "enqueue_swiglu: n={n}, gate.len() {}, up.len() {}, y.len() {}",
                gate.len(),
                up.len(),
                y.len()
            )
            .into());
        }
        let prep =
            self.module
                .prepare_swiglu(LaunchConfig1D::new(n.div_ceil(256) as u32, 256, 0))?;
        self.module.swiglu(stream, &prep, gate, up, n as u32, y)?;
        Ok(())
    }

    /// Enqueue `y = a + b` over `n` values (flat). Exact. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_add(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || a.len() < n || b.len() < n || y.len() < n {
            return Err(format!(
                "enqueue_add: n={n}, a.len() {}, b.len() {}, y.len() {}",
                a.len(),
                b.len(),
                y.len()
            )
            .into());
        }
        let prep = self
            .module
            .prepare_add(LaunchConfig1D::new(n.div_ceil(256) as u32, 256, 0))?;
        self.module.add(stream, &prep, a, b, n as u32, y)?;
        Ok(())
    }

    /// Enqueue the routed-expert combine: `down` holds `m` tokens' stacks of
    /// `n_exp` expert outputs of `rows` values (token-major `[rows, n_exp,
    /// m]`), `w` the router weights (`n_exp * m` f32, `[n_exp, m]`); `y`
    /// holds `rows * m` f32, token-major. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_weighted_sum(
        &self,
        stream: &CudaStream,
        down: &DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        rows: usize,
        n_exp: u32,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if rows == 0 || n_exp == 0 || m == 0 {
            return Err(format!(
                "enqueue_weighted_sum: need rows/n_exp/m >= 1, got {rows}/{n_exp}/{m}"
            )
            .into());
        }
        if down.len() < rows * n_exp as usize * m
            || w.len() < n_exp as usize * m
            || y.len() < rows * m
        {
            return Err(format!(
                "enqueue_weighted_sum: down.len() {} (need {}), w.len() {} (need {}), y.len() {} (need {})",
                down.len(),
                rows * n_exp as usize * m,
                w.len(),
                n_exp as usize * m,
                y.len(),
                rows * m
            )
            .into());
        }
        let prep = self.module.prepare_weighted_sum(LaunchConfig1D::new(
            (rows * m).div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module
            .weighted_sum(stream, &prep, down, w, rows as u32, n_exp, m as u32, y)?;
        Ok(())
    }

    /// Enqueue the argmax of `n` f32 into `out[0]` (u32, device-resident;
    /// ties to the lower index). One warp walks the whole row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_argmax(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        if n == 0 || x.len() < n || out.len() < 1 {
            return Err(format!(
                "enqueue_argmax: n={n}, x.len() {}, out.len() {}",
                x.len(),
                out.len()
            )
            .into());
        }
        let prep = self.module.prepare_argmax(LaunchConfig1D::new(1, 32, 0))?;
        self.module.argmax(stream, &prep, x, n as u32, out)?;
        Ok(())
    }
}
