//! `GpuModel` — the GPU engine that stands beside `forward::step`
//! (docs/gpu-design.md decisions 1 and 7). Weights, KV and scratch live on
//! the device from `load`, split into stages by layer range; `step` enqueues
//! one decode step stage by stage and synchronizes once for the argmax.
//!
//! The chain is layer-indexed at m = 1: an attention half shared by every
//! layer (MLA over the layer's own KV cache) followed by one of two FFN
//! halves — the fused dense FFN for a layer without a router, the routed
//! MoE half (router → six experts through `sel` → shared expert → combine)
//! for a layer with one. Block 0 additionally embeds its token in front.
//! Every per-replay quantity (position, live key count, token id, rope
//! cos/sin cache) lives in a device buffer refreshed before the
//! enqueue/replay, so one captured graph serves every position, and the
//! routed expert ids live in a device buffer the expert kernels read per
//! launch, so one captured graph serves every routing.
//! Stitching the layers of a stage into one [`GpuModel::step`] is another
//! round; that entry names what is missing.
//!
//! The op set of P1–P7 has no f32 concat or strided per-head addressing
//! (both live in the CPU chain: the `kvr`/flash-row concats and the
//! per-head `wv_b`/q_nope2 legs of `attn.rs`), so this file carries one
//! small `#[cuda_module]` of its own — decision 6's per-file module rule:
//! a pair-table gather and two per-head gemv wrappers. The wrappers give
//! the q_nope2 (derived Q8_0) and wv_b (Q3_K) sites one launch each: one
//! warp per output row, the row's head selecting both the activation
//! slice it reads (an x base or a column of the quantized activation)
//! and the output slot it writes, m = 1 through the gated row bodies
//! (`q8f32::q8_0_lane_partials`, `cores::q3k_row_dot`) — so every dot
//! equals the plain gemv's on the same row and column bit for bit.
//! Everything else runs the gated kernels verbatim.

use crate::q5::Q8Blocks32;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, Graph};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use model::attn::MlaParams;
use std::ops::Range;
use std::sync::Arc;

/// Compile-time probe of the dependency direction (gpu → model → gguf): the
/// device-bundle crate reads model metadata through `bloomery-model`.
pub fn mla_width(gguf: &gguf::Gguf) -> Result<usize, GpuError> {
    let p = MlaParams::read(gguf, 0)?;
    Ok(p.rope_dims + p.latent)
}

// ------------------------------------------------------------- step kernels

#[cuda_module]
mod step_kernels {
    use super::*;
    use crate::cores::q3k_row_dot;
    use crate::q8f32::q8_0_lane_partials;
    use cuda_device::warp;

    /// `y[dst_idx[i]] = x[src_idx[i]]` for `i < n` — the f32 concat/extract
    /// the step needs and no P4 op provides. Both tables are load-time
    /// constants built by the host side from the buffer shapes it owns, so a
    /// pair is always in range; the guard keeps a wrong table a deterministic
    /// no-op rather than an out-of-bounds access (the policy `embed_rows`
    /// takes for its device-resident ids). Destinations must not overlap.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= 1,
            src_idx.len() >= n,
            dst_idx.len() >= n,
            y.len() >= 1
        )
    )]
    pub fn gather_pairs(
        x: &[f32],
        src_idx: &[u32],
        dst_idx: &[u32],
        n: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= src_idx.len() by the launch contract.
        let s = unsafe { *src_idx.get_unchecked(i) } as usize;
        // SAFETY: i < n <= dst_idx.len() by the launch contract.
        let d = unsafe { *dst_idx.get_unchecked(i) } as usize;
        if s < x.len() && d < y.len() {
            // SAFETY: both indices guarded against their buffers' lengths.
            unsafe {
                *y.get_unchecked_mut(d) = *x.get_unchecked(s);
            }
        }
    }

    /// Q8_0 gemv over per-head activation slices: launch row `r` belongs to
    /// head `h = r / rows_per_head` and output slot `r % rows_per_head` of
    /// that head; it dots weight row `r` of the derived planes against
    /// `x[h*x_head_stride .. +k]` (m = 1 through the gated row body, the
    /// `q8_0_gemv` skeleton with one warp per row, 8 rows per 256-thread
    /// block), lane 0 storing `y[h*y_head_stride + y_off + r %
    /// rows_per_head]`. `n_rows` must be `n_heads * rows_per_head`; every
    /// store is disjoint because each (head, slot) pair belongs to exactly
    /// one row.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= (n_heads - 1) * x_head_stride + k,
            y.len() >= (n_heads - 1) * y_head_stride + y_off + rows_per_head
        )
    )]
    pub fn q8_0_gemv_heads(
        qs: &[u32],
        d: &[f32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        n_heads: u32,
        rows_per_head: u32,
        x_head_stride: u32,
        y_head_stride: u32,
        y_off: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let h = row / rows_per_head as usize;
        // The launch contract cannot bind n_rows to n_heads*rows_per_head
        // (no division in its grammar); this guard keeps a violated
        // divisibility from writing outside the last head's y span.
        if h >= n_heads as usize {
            return;
        }
        let j = row % rows_per_head as usize;
        let lane = warp::lane_id() as usize;
        // SAFETY: the launch contract bounds the row's x window the way
        // `q8_0_lane_partials` demands: x0 = h*x_head_stride <=
        // (n_heads-1)*x_head_stride and x.len() >= x0 + k.
        let f = q8_0_lane_partials(qs, d, x, k, row, h * x_head_stride as usize, 1, lane);
        let s0 = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes; the slot
            // h*y_head_stride + y_off + j is inside y by the launch contract
            // and belongs to this row alone.
            unsafe {
                *y.get_unchecked_mut(h * y_head_stride as usize + y_off as usize + j) = s0;
            }
        }
    }

    /// Q3_K gemv over per-head activation columns: launch row `r` belongs
    /// to head `h = head_base + r / rows_per_head`, dots absolute weight
    /// row `h * row_stride_per_head + row_off + r % rows_per_head` against
    /// activation column `r / rows_per_head` of `q`/`d8` (m = 1 through
    /// the gated row body, one warp per row, 8 rows per 256-thread block),
    /// lane 0 storing `y[h * y_head_stride + r % rows_per_head]`.
    /// `n_rows` must be `heads * rows_per_head` and
    /// `row_off + rows_per_head <= row_stride_per_head` (host-validated),
    /// which keeps every `row_abs` below
    /// `(head_base + heads) * row_stride_per_head` — the bound the launch
    /// contract puts on `w`. Stores are disjoint: each (head, slot) pair
    /// belongs to exactly one row.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= (head_base + heads) * row_stride_per_head * 110 * n_sb,
            q.len() >= heads * 64 * iters,
            d8.len() >= heads * 2 * n_sb,
            y.len() >= (head_base + heads - 1) * y_head_stride + rows_per_head
        )
    )]
    pub fn q3k_gemv_heads(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        head_base: u32,
        heads: u32,
        rows_per_head: u32,
        row_stride_per_head: u32,
        row_off: u32,
        y_head_stride: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let hi = row / rows_per_head as usize;
        // The launch contract cannot bind n_rows to heads*rows_per_head
        // (no division in its grammar); this guard keeps a violated
        // divisibility from reading weight rows past the last head's
        // block.
        if hi >= heads as usize {
            return;
        }
        let row_in = row % rows_per_head as usize;
        let h = head_base as usize + hi;
        let row_abs = h * row_stride_per_head as usize + row_off as usize + row_in;
        let lane = warp::lane_id() as usize;
        // SAFETY: row_abs + 1 <= (head_base + heads) * row_stride_per_head
        // (the contract's bound on w, given the host-validated
        // row_off + rows_per_head <= row_stride_per_head); column hi <
        // heads keeps q/d8 inside their contract bounds.
        let f = q3k_row_dot(w, q, d8, n_sb as usize, iters, row_abs, hi, 1, lane);
        let s0 = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes; the slot
            // h*y_head_stride + row_in is inside y by the launch contract
            // and belongs to this row alone.
            unsafe {
                *y.get_unchecked_mut(h * y_head_stride as usize + row_in) = s0;
            }
        }
    }
}

/// The loaded step module of this file. Owns no context and no stream —
/// every enqueue takes the engine stream, so it orders with the rest of the
/// step and is capturable.
pub struct StepKernels {
    module: step_kernels::LoadedModule,
}

impl StepKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<StepKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { step_kernels::load(ctx)? };
        Ok(StepKernels { module })
    }

    /// Enqueue the pair-table gather of `n` pairs. Asynchronous,
    /// allocation-free, capturable. The tables must be built for the shapes
    /// of `x` and `y` this call pairs (load-time construction owns that).
    pub fn enqueue_gather(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        src_idx: &DeviceBuffer<u32>,
        dst_idx: &DeviceBuffer<u32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || src_idx.len() < n || dst_idx.len() < n || x.is_empty() || y.is_empty() {
            return Err(format!(
                "enqueue_gather: n={n}, src_idx.len() {}, dst_idx.len() {}, x.len() {}, \
                 y.len() {}",
                src_idx.len(),
                dst_idx.len(),
                x.len(),
                y.len()
            )
            .into());
        }
        let prep = self.module.prepare_gather_pairs(LaunchConfig1D::new(
            n.div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module
            .gather_pairs(stream, &prep, x, src_idx, dst_idx, n as u32, y)?;
        Ok(())
    }

    /// Enqueue the per-head Q8_0 gemv: `d.rows()` weight rows (the derived
    /// planes, `qs`/`d` as `enqueue_q8_0_gemv` takes them, k =
    /// `d.cols() * 32`), each dotted against its head's slice of `x` —
    /// head `h` reads `x[h*x_head_stride .. +k]`, m = 1 — with lane 0
    /// writing `y[h*y_head_stride + y_off + j]` for weight row
    /// `h*rows_per_head + j`. `d.rows()` must be a multiple of
    /// `rows_per_head` (then `n_heads = d.rows() / rows_per_head`). Asynchronous,
    /// allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_q8_0_gemv_heads(
        &self,
        stream: &CudaStream,
        qs: &DeviceTensor<u32>,
        d: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        rows_per_head: usize,
        x_head_stride: usize,
        y_head_stride: usize,
        y_off: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_rows = d.rows();
        let k = d.cols() * 32;
        if k == 0 || !k.is_multiple_of(32) {
            return Err(format!(
                "enqueue_q8_0_gemv_heads: need k a positive multiple of 32, got k={k} \
                 (d is {}x{})",
                d.rows(),
                d.cols()
            )
            .into());
        }
        if qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
            return Err(format!(
                "enqueue_q8_0_gemv_heads: qs is {}x{}, want {}x{} (k/4 words per row, k = \
                 d.cols()*32 = {k})",
                qs.rows(),
                qs.cols(),
                n_rows,
                d.cols() * 8
            )
            .into());
        }
        if rows_per_head == 0 || n_rows % rows_per_head != 0 {
            return Err(format!(
                "enqueue_q8_0_gemv_heads: n_rows={n_rows} is not a positive multiple of \
                 rows_per_head={rows_per_head}"
            )
            .into());
        }
        let n_heads = n_rows / rows_per_head;
        if x.len() < (n_heads - 1) * x_head_stride + k {
            return Err(format!(
                "enqueue_q8_0_gemv_heads: x.len() {} < (n_heads-1)*x_head_stride + k = \
                 {}*{x_head_stride} + {k}",
                x.len(),
                n_heads - 1
            )
            .into());
        }
        if y.len() < (n_heads - 1) * y_head_stride + y_off + rows_per_head {
            return Err(format!(
                "enqueue_q8_0_gemv_heads: y.len() {} < (n_heads-1)*y_head_stride + y_off + \
                 rows_per_head = {}*{y_head_stride} + {y_off} + {rows_per_head}",
                y.len(),
                n_heads - 1
            )
            .into());
        }
        let prep = self.module.prepare_q8_0_gemv_heads(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.q8_0_gemv_heads(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            n_rows as u32,
            k as u32,
            n_heads as u32,
            rows_per_head as u32,
            x_head_stride as u32,
            y_head_stride as u32,
            y_off as u32,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the per-head Q3_K gemv: `heads` heads starting at
    /// `head_base` (of the `w.rows()`-row weight, `row_stride_per_head`
    /// rows per head), each head's `rows_per_head` output rows dotting
    /// absolute weight row `h*row_stride_per_head + row_off + j` against
    /// activation column `h - head_base` of `act` (m = 1), lane 0 writing
    /// `y[h*y_head_stride + j]`. `heads` must be `act.m()` (one column per
    /// head), `w.cols()` `110 * n_sb / 4` words with even `n_sb` (as
    /// `enqueue_gemv_q3k`), and `row_off + rows_per_head <=
    /// row_stride_per_head` so a head's rows stay on its own block.
    /// Asynchronous, allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_q3k_gemv_heads(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        head_base: usize,
        rows_per_head: usize,
        row_stride_per_head: usize,
        row_off: usize,
        y_head_stride: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_sb = act.n_sb();
        let heads = act.m();
        if !n_sb.is_multiple_of(2) {
            return Err(format!(
                "enqueue_q3k_gemv_heads: odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                act.k()
            )
            .into());
        }
        if w.cols() != 110 * n_sb / 4 {
            return Err(format!(
                "enqueue_q3k_gemv_heads: Q3_K rows are 110*{n_sb}/4 = {} words at K={}, got {}",
                110 * n_sb / 4,
                act.k(),
                w.cols()
            )
            .into());
        }
        if row_off + rows_per_head > row_stride_per_head || rows_per_head == 0 {
            return Err(format!(
                "enqueue_q3k_gemv_heads: row_off {row_off} + rows_per_head {rows_per_head} \
                 must lie inside row_stride_per_head {row_stride_per_head}"
            )
            .into());
        }
        if (head_base + heads) * row_stride_per_head > w.rows() {
            return Err(format!(
                "enqueue_q3k_gemv_heads: heads {head_base}..{} need (head_base+heads)*\
                 row_stride_per_head = {} rows, w has {}",
                head_base + heads,
                (head_base + heads) * row_stride_per_head,
                w.rows()
            )
            .into());
        }
        let n_rows = heads * rows_per_head;
        if y.len() < (head_base + heads - 1) * y_head_stride + rows_per_head {
            return Err(format!(
                "enqueue_q3k_gemv_heads: y.len() {} < (head_base+heads-1)*y_head_stride + \
                 rows_per_head = {}*{y_head_stride} + {rows_per_head}",
                y.len(),
                head_base + heads - 1
            )
            .into());
        }
        let prep = self.module.prepare_q3k_gemv_heads(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.q3k_gemv_heads(
            stream,
            &prep,
            w.buf(),
            &act.q3,
            &act.d8,
            n_rows as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            head_base as u32,
            heads as u32,
            rows_per_head as u32,
            row_stride_per_head as u32,
            row_off as u32,
            y_head_stride as u32,
            y,
        )?;
        Ok(())
    }
}

// ------------------------------------------------------------------ scratch

/// The shapes the block-0 scratch is sized for, derived once from the
/// resident weights and `MlaParams` — never literals.
struct Dims {
    hidden: usize,
    /// `q_rows / rope_dims` — the 64-value columns `enqueue_rope` walks over
    /// the q projection.
    q_cols: usize,
    /// `kv_width / rope_dims` — same for the kv_a projection.
    kv_cols: usize,
}

/// One device index table for the gather (source and destination indices of
/// one permutation).
struct Gather {
    src: DeviceBuffer<u32>,
    dst: DeviceBuffer<u32>,
    n: usize,
}

impl Gather {
    /// Build the table for `pairs: (src, dst)`. Load-time only.
    fn new(
        stream: &CudaStream,
        pairs: impl Iterator<Item = (usize, usize)>,
        what: &str,
    ) -> Result<Gather, GpuError> {
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for (s, d) in pairs {
            if s > u32::MAX as usize || d > u32::MAX as usize {
                return Err(format!("Gather::new {what}: index overflows u32").into());
            }
            src.push(s as u32);
            dst.push(d as u32);
        }
        if src.is_empty() {
            return Err(format!("Gather::new {what}: empty table").into());
        }
        Ok(Gather {
            n: src.len(),
            src: DeviceBuffer::from_host(stream, &src)?,
            dst: DeviceBuffer::from_host(stream, &dst)?,
        })
    }
}

/// The weight names one layer's chain looks up, built once at load so the
/// step never formats a string (decision 4). `routed` is read from the
/// file, not from the layer index: a layer is MoE exactly when its router
/// weight is resident.
struct LayerNames {
    layer: usize,
    attn_norm: String,
    attn_q: String,
    attn_kv_a_mqa: String,
    attn_kv_a_norm: String,
    attn_kv_b: String,
    attn_output: String,
    ffn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
    ffn_gate_inp: String,
    ffn_gate_exps: String,
    ffn_up_exps: String,
    ffn_down_exps: String,
    ffn_gate_shexp: String,
    ffn_up_shexp: String,
    ffn_down_shexp: String,
    routed: bool,
}

impl LayerNames {
    /// Every name of block `l`, and whether that block routes.
    fn new(w: &Weights, l: usize) -> LayerNames {
        let n = |stem: &str| format!("blk.{l}.{stem}.weight");
        let ffn_gate_inp = n("ffn_gate_inp");
        LayerNames {
            layer: l,
            routed: w.get(&ffn_gate_inp).is_some(),
            attn_norm: n("attn_norm"),
            attn_q: n("attn_q"),
            attn_kv_a_mqa: n("attn_kv_a_mqa"),
            attn_kv_a_norm: n("attn_kv_a_norm"),
            attn_kv_b: n("attn_kv_b"),
            attn_output: n("attn_output"),
            ffn_norm: n("ffn_norm"),
            ffn_gate: n("ffn_gate"),
            ffn_up: n("ffn_up"),
            ffn_down: n("ffn_down"),
            ffn_gate_inp,
            ffn_gate_exps: n("ffn_gate_exps"),
            ffn_up_exps: n("ffn_up_exps"),
            ffn_down_exps: n("ffn_down_exps"),
            ffn_gate_shexp: n("ffn_gate_shexp"),
            ffn_up_shexp: n("ffn_up_shexp"),
            ffn_down_shexp: n("ffn_down_shexp"),
        }
    }
}

/// The MoE shapes of this model, read from the file's metadata and from the
/// resident expert stacks — never literals. Present only when the stage
/// holds a routed layer.
#[derive(Clone)]
struct MoeDims {
    n_expert: usize,
    n_used: usize,
    /// `expert_feed_forward_length` — one routed expert's hidden width.
    ff: usize,
    /// `expert_weights_scale`, applied to every router weight.
    scale: f32,
    /// The shared expert's hidden width (its gate/up row count).
    shexp_ff: usize,
}

/// The routed FFN half's arena, sized at load from [`MoeDims`].
struct MoeScratch {
    /// `ffn_norm(ffn_inp)` as f32 — the router's input and the block's
    /// `ffn_norm` tap. The experts read the q8_1 form in `act_ffn`.
    normed: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    /// The router's expert ids — at m = 1 this buffer IS the `sel` the
    /// expert kernels read, so a captured graph follows the routing.
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    h_exp: DeviceBuffer<f32>,
    act32_exp: Q8Blocks32,
    down: DeviceBuffer<f32>,
    h_sh: DeviceBuffer<f32>,
    act_sh: Q8Act,
    shexp: DeviceBuffer<f32>,
}

impl MoeScratch {
    fn bytes(&self) -> usize {
        let mut total = [
            &self.normed,
            &self.logits,
            &self.probs,
            &self.weights,
            &self.h_exp,
            &self.down,
            &self.h_sh,
            &self.shexp,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        total += self.ids.num_bytes();
        total += self.act32_exp.q.num_bytes()
            + self.act32_exp.s8.num_bytes()
            + self.act32_exp.d8.num_bytes();
        total += self.act_sh.q3.num_bytes()
            + self.act_sh.q4.num_bytes()
            + self.act_sh.q6.num_bytes()
            + self.act_sh.s8.num_bytes()
            + self.act_sh.d8.num_bytes();
        total
    }
}

/// The dense FFN half's arena: the swiglu intermediate and its 32-value
/// quantization.
struct DenseScratch {
    h: DeviceBuffer<f32>,
    act32: Q8Blocks32,
}

impl DenseScratch {
    fn bytes(&self) -> usize {
        self.h.num_bytes()
            + self.act32.q.num_bytes()
            + self.act32.s8.num_bytes()
            + self.act32.d8.num_bytes()
    }
}

/// The layer scratch arena and the device-side step parameters, sized at
/// load for m = 1 (decision 4: nothing here is allocated per step). One
/// arena serves every layer of a stage — layers run sequentially, so the
/// intermediates are reused; only the KV caches are per layer.
struct LayerScratch {
    dims: Dims,
    x: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    act_q: Q8Act,
    q: DeviceBuffer<f32>,
    q_rope_all: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    /// `[kv_compressed | k_rope]`: rope's last column writes the tail, then
    /// the kv norm overwrites the latent head.
    kv_s: DeviceBuffer<f32>,
    /// `[k_rope | kv_compressed]` — the row the KV append consumes.
    kvr: DeviceBuffer<f32>,
    /// Flash query rows, `[q_rope | q_nope2]` per head.
    f_rows: DeviceBuffer<f32>,
    kqvc: DeviceBuffer<f32>,
    act_kv_lo: Q8Act,
    act_kv_hi: Q8Act,
    kqv_2d: DeviceBuffer<f32>,
    act_ao: Q8Act,
    attn_out: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    /// The q8_1 form of the FFN-normed vector — read by the dense gate/up,
    /// by the routed experts and by the shared expert alike.
    act_ffn: Q8Act,
    /// The dense half's arena — present iff the stage holds a dense layer.
    dense: Option<DenseScratch>,
    /// The routed half's arena — present iff the stage holds a routed layer.
    moe: Option<MoeScratch>,
    l_out: DeviceBuffer<f32>,
    g_kvr: Gather,
    g_f_rope_lo: Gather,
    g_f_rope_hi: Gather,
    pos_buf: DeviceBuffer<u32>,
    n_keys_buf: DeviceBuffer<u32>,
    token_buf: DeviceBuffer<u32>,
    cs_buf: DeviceBuffer<f32>,
}

impl LayerScratch {
    /// Device bytes of the arena and the parameter buffers (weights and KV
    /// are counted by their owners).
    fn bytes(&self) -> usize {
        let mut total = [
            &self.x,
            &self.normed,
            &self.q,
            &self.q_rope_all,
            &self.kv_a,
            &self.kv_s,
            &self.kvr,
            &self.f_rows,
            &self.kqvc,
            &self.kqv_2d,
            &self.attn_out,
            &self.ffn_inp,
            &self.l_out,
            &self.cs_buf,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        total +=
            self.pos_buf.num_bytes() + self.n_keys_buf.num_bytes() + self.token_buf.num_bytes();
        let act = |a: &Q8Act| {
            a.q3.num_bytes()
                + a.q4.num_bytes()
                + a.q6.num_bytes()
                + a.s8.num_bytes()
                + a.d8.num_bytes()
        };
        total += act(&self.act_q)
            + act(&self.act_kv_lo)
            + act(&self.act_kv_hi)
            + act(&self.act_ao)
            + act(&self.act_ffn);
        total += [&self.g_kvr, &self.g_f_rope_lo, &self.g_f_rope_hi]
            .iter()
            .map(|g| g.src.num_bytes() + g.dst.num_bytes())
            .sum::<usize>();
        total += self.dense.as_ref().map_or(0, DenseScratch::bytes);
        total += self.moe.as_ref().map_or(0, MoeScratch::bytes);
        total
    }
}

// ------------------------------------------------------------------- stage

/// A contiguous range of blocks resident on one device (docs/gpu-design.md
/// decision 7). A stage owns its `Gpu` — context, stream, modules — and its
/// weights, KV rows and scratch once loaded with residency; what crosses a
/// stage boundary is one hidden vector. Two stages may sit on the same
/// card: that is the shape the 2-stage = 1-stage bit-identity gate runs in.
///
/// `residency` is filled by [`GpuModel::load_blocks`]; a stage from
/// [`GpuModel::load_staged`] carries only the metadata (that entry must stay
/// cheap — metadata probes call it).
pub struct Stage {
    gpu: Gpu,
    /// Blocks `layers.start..layers.end` of the model, in order.
    layers: std::ops::Range<usize>,
    /// The captured decode step of this stage, once one is assembled.
    graph: Option<Graph>,
    /// What `graph` recorded: the layer and whether it embeds in front. A
    /// replay names what it expects, so a graph of another layer is an
    /// error, not a silent replay of the wrong chain.
    graph_of: Option<(usize, bool)>,
    /// Weights, KV and scratch — present once loaded with residency.
    residency: Option<Residency>,
}

/// Everything a stage's step touches, allocated at load (decision 4).
struct Residency {
    weights: Weights,
    /// One `[ctx_max, kv_width]` u16 cache per layer, `kvr` row layout.
    kv: Vec<DeviceTensor<u16>>,
    /// The weight names of each layer of the stage, in layer order.
    names: Vec<LayerNames>,
    scratch: LayerScratch,
    step: StepKernels,
}

impl Stage {
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layers.clone()
    }

    /// Whether this stage replays a captured graph or enqueues eagerly.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Device bytes held by the residency: weights, KV caches, scratch.
    /// Zero for a metadata-only stage.
    pub fn resident_bytes(&self) -> usize {
        self.residency.as_ref().map_or(0, |r| {
            r.weights.resident_bytes()
                + r.kv.iter().map(|c| c.buf().len() * 2).sum::<usize>()
                + r.scratch.bytes()
        })
    }
}

/// Host copies of block 0's tap tensors for one position, in the dump's
/// logical order at that position. The fused FFN exposes neither
/// `ffn_norm-0` nor the down-projection `ffn_out-0` (the norm feeds the
/// quantizer in registers; the down store folds the residual) — `l_out-0`
/// carries that span.
pub struct Block0Taps {
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_rope_compressed: Vec<f32>,
    /// Head-major `q_rope(h)` per head, the dump's `(d, h, t)` at one t.
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub kv_compressed: Vec<f32>,
    /// Head-major `kqv(h)` per head.
    pub kqv_compressed: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub l_out: Vec<f32>,
}

impl Block0Taps {
    /// First tap holding a non-finite value, if any — the gate's finiteness
    /// check over every span, including the ones not compared.
    pub fn non_finite(&self) -> Option<&'static str> {
        let all: [&str; 10] = [
            "attn_norm",
            "q",
            "kv_rope_compressed",
            "q_rope",
            "k_rope",
            "kv_compressed",
            "kqv_compressed",
            "kqv_out",
            "ffn_inp",
            "l_out",
        ];
        let vals: [&[f32]; 10] = [
            &self.attn_norm,
            &self.q,
            &self.kv_rope_compressed,
            &self.q_rope,
            &self.k_rope,
            &self.kv_compressed,
            &self.kqv_compressed,
            &self.kqv_out,
            &self.ffn_inp,
            &self.l_out,
        ];
        all.iter()
            .zip(vals)
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
            .map(|(n, _)| *n)
    }

    /// Bit equality of every tap — the gate's rerun/replay checks.
    pub fn bits_equal(&self, other: &Block0Taps) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        eq(&self.attn_norm, &other.attn_norm)
            && eq(&self.q, &other.q)
            && eq(&self.kv_rope_compressed, &other.kv_rope_compressed)
            && eq(&self.q_rope, &other.q_rope)
            && eq(&self.k_rope, &other.k_rope)
            && eq(&self.kv_compressed, &other.kv_compressed)
            && eq(&self.kqv_compressed, &other.kqv_compressed)
            && eq(&self.kqv_out, &other.kqv_out)
            && eq(&self.ffn_inp, &other.ffn_inp)
            && eq(&self.l_out, &other.l_out)
    }
}

/// Host copies of one layer's tap tensors for one position, in the dump's
/// logical order at that position. The MoE spans are empty for a layer
/// without a router, and so is `ffn_norm` (the fused dense FFN keeps its
/// normed vector in registers). The routed
/// half's `ffn_moe_out` and `ffn_out` are not here: `moe_combine` folds the
/// weighted sum, the shared expert and the residual into one store, so
/// `l_out` carries that span — `expert_down` and `moe_weights` are the
/// operands a caller can recombine.
pub struct LayerTaps {
    pub layer: usize,
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_rope_compressed: Vec<f32>,
    /// Head-major `q_rope(h)` per head, the dump's `(d, h, t)` at one t.
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub kv_compressed: Vec<f32>,
    /// Head-major `kqv(h)` per head.
    pub kqv_compressed: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub moe_logits: Vec<f32>,
    /// The router's chosen expert ids, rank order — the `sel` the expert
    /// kernels read.
    pub moe_ids: Vec<u32>,
    /// The router weight of each chosen expert, same order.
    pub moe_weights: Vec<f32>,
    /// Each slot's down projection, slot-major (`n_used * hidden`).
    pub expert_down: Vec<f32>,
    pub ffn_shexp: Vec<f32>,
    pub l_out: Vec<f32>,
}

impl LayerTaps {
    /// Every f32 span with its name, in forward order — the finiteness and
    /// bit-equality checks walk this one list so a new tap cannot be added
    /// to the struct and forgotten by the checks.
    fn spans(&self) -> [(&'static str, &[f32]); 14] {
        [
            ("attn_norm", &self.attn_norm),
            ("q", &self.q),
            ("kv_rope_compressed", &self.kv_rope_compressed),
            ("q_rope", &self.q_rope),
            ("k_rope", &self.k_rope),
            ("kv_compressed", &self.kv_compressed),
            ("kqv_compressed", &self.kqv_compressed),
            ("kqv_out", &self.kqv_out),
            ("ffn_inp", &self.ffn_inp),
            ("ffn_norm", &self.ffn_norm),
            ("moe_logits", &self.moe_logits),
            ("moe_weights", &self.moe_weights),
            ("expert_down", &self.expert_down),
            ("ffn_shexp", &self.ffn_shexp),
        ]
    }

    /// First tap holding a non-finite value, if any — the gate's finiteness
    /// check over every span, including the ones not compared.
    pub fn non_finite(&self) -> Option<&'static str> {
        self.spans()
            .into_iter()
            .chain([("l_out", self.l_out.as_slice())])
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
            .map(|(n, _)| n)
    }

    /// Bit equality of every tap, the routed ids included — the gate's
    /// rerun/replay checks.
    pub fn bits_equal(&self, other: &LayerTaps) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.layer == other.layer
            && self.moe_ids == other.moe_ids
            && eq(&self.l_out, &other.l_out)
            && self
                .spans()
                .into_iter()
                .zip(other.spans())
                .all(|((_, a), (_, b))| eq(a, b))
    }
}

/// Per-op timing of one block-0 chain run from
/// [`GpuModel::profile_block0`]: `us_mean`/`us_min` over the measured reps
/// of that op's eager launch + body + the one stream synchronize the
/// profiling observer issues after it. Every op carries the same sync
/// overhead; subtract a touch-launch+sync constant (the gate's
/// `sync_floor_us`) to compare op bodies.
pub struct OpTime {
    pub index: usize,
    pub name: &'static str,
    pub us_mean: f64,
    pub us_min: f64,
}

/// The profiling observer's state: per-op sample lists and the wall clock
/// of the previous synchronize. Owned by [`GpuModel::profile_block0`]'s rep
/// loop, borrowed by the observer closure for one chain run at a time.
struct ProfRec {
    ops: Vec<(&'static str, Vec<f64>)>,
    last: std::time::Instant,
}

impl ProfRec {
    /// Synchronize after op `i`'s enqueue and record the wall time since the
    /// previous sync — the op's eager launch + body + one synchronize.
    fn observe(
        &mut self,
        i: usize,
        name: &'static str,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        stream.synchronize()?;
        let now = std::time::Instant::now();
        let us = now.duration_since(self.last).as_secs_f64() * 1e6;
        self.last = now;
        if i == self.ops.len() {
            self.ops.push((name, Vec::new()));
        }
        let slot = &mut self.ops[i];
        if slot.0 != name {
            return Err(format!(
                "profile_block0: op {i} was {} on an earlier rep, now {name}",
                slot.0
            )
            .into());
        }
        slot.1.push(us);
        Ok(())
    }
}

/// One resident model: its stages in layer order, covering every block
/// exactly once. Everything `step` touches is allocated at load, never per
/// step.
pub struct GpuModel {
    stages: Vec<Stage>,
    mla: MlaParams,
    /// The MoE shapes — `None` when no resident layer routes.
    moe: Option<MoeDims>,
    /// KV rows the resident cache was sized for; `step` refuses to grow it.
    ctx_max: usize,
}

impl GpuModel {
    /// The whole model as one stage (metadata only — a resident stage comes
    /// from [`GpuModel::load_blocks`]).
    pub fn load(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel, GpuError> {
        GpuModel::load_staged(gguf, ctx_max, &[])
    }

    /// Split the blocks at `cuts` (strictly ascending, each in
    /// `1..block_count`): `cuts.len() + 1` stages. Reads the metadata `step`
    /// needs; the stages carry no residency until [`GpuModel::load_blocks`]
    /// fills one.
    pub fn load_staged(
        gguf: &gguf::Gguf,
        ctx_max: usize,
        cuts: &[usize],
    ) -> Result<GpuModel, GpuError> {
        if ctx_max == 0 {
            return Err("GpuModel::load: ctx_max must be >= 1".into());
        }
        let n_layers =
            gguf.block_count()
                .ok_or("GpuModel::load: metadata key block_count missing")? as usize;
        let mut bounds = vec![0usize];
        for &c in cuts {
            if c <= *bounds.last().unwrap_or(&0) || c >= n_layers {
                return Err(format!(
                    "GpuModel::load_staged: cuts must ascend strictly inside 1..{n_layers}, got {cuts:?}"
                )
                .into());
            }
            bounds.push(c);
        }
        bounds.push(n_layers);
        let mut stages = Vec::with_capacity(bounds.len() - 1);
        for w in bounds.windows(2) {
            stages.push(Stage {
                gpu: Gpu::new()?,
                layers: w[0]..w[1],
                graph: None,
                graph_of: None,
                residency: None,
            });
        }
        let mla = MlaParams::read(gguf, 0)?;
        Ok(GpuModel {
            stages,
            mla,
            moe: None,
            ctx_max,
        })
    }

    /// One stage over `layers` WITH residency: every tensor of that range
    /// plus the globals in its kernels' device format, one KV cache per
    /// layer, the m = 1 layer scratch and this file's step module. The
    /// geometry checks below run at load, not mid-step; a range holding a
    /// routed layer also sizes the MoE half's arena.
    pub fn load_blocks(
        gguf: &gguf::Gguf,
        ctx_max: usize,
        layers: Range<usize>,
    ) -> Result<GpuModel, GpuError> {
        if ctx_max == 0 {
            return Err("GpuModel::load_blocks: ctx_max must be >= 1".into());
        }
        let n_layers = gguf
            .block_count()
            .ok_or("GpuModel::load_blocks: metadata key block_count missing")?
            as usize;
        if layers.start >= layers.end || layers.end > n_layers {
            return Err(format!(
                "GpuModel::load_blocks: layer range {layers:?} outside 0..{n_layers}"
            )
            .into());
        }
        let mla = MlaParams::read(gguf, 0)?;
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let weights = Weights::load(stream, gguf, layers.clone(), true)?;
        let names: Vec<LayerNames> = layers
            .clone()
            .map(|l| LayerNames::new(&weights, l))
            .collect();
        let moe = match names.iter().find(|n| n.routed) {
            Some(n) => Some(MoeDims::read(gguf, &weights, n)?),
            None => None,
        };
        let scratch = LayerScratch::new(stream, &weights, &mla, &names, moe.as_ref())?;
        let kv_width = mla.latent + mla.rope_dims;
        let kv = (0..layers.len())
            .map(|_| DeviceTensor::<u16>::zeroed(stream, ctx_max, kv_width))
            .collect::<Result<Vec<_>, _>>()?;
        let step = StepKernels::load(gpu.context())?;
        Ok(GpuModel {
            stages: vec![Stage {
                gpu,
                layers,
                graph: None,
                graph_of: None,
                residency: Some(Residency {
                    weights,
                    kv,
                    names,
                    scratch,
                    step,
                }),
            }],
            mla,
            moe,
            ctx_max,
        })
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    pub fn mla(&self) -> &MlaParams {
        &self.mla
    }

    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    /// One decode step: append `tokens` to the KV cache and return the argmax
    /// of the last position's logits. The layers are assembled one at a time
    /// (`step_layer_taps`) but not yet stitched into a stage-wide chain, and
    /// the lm_head lives in `head.rs`, so this is an error, not a panic, and
    /// a caller can probe for it.
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        if tokens.is_empty() {
            return Err("GpuModel::step: empty token slice".into());
        }
        Err(
            "GpuModel::step: the stage-wide chain (layer to layer residual, the \
             head) is another round; one layer at a time is assembled — see \
             step_block0_taps and step_layer_taps"
                .into(),
        )
    }

    // ------------------------------------------------- assembled layer step

    /// The one resident stage. Every assembled path needs exactly one stage
    /// carrying residency; `what` names the caller in the error.
    fn stage_parts(&mut self, what: &str) -> Result<(&mut Gpu, &mut Residency), GpuError> {
        if self.stages.len() != 1 {
            return Err(format!(
                "GpuModel::{what}: the assembled step needs the single stage of load_blocks"
            )
            .into());
        }
        let stage = &mut self.stages[0];
        let Some(residency) = stage.residency.as_mut() else {
            return Err(format!(
                "GpuModel::{what}: stage carries no residency (load_blocks fills it)"
            )
            .into());
        };
        Ok((&mut stage.gpu, residency))
    }

    /// The slot of layer `l` inside the resident range — the index of its KV
    /// cache and of its names.
    fn layer_slot(&self, l: usize, what: &str) -> Result<usize, GpuError> {
        if self.stages.len() != 1 {
            return Err(format!(
                "GpuModel::{what}: the assembled step needs the single stage of load_blocks"
            )
            .into());
        }
        let layers = self.stages[0].layers.clone();
        if !layers.contains(&l) {
            return Err(format!(
                "GpuModel::{what}: layer {l} is outside the resident range {layers:?}"
            )
            .into());
        }
        Ok(l - layers.start)
    }

    /// The one stage, requiring it to hold block 0 with residency.
    fn block0_parts(&mut self) -> Result<(&mut Gpu, &mut Residency), GpuError> {
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(
                "GpuModel: the assembled block-0 step needs a stage of load_blocks starting \
                 at layer 0"
                    .into(),
            );
        }
        self.stage_parts("block0")
    }

    /// Refresh every per-step device parameter for `(token, pos)`: the rope
    /// cos/sin cache (host YaRN math, one position), the token id, the KV
    /// landing row and the live key count. Runs before an eager enqueue or a
    /// graph replay; a captured graph reads these buffers at run time, which
    /// is what lets one graph serve every position.
    fn refresh_params(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        let mut cs = Vec::new();
        self.mla.rope.cache_into(pos, &mut cs);
        let (gpu, residency) = self.stage_parts("refresh_params")?;
        let stream = gpu.stream();
        let s = &mut residency.scratch;
        s.cs_buf.copy_from_host(stream, &cs)?;
        s.token_buf.copy_from_host(stream, &[token])?;
        s.pos_buf.copy_from_host(stream, &[pos])?;
        s.n_keys_buf.copy_from_host(stream, &[pos + 1])?;
        Ok(())
    }

    /// Enqueue the block-0 step on the engine stream (eager form). Pure
    /// enqueues — no allocation, no synchronization — so the same body is
    /// what [`GpuModel::capture_block0`] records.
    fn enqueue_step(&mut self) -> Result<(), GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let (gpu, residency) = self.block0_parts()?;
        let Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        } = residency;
        enqueue_layer(
            gpu,
            step,
            weights,
            &names[0],
            &mut kv[0],
            scratch,
            &mla,
            moe.as_ref(),
            true,
            &mut |_, _| Ok(()),
        )
    }

    /// Enqueue layer `l`'s step on the engine stream (eager form), the
    /// layer's input residual already in the resident input buffer. Pure
    /// enqueues — the same body [`GpuModel::capture_layer`] records.
    fn enqueue_layer_step(&mut self, l: usize) -> Result<(), GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let slot = self.layer_slot(l, "enqueue_layer_step")?;
        let (gpu, residency) = self.stage_parts("enqueue_layer_step")?;
        let Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        } = residency;
        enqueue_layer(
            gpu,
            step,
            weights,
            &names[slot],
            &mut kv[slot],
            scratch,
            &mla,
            moe.as_ref(),
            false,
            &mut |_, _| Ok(()),
        )
    }

    /// Eagerly run block 0's step for `token` at `pos` (`pos + 1` live keys,
    /// rows `0..pos` already in the cache — this call appends row `pos`) and
    /// read every tap back. Synchronizes; gate/debug use.
    pub fn step_block0_taps(&mut self, token: u32, pos: u32) -> Result<Block0Taps, GpuError> {
        self.check_pos(pos, "step_block0_taps")?;
        self.refresh_params(token, pos)?;
        self.enqueue_step()?;
        self.block0_taps()
    }

    /// Read the tap tensors of the last run (eager or replay). Synchronizes
    /// per readback; gate/debug use.
    pub fn block0_taps(&mut self) -> Result<Block0Taps, GpuError> {
        let mla = self.mla.clone();
        let (gpu, residency) = self.block0_parts()?;
        let stream = gpu.stream();
        let s = &residency.scratch;
        let rd =
            |b: &DeviceBuffer<f32>| -> Result<Vec<f32>, GpuError> { Ok(b.to_host_vec(stream)?) };
        let f_rows = rd(&s.f_rows)?;
        let width = mla.rope_dims + mla.latent;
        let mut q_rope = Vec::with_capacity(mla.n_head * mla.rope_dims);
        for h in 0..mla.n_head {
            q_rope.extend_from_slice(&f_rows[h * width..h * width + mla.rope_dims]);
        }
        let kv_s = rd(&s.kv_s)?;
        let kvr = rd(&s.kvr)?;
        Ok(Block0Taps {
            attn_norm: rd(&s.normed)?,
            q: rd(&s.q)?,
            kv_rope_compressed: rd(&s.kv_a)?,
            q_rope,
            k_rope: kvr[..mla.rope_dims].to_vec(),
            kv_compressed: kv_s[..mla.latent].to_vec(),
            kqv_compressed: rd(&s.kqvc)?,
            kqv_out: rd(&s.attn_out)?,
            ffn_inp: rd(&s.ffn_inp)?,
            l_out: rd(&s.l_out)?,
        })
    }

    /// Read layer `l`'s tap tensors of the last run (eager or replay). The
    /// MoE spans come back empty for a layer without a router.
    /// Synchronizes per readback; gate/debug use.
    pub fn layer_taps(&mut self, l: usize) -> Result<LayerTaps, GpuError> {
        let mla = self.mla.clone();
        let slot = self.layer_slot(l, "layer_taps")?;
        let (gpu, residency) = self.stage_parts("layer_taps")?;
        let stream = gpu.stream();
        let routed = residency.names[slot].routed;
        let s = &residency.scratch;
        let rd =
            |b: &DeviceBuffer<f32>| -> Result<Vec<f32>, GpuError> { Ok(b.to_host_vec(stream)?) };
        let f_rows = rd(&s.f_rows)?;
        let width = mla.rope_dims + mla.latent;
        let mut q_rope = Vec::with_capacity(mla.n_head * mla.rope_dims);
        for h in 0..mla.n_head {
            q_rope.extend_from_slice(&f_rows[h * width..h * width + mla.rope_dims]);
        }
        let kv_s = rd(&s.kv_s)?;
        let kvr = rd(&s.kvr)?;
        let moe = if routed { s.moe.as_ref() } else { None };
        let moe_rd = |pick: fn(&MoeScratch) -> &DeviceBuffer<f32>| -> Result<Vec<f32>, GpuError> {
            match moe {
                Some(m) => Ok(pick(m).to_host_vec(stream)?),
                None => Ok(Vec::new()),
            }
        };
        Ok(LayerTaps {
            layer: l,
            attn_norm: rd(&s.normed)?,
            q: rd(&s.q)?,
            kv_rope_compressed: rd(&s.kv_a)?,
            q_rope,
            k_rope: kvr[..mla.rope_dims].to_vec(),
            kv_compressed: kv_s[..mla.latent].to_vec(),
            kqv_compressed: rd(&s.kqvc)?,
            kqv_out: rd(&s.attn_out)?,
            ffn_inp: rd(&s.ffn_inp)?,
            ffn_norm: moe_rd(|m| &m.normed)?,
            moe_logits: moe_rd(|m| &m.logits)?,
            moe_ids: match moe {
                Some(m) => m.ids.to_host_vec(stream)?,
                None => Vec::new(),
            },
            moe_weights: moe_rd(|m| &m.weights)?,
            expert_down: moe_rd(|m| &m.down)?,
            ffn_shexp: moe_rd(|m| &m.shexp)?,
            l_out: rd(&s.l_out)?,
        })
    }

    /// Write `x_in` into the resident input buffer — the layer's input
    /// residual, which a lone layer has no embedding in front of to produce.
    /// Synchronizes; never inside a capture.
    pub fn set_layer_input(&mut self, x_in: &[f32]) -> Result<(), GpuError> {
        let (gpu, residency) = self.stage_parts("set_layer_input")?;
        let s = &mut residency.scratch;
        if x_in.len() != s.dims.hidden {
            return Err(format!(
                "GpuModel::set_layer_input: {} values, the hidden width is {}",
                x_in.len(),
                s.dims.hidden
            )
            .into());
        }
        s.x.copy_from_host(gpu.stream(), x_in)?;
        Ok(())
    }

    /// Eagerly run layer `l`'s step for the input residual `x_in` at `pos`
    /// (`pos + 1` live keys, rows `0..pos` already in that layer's cache —
    /// this call appends row `pos`) and read every tap back. Synchronizes;
    /// gate/debug use.
    pub fn step_layer_taps(
        &mut self,
        l: usize,
        x_in: &[f32],
        pos: u32,
    ) -> Result<LayerTaps, GpuError> {
        self.check_pos(pos, "step_layer_taps")?;
        self.set_layer_input(x_in)?;
        self.refresh_params(0, pos)?;
        self.enqueue_layer_step(l)?;
        self.layer_taps(l)
    }

    /// Capture layer `l`'s step into the stage's graph over the resident
    /// buffers (their addresses freeze — they were allocated at load). The
    /// input residual is read from the resident input buffer and the routed
    /// expert ids from the router's own device buffer, so one graph serves
    /// every input and every routing. Returns the node count.
    pub fn capture_layer(&mut self, l: usize) -> Result<usize, GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let slot = self.layer_slot(l, "capture_layer")?;
        let stage = &mut self.stages[0];
        let Some(Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        }) = stage.residency.as_mut()
        else {
            return Err("GpuModel::capture_layer: stage carries no residency".into());
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| {
            enqueue_layer(
                gpu,
                step,
                weights,
                &names[slot],
                &mut kv[slot],
                scratch,
                &mla,
                moe.as_ref(),
                false,
                &mut |_, _| Ok(()),
            )
        })?;
        let nodes = graph.node_count();
        stage.graph = Some(graph);
        stage.graph_of = Some((l, false));
        Ok(nodes)
    }

    /// Refresh the step parameters for `(x_in, pos)` and replay the captured
    /// layer graph. Synchronizes.
    pub fn replay_layer(&mut self, l: usize, x_in: &[f32], pos: u32) -> Result<(), GpuError> {
        self.check_pos(pos, "replay_layer")?;
        self.layer_slot(l, "replay_layer")?;
        self.set_layer_input(x_in)?;
        self.refresh_params(0, pos)?;
        self.launch_graph((l, false))?;
        self.stages[0].gpu.stream().synchronize()?;
        Ok(())
    }

    /// Write `rows` (whole `kv_width`-wide rows) at the head of layer `l`'s
    /// cache, zeroing the rest — the seeding path the gate uses to give the
    /// step a prefix of oracle rows. Synchronizes; never inside a capture.
    pub fn seed_layer_cache(&mut self, l: usize, rows: &[u16]) -> Result<(), GpuError> {
        let slot = self.layer_slot(l, "seed_layer_cache")?;
        let (gpu, residency) = self.stage_parts("seed_layer_cache")?;
        seed_cache(gpu, &mut residency.kv[slot], rows, "seed_layer_cache")
    }

    /// Capture the block-0 step into the stage's graph over the resident
    /// buffers (their addresses freeze — they were allocated at load).
    /// Returns the node count.
    pub fn capture_block0(&mut self) -> Result<usize, GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(
                "GpuModel::capture_block0: needs a stage of load_blocks starting at layer 0".into(),
            );
        }
        let stage = &mut self.stages[0];
        let Some(Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        }) = stage.residency.as_mut()
        else {
            return Err("GpuModel::capture_block0: stage carries no residency".into());
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| {
            enqueue_layer(
                gpu,
                step,
                weights,
                &names[0],
                &mut kv[0],
                scratch,
                &mla,
                moe.as_ref(),
                true,
                &mut |_, _| Ok(()),
            )
        })?;
        let nodes = graph.node_count();
        stage.graph = Some(graph);
        stage.graph_of = Some((0, true));
        Ok(nodes)
    }

    /// Enqueue one replay of the captured graph — no parameter refresh, no
    /// synchronization. The timing arm of the gate drives this in a loop.
    pub fn launch_block0_graph(&self) -> Result<(), GpuError> {
        self.launch_graph((0, true))
    }

    /// Launch the stage's graph, requiring it to be the capture of `want`
    /// (layer, embeds in front).
    fn launch_graph(&self, want: (usize, bool)) -> Result<(), GpuError> {
        let stage = &self.stages[0];
        let graph = stage
            .graph
            .as_ref()
            .ok_or("GpuModel::launch_graph: no captured graph")?;
        if stage.graph_of != Some(want) {
            return Err(format!(
                "GpuModel::launch_graph: the captured graph is {:?} (layer, embed), the replay \
                 wants {want:?}",
                stage.graph_of
            )
            .into());
        }
        graph.launch(stage.gpu.stream())
    }

    /// Refresh the step parameters for `(token, pos)` and replay the
    /// captured block-0 graph. Synchronizes.
    pub fn replay_block0(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        self.check_pos(pos, "replay_block0")?;
        self.refresh_params(token, pos)?;
        self.launch_block0_graph()?;
        let stream = self.stages[0].gpu.stream();
        stream.synchronize()?;
        Ok(())
    }

    /// Time the block-0 step op by op: `reps` measured chain runs (after 20
    /// warm-up runs, discarded), each enqueued eagerly at `(token, pos)` with
    /// an observer that synchronizes after every op and records wall time
    /// since the previous sync. Each sample is therefore an eager launch,
    /// body and one synchronize — it overstates every op by the same host
    /// sync cost, which the caller calibrates against a bare launch+sync
    /// constant. Parameters are refreshed once before the loop; the chain is
    /// idempotent at a fixed `(token, pos)` (it rewrites the same KV row and
    /// every scratch buffer it reads), so the runs leave the model exactly
    /// where one eager step would. Debug/profiling use — never inside a
    /// capture (the observer synchronizes).
    pub fn profile_block0(
        &mut self,
        token: u32,
        pos: u32,
        reps: u32,
    ) -> Result<Vec<OpTime>, GpuError> {
        self.check_pos(pos, "profile_block0")?;
        if reps == 0 {
            return Err("profile_block0: reps must be >= 1".into());
        }
        self.refresh_params(token, pos)?;
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let (gpu, residency) = self.block0_parts()?;
        let stream = gpu.stream();
        let Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        } = residency;
        let mut rec = ProfRec {
            // Slot i is created by op i's first firing (indices arrive in
            // chain order); the name is cross-checked on every later rep.
            ops: Vec::new(),
            last: std::time::Instant::now(),
        };
        const WARMUP: u32 = 20;
        for rep in 0..(WARMUP + reps) {
            rec.last = std::time::Instant::now();
            let mut obs = |i: usize, name: &'static str| rec.observe(i, name, stream);
            enqueue_layer(
                gpu,
                step,
                weights,
                &names[0],
                &mut kv[0],
                scratch,
                &mla,
                moe.as_ref(),
                true,
                &mut obs,
            )?;
            if rep < WARMUP {
                for (_, s) in rec.ops.iter_mut() {
                    s.clear();
                }
            }
        }
        stream.synchronize()?;
        Ok(rec
            .ops
            .into_iter()
            .enumerate()
            .map(|(index, (name, s))| OpTime {
                index,
                name,
                us_mean: s.iter().sum::<f64>() / s.len() as f64,
                us_min: s.iter().cloned().fold(f64::INFINITY, f64::min),
            })
            .collect())
    }

    /// Mean host wall time (µs) of one `refresh_params` call — the four
    /// host→device parameter copies that sit outside the captured graph but
    /// inside every real decode step. Same 20-run warm-up convention as
    /// [`GpuModel::profile_block0`]; the timed window is the call itself,
    /// not the copies' stream completion.
    pub fn refresh_params_us(&mut self, token: u32, pos: u32, reps: u32) -> Result<f64, GpuError> {
        self.check_pos(pos, "refresh_params_us")?;
        if reps == 0 {
            return Err("refresh_params_us: reps must be >= 1".into());
        }
        for _ in 0..20 {
            self.refresh_params(token, pos)?;
        }
        self.stages[0].gpu.stream().synchronize()?;
        let mut total = 0.0f64;
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            self.refresh_params(token, pos)?;
            total += t0.elapsed().as_secs_f64() * 1e6;
        }
        self.stages[0].gpu.stream().synchronize()?;
        Ok(total / f64::from(reps))
    }

    /// Write `rows` (whole `kv_width`-wide rows, `rows.len() <= ctx_max * kv_width`)
    /// at the head of layer 0's cache, zeroing the rest — the seeding path
    /// the gate uses to give the step a prefix of oracle rows. Synchronizes;
    /// never inside a capture.
    pub fn seed_block0_cache(&mut self, rows: &[u16]) -> Result<(), GpuError> {
        let (gpu, residency) = self.block0_parts()?;
        seed_cache(gpu, &mut residency.kv[0], rows, "seed_block0_cache")
    }

    fn check_pos(&self, pos: u32, what: &str) -> Result<(), GpuError> {
        if pos as usize + 1 > self.ctx_max {
            return Err(format!(
                "GpuModel::{what}: pos {pos} + 1 exceeds the resident cache's {} rows",
                self.ctx_max
            )
            .into());
        }
        Ok(())
    }
}

/// Write `rows` (whole cache rows) at the head of `cache`, zeroing the rest.
/// Synchronizes; never inside a capture.
fn seed_cache(
    gpu: &Gpu,
    cache: &mut DeviceTensor<u16>,
    rows: &[u16],
    what: &str,
) -> Result<(), GpuError> {
    let full_len = cache.rows() * cache.cols();
    if rows.is_empty() || rows.len() > full_len || !rows.len().is_multiple_of(cache.cols()) {
        return Err(format!(
            "{what}: {} values are not whole {}-wide rows inside {full_len}",
            rows.len(),
            cache.cols()
        )
        .into());
    }
    let mut full = vec![0u16; full_len];
    full[..rows.len()].copy_from_slice(rows);
    cache.buf_mut().copy_from_host(gpu.stream(), &full)?;
    Ok(())
}

impl MoeDims {
    /// Read the MoE shapes from the file's metadata and cross-check them
    /// against the resident expert stacks of `names` — the quantization
    /// types and row counts the enqueue leans on are proven here, at load.
    fn read(gguf: &gguf::Gguf, w: &Weights, names: &LayerNames) -> Result<MoeDims, GpuError> {
        use gguf::quant::GgmlType;

        let meta = model::moe::Meta::read(gguf)?;
        let hidden = f32_gain(w, &names.attn_norm)?.len();
        // The router kernel ranks a fixed 64 experts into a fixed 6 slots.
        if meta.n_expert != 64 || meta.n_used != 6 {
            return Err(format!(
                "MoeDims::read: the router kernel is 64 experts into 6 slots, the file says \
                 {} into {}",
                meta.n_expert, meta.n_used
            )
            .into());
        }
        let kq_ty = |name: &str, want: GgmlType| -> Result<(), GpuError> {
            match w.get(name) {
                Some(DevWeight::KQuant { ty, .. }) if *ty == want => Ok(()),
                Some(other) => Err(format!(
                    "MoeDims::read: {name} is resident as {} rows of k={}, want {want}",
                    other.rows(),
                    other.k()
                )
                .into()),
                None => Err(format!("MoeDims::read: {name} not resident").into()),
            }
        };
        kq_ty(&names.ffn_gate_exps, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_up_exps, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_gate_shexp, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_up_shexp, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_down_shexp, GgmlType::Q4_K)?;
        // The routed down stack's kernel is `q5_0_gemv_sel`: a Q5_1 or a
        // K-quant here is a different kernel, not a different constant.
        match w.get(&names.ffn_down_exps) {
            Some(DevWeight::Q5_0 { .. }) => {}
            Some(_) => {
                return Err(format!(
                    "MoeDims::read: {} is not resident as Q5_0 — the routed down projection \
                     runs q5_0_gemv_sel",
                    names.ffn_down_exps
                )
                .into());
            }
            None => {
                return Err(format!("MoeDims::read: {} not resident", names.ffn_down_exps).into());
            }
        }
        match w.get(&names.ffn_gate_inp) {
            Some(DevWeight::F32 { .. }) => {}
            _ => {
                return Err(format!(
                    "MoeDims::read: {} is not resident as F32 — the router is an f32 gemv",
                    names.ffn_gate_inp
                )
                .into());
            }
        }
        let gate_exps = kq_weight(w, &names.ffn_gate_exps)?;
        let down_exps = kq_weight(w, &names.ffn_down_exps)?;
        let shexp_ff = kq_weight(w, &names.ffn_gate_shexp)?.rows();
        let check = |what: &str, want: usize, got: usize| -> Result<(), GpuError> {
            if want == got {
                Ok(())
            } else {
                Err(format!("MoeDims::read: {what}: {want} != {got}").into())
            }
        };
        check(
            "ffn_gate_exps rows vs n_expert*expert_ff",
            gate_exps.rows(),
            meta.n_expert * meta.ff,
        )?;
        check(
            "ffn_up_exps rows vs n_expert*expert_ff",
            kq_weight(w, &names.ffn_up_exps)?.rows(),
            meta.n_expert * meta.ff,
        )?;
        check(
            "ffn_down_exps rows vs n_expert*hidden",
            down_exps.rows(),
            meta.n_expert * hidden,
        )?;
        check(
            "ffn_up_shexp rows vs ffn_gate_shexp rows",
            kq_weight(w, &names.ffn_up_shexp)?.rows(),
            shexp_ff,
        )?;
        check(
            "ffn_down_shexp rows vs hidden",
            kq_weight(w, &names.ffn_down_shexp)?.rows(),
            hidden,
        )?;
        Ok(MoeDims {
            n_expert: meta.n_expert,
            n_used: meta.n_used,
            ff: meta.ff,
            scale: meta.scale,
            shexp_ff,
        })
    }
}

impl LayerScratch {
    /// Derive every size from the resident weights and the MLA metadata,
    /// cross-check the geometry the enqueue leans on, and allocate the arena
    /// (plus the pair tables the gathers run). The dense FFN buffers are
    /// sized from the stage's first dense layer, the routed ones from
    /// `moe`. Load-time only.
    fn new(
        stream: &CudaStream,
        w: &Weights,
        mla: &MlaParams,
        names: &[LayerNames],
        moe: Option<&MoeDims>,
    ) -> Result<LayerScratch, GpuError> {
        let first = names
            .first()
            .ok_or("LayerScratch::new: the stage holds no layer")?;
        let hidden = f32_gain(w, &first.attn_norm)?.len();
        let q_rows = kq_weight(w, &first.attn_q)?.rows();
        let kv_width = kq_weight(w, &first.attn_kv_a_mqa)?.rows();
        // The dense arena is sized from the stage's first dense layer; a
        // stage with none carries no dense arena.
        let dense_ff = match names.iter().find(|n| !n.routed) {
            Some(n) => Some(kq_weight(w, &n.ffn_gate)?.rows()),
            None => None,
        };
        let (_, qn2_d) = q8_derived(w, first.layer)?;
        let (derived, derived_k) = (qn2_d.rows(), qn2_d.cols() * 32);
        let kv_b_rows = kq_weight(w, &first.attn_kv_b)?.rows();
        let check = |what: &str, want: usize, got: usize| -> Result<(), GpuError> {
            if want == got {
                Ok(())
            } else {
                Err(format!("LayerScratch: {what}: {want} != {got}").into())
            }
        };
        check(
            "attn_q rows vs n_head*kq_head",
            q_rows,
            mla.n_head * mla.kq_head,
        )?;
        check(
            "kv_a rows vs latent+rope",
            kv_width,
            mla.latent + mla.rope_dims,
        )?;
        check(
            "derived rows vs n_head*latent",
            derived,
            mla.n_head * mla.latent,
        )?;
        check("derived k vs nope", derived_k, mla.nope)?;
        check(
            "kv_b rows vs n_head*(nope+v_head)",
            kv_b_rows,
            mla.n_head * (mla.nope + mla.v_head),
        )?;
        if q_rows % mla.rope_dims != 0 || kv_width % mla.rope_dims != 0 {
            return Err(format!(
                "LayerScratch: rope walks 64-value columns from the buffer start; q rows \
                 {q_rows} / kv width {kv_width} are not {}-aligned",
                mla.rope_dims
            )
            .into());
        }
        let half = mla.n_head / 2;
        if !mla.n_head.is_multiple_of(2) || half > 8 {
            return Err(format!(
                "LayerScratch: the half-split m = 8 quantize at the wv_b site needs an even \
                 n_head <= 16, got {}",
                mla.n_head
            )
            .into());
        }
        let (rope, latent, nope, kq_head) = (mla.rope_dims, mla.latent, mla.nope, mla.kq_head);
        let dims = Dims {
            hidden,
            q_cols: q_rows / rope,
            kv_cols: kv_width / rope,
        };
        let f32n = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let dense = match dense_ff {
            Some(ff) => Some(DenseScratch {
                h: f32n(ff)?,
                act32: Q8Blocks32::new(stream, ff, 1)?,
            }),
            None => None,
        };
        let moe = match moe {
            Some(m) => Some(MoeScratch {
                normed: f32n(hidden)?,
                logits: f32n(m.n_expert)?,
                probs: f32n(m.n_expert)?,
                ids: DeviceBuffer::<u32>::zeroed(stream, m.n_used)?,
                weights: f32n(m.n_used)?,
                h_exp: f32n(m.n_used * m.ff)?,
                act32_exp: Q8Blocks32::new(stream, m.ff, m.n_used)?,
                down: f32n(m.n_used * hidden)?,
                h_sh: f32n(m.shexp_ff)?,
                act_sh: Q8Act::with_k(stream, 1, m.shexp_ff)?,
                shexp: f32n(hidden)?,
            }),
            None => None,
        };
        Ok(LayerScratch {
            x: f32n(hidden)?,
            normed: f32n(hidden)?,
            act_q: Q8Act::with_k(stream, 1, hidden)?,
            q: f32n(q_rows)?,
            q_rope_all: f32n(q_rows)?,
            kv_a: f32n(kv_width)?,
            kv_s: f32n(kv_width)?,
            kvr: f32n(kv_width)?,
            f_rows: f32n(mla.n_head * kv_width)?,
            kqvc: f32n(mla.n_head * latent)?,
            act_kv_lo: Q8Act::with_k(stream, 8, latent)?,
            act_kv_hi: Q8Act::with_k(stream, 8, latent)?,
            kqv_2d: f32n(mla.n_head * mla.v_head)?,
            act_ao: Q8Act::with_k(stream, 1, hidden)?,
            attn_out: f32n(hidden)?,
            ffn_inp: f32n(hidden)?,
            act_ffn: Q8Act::with_k(stream, 1, hidden)?,
            dense,
            moe,
            l_out: f32n(hidden)?,
            g_kvr: Gather::new(
                stream,
                (0..rope)
                    .map(|i| (latent + i, i))
                    .chain((0..latent).map(|i| (i, rope + i))),
                "kvr",
            )?,
            g_f_rope_lo: Gather::new(
                stream,
                (0..half).flat_map(|h| {
                    let col = (h * kq_head + nope) / rope;
                    (0..rope).map(move |d| (col * rope + d, h * kv_width + d))
                }),
                "f_rope_lo",
            )?,
            g_f_rope_hi: Gather::new(
                stream,
                (half..mla.n_head).flat_map(|h| {
                    let col = (h * kq_head + nope) / rope;
                    (0..rope).map(move |d| (col * rope + d, h * kv_width + d))
                }),
                "f_rope_hi",
            )?,
            pos_buf: DeviceBuffer::from_host(stream, &[0u32])?,
            n_keys_buf: DeviceBuffer::from_host(stream, &[1u32])?,
            token_buf: DeviceBuffer::from_host(stream, &[0u32])?,
            cs_buf: f32n(rope)?,
            dims,
        })
    }
}

/// The resident derived q_nope2 planes of block `l` (`qs` rows x k/4 code
/// words, `d` rows x k/32 scales, rows = n_head * latent, k = nope).
pub(crate) fn q8_derived(
    w: &Weights,
    l: usize,
) -> Result<(&DeviceTensor<u32>, &DeviceTensor<f32>), GpuError> {
    let name = crate::weights::derived_name(l);
    match w.get(&name) {
        Some(DevWeight::Q8_0Derived { qs, d, .. }) => Ok((qs, d)),
        Some(_) => Err(format!("q8_derived: {name} is not the derived variant").into()),
        None => Err(format!("q8_derived: {name} not resident").into()),
    }
}

/// The resident Q3_K/Q4_K/Q6_K/Q5 word plane of a weight, by name.
pub(crate) fn kq_weight<'a>(w: &'a Weights, name: &str) -> Result<&'a DeviceTensor<u32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::KQuant { w, .. })
        | Some(DevWeight::Q5_0 { w, .. })
        | Some(DevWeight::Q5_1 { w, .. }) => Ok(w),
        Some(_) => Err(format!("kq_weight: {name} is not a word-plane variant").into()),
        None => Err(format!("kq_weight: {name} not resident").into()),
    }
}

/// The resident F32 plane (norm gains), by name.
pub(crate) fn f32_gain<'a>(w: &'a Weights, name: &str) -> Result<&'a DeviceBuffer<f32>, GpuError> {
    Ok(f32_tensor(w, name)?.buf())
}

/// The resident F32 plane of a weight as a tensor — the router matrix, whose
/// gemv addresses rows.
pub(crate) fn f32_tensor<'a>(
    w: &'a Weights,
    name: &str,
) -> Result<&'a DeviceTensor<f32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w, .. }) => Ok(w),
        Some(_) => Err(format!("f32_tensor: {name} is not F32").into()),
        None => Err(format!("f32_tensor: {name} not resident").into()),
    }
}

/// Tick the observer for op `*i` and advance. `obs` fires on the host after
/// every enqueue, carrying the op's index (one per launch, in chain order)
/// and the name of the numbered step it belongs to; returning `Err` aborts
/// the chain at that op. The normal and captured paths pass a no-op, which
/// issues byte-for-byte the same launches in the same order as an
/// uninstrumented chain — the observer is host-side only and never touches
/// the stream. The op index is the graph node index of the same launch.
fn tick(
    i: &mut usize,
    obs: &mut dyn FnMut(usize, &'static str) -> Result<(), GpuError>,
    name: &'static str,
) -> Result<(), GpuError> {
    obs(*i, name)?;
    *i += 1;
    Ok(())
}

/// Enqueue one layer's whole step at m = 1: the token embedding when the
/// layer is the first of the model, the attention half every layer shares,
/// and the FFN half the layer's own weights select — the fused dense FFN
/// when the layer has no router, the routed MoE half when it has one.
/// Mirrors `model::attn::block_attn_cached` and `model::moe`'s op order with
/// the gated kernels plus this file's gather. Asynchronous throughout —
/// capturable as a body. A layer that does not embed reads its input
/// residual from the resident input buffer.
#[allow(clippy::too_many_arguments)]
fn enqueue_layer(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &LayerNames,
    kv_l: &mut DeviceTensor<u16>,
    s: &mut LayerScratch,
    mla: &MlaParams,
    moe: Option<&MoeDims>,
    embed: bool,
    obs: &mut dyn FnMut(usize, &'static str) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let mut i = 0usize;
    if embed {
        // 0. embed(token) — the block input x, kept intact for both residuals.
        gpu.elem().enqueue_embed_rows(
            gpu.stream(),
            kq_weight(w, "token_embd.weight")?,
            &s.token_buf,
            &mut s.x,
        )?;
        tick(&mut i, obs, "embed")?;
    }
    enqueue_attn(gpu, step, w, names, kv_l, s, mla, &mut i, obs)?;
    if names.routed {
        let dims = moe.ok_or_else(|| -> GpuError {
            format!(
                "enqueue_layer: layer {} routes but the stage carries no MoE shapes",
                names.layer
            )
            .into()
        })?;
        enqueue_ffn_moe(gpu, w, names, s, mla, dims, &mut i, obs)
    } else {
        enqueue_ffn_dense(gpu, w, names, s, mla, &mut i, obs)
    }
}

/// Enqueue the attention half of layer `names.layer`: `s.x` (the layer's
/// input residual) in, `s.ffn_inp` (that residual plus the attention output)
/// out, appending the step's key row to `kv_l` and attending over the live
/// rows. Every layer runs this same chain against its own weights and its
/// own cache.
#[allow(clippy::too_many_arguments)]
fn enqueue_attn(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &LayerNames,
    kv_l: &mut DeviceTensor<u16>,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut dyn FnMut(usize, &'static str) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (hidden, latent, rope) = (s.dims.hidden, mla.latent, mla.rope_dims);
    let kv_width = latent + rope;
    let gather =
        |gt: &Gather, src: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>| -> Result<(), GpuError> {
            step.enqueue_gather(stream, src, &gt.src, &gt.dst, gt.n, y)
        };

    // 1. attn_norm(x) — the dump's FUSED_RMS_NORM output.
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32_gain(w, &names.attn_norm)?,
        mla.eps,
        hidden,
        1,
        &mut s.normed,
    )?;
    tick(i, obs, "attn_norm")?;
    // 2-3. the two projections over one q8_1 activation of the normed x.
    gpu.enqueue_quantize_q8_1(&s.normed, &mut s.act_q)?;
    tick(i, obs, "quantize_q8_1(act_q)")?;
    gpu.enqueue_gemv_q3k(kq_weight(w, &names.attn_q)?, &s.act_q, &mut s.q)?;
    tick(i, obs, "gemv_q3k(attn_q)")?;
    gpu.enqueue_gemv_q3k(kq_weight(w, &names.attn_kv_a_mqa)?, &s.act_q, &mut s.kv_a)?;
    tick(i, obs, "gemv_q3k(attn_kv_a_mqa)")?;
    // 4. rope over every 64-value column of both projections: the q layout
    //    puts each head's rope slice on a column boundary (3 columns per
    //    head, the slice last), the kv layout its rope tail (last column);
    //    the rotated neighbours are never read.
    gpu.elem().enqueue_rope(
        stream,
        &s.q,
        &s.cs_buf,
        rope,
        s.dims.q_cols as u32,
        1,
        &mut s.q_rope_all,
    )?;
    tick(i, obs, "rope(q)")?;
    gpu.elem().enqueue_rope(
        stream,
        &s.kv_a,
        &s.cs_buf,
        rope,
        s.dims.kv_cols as u32,
        1,
        &mut s.kv_s,
    )?;
    tick(i, obs, "rope(kv_a)")?;
    // 5. the latent norm overwrites `kv_s`'s head, leaving the rope tail in
    //    place: `kv_s` ends as `[kv_compressed | k_rope]`.
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.kv_a,
        f32_gain(w, &names.attn_kv_a_norm)?,
        mla.eps,
        latent,
        1,
        &mut s.kv_s,
    )?;
    tick(i, obs, "rms_norm(attn_kv_a_norm)")?;
    // 6. kvr = [k_rope | kv_compressed] — the oracle's CONCAT order.
    gather(&s.g_kvr, &s.kv_s, &mut s.kvr)?;
    tick(i, obs, "gather(kvr)")?;
    // 7. q_nope2 per head: one per-head launch dots every derived wk_b row
    //    of head h against q's nope slice of that head (x base h*kq_head,
    //    m = 1 per row), writing the nope2 span of flash row h directly
    //    (y base h*kv_width + rope).
    let (qn2_qs, qn2_d) = q8_derived(w, names.layer)?;
    step.enqueue_q8_0_gemv_heads(
        stream,
        qn2_qs,
        qn2_d,
        &s.q,
        latent,
        mla.kq_head,
        kv_width,
        rope,
        &mut s.f_rows,
    )?;
    tick(i, obs, "gemv_q8_0_heads(q_nope2)")?;
    // 8. the flash q rows `[q_rope | q_nope2]` per head — the rope spans are
    //    copies of q_rope_all's per-head rope slices (the nope2 span is
    //    already in place).
    gather(&s.g_f_rope_lo, &s.q_rope_all, &mut s.f_rows)?;
    tick(i, obs, "gather(f_rope_lo)")?;
    gather(&s.g_f_rope_hi, &s.q_rope_all, &mut s.f_rows)?;
    tick(i, obs, "gather(f_rope_hi)")?;
    // 9. append the kvr row at `pos`, then attend over `n_keys` rows — both
    //    read from device buffers, so a captured replay follows the position.
    gpu.flash()
        .enqueue_kv_append_pos_buf(stream, &s.kvr, &s.pos_buf, kv_l, 1)?;
    tick(i, obs, "kv_append(pos)")?;
    gpu.flash().enqueue_flash_latent(
        stream,
        &s.f_rows,
        kv_l,
        &s.n_keys_buf,
        mla.kq_scale,
        1,
        mla.n_head,
        rope,
        latent,
        &mut s.kqvc,
    )?;
    tick(i, obs, "flash_latent")?;
    // 10. wv_b per head: flash's output is head-major — kqvc column h is
    //     head h's compressed values, the m-column layout the quantizer
    //     consumes; the heads 8..15 half quantizes from kqvc's base offset
    //     (the quantizer's x base), so no gathered copy. Each per-head
    //     launch dots only its heads' wv_b rows (absolute row
    //     h*(nope+v_head) + nope + j, activation column h - head_base,
    //     m = 1), writing the flat, head-major kqv_2d directly — the wk_b
    //     rows are never read.
    gpu.enqueue_quantize_q8_1(&s.kqvc, &mut s.act_kv_lo)?;
    tick(i, obs, "quantize_q8_1(kqvc_lo)")?;
    gpu.enqueue_quantize_q8_1_at(&s.kqvc, (mla.n_head / 2) * latent, &mut s.act_kv_hi)?;
    tick(i, obs, "quantize_q8_1(kqvc_hi)")?;
    let kv_b = kq_weight(w, &names.attn_kv_b)?;
    step.enqueue_q3k_gemv_heads(
        stream,
        kv_b,
        &s.act_kv_lo,
        0,
        mla.v_head,
        mla.nope + mla.v_head,
        mla.nope,
        mla.v_head,
        &mut s.kqv_2d,
    )?;
    tick(i, obs, "gemv_q3k_heads(wv_b_lo)")?;
    step.enqueue_q3k_gemv_heads(
        stream,
        kv_b,
        &s.act_kv_hi,
        mla.n_head / 2,
        mla.v_head,
        mla.nope + mla.v_head,
        mla.nope,
        mla.v_head,
        &mut s.kqv_2d,
    )?;
    tick(i, obs, "gemv_q3k_heads(wv_b_hi)")?;
    // 11. attn_output over the flat kqv_2d, then the attention residual.
    gpu.enqueue_quantize_q8_1(&s.kqv_2d, &mut s.act_ao)?;
    tick(i, obs, "quantize_q8_1(act_ao)")?;
    gpu.enqueue_gemv_q4k(
        kq_weight(w, &names.attn_output)?,
        &s.act_ao,
        &mut s.attn_out,
    )?;
    tick(i, obs, "gemv_q4k(attn_output)")?;
    gpu.elem()
        .enqueue_add(stream, &s.attn_out, &s.x, hidden, &mut s.ffn_inp)?;
    tick(i, obs, "add(attn_resid)")?;
    Ok(())
}

/// Enqueue the dense FFN half: norm+quantize, gate·up·swiglu, 32-value
/// quantize, down+residual — bit-identical to the op path (the P0b
/// contract). `s.ffn_inp` in, `s.l_out` out.
#[allow(clippy::too_many_arguments)]
fn enqueue_ffn_dense(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut dyn FnMut(usize, &'static str) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let LayerScratch {
        ffn_inp,
        act_ffn,
        dense,
        l_out,
        ..
    } = s;
    let d = dense.as_mut().ok_or_else(|| -> GpuError {
        format!(
            "enqueue_ffn_dense: layer {} is dense but the stage carries no dense arena",
            names.layer
        )
        .into()
    })?;
    gpu.fused().enqueue_norm_quant(
        stream,
        ffn_inp,
        f32_gain(w, &names.ffn_norm)?,
        mla.eps,
        act_ffn,
    )?;
    tick(i, obs, "ffn_norm_quant")?;
    gpu.fused().enqueue_gate_up_swiglu(
        stream,
        kq_weight(w, &names.ffn_gate)?,
        kq_weight(w, &names.ffn_up)?,
        act_ffn,
        &mut d.h,
    )?;
    tick(i, obs, "ffn_gate_up_swiglu")?;
    gpu.q5().enqueue_quantize_q8(stream, &d.h, &mut d.act32)?;
    tick(i, obs, "ffn_quantize_q8")?;
    gpu.fused().enqueue_down_add_q5_1(
        stream,
        kq_weight(w, &names.ffn_down)?,
        &d.act32,
        ffn_inp,
        l_out,
    )?;
    tick(i, obs, "ffn_down_add")?;
    Ok(())
}

/// Enqueue the routed MoE FFN half at m = 1: `s.ffn_inp` in, `s.l_out` out.
///
/// The norm is split from the quantizer here, unlike the dense half's fused
/// `norm_quant`: the router eats the f32 normed vector and the experts eat
/// its q8_1 form, so both must exist. The routed experts run through the
/// device-resident `sel` (the router's own ids buffer at m = 1), so a
/// captured graph follows the routing; the shared expert runs the dense
/// fused kernels at its own width on the same quantized input. The combine
/// keeps the dump's grouping — `(Σ w·down + shexp) + resid`.
#[allow(clippy::too_many_arguments)]
fn enqueue_ffn_moe(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    mla: &MlaParams,
    dims: &MoeDims,
    i: &mut usize,
    obs: &mut dyn FnMut(usize, &'static str) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let LayerScratch {
        ffn_inp,
        act_ffn,
        moe,
        l_out,
        ..
    } = s;
    let m = moe.as_mut().ok_or_else(|| -> GpuError {
        format!(
            "enqueue_ffn_moe: layer {} routes but the stage carries no MoE arena",
            names.layer
        )
        .into()
    })?;
    // 1-2. ffn_norm in both forms the half's consumers need.
    gpu.elem().enqueue_rms_norm(
        stream,
        ffn_inp,
        f32_gain(w, &names.ffn_norm)?,
        mla.eps,
        hidden,
        1,
        &mut m.normed,
    )?;
    tick(i, obs, "moe_ffn_norm")?;
    gpu.enqueue_quantize_q8_1(&m.normed, act_ffn)?;
    tick(i, obs, "moe_quantize_q8_1")?;
    // 3-4. the router: an f32 gemv over the normed vector, then softmax +
    //      top-k + scale. `m.ids` is the `sel` every expert launch reads.
    gpu.q8f32().enqueue_f32_gemv(
        stream,
        f32_tensor(w, &names.ffn_gate_inp)?,
        &m.normed,
        1,
        &mut m.logits,
    )?;
    tick(i, obs, "moe_router_gemv")?;
    gpu.router().enqueue_router_topk(
        stream,
        &m.logits,
        1,
        dims.scale,
        &mut m.probs,
        &mut m.ids,
        &mut m.weights,
    )?;
    tick(i, obs, "moe_router_topk")?;
    // 5-7. the routed experts, all six per launch through `sel`.
    gpu.moe_fused().enqueue_expert_gate_up_swiglu(
        stream,
        kq_weight(w, &names.ffn_gate_exps)?,
        kq_weight(w, &names.ffn_up_exps)?,
        act_ffn,
        &m.ids,
        dims.n_used,
        dims.ff,
        &mut m.h_exp,
    )?;
    tick(i, obs, "moe_expert_gate_up_swiglu")?;
    gpu.q5()
        .enqueue_quantize_q8(stream, &m.h_exp, &mut m.act32_exp)?;
    tick(i, obs, "moe_expert_quantize_q8")?;
    gpu.q5().enqueue_gemv_q5_0_sel(
        stream,
        kq_weight(w, &names.ffn_down_exps)?,
        &m.act32_exp,
        &m.ids,
        dims.n_used,
        hidden,
        &mut m.down,
    )?;
    tick(i, obs, "moe_expert_down")?;
    // 8-10. the shared expert: the dense fused gate·up·swiglu at its own
    //       width on the same quantized input, then a Q4_K down projection
    //       whose K is the shared width (odd super-block count, which the
    //       Q4_K row geometry takes).
    gpu.fused().enqueue_gate_up_swiglu(
        stream,
        kq_weight(w, &names.ffn_gate_shexp)?,
        kq_weight(w, &names.ffn_up_shexp)?,
        act_ffn,
        &mut m.h_sh,
    )?;
    tick(i, obs, "shexp_gate_up_swiglu")?;
    gpu.enqueue_quantize_q8_1(&m.h_sh, &mut m.act_sh)?;
    tick(i, obs, "shexp_quantize_q8_1")?;
    gpu.enqueue_gemv_q4k(
        kq_weight(w, &names.ffn_down_shexp)?,
        &m.act_sh,
        &mut m.shexp,
    )?;
    tick(i, obs, "shexp_down")?;
    // 11. (Σ w·down + shexp) + resid, the dump's own grouping.
    gpu.moe_fused().enqueue_moe_combine(
        stream,
        &m.down,
        &m.weights,
        &m.shexp,
        ffn_inp,
        hidden,
        dims.n_used,
        l_out,
    )?;
    tick(i, obs, "moe_combine")?;
    Ok(())
}
