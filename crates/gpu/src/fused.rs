//! Fused block kernels (package P0b — filled by its track): the dense FFN
//! half as four launches, bit-identical to the eight-launch op path.

use crate::GpuError;
use crate::cores::{q3_slot, q3k_row_dot, q4_slot, q6_slot, q8_quad};
use crate::elem::{rms_scale, silu_mul};
use crate::q5::{Q8Blocks32, q5_row_dot};
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

// Fusion rule (docs/gpu-design.md "P0b의 모양"): merge launches wherever
// there is NO true cross-thread dependency, keep a launch boundary where
// there is one — a grid-wide barrier costs more than the nodes it saves.
// The op path's eight launches:
//   rms_norm -> quantize q8_1(128) -> gate gemv -> up gemv -> swiglu
//   -> quantize q8(32) -> down gemv -> residual add
// The two boundaries that stay: a gemv reads the WHOLE quantized input, and
// the down gemv reads ALL 342 32-value blocks of the 10944 intermediates
// (one block spans 32 rows = 32 warps = 4 CUDA blocks, so it cannot close
// inside the gate/up launch). Everything else merges:
//   [norm+quantize] [gate·up·swiglu] [32-value quantize] [down+residual]
// m = 1 (decode) only. The arithmetic bodies are the cores the op-path
// kernels run (`cores::q3k_row_dot`, `elem::{rms_scale, silu_mul}`,
// `q5::q5_row_dot`, the quantizer tail), so the gate's contract with the op
// path is bit identity, not a band.

#[cuda_module]
mod fused_kernels {
    use super::*;

    /// rms_norm + 128-value q8_1 quantization, ONE warp per column (token):
    /// phase A is `elem::rms_norm`'s body verbatim (lane-strided partial
    /// sums of squares over the whole row, the fixed butterfly, `rms_scale`),
    /// phase B is `kernels::q3k_quantize_q8_1`'s body per 128-value block
    /// with the normalized value computed in registers as `(scale · gain) ·
    /// x` — the same expression and order `elem::rms_norm` stores — then the
    /// same block amax / scale / rounding / permuted stores. One warp owning
    /// the whole row is what removes the launch boundary: the block amax
    /// needs the normalized values, the normalized values need the row's sum
    /// of squares, and both reductions close inside the warp — no grid
    /// barrier. `k` a multiple of 128 (every `Q8Act` k is a multiple of
    /// 256); the column guard is warp-uniform, so every collective sees a
    /// full warp.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= k * m,
            gain.len() >= k,
            q3.len() >= m * 64 * half_it,
            q4.len() >= m * 256 * quad_it,
            q6.len() >= m * 128 * half_it,
            s8.len() >= m * 8 * n_sb,
            d8.len() >= m * 2 * n_sb
        )
    )]
    pub fn norm_quant(
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        m: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
    ) {
        // One 32-thread block per column: t is the COLUMN index (global
        // tid / 32), not the thread id.
        let t = thread::index_1d().get() / 32;
        if t >= m as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let k = k as usize;
        let n_sb = n_sb as usize;
        let base = t * k;

        // Phase A: the sum of squares, exactly `elem::rms_norm`'s loop and
        // reduction tree.
        let mut acc = 0.0f32;
        let mut it = lane;
        while it < k {
            // SAFETY: it < k <= x.len() - base by the launch contract.
            let v = unsafe { *x.get_unchecked(base + it) };
            acc += v * v;
            it += 32;
        }
        let scale = rms_scale(warp::reduce_sum_f32(acc), k as u32, eps);

        // Phase B: the quantizer's body per 128-value block, reading the
        // raw x and gain at the quantizer's four-consecutive-values
        // geometry (the op path round-trips through the norm's store; the
        // recomputation reproduces those bits).
        let blocks = k / 128; // = 2 * n_sb
        let mut b = 0usize;
        while b < blocks {
            let vb = base + 128 * b + 4 * lane;
            // SAFETY: vb + 3 < base + k <= x.len() and the gain reads stay
            // below 128*b + 4*lane + 3 < k <= gain.len() by the launch
            // contract.
            let (v0, v1, v2, v3, gn0, gn1, gn2, gn3) = unsafe {
                (
                    *x.get_unchecked(vb),
                    *x.get_unchecked(vb + 1),
                    *x.get_unchecked(vb + 2),
                    *x.get_unchecked(vb + 3),
                    *gain.get_unchecked(128 * b + 4 * lane),
                    *gain.get_unchecked(128 * b + 4 * lane + 1),
                    *gain.get_unchecked(128 * b + 4 * lane + 2),
                    *gain.get_unchecked(128 * b + 4 * lane + 3),
                )
            };
            // `elem::rms_norm`'s store expression, per value.
            let nv0 = (scale * gn0) * v0;
            let nv1 = (scale * gn1) * v1;
            let nv2 = (scale * gn2) * v2;
            let nv3 = (scale * gn3) * v3;
            let amax = warp::reduce_max_f32(nv0.abs().max(nv1.abs()).max(nv2.abs()).max(nv3.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let (word, quad) = q8_quad([nv0, nv1, nv2, nv3], d);

            // v4 = this word's index in value order within the column (the
            // quantizer's tie: the word covers values 128b + 4*lane .. +3).
            let v4 = (32 * b + lane) as u32;
            // Q3_K u64 pairing: the two fields of a gemv load PAIR (j, j^1)
            // are always held by quantize lanes lane and lane^8 of one
            // block, so all lanes run the collective shuffle and the
            // bit3-clear half stores.
            let g3 = q3_slot(v4);
            let partner = warp::shuffle_xor(word, 8);
            let cb = t * 64 * half_it as usize;
            let p4 = q4_slot(v4);
            let p6 = q6_slot(v4);
            let cu4 = t * 256 * quad_it as usize;
            let cu6 = t * 128 * half_it as usize;
            // SAFETY: q3_slot < 64*half_it, q4_slot < 256*quad_it and
            // q6_slot < 128*half_it per column (permutations of the column's
            // value words onto its group slots, host-verified bijections);
            // the three stores hit three distinct buffers, bit3-clear lanes
            // of a block write disjoint u64 positions, every lane its own
            // u32 position.
            unsafe {
                if lane & 8 == 0 {
                    *q3.get_unchecked_mut(cb + g3 as usize) =
                        (word as u64) | ((partner as u64) << 32);
                }
                *q4.get_unchecked_mut(cu4 + p4 as usize) = word;
                *q6.get_unchecked_mut(cu6 + p6 as usize) = word;
            }

            // 32-value-group signed sums: butterfly over the lane-local
            // quad sums (masks 1, 2, 4); lanes 8k write group 4b + k.
            let mut g = quad;
            g += warp::shuffle_xor(g as u32, 1) as i32;
            g += warp::shuffle_xor(g as u32, 2) as i32;
            g += warp::shuffle_xor(g as u32, 4) as i32;
            if lane & 7 == 0 {
                // SAFETY: group index 4b + lane/8 < 8*n_sb per column; s8
                // holds m*8*n_sb words and one lane writes each group.
                unsafe {
                    *s8.get_unchecked_mut(t * 8 * n_sb + 4 * b + (lane >> 3)) = g;
                }
            }
            if lane == 0 {
                // SAFETY: lane 0 of each warp writes its own d8 slot.
                unsafe {
                    *d8.get_unchecked_mut(t * 2 * n_sb + b) = d;
                }
            }

            b += 1;
        }
    }

    /// `h[r] = silu(gate_r · act) · (up_r · act)`, one warp per output row:
    /// `cores::q3k_row_dot` twice — the same body `q3k_gemv` and
    /// `q3k_gemv_sel` run — reduced with the same fixed warp tree, then the
    /// `elem::silu_mul` core on the two row dots. A row's output depends
    /// only on its own two weight rows and the shared quantized input, so
    /// gate·up·swiglu closes in the warp that owns the row. m = 1: every
    /// row dots the ONE quantized column of `q`/`d8` (gate and up read the
    /// same input). Weights as `enqueue_gemv_q3k` (rows of `110 * n_sb / 4`
    /// u32 words, even n_sb), gate and up of the same row count.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * wg.len() >= n_rows * 110 * n_sb,
            4 * wu.len() >= n_rows * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            h.len() >= n_rows
        )
    )]
    pub fn gate_up_swiglu_q3k(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row, 0, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row, 0, 1, lane);
        // The op path's per-row values: each gemv reduces its lane partials
        // with the fixed warp tree, then swiglu combines the two row dots.
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            // SAFETY: only lane 0 writes; warp `row` owns h[row].
            unsafe {
                *h.get_unchecked_mut(row) = silu_mul(g, u);
            }
        }
    }

    /// `y[row] = down_row(row0 + row) · act + resid[row]`, one warp per
    /// output row: `q5::q5_row_dot` (the body `q5_1_gemv` runs) against the
    /// ONE quantized column, the warp-summed dot, and the residual folded
    /// into the store as the same single f32 add `elem::add` performs —
    /// a = the down dot (the op path's ffn_out operand), b = `resid` (its
    /// ffn_inp operand). Weight layout as `enqueue_gemv_q5_1`
    /// (`q_stride + 2*k_blocks` words per row). m = 1.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= (row0 + n_rows) * (q_stride + 2 * k_blocks),
            q.len() >= q_stride,
            d8.len() >= k_blocks,
            s8.len() >= k_blocks,
            resid.len() >= n_rows,
            y.len() >= n_rows
        )
    )]
    pub fn down_add_q5_1(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        s8: &[i32],
        resid: &[f32],
        k_blocks: u32,
        q_stride: u32,
        row0: u32,
        n_rows: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let f = q5_row_dot(
            w,
            q,
            d8,
            s8,
            k_blocks as usize,
            q_stride as usize,
            row0 as usize + row,
            0,
            1,
            lane,
            true,
        );
        // `q5_1_gemv`'s m = 1 reduction is this one warp tree over f[0].
        let a = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: row < n_rows bounds the resid read and the y store by
            // the launch contract; only lane 0 of the warp writes y[row].
            unsafe {
                let b = *resid.get_unchecked(row);
                *y.get_unchecked_mut(row) = a + b;
            }
        }
    }
}

/// The loaded P0b device module and its enqueue API. Owns no context and no
/// stream — every enqueue takes the engine stream (`Gpu::stream()`), so
/// launches order with the rest of the step and are capturable.
pub struct FusedKernels {
    module: fused_kernels::LoadedModule,
}

impl FusedKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FusedKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { fused_kernels::load(ctx)? };
        Ok(FusedKernels { module })
    }

    /// Enqueue rms_norm + 128-value q8_1 quantization of `x` (`act.m()`
    /// columns of `act.k()` f32, token-major, one warp per column) by
    /// `gain`/`eps` into `act` — the same five buffers `elem::rms_norm`
    /// followed by `Gpu::enqueue_quantize_q8_1` produce, bit for bit.
    /// `act.k()` must be a multiple of 128 (every `Q8Act` k is a multiple
    /// of 256). Asynchronous, allocation-free, capturable.
    pub fn enqueue_norm_quant(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let (m, k, n_sb) = (act.m(), act.k(), act.n_sb());
        if k % 128 != 0 {
            return Err(format!(
                "enqueue_norm_quant: k must be a multiple of 128 (one q8_1 \
                 block per four lanes), got {k}"
            )
            .into());
        }
        if x.len() < m * k || gain.len() < k {
            return Err(format!(
                "enqueue_norm_quant: x.len() {} (need m*k = {mk}), gain.len() {gl} (need {k})",
                x.len(),
                mk = m * k,
                gl = gain.len()
            )
            .into());
        }
        let prep = self
            .module
            .prepare_norm_quant(LaunchConfig1D::new(m as u32, 32, 0))?;
        self.module.norm_quant(
            stream,
            &prep,
            x,
            gain,
            eps,
            k as u32,
            m as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            n_sb.div_ceil(4) as u32,
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }

    /// Enqueue `h[r] = silu(gate_r · act) · (up_r · act)` for the m = 1
    /// decode shape: `wg`/`wu` are the gate/up weights (Q3_K, `w.rows()`
    /// rows of `110 * n_sb / 4` u32 words, even n_sb — as
    /// `Gpu::enqueue_gemv_q3k` takes), both of the SAME row count, dotted
    /// against the ONE quantized column of `act`. `h` holds `rows` f32.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up_swiglu(
        &self,
        stream: &CudaStream,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
        act: &Q8Act,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_sb = act.n_sb();
        if act.m() != 1 {
            return Err(format!(
                "enqueue_gate_up_swiglu: m = 1 only (the decode shape), got act.m() = {}",
                act.m()
            )
            .into());
        }
        if n_sb % 2 != 0 {
            return Err(format!(
                "enqueue_gate_up_swiglu: odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                act.k()
            )
            .into());
        }
        if wg.cols() != 110 * n_sb / 4 || wu.cols() != 110 * n_sb / 4 {
            return Err(format!(
                "enqueue_gate_up_swiglu: Q3_K rows are 110*{n_sb}/4 = {} words at K={}, got \
                 gate {} x {}, up {} x {}",
                110 * n_sb / 4,
                act.k(),
                wg.rows(),
                wg.cols(),
                wu.rows(),
                wu.cols()
            )
            .into());
        }
        if wg.rows() != wu.rows() {
            return Err(format!(
                "enqueue_gate_up_swiglu: gate rows {} != up rows {}",
                wg.rows(),
                wu.rows()
            )
            .into());
        }
        let n_rows = wg.rows();
        if n_rows == 0 {
            return Err("enqueue_gate_up_swiglu: empty weight".into());
        }
        if h.len() < n_rows {
            return Err(format!(
                "enqueue_gate_up_swiglu: h.len() {} < rows {n_rows}",
                h.len()
            )
            .into());
        }
        let prep = self.module.prepare_gate_up_swiglu_q3k(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.gate_up_swiglu_q3k(
            stream,
            &prep,
            wg.buf(),
            wu.buf(),
            &act.q3,
            &act.d8,
            n_rows as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            h,
        )?;
        Ok(())
    }

    /// Enqueue `y[row] = w_row(row0 + row) · act + resid[row]` (Q5_1, the
    /// down projection with the residual folded into the store): `w` packed
    /// by `q5::pack_q5_1` with `cols = q_stride + 2*k/32` over the whole
    /// flat stack (`row0` reaches experts without a gather copy), `act` the
    /// ONE quantized 32-value-block column (m = 1), `resid`/`y` `n_rows`
    /// f32 each. Asynchronous, allocation-free, capturable.
    pub fn enqueue_down_add_q5_1(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Blocks32,
        resid: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let k_blocks = act.k() / 32;
        if act.m() != 1 {
            return Err(format!(
                "enqueue_down_add_q5_1: m = 1 only (the decode shape), got act.m() = {}",
                act.m()
            )
            .into());
        }
        if w.cols() != act.q_stride() + 2 * k_blocks {
            return Err(format!(
                "enqueue_down_add_q5_1: Q5_1 row is q_stride + 2*k/32 = {} words, got cols {}",
                act.q_stride() + 2 * k_blocks,
                w.cols()
            )
            .into());
        }
        let n_rows = w.rows();
        if n_rows == 0 {
            return Err("enqueue_down_add_q5_1: empty weight".into());
        }
        if resid.len() < n_rows || y.len() < n_rows {
            return Err(format!(
                "enqueue_down_add_q5_1: resid.len() {} and y.len() {} vs rows {n_rows}",
                resid.len(),
                y.len()
            )
            .into());
        }
        let prep = self.module.prepare_down_add_q5_1(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.down_add_q5_1(
            stream,
            &prep,
            w.buf(),
            &act.q,
            &act.d8,
            &act.s8,
            resid,
            k_blocks as u32,
            act.q_stride() as u32,
            0,
            n_rows as u32,
            y,
        )?;
        Ok(())
    }
}

/// The five buffers of a `Q8Act` on the host — the gate's step-1 comparison
/// between the op and fused paths (`Q8Act`'s fields are crate-private).
/// Diagnostic readback: synchronizes, so load-time/gate use only, never
/// inside a graph capture.
pub struct Q8ActHost {
    pub q3: Vec<u64>,
    pub q4: Vec<u32>,
    pub q6: Vec<u32>,
    pub s8: Vec<i32>,
    pub d8: Vec<f32>,
}

/// Read a `Q8Act`'s buffers back to the host. Synchronizes `stream`.
pub fn readback_q8act(stream: &CudaStream, act: &Q8Act) -> Result<Q8ActHost, GpuError> {
    Ok(Q8ActHost {
        q3: act.q3.to_host_vec(stream)?,
        q4: act.q4.to_host_vec(stream)?,
        q6: act.q6.to_host_vec(stream)?,
        s8: act.s8.to_host_vec(stream)?,
        d8: act.d8.to_host_vec(stream)?,
    })
}
