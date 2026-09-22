//! Fused block kernels (package P0b — filled by its track): the dense FFN
//! half as four launches, bit-identical to the eight-launch op path, plus the
//! two attention-chain fusions built on the same rule — `norm_quant`'s f32
//! side-output (the norm's two consumers from one launch) and the MLA key
//! path's `rope + rms_norm + gather + kv_append` as one launch.

use crate::GpuError;
use crate::cores::{q3_slot, q3k_row_dot, q4_slot, q6_slot, q8_quad};
use crate::elem::{
    RMS_THREADS, RMS_THREADS_U32, RMS_WARPS, rms_partial_sq, rms_scale, rms_warp_tree,
    rope_pair_core, silu_mul,
};
use crate::flash::f32_to_f16_bits;
use crate::launch_u32;
use crate::q5::{Q8Blocks32, q5_row_dot};
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

// Fusion rule (docs/gpu-design.md "P0b의 모양"): merge launches wherever
// there is NO true cross-thread dependency, keep a launch boundary where
// there is one — a grid-wide barrier, plus the cooperative launch it
// requires, costs more than the one node it saves. That is a measurement,
// not a belief: the `gap_*` arms of `gate_p8 --bench-kernels` price the
// barrier and the cooperative launch separately at the grids a fold here
// would use.
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

    /// rms_norm + 128-value q8_1 quantization, ONE [`RMS_THREADS`] block per
    /// column (token): phase A is `elem::rms_norm`'s body verbatim
    /// (`rms_partial_sq` per thread, the fixed butterfly per warp,
    /// `rms_warp_tree`, `rms_scale`), phase B is
    /// `kernels::q3k_quantize_q8_1`'s body per 128-value block with the
    /// normalized value computed in registers as `(scale · gain) · x` — the
    /// same expression and order `elem::rms_norm` stores — then the same
    /// block amax / scale / rounding / permuted stores, warp `w` taking
    /// blocks `w, w + RMS_WARPS, …`. One block owning the whole column is
    /// what removes the launch boundary: the 128-value amax needs the
    /// normalized values, the normalized values need the row's sum of
    /// squares, and both reductions close inside the block — no grid
    /// barrier. `k` a multiple of 128 (every `Q8Act` k is a multiple of
    /// 256); the column guard is block-uniform, so no barrier is skipped and
    /// every warp collective sees a full warp.
    ///
    /// `y` takes the f32 normalized vector as well — the same registers the
    /// quantizer consumes, so it holds `elem::rms_norm`'s store bit for bit.
    /// The attention norm's tap and the MoE router read it; a site with no
    /// f32 consumer passes a buffer it is about to overwrite anyway.
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
            x.len() >= k * m,
            gain.len() >= k,
            q3.len() >= m * 64 * half_it,
            q4.len() >= m * 256 * quad_it,
            q6.len() >= m * 128 * half_it,
            s8.len() >= m * 8 * n_sb,
            d8.len() >= m * 2 * n_sb,
            y.len() >= k * m
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
        mut y: DisjointSlice<f32>,
    ) {
        // One RMS_THREADS block per column: t is the COLUMN index (the block
        // index), not the thread id.
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_of = tid / 32;
        let k = k as usize;
        let n_sb = n_sb as usize;
        let base = t * k;

        // Phase A: the sum of squares, exactly `elem::rms_norm`'s cores and
        // reduction tree.
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        let part = warp::reduce_sum_f32(rms_partial_sq(x, base, k, tid));
        if lane == 0 {
            // SAFETY: warp_of < RMS_WARPS; one lane per warp writes its slot.
            unsafe {
                *ws.add(warp_of) = part;
            }
        }
        thread::sync_threads();
        // SAFETY: every slot was written above and is visible past the
        // barrier.
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
        let scale = rms_scale(rms_warp_tree(sums), k as u32, eps);

        // Phase B: the quantizer's body per 128-value block, reading the
        // raw x and gain at the quantizer's four-consecutive-values
        // geometry (the op path round-trips through the norm's store; the
        // recomputation reproduces those bits). A block belongs wholly to
        // one warp, so every collective below stays warp-uniform.
        let blocks = k / 128; // = 2 * n_sb
        let mut b = warp_of;
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
            // SAFETY: vb + 3 < base + k <= k*m <= y.len() by the launch
            // contract; each value of the column is written by exactly one
            // lane of exactly one warp.
            unsafe {
                *y.get_unchecked_mut(vb) = nv0;
                *y.get_unchecked_mut(vb + 1) = nv1;
                *y.get_unchecked_mut(vb + 2) = nv2;
                *y.get_unchecked_mut(vb + 3) = nv3;
            }
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

            b += RMS_WARPS;
        }
    }

    /// The MLA key path of one decode step as ONE launch: the latent norm,
    /// the rope of the key's rope tail, the `[k_rope | kv_compressed]`
    /// permutation and the f16 cache append, all off the single `kv_a` row
    /// of `latent + rope` f32. One [`RMS_THREADS`] block, m = 1.
    ///
    /// Bit identity with the four launches it replaces comes from running
    /// their bodies unchanged: the norm is `elem::rms_norm` at `k = latent`
    /// (`rms_partial_sq` on the same per-thread stride, the same warp
    /// butterfly, `rms_warp_tree`, `rms_scale`, the same
    /// `(scale · gain) · x` store), the tail is `elem::rope`'s
    /// `rope_pair_core` on the LAST 64-value column (the only column the
    /// chain reads back: the op path rotates all of them into `kv_s` and the
    /// norm immediately overwrites the first `latent`, so writing just these
    /// two spans leaves `kv_s` in the same state), the permutation is the
    /// `[k_rope | kv_compressed]` concat written from registers instead of
    /// through a gather's pair table, and the cache row is
    /// `flash::kv_append_pos_buf`'s `f32_to_f16_bits` at `pos_buf[0]`, with
    /// the same skip for a position at or past the cache's height.
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
            kv_a.len() >= latent + rope,
            gain.len() >= latent,
            cs.len() >= rope,
            pos_buf.len() >= 1,
            kv_s.len() >= latent + rope,
            kvr.len() >= latent + rope,
            cache.len() >= dst_rows * (latent + rope)
        )
    )]
    pub fn kv_norm_rope_append(
        kv_a: &[f32],
        gain: &[f32],
        cs: &[f32],
        pos_buf: &[u32],
        eps: f32,
        latent: u32,
        rope: u32,
        dst_rows: u32,
        mut kv_s: DisjointSlice<f32>,
        mut kvr: DisjointSlice<f32>,
        mut cache: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let k = latent as usize;
        let nd = rope as usize;
        let width = k + nd;

        // Phase A: `elem::rms_norm`'s sum of squares over the latent head.
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        let part = warp::reduce_sum_f32(rms_partial_sq(kv_a, 0, k, tid));
        if lane == 0 {
            // SAFETY: tid / 32 < RMS_WARPS; one lane per warp writes its slot.
            unsafe {
                *ws.add(tid / 32) = part;
            }
        }
        thread::sync_threads();
        // SAFETY: every slot was written above and is visible past the
        // barrier.
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
        let scale = rms_scale(rms_warp_tree(sums), latent, eps);

        // SAFETY: pos_buf.len() >= 1 by the launch contract.
        let pos = unsafe { *pos_buf.get_unchecked(0) } as usize;
        // The append's own bound: `pos` comes from device memory, so a row
        // at or past the cache's height is skipped, not clamped. Launch
        // uniform — every thread reads the same slot.
        let live = pos < dst_rows as usize;
        let crow = pos * width;

        // The latent half, on `elem::rms_norm`'s own loop order. Each value
        // lands in `kv_s` (the norm's output), at `kvr`'s permuted slot and,
        // rounded once, in the cache row.
        let mut it = tid;
        while it < k {
            // SAFETY: it < k bounds the gain and kv_a reads by the contract;
            // the kv_s slot is it < k < width <= kv_s.len(), the kvr slot is
            // nd + it < width <= kvr.len(), and the cache slot is
            // crow + nd + it < (pos + 1) * width <= dst_rows * width <=
            // cache.len() because `live` bounds pos below dst_rows.
            unsafe {
                let g = *gain.get_unchecked(it);
                let v = *kv_a.get_unchecked(it);
                let nv = (scale * g) * v;
                *kv_s.get_unchecked_mut(it) = nv;
                *kvr.get_unchecked_mut(nd + it) = nv;
                if live {
                    *cache.get_unchecked_mut(crow + nd + it) = f32_to_f16_bits(nv);
                }
            }
            it += RMS_THREADS;
        }

        // The rope tail: one thread per pair of the last 64-value column,
        // which `elem::rope` addresses as column `width/nd - 1` of token 0 —
        // so its cos/sin slots are `cs[d]`, `cs[d + 1]`.
        if 2 * tid < nd {
            let d = 2 * tid;
            // SAFETY: d + 1 < nd, so the kv_a reads stay below k + nd <=
            // kv_a.len(), the cs reads below nd <= cs.len(), the kv_s slots
            // below width, the kvr slots below nd <= width, and the cache
            // slots below (pos + 1) * width <= cache.len() under `live`.
            unsafe {
                let x0 = *kv_a.get_unchecked(k + d);
                let x1 = *kv_a.get_unchecked(k + d + 1);
                let c = *cs.get_unchecked(d);
                let s = *cs.get_unchecked(d + 1);
                let (y0, y1) = rope_pair_core(x0, x1, c, s);
                *kv_s.get_unchecked_mut(k + d) = y0;
                *kv_s.get_unchecked_mut(k + d + 1) = y1;
                *kvr.get_unchecked_mut(d) = y0;
                *kvr.get_unchecked_mut(d + 1) = y1;
                if live {
                    *cache.get_unchecked_mut(crow + d) = f32_to_f16_bits(y0);
                    *cache.get_unchecked_mut(crow + d + 1) = f32_to_f16_bits(y1);
                }
            }
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
    /// `gain`/`eps` into `act`, and the f32 normed vector into `y` — the
    /// same six buffers `elem::rms_norm` followed by
    /// `Gpu::enqueue_quantize_q8_1` produce, bit for bit. `y` holds
    /// `act.m() * act.k()` f32; a caller with no f32 consumer passes a
    /// buffer of that width it overwrites later in the chain. `act.k()`
    /// must be a multiple of 128 (every `Q8Act` k is a multiple of 256).
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_norm_quant(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        act: &mut Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (m, k, n_sb) = (act.m(), act.k(), act.n_sb());
        if k % 128 != 0 {
            return Err(GpuError::shape(
                "enqueue_norm_quant",
                format!(
                    "k must be a multiple of 128 (one q8_1 \
                 block per four lanes), got {k}"
                ),
            ));
        }
        if x.len() < m * k || gain.len() < k || y.len() < m * k {
            return Err(GpuError::shape(
                "enqueue_norm_quant",
                format!(
                    "x.len() {} (need m*k = {mk}), gain.len() {gl} (need {k}), \
                 y.len() {yl} (need {mk})",
                    x.len(),
                    mk = m * k,
                    gl = gain.len(),
                    yl = y.len()
                ),
            ));
        }
        let what = "enqueue_norm_quant";
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_norm_quant(LaunchConfig1D::new(m, RMS_THREADS_U32, 0))?;
        self.module.norm_quant(
            stream,
            &prep,
            x,
            gain,
            eps,
            k,
            m,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the MLA key path of one decode step: `kv_a` (the
    /// `latent + rope` f32 projection) normed by `gain`/`eps` over its
    /// latent head and roped over its rope tail into `kv_s`
    /// (`[kv_compressed | k_rope]`), permuted into `kvr`
    /// (`[k_rope | kv_compressed]`) and appended as f16 at row
    /// `pos_buf[0]` of `cache` — the same three buffers
    /// `elem::rope` + `elem::rms_norm` + the `kvr` gather +
    /// `FlashKernels::enqueue_kv_append_pos_buf` produce, bit for bit.
    /// `cache` rows are `latent + rope` wide; a position at or past its
    /// height leaves the cache untouched. m = 1. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_kv_norm_rope_append(
        &self,
        stream: &CudaStream,
        kv_a: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        cs: &DeviceBuffer<f32>,
        pos_buf: &DeviceBuffer<u32>,
        eps: f32,
        latent: usize,
        rope: usize,
        kv_s: &mut DeviceBuffer<f32>,
        kvr: &mut DeviceBuffer<f32>,
        cache: &mut DeviceTensor<u16>,
    ) -> Result<(), GpuError> {
        let width = latent + rope;
        if latent == 0 || !latent.is_multiple_of(32) {
            return Err(GpuError::shape(
                "enqueue_kv_norm_rope_append",
                format!(
                    "the norm's geometry needs a positive multiple of \
                 32, got latent {latent}"
                ),
            ));
        }
        if rope < 2 || !rope.is_multiple_of(2) || rope > 2 * RMS_THREADS {
            return Err(GpuError::shape(
                "enqueue_kv_norm_rope_append",
                format!(
                    "the rope tail is one pair per thread of the one \
                 {RMS_THREADS}-thread block, so rope must be even and at most {}, got {rope}",
                    2 * RMS_THREADS
                ),
            ));
        }
        if cache.cols() != width {
            return Err(GpuError::shape(
                "enqueue_kv_norm_rope_append",
                format!(
                    "cache rows are {} wide, the kvr row is \
                 latent + rope = {width}",
                    cache.cols()
                ),
            ));
        }
        if kv_a.len() < width || gain.len() < latent || cs.len() < rope {
            return Err(GpuError::shape(
                "enqueue_kv_norm_rope_append",
                format!(
                    "kv_a.len() {} (need {width}), gain.len() {} \
                 (need {latent}), cs.len() {} (need {rope})",
                    kv_a.len(),
                    gain.len(),
                    cs.len()
                ),
            ));
        }
        if kv_s.len() < width || kvr.len() < width {
            return Err(GpuError::shape(
                "enqueue_kv_norm_rope_append",
                format!(
                    "kv_s.len() {} and kvr.len() {} vs the row width \
                 {width}",
                    kv_s.len(),
                    kvr.len()
                ),
            ));
        }
        if pos_buf.is_empty() {
            return Err(GpuError::shape(
                "enqueue_kv_norm_rope_append",
                "pos_buf must hold 1 u32",
            ));
        }
        let what = "enqueue_kv_norm_rope_append";
        let latent = launch_u32(what, "latent", latent)?;
        let rope = launch_u32(what, "rope", rope)?;
        let rows = launch_u32(what, "cache.rows()", cache.rows())?;
        let prep = self
            .module
            .prepare_kv_norm_rope_append(LaunchConfig1D::new(1, RMS_THREADS_U32, 0))?;
        self.module.kv_norm_rope_append(
            stream,
            &prep,
            kv_a,
            gain,
            cs,
            pos_buf,
            eps,
            latent,
            rope,
            rows,
            kv_s,
            kvr,
            cache.buf_mut(),
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
            return Err(GpuError::shape(
                "enqueue_gate_up_swiglu",
                format!("m = 1 only (the decode shape), got act.m() = {}", act.m()),
            ));
        }
        if !n_sb.is_multiple_of(2) {
            return Err(GpuError::shape(
                "enqueue_gate_up_swiglu",
                format!(
                    "odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                    act.k()
                ),
            ));
        }
        if wg.cols() != 110 * n_sb / 4 || wu.cols() != 110 * n_sb / 4 {
            return Err(GpuError::shape(
                "enqueue_gate_up_swiglu",
                format!(
                    "Q3_K rows are 110*{n_sb}/4 = {} words at K={}, got \
                 gate {} x {}, up {} x {}",
                    110 * n_sb / 4,
                    act.k(),
                    wg.rows(),
                    wg.cols(),
                    wu.rows(),
                    wu.cols()
                ),
            ));
        }
        if wg.rows() != wu.rows() {
            return Err(GpuError::shape(
                "enqueue_gate_up_swiglu",
                format!("gate rows {} != up rows {}", wg.rows(), wu.rows()),
            ));
        }
        let n_rows = wg.rows();
        if n_rows == 0 {
            return Err(GpuError::shape("enqueue_gate_up_swiglu", "empty weight"));
        }
        if h.len() < n_rows {
            return Err(GpuError::shape(
                "enqueue_gate_up_swiglu",
                format!("h.len() {} < rows {n_rows}", h.len()),
            ));
        }
        let what = "enqueue_gate_up_swiglu";
        let n_rows = launch_u32(what, "rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self.module.prepare_gate_up_swiglu_q3k(LaunchConfig1D::new(
            n_rows.div_ceil(8),
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
            n_rows,
            n_sb,
            n_sb.div_ceil(2),
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
            return Err(GpuError::shape(
                "enqueue_down_add_q5_1",
                format!("m = 1 only (the decode shape), got act.m() = {}", act.m()),
            ));
        }
        if w.cols() != act.q_stride() + 2 * k_blocks {
            return Err(GpuError::shape(
                "enqueue_down_add_q5_1",
                format!(
                    "Q5_1 row is q_stride + 2*k/32 = {} words, got cols {}",
                    act.q_stride() + 2 * k_blocks,
                    w.cols()
                ),
            ));
        }
        let n_rows = w.rows();
        if n_rows == 0 {
            return Err(GpuError::shape("enqueue_down_add_q5_1", "empty weight"));
        }
        if resid.len() < n_rows || y.len() < n_rows {
            return Err(GpuError::shape(
                "enqueue_down_add_q5_1",
                format!(
                    "resid.len() {} and y.len() {} vs rows {n_rows}",
                    resid.len(),
                    y.len()
                ),
            ));
        }
        let what = "enqueue_down_add_q5_1";
        let k_blocks = launch_u32(what, "k_blocks", k_blocks)?;
        let q_stride = launch_u32(what, "q_stride", act.q_stride())?;
        let n_rows = launch_u32(what, "rows", n_rows)?;
        let prep =
            self.module
                .prepare_down_add_q5_1(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.down_add_q5_1(
            stream,
            &prep,
            w.buf(),
            &act.q,
            &act.d8,
            &act.s8,
            resid,
            k_blocks,
            q_stride,
            0,
            n_rows,
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
