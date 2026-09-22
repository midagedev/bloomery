//! A `#[cuda_module]` in a package that is NOT the package the device-side
//! cores live in. The two entries here are verbatim copies of
//! `bloomery_gpu`'s `q3k_quantize_q8_1` and `q3k_gemv_sel`: same launch
//! contracts, same bodies, same argument order. The entries carry an `xc_`
//! prefix because a kernel's host-side symbol is keyed by its entry name
//! alone — two packages declaring the same name do not link. The other
//! difference is where
//! the bodies they call are defined — `bloomery_gpu::q8_1_quant_block` and
//! `bloomery_gpu::cores::q3k_row_dot` are reached across the crate boundary,
//! which is the question this package exists to answer. Both are
//! `#[inline(always)]`, so their MIR travels in the rlib and the codegen
//! backend has a body to emit here.
//!
//! Because the bodies are the same and the accumulation order is the same,
//! this module's outputs must be bit-identical to the in-crate entries' on
//! the same input, and its PTX must be the same instructions.

use bloomery_gpu::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod xcrate_kernels {
    use super::*;
    use bloomery_gpu::cores::q3k_row_dot;
    use bloomery_gpu::q8_1_quant_block;

    /// Copy of `bloomery_gpu`'s `q3k_quantize_q8_1`. The whole body is the
    /// cross-crate `q8_1_quant_block`, so the bytes it writes are the bytes
    /// that entry writes.
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
            x.len() >= x0 + m_cols * 256 * n_sb,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn xc_q3k_quantize_q8_1(
        x: &[f32],
        x0: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
    ) {
        // One 32-thread block per 128-value quant block: blk is the BLOCK
        // index (global tid / 32), not the thread id.
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb; // 128-value blocks per column
        let total = m_cols as usize * blocks_per_col;
        if blk >= total {
            return;
        }
        let col = blk / blocks_per_col;
        let b = blk % blocks_per_col;
        let lane = warp::lane_id() as usize;
        // SAFETY: col < m_cols and b < 2*n_sb by the two lines above; the
        // launch contract carries the rest of `q8_1_quant_block`'s
        // preconditions, and the block index is warp-uniform.
        q8_1_quant_block(
            x,
            x0 as usize,
            col,
            b,
            n_sb,
            half_it,
            quad_it,
            lane,
            &mut q3,
            &mut q4,
            &mut q6,
            &mut s8,
            &mut d8,
        );
    }

    /// Copy of `bloomery_gpu`'s `q3k_gemv_sel`, the simplest m = 1 Q3_K gemv
    /// entry whose per-row body is `cores::q3k_row_dot` called with
    /// `m_cols = 1`. Thread geometry, guards, reduction and store are the
    /// in-crate entry's; only the path to the core differs.
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
            4 * w.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn xc_q3k_gemv_sel(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the out-of-range return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        // The m = 1 per-row body is `bloomery_gpu::cores::q3k_row_dot`, the
        // same core the in-crate entry calls.
        let f = q3k_row_dot(w, q, d8, n_sb as usize, iters, row_abs, 0, 1, lane);
        let s0 = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: row < n_slots*rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }
}

/// q8_1 activation scratch owned by this package, the shape
/// `bloomery_gpu::Q8Act` has for `m` columns of `k = 256 * n_sb` values.
/// It exists because that type's buffers are crate-private: the spike has to
/// hold its own to launch its own copies against them.
pub struct SpikeAct {
    pub q3: DeviceBuffer<u64>,
    pub q4: DeviceBuffer<u32>,
    pub q6: DeviceBuffer<u32>,
    pub s8: DeviceBuffer<i32>,
    pub d8: DeviceBuffer<f32>,
    pub m: usize,
    pub n_sb: usize,
}

impl SpikeAct {
    /// Allocate scratch for `m` columns of `256 * n_sb` values. Load-time
    /// only.
    pub fn new(stream: &CudaStream, m: usize, n_sb: usize) -> Result<SpikeAct, GpuError> {
        Ok(SpikeAct {
            q3: DeviceBuffer::zeroed(stream, m * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, m * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, m * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, m * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, m * 2 * n_sb)?,
            m,
            n_sb,
        })
    }
}

/// This package's device bundle, loaded into a context that another
/// package's bundle is already loaded into.
pub struct XcrateKernels {
    module: xcrate_kernels::LoadedModule,
}

impl XcrateKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<XcrateKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher below checks its launch contract.
        let module = unsafe { xcrate_kernels::load(ctx)? };
        Ok(XcrateKernels { module })
    }

    /// Enqueue this package's copy of the q8_1 quantizer over `x` into
    /// `act`, with the launch geometry `bloomery_gpu` uses for the same
    /// entry (one 32-thread block per 128-value quant block).
    pub fn enqueue_quantize(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        act: &mut SpikeAct,
    ) -> Result<(), GpuError> {
        let (m, n_sb) = (act.m as u32, act.n_sb as u32);
        let prep = self
            .module
            .prepare_xc_q3k_quantize_q8_1(LaunchConfig1D::new(m * n_sb * 2, 32, 0))?;
        self.module.xc_q3k_quantize_q8_1(
            stream,
            &prep,
            x,
            0,
            m,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }

    /// Enqueue this package's copy of `q3k_gemv_sel` over one expert
    /// (`n_experts = n_slots = 1`, `rows_per_expert = rows`), the geometry
    /// `bloomery_gpu::Gpu::enqueue_gemv_q3k_sel` launches.
    pub fn enqueue_gemv_sel(
        &self,
        stream: &CudaStream,
        w: &DeviceBuffer<u32>,
        act: &SpikeAct,
        sel: &DeviceBuffer<u32>,
        rows: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_sb = act.n_sb as u32;
        let rows = rows as u32;
        let prep =
            self.module
                .prepare_xc_q3k_gemv_sel(LaunchConfig1D::new(rows.div_ceil(8), 256, 0))?;
        self.module.xc_q3k_gemv_sel(
            stream,
            &prep,
            w,
            &act.q3,
            &act.d8,
            sel,
            1,
            rows,
            1,
            n_sb,
            n_sb.div_ceil(2),
            y,
        )?;
        Ok(())
    }
}
