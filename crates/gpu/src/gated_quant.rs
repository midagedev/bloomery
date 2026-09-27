//! The q8_1 quantizer of an attention output under a sigmoid gate: the input
//! of a gated attention layer's output projection (Qwen3.6's full-attention
//! layers), `x = attn · sigmoid(g)` quantized in the one launch the plain
//! input would take.
//!
//! `attn` is the attention's output, `k` values a column (token); `g` is read
//! at stride from the q+gate projection's rows, where each head's gate sits
//! next to its query: value `i` of column `c`, head `i / head` and dim `i %
//! head`, takes its gate from `g[c·col_stride + (i / head)·head_stride +
//! offset + i % head]` ([`GateLayout`]). Each value is `attn · sigmoid(g)` in
//! f32, `sigmoid` = `route_core::sigmoid` (`1 / (1 + e^-g)`), and the 128-value
//! blocks go through `q8_1_quant_vals`, the plain quantizer's rounding and
//! layout, unchanged. So a column's bytes are the plain quantizer's on the
//! product: bit for bit wherever the device's `sigmoid` equals the host's
//! (`g` of 0 or of large magnitude, where it is 1/2, 1 or 0), within one code
//! elsewhere.
//!
//! A gate that is not finite has no defined product: its value is taken as
//! NaN, so its block is refused as any non-finite block is — a NaN scale,
//! zero codes and sums — and [`FaultSite::QuantColumn`] is raised on the
//! caller's sink. A non-finite product (a non-finite `attn`) is refused the
//! same way by the quantizer itself.

use crate::fault::{FaultSink, FaultSite};
use crate::gemm::GemmAct;
use crate::route_core::sigmoid;
use crate::tensor::Q8Act;
use crate::{GpuError, launch_u32, q8_1_quant_vals};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Where each value's gate sits in the gate rows: head `h` of column `c`
/// starts at `c·col_stride + h·head_stride + offset`, `head` values a head.
/// Qwen3.6's q+gate rows: `head` 256, `head_stride` 512, `offset` 256,
/// `col_stride` 16 · 512.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateLayout {
    pub head: usize,
    pub head_stride: usize,
    pub offset: usize,
    pub col_stride: usize,
}

#[cuda_module]
mod gated_quant_kernels {
    use super::*;

    /// One 32-thread block per 128-value block of a column, as
    /// `q3k_quantize_q8_1`: block `blk` is column `blk / (2·n_sb)`, block
    /// `blk % (2·n_sb)`, lane `l` its values `4l .. +3`. A 128-value block lies
    /// in one head (`head` a multiple of 128, host-checked), so its gates are
    /// four consecutive values per lane. The launch contract bounds the gate
    /// rows at `m_cols` whole columns; the host checks that the heads of a
    /// column fit in one (`(k/head − 1)·head_stride + offset + head <=
    /// col_stride`).
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
            x.len() >= m_cols * 256 * n_sb,
            g.len() >= m_cols * col_stride,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn q8_1_quantize_gated(
        x: &[f32],
        g: &[f32],
        head: u32,
        head_stride: u32,
        offset: u32,
        col_stride: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb;
        if blk >= m_cols as usize * blocks_per_col {
            return; // block-uniform
        }
        let col = blk / blocks_per_col;
        let b = blk % blocks_per_col;
        let lane = warp::lane_id() as usize;
        let first = 128 * b;
        let hd = head as usize;
        let h = first / hd;
        let base = col * 256 * n_sb + first + 4 * lane;
        let gbase = col * col_stride as usize
            + h * head_stride as usize
            + offset as usize
            + (first - h * hd)
            + 4 * lane;
        // SAFETY: base + 3 < (col + 1)·256·n_sb <= x.len(); gbase + 3 lies in
        // column col's gate row (host-checked layout), inside m_cols·col_stride
        // <= g.len() by the launch contract.
        let (a, gv) = unsafe {
            (
                [
                    *x.get_unchecked(base),
                    *x.get_unchecked(base + 1),
                    *x.get_unchecked(base + 2),
                    *x.get_unchecked(base + 3),
                ],
                [
                    *g.get_unchecked(gbase),
                    *g.get_unchecked(gbase + 1),
                    *g.get_unchecked(gbase + 2),
                    *g.get_unchecked(gbase + 3),
                ],
            )
        };
        let mut v = [0.0f32; 4];
        let mut i = 0usize;
        #[unroll]
        while i < 4 {
            v[i] = if gv[i].is_finite() {
                a[i] * sigmoid(gv[i])
            } else {
                f32::NAN
            };
            i += 1;
        }
        // SAFETY: col < m_cols and b < 2·n_sb (above); the launch contract
        // bounds the five planes at m_cols columns; the block index is
        // warp-uniform; `v` holds values 128·b + 4·lane .. +3 of column col.
        let refused = unsafe {
            q8_1_quant_vals(
                v, col, b, n_sb, half_it, quad_it, lane, &mut q3, &mut q4, &mut q6, &mut s8,
                &mut d8,
            )
        };
        if refused && lane == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct GatedQuantKernels {
    module: gated_quant_kernels::LoadedModule,
}

/// The five q8_1 planes a launch writes, and their geometry.
struct Planes<'a> {
    q3: &'a mut DeviceBuffer<u64>,
    q4: &'a mut DeviceBuffer<u32>,
    q6: &'a mut DeviceBuffer<u32>,
    s8: &'a mut DeviceBuffer<i32>,
    d8: &'a mut DeviceBuffer<f32>,
    k: usize,
}

impl GatedQuantKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<GatedQuantKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { gated_quant_kernels::load(ctx)? };
        Ok(GatedQuantKernels { module })
    }

    /// Enqueue the gated quantization of the first `cols` columns of `x`
    /// (`act.k()` f32 each) under the gates `g` laid out as `gate`, into
    /// `act` (`1 <= cols <= act.m()`): the decode and pass input of a gated
    /// output projection, the counterpart of
    /// [`crate::Gpu::enqueue_quantize_q8_1_layer`]. A non-finite gate or
    /// product raises [`FaultSite::QuantColumn`] on `fault`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q8act(
        &self,
        stream: &CudaStream,
        (x, g): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        gate: GateLayout,
        act: &mut Q8Act,
        cols: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        if !(1..=act.m()).contains(&cols) {
            return Err(GpuError::shape(
                "gated_quant::enqueue_q8act",
                format!("{cols} columns of an activation of {}", act.m()),
            ));
        }
        let k = act.k();
        let planes = Planes {
            q3: &mut act.q3,
            q4: &mut act.q4,
            q6: &mut act.q6,
            s8: &mut act.s8,
            d8: &mut act.d8,
            k,
        };
        self.launch(
            "gated_quant::enqueue_q8act",
            stream,
            (x, g),
            gate,
            planes,
            cols,
            fault,
        )
    }

    /// [`GatedQuantKernels::enqueue_q8act`] into a GEMM's activations: the
    /// first `n_cols` columns (`1 <= n_cols <= act.cols()`), the ubatch input
    /// of a gated output projection — the counterpart of
    /// [`crate::Gpu::enqueue_quantize_gemm`]; column `c` holds the bytes a
    /// [`Q8Act`] quantized from the same values and gates holds.
    pub fn enqueue_gemm(
        &self,
        stream: &CudaStream,
        (x, g): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        gate: GateLayout,
        act: &mut GemmAct,
        n_cols: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        if n_cols == 0 || n_cols > act.cols() {
            return Err(GpuError::shape(
                "gated_quant::enqueue_gemm",
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", act.cols()),
            ));
        }
        let k = act.k();
        let p = act.planes_mut();
        let planes = Planes {
            q3: p.q3,
            q4: p.q4,
            q6: p.q6,
            s8: p.s8,
            d8: p.d8,
            k,
        };
        self.launch(
            "gated_quant::enqueue_gemm",
            stream,
            (x, g),
            gate,
            planes,
            n_cols,
            fault,
        )
    }

    /// The one launch both enqueues make, after the layout checks.
    #[allow(
        clippy::too_many_arguments,
        reason = "the enqueue's inputs, the layout, the planes and the sink, handed on"
    )]
    fn launch(
        &self,
        what: &'static str,
        stream: &CudaStream,
        (x, g): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        gate: GateLayout,
        p: Planes<'_>,
        cols: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let GateLayout {
            head,
            head_stride,
            offset,
            col_stride,
        } = gate;
        let k = p.k;
        if head == 0 || !head.is_multiple_of(128) || !k.is_multiple_of(head) {
            return Err(GpuError::shape(
                what,
                format!("head {head} must be a positive multiple of 128 dividing k = {k}"),
            ));
        }
        let last = (k / head - 1) * head_stride + offset + head;
        if last > col_stride {
            return Err(GpuError::shape(
                what,
                format!(
                    "the gates of a column reach {last} values past its start, past the column \
                     stride {col_stride} (head {head}, head stride {head_stride}, offset {offset})"
                ),
            ));
        }
        let lens = [("x", x.len(), cols * k), ("g", g.len(), cols * col_stride)];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let n_sb = k / 256;
        let grid = launch_u32(what, "grid", cols * n_sb * 2)?;
        let head = launch_u32(what, "head", head)?;
        let head_stride = launch_u32(what, "head_stride", head_stride)?;
        let offset = launch_u32(what, "offset", offset)?;
        let col_stride = launch_u32(what, "col_stride", col_stride)?;
        let m = launch_u32(what, "cols", cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q8_1_quantize_gated(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q8_1_quantize_gated(
            stream,
            &prep,
            x,
            g,
            head,
            head_stride,
            offset,
            col_stride,
            m,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            p.q3,
            p.q4,
            p.q6,
            p.s8,
            p.d8,
            fault,
        )?;
        Ok(())
    }
}
