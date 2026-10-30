//! The SwiGLU quantizer between a gate·up pair and its down: the launchers of
//! `gemm_swiglu_quant` and `gemm_swiglu_quant_sel` (declared in `kernels.rs`),
//! into the GEMM's [`GemmAct`] or a gemv's [`Q8Act`] — the two hold the same
//! q8_1 planes, so one kernel writes either — and the warp body both entries
//! share ([`swiglu_quant_block`]).

use super::act::GemmActPlanesMut;
use super::{GemmAct, GemmKernels};
use crate::fault::FaultSink;
use crate::kquant::Act;
use crate::tensor::Q8Act;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// `act`'s five planes as the launch writes them: a `Q8Act` holds the
/// layout [`GemmAct`] documents (the q3/q4/q6 permutations, the 32-value
/// code sums, the 128-value scales), per-column sizes equal.
fn q8_planes(act: &mut Q8Act) -> GemmActPlanesMut<'_> {
    GemmActPlanesMut {
        q3: &mut act.q3,
        q4: &mut act.q4,
        q6: &mut act.q6,
        s8: &mut act.s8,
        d8: &mut act.d8,
    }
}

/// The columns, values per column and planes of the activation one launch
/// fills.
struct Target<'a> {
    cols: usize,
    k: usize,
    planes: GemmActPlanesMut<'a>,
}

impl GemmKernels {
    /// Enqueue `act = q8_1(silu(g) · u)` over the first `n_cols` slot
    /// columns (`act.k()` values each, slot-major as the gate and up GEMMs
    /// write them): [`GemmKernels::enqueue_swiglu_act_quant`] under
    /// [`Act::SiluMul`], the bytes `ElemKernels::enqueue_swiglu` then
    /// [`Gpu::enqueue_quantize_gemm`] leave.
    ///
    /// [`Gpu::enqueue_quantize_gemm`]: crate::Gpu::enqueue_quantize_gemm
    pub fn enqueue_swiglu_quant(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.enqueue_swiglu_act_quant(stream, g, u, Act::SiluMul, n_cols, act, fault)
    }

    /// Enqueue `act = q8_1(rule(g, u))` over the first `n_cols` slot columns
    /// of `act`: one launch of `gemm_swiglu_quant`, `rule` the gate·up rule
    /// of the K-quant family (`kquant::act::apply`: `silu(g) · u`, ik's
    /// clamped SwiGLU or the OAI SwiGLU, the values of a fused `_sel` gate·up
    /// under the same rule). A 128-value block holding a non-finite `g` or
    /// `u` value is stored refused (NaN scale, zero codes) and raised on
    /// `fault` as [`FaultSite::QuantColumn`] under any rule; a rule
    /// [`Act::code`] refuses is refused by name. Asynchronous,
    /// allocation-free, capturable.
    ///
    /// [`FaultSite::QuantColumn`]: crate::FaultSite::QuantColumn
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_act_quant(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        rule: Act,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let target = Target {
            cols: act.cols,
            k: act.k,
            planes: act.planes_mut(),
        };
        self.launch(
            "GemmKernels::enqueue_swiglu_act_quant",
            stream,
            (g, u, rule),
            (n_cols, None),
            target,
            fault,
        )
    }

    /// [`GemmKernels::enqueue_swiglu_act_quant`] over the first `n_cols` slot
    /// columns a remapped route left the card — the slots whose place
    /// `sel[0..n_cols]` is below `n_card` (`gemm_swiglu_quant_sel`, declared
    /// in `kernels.rs`): a host slot's column (`crate::hybrid::HOST`) is
    /// left exactly as it stands — a GEMM over the route reads only the
    /// listed (card) columns, so a stale column a reset left or a non-finite
    /// `g`/`u` row of it raises nothing — and a place in `[n_card, HOST)`
    /// raises [`FaultSite::ExpertId`] on `fault` with nothing stored. The
    /// written columns' bytes are the plain launcher's under the same rule.
    /// Asynchronous, allocation-free, capturable.
    ///
    /// [`FaultSite::ExpertId`]: crate::FaultSite::ExpertId
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_act_quant_sel(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        rule: Act,
        sel: &DeviceBuffer<u32>,
        n_card: usize,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_swiglu_act_quant_sel";
        let n_card = launch_u32(what, "n_card", n_card)?;
        let target = Target {
            cols: act.cols,
            k: act.k,
            planes: act.planes_mut(),
        };
        self.launch(
            what,
            stream,
            (g, u, rule),
            (n_cols, Some((sel, n_card))),
            target,
            fault,
        )
    }

    /// [`GemmKernels::enqueue_swiglu_act_quant`] into the gemv activation
    /// `act` (`act.m()` columns of `act.k()` values): the input of a K-quant
    /// down's `site::kgemv`, whose column `c` holds the bytes a `Q8Act`
    /// quantized from the same values holds.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_act_quant_q8(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        rule: Act,
        n_cols: usize,
        act: &mut Q8Act,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let target = Target {
            cols: act.m(),
            k: act.k(),
            planes: q8_planes(act),
        };
        self.launch(
            "GemmKernels::enqueue_swiglu_act_quant_q8",
            stream,
            (g, u, rule),
            (n_cols, None),
            target,
            fault,
        )
    }

    /// The launch the enqueues share, over the first `n_cols` of `target`'s
    /// columns: `sel` `None` quantizes every one, `Some((sel, n_card))` only
    /// those whose place is below `n_card`.
    fn launch(
        &self,
        what: &'static str,
        stream: &CudaStream,
        (g, u, rule): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, Act),
        (n_cols, sel): (usize, Option<(&DeviceBuffer<u32>, u32)>),
        target: Target<'_>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let Target { cols, k, planes } = target;
        if n_cols == 0 || n_cols > cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act cols = {cols}, got {n_cols}"),
            ));
        }
        if g.len() < n_cols * k || u.len() < n_cols * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "g.len() {} and u.len() {} need n_cols*k = {n_cols}*{k}",
                    g.len(),
                    u.len(),
                ),
            ));
        }
        if let Some((sel, _)) = sel
            && sel.len() < n_cols
        {
            return Err(GpuError::shape(
                what,
                format!("sel.len() {} < n_cols {n_cols}", sel.len()),
            ));
        }
        let (act_code, limit) = rule.code(what)?;
        let n_sb = k / 256;
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let n_cols = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        if let Some((sel, n_card)) = sel {
            let prep = self
                .module
                .prepare_gemm_swiglu_quant_sel(LaunchConfig1D::new(grid, 32, 0))?;
            self.module.gemm_swiglu_quant_sel(
                stream,
                &prep,
                g,
                u,
                sel,
                n_card,
                act_code,
                limit,
                n_cols,
                n_sb,
                n_sb.div_ceil(2),
                n_sb.div_ceil(4),
                planes.q3,
                planes.q4,
                planes.q6,
                planes.s8,
                planes.d8,
                fault,
            )?;
        } else {
            let prep = self
                .module
                .prepare_gemm_swiglu_quant(LaunchConfig1D::new(grid, 32, 0))?;
            self.module.gemm_swiglu_quant(
                stream,
                &prep,
                g,
                u,
                act_code,
                limit,
                n_cols,
                n_sb,
                n_sb.div_ceil(2),
                n_sb.div_ceil(4),
                planes.q3,
                planes.q4,
                planes.q6,
                planes.s8,
                planes.d8,
                fault,
            )?;
        }
        Ok(())
    }
}

/// One warp's 128-value block `b` of column `col` of the K-quant SwiGLU
/// quantizer — the body both entries of `kernels.rs` run once any sel gate
/// has decided the column is the card's: lane `l` takes the values
/// `128·b + 4·l .. +3` of `g` and `u` of the column, forms the gate·up rule
/// `kquant::act::apply(act, limit, ..)` of each pair and hands the four to
/// `q8_1_quant_vals`, the values `elem::swiglu` stores under the first rule
/// and the bytes `q3k_quantize_q8_1` writes from them. A block holding a
/// non-finite `g` or `u` value is refused as the quantizer refuses it (NaN
/// scale, zero codes); the entry raises the fault on the return.
///
/// # Safety
///
/// All 32 lanes of the warp call it with one `(col, b)`; `col` is below the
/// launch's columns and `b` below `2 · n_sb`; `g` and `u` hold `256 · n_sb`
/// values per column, and `q3`, `q4`, `q6`, `s8`, `d8` the launch contract's
/// per-column sizes.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(super) unsafe fn swiglu_quant_block(
    g: &[f32],
    u: &[f32],
    act: u32,
    limit: f32,
    col: usize,
    b: usize,
    n_sb: usize,
    half_it: u32,
    quad_it: u32,
    lane: usize,
    q3: &mut cuda_device::DisjointSlice<u64>,
    q4: &mut cuda_device::DisjointSlice<u32>,
    q6: &mut cuda_device::DisjointSlice<u32>,
    s8: &mut cuda_device::DisjointSlice<i32>,
    d8: &mut cuda_device::DisjointSlice<f32>,
) -> bool {
    let base = col * 256 * n_sb + 128 * b + 4 * lane;
    // SAFETY: base + 3 < (col + 1)·256·n_sb <= n_cols·256·n_sb, inside g
    // and u by this fn's contract.
    let (gv, uv) = unsafe {
        (
            [
                *g.get_unchecked(base),
                *g.get_unchecked(base + 1),
                *g.get_unchecked(base + 2),
                *g.get_unchecked(base + 3),
            ],
            [
                *u.get_unchecked(base),
                *u.get_unchecked(base + 1),
                *u.get_unchecked(base + 2),
                *u.get_unchecked(base + 3),
            ],
        )
    };
    let v = if crate::fault::quad_finite(gv) & crate::fault::quad_finite(uv) {
        [
            crate::kquant::act::apply(act, limit, gv[0], uv[0]),
            crate::kquant::act::apply(act, limit, gv[1], uv[1]),
            crate::kquant::act::apply(act, limit, gv[2], uv[2]),
            crate::kquant::act::apply(act, limit, gv[3], uv[3]),
        ]
    } else {
        [f32::NAN; 4]
    };
    // SAFETY: col < n_cols and b < 2·n_sb by this fn's contract, the output
    // bounds are the launch contract's, the block is one warp with one
    // `(col, b)`, and `v` holds values 128·b + 4·lane .. +3 of column col.
    unsafe { crate::q8_1_quant_vals(v, col, b, n_sb, half_it, quad_it, lane, q3, q4, q6, s8, d8) }
}
