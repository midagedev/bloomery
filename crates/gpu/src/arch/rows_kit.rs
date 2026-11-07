//! The rows kit: what a pass of 1 to [`PASS_ROWS`] rows needs beyond the step's launches, shared
//! by every family whose layers run a pass of rows. A K-quant site's m-column gemv writes its
//! columns row-major, so it lands in a stage and one copy ([`RowKernels`], the token-major copy's
//! one owner) lays the rows token-major; the f32 and q8_0 sites' m-column launches write token-major
//! straight in. [`project_rows`] runs a site over a pass's rows, [`kgemv_rows`] a K-quant site
//! over rows already in q8_1 form, and each column is the site's one-column launch on that row
//! bit for bit. A site with no m-column launch ([`has_columns`]) runs its one-row launch on each
//! row's window.

use crate::GpuError;
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::linear::{LinearKernels, PASS_ROWS};
use crate::site::{self, Form, Order, SiteTy};
use crate::tensor::{Q8Act, Window, WindowMut};
use crate::token_major::TokenMajor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::sync::Arc;

/// The kernels a pass of more than one row launches beyond the step's: the
/// copy of a row-major K-quant output into token-major rows
/// ([`TokenMajor`], the one owner of that copy).
pub(crate) struct RowKernels {
    pub(crate) tm: TokenMajor,
}

impl RowKernels {
    /// Load the copy's module into `ctx`. Load-time only.
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<RowKernels, GpuError> {
        Ok(RowKernels {
            tm: TokenMajor::load(ctx)?,
        })
    }
}

/// What a pass of rows lends a layer beyond its buffers: the linear
/// families' kernels, the row kernels and the stage a row-major site's
/// columns land in before their copy, at least the widest stage any layer
/// the pass runs needs.
pub(crate) struct RowsKit<'a> {
    pub linear: &'a LinearKernels,
    pub rows: &'a RowKernels,
    pub stage: &'a mut DeviceBuffer<f32>,
}

/// The q8_1 forms of 2 to [`PASS_ROWS`] columns a pass of rows reads, at the
/// width of each K-quant site of `sites`, beside the one-column forms
/// [`Acts::new`] makes. Load-time only.
pub(crate) fn add_row_forms<'s>(
    stream: &CudaStream,
    acts: &mut Acts,
    sites: impl Iterator<Item = &'s Site>,
) -> Result<(), GpuError> {
    for s in sites.filter(|s| s.ty().reads() == Form::Q8x128) {
        for m in 2..=PASS_ROWS {
            acts.add_q8_1(stream, s.k(), m)?;
        }
    }
    Ok(())
}

/// Where site `t`'s m-column launch lands column `c`'s outputs: a K-quant's
/// by its gemv ([`site::kgemv_order`]), token-major on the gemv arm;
/// `None` for the Q5_0 form, which no row launch here reads.
pub(crate) fn order(t: &Site) -> Option<Order> {
    match t.ty().reads() {
        Form::Q8x128 => site::kgemv_order(t.ty()),
        Form::F32 | Form::Q8x32 => Some(Order::TokenMajor),
        Form::Q8Blocks32 => None,
    }
}

/// The refusal of site `t` at a pass of `m` rows, naming `what`.
fn no_rows(what: &'static str, t: &Site, m: usize) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "{} is {}: a pass of {m} rows runs no m-column launch of it",
            t.name(),
            t.ty()
        ),
    )
}

/// K-quant site `t` over the columns of the q8_1 form `act`, its `rows`
/// outputs of each column token-major into `y`: a row-major gemv through the
/// stage and the copy, a token-major one straight in. Refusals name
/// `cx.what`, as [`Site::project`]'s do.
pub(crate) fn kgemv_rows(
    cx: &GqaCx<'_>,
    kit: &mut RowsKit<'_>,
    (t, rows): (&Site, usize),
    act: &Q8Act,
    l: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let m = act.m();
    match site::kgemv_order(t.ty()) {
        Some(Order::TokenMajor) => t.kgemv(cx, act, l, y),
        Some(Order::RowMajor) => {
            {
                let mut stage = WindowMut::<f32>::of_mut(&mut *kit.stage, 0, rows * m)?;
                t.kgemv(cx, act, l, &mut stage)?;
            }
            kit.rows.tm.enqueue(cx.gpu.stream(), kit.stage, rows, m, y)
        }
        None => Err(no_rows(cx.what, t, m)),
    }
}

/// Whether site `t` runs over several rows in one launch: the f32 rows' gemv,
/// or a K-quant's m-column gemv. Not a Q6_K row of an odd super-block count,
/// which `site::kgemv` runs one column at a launch, nor the Q5_0 form.
pub(crate) fn has_columns(t: &Site) -> bool {
    match t.ty().reads() {
        Form::F32 | Form::Q8x32 => true,
        Form::Q8x128 => {
            let odd_q6k = matches!(t.ty(), SiteTy::Q6K) && (t.k() / 256) % 2 == 1;
            site::kgemv_order(t.ty()).is_some() && !odd_q6k
        }
        Form::Q8Blocks32 => false,
    }
}

/// The importance collector's hook for site `t` read over the `m` token-major rows of `x` in one
/// launch (a caller that launches the m-column gemv itself calls it beside it): nothing when no
/// collector is attached. A one-row launch is [`Site::project`], which feeds it itself.
pub(crate) fn collect_rows(
    cx: &GqaCx<'_>,
    t: &Site,
    x: &DeviceBuffer<f32>,
    m: usize,
) -> Result<(), GpuError> {
    match crate::imatrix::collector() {
        Some(c) => c.feed_dense(cx.gpu, t.name(), x, (m, t.k())),
        None => Ok(()),
    }
}

/// `y = W · x` for site `t` over the `m` token-major rows of `x`, its `rows`
/// outputs of each row token-major into `y`: a site that [`has_columns`] in
/// one launch over the rows (the gemv arm's m-column launch, or a K-quant's
/// gemv over `x`'s q8_1 form of `m` columns, [`kgemv_rows`]), any other the
/// one-row launch on each row's window. Each row is the site's one-row
/// [`Site::project`] bit for bit; at one row it is [`Site::project`].
pub(crate) fn project_rows(
    cx: &GqaCx<'_>,
    kit: &mut RowsKit<'_>,
    acts: &mut Acts,
    (l, m): (usize, usize),
    (t, rows): (&Site, usize),
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    if m == 1 {
        return t.project(cx, acts, l, x, y);
    }
    if !has_columns(t) {
        let (k, f) = (t.k(), size_of::<f32>());
        for r in 0..m {
            let xr = Window::<f32>::of(x, r * k * f, k)?;
            let mut yr = WindowMut::<f32>::of_mut(y, r * rows * f, rows)?;
            t.project(cx, acts, l, &xr, &mut yr)?;
        }
        return Ok(());
    }
    collect_rows(cx, t, x, m)?;
    match t.ty().reads() {
        Form::F32 | Form::Q8x32 => site::gemv(cx.gpu, cx.g32, (t.ty(), cx.w, t.name()), x, m, y),
        Form::Q8x128 => {
            let act = acts.q8_1_at(cx.what, t.k(), m)?;
            cx.gpu.enqueue_quantize_q8_1_layer(x, act, l)?;
            kgemv_rows(cx, kit, (t, rows), act, l, y)
        }
        Form::Q8Blocks32 => Err(no_rows(cx.what, t, m)),
    }
}
