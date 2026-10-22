//! Raon-1's feed-forward blocks over one row ([`ffn`]) or a pass's rows
//! ([`ffn_rows`]): the dense SiTU-GLU block of the leading layers and the
//! routed block with its shared expert (llama.cpp `src/models/raon1.cpp`:
//! `build_ffn` / `build_moe_ffn` at `LLM_FFN_SITU`, the shared expert added
//! after the routed sum).
//!
//! - dense: gate and up sites of the normed row, SiTU-GLU
//!   (`ElemKernels::enqueue_situ`), the down site;
//! - routed: the sigmoid router at 128 experts and the top 6
//!   ([`sigmoid_128_6`]: the selection bias steers the pick only, the kept
//!   unbiased scores renormalized and scaled), the card leg over every
//!   expert (`card_leg`: the gate·up `_sel` under [`Act::Situ`], the down
//!   `_sel`, the slots' weighted sum in slot order), then the shared expert
//!   as the dense block at its width, added to the routed sum.
//!
//! Every expert is resident: a slot's place in its stacks is its expert id,
//! so the router's ids are the leg's places.
//!
//! A pass's rows are each the one row's bits: the router over the rows is
//! `enqueue_router_rows` (what that many one-token routers write), the card
//! leg's slots are slot-local whatever its token count, the SiTU and the
//! add are element-wise, and each site runs over the rows as
//! [`project_rows`] says: one launch over the rows where the site has one
//! (the K-quants' m-column gemv through the pass's stage, the f32 rows'
//! gemv), else the one-row launch a row.

use super::rda::{self, RowsKit};
use super::state::PASS_ROWS;
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::kquant::Act;
use crate::site::{Form, SiteTy};
use crate::tensor::{Window, WindowMut};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::placement::kernels::DownAct;

pub use crate::route_body::sigmoid_128_6;

/// What the refusals of this file name.
const WHAT: &str = "raon1 ffn";

/// A gate·up·down block's three sites.
pub(crate) struct Glu {
    pub(crate) gate: Site,
    pub(crate) up: Site,
    pub(crate) down: Site,
    /// The block's width: the gate's and up's rows.
    pub(crate) ff: usize,
}

impl Glu {
    pub(crate) fn sites(&self) -> [&Site; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

/// A routed layer's resident pieces.
pub(crate) struct Routed {
    /// The F32 router rows and the selection bias.
    pub(crate) router: String,
    pub(crate) bias: String,
    /// The gate, up and down stacks' names.
    pub(crate) stacks: [String; 3],
    pub(crate) shared: Glu,
}

/// A layer's feed-forward block.
pub(crate) enum Ffn {
    Dense(Glu),
    Routed(Routed),
}

/// The buffers the blocks run over, one row.
pub(crate) struct FfnScratch {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    route: sigmoid_128_6::RouterOut,
    leg: LegBufs,
}

impl FfnScratch {
    /// The buffers for rows of `n` values, the widest dense block `ff` and
    /// routed experts `expert_ff` wide. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        n: usize,
        ff: usize,
        expert_ff: usize,
    ) -> Result<FfnScratch, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let shape = LegShape {
            cap: 1,
            slots_per_col: sigmoid_128_6::N_USED,
            n,
            ff: expert_ff,
        };
        Ok(FfnScratch {
            gate: z(ff)?,
            up: z(ff)?,
            act: z(ff)?,
            shared: z(n)?,
            routed: z(n)?,
            route: sigmoid_128_6::RouterOut::new(stream)?,
            leg: LegBufs::for_down(stream, shape, DownAct::Q8Blocks32)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        [&self.gate, &self.up, &self.act, &self.shared, &self.routed]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.leg.bytes()
            + 3 * size_of::<f32>() * sigmoid_128_6::N_EXPERT
    }
}

/// The buffers the blocks run over a pass's rows, up to `rows` of them,
/// token-major: the sites' rows, the router's scores, ids and weights (a
/// row each) and the card leg's.
pub(crate) struct FfnRows {
    rows: usize,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    leg: LegBufs,
}

impl FfnRows {
    /// The buffers for passes of up to `rows` rows of `n` values, the widest
    /// dense block `ff` and routed experts `expert_ff` wide. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        n: usize,
        (ff, expert_ff): (usize, usize),
        rows: usize,
    ) -> Result<FfnRows, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let shape = LegShape {
            cap: rows,
            slots_per_col: sigmoid_128_6::N_USED,
            n,
            ff: expert_ff,
        };
        Ok(FfnRows {
            rows,
            gate: z(rows * ff)?,
            up: z(rows * ff)?,
            act: z(rows * ff)?,
            shared: z(rows * n)?,
            routed: z(rows * n)?,
            probs: z(rows * sigmoid_128_6::N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, rows * sigmoid_128_6::N_USED)?,
            weights: z(rows * sigmoid_128_6::N_USED)?,
            leg: LegBufs::for_down(stream, shape, DownAct::Q8Blocks32)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        [
            &self.gate,
            &self.up,
            &self.act,
            &self.shared,
            &self.routed,
            &self.probs,
            &self.weights,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.ids.num_bytes()
            + self.leg.bytes()
    }
}

/// The kernels the blocks launch beyond the sites'.
pub(crate) struct FfnKernels {
    pub(crate) router: sigmoid_128_6::RouterKernels,
    pub(crate) leg: CardLeg,
}

impl FfnKernels {
    pub(crate) fn load(gpu: &Gpu) -> Result<FfnKernels, GpuError> {
        Ok(FfnKernels {
            router: sigmoid_128_6::RouterKernels::load(gpu.context())?,
            leg: CardLeg::load(gpu)?,
        })
    }
}

/// `y = down(situ(gate(xn), up(xn)))` over one row.
fn glu(
    cx: &GqaCx<'_>,
    (acts, rule): (&mut Acts, Act),
    l: usize,
    g: &Glu,
    xn: &DeviceBuffer<f32>,
    s: (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (gate, up, act) = s;
    g.gate.project(cx, acts, l, xn, gate)?;
    g.up.project(cx, acts, l, xn, up)?;
    cx.gpu
        .elem()
        .enqueue_situ(cx.gpu.stream(), rule, (gate, up), g.ff, act)?;
    g.down.project(cx, acts, l, act, y)
}

/// Layer `l`'s block over its normed row `xn` into `y` (the branch's
/// output): the dense block, or the routed sum plus the shared expert.
#[allow(
    clippy::too_many_arguments,
    reason = "one block's inputs: the launch context, the kernels, the forms, the rule, the layer, \
              the row and the scratch"
)]
pub(crate) fn ffn(
    cx: &GqaCx<'_>,
    k: &FfnKernels,
    acts: &mut Acts,
    (rule, scale): (Act, f32),
    (l, f): (usize, &Ffn),
    xn: &DeviceBuffer<f32>,
    s: &mut FfnScratch,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let FfnScratch {
        gate,
        up,
        act,
        shared,
        routed,
        route,
        leg,
    } = s;
    match f {
        Ffn::Dense(g) => glu(cx, (acts, rule), l, g, xn, (gate, up, act), y),
        Ffn::Routed(r) => {
            let (gpu, w) = (cx.gpu, cx.w);
            let stream = gpu.stream();
            k.router.enqueue_router(
                stream,
                w.f32_tensor(WHAT, &r.router)?,
                xn,
                w.f32_buf(WHAT, &r.bias)?,
                scale,
                route,
                gpu.layer_sink(l)?,
            )?;
            let picks = (&route.ids, &route.weights);
            routed_sum(gpu, w, &k.leg, (l, r, rule), (xn, 1), picks, leg, routed)?;
            glu(cx, (acts, rule), l, &r.shared, xn, (gate, up, act), shared)?;
            gpu.elem().enqueue_add(stream, routed, shared, cx.hidden, y)
        }
    }
}

/// f32s of the stage ([`RowsKit`]) a pass of up to [`PASS_ROWS`] rows of
/// blocks at most `ff` wide writes: a row-major gate or up site's columns.
#[must_use]
pub(crate) fn stage_len(ff: usize) -> usize {
    PASS_ROWS * ff
}

/// Whether site `s` runs over several rows in one launch: the f32 rows'
/// gemv, or a K-quant's m-column gemv. Not a Q6_K row of an odd
/// super-block count, which `site::kgemv` runs one column at a launch, nor
/// the Q5_0 form.
fn has_columns(s: &Site) -> bool {
    match s.ty().reads() {
        Form::F32 | Form::Q8x32 => true,
        Form::Q8x128 => {
            let odd_q6k = matches!(s.ty(), SiteTy::Q6K) && (s.k() / 256) % 2 == 1;
            s.ty().kgemv_order().is_some() && !odd_q6k
        }
        Form::Q8Blocks32 => false,
    }
}

/// `y = W · x` for site `s` over `m` rows of `x` (token-major, `s.k()`
/// values a row) into `y` (`rows` values a row), each row the one-row
/// [`Site::project`]'s bits: a site [`has_columns`] in one launch over the
/// rows ([`rda::project_rows`]: each column its one-column launch bit for
/// bit), any other the one-row launch on each row's window. At one row it
/// is [`Site::project`].
#[allow(
    clippy::too_many_arguments,
    reason = "a site's launch inputs: the context, the row kit, the forms, the layer, the site, \
              the rows and the buffers"
)]
fn project_rows(
    cx: &GqaCx<'_>,
    kit: &mut RowsKit<'_>,
    acts: &mut Acts,
    l: usize,
    s: &Site,
    (m, rows): (usize, usize),
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    if m == 1 {
        return s.project(cx, acts, l, x, y);
    }
    if has_columns(s) {
        return rda::project_rows(cx, kit, acts, (l, m), (s, rows), x, y);
    }
    let (k, f) = (s.k(), size_of::<f32>());
    for r in 0..m {
        let xr = Window::<f32>::of(x, r * k * f, k)?;
        let mut yr = WindowMut::<f32>::of_mut(y, r * rows * f, rows)?;
        s.project(cx, acts, l, &xr, &mut yr)?;
    }
    Ok(())
}

/// [`glu`] over `m` rows of `xn`: each site over the rows
/// ([`project_rows`]; a site's rows run before the next site's, so a
/// per-row form reads its weight from the cache after the first row), the
/// SiTU over every row in one launch.
#[allow(
    clippy::too_many_arguments,
    reason = "glu's inputs, the row count and the row kit"
)]
fn glu_rows(
    cx: &GqaCx<'_>,
    kit: &mut RowsKit<'_>,
    (acts, rule): (&mut Acts, Act),
    (l, m): (usize, usize),
    g: &Glu,
    xn: &DeviceBuffer<f32>,
    s: (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (gate, up, act) = s;
    project_rows(cx, kit, acts, l, &g.gate, (m, g.ff), xn, gate)?;
    project_rows(cx, kit, acts, l, &g.up, (m, g.ff), xn, up)?;
    cx.gpu
        .elem()
        .enqueue_situ(cx.gpu.stream(), rule, (gate, up), m * g.ff, act)?;
    project_rows(cx, kit, acts, l, &g.down, (m, cx.hidden), act, y)
}

/// [`ffn`] over `m` rows of `xn` (token-major) into `y`, each row the
/// one-row block's bits (module doc): `m` is 1 to the rows `s` was made for;
/// `kit`'s stage holds at least the [`stage_len`] of the widest block.
#[allow(
    clippy::too_many_arguments,
    reason = "ffn's inputs, the row count and the row kit"
)]
pub(crate) fn ffn_rows(
    cx: &GqaCx<'_>,
    k: &FfnKernels,
    acts: &mut Acts,
    (rule, scale): (Act, f32),
    (l, f): (usize, &Ffn),
    m: usize,
    xn: &DeviceBuffer<f32>,
    s: &mut FfnRows,
    y: &mut DeviceBuffer<f32>,
    mut kit: RowsKit<'_>,
) -> Result<(), GpuError> {
    if m == 0 || m > s.rows {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}: a pass of {m} rows on buffers of {} (1..={})",
                s.rows, s.rows
            ),
        ));
    }
    let FfnRows {
        gate,
        up,
        act,
        shared,
        routed,
        probs,
        ids,
        weights,
        leg,
        ..
    } = s;
    match f {
        Ffn::Dense(g) => glu_rows(
            cx,
            &mut kit,
            (acts, rule),
            (l, m),
            g,
            xn,
            (gate, up, act),
            y,
        ),
        Ffn::Routed(r) => {
            let (gpu, w) = (cx.gpu, cx.w);
            let stream = gpu.stream();
            k.router.enqueue_router_rows(
                stream,
                gpu,
                cx.g32,
                w.f32_tensor(WHAT, &r.router)?,
                xn,
                w.f32_buf(WHAT, &r.bias)?,
                scale,
                m,
                probs,
                ids,
                weights,
                gpu.layer_sink(l)?,
            )?;
            let picks = (&*ids, &*weights);
            routed_sum(gpu, w, &k.leg, (l, r, rule), (xn, m), picks, leg, routed)?;
            glu_rows(
                cx,
                &mut kit,
                (acts, rule),
                (l, m),
                &r.shared,
                xn,
                (gate, up, act),
                shared,
            )?;
            gpu.elem()
                .enqueue_add(stream, routed, shared, m * cx.hidden, y)
        }
    }
}

/// The card leg over every expert of layer `l` for `m` rows of `xn`, the
/// router's ids as the slots' places, its weights the sum's, into `acc`.
#[allow(
    clippy::too_many_arguments,
    reason = "the leg's inputs: the card, the weights, the kernels, the layer, the rows, the picks \
              and the buffers"
)]
fn routed_sum(
    gpu: &Gpu,
    w: &Weights,
    leg: &CardLeg,
    (l, r, rule): (usize, &Routed, Act),
    (xn, m): (&DeviceBuffer<f32>, usize),
    (sel, weights): (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
    bufs: &mut LegBufs,
    acc: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let names = r.stacks.each_ref().map(String::as_str);
    let stacks = LegStacks::of_resident(w, names)?;
    let run = LegRun {
        layer: l,
        stacks: &stacks,
        normed: xn,
        m,
        sel,
        n_card: sigmoid_128_6::N_EXPERT,
        rule,
    };
    leg.enqueue(gpu, bufs, &run, Some(LegSum { w: weights, acc }))
}
