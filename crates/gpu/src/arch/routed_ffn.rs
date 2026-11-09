//! The SiTU feed-forward blocks over one row ([`ffn`]) or a pass's rows ([`ffn_rows`]), common to
//! the families that run them (Raon-1: llama.cpp `src/models/raon1.cpp`; Kimi K3: `src/models/kimi-k3.cpp`):
//! the dense SiTU-GLU block of the leading layers and the routed block with its shared expert
//! (`build_ffn` / `build_moe_ffn` at `LLM_FFN_SITU`, the shared expert added after the routed
//! sum).
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
//! Two stages a routed layer may add around the leg, each `None` for a family without them
//! (llama.cpp `src/models/kimi-k3.cpp` `build_latent_moe`, Kimi K3's block): a latent pair ([`Latent`]: the experts read and write a
//! row of the latent width, the pair projects the normed row down to it
//! and the routed sum back up) and an RMS norm on the routed sum
//! ([`Routed::sum_norm`], at the block's eps, before the latent up). The
//! router is a parameter too (`R`, any [`RouterEntry`] instance); the
//! default is Raon-1's.
//!
//! A pass's rows are each the one row's bits: the router over the rows is
//! `enqueue_router_rows` (what that many one-token routers write), the card
//! leg's slots are slot-local whatever its token count, the SiTU and the
//! add are element-wise, and each site runs over the rows as
//! [`rows_kit::project_rows`] says: one launch over the rows where the site has one
//! (the K-quants' m-column gemv through the pass's stage, the f32 rows'
//! gemv), else the one-row launch a row.

use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::arch::rows_kit::{self, RowsKit};
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::kquant::Act;
use crate::linear::PASS_ROWS;
use crate::routed_block::{RouteIn, RoutedRows, RouterEntry};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::placement::kernels::DownAct;
use std::marker::PhantomData;

pub use crate::route_body::sigmoid_128_6;

/// What the refusals of this file name.
const WHAT: &str = "routed ffn";

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

/// The latent pair around a routed layer's experts: the two sites and the
/// latent width the experts run at.
pub(crate) struct Latent {
    /// `hidden → width`: the normed row the experts read.
    pub(crate) down: Site,
    /// `width → hidden`: the routed sum.
    pub(crate) up: Site,
    /// The width the experts read and write.
    pub(crate) width: usize,
}

/// A routed layer's resident pieces.
pub(crate) struct Routed {
    /// The F32 router rows and the selection bias.
    pub(crate) router: String,
    pub(crate) bias: String,
    /// The gate, up and down stacks' names.
    pub(crate) stacks: [String; 3],
    pub(crate) shared: Glu,
    /// The latent pair, for a file whose experts run in a latent space; the
    /// router and the shared expert read the full-width row either way.
    pub(crate) latent: Option<Box<Latent>>,
    /// The gain of the RMS norm on the routed sum (over the experts' width,
    /// before the latent up), for a file that has one.
    pub(crate) sum_norm: Option<String>,
}

impl Routed {
    /// The sites a layer's launches read: the shared expert's, then the
    /// latent pair's.
    pub(crate) fn sites(&self) -> Vec<&Site> {
        let mut out: Vec<&Site> = self.shared.sites().to_vec();
        if let Some(lat) = &self.latent {
            out.extend([&lat.down, &lat.up]);
        }
        out
    }

    /// The width the experts read and write: the latent's, else the model's
    /// `hidden`.
    fn expert_n(&self, hidden: usize) -> usize {
        self.latent.as_ref().map_or(hidden, |lat| lat.width)
    }
}

/// What the routed block's buffers are sized for beside the model width and
/// the dense block's: the routed experts' width, the latent's (where the
/// layers have one) and the form the card leg's down reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoutedShape {
    /// A routed expert's width: a gate·up slot's rows.
    pub(crate) expert_ff: usize,
    /// The latent width, for layers with a [`Latent`] pair or a sum norm
    /// over a latent row; `None` runs the experts at the model's width.
    pub(crate) latent: Option<usize>,
    /// Whether any layer norms the routed sum.
    pub(crate) sum_norm: bool,
    /// The form the leg's down reads (its stacks' type).
    pub(crate) down: DownAct,
}

impl RoutedShape {
    /// Raon-1's: experts at the model's width, no sum norm, a down reading
    /// q8 blocks of 32.
    pub(crate) const fn plain(expert_ff: usize) -> RoutedShape {
        RoutedShape {
            expert_ff,
            latent: None,
            sum_norm: false,
            down: DownAct::Q8Blocks32,
        }
    }

    /// The experts' width for a model of `n` values a row.
    const fn expert_n(self, n: usize) -> usize {
        match self.latent {
            Some(w) => w,
            None => n,
        }
    }
}

/// The latent path's buffers over `rows` rows: the latent down's output (the
/// leg's rows) and the latent up's output (the routed delta).
struct LatentBufs {
    x: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
}

/// The buffers the latent and sum-norm stages need, each present when the
/// shape asks for it.
struct Stages {
    latent: Option<LatentBufs>,
    normed: Option<DeviceBuffer<f32>>,
}

impl Stages {
    /// The stages' buffers for `rows` rows of `n` model values.
    fn new(
        stream: &CudaStream,
        n: usize,
        rows: usize,
        shape: RoutedShape,
    ) -> Result<Stages, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let w = shape.expert_n(n);
        Ok(Stages {
            latent: match shape.latent {
                Some(w) => Some(LatentBufs {
                    x: z(rows * w)?,
                    up: z(rows * n)?,
                }),
                None => None,
            },
            normed: if shape.sum_norm {
                Some(z(rows * w)?)
            } else {
                None
            },
        })
    }

    fn bytes(&self) -> usize {
        self.latent
            .as_ref()
            .map_or(0, |l| l.x.num_bytes() + l.up.num_bytes())
            + self.normed.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

/// A layer's feed-forward block.
pub(crate) enum Ffn {
    Dense(Glu),
    Routed(Routed),
}

/// The buffers the blocks run over, one row.
pub(crate) struct FfnScratch<R: RouterEntry = sigmoid_128_6::RouterKernels> {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    route: R::Out,
    leg: LegBufs,
    stages: Stages,
}

impl<R: RouterEntry> FfnScratch<R> {
    /// The buffers for rows of `n` values, the widest dense block `ff` and
    /// the routed experts `shape` describes. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        n: usize,
        ff: usize,
        shape: RoutedShape,
    ) -> Result<FfnScratch<R>, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let en = shape.expert_n(n);
        let leg = LegShape {
            cap: 1,
            slots_per_col: R::N_USED,
            n: en,
            ff: shape.expert_ff,
        };
        Ok(FfnScratch {
            gate: z(ff)?,
            up: z(ff)?,
            act: z(ff)?,
            shared: z(n)?,
            routed: z(en)?,
            route: R::new_out(stream)?,
            leg: LegBufs::for_down(stream, leg, shape.down)?,
            stages: Stages::new(stream, n, 1, shape)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        [&self.gate, &self.up, &self.act, &self.shared, &self.routed]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.leg.bytes()
            + self.stages.bytes()
            + 3 * size_of::<f32>() * R::N_EXPERT
    }

    /// The last block's picks read back: the kept expert ids in rank order
    /// and their weights (a gate's instrument; it synchronizes `stream`).
    pub(crate) fn picks(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        Ok((
            R::out_ids(&self.route).to_host_vec(stream)?,
            R::out_weights(&self.route).to_host_vec(stream)?,
        ))
    }
}

/// The buffers the blocks run over a pass's rows, up to `rows` of them,
/// token-major: the sites' rows, the router's scores, ids and weights (a
/// row each) and the card leg's.
pub(crate) struct FfnRows<R: RouterEntry = sigmoid_128_6::RouterKernels> {
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
    stages: Stages,
    router: PhantomData<fn() -> R>,
}

impl<R: RouterEntry> FfnRows<R> {
    /// The buffers for passes of up to `rows` rows of `n` values, the widest
    /// dense block `ff` and the routed experts `shape` describes.
    /// Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        n: usize,
        (ff, shape): (usize, RoutedShape),
        rows: usize,
    ) -> Result<FfnRows<R>, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let en = shape.expert_n(n);
        let leg = LegShape {
            cap: rows,
            slots_per_col: R::N_USED,
            n: en,
            ff: shape.expert_ff,
        };
        Ok(FfnRows {
            rows,
            gate: z(rows * ff)?,
            up: z(rows * ff)?,
            act: z(rows * ff)?,
            shared: z(rows * n)?,
            routed: z(rows * en)?,
            probs: z(rows * R::N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, rows * R::N_USED)?,
            weights: z(rows * R::N_USED)?,
            leg: LegBufs::for_down(stream, leg, shape.down)?,
            stages: Stages::new(stream, n, rows, shape)?,
            router: PhantomData,
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
            + self.stages.bytes()
    }

    /// The last pass's picks read back: each row's kept expert ids in rank
    /// order and its weights, row-major (a gate's instrument; it synchronizes
    /// `stream`).
    pub(crate) fn picks(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        Ok((
            self.ids.to_host_vec(stream)?,
            self.weights.to_host_vec(stream)?,
        ))
    }
}

/// The kernels the blocks launch beyond the sites'.
pub(crate) struct FfnKernels<R: RouterEntry = sigmoid_128_6::RouterKernels> {
    pub(crate) router: R,
    pub(crate) leg: CardLeg,
}

impl<R: RouterEntry> FfnKernels<R> {
    pub(crate) fn load(gpu: &Gpu) -> Result<FfnKernels<R>, GpuError> {
        Ok(FfnKernels {
            router: R::load(gpu.context())?,
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
/// output): the dense block, or the routed sum (through the latent pair and
/// the sum norm where the layer has them) plus the shared expert.
#[allow(
    clippy::too_many_arguments,
    reason = "one block's inputs: the launch context, the kernels, the forms, the rule, the layer, \
              the row and the scratch"
)]
pub(crate) fn ffn<R: RouterEntry>(
    cx: &GqaCx<'_>,
    k: &FfnKernels<R>,
    acts: &mut Acts,
    (rule, scale): (Act, f32),
    (l, f): (usize, &Ffn),
    xn: &DeviceBuffer<f32>,
    s: &mut FfnScratch<R>,
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
        stages,
    } = s;
    match f {
        Ffn::Dense(g) => glu(cx, (acts, rule), l, g, xn, (gate, up, act), y),
        Ffn::Routed(r) => {
            let (gpu, w) = (cx.gpu, cx.w);
            let stream = gpu.stream();
            let rin = RouteIn {
                w: w.f32_tensor(WHAT, &r.router)?,
                bias: Some(w.f32_buf(WHAT, &r.bias)?),
                scale,
                fault: gpu.layer_sink(l)?,
            };
            k.router.route(stream, xn, &rin, route)?;
            if let Some(c) = crate::imatrix::collector() {
                c.feed_dense(gpu, &r.router, xn, (1, cx.hidden))?;
            }
            let Stages { latent, normed } = stages;
            let mut lat = latent_of(l, r, latent.as_mut())?;
            // The experts' rows: the latent down's output, or the normed row.
            let ex: &DeviceBuffer<f32> = match lat.as_mut() {
                Some(LatentRun { pair, x, .. }) => {
                    pair.down.project(cx, acts, l, xn, x)?;
                    x
                }
                None => xn,
            };
            let picks = (R::out_ids(route), R::out_weights(route));
            routed_sum::<R>(gpu, w, &k.leg, (l, r, rule), (ex, 1), picks, leg, routed)?;
            let summed = sum_norm(cx, (l, r), (&*routed, normed.as_mut()), 1)?;
            let delta: &DeviceBuffer<f32> = match lat {
                Some(LatentRun { pair, up, .. }) => {
                    pair.up.project(cx, acts, l, summed, up)?;
                    up
                }
                None => summed,
            };
            glu(cx, (acts, rule), l, &r.shared, xn, (gate, up, act), shared)?;
            gpu.elem().enqueue_add(stream, delta, shared, cx.hidden, y)
        }
    }
}

/// A layer's latent pair with the buffers its two projections write.
struct LatentRun<'a> {
    pair: &'a Latent,
    x: &'a mut DeviceBuffer<f32>,
    up: &'a mut DeviceBuffer<f32>,
}

/// The buffers of layer `l`'s latent path beside its block's latent pair, the
/// two agreeing: both there, or neither. A block and buffers made apart are
/// refused by name.
fn latent_of<'a>(
    l: usize,
    r: &'a Routed,
    bufs: Option<&'a mut LatentBufs>,
) -> Result<Option<LatentRun<'a>>, GpuError> {
    match (&r.latent, bufs) {
        (Some(pair), Some(LatentBufs { x, up })) => Ok(Some(LatentRun { pair, x, up })),
        (None, None) => Ok(None),
        (Some(_), None) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a latent routed block on buffers made without a latent width"),
        )),
        (None, Some(_)) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a routed block with no latent on buffers made with a latent width"),
        )),
    }
}

/// Layer `l`'s routed sum `routed` (`m` rows), normed by the layer's gain
/// into `normed` where the layer has a sum norm, else `routed` itself.
fn sum_norm<'a>(
    cx: &GqaCx<'_>,
    (l, r): (usize, &Routed),
    (routed, normed): (&'a DeviceBuffer<f32>, Option<&'a mut DeviceBuffer<f32>>),
    m: usize,
) -> Result<&'a DeviceBuffer<f32>, GpuError> {
    match (&r.sum_norm, normed) {
        (Some(gain), Some(dst)) => {
            let width = r.expert_n(cx.hidden);
            cx.gpu.elem().enqueue_rms_norm(
                cx.gpu.stream(),
                routed,
                cx.w.f32_buf(WHAT, gain)?,
                cx.eps,
                width,
                m,
                dst,
            )?;
            Ok(dst)
        }
        (None, None) => Ok(routed),
        (Some(_), None) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a routed block with a sum norm on buffers made without one"),
        )),
        (None, Some(_)) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a routed block with no sum norm on buffers made with one"),
        )),
    }
}

/// f32s of the stage ([`RowsKit`]) a pass of up to [`PASS_ROWS`] rows of
/// blocks at most `ff` wide writes: a row-major gate or up site's columns.
#[must_use]
pub(crate) fn stage_len(ff: usize) -> usize {
    PASS_ROWS * ff
}

/// [`glu`] over `m` rows of `xn`: each site over the rows
/// ([`rows_kit::project_rows`]; a site's rows run before the next site's, so a
/// per-row form reads its weight from the cache after the first row), the
/// SiTU over every row in one launch.
#[allow(
    clippy::too_many_arguments,
    reason = "glu's inputs, the row count and the row kit"
)]
pub(crate) fn glu_rows(
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
    rows_kit::project_rows(cx, kit, acts, (l, m), (&g.gate, g.ff), xn, gate)?;
    rows_kit::project_rows(cx, kit, acts, (l, m), (&g.up, g.ff), xn, up)?;
    cx.gpu
        .elem()
        .enqueue_situ(cx.gpu.stream(), rule, (gate, up), m * g.ff, act)?;
    rows_kit::project_rows(cx, kit, acts, (l, m), (&g.down, cx.hidden), act, y)
}

/// [`ffn`] over `m` rows of `xn` (token-major) into `y`, each row the
/// one-row block's bits (module doc): `m` is 1 to the rows `s` was made for;
/// `kit`'s stage holds at least the [`stage_len`] of the widest block.
#[allow(
    clippy::too_many_arguments,
    reason = "ffn's inputs, the row count and the row kit"
)]
pub(crate) fn ffn_rows<R: RouterEntry>(
    cx: &GqaCx<'_>,
    k: &FfnKernels<R>,
    acts: &mut Acts,
    (rule, scale): (Act, f32),
    (l, f): (usize, &Ffn),
    m: usize,
    xn: &DeviceBuffer<f32>,
    s: &mut FfnRows<R>,
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
        stages,
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
            let rin = RouteIn {
                w: w.f32_tensor(WHAT, &r.router)?,
                bias: Some(w.f32_buf(WHAT, &r.bias)?),
                scale,
                fault: gpu.layer_sink(l)?,
            };
            let rows = RoutedRows {
                probs,
                logits: None,
                ids,
                weights,
            };
            k.router
                .route_rows((stream, gpu, cx.g32), xn, &rin, m, rows)?;
            if let Some(c) = crate::imatrix::collector() {
                c.feed_dense(gpu, &r.router, xn, (m, cx.hidden))?;
            }
            let Stages { latent, normed } = stages;
            let mut lat = latent_of(l, r, latent.as_mut())?;
            let ex: &DeviceBuffer<f32> = match lat.as_mut() {
                Some(LatentRun { pair, x, .. }) => {
                    let width = pair.width;
                    rows_kit::project_rows(cx, &mut kit, acts, (l, m), (&pair.down, width), xn, x)?;
                    x
                }
                None => xn,
            };
            let picks = (&*ids, &*weights);
            routed_sum::<R>(gpu, w, &k.leg, (l, r, rule), (ex, m), picks, leg, routed)?;
            let summed = sum_norm(cx, (l, r), (&*routed, normed.as_mut()), m)?;
            let delta: &DeviceBuffer<f32> = match lat {
                Some(LatentRun { pair, up, .. }) => {
                    rows_kit::project_rows(
                        cx,
                        &mut kit,
                        acts,
                        (l, m),
                        (&pair.up, cx.hidden),
                        summed,
                        up,
                    )?;
                    up
                }
                None => summed,
            };
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
                .enqueue_add(stream, delta, shared, m * cx.hidden, y)
        }
    }
}

/// The card leg over every expert of layer `l` for `m` rows of `xn` (the
/// experts' rows: the latent down's output where the layer has one), the
/// router's ids as the slots' places, its weights the sum's, into `acc`.
#[allow(
    clippy::too_many_arguments,
    reason = "the leg's inputs: the card, the weights, the kernels, the layer, the rows, the picks \
              and the buffers"
)]
fn routed_sum<R: RouterEntry>(
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
        n_card: R::N_EXPERT,
        rule,
    };
    leg.enqueue(gpu, bufs, &run, Some(LegSum { w: weights, acc }))?;
    if let Some(c) = crate::imatrix::collector() {
        // Every expert is resident, so `sel` is the picks' expert ids and the gate·up's rows the
        // leg left are the down's input rows.
        let shape = bufs.shape();
        let dims = (m, shape.slots_per_col, shape.n, R::N_EXPERT);
        c.feed_routed(gpu, &r.stacks[0], (xn, sel), dims, false)?;
        c.feed_routed(gpu, &r.stacks[1], (xn, sel), dims, false)?;
        let dims = (m, shape.slots_per_col, shape.ff, R::N_EXPERT);
        c.feed_routed(gpu, &r.stacks[2], (bufs.act_rows(), sel), dims, true)?;
    }
    Ok(())
}
