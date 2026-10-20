//! The routed block with its shared expert: the router `sigmoid_320_8`
//! (so3's), the go to the host leg, the shared expert (`block::enqueue_block`
//! at ff 1280) and the card experts (`card_experts`) in the leg's shadow, the
//! add of the host sum. Over one row (the step: [`front`], [`shadow`],
//! [`back`]) or a unit's rows (the batch: [`Routed::front_rows`] and its
//! three). Both feeds leave the same bits but the shared expert's rows of a
//! unit of [`GATED_FROM`] rows or more, which its GEMMs write.
//!
//! Every layer is a routed layer with one shared expert, so the block has no
//! dense arm. The router, the handoff, the go, the card experts' call and the
//! wait and add around the host leg are the common routed block
//! ([`bloomery_gpu::routed_block`]); this file holds the shared expert and the
//! base the card sum is added to. Its parts around the host leg:
//! - the front: `ffn_norm` RMS of the rows of the residual after the mixer
//!   `x1`, and the router over them: the scores, the selection bias (the
//!   choice only), the top 8, the weights from the unbiased scores
//!   normalized and scaled. The step goes on to the handoff into the
//!   host-mapped image and the go; four launches. The batch downloads the
//!   rows, weights and ids to the host ([`Routed::download`]).
//! - the shadow: while the host computes, the shared expert, its output
//!   summed into `x1` (`x1s`), then the card experts the slot map puts on the
//!   card with `x1s` as the base their sum is added to; none of the latter on
//!   a layer with no card experts. The base is `x1 + shared` before the card
//!   sum is added, in the step and the batch alike.
//! - the back: the wait, then the add of the host's routed sum and the
//!   shadow's `card + x1s` (`x1s` alone on a layer with no card experts).
//!   One launch after the wait.
//!
//! Solar facts that fix this file: the routed experts' SwiGLU and the shared
//! expert's are the description's limits (0: plain); the shared expert has no
//! sigmoid gate; the router's selection bias is in every layer; the shared
//! expert's down is Q6_K or Q4_K by layer, so its block plan is read per layer.

use std::sync::Arc;

use bloomery_gpu::card_experts::{CardBatch, CardExperts, CardFamily};
use bloomery_gpu::gemm::{Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::{BatchKey, Boundary, Hybrid, SlotMap};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::routed_block::{RouteBlock, RouteBufs, RouteDims, RouteLayer, RoutedRows};
use bloomery_gpu::site::{self, Gated, GatedIo, SiteTy, SiteWeight, WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use bloomery_gpu_deepseek41::block::{
    Block, BlockIo, BlockKernels, BlockPlan, BlockScratch, KBlockKernels, enqueue_block,
    enqueue_block_cols,
};
use bloomery_gpu_deepseek41::experts::ExpertKernels;
use bloomery_gpu_deepseek41::router::sigmoid_320_8::{N_EXPERT, N_USED, RouterKernels};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::names;
use models::{Act as ActRule, Ffn, LayerSpec, Score};

/// What the refusals of this file name.
const WHAT: &str = "solar2 ffn";

/// The fewest rows of a unit whose shared expert runs on `site::gated` (one
/// GEMM a site): below it `enqueue_block_cols` runs the step's gemvs on each
/// column (at most 8 columns), so a unit of up to 8 rows writes the step's
/// bits and from it a launch count of 6 replaces 4 a column.
pub const GATED_FROM: usize = 9;

/// The card experts' family: the file's stack names and the router's counts.
pub const CARD: CardFamily = CardFamily {
    what: "solar2 CardExperts",
    n_expert: N_EXPERT,
    n_used: N_USED,
    stacks: names::routed_stacks,
};

/// What the load takes of the model besides the description and the file.
#[derive(Clone, Copy, Debug)]
pub struct RoutedDims {
    /// The model's width.
    pub embd: usize,
    pub rms_eps: f32,
    /// The most rows a unit of the batch holds; at most one: no batch buffers.
    pub batch_rows: usize,
}

/// One layer's shared expert: its tensors, named once at load, and its plan.
struct LayerFfn {
    /// The shared expert's gate, up and down.
    shared: [String; 3],
    plan: BlockPlan,
}

impl LayerFfn {
    /// The shared expert's gate, up and down names.
    fn shared(&self) -> [&str; 3] {
        self.shared.each_ref().map(String::as_str)
    }

    /// Layer `l`'s tensors of a shared expert `ff` wide over `embd` values,
    /// the block plan read from the file's header and checked against the
    /// resident weights.
    fn load(
        (file, w): (&Split, &Weights),
        l: usize,
        (ff, embd): (usize, usize),
    ) -> Result<LayerFfn, GpuError> {
        let shared = [
            names::ffn_gate_shexp(l),
            names::ffn_up_shexp(l),
            names::ffn_down_shexp(l),
        ];
        let tensors = shared.each_ref().map(String::as_str);
        let plan = BlockPlan::of_file(file, WHAT, tensors, (ff, embd))?;
        plan.check_resident(w, WHAT, tensors, (ff, embd))?;
        Ok(LayerFfn { shared, plan })
    }
}

/// Layer `l`'s routed block for the common one: its norm, router and bias
/// tensors (every layer has a bias) and its row in `map`.
fn route_layer(map: &SlotMap, l: usize) -> Result<RouteLayer, GpuError> {
    let row_off = map
        .row_offset(l)
        .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l}: no row in the slot map")))?;
    Ok(RouteLayer {
        norm: names::ffn_norm(l),
        router: names::ffn_gate_inp(l),
        bias: Some(names::exp_probs_b(l)),
        row_off,
    })
}

/// The kernels the routed block launches beyond the common block's and the
/// card experts' own.
struct Kernels {
    /// The shared expert's wide quantize of a batch's rows.
    g32: Gemm32Kernels,
    /// The shared expert's pair.
    experts: ExpertKernels,
    /// The K-quant block's kernels: the shared expert is K-quant in this file.
    block: Option<KBlockKernels>,
}

/// The step's buffers: the router's outputs and each slot's place, and the
/// shared expert's output and the shadow's base.
struct StepBufs {
    route: RouteBufs<RouterKernels>,
    sh_y: DeviceBuffer<f32>,
    x1s: DeviceBuffer<f32>,
}

/// The shared expert on `site::gated` over a unit's rows: the one-expert
/// table, the input and SwiGLU rows in the forms the layers' types read, and
/// the gate and up outputs.
struct SharedWide {
    gemm: GemmKernels,
    route: GemmRoute,
    x: WideActs,
    h: WideActs,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
}

impl SharedWide {
    /// Buffers for up to `cap` rows over the layers' shared experts `per` of
    /// width `ff` on `embd` values. Load-time only.
    fn new(
        gpu: &Gpu,
        per: &[LayerFfn],
        (cap, embd, ff): (usize, usize, usize),
    ) -> Result<SharedWide, GpuError> {
        let stream = gpu.stream();
        let gate_up: Vec<SiteTy> = per
            .iter()
            .flat_map(|l| {
                let [gate, up, _] = l.plan.types();
                [gate, up]
            })
            .collect();
        let down: Vec<SiteTy> = per.iter().map(|l| l.plan.types()[2]).collect();
        Ok(SharedWide {
            gemm: GemmKernels::load(gpu.context())?,
            route: GemmRoute::new(stream, cap, 1)?,
            x: WideActs::new(stream, WideActs::wants_of(&gate_up), cap, embd)?,
            h: WideActs::new(stream, WideActs::wants_of(&down), cap, ff)?,
            g: DeviceBuffer::zeroed(stream, cap * ff)?,
            u: DeviceBuffer::zeroed(stream, cap * ff)?,
        })
    }

    fn bytes(&self) -> usize {
        self.route.bytes()
            + self.x.bytes()
            + self.h.bytes()
            + self.g.num_bytes()
            + self.u.num_bytes()
    }

    /// Layer `l`'s shared expert over the `m` rows of `x` into `y`: the rows
    /// quantized once into the forms gate and up read, the two GEMMs, the
    /// clamped SwiGLU into the form the down reads, the down GEMM. Six
    /// launches, whatever `m`.
    fn run(
        &mut self,
        (gpu, w, g32): (&Gpu, &Weights, &Gemm32Kernels),
        blk: &Block<'_>,
        (m, embd): (usize, usize),
        (x, y): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let fault = gpu.layer_sink(blk.layer)?;
        let SharedWide {
            gemm,
            route,
            x: xa,
            h,
            g,
            u,
        } = self;
        if route.filled() != Some(m) {
            gemm.enqueue_route_dense(gpu.stream(), m, route, fault)?;
        }
        let [gate, up, down] = blk.plan.types();
        let [gate_n, up_n, down_n] = blk.names;
        let kernels = WideKernels {
            gemm: &*gemm,
            g32: Some(g32),
        };
        xa.quantize(gpu, &kernels, fault, x, m, &[gate, up])?;
        let block = Gated {
            w,
            route: &*route,
            gate: (gate, SiteWeight::Resident(gate_n)),
            up: (up, SiteWeight::Resident(up_n)),
            down: (down, SiteWeight::Resident(down_n)),
            ff: blk.ff,
            hidden: embd,
            rule: Act::SwigluClamp { limit: blk.limit },
            input: GemmInput::PerSlot,
        };
        let input = xa.input(x);
        let io = GatedIo {
            x: &input,
            gate: g,
            up: u,
            rows: h.swiglu_rows(None),
            y,
        };
        site::gated(gpu, &kernels, &block, (m, m), io, fault)
    }
}

/// A unit's buffers, for up to `cap` rows of the batch: the normed rows, the
/// router's scores, ids and weights ([`N_EXPERT`] and [`N_USED`] a row), the
/// shared expert's output rows, the shadow's base rows and the card experts'
/// rows. One set serves every unit of a walk: each part writes what the next
/// part of the same item reads, and the next item's front writes only the
/// normed rows and the router's.
struct BatchBufs {
    cap: usize,
    normed: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    x1s: DeviceBuffer<f32>,
    card: Option<CardBatch>,
    /// Made when a unit can reach [`GATED_FROM`] rows.
    wide: Option<SharedWide>,
}

impl BatchBufs {
    fn new(
        gpu: &Gpu,
        (card, per): (&CardExperts, &[LayerFfn]),
        (cap, embd, ff): (usize, usize, usize),
    ) -> Result<Self, GpuError> {
        let stream = gpu.stream();
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(BatchBufs {
            cap,
            normed: z(cap * embd)?,
            probs: z(cap * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, cap * N_USED)?,
            weights: z(cap * N_USED)?,
            sh: z(cap * embd)?,
            x1s: z(cap * embd)?,
            card: card.any().then(|| card.batch(gpu, cap)).transpose()?,
            wide: (cap >= GATED_FROM)
                .then(|| SharedWide::new(gpu, per, (cap, embd, ff)))
                .transpose()?,
        })
    }

    fn bytes(&self) -> usize {
        [
            &self.normed,
            &self.probs,
            &self.weights,
            &self.sh,
            &self.x1s,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.ids.num_bytes()
            + self.card.as_ref().map_or(0, CardBatch::bytes)
            + self.wide.as_ref().map_or(0, SharedWide::bytes)
    }

    /// `m` rows are a unit this buffer set holds.
    fn check(&self, l: usize, m: usize) -> Result<(), GpuError> {
        if m == 0 || m > self.cap {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l}: {m} rows through batch buffers of {}", self.cap),
            ));
        }
        Ok(())
    }
}

/// The routed blocks of every layer: the common block, the shared experts'
/// plans, the kernels, the card experts, and the step's and the batch's
/// buffers.
pub struct Routed {
    d: RoutedDims,
    layers: Vec<LayerFfn>,
    k: Kernels,
    /// The router, the handoff and the layers' tensors and slot-map rows.
    route: RouteBlock<RouterKernels>,
    card: CardExperts,
    /// The routed experts' and the shared expert's SwiGLU limits (0: plain).
    limits: (f32, f32),
    /// The shared expert's width.
    shared_ff: usize,
    /// The shared expert's `h` row and its K-quant block's buffers, one row at
    /// a time in the step and the batch alike.
    h: DeviceBuffer<f32>,
    block: Option<BlockScratch>,
    step: StepBufs,
    /// The batch's buffers, on a load that runs one.
    batch: Option<BatchBufs>,
}

/// The description of layer `l`'s routed block, or a named refusal.
fn moe_of(layers: &[LayerSpec], l: usize) -> Result<&models::Moe, GpuError> {
    match layers.get(l).map(|spec| &spec.ffn) {
        Some(Ffn::Moe(m)) => Ok(m),
        _ => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: not a routed block; this program runs routed blocks only"),
        )),
    }
}

/// A SwiGLU rule's limit; any other activation is refused by name.
fn swiglu_limit(what: &str, l: usize, act: ActRule) -> Result<f32, GpuError> {
    match act {
        ActRule::SwiGlu { limit } => Ok(limit.unwrap_or(0.0)),
        other => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: the {what} is {other:?}; this program runs SwiGLU"),
        )),
    }
}

/// What layers agree on: the first's `value`, every other layer's equal to it.
fn agreed<T: PartialEq + std::fmt::Debug>(
    what: &str,
    l: usize,
    first: &mut Option<T>,
    value: T,
) -> Result<(), GpuError> {
    match first {
        Some(f) if *f != value => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: {what} is {value:?}, an earlier layer's {f:?}"),
        )),
        _ => {
            first.get_or_insert(value);
            Ok(())
        }
    }
}

/// What the description's routed blocks agree on, every layer's the same.
struct Described {
    /// `expert_weights_scale`.
    scale: f32,
    expert_ff: usize,
    shared_ff: usize,
    /// The routed experts' and the shared expert's SwiGLU limits.
    limits: (f32, f32),
}

/// The description of `layers` as the compiled router, handoff and shared
/// block run it: 320 experts, the top 8, sigmoid scores with a bias and
/// normalized weights, a SwiGLU shared expert without a gate. Anything else
/// is refused by name with its layer.
fn describe(layers: &[LayerSpec]) -> Result<Described, GpuError> {
    let (mut scale, mut expert_ff, mut shared_ff) = (None, None, None);
    let (mut routed_limit, mut shared_limit) = (None, None);
    for l in 0..layers.len() {
        let m = moe_of(layers, l)?;
        let Some(shared) = m.shared else {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l}: no shared expert in the description"),
            ));
        };
        let r = m.router;
        if m.experts as usize != N_EXPERT
            || m.top_k as usize != N_USED
            || m.latent.is_some()
            || r.score != Score::Sigmoid
            || !r.bias
            || !r.norm
            || r.hash
            || shared.sigmoid_gate
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: {m:?} is not the {N_EXPERT}-expert top-{N_USED} sigmoid router \
                     with a bias, normalized weights and an ungated shared expert"
                ),
            ));
        }
        let routed = swiglu_limit("routed activation", l, m.act)?;
        let shared_l = swiglu_limit("shared activation", l, shared.act)?;
        agreed("the weights' scale", l, &mut scale, r.scale.to_bits())?;
        agreed("the routed expert width", l, &mut expert_ff, m.expert_ff)?;
        agreed("the shared expert width", l, &mut shared_ff, shared.ff)?;
        agreed("the routed limit", l, &mut routed_limit, routed.to_bits())?;
        agreed("the shared limit", l, &mut shared_limit, shared_l.to_bits())?;
    }
    let (Some(scale), Some(expert_ff), Some(shared_ff), Some(routed), Some(shared)) =
        (scale, expert_ff, shared_ff, routed_limit, shared_limit)
    else {
        return Err(GpuError::shape(WHAT, "a description with no layer"));
    };
    Ok(Described {
        scale: f32::from_bits(scale),
        expert_ff: expert_ff as usize,
        shared_ff: shared_ff as usize,
        limits: (f32::from_bits(routed), f32::from_bits(shared)),
    })
}

impl Routed {
    /// The routed blocks of `layers` (the description, [`describe`]) as the
    /// file `file` and the resident weights `w` hold them, the card experts as
    /// `map` places them and `slots` is the map's card copy. Load-time only.
    pub fn new(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        (layers, map, slots): (&[LayerSpec], &SlotMap, &Arc<DeviceTensor<u32>>),
        d: RoutedDims,
    ) -> Result<Routed, GpuError> {
        let desc = describe(layers)?;
        let per = (0..layers.len())
            .map(|l| LayerFfn::load((file, w), l, (desc.shared_ff, d.embd)))
            .collect::<Result<Vec<_>, _>>()?;
        let routes = (0..layers.len())
            .map(|l| route_layer(map, l).map(Some))
            .collect::<Result<Vec<_>, _>>()?;
        let route_dims = RouteDims {
            embd: d.embd,
            rms_eps: d.rms_eps,
            scale: desc.scale,
        };
        let route = RouteBlock::new(gpu, w, WHAT, (routes, slots), route_dims)?;
        let ctx = gpu.context();
        let stream = gpu.stream();
        let k = Kernels {
            g32: Gemm32Kernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            block: per
                .iter()
                .any(|l| !l.plan.is_q8())
                .then(|| KBlockKernels::load(gpu))
                .transpose()?,
        };
        let blocks = per.iter().map(|l| (l.plan, desc.shared_ff));
        let block = BlockScratch::new(stream, d.embd, blocks)?;
        let card = CardExperts::new(gpu, w, CARD, layers, map, (d.embd, desc.expert_ff), 1)?;
        let batch = (d.batch_rows > 1)
            .then(|| BatchBufs::new(gpu, (&card, &per), (d.batch_rows, d.embd, desc.shared_ff)))
            .transpose()?;
        Ok(Routed {
            layers: per,
            k,
            route,
            card,
            limits: desc.limits,
            shared_ff: desc.shared_ff,
            h: DeviceBuffer::zeroed(stream, desc.shared_ff)?,
            block,
            step: StepBufs {
                route: RouteBufs::new(stream, 1)?,
                sh_y: DeviceBuffer::zeroed(stream, d.embd)?,
                x1s: DeviceBuffer::zeroed(stream, d.embd)?,
            },
            batch,
            d,
        })
    }

    /// Device bytes of the step's buffers, the card experts' and the kernels'
    /// included; the batch's are [`Routed::batch_bytes`].
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.card.bytes()
            + self.block.as_ref().map_or(0, BlockScratch::bytes)
            + self.k.block.as_ref().map_or(0, KBlockKernels::device_bytes)
            + self.h.num_bytes()
            + self.route.bytes()
            + self.step.route.bytes()
            + self.step.sh_y.num_bytes()
            + self.step.x1s.num_bytes()
    }

    /// Device bytes of the batch's buffers.
    #[must_use]
    pub fn batch_bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, BatchBufs::bytes)
    }
}

/// Layer `l`'s shared expert, or a named refusal.
fn layer_of(layers: &[LayerFfn], l: usize) -> Result<&LayerFfn, GpuError> {
    layers
        .get(l)
        .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l} of a program of {}", layers.len())))
}

/// Enqueue layer `l`'s routed block of the step up to its go (module doc):
/// the norm of `x1` into the boundary's handoff region, the router, the
/// handoff and the go.
pub fn front(
    gpu: &Gpu,
    w: &Weights,
    r: &mut Routed,
    x1: &DeviceBuffer<f32>,
    (hybrid, l): (&mut Hybrid<HostRun>, usize),
) -> Result<(), GpuError> {
    r.route.front((gpu, w), &mut r.step.route, x1, (hybrid, l))
}

/// The shared expert's launch of layer `l`: the kernels and the block, whose
/// width, limit and layer are `(ff, limit, l)`.
fn shared_launch<'a>(
    k: &'a Kernels,
    layer: &'a LayerFfn,
    (ff, limit, l): (usize, f32, usize),
) -> (BlockKernels<'a>, Block<'a>) {
    let kernels = BlockKernels {
        experts: &k.experts,
        kquant: k.block.as_ref(),
    };
    let blk = Block {
        plan: layer.plan,
        names: layer.shared(),
        ff,
        limit,
        layer: l,
    };
    (kernels, blk)
}

/// Enqueue layer `l`'s shared expert and card experts of the step in its host
/// leg's shadow, after its [`front`]: `x1s` is `x1` plus the shared expert's
/// output, and the card sum plus `x1s` goes to the card experts' `pre` where
/// the layer has any.
pub fn shadow(
    gpu: &Gpu,
    w: &Weights,
    r: &mut Routed,
    x1: &DeviceBuffer<f32>,
    (boundary, l): (&Boundary, usize),
) -> Result<(), GpuError> {
    let layer = layer_of(&r.layers, l)?;
    let stream = gpu.stream();
    let (kernels, blk) = shared_launch(&r.k, layer, (r.shared_ff, r.limits.1, l));
    let io = BlockIo {
        x: boundary.normed(),
        h: &mut r.h,
        y: &mut r.step.sh_y,
    };
    enqueue_block(gpu, w, &kernels, &mut r.block, &blk, io)?;
    gpu.elem()
        .enqueue_add(stream, x1, &r.step.sh_y, r.d.embd, &mut r.step.x1s)?;
    let rule = Act::SwigluClamp { limit: r.limits.0 };
    r.route.shadow(
        (gpu, w),
        &mut r.card,
        &r.step.route,
        (boundary, l),
        (&r.step.x1s, rule),
    )
}

/// Enqueue layer `l`'s wait and the add of the host's routed sum to the
/// shadow's (module doc) into `dest`, after its [`shadow`].
pub fn back(
    gpu: &Gpu,
    r: &Routed,
    hybrid: &Hybrid<HostRun>,
    l: usize,
    dest: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    r.route.back(gpu, hybrid, &r.card, l, (&r.step.x1s, dest))
}

/// The batch: the parts of a unit of `m` rows, the step's launches over its
/// rows. A unit runs `front_rows`, `download`, `shadow_rows` and `back_rows`
/// in that order, no other unit's `front_rows` between the first two and the
/// next two.
impl Routed {
    /// Layer `l`'s `ffn_norm` of the unit's `m` rows of the residual `x1` and
    /// the router over them, into the batch's buffers.
    pub fn front_rows(
        &mut self,
        (gpu, w): (&Gpu, &Weights),
        (l, m): (usize, usize),
        x1: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let b = self.batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let rows = RoutedRows {
            probs: &mut b.probs,
            ids: &mut b.ids,
            weights: &mut b.weights,
        };
        self.route
            .route_rows((gpu, w), (l, m), x1, (&mut b.normed, rows))
    }

    /// The copies of [`Routed::front_rows`]'s normed rows, weights and ids to
    /// the host's next batch set under `key`.
    pub fn download(
        &self,
        hybrid: &mut Hybrid<HostRun>,
        stream: &CudaStream,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        let b = self.batch.as_ref().ok_or_else(no_batch)?;
        hybrid.enqueue_download(stream, [&b.normed, &b.weights], &b.ids, key)
    }

    /// Layer `l`'s shared expert and, where the slot map puts any, card
    /// experts over the `m` rows [`Routed::front_rows`] wrote, under the host
    /// leg: the shared expert's rows (per column below [`GATED_FROM`] rows, on
    /// `site::gated` from it), `x1s = x1 + shared`, and the card sum plus
    /// `x1s`.
    pub fn shadow_rows(
        &mut self,
        (gpu, w): (&Gpu, &Weights),
        (l, m): (usize, usize),
        x1: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Routed {
            layers,
            k,
            route,
            card,
            limits,
            shared_ff,
            h,
            block,
            batch,
            d,
            ..
        } = self;
        let b = batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let layer = layer_of(layers, l)?;
        let (kernels, blk) = shared_launch(k, layer, (*shared_ff, limits.1, l));
        let n = d.embd;
        if m < GATED_FROM {
            let xs = span(WHAT, &b.normed, 0, m * n)?;
            let mut ys = span_mut(WHAT, &mut b.sh, 0, m * n)?;
            let io = BlockIo {
                x: &xs,
                h: &mut *h,
                y: &mut ys,
            };
            enqueue_block_cols(gpu, w, &kernels, block, &blk, m, io)?;
        } else {
            let wide = b.wide.as_mut().ok_or(GpuError::State {
                what: WHAT,
                missing: "the batch's GEMM buffers of the shared expert",
            })?;
            wide.run((gpu, w, &k.g32), &blk, (m, n), (&b.normed, &mut b.sh))?;
        }
        gpu.elem()
            .enqueue_add(gpu.stream(), x1, &b.sh, m * n, &mut b.x1s)?;
        let rule = Act::SwigluClamp { limit: limits.0 };
        route.card_rows(
            (gpu, w),
            (card, b.card.as_mut()),
            (l, m),
            (&b.x1s, rule),
            (&b.normed, &b.weights, &b.ids),
        )
    }

    /// `x = hsum +` the shadow's rows (the card sum plus `x1s`, or `x1s` on a
    /// layer with no card experts) over the unit's `m` rows, after
    /// [`Routed::shadow_rows`] and the host sums' upload.
    pub fn back_rows(
        &self,
        gpu: &Gpu,
        (l, m): (usize, usize),
        (hsum, x): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let b = self.batch.as_ref().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let shadowed = match b.card.as_ref().filter(|_| self.card.has(l)) {
            Some(cb) => cb.pre(),
            None => &b.x1s,
        };
        gpu.elem()
            .enqueue_add(gpu.stream(), hsum, shadowed, m * self.d.embd, x)
    }
}

/// A batch call on a load that runs no batch.
fn no_batch() -> GpuError {
    GpuError::State {
        what: WHAT,
        missing: "the batch's buffers",
    }
}
