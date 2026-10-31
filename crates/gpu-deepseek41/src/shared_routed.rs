//! The routed block with one ungated shared expert, over a sigmoid router instance: the router (`R`), the go to the
//! host leg, the shared expert ([`block::enqueue_block_act`]) and the card experts
//! ([`bloomery_gpu::card_experts`]) in the leg's shadow, the add of the host sum. Over one row (the step: [`front`],
//! [`shadow`], [`back`]) or a unit's rows (the batch: [`Routed::front_rows`] and its three). Both feeds leave the same
//! bits but the shared expert's rows of a unit of [`GATED_FROM`] rows or more, which its GEMMs write.
//!
//! Dense leading layers run the common dense block; routed layers have one ungated shared expert. The router,
//! handoff, go, card experts and host wait are the common routed block ([`bloomery_gpu::routed_block`]).
//! [`FfnConfig`] resolves the residual association and expert-input range protection once at load:
//! - the front normalizes the attention residual, scores and selects through `R`'s normalization and bias
//!   policy, then hands the normed row and picks to the host;
//! - the shadow computes the shared expert and card experts while the host runs. The default saves
//!   `residual + shared` as the base. [`ResidualOrder::RoutedSharedResidual`] keeps the residual separate;
//! - the back joins the host and card routed sums. The default adds the host to the card-plus-base rows;
//!   the other association computes `(routed + shared) + residual`, after the complete routed sum is ready.
//!
//! The step and prompt batch use the same loaded association, including tier-card joins.
//!
//! A load whose slot map names an expert tier card ([`TierSide`]) runs a layer the tier holds experts of as a tier
//! layer: the front's handoff also writes the tier's image and each slot's tier place and the go is the host's and the
//! tier's; the shadow runs the card slots' downs but not their sum; the back is the common step join
//! ([`crate::tier_join`]): it waits for the host and the tier, then sums the card's and the tier's rows in one weighted
//! sum (`ds41_ffn_card_acc_8_tier`, the card sum's order with the tier's slots in it) plus the shadow's base, and adds
//! the host's sum. A unit of the batch does the same through [`Routed::download_rows`], [`Routed::shadow_rows`] and
//! [`Routed::join_rows`] (the host leg's [`bloomery_gpu::host::BatchLeg::join_tiered`] hands the tier's rows to the
//! join, then [`Routed::back_rows`]).
//!
//! What a family fixes is its router instance, its [`CardFamily`] (stack names, counts), the loaded arithmetic
//! facts, and its description: the routed and shared activations ([`rule_of`]), scales and widths
//! ([`describe`]). Shared and dense block plans are read per layer and support mixed sites, including IQ4_XS.
//! A Q8_0 shared pair runs the clamp alone; a SwiGLU-OAI shared expert on it is refused before launch.

use std::sync::Arc;

use bloomery_gpu::card_experts::{CardBatch, CardExperts, CardFamily};
use bloomery_gpu::card_leg::ExpertInput;
use bloomery_gpu::gemm::{Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
use bloomery_gpu::host::leg_tier::TierSide;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::xstream::{LandBatch, RingLayer};
use bloomery_gpu::hybrid::{BatchKey, Boundary, Hybrid, SlotMap};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::model::agreed;
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::routed_block::{RouteBlock, RouteBufs, RouteDims, RouteLayer, RoutedRows};
use bloomery_gpu::site::{self, Gated, GatedIo, SiteTy, SiteWeight, WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};

use crate::block::{
    Block, BlockIo, BlockKernels, BlockPlan, BlockScratch, DenseIo, KBlockKernels,
    enqueue_block_act, enqueue_block_cols_act, enqueue_dense_layer,
};
use crate::chain::ffn::FfnBatchKernels;
use crate::experts::ExpertKernels;
use crate::span::{span, span_mut};
use crate::tier_join::{self, TierBatch, TierSlots};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::add_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::Split;
use model::arch::names;
use model::placement::kernels::ffn;
use models::{Act as ActRule, Ffn, LayerSpec, Score};

/// What the refusals of this file name.
const WHAT: &str = "shared routed block";

/// The fewest rows of a unit whose shared expert runs on `site::gated` (one
/// GEMM a site): below it `enqueue_block_cols` runs the step's gemvs on each
/// column (at most 8 columns), so a unit of up to 8 rows writes the step's
/// bits and from it a launch count of 6 replaces 4 a column.
pub const GATED_FROM: usize = ffn::GATED_FROM as usize;

/// The association of a routed block's shared output and attention residual.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidualOrder {
    /// The shared output joins the residual before the routed sum.
    #[default]
    ResidualSharedRouted,
    /// The complete routed sum joins the shared output before the residual.
    RoutedSharedResidual,
}

/// Arithmetic facts resolved once when the routed block is loaded.
#[derive(Clone, Copy, Debug)]
pub struct FfnConfig {
    pub residual: ResidualOrder,
    pub expert_input: ExpertInput,
}

impl Default for FfnConfig {
    fn default() -> Self {
        Self {
            residual: ResidualOrder::ResidualSharedRouted,
            expert_input: ExpertInput::Direct,
        }
    }
}

#[cuda_module]
mod residual_kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            host.len() >= n,
            card.len() >= n,
            shared.len() >= n,
            residual.len() >= n,
            dest.len() >= n
        )
    )]
    pub fn routed_shared_residual(
        host: &[f32],
        card: &[f32],
        shared: &[f32],
        residual: &[f32],
        has_card: u32,
        n: u32,
        mut dest: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the launch bounds every input and output by n; i is guarded.
        unsafe {
            let mut routed = *host.get_unchecked(i);
            if has_card != 0 {
                routed = add_rn_f32(routed, *card.get_unchecked(i));
            }
            let ffn = add_rn_f32(routed, *shared.get_unchecked(i));
            *dest.get_unchecked_mut(i) = add_rn_f32(ffn, *residual.get_unchecked(i));
        }
    }
}

struct ResidualKernels {
    module: residual_kernels::LoadedModule,
}

impl ResidualKernels {
    fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
        // SAFETY: this package owns the embedded bundle; enqueue checks its contract.
        let module = unsafe { bloomery_gpu::shared_module!(residual_kernels, ctx)? };
        Ok(Self { module })
    }

    fn enqueue(
        &self,
        stream: &CudaStream,
        (host, card): (&DeviceBuffer<f32>, Option<&DeviceBuffer<f32>>),
        (shared, residual): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        n: usize,
        dest: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0
            || host.len() < n
            || card.is_some_and(|c| c.len() < n)
            || shared.len() < n
            || residual.len() < n
            || dest.len() < n
        {
            return Err(GpuError::shape(
                WHAT,
                "routed residual join outside its buffers",
            ));
        }
        let n = bloomery_gpu::launch_u32(WHAT, "residual values", n)?;
        let prep = self
            .module
            .prepare_routed_shared_residual(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.routed_shared_residual(
            stream,
            &prep,
            host,
            card.unwrap_or(host),
            shared,
            residual,
            u32::from(card.is_some()),
            n,
            dest,
        )?;
        Ok(())
    }
}

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

/// One dense layer's block: the layer's three tensors, its plan, width, rule and `ffn_norm`.
struct DenseFfn {
    names: [String; 3],
    plan: BlockPlan,
    ff: usize,
    rule: Act,
    norm: String,
}

impl DenseFfn {
    /// Layer `l`'s dense block of `ff` rows over `embd` values under `rule`, the plan read from the file's header and
    /// checked against the resident weights.
    fn load(
        (file, w): (&Split, &Weights),
        l: usize,
        (ff, embd, rule): (usize, usize, Act),
    ) -> Result<DenseFfn, GpuError> {
        let names = [names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)];
        let tensors = names.each_ref().map(String::as_str);
        let plan = BlockPlan::of_file(file, WHAT, tensors, (ff, embd))?;
        plan.check_resident(w, WHAT, tensors, (ff, embd))?;
        Ok(DenseFfn {
            names,
            plan,
            ff,
            rule,
            norm: names::ffn_norm(l),
        })
    }
}

/// Layer `l`'s norm, router, optional selection bias and row in `map`.
fn route_layer(map: &SlotMap, l: usize, bias: bool) -> Result<RouteLayer, GpuError> {
    let row_off = map
        .row_offset(l)
        .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l}: no row in the slot map")))?;
    Ok(RouteLayer {
        norm: names::ffn_norm(l),
        router: names::ffn_gate_inp(l),
        bias: bias.then(|| names::exp_probs_b(l)),
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
    /// Loaded only for the routed-plus-shared, then residual association.
    residual: Option<ResidualKernels>,
}

/// The step's buffers: the router's outputs and each slot's place, and the
/// shared expert's output and the shadow's base.
struct StepBufs<R: RouterEntry> {
    route: RouteBufs<R>,
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

fn wide_layout(
    (gate_up, down): (&[SiteTy], &[SiteTy]),
    (embd, ff): (usize, usize),
) -> ffn::WideBlock {
    let forms = |types: &[SiteTy]| {
        types.iter().fold(ffn::WideForms::default(), |forms, ty| {
            forms.with(ty.reads())
        })
    };
    ffn::WideBlock {
        hidden: embd as u64,
        ff: ff as u64,
        input: forms(gate_up).without_f32(),
        down: forms(down),
    }
}

fn device_forms(forms: ffn::WideForms) -> site::WideForms {
    site::WideForms {
        q128: forms.quant.q128,
        q32: forms.quant.q32,
        f32: forms.f32,
    }
}

impl SharedWide {
    /// Buffers for up to `cap` rows over a block of width `ff` on `embd` values, whose gate and up read
    /// the types `gate_up` and whose down reads the types `down`. Load-time only.
    fn new(
        gpu: &Gpu,
        (gate_up, down): (&[SiteTy], &[SiteTy]),
        (cap, embd, ff): (usize, usize, usize),
    ) -> Result<SharedWide, GpuError> {
        let stream = gpu.stream();
        let layout = wide_layout((gate_up, down), (embd, ff));
        Ok(SharedWide {
            gemm: GemmKernels::load(gpu.context())?,
            route: GemmRoute::new(stream, cap, 1)?,
            x: WideActs::new(stream, device_forms(layout.input), cap, embd)?,
            h: WideActs::new(stream, device_forms(layout.down), cap, ff)?,
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
        (blk, rule): (&Block<'_>, Act),
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
            rule,
            input: GemmInput::PerSlot,
        };
        let input = xa.input(x);
        let io = GatedIo {
            x: &input,
            gate: g,
            up: u,
            rows: h.swiglu_rows(),
            y,
        };
        site::gated(gpu, &kernels, &block, (m, m), io, fault)
    }
}

/// A unit's buffers, for up to `cap` rows of the batch: the normed rows, the
/// router's scores, ids and weights (`R::N_EXPERT` and `R::N_USED` a row), the
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
    /// The shared expert's `h` row: the cols path writes a row a column, so a unit of
    /// fewer rows than [`GATED_FROM`] needs a row a column.
    h: DeviceBuffer<f32>,
    /// The dense blocks' normed rows and their output, over the unit's rows.
    /// A plan with no dense block holds none of them.
    dense_xn: Option<DeviceBuffer<f32>>,
    dense_out: Option<DeviceBuffer<f32>>,
    /// Made when the plan holds a dense block and a unit can reach [`GATED_FROM`] rows.
    dense_wide: Option<SharedWide>,
    /// The tier layers' places and the join's copies, on a load with an expert tier card.
    tier: Option<TierBatch>,
}

/// The types the blocks' gate and up read, and the types their down reads, of the plans `plans` names.
fn block_types<'a>(plans: impl Iterator<Item = &'a BlockPlan>) -> (Vec<SiteTy>, Vec<SiteTy>) {
    let mut gate_up = Vec::new();
    let mut down = Vec::new();
    for p in plans {
        let [gate, up, d] = p.types();
        gate_up.push(gate);
        gate_up.push(up);
        down.push(d);
    }
    (gate_up, down)
}

impl BatchBufs {
    fn new<R: RouterEntry>(
        gpu: &Gpu,
        (card, per, dense): (&CardExperts, &[Option<LayerFfn>], &[Option<DenseFfn>]),
        (cap, embd, ff, dense_ff): (usize, usize, usize, usize),
        tiered: bool,
    ) -> Result<Self, GpuError> {
        let stream = gpu.stream();
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let (shared_gate_up, shared_down) = block_types(per.iter().flatten().map(|l| &l.plan));
        let (dense_gate_up, dense_down) = block_types(dense.iter().flatten().map(|l| &l.plan));
        let layout = ffn::RoutedBatch {
            hidden: embd as u64,
            experts: R::N_EXPERT as u64,
            used: R::N_USED as u64,
            shared: wide_layout((&shared_gate_up, &shared_down), (embd, ff)),
            dense: (dense_ff > 0)
                .then(|| wide_layout((&dense_gate_up, &dense_down), (embd, dense_ff))),
            card: (card.any() || card.any_tier()).then(|| card.batch_layout(cap)),
            tiered,
        };
        let expected = layout
            .bytes(cap as u64)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| GpuError::shape(WHAT, "routed batch scratch byte overflow"))?;
        let buffers = BatchBufs {
            cap,
            normed: z(cap * embd)?,
            probs: z(cap * R::N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, cap * R::N_USED)?,
            weights: z(cap * R::N_USED)?,
            sh: z(cap * embd)?,
            x1s: z(cap * embd)?,
            card: (card.any() || card.any_tier())
                .then(|| card.batch(gpu, cap))
                .transpose()?,
            h: z(cap * ff.max(dense_ff))?,
            wide: (cap >= GATED_FROM)
                .then(|| SharedWide::new(gpu, (&shared_gate_up, &shared_down), (cap, embd, ff)))
                .transpose()?,
            dense_xn: (dense_ff > 0).then(|| z(cap * embd)).transpose()?,
            dense_out: (dense_ff > 0).then(|| z(cap * embd)).transpose()?,
            dense_wide: (cap >= GATED_FROM && dense_ff > 0)
                .then(|| SharedWide::new(gpu, (&dense_gate_up, &dense_down), (cap, embd, dense_ff)))
                .transpose()?,
            tier: tiered
                .then(|| TierBatch::new(stream, cap, R::N_USED))
                .transpose()?,
        };
        if buffers.bytes() != expected {
            return Err(GpuError::shape(
                WHAT,
                "routed batch allocation differs from its common scratch descriptor",
            ));
        }
        Ok(buffers)
    }

    fn bytes(&self) -> usize {
        [
            Some(&self.normed),
            Some(&self.probs),
            Some(&self.weights),
            Some(&self.sh),
            Some(&self.x1s),
            Some(&self.h),
            self.dense_xn.as_ref(),
            self.dense_out.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.ids.num_bytes()
            + self.card.as_ref().map_or(0, CardBatch::bytes)
            + self.wide.as_ref().map_or(0, SharedWide::bytes)
            + self.dense_wide.as_ref().map_or(0, SharedWide::bytes)
            + self.tier.as_ref().map_or(0, TierBatch::bytes)
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
pub struct Routed<R: RouterEntry> {
    d: RoutedDims,
    /// Each layer's shared expert; `None` on a dense layer.
    layers: Vec<Option<LayerFfn>>,
    /// Each dense layer's block; `None` on a routed layer.
    dense: Vec<Option<DenseFfn>>,
    k: Kernels,
    /// The router, the handoff and the layers' tensors and slot-map rows.
    route: RouteBlock<R>,
    card: CardExperts,
    /// The routed experts' and the shared expert's activations.
    rules: (Act, Act),
    /// The shared expert's width.
    shared_ff: usize,
    /// The shared expert's and the dense blocks' `h` row, and their K-quant blocks' buffers, one row at
    /// a time in the step and the batch alike.
    h: DeviceBuffer<f32>,
    block: Option<BlockScratch>,
    step: StepBufs<R>,
    /// The batch's buffers, on a load that runs one.
    batch: Option<BatchBufs>,
    /// The stage card's side of the tier layers, and the kernel of their card sum, on a load whose slot map names an
    /// expert tier card.
    tier: Option<(TierSide, FfnBatchKernels)>,
}

/// The description of layer `l`'s routed block; `None` for a dense layer.
fn moe_of(layers: &[LayerSpec], l: usize) -> Option<&models::Moe> {
    match layers.get(l).map(|spec| &spec.ffn) {
        Some(Ffn::Moe(m)) => Some(m),
        _ => None,
    }
}

/// The device rule of a description's activation: a SwiGLU clamp (no limit: plain) or SwiGLU-OAI; any other
/// activation is refused by name.
pub fn rule_of(what: &str, l: usize, act: ActRule) -> Result<Act, GpuError> {
    match act {
        ActRule::SwiGlu { limit } => Ok(Act::SwigluClamp {
            limit: limit.unwrap_or(0.0),
        }),
        ActRule::SwigluOai { alpha, limit } => Ok(Act::SwigluOai { alpha, limit }),
        other => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: the {what} is {other:?}; this block runs SwiGLU and SwiGLU-OAI"),
        )),
    }
}

/// What the description's routed blocks agree on, every layer's the same.
struct Described {
    /// `expert_weights_scale`.
    scale: f32,
    expert_ff: usize,
    shared_ff: usize,
    /// The routed experts' and the shared expert's activations.
    rules: (Act, Act),
}

/// The description of `layers` as the compiled router, handoff and shared
/// block run it: `R`'s experts, picks, normalization and bias policy; sigmoid scores and a
/// SwiGLU or SwiGLU-OAI shared expert without a gate.
/// Anything else is refused by name with its layer.
fn describe<R: RouterEntry>(layers: &[LayerSpec]) -> Result<Described, GpuError> {
    let (mut scale, mut expert_ff, mut shared_ff) = (None, None, None);
    let (mut routed_rule, mut shared_rule) = (None, None);
    for l in 0..layers.len() {
        let Some(m) = moe_of(layers, l) else {
            continue;
        };
        let Some(shared) = m.shared else {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l}: no shared expert in the description"),
            ));
        };
        let r = m.router;
        if m.experts as usize != R::N_EXPERT
            || m.top_k as usize != R::N_USED
            || m.latent.is_some()
            || r.score != Score::Sigmoid
            || (!r.bias && !R::OPTIONAL_BIAS)
            || r.norm != R::NORM
            || r.hash
            || shared.sigmoid_gate
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: {m:?} is not the {}-expert top-{} sigmoid router \
                     with {:?} normalization, optional bias={} and an ungated shared expert",
                    R::N_EXPERT,
                    R::N_USED,
                    R::NORM,
                    R::OPTIONAL_BIAS
                ),
            ));
        }
        let routed = rule_of("routed activation", l, m.act)?;
        let shared_l = rule_of("shared activation", l, shared.act)?;
        agreed(WHAT, "the weights' scale", l, &mut scale, r.scale.to_bits())?;
        agreed(
            WHAT,
            "the routed expert width",
            l,
            &mut expert_ff,
            m.expert_ff,
        )?;
        agreed(
            WHAT,
            "the shared expert width",
            l,
            &mut shared_ff,
            shared.ff,
        )?;
        agreed(WHAT, "the routed rule", l, &mut routed_rule, routed)?;
        agreed(WHAT, "the shared rule", l, &mut shared_rule, shared_l)?;
    }
    let (Some(scale), Some(expert_ff), Some(shared_ff), Some(routed), Some(shared)) =
        (scale, expert_ff, shared_ff, routed_rule, shared_rule)
    else {
        return Err(GpuError::shape(WHAT, "a description with no layer"));
    };
    Ok(Described {
        scale: f32::from_bits(scale),
        expert_ff: expert_ff as usize,
        shared_ff: shared_ff as usize,
        rules: (routed, shared),
    })
}

impl<R: RouterEntry> Routed<R> {
    /// The routed blocks of `layers` (the description, [`describe`]) as the
    /// file `file` and the resident weights `w` hold them, the card experts of
    /// the family `card` as `map` places them and `slots` is the map's card
    /// copy. The family's counts are `R`'s, or the load is refused by name.
    /// Load-time only.
    pub fn new(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        (layers, map, slots): (&[LayerSpec], &SlotMap, &Arc<DeviceTensor<u32>>),
        d: RoutedDims,
        card: CardFamily,
    ) -> Result<Routed<R>, GpuError> {
        Self::new_configured(
            gpu,
            file,
            w,
            (layers, map, slots),
            d,
            card,
            FfnConfig::default(),
        )
    }

    /// Load the same common block with explicit arithmetic facts. The default keeps the legacy
    /// association and expert input unchanged; the selected policies apply to every routed layer.
    pub fn new_configured(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        (layers, map, slots): (&[LayerSpec], &SlotMap, &Arc<DeviceTensor<u32>>),
        d: RoutedDims,
        card: CardFamily,
        config: FfnConfig,
    ) -> Result<Routed<R>, GpuError> {
        if (card.n_expert, card.n_used) != (R::N_EXPERT, R::N_USED) {
            return Err(GpuError::shape(
                card.what,
                format!(
                    "card experts of {} and {} a token under a router of {} and {}",
                    card.n_expert,
                    card.n_used,
                    R::N_EXPERT,
                    R::N_USED
                ),
            ));
        }
        let desc = describe::<R>(layers)?;
        let routed = |l: usize| moe_of(layers, l).is_some();
        let per = (0..layers.len())
            .map(|l| {
                routed(l)
                    .then(|| LayerFfn::load((file, w), l, (desc.shared_ff, d.embd)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dense = layers
            .iter()
            .enumerate()
            .map(|(l, spec)| match spec.ffn {
                Ffn::Dense { ff, act } => {
                    let rule = rule_of("dense activation", l, act)?;
                    DenseFfn::load((file, w), l, (ff as usize, d.embd, rule)).map(Some)
                }
                Ffn::Moe(_) => Ok(None),
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let routes = (0..layers.len())
            .map(|l| {
                moe_of(layers, l)
                    .map(|m| route_layer(map, l, m.router.bias))
                    .transpose()
            })
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
            residual: (config.residual == ResidualOrder::RoutedSharedResidual)
                .then(|| ResidualKernels::load(ctx))
                .transpose()?,
            block: per
                .iter()
                .flatten()
                .map(|l| l.plan)
                .chain(dense.iter().flatten().map(|l| l.plan))
                .any(|p| !p.is_q8())
                .then(|| KBlockKernels::load(gpu))
                .transpose()?,
        };
        let blocks = per
            .iter()
            .flatten()
            .map(|l| (l.plan, desc.shared_ff))
            .chain(dense.iter().flatten().map(|l| (l.plan, l.ff)));
        let block = BlockScratch::new(stream, d.embd, blocks)?;
        let h_len = dense
            .iter()
            .flatten()
            .map(|l| l.ff)
            .fold(desc.shared_ff, usize::max);
        let tier = (map.tiers() > 0)
            .then(|| {
                Ok::<_, GpuError>((
                    TierSide::new(gpu, map, layers.len(), (1, R::N_USED))?,
                    FfnBatchKernels::load(ctx)?,
                ))
            })
            .transpose()?;
        let dims = (d.embd, desc.expert_ff);
        let mut card = match tier {
            Some(_) => CardExperts::tiered(gpu, w, card, layers, map, dims, 1)?,
            None => CardExperts::new(gpu, w, card, layers, map, dims, 1)?,
        };
        card.set_expert_input(gpu, config.expert_input)?;
        card.set_add_base(config.residual == ResidualOrder::ResidualSharedRouted);
        let dense_ff = dense.iter().flatten().map(|l| l.ff).max().unwrap_or(0);
        let batch = (d.batch_rows > 1)
            .then(|| {
                BatchBufs::new::<R>(
                    gpu,
                    (&card, &per, &dense),
                    (d.batch_rows, d.embd, desc.shared_ff, dense_ff),
                    tier.is_some(),
                )
            })
            .transpose()?;
        Ok(Routed {
            layers: per,
            dense,
            k,
            route,
            card,
            rules: desc.rules,
            shared_ff: desc.shared_ff,
            h: DeviceBuffer::zeroed(stream, h_len)?,
            block,
            step: StepBufs {
                route: RouteBufs::<R>::new(stream, 1)?,
                sh_y: DeviceBuffer::zeroed(stream, d.embd)?,
                x1s: DeviceBuffer::zeroed(stream, d.embd)?,
            },
            batch,
            tier,
            d,
        })
    }

    /// Layer `l` computes routed experts on the card: the layers a
    /// residency machine holds stage stacks of.
    #[must_use]
    pub fn card_has(&self, l: usize) -> bool {
        self.card.has(l)
    }

    /// The experts of layer `l` the expert tier holds: 0 off the tier, and on
    /// every layer of a load with none.
    #[must_use]
    pub fn tier_k(&self, l: usize) -> usize {
        self.card.tier_k(l)
    }

    /// The stage card's side of the tier layers, on a load whose slot map names
    /// an expert tier card.
    #[must_use]
    pub fn tier(&self) -> Option<&TierSide> {
        self.tier.as_ref().map(|(side, _)| side)
    }

    /// Device bytes of the step's buffers, the card experts' and the kernels'
    /// included; the batch's are [`Routed::batch_bytes`].
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.card.bytes()
            + self.tier().map_or(0, TierSide::bytes)
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

    /// A routed expert's gate, up and down widths: the model's and the
    /// expert's, which the expert stream prices its card costs from.
    #[must_use]
    pub fn expert_widths(&self) -> (usize, usize) {
        (self.d.embd, self.card.ff())
    }

    /// The ring pass's GEMM buffers of the batch's card experts for a half of
    /// `half_slots` slots of the expert stream's ring
    /// ([`CardExperts::arm_ring`]), made when the stream starts. Nothing on a
    /// load with no batch or no card experts in it: the ring pass then
    /// refuses by name.
    pub fn arm_ring(&mut self, gpu: &Gpu, half_slots: usize) -> Result<(), GpuError> {
        match self.batch.as_mut().and_then(|b| b.card.as_mut()) {
            Some(cb) => self.card.arm_ring(gpu, cb, half_slots),
            None => Ok(()),
        }
    }

    /// The last token's routed expert ids, in router rank order: the picks
    /// the block made, for a gate that holds them against a reference's.
    #[must_use]
    pub fn route_ids(&self) -> &DeviceBuffer<u32> {
        self.step.route.ids()
    }

    /// Layer `l`'s selection bias for reference-pick inspection. A layer with no bias is refused by
    /// name; execution handles it through the router's optional-bias policy.
    pub fn route_bias<'a>(
        &'a self,
        w: &'a Weights,
        l: usize,
    ) -> Result<&'a DeviceBuffer<f32>, GpuError> {
        self.route.bias_of(w, l)
    }
}

/// Layer `l`'s shared expert, or a named refusal.
fn layer_of(layers: &[Option<LayerFfn>], l: usize) -> Result<&LayerFfn, GpuError> {
    layers
        .get(l)
        .ok_or_else(|| {
            GpuError::shape(WHAT, format!("layer {l} of a program of {}", layers.len()))
        })?
        .as_ref()
        .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l}: a dense block, not a routed one")))
}

/// Enqueue layer `l`'s routed block of the step up to its go (module doc):
/// the norm of `x1` into the boundary's handoff region, the router, the
/// handoff and the go.
pub fn front<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    r: &mut Routed<R>,
    x1: &DeviceBuffer<f32>,
    (hybrid, l): (&mut Hybrid<HostRun>, usize),
) -> Result<(), GpuError> {
    let side = r.tier.as_mut().map(|(side, _)| side);
    r.route
        .front_tiered((gpu, w), &mut r.step.route, x1, (hybrid, l), side)
}

/// The shared expert's launch of layer `l`: the kernels and the block, whose
/// width and layer are `(ff, l)`; the block's `limit` is the clamp rule's (0
/// under another rule, which the launches read from the rule they are given).
fn shared_launch<'a>(
    k: &'a Kernels,
    layer: &'a LayerFfn,
    (ff, rule, l): (usize, Act, usize),
) -> (BlockKernels<'a>, Block<'a>) {
    let kernels = BlockKernels {
        experts: &k.experts,
        kquant: k.block.as_ref(),
    };
    let blk = Block {
        plan: layer.plan,
        names: layer.shared(),
        ff,
        limit: match rule {
            Act::SwigluClamp { limit } => limit,
            _ => 0.0,
        },
        layer: l,
    };
    (kernels, blk)
}

/// Enqueue layer `l`'s shared and card experts after [`front`], while the host runs. `x1s` holds the
/// legacy residual-plus-shared base, or the saved residual when the loaded association joins it last.
pub fn shadow<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    r: &mut Routed<R>,
    x1: &DeviceBuffer<f32>,
    (boundary, l): (&Boundary, usize),
) -> Result<(), GpuError> {
    let layer = layer_of(&r.layers, l)?;
    let stream = gpu.stream();
    let (kernels, blk) = shared_launch(&r.k, layer, (r.shared_ff, r.rules.1, l));
    let io = BlockIo {
        x: boundary.normed(),
        h: &mut r.h,
        y: &mut r.step.sh_y,
    };
    enqueue_block_act(gpu, w, &kernels, &mut r.block, (&blk, r.rules.1), io)?;
    if r.k.residual.is_some() {
        span_mut(WHAT, &mut r.step.x1s, 0, r.d.embd)?
            .copy_from_device_async(&*span(WHAT, x1, 0, r.d.embd)?, stream)?;
    } else {
        gpu.elem()
            .enqueue_add(stream, x1, &r.step.sh_y, r.d.embd, &mut r.step.x1s)?;
    }
    let rule = r.rules.0;
    r.route.shadow(
        (gpu, w),
        &mut r.card,
        &r.step.route,
        (boundary, l),
        (&r.step.x1s, rule),
    )
}

/// Enqueue layer `l`'s wait and the add of the host's routed sum to the
/// shadow's (module doc) into `dest`, after its [`shadow`]. On a layer the
/// expert tier holds experts of, the wait is the host's and the tier's, and
/// the shadow's rows are the card sum over the stage card's and the tier's
/// slots plus the base, which this enqueues first.
pub fn back<R: RouterEntry>(
    gpu: &Gpu,
    r: &mut Routed<R>,
    hybrid: &Hybrid<HostRun>,
    l: usize,
    dest: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let n_tier = r.card.tier_k(l);
    if n_tier == 0 {
        if let Some(join) = &r.k.residual {
            hybrid.boundary().enqueue_back_of(gpu.stream(), 0, l)?;
            return join.enqueue(
                gpu.stream(),
                (
                    hybrid.boundary().hsum_of(0)?,
                    r.card.has(l).then(|| r.card.acc()),
                ),
                (&r.step.sh_y, &r.step.x1s),
                r.d.embd,
                dest,
            );
        }
        return r.route.back(gpu, hybrid, &r.card, l, (&r.step.x1s, dest));
    }
    const W: &str = "shared routed back (tier layer)";
    let (side, kernels) = r.tier.as_ref().ok_or(GpuError::State {
        what: W,
        missing: "the stage card's tier side",
    })?;
    let slots = TierSlots {
        w: r.step.route.weights(),
        tsel: side.tsel(0)?,
        n: r.d.embd,
        n_card: r.card.n_card(l),
        n_tier,
        n_used: R::N_USED,
    };
    if let Some(residual) = &r.k.residual {
        hybrid.enqueue_tier_back(gpu.stream(), l, 0)?;
        let mut join = r.card.step_join(r.step.route.sel());
        tier_join::enqueue_acc(
            (gpu.stream(), kernels),
            (&mut join, &slots),
            (hybrid.tier_rows(0)?, 1),
        )?;
        return residual.enqueue(
            gpu.stream(),
            (hybrid.boundary().hsum_of(0)?, Some(join.acc)),
            (&r.step.sh_y, &r.step.x1s),
            r.d.embd,
            dest,
        );
    }
    tier_join::enqueue_step_join(
        gpu,
        kernels,
        (hybrid, l, 0),
        (r.card.step_join(r.step.route.sel()), &slots),
        (&r.step.x1s, dest),
    )
}

/// Enqueue dense layer `l`'s feed-forward sub-layer of the step: `dest = x1 + block(rms_norm(x1))` through `xn` and
/// `out` ([`enqueue_dense_layer`]), with the layer's own rule and no host leg.
pub fn dense<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    r: &mut Routed<R>,
    l: usize,
    (x1, xn, out): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    dest: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let layer = r
        .dense
        .get(l)
        .and_then(Option::as_ref)
        .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l}: not a dense block")))?;
    let kernels = BlockKernels {
        experts: &r.k.experts,
        kquant: r.k.block.as_ref(),
    };
    let blk = Block {
        plan: layer.plan,
        names: layer.names.each_ref().map(String::as_str),
        ff: layer.ff,
        limit: 0.0,
        layer: l,
    };
    let norm = w.f32_buf(WHAT, &layer.norm)?;
    let io = DenseIo {
        x1,
        xn,
        h: &mut r.h,
        out,
        dest,
    };
    enqueue_dense_layer(
        gpu,
        w,
        &kernels,
        &mut r.block,
        (&blk, layer.rule),
        (norm, r.d.rms_eps),
        io,
    )
}

/// The batch: the parts of a unit of `m` rows, the step's launches over its
/// rows. A unit runs `front_rows`, `download`, `shadow_rows` (or its two
/// halves, `shared_rows` then `card_rows`) and `back_rows` in that order, no
/// other unit's `front_rows` between the first two and the next two.
impl<R: RouterEntry> Routed<R> {
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

    /// [`Routed::download`] of the unit's `m` rows for a load that may hang an
    /// expert tier card: a layer the tier holds experts of also carries each
    /// slot's tier place, which the tier's service reads
    /// ([`Hybrid::enqueue_download_tiered`]).
    pub fn download_rows(
        &mut self,
        gpu: &Gpu,
        hybrid: &mut Hybrid<HostRun>,
        (key, m): (BatchKey, usize),
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let l = key.layer;
        let Some((side, _)) = self.tier.as_ref().filter(|(side, _)| side.k(l) > 0) else {
            return self.download(hybrid, stream, key);
        };
        let b = self.batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let tb = b.tier.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier places of a batch made with a tier",
        })?;
        self.route
            .tier_places_rows(gpu, (l, m), &b.ids, (side.places(), tb.places_mut()))?;
        hybrid.enqueue_download_tiered(stream, [&b.normed, &b.weights], &b.ids, &[tb.places()], key)
    }

    /// Layer `l`'s shared expert and, where the slot map puts any, card
    /// experts over the `m` rows [`Routed::front_rows`] wrote, under the host
    /// leg: [`Routed::shared_rows`], then [`Routed::card_rows`] with no
    /// landing to wait for.
    pub fn shadow_rows(
        &mut self,
        gw: (&Gpu, &Weights),
        lm: (usize, usize),
        x1: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.shared_rows(gw, lm, x1)?;
        self.card_rows(gw, lm, (&[], None))
    }

    /// Layer `l`'s shared expert over the `m` normed rows, per column below [`GATED_FROM`] and on
    /// `site::gated` from it. Keep the residual-plus-shared base or the separate residual, as loaded.
    pub fn shared_rows(
        &mut self,
        (gpu, w): (&Gpu, &Weights),
        (l, m): (usize, usize),
        x1: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Routed {
            layers,
            k,
            rules,
            shared_ff,
            block,
            batch,
            d,
            ..
        } = self;
        let b = batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let layer = layer_of(layers, l)?;
        let (kernels, blk) = shared_launch(k, layer, (*shared_ff, rules.1, l));
        let n = d.embd;
        if m < GATED_FROM {
            let xs = span(WHAT, &b.normed, 0, m * n)?;
            let mut ys = span_mut(WHAT, &mut b.sh, 0, m * n)?;
            let io = BlockIo {
                x: &xs,
                h: &mut b.h,
                y: &mut ys,
            };
            enqueue_block_cols_act(gpu, w, &kernels, block, (&blk, rules.1), m, io)?;
        } else {
            let wide = b.wide.as_mut().ok_or(GpuError::State {
                what: WHAT,
                missing: "the batch's GEMM buffers of the shared expert",
            })?;
            wide.run(
                (gpu, w, &k.g32),
                (&blk, rules.1),
                (m, n),
                (&b.normed, &mut b.sh),
            )?;
        }
        if k.residual.is_some() {
            span_mut(WHAT, &mut b.x1s, 0, m * n)?
                .copy_from_device_async(&*span(WHAT, x1, 0, m * n)?, gpu.stream())?;
            Ok(())
        } else {
            gpu.elem()
                .enqueue_add(gpu.stream(), x1, &b.sh, m * n, &mut b.x1s)
        }
    }

    /// Layer `l`'s card experts, where the slot map puts any, over the `m`
    /// rows [`Routed::front_rows`] wrote: the card sum plus the base
    /// [`Routed::shared_rows`] wrote, after the engine stream has waited each
    /// landing batch's event of the layer's pick (`land`, the residency
    /// machine's admits into the stage stacks, which the slot map's card
    /// copy the route reads names already). One route over the whole map,
    /// each slot's rows the step's under the same map. With the expert
    /// stream's `ring` of the layer (a walk under `split`) the streamed
    /// experts' slots run over the ring's half into the same card sum, one
    /// GEMM batch a landing batch of the ring, each behind its own batch's
    /// copies, so a batch computes while the next lands
    /// ([`RouteBlock::card_rows_ring`]); the host's sum leaves them out.
    pub fn card_rows(
        &mut self,
        (gpu, w): (&Gpu, &Weights),
        (l, m): (usize, usize),
        (land, ring): (&[LandBatch], Option<&RingLayer>),
    ) -> Result<(), GpuError> {
        let Routed {
            route,
            card,
            rules,
            batch,
            k,
            ..
        } = self;
        let b = batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        for event in land.iter().filter_map(|lb| lb.event.as_deref()) {
            gpu.stream().wait(event)?;
        }
        let rule = rules.0;
        let cb = b.card.as_mut();
        let rows = (&b.normed, &b.weights, &b.ids);
        match ring {
            Some(r) => route.card_rows_ring(
                (gpu, w, &k.g32),
                (card, cb),
                (l, m),
                (&b.x1s, rule),
                rows,
                r,
            )?,
            None => route.card_rows((gpu, w), (card, cb), (l, m), (&b.x1s, rule), rows)?,
        }
        if card.tier_k(l) == 0 {
            return Ok(());
        }
        // The join reads this unit's weights and tier places after the next unit's front has routed: keep them.
        let tb = b.tier.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier places of a batch made with a tier",
        })?;
        tb.keep(gpu.stream(), m * R::N_USED, &b.weights)
    }

    /// Layer `l`'s join of a tier layer over the unit's `m` rows, once the host leg has landed the tier's rows `trows`
    /// ([`bloomery_gpu::host::BatchLeg::join_tiered`], which hands them over, then uploads the host sums): the common
    /// join ([`tier_join::enqueue_join`]) of every token's slots in slot order from the stage card's downs or the
    /// tier's rows, then the shadow's base added — the card sum plus the base of a one-card layer, which
    /// [`Routed::back_rows`] reads. A layer the tier holds no expert of, and a load with no tier, are refused by name.
    pub fn join_rows(
        &mut self,
        gpu: &Gpu,
        (l, m): (usize, usize),
        trows: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const W: &str = "shared routed join (tier layer)";
        let n_tier = self.card.tier_k(l);
        let (Some((_, kernels)), true) = (self.tier.as_ref(), n_tier > 0) else {
            return Err(GpuError::shape(
                W,
                format!("layer {l}: a join of a layer no expert tier holds an expert of"),
            ));
        };
        let n_card = self.card.n_card(l);
        let n = self.d.embd;
        let b = self.batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let (Some(tb), Some(cb)) = (b.tier.as_ref(), b.card.as_mut()) else {
            return Err(GpuError::State {
                what: W,
                missing: "the batch's tier places and card buffers",
            });
        };
        let slots = tb.slots(n, (n_card, n_tier, R::N_USED));
        if self.k.residual.is_some() {
            return tier_join::enqueue_acc(
                (gpu.stream(), kernels),
                (&mut cb.join(), &slots),
                (trows, m),
            );
        }
        tier_join::enqueue_join(
            gpu,
            (gpu.stream(), kernels),
            (&mut cb.join(), &slots),
            (trows, m),
            &b.x1s,
        )
    }

    /// Join the uploaded host sums and shadow results over `m` rows with the loaded association.
    /// On a tier layer, [`Routed::join_rows`] first forms the raw routed sum over both cards.
    pub fn back_rows(
        &self,
        gpu: &Gpu,
        (l, m): (usize, usize),
        (hsum, x): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let b = self.batch.as_ref().ok_or_else(no_batch)?;
        b.check(l, m)?;
        if let Some(join) = &self.k.residual {
            let card = b
                .card
                .as_ref()
                .filter(|_| self.card.sums(l))
                .map(CardBatch::acc);
            return join.enqueue(
                gpu.stream(),
                (hsum, card),
                (&b.sh, &b.x1s),
                m * self.d.embd,
                x,
            );
        }
        let shadowed = match b.card.as_ref().filter(|_| self.card.sums(l)) {
            Some(cb) => cb.pre(),
            None => &b.x1s,
        };
        gpu.elem()
            .enqueue_add(gpu.stream(), hsum, shadowed, m * self.d.embd, x)
    }

    /// Enqueue dense layer `l`'s feed-forward over the unit's `m` rows:
    /// `dest = x1 + block(rms_norm(x1))` with the layer's own rule and no
    /// host leg. The block runs on the wide GEMMs, as the shared expert's
    /// does above [`GATED_FROM`] rows.
    pub fn dense_rows(
        &mut self,
        (gpu, w): (&Gpu, &Weights),
        (l, m): (usize, usize),
        (x1, dest): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let layer = self
            .dense
            .get(l)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l}: not a dense block")))?;
        let b = self.batch.as_mut().ok_or_else(no_batch)?;
        b.check(l, m)?;
        let n = self.d.embd;
        let norm = w.f32_buf(WHAT, &layer.norm)?;
        let dense_xn = b.dense_xn.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's normed rows of the dense blocks",
        })?;
        gpu.elem()
            .enqueue_rms_norm(gpu.stream(), x1, norm, self.d.rms_eps, n, m, dense_xn)?;
        let blk = Block {
            plan: layer.plan,
            names: layer.names.each_ref().map(String::as_str),
            ff: layer.ff,
            limit: 0.0,
            layer: l,
        };
        let wide = b.dense_wide.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's GEMM buffers of the dense blocks",
        })?;
        let dense_out = b.dense_out.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's output rows of the dense blocks",
        })?;
        wide.run(
            (gpu, w, &self.k.g32),
            (&blk, layer.rule),
            (m, n),
            (dense_xn, dense_out),
        )?;
        gpu.elem()
            .enqueue_add(gpu.stream(), x1, dense_out, m * n, dest)
    }
}

/// A batch call on a load that runs no batch.
fn no_batch() -> GpuError {
    GpuError::State {
        what: WHAT,
        missing: "the batch's buffers",
    }
}

#[cfg(test)]
mod tests {
    use super::SharedWide;
    use crate::block::{
        Block, BlockIo, BlockKernels, BlockPlan, BlockScratch, KBlockKernels,
        enqueue_block_cols_act,
    };
    use crate::experts::ExpertKernels;
    use bloomery_gpu::Gpu;
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::kquant::Act;
    use bloomery_gpu::weights::Weights;
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use gguf::write::{Layout, TensorDecl, Writer};

    const N: usize = 256;
    const FF: usize = 512;
    const NAMES: [&str; 3] = [
        "blk.42.ffn_gate_shexp.weight",
        "blk.42.ffn_up_shexp.weight",
        "blk.42.ffn_down_shexp.weight",
    ];

    fn iq4_shared_file() -> (std::path::PathBuf, Split) {
        let path =
            std::env::temp_dir().join(format!("bloomery-iq4-shared-{}.gguf", std::process::id()));
        let specs = [(N, FF, 0x1c00u16), (N, FF, 0x1800), (FF, N, 0x1800)];
        let payloads: Vec<Vec<u8>> = specs
            .iter()
            .enumerate()
            .map(|(i, &(k, rows, scale))| {
                let mut bytes = Vec::with_capacity(rows * (k / 256) * 136);
                for row in 0..rows {
                    let d = if i == 2 && row % 2 == 1 {
                        scale - 0x400
                    } else {
                        scale
                    };
                    let mut block = [0x88u8; 136];
                    block[..2].copy_from_slice(&d.to_le_bytes());
                    block[2..4].copy_from_slice(&0xaaaau16.to_le_bytes());
                    block[4..8].fill(0x11);
                    for _ in 0..k / 256 {
                        bytes.extend_from_slice(&block);
                    }
                }
                bytes
            })
            .collect();
        let decls = NAMES
            .iter()
            .zip(specs)
            .zip(&payloads)
            .map(|((&name, (k, rows, _)), bytes)| TensorDecl {
                name: name.to_owned(),
                dims: vec![u64::try_from(k).unwrap(), u64::try_from(rows).unwrap()],
                type_id: GgmlType::IQ4_XS.as_u32(),
                nbytes: u64::try_from(bytes.len()).unwrap(),
            })
            .collect();
        let layout = Layout::new(&[], decls).unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let mut writer = Writer::new(file, layout).unwrap();
        for (name, bytes) in NAMES.iter().zip(payloads) {
            writer.tensor(name, &bytes).unwrap();
        }
        writer.finish().unwrap();
        let split = Split::open(&path).unwrap();
        (path, split)
    }

    fn check_shared_rows(input: &[f32], rows: &[f32]) {
        // Uniform positive rows quantize to code127 without integer rounding.
        // The gate is at least128, so SiLU rounds to the gate itself. The
        // scalar positive-dot upper bound has four input-width operations,
        // two output-width operations and the scale/product roundings.
        let operations = 4.0 * 256.0 + 2.0 * 512.0 + 16.0;
        let eps = f64::from(f32::EPSILON);
        let gamma = operations * eps / (1.0 - operations * eps);
        for (c, row) in rows.as_chunks::<N>().0.iter().enumerate() {
            let x = f64::from(input[c * N]);
            for (r, &got) in row.iter().enumerate() {
                let expected = x * x * if r % 2 == 0 { 0.5 } else { 0.25 };
                assert!(got.is_finite());
                assert!(
                    (f64::from(got) - expected).abs() <= gamma * expected,
                    "column {c} row {r}: {got} vs {expected}, relative bound {gamma}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_iq4_shared_loads_and_runs_narrow_columns_and_wide_replay() {
        let (path, split) = iq4_shared_file();
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let weights = Weights::load_where(stream, &split, |name| NAMES.contains(&name)).unwrap();
        assert_eq!(
            model::placement::card_format_of_role(GgmlType::IQ4_XS, models::Role::SharedExpert),
            Some(model::placement::CardFormat::KQuant)
        );
        for name in NAMES {
            let Some(bloomery_gpu::weights::DevWeight::KQuant { ty, w, .. }) = weights.get(name)
            else {
                panic!("{name} must retain its IQ4_XS words");
            };
            assert_eq!(*ty, GgmlType::IQ4_XS);
            let words = w.buf().to_host_vec(stream).unwrap();
            let stored: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            assert_eq!(stored, split.data(name).unwrap(), "{name}");
        }
        let plan = BlockPlan::of_file(&split, "IQ4_XS shared", NAMES, (FF, N)).unwrap();
        plan.check_resident(&weights, "IQ4_XS shared", NAMES, (FF, N))
            .unwrap();
        let block = Block {
            plan,
            names: NAMES,
            ff: FF,
            limit: 0.0,
            layer: 42,
        };
        let experts = ExpertKernels::load(gpu.context()).unwrap();
        let quant = KBlockKernels::load(&gpu).unwrap();
        let kernels = BlockKernels {
            experts: &experts,
            kquant: Some(&quant),
        };
        let mut scratch = BlockScratch::new(stream, N, [(plan, FF)]).unwrap();
        let mut h = DeviceBuffer::zeroed(stream, FF).unwrap();
        let g32 = Gemm32Kernels::load(gpu.context()).unwrap();
        for cols in [1, 8, super::GATED_FROM] {
            let input: Vec<f32> = (0..cols)
                .flat_map(|c| std::iter::repeat_n([128.0, 256.0, 384.0][c % 3], N))
                .collect();
            let x = DeviceBuffer::from_host(stream, &input).unwrap();
            let mut y = DeviceBuffer::zeroed(stream, cols * N).unwrap();
            if cols < super::GATED_FROM {
                enqueue_block_cols_act(
                    &gpu,
                    &weights,
                    &kernels,
                    &mut scratch,
                    (&block, Act::SiluMul),
                    cols,
                    BlockIo {
                        x: &x,
                        h: &mut h,
                        y: &mut y,
                    },
                )
                .unwrap();
            } else {
                let types = plan.types();
                let mut wide =
                    SharedWide::new(&gpu, (&types[..2], &types[2..]), (cols, N, FF)).unwrap();
                assert!(wide.x.q128.is_some() && wide.h.q128.is_some());
                assert!(wide.x.q32.is_none() && wide.h.q32.is_none());
                assert!(wide.x.f32.is_none() && wide.h.f32.is_none());
                wide.run(
                    (&gpu, &weights, &g32),
                    (&block, Act::SiluMul),
                    (cols, N),
                    (&x, &mut y),
                )
                .unwrap();
                let eager = y.to_host_vec(stream).unwrap();
                check_shared_rows(&input, &eager);
                let graph = gpu
                    .capture(|_| {
                        wide.run(
                            (&gpu, &weights, &g32),
                            (&block, Act::SiluMul),
                            (cols, N),
                            (&x, &mut y),
                        )
                    })
                    .unwrap();
                graph.launch(stream).unwrap();
                let replay = y.to_host_vec(stream).unwrap();
                assert!(
                    eager
                        .iter()
                        .zip(&replay)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            check_shared_rows(&input, &y.to_host_vec(stream).unwrap());
            assert_eq!(gpu.take_fault().unwrap(), None);
        }
        stream.synchronize().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
