//! The muse-spark `ChainBody`, whole-card or placed. The file's shape is read and
//! checked once at load, each layer's names, site types and router instance
//! resolved once, the K/V planes, the rope table, the router results and the
//! step scratch allocated once; the per-step record and the position's rope
//! row are refreshed before every enqueue or replay.
//!
//! One layer, in the graph's order (`src/models/muse-spark.cpp`, the
//! numerical contract of docs/research/muse-spark-attach.md §2):
//!
//! 1. `attn_norm` (weighted RMS at the block eps), then the q (Q6_K) and the
//!    k, v and gate (Q8_0) projections of the normed row;
//! 2. the per-head q/k RMS norms with the file's constant head gains (the q
//!    one carries the MetaP scale, the k one is ones), then the NORM-pair
//!    rope at the sliding base on the sliding layers — nothing on the full
//!    ones — over the position's uploaded cos/sin row, then the K/V appends
//!    and the head-64 flash pass at the layer's window (0 on a full layer);
//! 3. the per-head weightless RMS of the attention output, `× σ(gate)` (the
//!    gate's own projection, kept beside its sigmoid), the Q6_K output
//!    projection;
//! 4. the weightless post-attention RMS at `post_norm_epsilon`, then the α/β
//!    residual `src ⊙ α + cur ⊙ β` (`elem`'s `mix_rows`);
//! 5. `ffn_norm`, the router on the full stream (its rows pre-scaled by the
//!    MetaP output multiplier at conversion — nothing is multiplied here),
//!    `ffn_latent_down`, the weighted RMS on the latent (`ffn_pre_exp_norm`),
//!    the experts (`iq2_xxs_gate_up_sel` with SiLU, `iq2_xxs_gemv_sel`),
//!    their aggregate scaled by `ffn_post_exp_norm` then normed at the post
//!    eps — before the router weights apply, the graph's own order — the
//!    weighted sum, `ffn_latent_up`;
//! 6. `post_ffw_norm` (scale then norm, `elem`'s `gain_rms_norm`), the α/β
//!    residual again — the next layer's input.
//!
//! Two stretches of a layer run on the step's [`Branch`] beside the main
//! stream, each joined before the first launch that reads it: step 2 and
//! step 3's per-head output norm beside the gate's projection and its
//! sigmoid, and the router beside `ffn_latent_down` and its norm. The gate
//! and the latent down are bandwidth-bound reads of whole matrices; the small
//! launches beside them take the slots their blocks free. Every kernel reads
//! and writes what it does in the serial order, so the bits are that order's.
//!
//! The head is the shared one over the Q6_K `output` rows, the file's scale
//! and tanh softcap folded into its tail. Every launch is a shared kernel:
//! the family's own code is the order they run in.
//!
//! A load whose routed experts the card cannot all hold keeps the plan's card
//! share of each layer's stacks resident and serves the rest on the common
//! host tier ([`placed`]): the FFN half of a layer with a host share is the
//! three parts of the tier's step walk around the host's go, the card's slots
//! under it.

use crate::arch::hybrid::{Acts, DecodeInput, GqaCx, Names, Site, norm_rows};
use crate::card_slots::CardAcc;
use crate::checkpoint::saved::Identity;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::PlaneAppend;
use crate::flash_gqa::{FlashGqaKernels, GROUP, Gqa64Args, HEAD_64, SEGMENTS, window_segments};
use crate::gemm::Gemm32Kernels;
use crate::head::{Head, HeadNorm};
use crate::host::StepLeg;
use crate::host::swap::Residency;
use crate::host::swap_source::{DEADLINE, LIVE_DELAY, ResidencySpec};
use crate::iq_sel::IqGateUp;
use crate::kquant::{Act as GateAct, SelDown};
use crate::model::{ChainBody, GpuModel, HostServed, Rollback, StepMode, refuse_scratch_past};
use crate::quant_sel::QuantSelKernels;
use crate::rope_table::{Direction, RopeRows, RopeSpec, RopeTable};
use crate::route_body::{sigmoid_16_8, sigmoid_256_8};
use crate::site::{Form, KGemvKernels, SiteTy};
use crate::tensor::{Q8ACT_MAX_K, dtod_2d};
use crate::weights::{DevWeight, Weights};
use crate::{Branch, DeviceTensor, Gpu, GpuError, Window};
use bloomery_levers::{ChurnHosting, HostCfg};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{GgmlType, Split};
use model::arch::Arch;
use model::arch::musespark::hparams::Hparams;
use model::arch::musespark::names::{
    ATTN_GATE, ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_RESID_ALPHA,
    ATTN_RESID_BETA, ATTN_V, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_GATE_EXPS, FFN_GATE_INP,
    FFN_LATENT_DOWN, FFN_LATENT_UP, FFN_NORM, FFN_POST_EXP_NORM, FFN_PRE_EXP_NORM, FFN_RESID_ALPHA,
    FFN_RESID_BETA, FFN_UP_EXPS, POST_FFW_NORM,
};
use model::arch::musespark::place::stream_ring_bytes;
use model::arch::names::{blk, token_embd};
use model::placement::Plan;
use std::ops::Range;
use std::sync::Arc;

mod placed;
mod prefill;
mod seq;
mod stream;
mod tier;

pub use placed::Placed;
use placed::{PlacedOpen, StepWalk, plan_map};
use prefill::{PASS_ROWS_MAX, PassSizes, Prompt, choose_pass_rows, placed_rows};
pub use stream::STREAM_FLOOR;
use stream::XSide;

const WHAT: &str = "musespark::Body::load";
const STEP: &str = "musespark::Body::enqueue_chain";

/// The step record: the position, the token, then one row word a key head.
const WORD_POS: usize = 0;
const WORD_TOKEN: usize = 1;
const ROW_WORDS: usize = 2;

/// The rows in flight a load's residency tally holds: the step's one (the file
/// has no draft to verify rows of; a prompt call keeps no rows).
const LOAD_ROWS: usize = 1;

/// The site types the projections read: Q6_K q and o through the q8_1 kgemv,
/// Q8_0 k, v, gate and the latent pair through the f32 gemv.
const SITE_TYPES: &[SiteTy] = &[SiteTy::Q6K, SiteTy::Q8_0];

/// One router at each expert count a muse-spark file carries: the real
/// file's 256 and the fixture's 16, both sigmoid with the top 8
/// (`route_body`'s instances, the rule both families share).
struct Routers {
    e256: sigmoid_256_8::RouterKernels,
    e16: sigmoid_16_8::RouterKernels,
}

/// One layer's router results at its own expert count.
enum RouteOut {
    E256(sigmoid_256_8::RouterOut),
    E16(sigmoid_16_8::RouterOut),
}

impl RouteOut {
    /// The buffers of a layer of `n_expert` experts keeping `n_used`; a
    /// count or pick no instance is built for is refused by name as `what`'s.
    fn new(
        what: &'static str,
        stream: &CudaStream,
        (n_expert, n_used): (usize, usize),
    ) -> Result<RouteOut, GpuError> {
        Ok(match (n_expert, n_used) {
            (sigmoid_256_8::N_EXPERT, sigmoid_256_8::N_USED) => {
                RouteOut::E256(sigmoid_256_8::RouterOut::new(stream)?)
            }
            (sigmoid_16_8::N_EXPERT, sigmoid_16_8::N_USED) => {
                RouteOut::E16(sigmoid_16_8::RouterOut::new(stream)?)
            }
            (n, k) => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "a sigmoid router of {n} experts keeping {k}: the instances route over \
                         256 and 16, both keeping 8"
                    ),
                ));
            }
        })
    }

    /// The experts' raw logits of the last routing, pre-sigmoid (the
    /// oracle's `ffn_moe_logits`).
    fn logits(&self) -> &DeviceBuffer<f32> {
        match self {
            RouteOut::E256(o) => &o.logits,
            RouteOut::E16(o) => &o.logits,
        }
    }

    /// The experts' scores of the last routing: the sigmoid of the logits, which the selection adds its bias to. After a
    /// pass the last row's (the pass walk's `snap`), where the logits keep the last step's.
    fn probs(&self) -> &DeviceBuffer<f32> {
        match self {
            RouteOut::E256(o) => &o.probs,
            RouteOut::E16(o) => &o.probs,
        }
    }

    /// The kept expert ids of the last routing, in rank order.
    fn ids(&self) -> &DeviceBuffer<u32> {
        match self {
            RouteOut::E256(o) => &o.ids,
            RouteOut::E16(o) => &o.ids,
        }
    }

    /// The kept ids' weights — the unbiased scores renormalized — in the
    /// ids' order.
    fn weights(&self) -> &DeviceBuffer<f32> {
        match self {
            RouteOut::E256(o) => &o.weights,
            RouteOut::E16(o) => &o.weights,
        }
    }

    /// Device bytes the results hold.
    fn bytes(&self) -> usize {
        let of = |logits: usize, probs: usize, ids: usize| logits + probs + 2 * ids + 4;
        match self {
            RouteOut::E256(o) => of(o.logits.num_bytes(), o.probs.num_bytes(), o.ids.num_bytes()),
            RouteOut::E16(o) => of(o.logits.num_bytes(), o.probs.num_bytes(), o.ids.num_bytes()),
        }
    }
}

impl Routers {
    fn load(ctx: &Arc<cuda_core::CudaContext>) -> Result<Routers, GpuError> {
        Ok(Routers {
            e256: sigmoid_256_8::RouterKernels::load(ctx)?,
            e16: sigmoid_16_8::RouterKernels::load(ctx)?,
        })
    }

    /// Enqueue one token's routing: `w` the router rows as f32, `x` the
    /// normed row, `bias` the selection bias, `scale` the file's
    /// `expert_weights_scale`. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
    fn enqueue(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        out: &mut RouteOut,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        match out {
            RouteOut::E256(o) => self
                .e256
                .enqueue_router(stream, w, x, bias, scale, o, fault),
            RouteOut::E16(o) => self.e16.enqueue_router(stream, w, x, bias, scale, o, fault),
        }
    }
}

/// One layer: every name the step reads, its K/V planes and its router's
/// results.
struct Layer {
    /// Whether the layer ropes (a sliding layer), and its window (0 when it
    /// has none).
    sliding: bool,
    window: usize,
    attn_norm: String,
    q_norm: String,
    k_norm: String,
    q: Site,
    k: Site,
    v: Site,
    gate: Site,
    o: Site,
    attn_alpha: String,
    attn_beta: String,
    ffn_norm: String,
    latent_down: Site,
    pre_exp_norm: String,
    router: String,
    bias: String,
    /// The experts the card holds of the layer: the id prefix `[0, n_card)`,
    /// all of them on a whole-card load.
    n_card: usize,
    /// The resident stacks of those experts; `None` for a layer whose experts
    /// all stay on the host.
    experts: Option<Stacks>,
    post_exp_norm: String,
    latent_up: Site,
    post_ffw_norm: String,
    ffn_alpha: String,
    ffn_beta: String,
    /// The head-major planes: head `h`, position `p` at row `h · ctx + p`.
    kc: DeviceTensor<u16>,
    vc: DeviceTensor<u16>,
    route: RouteOut,
}

/// The IQ2_XXS stacks of a layer's card experts, in the file's blocks.
struct Stacks {
    gate: String,
    up: String,
    down: String,
}

impl Layer {
    /// The layer's resident expert stacks; a layer whose experts all stay on
    /// the host holds none, and a launch that asks for them is refused by name.
    fn stacks(&self, what: &'static str) -> Result<&Stacks, GpuError> {
        self.experts.as_ref().ok_or(GpuError::state(
            what,
            "the card's expert stacks of a layer that holds experts on the card",
        ))
    }

    /// Every site the layer launches, for the load's [`Acts::new`].
    fn sites(&self) -> Vec<&Site> {
        vec![
            &self.q,
            &self.k,
            &self.v,
            &self.gate,
            &self.o,
            &self.latent_down,
            &self.latent_up,
        ]
    }

    /// The K/V planes' bytes.
    fn bytes(&self) -> usize {
        self.kc.buf().num_bytes() + self.vc.buf().num_bytes()
    }
}

/// The widths every layer shares.
#[derive(Clone, Copy)]
struct Shape {
    hidden: usize,
    /// The head width (the flash instance's 64).
    head: usize,
    n_head: usize,
    n_kv: usize,
    ctx: usize,
    eps: f32,
    post_eps: f32,
    latent: usize,
    n_expert: usize,
    top_k: usize,
    ff: usize,
    weights_scale: f32,
}

impl Shape {
    /// The card slots' sum over `m` rows of the layer's `n_card` card slots: `down` their
    /// rows, `w` the rows' weights and `sel` the slots' places. The latent width, the top-k
    /// and the weights' pitch are the shape's.
    fn card_acc<'a>(
        &self,
        (down, w, sel): (
            &'a DeviceBuffer<f32>,
            &'a DeviceBuffer<f32>,
            &'a DeviceBuffer<u32>,
        ),
        (m, n_card): (usize, usize),
        fault: FaultSink,
    ) -> CardAcc<'a> {
        CardAcc {
            down,
            w,
            sel,
            n: self.latent,
            m,
            n_card,
            n_used: self.top_k,
            w_pitch: self.top_k,
            fault,
        }
    }
}

/// The kernels a step launches beyond the `Gpu`'s own.
struct Kernels {
    flash: FlashGqaKernels,
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    iq: QuantSelKernels,
    routers: Routers,
}

/// The buffers one step runs over.
struct Scratch {
    /// The normed row every projection of a sublayer reads.
    xn: DeviceBuffer<f32>,
    /// The attention output projection's rows.
    delta: DeviceBuffer<f32>,
    /// The latent up's rows.
    delta2: DeviceBuffer<f32>,
    /// The weightless post-attention norm's rows.
    post: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    k2: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    gate_sig: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    lat: DeviceBuffer<f32>,
    lat2: DeviceBuffer<f32>,
    /// The experts' downs, then their scaled-and-normed aggregate, a slot a
    /// row of the latent.
    rows: DeviceBuffer<f32>,
    rows2: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// The embedding's row outputs: the position word the refresh's rope row
    /// follows and the live key count the flash reads.
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    /// The step record: the position, the token, one row word a key head.
    words: DeviceBuffer<u32>,
    /// This position's rope cos/sin row, uploaded by every refresh.
    cs: DeviceBuffer<f32>,
    /// Ones at the hidden and the head: the gains of the weightless norms.
    ones_hidden: DeviceBuffer<f32>,
    ones_head: DeviceBuffer<f32>,
    /// The step record and the rope row on the host, rewritten by every
    /// refresh.
    host: Vec<u32>,
    host_cs: Vec<f32>,
    /// The second stream the attention chain and the router run on beside
    /// the gate's and the latent down's projections ([`attention`],
    /// [`ffn_pre`]).
    branch: Branch,
}

impl Scratch {
    fn bytes(&self) -> usize {
        [
            &self.xn,
            &self.delta,
            &self.delta2,
            &self.post,
            &self.q,
            &self.qn,
            &self.k,
            &self.k2,
            &self.v,
            &self.gate,
            &self.gate_sig,
            &self.attn,
            &self.lat,
            &self.lat2,
            &self.rows,
            &self.rows2,
            &self.up,
            &self.moe,
            &self.part_v,
            &self.part_ms,
            &self.ones_hidden,
            &self.ones_head,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
            + self.words.num_bytes()
            + self.cs.num_bytes()
    }
}

/// One layer's compared activations. The step scratch keeps one buffer a
/// node and the next layer rewrites it, so each node is copied here at the
/// end of its layer. The router's logits, ids and weights stay on the layer.
struct LayerTap {
    attn_gate_proj: DeviceBuffer<f32>,
    attn_gate_sig: DeviceBuffer<f32>,
    attn_gated: DeviceBuffer<f32>,
    attn_o_norm: DeviceBuffer<f32>,
    attn_o_proj: DeviceBuffer<f32>,
    attn_post_norm: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    ffn_latent_down: DeviceBuffer<f32>,
    ffn_moe_out: DeviceBuffer<f32>,
    ffn_latent_up: DeviceBuffer<f32>,
    l_out: DeviceBuffer<f32>,
}

impl LayerTap {
    /// A layer's copies: the q-wide attention nodes, the hidden residuals
    /// and the latent pair.
    fn new(stream: &CudaStream, sh: Shape) -> Result<LayerTap, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let q = sh.n_head * sh.head;
        Ok(LayerTap {
            attn_gate_proj: z(q)?,
            attn_gate_sig: z(q)?,
            attn_gated: z(q)?,
            attn_o_norm: z(q)?,
            attn_o_proj: z(sh.hidden)?,
            attn_post_norm: z(sh.hidden)?,
            ffn_inp: z(sh.hidden)?,
            ffn_latent_down: z(sh.latent)?,
            ffn_moe_out: z(sh.latent)?,
            ffn_latent_up: z(sh.hidden)?,
            l_out: z(sh.hidden)?,
        })
    }

    /// This layer's scratch, copied before the next layer rewrites it.
    fn snap(
        &mut self,
        stream: &CudaStream,
        s: &Scratch,
        ffn_inp: &DeviceBuffer<f32>,
        l_out: &DeviceBuffer<f32>,
        sh: Shape,
    ) -> Result<(), GpuError> {
        let q = sh.n_head * sh.head;
        copy_row(stream, &s.gate, &mut self.attn_gate_proj, q)?;
        copy_row(stream, &s.gate_sig, &mut self.attn_gate_sig, q)?;
        copy_row(stream, &s.attn, &mut self.attn_gated, q)?;
        copy_row(stream, &s.qn, &mut self.attn_o_norm, q)?;
        copy_row(stream, &s.delta, &mut self.attn_o_proj, sh.hidden)?;
        copy_row(stream, &s.post, &mut self.attn_post_norm, sh.hidden)?;
        copy_row(stream, ffn_inp, &mut self.ffn_inp, sh.hidden)?;
        copy_row(stream, &s.lat, &mut self.ffn_latent_down, sh.latent)?;
        copy_row(stream, &s.moe, &mut self.ffn_moe_out, sh.latent)?;
        copy_row(stream, &s.delta2, &mut self.ffn_latent_up, sh.hidden)?;
        copy_row(stream, l_out, &mut self.l_out, sh.hidden)
    }

    /// Device bytes the copies hold.
    fn bytes(&self) -> usize {
        [
            &self.attn_gate_proj,
            &self.attn_gate_sig,
            &self.attn_gated,
            &self.attn_o_norm,
            &self.attn_o_proj,
            &self.attn_post_norm,
            &self.ffn_inp,
            &self.ffn_latent_down,
            &self.ffn_moe_out,
            &self.ffn_latent_up,
            &self.l_out,
        ]
        .into_iter()
        .map(|b| b.num_bytes())
        .sum()
    }
}

/// The per-layer staging a gate's taps read, one [`LayerTap`] a layer, and
/// the row a gate substitutes for one layer's input.
struct Taps {
    layers: Vec<LayerTap>,
    inject: Option<(usize, DeviceBuffer<f32>)>,
    /// The layers a step runs, when a gate probes a span of them; `None` runs
    /// every layer.
    span: Option<Range<usize>>,
}

impl Taps {
    /// Whether the step runs layer `l`.
    fn runs(&self, l: usize) -> bool {
        self.span.as_ref().is_none_or(|s| s.contains(&l))
    }
}

/// The muse-spark chain over one sequence on one card.
pub struct Body {
    sh: Shape,
    layers: Vec<Layer>,
    k: Kernels,
    acts: Acts,
    /// The residual stream's second buffer: each layer's attention half
    /// writes it, its FFN half reads it.
    r: DeviceBuffer<f32>,
    /// The embedding table's name and the head's epilogue (the logit scale
    /// and the softcap).
    embed: String,
    epilogue: (f32, f32),
    /// The rope of the sliding layers (the full ones rope nothing), and its
    /// table, one row a cache position.
    table: RopeTable,
    rope: RopeRows,
    s: Scratch,
    taps: Option<Taps>,
    /// The host tier and the card's side of it; `None` on a whole-card load.
    placed: Option<Placed>,
    /// The prompt pass's buffers ([`GpuModel::prefill`]).
    prompt: Prompt,
    /// The prompt calls' host streaming ([`stream`]).
    xs: XSide,
    /// What a saved sequence belongs to ([`seq`]).
    who: Identity,
}

/// How a load holds the routed experts.
enum Open<'a> {
    /// Every expert resident.
    Whole,
    /// The plan's card segment resident and the rest on the host tier.
    Placed(Box<PlacedOpen<'a>>),
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format (the IQ2_XXS stacks in the
    /// file's blocks), K/V planes of `ctx` rows a key head, the router's
    /// results, the rope table and the step scratch. The model takes the
    /// file and closes it once the weights are resident.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = Hparams::read(&file)
            .map_err(|e| GpuError::plan(WHAT, e))?
            .n_layer;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            Body::load(gpu, &file, w, ctx, Open::Whole)
        })
    }

    /// Card `card` of `plan` resident: the weights the plan puts on the card
    /// (each routed stack's card experts, the id prefix the plan keeps), the
    /// K/V planes of the plan's `ctx_max` rows, the step's buffers, and the
    /// host tier over the layers whose experts the plan leaves on the host,
    /// SiLU-gated in the latent through the common host run ([`crate::host`]),
    /// each expert's output row normed on the host before its weight
    /// ([`placed`]). A plan's expert tier card (plan (b′)) is hung under the
    /// host tier and computes the next ids of each layer ([`tier`]). The prompt
    /// runs as passes through the tier's batch port ([`prefill`]). The load's
    /// slot map stays for the model's life ([`Residency::Off`];
    /// [`GpuModel::open_placed_with`] runs a residency machine). Refused by
    /// name: a plan of other than one stage card holding every layer or of more
    /// than one expert tier, a layer whose card count the file's stacks do not
    /// admit, and scratch past the plan's term for it.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        GpuModel::<Body>::open_placed_with(file, plan, card, host, Residency::Off)
    }

    /// [`GpuModel::open_placed`] under `residency`: under [`Residency::Mid`]
    /// the host set also holds each layer's churn pool and the common
    /// residency machine runs over the load's slot map, its source the file's
    /// IQ2_XXS blocks as the card's stacks hold them ([`stream::stacks`]),
    /// started once the body's pieces are sized
    /// ([`HostServed::start_residency`]); its boundaries are the generic
    /// model's, a prompt call one pass ([`prefill`]). The machine is the
    /// expert stream's source ([`stream`]). Refused by name as
    /// [`GpuModel::open_placed`] is, and by the machine's own refusals (a
    /// layer's card slots too few for its pinned and spare ones, a host set the
    /// load did not read in).
    pub fn open_placed_with(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        residency: Residency,
    ) -> Result<GpuModel<Body>, GpuError> {
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan's ctx_max {}", plan.ctx_max)))?;
        let hp = Hparams::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
        let n_used = hp.n_used;
        let ring_room = stream_ring_bytes(&hp, plan.model).map_err(|e| GpuError::plan(WHAT, e))?;
        let spec = ResidencySpec {
            lever: residency,
            pool: ChurnHosting::Hosted,
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k: n_used,
            max_rows: LOAD_ROWS,
            stacks: Arc::new(stream::stacks(plan, plan.model.layers)?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set, glue| {
                let open = PlacedOpen {
                    plan,
                    file: Arc::clone(file),
                    host,
                    set,
                    glue,
                };
                let mut body = Body::load(gpu, file, w, ctx, Open::Placed(Box::new(open)))?;
                body.xs.set_ring_room(ring_room);
                refuse_scratch_past(
                    WHAT,
                    plan,
                    card,
                    body.scratch_bytes(),
                    "the step's buffers, the routers' results, the quantized forms, the host \
                     boundary, the slot map's copy",
                )?;
                if let Some(p) = body.placed.as_mut() {
                    p.start_residency(gpu)?;
                }
                Ok(body)
            },
        )
    }
}

impl Body {
    /// Every site's width must be one the form its weight type reads takes:
    /// a K-quant site's q8_1 blocks of 256 values, a Q8_0 site's 32.
    fn refuse_widths(layers: &[Layer]) -> Result<(), GpuError> {
        for l in layers {
            for s in l.sites() {
                let gran = match s.ty.reads() {
                    Form::Q8x128 => 256,
                    Form::Q8Blocks32 | Form::Q8x32 => 32,
                    Form::F32 => continue,
                };
                if s.k == 0 || !s.k.is_multiple_of(gran) || s.k > Q8ACT_MAX_K {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "{}: a form of {gran} values takes a width that is a positive \
                             multiple of {gran} up to {Q8ACT_MAX_K}, got {}",
                            s.name(),
                            s.k()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        ctx: usize,
        open: Open<'_>,
    ) -> Result<Body, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let map = match &open {
            Open::Whole => None,
            Open::Placed(o) => Some(plan_map(o.plan, &hp)?),
        };
        if hp.head_dim != HEAD_64 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the head is {}, the flash instance runs {HEAD_64}",
                    hp.head_dim
                ),
            ));
        }
        // The step and a pass rope each row's heads by one row of the table
        // a position, so the table's width is the head's.
        if hp.rope_dims != hp.head_dim {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a rope of {} dims over heads of {}: a step and a pass rope each row's \
                     heads by one row of the table a position",
                    hp.rope_dims, hp.head_dim
                ),
            ));
        }
        if hp.n_head != hp.n_head_kv * GROUP {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "attention of {} query heads over {} key heads: the flash runs {GROUP} \
                     query heads a key head",
                    hp.n_head, hp.n_head_kv
                ),
            ));
        }
        // The experts' gate·up quantizes the latent into whole 256-value
        // super-blocks, and the q8_1 form of the q and o sites the hidden.
        for (name, width) in [("the hidden", hp.n_embd), ("moe_latent_size", hp.latent)] {
            if width == 0 || !width.is_multiple_of(256) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is {width}, the experts' blocks take a multiple of 256"),
                ));
            }
        }
        if !hp.weights_norm {
            return Err(GpuError::shape(
                WHAT,
                "expert_weights_norm is false: the sigmoid instances renormalize the kept \
                 weights, a file that keeps them raw diverges from them",
            ));
        }
        let stream = gpu.stream();
        let names = Names {
            what: WHAT,
            file,
            w,
            types: SITE_TYPES,
        };
        let embed = token_embd();
        match w.resident(WHAT, &embed)? {
            DevWeight::KQuant {
                ty: GgmlType::Q4_K,
                w: t,
                ..
            } if t.cols().is_multiple_of(36) => {}
            _ => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{embed} is not a Q4_K table of 36-word super-block rows"),
                ));
            }
        }
        let (hidden, head, n_head, n_kv) = (hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv);
        let q_width = n_head * head;
        let kv_width = n_kv * head;
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let proj = |stem: &str, rows: usize, k: usize| names.proj(l, stem, (rows, k));
            let n_card = match &map {
                Some((m, _)) => m.on_card(l)?,
                None => hp.n_expert,
            };
            let ff = hp.expert_ff[l];
            let experts = if n_card > 0 {
                Some(Stacks {
                    gate: iq2_xxs_stack(w, l, FFN_GATE_EXPS, (n_card * ff, hp.latent))?,
                    up: iq2_xxs_stack(w, l, FFN_UP_EXPS, (n_card * ff, hp.latent))?,
                    down: iq2_xxs_stack(w, l, FFN_DOWN_EXPS, (n_card * hp.latent, ff))?,
                })
            } else {
                None
            };
            layers.push(Layer {
                sliding: hp.sliding[l],
                window: if hp.sliding[l] { hp.window } else { 0 },
                attn_norm: names.vector(l, ATTN_NORM, hidden)?,
                q_norm: names.vector(l, ATTN_Q_NORM, head)?,
                k_norm: names.vector(l, ATTN_K_NORM, head)?,
                q: proj(ATTN_Q, q_width, hidden)?,
                k: proj(ATTN_K, kv_width, hidden)?,
                v: proj(ATTN_V, kv_width, hidden)?,
                gate: proj(ATTN_GATE, q_width, hidden)?,
                o: proj(ATTN_OUTPUT, hidden, q_width)?,
                attn_alpha: names.vector(l, ATTN_RESID_ALPHA, hidden)?,
                attn_beta: names.vector(l, ATTN_RESID_BETA, hidden)?,
                ffn_norm: names.vector(l, FFN_NORM, hidden)?,
                latent_down: proj(FFN_LATENT_DOWN, hp.latent, hidden)?,
                pre_exp_norm: names.vector(l, FFN_PRE_EXP_NORM, hp.latent)?,
                // A matrix, not a vector: its F32 shape check is the loop
                // below the layers.
                router: blk(l, FFN_GATE_INP),
                bias: names.vector(l, EXP_PROBS_B, hp.n_expert)?,
                n_card,
                experts,
                post_exp_norm: names.vector(l, FFN_POST_EXP_NORM, hp.latent)?,
                latent_up: proj(FFN_LATENT_UP, hidden, hp.latent)?,
                post_ffw_norm: names.vector(l, POST_FFW_NORM, hidden)?,
                ffn_alpha: names.vector(l, FFN_RESID_ALPHA, hidden)?,
                ffn_beta: names.vector(l, FFN_RESID_BETA, hidden)?,
                kc: DeviceTensor::zeroed(stream, n_kv * ctx, head)?,
                vc: DeviceTensor::zeroed(stream, n_kv * ctx, head)?,
                route: RouteOut::new(WHAT, stream, (hp.n_expert, hp.n_used))?,
            });
        }
        Body::refuse_widths(&layers)?;
        // The router's rows are the file's F32 `ffn_gate_inp`, whole and
        // pre-scaled.
        for l in 0..hp.n_layer {
            let name = blk(l, FFN_GATE_INP);
            let rows = w.f32_tensor(WHAT, &name)?;
            if (rows.rows(), rows.cols()) != (hp.n_expert, hidden) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not {} f32 rows of {hidden} values", hp.n_expert),
                ));
            }
        }
        let mut acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        for (k, m) in [(hp.latent, 1), (hp.expert_ff[0], hp.n_used)] {
            acts.add_q8_1(stream, k, m)?;
        }
        let sh = Shape {
            hidden,
            head,
            n_head,
            n_kv,
            ctx,
            eps: hp.rms_eps,
            post_eps: hp.post_eps,
            latent: hp.latent,
            n_expert: hp.n_expert,
            top_k: hp.n_used,
            ff: hp.expert_ff[0],
            weights_scale: hp.weights_scale,
        };
        let table = RopeTable::new(&RopeSpec::window(hp.rope_base, hp.rope_dims))?;
        let rope = RopeRows::new(stream, &table, hp.rope_dims, ctx)?;
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n.max(1))?)
        };
        // A windowed row is cut into whole segments of the window, a
        // windowless one into SEGMENTS: the partials hold the larger cut.
        let segs = SEGMENTS.max(window_segments(hp.window));
        let k = {
            let ctx = gpu.context();
            Kernels {
                flash: FlashGqaKernels::load(ctx)?,
                kgemv: KGemvKernels::load(gpu)?,
                g32: Gemm32Kernels::load(ctx)?,
                iq: QuantSelKernels::load(ctx, gpu.fault_word())?,
                routers: Routers::load(ctx)?,
            }
        };
        let s = Scratch {
            xn: z(hidden)?,
            delta: z(hidden)?,
            delta2: z(hidden)?,
            post: z(hidden)?,
            q: z(q_width)?,
            qn: z(q_width)?,
            k: z(kv_width)?,
            k2: z(kv_width)?,
            v: z(kv_width)?,
            gate: z(q_width)?,
            gate_sig: z(q_width)?,
            attn: z(q_width)?,
            lat: z(sh.latent)?,
            lat2: z(sh.latent)?,
            rows: z(sh.top_k * sh.latent)?,
            rows2: z(sh.top_k * sh.latent)?,
            up: z(sh.top_k * sh.ff)?,
            moe: z(sh.latent)?,
            part_v: DeviceBuffer::zeroed(stream, n_head * segs * head)?,
            part_ms: DeviceBuffer::zeroed(stream, n_head * segs * 2)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
            words: DeviceBuffer::zeroed(stream, ROW_WORDS + n_kv)?,
            cs: DeviceBuffer::zeroed(stream, hp.rope_dims)?,
            ones_hidden: DeviceBuffer::from_host(stream, &vec![1.0f32; hidden])?,
            ones_head: DeviceBuffer::from_host(stream, &vec![1.0f32; head])?,
            host: vec![0; ROW_WORDS + n_kv],
            host_cs: vec![0.0; hp.rope_dims],
            branch: Branch::new(gpu.stream_handle())?,
        };
        // The prompt pass's rows come from the card's room after the weights,
        // the stores and the step's buffers: the largest pass whose buffers
        // fit what is left, at most a whole-card load's cap or the rows a
        // placed load's host union takes in one call. A placed load's plan
        // reserved the buffers of that cap: the load cuts them from the shape
        // the plan counted.
        let sizes = PassSizes::of(sh, &layers);
        if let Open::Placed(o) = &open {
            sizes.refuse_unplanned(&hp, o.plan)?;
        }
        let cap = match &open {
            Open::Whole => PASS_ROWS_MAX,
            Open::Placed(_) => placed_rows(hp.n_used),
        };
        let rows = choose_pass_rows(&sizes, ctx, gpu.alloc_room()?, cap)?;
        let prompt = Prompt::new(gpu, &sizes, &layers, rows)?;
        let placed = match open {
            Open::Whole => None,
            Open::Placed(o) => {
                let map = map.ok_or(GpuError::state(WHAT, "a slot map for a placed load"))?;
                Some(Placed::new(gpu, &hp, &layers, map, (*o, rows))?)
            }
        };
        Ok(Body {
            sh,
            layers,
            k,
            acts,
            r: z(hidden)?,
            embed,
            epilogue: (hp.logit_scale, hp.softcap),
            table,
            rope,
            s,
            taps: None,
            placed,
            prompt,
            xs: XSide::default(),
            who: Identity::of_load("musespark", file, gpu, ctx)?,
        })
    }

    /// Device bytes the body holds past the weights and the stores: what the
    /// plan sets aside as scratch, the one owner of the term the load checks
    /// against it (the rope rows are the plan's own stage term, the K/V planes
    /// its cache's; the prompt pass's buffers are the plan's own reserve,
    /// `PlanInputs::prompt_reserve`, not its scratch).
    fn scratch_bytes(&self) -> usize {
        let routes: usize = self.layers.iter().map(|l| l.route.bytes()).sum();
        routes
            + self.s.bytes()
            + self.acts.bytes()
            + self.r.num_bytes()
            + self.k.kgemv.device_bytes()
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
    }

    /// The placed side of a placed load (the host tier and the card's side of
    /// it), `None` on a whole-card load: a gate's instrument.
    #[must_use]
    pub fn placed(&self) -> Option<&Placed> {
        self.placed.as_ref()
    }

    /// The whole chain: the embedding and its weightless norm, every layer's
    /// two halves, then the head with the file's epilogue in its tail.
    fn enqueue(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let Body {
            sh,
            layers,
            k,
            acts,
            r,
            embed,
            epilogue,
            table: _table,
            rope: _rope,
            s,
            taps,
            placed,
            prompt: _prompt,
            xs: _xs,
            who: _who,
        } = self;
        let sh = *sh;
        let cx = GqaCx {
            what: STEP,
            gpu,
            w,
            kgemv: &k.kgemv,
            g32: &k.g32,
            flash: &k.flash,
            mma: false,
            out_gate: None,
            hidden: sh.hidden,
            eps: sh.eps,
            n_kv: sh.n_kv,
            ctx: sh.ctx,
        };
        // The embedding row and its weightless norm: the layer input.
        let table_rows = stack(w, embed)?;
        {
            let ids = Window::<u32>::of(&s.words, WORD_TOKEN * 4, 1)?;
            let pos0 = Window::<u32>::of(&s.words, WORD_POS * 4, 1)?;
            gpu.elem().enqueue_embed_rows_q4k(
                stream,
                crate::elem::EmbedRowsArgs {
                    w: table_rows,
                    ids: &ids,
                    pos0: &pos0,
                    first: 0,
                    y: &mut s.xn,
                    pos: &mut s.pos,
                    n_keys: &mut s.n_keys,
                },
            )?;
            gpu.elem().enqueue_rms_norm(
                stream,
                &s.xn,
                &s.ones_hidden,
                sh.eps,
                sh.hidden,
                1,
                head.input_mut(),
            )?;
        }
        // A placed load walks its layers through the host tier's step port: a
        // layer with a host share is its front, its shadow under the host's
        // work and its back.
        if let Some(Placed { hybrid, side, .. }) = placed.as_mut() {
            let mut leg = StepLeg::new(stream, hybrid);
            StepWalk {
                cx,
                k,
                sh,
                layers,
                s,
                acts,
                side,
                taps: taps.as_mut(),
                hin: head.input_mut(),
                r,
            }
            .walk(&mut leg)?;
        } else {
            for (l, a) in layers.iter_mut().enumerate() {
                if taps.as_ref().is_some_and(|t| !t.runs(l)) {
                    continue;
                }
                if let Some((at, row)) = taps.as_ref().and_then(|t| t.inject.as_ref())
                    && *at == l
                {
                    copy_row(stream, row, head.input_mut(), sh.hidden)?;
                }
                attention(
                    gpu,
                    &cx,
                    (l, a),
                    (s, acts),
                    (&*head.input_mut(), &mut *r),
                    sh,
                )?;
                ffn(gpu, &cx, k, (l, a), (s, acts), (r, head.input_mut()), sh)?;
                if let Some(t) = taps.as_mut() {
                    t.layers[l].snap(stream, s, r, head.input_mut(), sh)?;
                }
            }
        }
        enqueue_head(gpu, w, head, *epilogue)
    }
}

/// The head over its input row with the file's epilogue in its tail: the
/// Q6_K `output` rows, the logit scale, the softcap, the argmax — the step's
/// and a prompt pass's last row's.
fn enqueue_head(
    gpu: &Gpu,
    w: &Weights,
    head: &mut Head,
    (scale, cap): (f32, f32),
) -> Result<(), GpuError> {
    let n_vocab = head.n_vocab();
    head.enqueue_with_tail(gpu, w, |act, out_w, fault, logits, token_out| {
        gpu.enqueue_gemv_q6k(out_w, act, logits)?;
        let elem = gpu.elem();
        elem.enqueue_scale_rows(gpu.stream(), logits, n_vocab, scale)?;
        elem.enqueue_softcap_rows(gpu.stream(), logits, n_vocab, cap)?;
        elem.enqueue_argmax_fault(gpu.stream(), logits, n_vocab, fault, token_out)
    })
}

/// `n` f32 of `src` copied into `dst`, one row.
fn copy_row(
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    n: usize,
) -> Result<(), GpuError> {
    let width = n * 4;
    dtod_2d(
        stream,
        (src, 0),
        (dst, 0),
        crate::tensor::Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        },
    )
}

/// Key head `h`'s row of `buf`, the view an append reads.
fn head_rows<'b>(
    buf: &'b DeviceBuffer<f32>,
    head: usize,
) -> impl FnMut(usize) -> Result<Window<'b, f32>, GpuError> {
    move |h| Window::<f32>::of(buf, h * head * 4, head)
}

/// One attention half: the norm, the projections, the per-head q/k norms,
/// the rope on a sliding layer, the appends, the flash, the per-head output
/// norm, the gate, the output projection, the weightless post-norm and the
/// α/β mix of `src` into `dst`.
fn attention(
    gpu: &Gpu,
    cx: &GqaCx<'_>,
    (l, a): (usize, &mut Layer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    sh: Shape,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let elem = gpu.elem();
    norm_rows(
        (gpu, cx.w),
        (sh.eps, sh.hidden, 1),
        &a.attn_norm,
        src,
        &mut s.xn,
    )?;
    a.q.project(cx, acts, l, &s.xn, &mut s.q)?;
    a.k.project(cx, acts, l, &s.xn, &mut s.k)?;
    a.v.project(cx, acts, l, &s.xn, &mut s.v)?;
    // The gate's projection and its sigmoid on the main stream, the chain from
    // the per-head norms to the per-head output norm on the branch beside
    // them; the gated product after the join reads both.
    let fork = s.branch.fork()?;
    a.gate.project(cx, acts, l, &s.xn, &mut s.gate)?;
    // The gate's projection and its sigmoid both survive the step: the
    // projection's rows and, beside them, their σ.
    let width = sh.n_head * sh.head * 4;
    dtod_2d(
        stream,
        (&s.gate, 0),
        (&mut s.gate_sig, 0),
        crate::tensor::Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        },
    )?;
    elem.enqueue_sigmoid_rows(stream, &mut s.gate_sig, sh.n_head * sh.head)?;
    let side = fork.stream();
    elem.enqueue_rms_norm(
        side,
        &s.q,
        cx.w.f32_buf(STEP, &a.q_norm)?,
        sh.eps,
        sh.head,
        sh.n_head,
        &mut s.qn,
    )?;
    elem.enqueue_rms_norm(
        side,
        &s.k,
        cx.w.f32_buf(STEP, &a.k_norm)?,
        sh.eps,
        sh.head,
        sh.n_kv,
        &mut s.k2,
    )?;
    // The sliding layers rope in NORM pairs over this position's uploaded
    // row; the full layers rope nothing and the flash and the appends read
    // their norms' outputs.
    let (q_in, k_in) = if a.sliding {
        elem.enqueue_rope(side, &s.qn, &s.cs, sh.head, sh.n_head as u32, 1, &mut s.q)?;
        elem.enqueue_rope(side, &s.k2, &s.cs, sh.head, sh.n_kv as u32, 1, &mut s.k)?;
        (&s.q, &s.k)
    } else {
        (&s.qn, &s.k2)
    };
    let at = PlaneAppend {
        words: &s.words,
        first_word: ROW_WORDS,
        heads: sh.n_kv,
        m: 1,
        fault,
    };
    gpu.flash()
        .enqueue_kv_append_planes(side, &at, head_rows(k_in, sh.head), &mut a.kc)?;
    gpu.flash()
        .enqueue_kv_append_planes(side, &at, head_rows(&s.v, sh.head), &mut a.vc)?;
    cx.flash.enqueue_pass_64(
        side,
        Gqa64Args {
            q: q_in,
            kc: a.kc.buf(),
            vc: a.vc.buf(),
            n_keys: &s.n_keys,
            scale: 1.0 / (sh.head as f32).sqrt(),
            n_kv: sh.n_kv,
            ctx: sh.ctx,
            m: 1,
            window: a.window,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            fault,
            y: &mut s.attn,
        },
        false,
    )?;
    // The per-head weightless norm of the attention output, then the gated
    // product in the flash's own rows.
    elem.enqueue_rms_norm(
        side,
        &s.attn,
        &s.ones_head,
        sh.eps,
        sh.head,
        sh.n_head,
        &mut s.qn,
    )?;
    fork.join()?;
    elem.enqueue_mul_rows(stream, &s.qn, &s.gate_sig, sh.n_head * sh.head, &mut s.attn)?;
    a.o.project(cx, acts, l, &s.attn, &mut s.delta)?;
    elem.enqueue_rms_norm(
        stream,
        &s.delta,
        &s.ones_hidden,
        sh.post_eps,
        sh.hidden,
        1,
        &mut s.post,
    )?;
    elem.enqueue_mix_rows(
        stream,
        src,
        cx.w.f32_buf(STEP, &a.attn_alpha)?,
        &s.post,
        cx.w.f32_buf(STEP, &a.attn_beta)?,
        sh.hidden,
        dst,
    )
}

/// One FFN half: the norm, the router on the full stream, the latent down
/// and its weighted norm ([`ffn_pre`]), the experts and their scaled-and-normed
/// rows ([`experts_card`]), the rows under the router weights, then the latent
/// up, the scale-then-norm and the α/β mix of `src` into `dst`
/// ([`ffn_post`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the half's launches: its layer, the site context, the kernels, the scratch and its input and output (rust-quality R8)"
)]
fn ffn(
    gpu: &Gpu,
    cx: &GqaCx<'_>,
    k: &Kernels,
    (l, a): (usize, &mut Layer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    sh: Shape,
) -> Result<(), GpuError> {
    ffn_pre(gpu, cx, k, (l, &mut *a), (s, acts), src, None, sh)?;
    experts_card(gpu, cx, k, (l, a), (s, acts), (None, None), sh)?;
    // The weights apply after the norm, the graph's own order.
    gpu.elem().enqueue_weighted_sum(
        gpu.stream(),
        &s.rows2,
        a.route.weights(),
        sh.latent,
        sh.top_k as u32,
        1,
        &mut s.moe,
    )?;
    ffn_post(gpu, cx, (l, a), (s, acts), (src, dst), sh)
}

/// The FFN half's front: the norm of `src`, the router on the full stream (on
/// the step's branch), the latent down and the weighted norm on the latent
/// (`ffn_pre_exp_norm`) into `lat2`, the row the experts read — the step's own
/// buffer, or the buffer given (a hosted layer's boundary activation, which
/// the host reads). The router is joined before the function returns.
#[allow(
    clippy::too_many_arguments,
    reason = "the half's launches: its layer, the site context, the kernels, the scratch, its input and the experts' row (rust-quality R8)"
)]
fn ffn_pre(
    gpu: &Gpu,
    cx: &GqaCx<'_>,
    k: &Kernels,
    (l, a): (usize, &mut Layer),
    (s, acts): (&mut Scratch, &mut Acts),
    src: &DeviceBuffer<f32>,
    lat2: Option<&mut DeviceBuffer<f32>>,
    sh: Shape,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    norm_rows(
        (gpu, cx.w),
        (sh.eps, sh.hidden, 1),
        &a.ffn_norm,
        src,
        &mut s.xn,
    )?;
    // The latent down and its norm on the main stream, the router on the
    // branch beside them; the half's next launch, after the join, reads the
    // router's ids and weights.
    let fork = s.branch.fork()?;
    a.latent_down.project(cx, acts, l, &s.xn, &mut s.lat)?;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.lat,
        cx.w.f32_buf(STEP, &a.pre_exp_norm)?,
        sh.eps,
        sh.latent,
        1,
        match lat2 {
            Some(buf) => buf,
            None => &mut s.lat2,
        },
    )?;
    k.routers.enqueue(
        fork.stream(),
        cx.w.f32_tensor(STEP, &a.router)?,
        &s.xn,
        cx.w.f32_buf(STEP, &a.bias)?,
        sh.weights_scale,
        &mut a.route,
        fault,
    )?;
    fork.join()
}

/// The card's experts of a layer for one token: `lat` (the step's `lat2` when
/// `None`) quantized as q8_1, the SiLU-gated gate·up and the down over the
/// slots `sel` places (the router's ids when `None`; a hosted layer's places,
/// whose [`HOST`](crate::hybrid::HOST) slots the launches skip), each slot's
/// down row scaled by `ffn_post_exp_norm` then normed at the post epsilon —
/// before the router weights apply, the graph's own order — into `rows2`.
fn experts_card(
    gpu: &Gpu,
    cx: &GqaCx<'_>,
    k: &Kernels,
    (l, a): (usize, &Layer),
    (s, acts): (&mut Scratch, &mut Acts),
    (lat, sel): (Option<&DeviceBuffer<f32>>, Option<&DeviceBuffer<u32>>),
    sh: Shape,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let stacks = a.stacks(STEP)?;
    let lat = lat.unwrap_or(&s.lat2);
    let sel = sel.unwrap_or_else(|| a.route.ids());
    let act = acts.q8_1_at(STEP, sh.latent, 1)?;
    gpu.enqueue_quantize_q8_1_layer(lat, act, l)?;
    k.iq.enqueue_gate_up(
        stream,
        &IqGateUp {
            ty: GgmlType::IQ2_XXS,
            wg: stack(cx.w, &stacks.gate)?,
            wu: stack(cx.w, &stacks.up)?,
            act,
            sel,
            n_slots: sh.top_k,
            rows_per_expert: sh.ff,
            // One latent column: slot `s` reads column `s / top_k`.
            slots_per_col: sh.top_k,
            rule: GateAct::SiluMul,
        },
        fault,
        &mut s.up,
    )?;
    let act2 = acts.q8_1_at(STEP, sh.ff, sh.top_k)?;
    gpu.enqueue_quantize_q8_1_layer(&s.up, act2, l)?;
    k.iq.enqueue_gemv_sel(
        stream,
        GgmlType::IQ2_XXS,
        &SelDown {
            w: stack(cx.w, &stacks.down)?,
            act: act2,
            sel,
            n_slots: sh.top_k,
            rows_per_expert: sh.latent,
        },
        fault,
        &mut s.rows,
    )?;
    gpu.elem().enqueue_gain_rms_norm(
        stream,
        &s.rows,
        cx.w.f32_buf(STEP, &a.post_exp_norm)?,
        sh.post_eps,
        sh.latent,
        sh.top_k,
        &mut s.rows2,
    )
}

/// The FFN half's back: the latent up of the experts' weighted sum (`s.moe`),
/// the scale-then-norm (`post_ffw_norm`) and the α/β mix of `src` into `dst`.
fn ffn_post(
    gpu: &Gpu,
    cx: &GqaCx<'_>,
    (l, a): (usize, &Layer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    sh: Shape,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    a.latent_up.project(cx, acts, l, &s.moe, &mut s.delta2)?;
    elem.enqueue_gain_rms_norm(
        stream,
        &s.delta2,
        cx.w.f32_buf(STEP, &a.post_ffw_norm)?,
        sh.post_eps,
        sh.hidden,
        1,
        &mut s.xn,
    )?;
    elem.enqueue_mix_rows(
        stream,
        src,
        cx.w.f32_buf(STEP, &a.ffn_alpha)?,
        &s.xn,
        cx.w.f32_buf(STEP, &a.ffn_beta)?,
        sh.hidden,
        dst,
    )
}

/// The resident IQ2_XXS stack `name`, in the file's blocks.
fn stack<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(STEP, name)? {
        DevWeight::KQuant { w, .. } => Ok(w),
        _ => Err(GpuError::tensor(
            STEP,
            name,
            "a weight in the file's blocks",
        )),
    }
}

/// The resident IQ2_XXS stack `stem` of layer `l`, `rows` rows of `k`
/// values, in the file's blocks; any other variant, type or shape is refused
/// by name as `what`'s.
fn iq2_xxs_stack(
    w: &Weights,
    l: usize,
    stem: &str,
    (rows, k): (usize, usize),
) -> Result<String, GpuError> {
    let name = blk(l, stem);
    match w.resident(WHAT, &name)? {
        DevWeight::KQuant {
            ty: GgmlType::IQ2_XXS,
            w: t,
            k: got,
        } if t.rows() == rows && *got == k => Ok(name),
        _ => Err(GpuError::shape(
            WHAT,
            format!(
                "{name} is not an IQ2_XXS weight of {rows} rows of {k} values in the file's blocks"
            ),
        )),
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = Placed;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::MuseSpark
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The step record and the position's rope row, each in one copy.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        const WHAT: &str = "musespark::Body::refresh";
        let DecodeInput { token, pos } = *input;
        let (ctx, n_kv) = (self.sh.ctx, self.sh.n_kv);
        self.s.host[WORD_POS] = pos;
        self.s.host[WORD_TOKEN] = token;
        for h in 0..n_kv {
            let row = h
                .checked_mul(ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("row {h} · {ctx} + {pos} does not fit a u32 row word"),
                    )
                })?;
            self.s.host[ROW_WORDS + h] = row;
        }
        self.s.host_cs.clear();
        self.table
            .push(pos, Direction::Forward, &mut self.s.host_cs);
        self.s.words.copy_from_host(stream, &self.s.host)?;
        self.s.cs.copy_from_host(stream, &self.s.host_cs)?;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue(gpu, w, head)
    }

    /// Rewind every cache to empty. The K/V planes need nothing: the flash
    /// never reads a row at or past the live count, and every row below it is
    /// written by its own step first. The router's results are written whole
    /// by every step before any read. A placed load's expert rows are cleared:
    /// the launches that fill them skip the slots the host serves, and the
    /// quantize after the up reads every slot, so the non-finite rows a
    /// faulted call left would fault the next valid one.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if let Some(p) = self.placed.as_mut() {
            self.s.up.zero_async(gpu.stream())?;
            self.s.rows.zero_async(gpu.stream())?;
            self.s.rows2.zero_async(gpu.stream())?;
            p.reset(gpu.stream())?;
        }
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn head_norm(&self) -> HeadNorm {
        HeadNorm::Rms
    }

    fn resident_bytes(&self) -> usize {
        let stores: usize = self.layers.iter().map(Layer::bytes).sum();
        let routes: usize = self.layers.iter().map(|l| l.route.bytes()).sum();
        stores
            + routes
            + self.s.bytes()
            + self.acts.bytes()
            + self.r.num_bytes()
            + self.rope.table.num_bytes()
            + self.k.kgemv.device_bytes()
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
            + self.prompt.bytes()
            + self
                .taps
                .as_ref()
                .map_or(0, |t| t.layers.iter().map(LayerTap::bytes).sum())
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}

impl Rollback for Body {
    /// A cut to `pos`: every store is a plane of positions the flash reads
    /// only below the live count and each step writes its position's row
    /// before it reads it, so a cut leaves nothing to take back.
    fn rollback(&mut self, _pos: u32) -> Result<(), GpuError> {
        Ok(())
    }
}

/// One layer's compared values, read back after a step (a gate's
/// instrument; the oracle's own `cb()` names).
pub struct LayerNodes {
    pub attn_gate_proj: Vec<f32>,
    pub attn_gate_sig: Vec<f32>,
    pub attn_gated: Vec<f32>,
    pub attn_o_norm: Vec<f32>,
    pub attn_o_proj: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub ffn_latent_down: Vec<f32>,
    pub ffn_moe_logits: Vec<f32>,
    /// The router's scores (the logits' sigmoid): the step's, or after a pass the pass's last row's.
    pub ffn_moe_probs: Vec<f32>,
    pub ffn_moe_topk: Vec<u32>,
    pub ffn_moe_weights: Vec<f32>,
    pub ffn_moe_out: Vec<f32>,
    pub ffn_latent_up: Vec<f32>,
    pub l_out: Vec<f32>,
}

/// One layer's routing as the last step or pass left it, read back ([`GpuModel::layer_routes`]).
pub struct LayerRoute {
    /// The kept expert ids, in rank order.
    pub ids: Vec<u32>,
    /// The router's logits: the last step's, a pass leaves them as they were.
    pub logits: Vec<f32>,
    /// The router's scores (the logits' sigmoid): the last step's, or the last pass row's.
    pub probs: Vec<f32>,
}

/// Refused by name in graph mode: a captured step replays the walk it recorded, so a layer span or an input override
/// set after the capture would be ignored by every replay.
fn refuse_graph(what: &'static str, mode: StepMode) -> Result<(), GpuError> {
    match mode {
        StepMode::Eager => Ok(()),
        StepMode::Graph => Err(GpuError::state(
            what,
            "eager steps (set_mode(StepMode::Eager)): a captured step replays the walk it recorded",
        )),
    }
}

impl GpuModel<Body> {
    /// Arm or disarm the per-layer node taps (every compared activation a
    /// gate's [`nodes`](GpuModel::nodes) reads back; arming while a capture
    /// exists is refused by name, for the graph would not hold the copies).
    /// Load-time work only; the copies themselves are enqueues.
    pub fn set_taps(&mut self, on: bool) -> Result<(), GpuError> {
        const WHAT: &str = "musespark::set_taps";
        let captured = self.has_capture();
        let (gpu, _w, body) = self.body_parts(WHAT)?;
        // A two-card load may leave the tier card's context current (any read
        // of the tier binds it); the stage's buffers are made in its own.
        gpu.context().bind_to_thread()?;
        if on && captured {
            return Err(GpuError::state(
                WHAT,
                "a captured graph would not hold the taps' copies",
            ));
        }
        if !on {
            body.taps = None;
            return Ok(());
        }
        let stream = gpu.stream();
        let n = body.layers.len();
        let sh = body.sh;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(LayerTap::new(stream, sh)?);
        }
        body.taps = Some(Taps {
            layers,
            inject: None,
            span: None,
        });
        Ok(())
    }

    /// Run only the layers of `span` on the steps that follow, the layers
    /// outside it left unrun (their taps hold what an earlier step left, and
    /// their K/V planes get no row): with [`GpuModel::set_input_override`] on
    /// the span's first layer, a gate's way to probe those layers alone on the
    /// oracle's own input at the cost of the span rather than the chain. The
    /// head runs on whatever the last run layer left. `None` runs every layer.
    /// Eager steps only, so the taps must be armed; a span outside the model
    /// and any call in graph mode are refused by name.
    pub fn set_layer_span(&mut self, span: Option<Range<usize>>) -> Result<(), GpuError> {
        const WHAT: &str = "musespark::set_layer_span";
        refuse_graph(WHAT, self.mode())?;
        let (_gpu, _w, body) = self.body_parts(WHAT)?;
        let taps = body
            .taps
            .as_mut()
            .ok_or_else(|| GpuError::state(WHAT, "the taps are not armed (set_taps)"))?;
        if let Some(s) = &span
            && (s.start >= s.end || s.end > taps.layers.len())
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the span {s:?} is empty or past the model's {} layers",
                    taps.layers.len()
                ),
            ));
        }
        taps.span = span;
        Ok(())
    }

    /// Make `row` layer `layer`'s input on the steps that follow, in place of
    /// the residual stream the layer before it wrote: a gate's way to run one
    /// layer on the oracle's own input, so its nodes carry that layer's
    /// rounding alone. `None` clears it. Eager steps only (a captured graph
    /// holds no such copy), so the taps must be armed and graph mode is
    /// refused by name.
    pub fn set_input_override(
        &mut self,
        layer: usize,
        row: Option<&[f32]>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "musespark::set_input_override";
        refuse_graph(WHAT, self.mode())?;
        let (gpu, _w, body) = self.body_parts(WHAT)?;
        gpu.context().bind_to_thread()?;
        let hidden = body.sh.hidden;
        let taps = body
            .taps
            .as_mut()
            .ok_or_else(|| GpuError::state(WHAT, "the taps are not armed (set_taps)"))?;
        taps.inject = match row {
            None => None,
            Some(r) if r.len() == hidden && layer < taps.layers.len() => {
                Some((layer, DeviceBuffer::from_host(gpu.stream(), r)?))
            }
            Some(r) => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {layer} row of {} values: the model has {} layers of {hidden}",
                        r.len(),
                        taps.layers.len()
                    ),
                ));
            }
        };
        Ok(())
    }

    /// Every layer's compared values of the last step, read back (a gate's
    /// instrument; it synchronizes the stream, and the taps must be armed).
    pub fn nodes(&mut self) -> Result<Vec<LayerNodes>, GpuError> {
        const WHAT: &str = "musespark::nodes";
        let (gpu, _w, body) = self.body_parts(WHAT)?;
        let stream = gpu.stream();
        let taps = body
            .taps
            .as_ref()
            .ok_or_else(|| GpuError::state(WHAT, "the taps are not armed (set_taps)"))?;
        let mut out = Vec::with_capacity(body.layers.len());
        for (l, a) in body.layers.iter().enumerate() {
            let t = &taps.layers[l];
            out.push(LayerNodes {
                attn_gate_proj: t.attn_gate_proj.to_host_vec(stream)?,
                attn_gate_sig: t.attn_gate_sig.to_host_vec(stream)?,
                attn_gated: t.attn_gated.to_host_vec(stream)?,
                attn_o_norm: t.attn_o_norm.to_host_vec(stream)?,
                attn_o_proj: t.attn_o_proj.to_host_vec(stream)?,
                attn_post_norm: t.attn_post_norm.to_host_vec(stream)?,
                ffn_inp: t.ffn_inp.to_host_vec(stream)?,
                ffn_latent_down: t.ffn_latent_down.to_host_vec(stream)?,
                ffn_moe_logits: a.route.logits().to_host_vec(stream)?,
                ffn_moe_probs: a.route.probs().to_host_vec(stream)?,
                ffn_moe_topk: a.route.ids().to_host_vec(stream)?,
                ffn_moe_weights: a.route.weights().to_host_vec(stream)?,
                ffn_moe_out: t.ffn_moe_out.to_host_vec(stream)?,
                ffn_latent_up: t.ffn_latent_up.to_host_vec(stream)?,
                l_out: t.l_out.to_host_vec(stream)?,
            });
        }
        Ok(out)
    }

    /// The routing of the layers of `layers` as the last step or pass left it, read back (a gate's instrument; it
    /// synchronizes the stream): [`GpuModel::nodes`]'s routing fields alone, three buffers a layer where `nodes` reads
    /// fourteen. Refused by name: a range past the model's layers.
    pub fn layer_routes(&mut self, layers: Range<usize>) -> Result<Vec<LayerRoute>, GpuError> {
        const WHAT: &str = "musespark::layer_routes";
        let (gpu, _w, body) = self.body_parts(WHAT)?;
        let stream = gpu.stream();
        let n = body.layers.len();
        if layers.start > layers.end || layers.end > n {
            return Err(GpuError::shape(
                WHAT,
                format!("the layers {layers:?} are past the model's {n} layers"),
            ));
        }
        let mut out = Vec::with_capacity(layers.len());
        for a in &body.layers[layers] {
            out.push(LayerRoute {
                ids: a.route.ids().to_host_vec(stream)?,
                logits: a.route.logits().to_host_vec(stream)?,
                probs: a.route.probs().to_host_vec(stream)?,
            });
        }
        Ok(out)
    }

    /// One router launch of layer `layer`'s own instance over a planted
    /// weight whose expert 0 row is NaN and the rest finite and strictly
    /// descending, at a one-hot input: a NaN lane must raise
    /// [`FaultSite::Router`] on the fault word and be picked by no slot
    /// (a gate's instrument; it synchronizes the stream).
    pub fn router_lane_nan(&mut self, layer: usize) -> Result<(bool, bool), GpuError> {
        const WHAT: &str = "musespark::router_lane_nan";
        let (gpu, _w, body) = self.body_parts(WHAT)?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let a = body
            .layers
            .get_mut(layer)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} is none")))?;
        let n_expert = a.route.logits().len();
        let hidden = body.sh.hidden;
        let mut w = vec![0.0f32; n_expert * hidden];
        w[..hidden].fill(f32::NAN);
        for (r, logit) in w.chunks_mut(hidden).enumerate().skip(1) {
            logit[0] = 20.0 - r as f32;
        }
        let mut x = vec![0.0f32; hidden];
        x[0] = 1.0;
        let w_dev = DeviceTensor::upload(stream, &w, n_expert, hidden)?;
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let bias = DeviceBuffer::zeroed(stream, n_expert)?;
        body.k.routers.enqueue(
            stream,
            &w_dev,
            &x_dev,
            &bias,
            body.sh.weights_scale,
            &mut a.route,
            gpu.layer_sink(layer)?,
        )?;
        let ids = a.route.ids().to_host_vec(stream)?;
        let weights = a.route.weights().to_host_vec(stream)?;
        let raised = gpu
            .take_fault()?
            .is_some_and(|f| f.code == FaultSite::Router as u32);
        let defined = ids.len() == body.sh.top_k
            && ids
                .iter()
                .all(|&i| usize::try_from(i).is_ok_and(|i| i > 0 && i < n_expert))
            && ids.iter().collect::<std::collections::BTreeSet<_>>().len() == ids.len()
            && weights.iter().all(|v| v.is_finite());
        Ok((raised, defined))
    }
}
