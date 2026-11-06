//! Kimi K3's chain over one sequence on one card: the one-token step, from llama.cpp
//! `src/models/kimi-k3.cpp`.
//!
//! The step embeds the token, runs every layer and the output head. A layer reads the running
//! sum through the cross-layer residual (AttnRes, [`crate::attn_res`]): the sublayer's input is a
//! softmax mix of the banked block inputs and the sum, scored by the layer's own vector.
//!
//! 1. the mix of the banks and the sum `x` (`attn_res_score`); with no bank yet, `x` itself;
//! 2. on every `attn_res.block_size`th layer, `x` is banked (the raw sum, not the mix);
//! 3. the mixer: a delta layer ([`super::kda`]) or the gated latent layer
//!    ([`crate::arch::gated_mla`], the one gated latent core both Raon-1 and K3 run), each
//!    taking its own attn norm of the mix;
//! 4. the sum after the mixer is the mixer's update alone on a banking layer (the stream
//!    restarts), else `x` plus the update;
//! 5. the mix of the banks and that sum (`ffn_res_score`), the layer's `ffn_norm`, the
//!    feed-forward block (the dense `situ` block or the latent routed block,
//!    [`crate::arch::routed_ffn`]), the sum plus its update.
//!
//! After the last layer one more mix (`output_res_score`) over every bank and the sum goes to the
//! output head, which norms it and reads the logits.
//!
//! Two row buffers ping-pong the sum, so the sum is back in the first after every layer.
//! The state book is [`super::state`]'s: the delta layers' one-lane stores and the latent layers'
//! caches.

use super::kda::{self, KdaSites};
use super::moe::{MoeDims, routed_of};
use super::state::{self, Lane, StepIn};
use crate::arch::gated_mla::{self as mla, MlaBufs, MlaSites};
use crate::arch::hybrid::{Acts, DecodeInput, GqaCx, Site};
use crate::arch::routed_ffn::{self as ffn, Ffn, FfnKernels, FfnScratch, Glu};
use crate::attn_res::{AttnResArgs, AttnResKernels, MAX_VECS};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::FlashGqaKernels;
use crate::gemm::Gemm32Kernels;
use crate::head::Head;
use crate::host::StepLeg;
use crate::host::handoff::HandoffSlots;
use crate::host::run::HostRun;
use crate::host::swap::Residency;
use crate::host::swap_source::{DEADLINE, LIVE_DELAY, ResidencySpec, hosted_stacks};
use crate::host::tier::TierOpen;
use crate::hybrid::{Hybrid, SlotMap};
use crate::kquant::Act;
use crate::latent_gated::{self, MlaDims, MlaKernels};
use crate::linear::mixer_kda::{KdaBufs, KdaDims};
use crate::linear::{HEAD, LinearKernels, PASS_ROWS};
use crate::model::{
    ChainBody, GpuModel, HostServed, block_count, refuse_reserve_past, refuse_scratch_past,
    stage_run,
};
use crate::recur::RecStore;
use crate::route_body::{sigmoid_8_2, sigmoid_896_16};
use crate::routed_block::RouterEntry;
use crate::site::{KGemvKernels, SiteTy};
use crate::tensor::{WindowMut, copy_row};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, LAYER_HEAD};
use bloomery_levers::{ChurnHosting, HostCfg};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::kimi_k3::hparams::{Hparams, Kind};
use model::arch::kimi_k3::names;
use model::arch::kimi_k3::place::PlanInputs;
use model::placement::Plan;
use placed::{Open, Placed};
use runtime::layer::hosted;
use std::ops::Range;
use std::sync::Arc;

#[path = "placed.rs"]
pub(super) mod placed;
#[path = "rows.rs"]
pub(super) mod rows;

const WHAT: &str = "kimi_k3::Body::load";
const STEP: &str = "kimi_k3::Body::enqueue_chain";
const OPEN: &str = "kimi_k3 GpuModel::open";

/// The types the dense layer's projections have in the released set.
const DENSE_TYPES: &[SiteTy] = &[SiteTy::Q6K];

/// A layer's mixer and its store.
enum Mixer {
    /// The recurrent state, its stamp and the conv ring ([`state::kda_store`]).
    Kda {
        sites: Box<KdaSites>,
        store: RecStore,
    },
    /// The latent cache, `[ctx][row]` f16.
    Mla {
        sites: Box<MlaSites>,
        cache: DeviceTensor<u16>,
    },
}

/// One layer: its mixer, its feed-forward block, the name of its `ffn_norm`, the names of its two
/// AttnRes score vectors (the mixer's, the block's) and whether it banks its input.
struct Layer {
    mixer: Mixer,
    ffn: Ffn,
    ffn_norm: String,
    scores: [String; 2],
    banks: bool,
}

impl Layer {
    /// Every site the layer launches, for the load's [`Acts::new`].
    fn sites(&self) -> Vec<&Site> {
        let mut out: Vec<&Site> = match &self.mixer {
            Mixer::Kda { sites, .. } => sites.sites().to_vec(),
            Mixer::Mla { sites, .. } => sites.sites().to_vec(),
        };
        match &self.ffn {
            Ffn::Dense(g) => out.extend(g.sites()),
            Ffn::Routed(r) => out.extend(r.sites()),
        }
        out
    }

    fn store_bytes(&self) -> usize {
        match &self.mixer {
            Mixer::Kda { store, .. } => store.bytes(),
            Mixer::Mla { cache, .. } => cache.buf().num_bytes(),
        }
    }
}

/// What a mixer launch reads and writes: the layer's input row, the step's position words, the
/// scratch and the row the update lands in.
struct MixerIo<'a> {
    input: &'a DeviceBuffer<f32>,
    pos: &'a DeviceBuffer<u32>,
    keys: &'a DeviceBuffer<u32>,
    xn: &'a mut DeviceBuffer<f32>,
    kb: &'a mut KdaBuffers,
    mb: &'a mut MlaBuffers,
    out: &'a mut DeviceBuffer<f32>,
}

/// Layer `l`'s mixer over the row `io.input`: the delta layer's step on its one-lane store, or the
/// gated latent layer's on its cache; the update lands in `io.out`.
fn run_mixer(
    cx: &GqaCx<'_>,
    (k, acts, lane): (&Kernels, &mut Acts, &Lane),
    (l, mixer): (usize, &mut Mixer),
    (kd, md): (&KdaDims, &MlaDims),
    io: MixerIo<'_>,
) -> Result<(), GpuError> {
    let MixerIo {
        input,
        pos,
        keys,
        xn,
        kb,
        mb,
        out,
    } = io;
    match mixer {
        Mixer::Kda { sites, store } => {
            let st = lane.state(store)?;
            let bufs = KdaBufs {
                x: input,
                pos,
                xn,
                qkv: &mut kb.qkv,
                fa: &mut kb.fa,
                ga: &mut kb.ga,
                beta_raw: &mut kb.beta_raw,
                write_raw: None,
                f: &mut kb.f,
                z: &mut kb.z,
                conv: &mut kb.conv,
                beta: &mut kb.beta,
                decay: &mut kb.decay,
                o: &mut kb.o,
                gated: &mut kb.gated,
                out,
            };
            kda::step(cx, &k.linear, acts, sites, (l, kd), st, bufs)
        }
        Mixer::Mla { sites, cache } => {
            let bufs = MlaBufs {
                x: input,
                pos,
                n_keys: keys,
                xn,
                qa: &mut mb.qa,
                qr: &mut mb.qr,
                kv: &mut mb.kv,
                gate: &mut mb.gate,
                q: &mut mb.q,
                qabs: &mut mb.qabs,
                att: &mut mb.att,
                av: &mut mb.av,
                gated: &mut mb.gated,
                part_v: &mut mb.part_v,
                part_ms: &mut mb.part_ms,
                out,
                cache,
            };
            mla::step(cx, &k.mla, acts, (&**sites, md), l, bufs)
        }
    }
}

/// The launch context of the step's layers: the kernels' sites at one row and `ctx` cache rows.
fn step_cx<'a>(
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    hp: &Hparams,
    ctx: usize,
) -> GqaCx<'a> {
    GqaCx {
        what: STEP,
        gpu,
        w,
        kgemv: &k.kgemv,
        g32: &k.g32,
        flash: &k.flash,
        mma: false,
        out_gate: None,
        hidden: hp.n_embd,
        eps: hp.rms_eps,
        n_kv: 1,
        ctx,
    }
}

/// The embedding of the token `step_in.ids` into `y`, and the position words of the row at
/// `step_in.pos0`.
fn embed(
    gpu: &Gpu,
    w: &Weights,
    step_in: &mut StepIn,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let embd = names::token_embd();
    let Some(DevWeight::KQuant { ty, w: table, .. }) = w.get(&embd) else {
        return Err(GpuError::tensor(STEP, &embd, "K-quant rows"));
    };
    gpu.elem().enqueue_embed_rows_kquant(
        gpu.stream(),
        *ty,
        EmbedRowsArgs {
            w: table,
            ids: &step_in.ids,
            pos0: &step_in.pos0,
            first: 0,
            y,
            pos: &mut step_in.pos,
            n_keys: &mut step_in.n_keys,
        },
    )
}

/// The kernels the step launches beside the `Gpu`'s own and the router instance's.
struct Kernels {
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash: FlashGqaKernels,
    linear: LinearKernels,
    mla: MlaKernels,
    attn_res: AttnResKernels,
}

impl Kernels {
    fn load(gpu: &Gpu, mla: &MlaDims) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            mla: MlaKernels::load(gpu, mla, PASS_ROWS)?,
            attn_res: AttnResKernels::load(ctx)?,
        })
    }
}

/// The feed-forward kernels and one row's scratch at one router instance.
struct Router<R: RouterEntry> {
    k: FfnKernels<R>,
    s: FfnScratch<R>,
    /// The hosted layers' buffers, on a placed load.
    p: Option<placed::Bufs<R>>,
}

impl<R: RouterEntry> Router<R> {
    fn load(
        gpu: &Gpu,
        n: usize,
        ff: usize,
        dims: &MoeDims,
        placed: bool,
    ) -> Result<Router<R>, GpuError> {
        Ok(Router {
            k: FfnKernels::load(gpu)?,
            s: FfnScratch::new(gpu.stream(), n, ff, dims.shape())?,
            p: placed
                .then(|| placed::Bufs::new(gpu.stream(), n, dims))
                .transpose()?,
        })
    }

    /// The kernels and the hosted layers' buffers; refused by name on a load that is not placed.
    fn hosted(&mut self) -> Result<(&FfnKernels<R>, &mut placed::Bufs<R>), GpuError> {
        let Router { k, p, .. } = self;
        let p = p
            .as_mut()
            .ok_or_else(|| GpuError::state(STEP, "a placed load's hosted-layer buffers"))?;
        Ok((k, p))
    }
}

/// The router instance the file's expert counts select (the file's 896 experts over the top 16,
/// the fixture's 8 over the top 2): one constant of the step, so one match on it.
enum Instance {
    E896(Router<sigmoid_896_16::RouterKernels>),
    E8(Router<sigmoid_8_2::RouterKernels>),
}

impl Instance {
    /// Layer `l`'s block over its normed row `xn` into `y`, the branch's update.
    fn run(
        &mut self,
        cx: &GqaCx<'_>,
        acts: &mut Acts,
        (rule, scale): (Act, f32),
        (l, f): (usize, &Ffn),
        xn: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        match self {
            Instance::E896(r) => ffn::ffn(cx, &r.k, acts, (rule, scale), (l, f), xn, &mut r.s, y),
            Instance::E8(r) => ffn::ffn(cx, &r.k, acts, (rule, scale), (l, f), xn, &mut r.s, y),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Instance::E896(r) => r.s.bytes() + r.p.as_ref().map_or(0, placed::Bufs::bytes),
            Instance::E8(r) => r.s.bytes() + r.p.as_ref().map_or(0, placed::Bufs::bytes),
        }
    }

    /// A hosted layer's front ([`placed::front`]) at this instance.
    fn front(
        &mut self,
        b: &placed::Block<'_>,
        acts: &mut Acts,
        side: &mut placed::Side,
        hy: &mut Hybrid<HostRun>,
    ) -> Result<(), GpuError> {
        match self {
            Instance::E896(r) => placed::front(r.hosted()?, b, acts, side, hy),
            Instance::E8(r) => placed::front(r.hosted()?, b, acts, side, hy),
        }
    }

    /// A hosted layer's shadow ([`placed::shadow`]) at this instance.
    fn shadow(
        &mut self,
        b: &placed::Block<'_>,
        acts: &mut Acts,
        side: &placed::Side,
        hy: &Hybrid<HostRun>,
    ) -> Result<(), GpuError> {
        match self {
            Instance::E896(r) => placed::shadow(r.hosted()?, b, acts, side, hy),
            Instance::E8(r) => placed::shadow(r.hosted()?, b, acts, side, hy),
        }
    }

    /// A hosted layer's back ([`placed::back`]) at this instance.
    fn back(
        &mut self,
        b: &placed::Block<'_>,
        acts: &mut Acts,
        side: &placed::Side,
        (hy, update): (&mut Hybrid<HostRun>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        match self {
            Instance::E896(r) => placed::back(r.hosted()?.1, b, acts, side, hy, update),
            Instance::E8(r) => placed::back(r.hosted()?.1, b, acts, side, hy, update),
        }
    }

    fn picks(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        match self {
            Instance::E896(r) => r.s.picks(stream),
            Instance::E8(r) => r.s.picks(stream),
        }
    }
}

/// The step's row buffers, `n` or the mixer's widths, one row each.
struct Scratch {
    /// The running sum, ping-ponged (module doc).
    sum: [DeviceBuffer<f32>; 2],
    /// A mix's output, a mixer's update, the block's update.
    mixed: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    /// The mixer's normed row and the block's.
    xn: DeviceBuffer<f32>,
    ffn_xn: DeviceBuffer<f32>,
    kda: KdaBuffers,
    mla: MlaBuffers,
}

/// The delta mixer's intermediates ([`KdaBufs`]).
struct KdaBuffers {
    qkv: DeviceBuffer<f32>,
    fa: DeviceBuffer<f32>,
    ga: DeviceBuffer<f32>,
    beta_raw: DeviceBuffer<f32>,
    f: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    conv: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    decay: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
}

/// The latent mixer's intermediates ([`MlaBufs`]).
struct MlaBuffers {
    qa: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qabs: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
}

impl Scratch {
    fn new(
        stream: &CudaStream,
        n: usize,
        (kd, md): (&KdaDims, &MlaDims),
        ctx: usize,
    ) -> Result<Scratch, GpuError> {
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len.max(1));
        let lin = kd.shape;
        let (channels, v, n_v) = (lin.channels(), lin.n_v * HEAD, lin.n_v);
        let (part_v, part_ms) = latent_gated::partials_lens_rows(md, ctx, 1);
        let (heads, row) = (md.heads, md.row());
        Ok(Scratch {
            sum: [z(n)?, z(n)?],
            mixed: z(n)?,
            update: z(n)?,
            xn: z(n)?,
            ffn_xn: z(n)?,
            kda: KdaBuffers {
                qkv: z(channels)?,
                fa: z(HEAD)?,
                ga: z(1)?,
                beta_raw: z(n_v)?,
                f: z(v)?,
                z: z(v)?,
                conv: z(channels)?,
                beta: z(n_v)?,
                decay: z(v)?,
                o: z(v)?,
                gated: z(v)?,
            },
            mla: MlaBuffers {
                qa: z(md.q_lora)?,
                qr: z(md.q_lora)?,
                kv: z(row)?,
                gate: z(heads * md.v_head)?,
                q: z(heads * md.head_k())?,
                qabs: z(heads * row)?,
                att: z(heads * crate::latent::LATENT)?,
                av: z(heads * md.v_head)?,
                gated: z(heads * md.v_head)?,
                part_v: z(part_v)?,
                part_ms: z(part_ms)?,
            },
        })
    }

    fn bytes(&self) -> usize {
        let k = &self.kda;
        let m = &self.mla;
        [
            &self.sum[0],
            &self.sum[1],
            &self.mixed,
            &self.update,
            &self.xn,
            &self.ffn_xn,
            &k.qkv,
            &k.fa,
            &k.ga,
            &k.beta_raw,
            &k.f,
            &k.z,
            &k.conv,
            &k.beta,
            &k.decay,
            &k.o,
            &k.gated,
            &m.qa,
            &m.qr,
            &m.kv,
            &m.gate,
            &m.q,
            &m.qabs,
            &m.att,
            &m.av,
            &m.gated,
            &m.part_v,
            &m.part_ms,
        ]
        .into_iter()
        .map(DeviceBuffer::num_bytes)
        .sum()
    }
}

/// The Kimi K3 chain over one sequence on one card.
/// The stages a tap snapshots in each layer, named as the vendor graph names its nodes
/// (`mixer_out` is `kda_out` or `mla_out`): [`Body::taps`] holds them layer by layer in this order.
pub const TAP_STAGES: [&str; 10] = [
    "attn_norm",
    "kda_beta",
    "kda_g1",
    "kda_scan_out",
    "kda_g2",
    "mixer_out",
    "prefix_sum_attn",
    "ffn_norm",
    "ffn_out",
    "l_out",
];
const TAP_ATTN_NORM: usize = 0;
const TAP_BETA: usize = 1;
const TAP_G1: usize = 2;
const TAP_SCAN: usize = 3;
const TAP_G2: usize = 4;
const TAP_MIXER_OUT: usize = 5;
const TAP_PREFIX: usize = 6;
const TAP_FFN_NORM: usize = 7;
const TAP_FFN_OUT: usize = 8;
const TAP_L_OUT: usize = 9;

/// The stage rows of one step, `stride` values a stage (the widest row a stage holds).
struct Taps {
    buf: DeviceBuffer<f32>,
    stride: usize,
}

impl Taps {
    /// Stage `stage` of layer `l` takes a copy of `src` (cut at the stride). Asynchronous.
    fn put(
        &mut self,
        stream: &CudaStream,
        (l, stage): (usize, usize),
        src: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let len = src.len().min(self.stride);
        let at = (l * TAP_STAGES.len() + stage) * self.stride * 4;
        let mut slot = WindowMut::<f32>::of_mut(&mut self.buf, at, len)?;
        copy_row(stream, src, &mut slot, len)
    }
}

pub struct Body {
    hp: Hparams,
    kda: KdaDims,
    mla: MlaDims,
    act: Act,
    layers: Vec<Layer>,
    k: Kernels,
    acts: Acts,
    s: Scratch,
    /// The banked block inputs, `stride = n_embd` values apart, one slot a banking layer.
    banks: DeviceBuffer<f32>,
    step_in: StepIn,
    lane: Lane,
    router: Instance,
    /// The host tier and what the walk reads beside it, on a placed load ([`placed`]).
    placed: Option<Placed>,
    /// The stage snapshots of the step's position: armed by [`Body::arm_taps`], absent in a
    /// serving load so no launch is recorded for them.
    taps: Option<Taps>,
    /// The position the stores stand at: the next step's.
    held: u32,
    ctx: usize,
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head: every tensor in its
    /// kernels' device format, the latent layers' caches of `ctx` rows and the delta layers' stores
    /// of one lane. The model takes the file and closes it once the weights are resident.
    ///
    /// A file whose tensors exceed the card's free bytes is refused by name before any tensor is
    /// read: the released file does not fit one card, and its placement over two stage cards, the
    /// host tier and the NVMe tier has no loader.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, OPEN)?;
        let (free, total) = gpu.mem_info()?;
        let bytes: u64 = file.iter_tensors().map(|(_, t)| t.nbytes).sum();
        if usize::try_from(bytes).map_or(true, |b| b > free) {
            return Err(GpuError::shape(
                OPEN,
                format!(
                    "the file's tensors are {bytes} B and the card has {free} of {total} B free: \
                     the body holds the whole model on one card, and no loader places it over \
                     two stage cards, the host tier and the NVMe tier"
                ),
            ));
        }
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            let hp = Hparams::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
            for l in (0..hp.n_layer).filter(|&l| hp.kind(l) == Kind::Kda) {
                kda::derive(gpu.stream(), w, l)?;
            }
            Body::load(gpu, &file, w, (hp, ctx), None)
        })
    }

    /// Card `card` of `plan`, which `inputs` made, resident: the dense trunk and each routed
    /// layer's card experts the plan puts there, the stores at the plan's `ctx_max`, the host tier
    /// over the layers whose experts the card does not hold whole with the plan's expert tier
    /// card hung under it, and the embedding as a host row ([`placed`]). The load's slot map
    /// stays for the model's life ([`Residency::Off`]). The prompt is stepped one id at a time.
    /// Refused by name: a plan of other than one stage card running every layer, a plan that
    /// reads the head on the host (no card carries it: the step ends in a card head, and no host
    /// head stage exists), of more than one expert tier, a layer whose stacks are not its card
    /// count's, a routed layer whose experts the host holds some of and no tier card any (its
    /// handoff is the host's own entry, which the slot count a token must be one of
    /// [`HandoffSlots`] builds), and scratch or tier places past the plan's terms for them.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        const PLAN: &str = "kimi_k3::Body::open_placed";
        let hp = &inputs.hp;
        let (_, ctx) = stage_run(PLAN, plan, card, hp.n_layer)?;
        if !plan.machine.cards.get(card).is_some_and(|c| c.head) {
            return Err(GpuError::shape(
                PLAN,
                format!(
                    "the plan reads the head on the host (card {card} does not carry it): the \
                     step ends in a card head and the body has no host head stage"
                ),
            ));
        }
        let run = hosted(&inputs.spec.layers)
            .map_err(|e| GpuError::shape(PLAN, format!("the routed layers: {e}")))?;
        refuse_tierless_host_layers(PLAN, plan, run.clone(), hp)?;
        let spec = ResidencySpec {
            lever: Residency::Off,
            pool: ChurnHosting::Hosted,
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k: hp.experts.n_used,
            max_rows: placed::LOAD_ROWS,
            stacks: Arc::new(hosted_stacks(&inputs.model, PLAN, (run.len(), true))?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |stream, _, layers, w| {
                for l in layers.filter(|&l| hp.kind(l) == Kind::Kda) {
                    kda::derive(stream, w, l)?;
                }
                Ok(())
            },
            |gpu, file, w, set, glue| {
                let open = Open {
                    file,
                    plan,
                    inputs,
                    host,
                    set,
                    glue,
                };
                let mut body =
                    Body::load(gpu, file, w, (inputs.hp.clone(), ctx), Some((open, card)))?;
                refuse_scratch_past(
                    PLAN,
                    plan,
                    card,
                    body.scratch_bytes(),
                    "the step's buffers, the hosted layers' buffers, the host boundary, the slot \
                     map's copy",
                )?;
                let reserve = inputs
                    .prompt_batch_bytes(plan.machine)
                    .map_err(|e| GpuError::plan(PLAN, e))?;
                if let Some(p) = body.placed.as_mut() {
                    refuse_reserve_past(
                        PLAN,
                        "tier places",
                        reserve,
                        p.tier_bytes() as u64,
                        "the slot map's tier view and the step's tier places",
                    )?;
                    p.start_residency(gpu)?;
                }
                Ok(body)
            },
        )
    }
}

/// The dense layer's projections of layer `l`.
fn dense_glu(file: &Split, w: &Weights, l: usize, hp: &Hparams) -> Result<Glu, GpuError> {
    let (n, ff) = (hp.n_embd, hp.dense_ff);
    let site = |name: String, dims: (usize, usize)| {
        Site::of_file((file, w), WHAT, name, dims, DENSE_TYPES)
    };
    let build = || -> Result<Glu, GpuError> {
        Ok(Glu {
            gate: site(names::ffn_gate(l), (ff, n))?,
            up: site(names::ffn_up(l), (ff, n))?,
            down: site(names::ffn_down(l), (n, ff))?,
            ff,
        })
    };
    build().map_err(|e| GpuError::shape(WHAT, format!("layer {l}'s dense block: {e}")))
}

/// The routed layers `run` the host holds experts of and no tier card holds any, refused by name
/// when the slot count a token is none of the host handoff's entries ([`HandoffSlots`]): such a
/// layer's front is the plain host handoff, and a tier layer's is the any-count tier entry.
fn refuse_tierless_host_layers(
    what: &'static str,
    plan: &Plan<'_>,
    run: Range<usize>,
    hp: &Hparams,
) -> Result<(), GpuError> {
    if HandoffSlots::of(hp.experts.n_used).is_ok() {
        return Ok(());
    }
    let at = |per: &[u64], l: usize| per.get(l).copied().unwrap_or(0);
    let tierless: Vec<(usize, u64)> = run
        .filter_map(|l| {
            let tier: u64 = plan.tier_n_l.iter().map(|t| at(t, l)).sum();
            let held = at(&plan.n_l, l) + tier;
            let n_expert = hp.experts.n_expert as u64;
            (tier == 0 && held < n_expert).then_some((l, n_expert - held))
        })
        .collect();
    if tierless.is_empty() {
        return Ok(());
    }
    Err(GpuError::shape(
        what,
        format!(
            "{} routed layers keep experts on the host and none on a tier card (layer, host \
             experts): {:?}; the host handoff's entries are built for 4, 6, 8, 9 and 10 slots a \
             token and this model routes {}, which only a tier layer's any-count entry serves",
            tierless.len(),
            &tierless[..tierless.len().min(8)],
            hp.experts.n_used,
        ),
    ))
}

/// The slot map of a placed load's plan over the routed run, the stage card `card`'s and the
/// expert tier cards': one row of `n_expert` places a routed layer. Refused by name: a plan of
/// other than one stage card running every layer, more tier cards than the host tier serves, and
/// the map's own refusals.
fn placed_map(open: &Open<'_>, card: usize, hp: &Hparams) -> Result<SlotMap, GpuError> {
    stage_run(WHAT, open.plan, card, hp.n_layer)?;
    let tiers = TierOpen::of_machine(open.plan.machine)?;
    let tier_cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
    let run = hosted(&open.inputs.spec.layers)
        .map_err(|e| GpuError::shape(WHAT, format!("the routed layers: {e}")))?;
    SlotMap::of_plan_tiers(open.plan, card, &tier_cards, run, hp.experts.n_expert)
}

/// The stores the load made against the plan's count of them: the plan counts the layers' stores
/// by the header's widths, and a load whose stores differ holds another number of bytes than the
/// plan fitted. Refused by name with both figures.
fn refuse_stores(layers: &[Layer], open: &Open<'_>, card: usize) -> Result<(), GpuError> {
    let held: u64 = layers.iter().map(|l| l.store_bytes() as u64).sum();
    let planned = open.inputs.kv.bytes(0..layers.len(), open.plan.ctx_max);
    let counted = open
        .plan
        .cards
        .get(card)
        .map(|t| t.kv_bytes)
        .ok_or_else(|| GpuError::shape(WHAT, format!("the plan has no card {card}")))?;
    if held != planned || counted != planned {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the stores hold {held} device bytes; the plan's layout counts {planned} and \
                 card {card}'s plan {counted}"
            ),
        ));
    }
    Ok(())
}

impl Body {
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        (hp, ctx): (Hparams, usize),
        placed: Option<(Open<'_>, usize)>,
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        let map = placed
            .as_ref()
            .map(|(open, card)| placed_map(open, *card, &hp))
            .transpose()?;
        let n = hp.n_embd;
        if hp.attn_res_block == 0 {
            return Err(GpuError::shape(
                WHAT,
                "attn_res.block_size is 0: every K3 file mixes through the cross-layer residual",
            ));
        }
        let bank_slots = hp.n_layer.div_ceil(hp.attn_res_block);
        if bank_slots + 1 > MAX_VECS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} layers in blocks of {} bank {bank_slots} inputs; one mix takes at most \
                     {MAX_VECS} vectors with the sum",
                    hp.n_layer, hp.attn_res_block
                ),
            ));
        }
        let kd = kda::dims(&hp)?;
        let md = MlaDims {
            embd: n,
            heads: hp.n_head,
            q_lora: hp.mla.q_lora,
            nope: hp.mla.nope(),
            extra: hp.mla.pe,
            v_head: hp.mla.head_v,
            rms_eps: hp.rms_eps,
            latent_eps: hp.rms_eps,
        };
        if md.row() != hp.mla.row() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the latent row is {} values in the file; the card's latent layer caches {}",
                    hp.mla.row(),
                    md.row()
                ),
            ));
        }
        let moe = MoeDims::of(&hp);
        if (0..hp.n_layer).any(|l| !hp.dense(l)) {
            moe.check()?;
        }
        let act = moe.rule();
        act.code(WHAT)?;
        let vector = |name: String| crate::site::vector(w, WHAT, name, n);
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mixer = match hp.kind(l) {
                Kind::Kda => Mixer::Kda {
                    sites: Box::new(KdaSites::load(file, w, &kd, l)?),
                    store: state::kda_store(stream, kd.shape)?,
                },
                Kind::Mla => Mixer::Mla {
                    sites: Box::new(MlaSites::load(file, w, &md, l)?),
                    cache: DeviceTensor::zeroed(stream, ctx, md.row())?,
                },
            };
            let (ffn, ffn_norm) = if hp.dense(l) {
                (
                    Ffn::Dense(dense_glu(file, w, l, &hp)?),
                    vector(names::ffn_norm(l))?,
                )
            } else {
                let stacks = match &map {
                    Some(m) => m.on_card(l)? > 0,
                    None => true,
                };
                let (routed, norm) = routed_of((file, w), l, &moe, stacks)?;
                (Ffn::Routed(routed), norm)
            };
            layers.push(Layer {
                mixer,
                ffn,
                ffn_norm,
                scores: [
                    vector(names::attn_res_score(l))?,
                    vector(names::ffn_res_score(l))?,
                ],
                banks: hp.banks(l),
            });
        }
        vector(names::OUTPUT_RES_SCORE.to_owned())?;
        let acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        let ff = hp.dense_ff.max(moe.shared_ff);
        let hosted_load = placed.is_some();
        let router = if (moe.n_expert, moe.n_used) == super::moe::INSTANCES[0] {
            Instance::E896(Router::load(gpu, n, ff, &moe, hosted_load)?)
        } else {
            Instance::E8(Router::load(gpu, n, ff, &moe, hosted_load)?)
        };
        let placed = match (placed, map) {
            (Some((open, card)), Some(map)) => {
                refuse_stores(&layers, &open, card)?;
                Some(Placed::new((gpu, w), &layers, map, open)?)
            }
            _ => None,
        };
        Ok(Body {
            placed,
            k: Kernels::load(gpu, &md)?,
            s: Scratch::new(stream, n, (&kd, &md), ctx)?,
            banks: DeviceBuffer::zeroed(stream, bank_slots * n)?,
            step_in: StepIn::new(stream)?,
            lane: Lane::new(stream)?,
            router,
            taps: None,
            held: 0,
            kda: kd,
            mla: md,
            act,
            layers,
            acts,
            ctx,
            hp,
        })
    }

    /// Device bytes the body holds past the weights and the stores.
    fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.banks.num_bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
            + self.step_in.bytes()
            + self.lane.bytes()
            + self.router.bytes()
            + self.acts.bytes()
            + self.k.kgemv.device_bytes()
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
    }

    /// The placed side of the load: `None` on the whole-card load.
    #[must_use]
    pub fn placed(&self) -> Option<&Placed> {
        self.placed.as_ref()
    }

    /// Device bytes of the sequence's stores: the delta layers' states and rings, the latent
    /// layers' caches.
    pub fn store_bytes(&self) -> usize {
        self.layers.iter().map(Layer::store_bytes).sum()
    }

    /// Arms the stage taps: every later step also copies [`TAP_STAGES`] of each layer (the
    /// vendor's node values) for [`Body::taps`]. Arm before the first step in graph mode: a
    /// captured chain keeps what it recorded.
    ///
    /// # Errors
    /// [`GpuError`] from the allocation.
    pub fn arm_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stride = self.hp.n_embd.max(self.hp.n_head * self.hp.kda.head_dim);
        self.taps = Some(Taps {
            buf: DeviceBuffer::zeroed(gpu.stream(), self.layers.len() * TAP_STAGES.len() * stride)?,
            stride,
        });
        Ok(())
    }

    /// The stage rows of the last step's position: `(layers · stages) · stride` values, layer by
    /// layer in [`TAP_STAGES`] order, and the stride. A delta layer's stages hold zeros on a
    /// latent layer. Synchronizes the stream. Refused by name until [`Body::arm_taps`].
    ///
    /// # Errors
    /// [`GpuError`] when the taps are not armed or from the readback.
    pub fn taps(&self, stream: &CudaStream) -> Result<(Vec<f32>, usize), GpuError> {
        let t = self
            .taps
            .as_ref()
            .ok_or_else(|| GpuError::shape("kimi_k3::Body::taps", "taps read before arm_taps"))?;
        Ok((t.buf.to_host_vec(stream)?, t.stride))
    }

    /// The last routed block's picks as the step left them: the kept expert ids in rank order and
    /// their weights. Synchronizes the stream.
    ///
    /// # Errors
    /// [`GpuError`] from the readback.
    pub fn last_picks(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        self.router.picks(stream)
    }

    /// Layer `l`'s mixer run alone over `rows`, the vendor's normed inputs (`attn_norm`) at
    /// positions `0..rows.len()`, from a fresh store: the update at the last position. The rows
    /// are divided by the layer's norm gain so the mixer's own norm gives them back. The layer's
    /// store is left at the last position: reset the model before the next step.
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model, an empty or too long `rows`, a row not `n_embd`
    /// wide, a norm gain that is zero or not finite, or from the launches.
    pub fn mixer_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rows: &[Vec<f32>],
    ) -> Result<Vec<f32>, GpuError> {
        const ALONE: &str = "kimi_k3::Body::mixer_alone";
        let stream = gpu.stream();
        let n = self.hp.n_embd;
        if rows.is_empty() || rows.len() > self.ctx {
            return Err(GpuError::shape(
                ALONE,
                format!("{} rows in caches of {}", rows.len(), self.ctx),
            ));
        }
        let gain = w
            .f32_buf(ALONE, &names::attn_norm(l))?
            .to_host_vec(stream)?;
        if gain.len() != n || gain.iter().any(|g| *g == 0.0 || !g.is_finite()) {
            return Err(GpuError::shape(
                ALONE,
                format!(
                    "layer {l}: a norm gain of {} values that is not all nonzero finite",
                    gain.len()
                ),
            ));
        }
        let Body {
            hp,
            kda: kd,
            mla: md,
            layers,
            k,
            acts,
            s,
            step_in,
            lane,
            ctx,
            ..
        } = self;
        let n_layers = layers.len();
        let layer = layers
            .get_mut(l)
            .ok_or_else(|| GpuError::shape(ALONE, format!("layer {l} of {n_layers}")))?;
        if let Mixer::Kda { store, .. } = &mut layer.mixer {
            state::clear_store(stream, store)?;
        }
        let cx = step_cx(gpu, w, k, hp, *ctx);
        let Scratch {
            sum,
            mixed,
            update,
            xn,
            kda: kb,
            mla: mb,
            ..
        } = s;
        for (t, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(GpuError::shape(
                    ALONE,
                    format!("row {t} holds {} values, a layer's input is {n}", row.len()),
                ));
            }
            let x: Vec<f32> = row.iter().zip(&gain).map(|(v, g)| v / g).collect();
            step_in.ids.copy_from_host(stream, &[0])?;
            step_in.pos0.copy_from_host(
                stream,
                &[u32::try_from(t).map_err(|e| GpuError::shape(ALONE, e.to_string()))?],
            )?;
            embed(gpu, w, step_in, &mut sum[0])?;
            mixed.copy_from_host(stream, &x)?;
            let io = MixerIo {
                input: &*mixed,
                pos: &step_in.pos,
                keys: &step_in.n_keys,
                xn: &mut *xn,
                kb: &mut *kb,
                mb: &mut *mb,
                out: &mut *update,
            };
            run_mixer(
                &cx,
                (k, &mut *acts, &*lane),
                (l, &mut layer.mixer),
                (kd, md),
                io,
            )?;
        }
        Ok(update.to_host_vec(stream)?)
    }

    /// Layer `l`'s feed-forward block run alone on the normed row `xn`: the block's update, and a
    /// routed block's picked expert ids in rank order. Touches no stored state.
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model, a row not `n_embd` wide or from the launches.
    pub fn ffn_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        xn: &[f32],
    ) -> Result<(Vec<f32>, Option<Vec<u32>>), GpuError> {
        const ALONE: &str = "kimi_k3::Body::ffn_alone";
        let stream = gpu.stream();
        if self.placed.is_some() {
            return Err(GpuError::shape(
                ALONE,
                "a placed body's routed block is split across the card, the tier and the host: \
                 it runs no block alone",
            ));
        }
        if xn.len() != self.hp.n_embd {
            return Err(GpuError::shape(
                ALONE,
                format!(
                    "a row of {} values, a layer's input is {}",
                    xn.len(),
                    self.hp.n_embd
                ),
            ));
        }
        let Body {
            hp,
            act,
            layers,
            k,
            acts,
            s,
            router,
            ctx,
            ..
        } = self;
        let layer = layers
            .get(l)
            .ok_or_else(|| GpuError::shape(ALONE, format!("layer {l} of {}", layers.len())))?;
        let cx = step_cx(gpu, w, k, hp, *ctx);
        s.ffn_xn.copy_from_host(stream, xn)?;
        router.run(
            &cx,
            acts,
            (*act, hp.experts.weights_scale),
            (l, &layer.ffn),
            &s.ffn_xn,
            &mut s.update,
        )?;
        let out = s.update.to_host_vec(stream)?;
        let picks = match layer.ffn {
            Ffn::Routed(_) => Some(router.picks(stream)?.0),
            Ffn::Dense(_) => None,
        };
        Ok((out, picks))
    }

    /// The step's launches in order (module doc): the embedding, every layer, the final mix, the
    /// head. A placed load has no embedding launch (its row is the refresh's) and walks its layers
    /// through the host tier's step port ([`placed`]).
    fn walk(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let Body {
            hp,
            kda: kd,
            mla: md,
            act,
            layers,
            k,
            acts,
            s,
            banks,
            step_in,
            lane,
            router,
            placed,
            taps,
            ctx,
            ..
        } = self;
        if placed.is_none() {
            embed(gpu, w, step_in, &mut s.sum[0])?;
        }
        let scale = hp.experts.weights_scale;
        let cx = step_cx(gpu, w, k, hp, *ctx);
        let mut walk = Walk {
            cx,
            k,
            acts,
            lane,
            kd,
            md,
            s,
            banks,
            step_in,
            taps: taps.as_mut(),
            n: hp.n_embd,
            eps: hp.rms_eps,
            pushed: 0,
        };
        match placed.as_mut() {
            None => {
                for (l, layer) in layers.iter_mut().enumerate() {
                    walk.attn_half(l, layer)?;
                    router.run(
                        &walk.cx,
                        &mut *walk.acts,
                        (*act, scale),
                        (l, &layer.ffn),
                        &walk.s.ffn_xn,
                        &mut walk.s.update,
                    )?;
                    walk.close(l)?;
                }
            }
            Some(Placed { shell, side, .. }) => {
                let mut leg = StepLeg::new(gpu.stream(), &mut shell.hybrid);
                let mut step = placed::StepWalk {
                    walk,
                    layers,
                    router,
                    side,
                    rule: (*act, scale),
                    latent: hp.latent,
                };
                step.run(&mut leg)?;
                walk = step.walk;
            }
        }
        walk.finish(head)
    }
}

/// The step's borrowed parts and the walk's one counter, the layers' common sequence (module
/// doc): [`Walk::attn_half`] runs a layer to its normed feed-forward input, the block runs
/// (whole, or around the host tier's go, [`placed`]) and [`Walk::close`] adds its update to the
/// sum.
struct Walk<'a> {
    cx: GqaCx<'a>,
    k: &'a Kernels,
    acts: &'a mut Acts,
    lane: &'a Lane,
    kd: &'a KdaDims,
    md: &'a MlaDims,
    s: &'a mut Scratch,
    banks: &'a mut DeviceBuffer<f32>,
    step_in: &'a StepIn,
    taps: Option<&'a mut Taps>,
    n: usize,
    eps: f32,
    /// Block inputs banked so far this step.
    pushed: usize,
}

impl Walk<'_> {
    /// Layer `l`'s steps 1 to 5 up to the block: the mix for the mixer, the bank, the mixer, the
    /// sum after it (`pb`), and the feed-forward mix normed into `ffn_xn`.
    fn attn_half(&mut self, l: usize, layer: &mut Layer) -> Result<(), GpuError> {
        let Walk {
            cx,
            k,
            acts,
            lane,
            kd,
            md,
            s,
            banks,
            step_in,
            taps,
            n,
            eps,
            pushed,
        } = self;
        let (gpu, w) = (cx.gpu, cx.w);
        let stream = gpu.stream();
        let (n, eps) = (*n, *eps);
        let Scratch {
            sum,
            mixed,
            update,
            xn,
            ffn_xn,
            kda: kb,
            mla: mb,
        } = &mut **s;
        let [pa, pb] = sum;
        let fault = gpu.layer_sink(l)?;
        let [attn_score, ffn_score] = &layer.scores;
        let (pos, keys) = (&step_in.pos, &step_in.n_keys);
        // 1. The mixer's input: the mix over the banks, or the sum itself with no bank yet.
        let input: &DeviceBuffer<f32> = if *pushed > 0 {
            k.attn_res.enqueue(
                stream,
                AttnResArgs {
                    banks,
                    n_banks: *pushed,
                    stride: n,
                    cur: &*pa,
                    score: w.f32_buf(STEP, attn_score)?,
                    eps,
                    n,
                    m: 1,
                    fault,
                },
                mixed,
            )?;
            &*mixed
        } else {
            &*pa
        };
        // 2. A banking layer banks the raw sum.
        if layer.banks {
            let mut slot = WindowMut::<f32>::of_mut(&mut **banks, *pushed * n * 4, n)?;
            copy_row(stream, &*pa, &mut slot, n)?;
            *pushed += 1;
        }
        // 3. The mixer. A banking layer's update is the next sum: it lands in the other sum
        // buffer.
        let out: &mut DeviceBuffer<f32> = if layer.banks { &mut *pb } else { &mut *update };
        let io = MixerIo {
            input,
            pos,
            keys,
            xn: &mut *xn,
            kb: &mut *kb,
            mb: &mut *mb,
            out,
        };
        run_mixer(
            cx,
            (k, &mut **acts, lane),
            (l, &mut layer.mixer),
            (kd, md),
            io,
        )?;
        if let Some(t) = taps.as_mut() {
            let mixer_out: &DeviceBuffer<f32> = if layer.banks { &*pb } else { &*update };
            t.put(stream, (l, TAP_ATTN_NORM), &*xn)?;
            t.put(stream, (l, TAP_MIXER_OUT), mixer_out)?;
            if matches!(layer.mixer, Mixer::Kda { .. }) {
                t.put(stream, (l, TAP_BETA), &kb.beta)?;
                t.put(stream, (l, TAP_G1), &kb.decay)?;
                t.put(stream, (l, TAP_SCAN), &kb.o)?;
                t.put(stream, (l, TAP_G2), &kb.z)?;
            }
        }
        // 4. The sum after the mixer: its update alone on a banking layer, else the sum plus it.
        if !layer.banks {
            gpu.elem().enqueue_add(stream, &*pa, &*update, n, pb)?;
        }
        if let Some(t) = taps.as_mut() {
            t.put(stream, (l, TAP_PREFIX), &*pb)?;
        }
        // 5. The block's input: the mix of the banks and the sum, the layer's norm.
        k.attn_res.enqueue(
            stream,
            AttnResArgs {
                banks,
                n_banks: *pushed,
                stride: n,
                cur: &*pb,
                score: w.f32_buf(STEP, ffn_score)?,
                eps,
                n,
                m: 1,
                fault,
            },
            mixed,
        )?;
        gpu.elem().enqueue_rms_norm(
            stream,
            &*mixed,
            w.f32_buf(STEP, &layer.ffn_norm)?,
            eps,
            n,
            1,
            ffn_xn,
        )
    }

    /// Layer `l`'s block update (`update`) added to the sum after the mixer (`pb`) into the
    /// first sum buffer, and the layer's block taps.
    fn close(&mut self, l: usize) -> Result<(), GpuError> {
        let stream = self.cx.gpu.stream();
        let n = self.n;
        let Scratch {
            sum,
            update,
            ffn_xn,
            ..
        } = &mut *self.s;
        let [pa, pb] = sum;
        self.cx
            .gpu
            .elem()
            .enqueue_add(stream, &*pb, &*update, n, pa)?;
        if let Some(t) = self.taps.as_mut() {
            t.put(stream, (l, TAP_FFN_NORM), &*ffn_xn)?;
            t.put(stream, (l, TAP_FFN_OUT), &*update)?;
            t.put(stream, (l, TAP_L_OUT), &*pa)?;
        }
        Ok(())
    }

    /// The final mix over every bank and the sum into the head's input, and the head.
    fn finish(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let n = self.n;
        self.k.attn_res.enqueue(
            gpu.stream(),
            AttnResArgs {
                banks: &*self.banks,
                n_banks: self.pushed,
                stride: n,
                cur: &self.s.sum[0],
                score: w.f32_buf(STEP, names::OUTPUT_RES_SCORE)?,
                eps: self.eps,
                n,
                m: 1,
                fault: gpu.fault_sink(LAYER_HEAD),
            },
            head.input_mut(),
        )?;
        head.enqueue(gpu, w)
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = Placed;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::KimiK3
    }

    /// The step at `pos`; a position past the caches is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(GpuError::shape(
                "kimi_k3::Body::decode_input",
                format!("a step at position {pos} in caches of {}", self.ctx),
            ));
        }
        Ok(DecodeInput { token, pos })
    }

    /// The token and the position: two copies. Refused by name, before either, off the position
    /// the stores stand at.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        if input.pos != self.held {
            return Err(GpuError::shape(
                STEP,
                format!(
                    "a step at position {}; the stores stand at {} (a call that failed \
                     part-way: reset)",
                    input.pos, self.held
                ),
            ));
        }
        match self.placed.as_mut() {
            None => {
                self.step_in.ids.copy_from_host(stream, &[input.token])?;
                self.step_in.pos0.copy_from_host(stream, &[input.pos])?;
            }
            Some(p) => p.refresh(
                stream,
                input.token,
                (
                    &mut self.s.sum[0],
                    &mut self.step_in.pos,
                    &mut self.step_in.n_keys,
                ),
                input.pos,
            )?,
        }
        self.held = input.pos + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.walk(gpu, w, head)
    }

    /// Every delta store back to a fresh one in place (a captured chain keeps their addresses,
    /// [`state::clear_store`]). The latent caches need nothing: no row at or past the live count
    /// is read, and every row below it is written by its own step first.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for layer in &mut self.layers {
            if let Mixer::Kda { store, .. } = &mut layer.mixer {
                state::clear_store(stream, store)?;
            }
        }
        if let Some(p) = self.placed.as_mut() {
            p.reset(stream)?;
        }
        self.held = 0;
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every block norm's, the head's included.
    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes() + self.scratch_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}
