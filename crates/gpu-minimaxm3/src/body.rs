//! The MiniMax-M3 body: the load, each layer's store, the buffers the step runs over, and [`Body`] behind
//! `GpuModel`.
//!
//! All 60 layers are one card's chain, each a GQA layer with a per-head q/k norm and a partial NEOX rope, over a
//! feed-forward block: the leading layers a dense block of their own width, the rest a routed block with one shared
//! expert, whose experts run on the card where the plan puts them ([`crate::ffn::Routed`]) and on the host tier
//! otherwise. A sparse layer also holds an index key (one f16 row of the indexer's width) for every position and
//! attends the blocks its indexer picks ([`crate::attn`]); a dense layer attends every position. Every store is a
//! plane of positions that a later position never reads before it writes, so a cut back to a position is the count
//! alone. The embedding is a host row, dequantized per step and copied into `x` before the launch.
//!
//! What the load refuses, by name: a plan of other than one card with no expert tier, a layer that is not a GQA
//! layer with the q/k norm and the partial rope over a block of the kinds above, widths the kernels have no
//! geometry for, a tensor whose type no site launches or that is not resident in the rows its launch reads, stores
//! whose bytes are not the plan's, a step scratch past the plan's term, and `slots` other than one.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::arch::hybrid::{Acts, GqaGate, GqaLayer, Site};
use bloomery_gpu::flash_gqa::{GROUP_16, HEAD, partials_ms_len, partials_v_len};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::served::{ResidencyParts, TierBody};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::{
    DEADLINE, LIVE_DELAY, PlanStacks, ResidencyGlue, ResidencySpec, hosted_stacks,
};
use bloomery_gpu::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use bloomery_gpu::model::{
    ChainBody, HostServed, Rollback, agreed, hosted_hybrid, one_card_run, refuse_scratch_past,
};
use bloomery_gpu::rope_neox::ROT_PARTIAL;
use bloomery_gpu::rope_table::{RopeRows, RopeSpec, RopeTable};
use bloomery_gpu::site::{self, SiteTy};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use bloomery_levers::{ChurnHosting, HostCfg};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::minimaxm3::host as minimax_host;
use model::arch::minimaxm3::hparams::Kind;
use model::arch::minimaxm3::place::PlanInputs;
use model::arch::names;
use model::embed::EmbedTable;
use model::placement::Plan;
use models::{Act, Ffn, Mixer, OutGate, Selector};
use runtime::layer::{Layer, MixerKind, ResidualKind, hosted};

use crate::attn::{DIM, HEADS, MsaScratch, MsaShape, N_HEAD, ROT, partials_lens};
use crate::ffn::{CARD, Routed, RoutedDims};
use crate::step::Kernels;

/// What the body's refusals name.
pub(crate) const WHAT: &str = "minimaxm3 Body";

/// The rows in flight a load's host boundary and residency tally hold: the step's one (the file has no draft to
/// verify rows of).
pub const LOAD_ROWS: usize = 1;

/// The MiniMax-M3 model: one card, the skeleton over this body.
pub type MiniMaxM3Model = GpuModel<Body>;

/// The input of one decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInput {
    pub token: u32,
    pub pos: u32,
}

/// The file's widths and constants, read once.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub embd: usize,
    pub vocab: usize,
    /// Query heads and key heads of a GQA layer (64 and 4: group 16).
    pub heads: usize,
    pub kv_heads: usize,
    pub rms_eps: f32,
    /// The rope's base; its table is [`ROT`] values a position.
    pub rope_base: f32,
    /// Routed experts' and the shared expert's widths.
    pub expert_ff: usize,
    pub shared_ff: usize,
    pub ctx: usize,
}

/// A sparse layer's indexer: its two projections, the gains of its two heads' norms and the index keys of every
/// position (f16, [`DIM`] values a position).
pub struct IndexLayer {
    pub q: Site,
    pub k: Site,
    pub q_norm: String,
    pub k_norm: String,
    pub keys: DeviceBuffer<u16>,
}

/// One layer's store and typed sites: the GQA layer's four sites and K/V planes (common type), the gains of the
/// per-head q/k norms, and the indexer on a sparse layer.
pub struct Store {
    pub gqa: GqaLayer,
    pub q_norm: String,
    pub k_norm: String,
    pub index: Option<IndexLayer>,
}

impl Store {
    /// Device bytes the store holds, the sites' weights not included.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.gqa.bytes() + self.index.as_ref().map_or(0, |i| i.keys.num_bytes())
    }

    /// Every site the layer launches.
    pub(crate) fn sites(&self) -> Vec<&Site> {
        let mut sites = self.gqa.sites();
        if let Some(i) = &self.index {
            sites.extend([&i.q, &i.k]);
        }
        sites
    }
}

/// The step's one-row buffers: the streams, the attention's intermediates, the indexer's, the selection's and the
/// activation forms.
pub struct Scratch {
    pub x: DeviceBuffer<f32>,
    pub x1: DeviceBuffer<f32>,
    pub xn: DeviceBuffer<f32>,
    /// A sub-layer's output before its residual add.
    pub delta: DeviceBuffer<f32>,
    /// The feed-forward block's input of a forced row, when the caller gave
    /// one in place of the attention sub-layer's residual.
    pub forced_x: DeviceBuffer<f32>,
    pub pos: DeviceBuffer<u32>,
    pub n_keys: DeviceBuffer<u32>,
    /// The query projection, its normed and turned heads, the key and value rows, the flash's output.
    pub q_raw: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub attn: DeviceBuffer<f32>,
    pub part_v: DeviceBuffer<f32>,
    pub part_ms: DeviceBuffer<f32>,
    /// The index heads' projection, their normed and turned heads, the index key row, and the value row and the
    /// plane the head kernel needs for the index key it appends ([`crate::attn::IndexArgs`]).
    pub iq_raw: DeviceBuffer<f32>,
    pub iq: DeviceBuffer<f32>,
    pub ik: DeviceBuffer<f32>,
    pub zeros: DeviceBuffer<f32>,
    pub sink: DeviceBuffer<u16>,
    pub msa: MsaScratch,
    pub acts: Acts,
}

impl Scratch {
    /// The one row's buffers over every width the sites read, for a cache of `d.ctx` positions. Load-time only.
    fn new<'s>(
        stream: &CudaStream,
        (d, shape): (&Dims, &MsaShape),
        sites: impl Iterator<Item = &'s Site>,
    ) -> Result<Scratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        let (q_width, kv_width) = (d.heads * HEAD, d.kv_heads * HEAD);
        let (msa_v, msa_ms) = partials_lens(shape, 1);
        Ok(Scratch {
            x: z(d.embd)?,
            x1: z(d.embd)?,
            xn: z(d.embd)?,
            delta: z(d.embd)?,
            forced_x: z(d.embd)?,
            pos: u(1)?,
            n_keys: u(1)?,
            q_raw: z(q_width)?,
            q: z(q_width)?,
            k: z(kv_width)?,
            v: z(kv_width)?,
            attn: z(q_width)?,
            part_v: z(partials_v_len(1, d.heads).max(msa_v))?,
            part_ms: z(partials_ms_len(1, d.heads).max(msa_ms))?,
            iq_raw: z(HEADS * DIM)?,
            iq: z(HEADS * DIM)?,
            ik: z(DIM)?,
            zeros: z(DIM)?,
            sink: DeviceBuffer::zeroed(stream, d.ctx * DIM)?,
            msa: MsaScratch::new(stream, shape, 1, d.ctx)?,
            acts: Acts::new(stream, sites)?,
        })
    }

    /// Device bytes of every buffer and activation form.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let f32s = [
            &self.x,
            &self.x1,
            &self.xn,
            &self.delta,
            &self.forced_x,
            &self.q_raw,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.part_v,
            &self.part_ms,
            &self.iq_raw,
            &self.iq,
            &self.ik,
            &self.zeros,
        ]
        .into_iter()
        .map(DeviceBuffer::num_bytes)
        .sum::<usize>();
        let words = [&self.pos, &self.n_keys]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();
        f32s + words + self.sink.num_bytes() + self.msa.bytes() + self.acts.bytes()
    }
}

/// The token embedding as the host reads it, and the step's host row.
pub(crate) struct HostIn {
    pub(crate) table: EmbedTable,
    row: Vec<f32>,
}

/// The MiniMax body. Field order is drop order: the shell, whose host tier (its lock over the host set and its page
/// windows) comes first, before the buffers.
pub struct Body {
    /// The host tier, the residency glue and the taps.
    pub(crate) shell: TierShell<HostRun>,
    pub(crate) dims: Dims,
    pub(crate) cfg: Vec<Layer>,
    pub(crate) stores: Vec<Store>,
    pub(crate) s: Scratch,
    pub(crate) slots: Arc<DeviceTensor<u32>>,
    /// The positions the stores hold: a step at `pos` needs `pos <= at`, and it leaves `pos + 1`.
    pub(crate) at: u32,
    /// The kernels the layers launch besides `Gpu`'s own.
    pub(crate) k: Kernels,
    /// The rope table of the 64 turned values a position, which the head kernels of the attention and the indexer
    /// both read.
    pub(crate) rope: RopeRows,
    pub(crate) shape: MsaShape,
    /// The routed blocks and the dense ones, the card experts among them.
    pub(crate) ffn: Routed,
    /// The step the last [`ChainBody::decode_input`] accepted: what `enqueue_chain` walks.
    cur: StepInput,
    pub(crate) embd: HostIn,
}

/// A shape or state the load or the step refuses, by name.
pub(crate) fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// Site types a projection of the file may be: Q8_0 (the released file's), and the K-quants.
const PROJ: &[SiteTy] = &[
    SiteTy::Q8_0,
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
];

/// Site types of the indexer's projections: the released file's f32, or any projection type.
const INDEX: &[SiteTy] = &[
    SiteTy::F32,
    SiteTy::Q8_0,
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
];

/// A layer's description held to what the kernels run: a GQA layer over every position of [`HEAD`]-wide heads at
/// group [`GROUP_16`] with the per-head q/k norm and the partial rope, and the block selector exactly when `sparse`.
fn attention_checked(l: usize, mixer: &Mixer, sparse: bool) -> Result<(), GpuError> {
    let Mixer::Gqa(g) = mixer else {
        return Err(shape(format!(
            "layer {l}: a {mixer:?} mixer; this program runs GQA"
        )));
    };
    let plain = g.qk_norm
        && g.out_gate == OutGate::None
        && g.window.is_none()
        && !g.sinks
        && g.value_scale.is_none()
        && matches!(g.select, Some(Selector::BlockMax { .. })) == sparse
        && (g.select.is_none() || sparse);
    if !plain {
        return Err(shape(format!(
            "layer {l}: {g:?} is not a q/k-normed GQA layer over {} positions",
            if sparse {
                "the picked blocks"
            } else {
                "every position"
            }
        )));
    }
    if g.head_dim as usize != HEAD || g.value_dim as usize != HEAD {
        return Err(shape(format!(
            "layer {l}: heads of {} keys and {} values; the kernels are built for {HEAD}",
            g.head_dim, g.value_dim
        )));
    }
    if g.group() as usize != GROUP_16 {
        return Err(shape(format!(
            "layer {l}: {} query heads over {} key heads; the flash runs {GROUP_16} a key head",
            g.heads, g.kv_heads
        )));
    }
    Ok(())
}

/// The file's widths and constants, each checked against the kernel that fixes it, from the description's layers.
fn dims_of(inputs: &PlanInputs, ctx: usize) -> Result<(Dims, MsaShape), GpuError> {
    let (hp, spec) = (&inputs.hp, &inputs.spec);
    let msa = MsaShape::of(hp)?;
    if !hp.n_embd.is_multiple_of(256) {
        return Err(shape(format!(
            "embedding_length {} is not whole K-quant blocks of 256",
            hp.n_embd
        )));
    }
    if hp.rope.dims != ROT || hp.n_head != N_HEAD {
        return Err(shape(format!(
            "{} query heads and a rope over {} values; the kernels are built for {N_HEAD} and {ROT}",
            hp.n_head, hp.rope.dims
        )));
    }
    let mut kv = None;
    for (l, s) in spec.layers.iter().enumerate() {
        attention_checked(
            l,
            &s.mixer,
            matches!(&s.mixer, Mixer::Gqa(g) if g.select.is_some()),
        )?;
        if let Mixer::Gqa(g) = &s.mixer {
            agreed(WHAT, "the key head count", l, &mut kv, g.kv_heads as usize)?;
        }
    }
    let kv_heads = kv.ok_or_else(|| shape("no layer in the description"))?;
    Ok((
        Dims {
            embd: hp.n_embd,
            vocab: hp.n_vocab,
            heads: hp.n_head,
            kv_heads,
            rms_eps: hp.rms_eps,
            rope_base: hp.rope.base,
            expert_ff: hp.experts.ff,
            shared_ff: hp.experts.shared_ff(),
            ctx,
        },
        msa,
    ))
}

/// The stacks of the layers `inputs` describes for the residency machine's file source: each routed layer's gate,
/// up and down, their types read from its tensors; the dense layers hold none. Refused by name: a layer that holds
/// some of its three stacks but not all, and a description whose routed layers are not one run.
fn stacks(inputs: &PlanInputs) -> Result<PlanStacks, GpuError> {
    let run = hosted(&inputs.spec.layers).map_err(|e| shape(format!("the routed layers: {e}")))?;
    hosted_stacks(&inputs.model, "minimaxm3 stacks", (run.len(), true))
}

impl Body {
    /// Card `card` of `plan`, which `inputs` made, resident: the layers' weights the plan puts on the card, the
    /// stores at the plan's `ctx_max`, the host tier over every routed layer and the embedding as a host row; the
    /// load's slot map for the model's life ([`Residency::Off`]). Refused by name (module doc).
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<MiniMaxM3Model, GpuError> {
        Body::open_placed_slots(
            file,
            plan,
            inputs,
            card,
            host,
            Residency::Off,
            NonZeroUsize::MIN,
        )
    }

    /// [`Body::open_placed`] by a plan that counts `slots` resident sequences ([`PlanInputs::plan_with_slots`]).
    /// The body holds one: more are refused by name, and a plan that counts another number of sequences is refused
    /// at the load.
    ///
    /// Under `residency` the host set also holds each layer's churn pool ([`GpuModel::load_placed_with`]) and the
    /// body starts the machine over its slot map once its pieces are sized ([`HostServed::start_residency`]);
    /// [`Residency::Off`] keeps the load's slot map for the model's life.
    pub fn open_placed_slots(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
        slots: NonZeroUsize,
    ) -> Result<MiniMaxM3Model, GpuError> {
        if slots.get() > 1 {
            return Err(shape(format!(
                "{slots} resident sequences: the program holds one sequence's stores and has no slots to add"
            )));
        }
        let spec = ResidencySpec {
            lever: residency,
            pool: ChurnHosting::Hosted,
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k: inputs.hp.experts.n_used,
            max_rows: LOAD_ROWS,
            stacks: Arc::new(stacks(inputs)?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set, glue| {
                Body::load(gpu, file, w, (plan, inputs, card), host, (set, glue))
            },
        )
    }

    /// The body of card `card`: its layers' programs and values from the plan's description, the stores at the
    /// plan's `ctx_max`, the step's buffers, the slot map the plan's routed segments make, the routed blocks and
    /// the host tier over the routed run. `glue` is the load's residency side, whose machine the body starts once
    /// its pieces are sized.
    fn load(
        gpu: &Gpu,
        file: &Arc<Split>,
        w: &Weights,
        (plan, inputs, card): (&Plan<'_>, &PlanInputs, usize),
        host: HostCfg,
        (set, glue): (HostResidency, ResidencyGlue),
    ) -> Result<Body, GpuError> {
        let (hp, spec) = (&inputs.hp, &inputs.spec);
        gpu.context().bind_to_thread()?;
        let (layers, ctx) = one_card_run(WHAT, plan, card, spec.layers.len())?;
        let (dims, msa) = dims_of(inputs, ctx)?;
        let cfg: Vec<Layer> = spec.layers.iter().map(Layer::of).collect();
        for (l, kind) in cfg.iter().enumerate() {
            if kind.residual != ResidualKind::Plain || kind.mixer != MixerKind::Gqa {
                return Err(shape(format!(
                    "layer {l}: {kind:?}; the minimaxm3 chain is a plain residual over GQA layers"
                )));
            }
        }
        let run = hosted(&spec.layers).map_err(|e| shape(e.to_string()))?;
        let map = SlotMap::of_plan(plan, card, None, run.clone(), hp.experts.n_expert)?;
        let stores = (0..spec.layers.len())
            .map(|l| store(gpu, file, w, &dims, l, hp.kind(l) == Kind::Sparse))
            .collect::<Result<Vec<_>, _>>()?;
        let held: u64 = stores.iter().map(|s| s.bytes() as u64).sum();
        let planned = inputs.kv.bytes(layers.clone(), plan.ctx_max);
        if held != planned {
            return Err(shape(format!(
                "the stores hold {held} device bytes; the plan's layout counts {planned}"
            )));
        }
        let counted = plan
            .cards
            .get(card)
            .map(|t| t.kv_bytes)
            .ok_or_else(|| shape(format!("the plan has no card {card}")))?;
        if counted != planned {
            return Err(shape(format!(
                "the plan counts {counted} store bytes on card {card}; the load holds {planned}"
            )));
        }
        let stream = gpu.stream();
        let s = Scratch::new(stream, (&dims, &msa), stores.iter().flat_map(Store::sites))?;
        let table = RopeTable::new(&RopeSpec::window(dims.rope_base, ROT_PARTIAL))?;
        let rope = RopeRows::new(stream, &table, ROT_PARTIAL, ctx)?;
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            run.len(),
            hp.experts.n_expert,
        )?);
        let ffn = Routed::new(
            gpu,
            file,
            w,
            (&spec.layers, &map, &slots),
            RoutedDims {
                embd: dims.embd,
                rms_eps: dims.rms_eps,
                batch_rows: 1,
            },
            CARD,
        )?;
        let rules = (0..spec.layers.len())
            .map(|l| routed_oai(spec, l))
            .collect::<Result<Vec<_>, _>>()?;
        let hybrid = hosted_hybrid(
            (gpu, file, plan),
            (run.clone(), map),
            (host, set, &glue),
            (
                HostWidths {
                    embd: hp.n_embd,
                    ff: hp.experts.ff,
                    n_used: hp.experts.n_used,
                },
                LOAD_ROWS,
            ),
            |src| {
                run.clone()
                    .map(|l| minimax_host::layer(src, hp, l, rules[l].0, rules[l].1))
                    .collect()
            },
        )?;
        let embd = EmbedTable::new(Arc::clone(file), &names::token_embd(), dims.embd)?;
        if embd.n_vocab() != hp.n_vocab {
            return Err(shape(format!(
                "the embedding has {} rows, the file's vocabulary {}",
                embd.n_vocab(),
                hp.n_vocab
            )));
        }
        let mut body = Body {
            shell: TierShell::new(hybrid, glue, dims.embd, WHAT),
            cfg,
            stores,
            s,
            slots,
            at: 0,
            k: Kernels::load(gpu)?,
            rope,
            shape: msa,
            ffn,
            embd: HostIn {
                table: embd,
                row: vec![0.0; dims.embd],
            },
            cur: StepInput { token: 0, pos: 0 },
            dims,
        };
        refuse_scratch_past(
            WHAT,
            plan,
            card,
            body.scratch_bytes(),
            "the step's buffers, the routed blocks' buffers and card experts, the kernels, the rope table, the \
             host boundary, the slot map's copy",
        )?;
        // Last: the machine frees each layer's spare slots, and every piece above sized itself from the load's
        // map, capacity = live.
        body.start_residency(gpu)?;
        Ok(body)
    }

    /// Positions every store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.dims.ctx
    }

    /// The file's widths and constants the load read.
    #[must_use]
    pub fn dims(&self) -> &Dims {
        &self.dims
    }

    /// Each layer's programs, in layer order.
    #[must_use]
    pub fn kinds(&self) -> Vec<Layer> {
        self.cfg.clone()
    }

    /// The positions the stores hold.
    #[must_use]
    pub fn held(&self) -> u32 {
        self.at
    }

    /// Device bytes of the stores: every layer's planes and the sparse layers' index keys.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store::bytes).sum()
    }

    /// Device bytes of the step's side of the card beside the stores and the weights: the step's buffers, the
    /// routed blocks' buffers and card experts, the kernels' own, the rope table, the host boundary and the slot
    /// map's copy. The term the plan sets aside as scratch and stage rows.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.ffn.bytes()
            + self.k.bytes()
            + self.rope.table.num_bytes()
            + self.shell.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
    }

    /// Every store's and buffer's bytes on the card, the armed taps included.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.store_bytes()
            + self.scratch_bytes()
            + self
                .shell
                .taps
                .as_ref()
                .map_or(0, |t| t.iter().map(DeviceBuffer::num_bytes).sum())
    }

    /// The token embedding's row of `token`, dequantized into `row` (one row of `n_embd` values); a token past the
    /// vocabulary is refused by name.
    pub fn embed(&self, token: u32, row: &mut [f32]) -> Result<(), GpuError> {
        self.embd.table.row_into(token, row)?;
        Ok(())
    }

    /// Arm (or disarm) the per-layer taps: after each layer the chain copies its output into the layer's tap. The
    /// caller drops its captured chains first: a capture holds the tap copies it was taken with.
    pub fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        self.shell.set_taps(gpu, self.cfg.len(), on)
    }

    /// Every layer's output after the last step, `n_embd` a layer. Blocking; refused when the taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        self.shell.taps(gpu)
    }

    /// A cut to `pos`: the stores hold `pos` positions again, those past it overwritten before any later position
    /// reads them. A position past the held ones is refused by name.
    pub fn cut(&mut self, pos: u32) -> Result<(), GpuError> {
        if pos > self.at {
            return Err(shape(format!(
                "a cut to position {pos} on stores holding {}",
                self.at
            )));
        }
        self.at = pos;
        Ok(())
    }
}

/// Arm (or disarm) `m`'s per-layer taps ([`Body::taps`] reads them back), the model first set to eager steps: a
/// captured chain holds the tap copies it was taken with, so arming under one would leave the taps unwritten and
/// disarming would free buffers its replays write. A model set back to graph steps afterwards captures its chain
/// with the taps.
pub fn set_taps(m: &mut MiniMaxM3Model, on: bool) -> Result<(), GpuError> {
    m.set_mode(bloomery_gpu::model::StepMode::Eager);
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.set_taps(gpu, on)
}

/// Layer `l`'s routed experts' SwiGLU-OAI constants `(alpha, limit)`; any other activation is refused by name.
fn routed_oai(spec: &models::ModelSpec, l: usize) -> Result<(f32, f32), GpuError> {
    match spec.layers.get(l).map(|s| &s.ffn) {
        Some(Ffn::Moe(m)) => match m.act {
            Act::SwigluOai { alpha, limit } => Ok((alpha, limit)),
            other => Err(shape(format!(
                "layer {l}: the routed activation is {other:?}; this program runs SwiGLU-OAI"
            ))),
        },
        // A dense layer has no routed experts: its entry is never read.
        Some(_) => Ok((0.0, 0.0)),
        None => Err(shape(format!("layer {l} past the description"))),
    }
}

/// Layer `l`'s store and typed sites: the sites' types from the file, checked against the resident weights; the
/// indexer's on a sparse layer.
fn store(
    gpu: &Gpu,
    file: &Split,
    w: &Weights,
    d: &Dims,
    l: usize,
    sparse: bool,
) -> Result<Store, GpuError> {
    let stream = gpu.stream();
    let site = |name: String, rows_k: (usize, usize), allowed: &[SiteTy]| {
        Site::of_file((file, w), WHAT, name, rows_k, allowed)
    };
    let embd = d.embd;
    let (q_rows, kv_rows) = (d.heads * HEAD, d.kv_heads * HEAD);
    let gqa = GqaLayer {
        norm: site::vector(w, WHAT, names::attn_norm(l), embd)?,
        q: site(names::attn_q(l), (q_rows, embd), PROJ)?,
        k: site(names::attn_k(l), (kv_rows, embd), PROJ)?,
        v: site(names::attn_v(l), (kv_rows, embd), PROJ)?,
        o: site(names::attn_output(l), (embd, q_rows), PROJ)?,
        gate: GqaGate::None,
        kc: DeviceTensor::zeroed(stream, d.kv_heads * d.ctx, HEAD)?,
        vc: DeviceTensor::zeroed(stream, d.kv_heads * d.ctx, HEAD)?,
    };
    let index = if sparse {
        Some(IndexLayer {
            q: site(names::indexer_q_proj(l), (HEADS * DIM, embd), INDEX)?,
            k: site(names::indexer_k_proj(l), (DIM, embd), INDEX)?,
            q_norm: site::vector(w, WHAT, names::indexer_q_norm(l), DIM)?,
            k_norm: site::vector(w, WHAT, names::indexer_k_norm(l), DIM)?,
            keys: DeviceBuffer::zeroed(stream, d.ctx * DIM)?,
        })
    } else {
        None
    };
    Ok(Store {
        gqa,
        q_norm: site::vector(w, WHAT, names::attn_q_norm(l), HEAD)?,
        k_norm: site::vector(w, WHAT, names::attn_k_norm(l), HEAD)?,
        index,
    })
}

impl ChainBody for Body {
    type Input = StepInput;
    type Host = Body;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::MiniMaxM3
    }

    /// The step at `pos`: its embedding row read from the file. A position past the stores, a token past the
    /// vocabulary, or a position past the ones the stores hold is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        if pos as usize >= self.dims.ctx {
            return Err(shape(format!(
                "a step at position {pos} in stores of {}",
                self.dims.ctx
            )));
        }
        if pos > self.at {
            return Err(shape(format!(
                "a step at position {pos} on stores holding {} positions",
                self.at
            )));
        }
        self.embd.table.row_into(token, &mut self.embd.row)?;
        self.cur = StepInput { token, pos };
        Ok(self.cur)
    }

    /// The embedding row into `x`, the position and the live key count: three host-to-device copies. Once they are
    /// sent the stores count the position: what follows launches.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        let pos = input.pos;
        self.s.x.copy_from_host(stream, &self.embd.row)?;
        self.s.pos.copy_from_host(stream, &[pos])?;
        self.s.n_keys.copy_from_host(stream, &[pos + 1])?;
        self.at = pos + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let input = self.cur;
        crate::step::enqueue_step(gpu, w, self, input, head)
    }

    /// The host tier settled and the stores empty by count. The planes keep their rows: a position is written
    /// before a later one reads it.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.shell.hybrid.settle(gpu.stream())?;
        self.at = 0;
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every RMS norm's, the head's included.
    fn head_eps(&self) -> f32 {
        self.dims.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.cfg.len()
    }

    fn host(&mut self) -> Option<&mut Body> {
        Some(self)
    }
}

impl TierBody for Body {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.shell.hybrid()
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.shell.hybrid_mut()
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.shell.residency_parts(&self.slots)
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.shell.hybrid.serve_captured_of(chain)
    }
}

impl Drop for Body {
    /// The residency machine stops before any field frees card or pinned memory ([`Hybrid::stop_swap`]): the host
    /// tier is the first field and stops it in its own drop, and this holds that order as the machine's rule
    /// rather than the field list's.
    fn drop(&mut self) {
        self.shell.hybrid.stop_swap();
    }
}

impl Rollback for Body {
    /// A cut to `pos` ([`Body::cut`]).
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.cut(pos)
    }
}
