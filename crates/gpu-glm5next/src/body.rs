//! The GLM-5.3-Flash body behind `GpuModel`: the load by a placement plan
//! ([`Body::open_placed`]), each layer's store, the step's buffers, and the
//! chain the skeleton captures ([`ChainBody`]).
//!
//! A layer's programs come from its description once, at load
//! ([`runtime::layer::Layer::of`] over the plan's `ModelSpec`), never from
//! its number: a KDA or a latent mixer, a dense or a routed block, both
//! wrapped in the four hyper-connection streams. Each sub-layer mixes the
//! streams by its own `hc_pre` (the Own rule): `hc_pre` and the fold give its
//! input, `hc_post` writes the next streams from its output, and the fold
//! `hc_post` also writes is never read. After the last layer the streams'
//! mean is the head's input.
//!
//! The routed experts run on the card where the plan puts them and on the
//! host tier otherwise, the slot map made from the plan saying which: a
//! routed layer's front ends at the go, its shadow is its card experts and
//! the shared expert, its back the wait, the sum and `hc_post`
//! ([`crate::ffn`]). The step walks the layers with the runtime's
//! one-token walk ([`crate::program`]).
//!
//! The token's embedding row is read from the file on the host and written
//! into the first stream buffer four times, one copy before the chain; its
//! position and the attention's visible counts are two more words the
//! captured chain reads, and the step's live count (its position plus one)
//! a third, which the k-pool selector reads. Each latent layer caches every
//! position it has seen, its index rows and the key of every pool of four
//! they complete, and attends the positions its selector lists (every one
//! while it sees at most `top_k / kpool` pools, [`crate::mla`]); the plan
//! serves a context up to the deepest one a reference set checks the
//! selector at (`place::ORACLE_POSITIONS`). The next-token (MTP) layer is carried by the
//! file and not loaded: its tensors are `Role::Unused`, as ik loads and does
//! not run it.
//!
//! A cut behind the fed positions finds a KDA layer's state only in a
//! checkpoint ([`bloomery_gpu::checkpoint`]): each prompt call ([`prompt`])
//! copies every KDA layer's state and conv ring to the host at the marks of
//! `runtime::seqstate`, and a cut to one of them copies it back at the next
//! step; the latent rows are cut by position.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::checkpoint::Checkpoints;
use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::{
    Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap, refuse_expert_tiers,
};
use bloomery_gpu::kpool::{self, KpoolKernels};
use bloomery_gpu::latent::{INDEX_HEAD, INDEX_ROW, LATENT, LatentKernels, POOL, pools_for};
use bloomery_gpu::linear::{HEAD, KHeadMap, LinearKernels, LinearShape, PASS_ROWS};
use bloomery_gpu::model::{ChainBody, HostServed, Rollback, StepKernels, StepMode};
use bloomery_gpu::qsa::{QsaKernels, list_width};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Branch, DeviceTensor, Gpu, GpuError, GpuModel};
use bloomery_gpu_deepseek41::attn::{self, AttnKernels};
use bloomery_gpu_deepseek41::chain::ffn::FfnKernels;
use bloomery_gpu_deepseek41::experts::ExpertKernels;
use bloomery_gpu_deepseek41::hc::{HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcPreScratch};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED, RouterKernels, RouterOut};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::quant::dequant_row;
use gguf::{GgmlType, Split, TensorInfo};
use model::arch::Arch;
use model::arch::glm5next::names;
use model::arch::glm5next::place::{self, PlanInputs};
use model::placement::Plan;
use models::{Act, Ffn, LayerSpec, Mixer, Score};
use runtime::layer::{FfnKind, Layer, MixerKind, ResidualKind, hosted};
use runtime::seqstate::{HOST_BUDGET, Kept, Take};

use crate::ffn::CardExperts;
use crate::host::GlmHost;
use crate::program;
use crate::tensors::LayerNames;

#[path = "prefill.rs"]
pub mod prefill;

/// What the body's errors name.
const WHAT: &str = "glm5next Body";

// The plan sizes each KDA layer's conv ring by its own pass width.
const _: () = assert!(place::PASS_ROWS == PASS_ROWS);

/// The GLM model: one card, the skeleton over this body.
pub type Glm5nextModel = GpuModel<Body>;

/// The spacing of a prompt call's inner checkpoints; past the host budget's
/// slots the oldest is evicted (`runtime::seqstate`).
pub const CHECKPOINT_EVERY: u32 = 512;

/// The kernels the step launches, loaded once.
pub(crate) struct Kernels {
    pub step: StepKernels,
    pub linear: LinearKernels,
    pub latent: LatentKernels,
    pub hc: HcKernels,
    pub attn: AttnKernels,
    pub router: RouterKernels,
    pub experts: ExpertKernels,
    pub ffn: FfnKernels,
    /// The k-pool selector's score and top-k passes, and the stream its
    /// launches run on beside the latent mixer's query ([`crate::mla`]).
    pub kpool: KpoolKernels,
    pub qsa: QsaKernels,
    pub branch: Branch,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            step: StepKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            latent: LatentKernels::load(ctx)?,
            hc: HcKernels::load(ctx)?,
            attn: AttnKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            ffn: FfnKernels::load(ctx)?,
            kpool: KpoolKernels::load(ctx)?,
            qsa: QsaKernels::load(ctx)?,
            branch: Branch::new(ctx)?,
        })
    }
}

/// The file's widths and constants the launches take, read once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dims {
    pub embd: usize,
    /// Latent query heads, and KDA key and value heads.
    pub heads: usize,
    pub kda: LinearShape,
    pub q_lora: usize,
    /// A query head's values before absorption, and the attention's scale's
    /// root.
    pub head_k: usize,
    /// A head's output values after `attn_v_b`.
    pub head_v: usize,
    /// The index key's width, and its pool gate's.
    pub index_d: usize,
    pub rms_eps: f32,
    /// The index keys' LayerNorm.
    pub norm_eps: f32,
    /// Pools the k-pool selector keeps: `top_k / kpool`.
    pub kept: usize,
    pub hc_eps: f32,
    pub hc_iters: u32,
    /// The KDA decay's lower bound.
    pub lb: f32,
    /// `expert_weights_scale`.
    pub scale: f32,
}

impl Dims {
    /// The row of the joined `[q_a; latent; key; gate]` projection.
    pub(crate) fn stack(&self) -> usize {
        self.q_lora + LATENT + 2 * self.index_d
    }
}

/// One layer's programs and the values its launches take.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerCfg {
    pub kind: Layer,
    /// The dense block's or the shared expert's SwiGLU limit; 0 for the
    /// plain combine.
    pub limit: f32,
    /// The dense block's or the shared expert's width.
    pub ff: usize,
    /// The router's selection bias is in the file.
    pub bias: bool,
    /// The layer's row in the slot map's card copy, as a word offset
    /// ([`SlotMap::row_offset`]).
    pub row_off: usize,
}

/// A layer's store: a KDA layer's recurrent state and conv ring, a latent
/// layer's latent rows and index rows, one row a position, and its pool
/// plane, one key a pool of [`POOL`] positions.
pub(crate) enum Store {
    Kda {
        state: DeviceBuffer<f32>,
        ring: DeviceBuffer<f32>,
    },
    Latent {
        latent: DeviceTensor<u16>,
        index: DeviceTensor<u16>,
        pooled: DeviceTensor<u16>,
    },
}

impl Store {
    fn bytes(&self) -> usize {
        match self {
            Store::Kda { state, ring } => state.num_bytes() + ring.num_bytes(),
            Store::Latent {
                latent,
                index,
                pooled,
            } => latent.buf().num_bytes() + index.buf().num_bytes() + pooled.buf().num_bytes(),
        }
    }

    /// A KDA layer's state and conv ring, the stores a checkpoint copies.
    fn copied(&mut self) -> Option<[&mut DeviceBuffer<f32>; 2]> {
        match self {
            Store::Kda { state, ring } => Some([state, ring]),
            Store::Latent { .. } => None,
        }
    }

    fn zero(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        match self {
            Store::Kda { state, ring } => {
                state.zero_async(stream)?;
                ring.zero_async(stream)?;
            }
            Store::Latent {
                latent,
                index,
                pooled,
            } => {
                latent.buf_mut().zero_async(stream)?;
                index.buf_mut().zero_async(stream)?;
                pooled.buf_mut().zero_async(stream)?;
            }
        }
        Ok(())
    }
}

/// Every buffer the step writes or reads besides the stores, one token's.
pub(crate) struct Scratch {
    /// The four streams, ping-ponged: a sub-layer reads one and writes the
    /// other. The embedding writes buffer 0 before the chain, and every
    /// chain starts there.
    pub streams: [DeviceBuffer<f32>; 2],
    /// A sub-layer's input (the fold by its own mix), its normed form and
    /// its output.
    pub x: DeviceBuffer<f32>,
    pub xn: DeviceBuffer<f32>,
    pub out: DeviceBuffer<f32>,
    /// The fold `hc_post` writes beside the streams: never read.
    pub fold: DeviceBuffer<f32>,
    pub mixes: DeviceBuffer<f32>,
    pub hc: DeviceBuffer<f32>,
    pub hc_scratch: HcPreScratch,
    // A KDA mixer's.
    pub qkv: DeviceBuffer<f32>,
    pub conv: DeviceBuffer<f32>,
    pub fa: DeviceBuffer<f32>,
    pub ga: DeviceBuffer<f32>,
    pub beta_raw: DeviceBuffer<f32>,
    pub beta: DeviceBuffer<f32>,
    pub f: DeviceBuffer<f32>,
    pub z: DeviceBuffer<f32>,
    pub decay: DeviceBuffer<f32>,
    pub o: DeviceBuffer<f32>,
    pub gated: DeviceBuffer<f32>,
    // A latent mixer's.
    pub stack: DeviceBuffer<f32>,
    pub qr: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub qabs: DeviceBuffer<f32>,
    pub att: DeviceBuffer<f32>,
    pub av: DeviceBuffer<f32>,
    pub part_v: DeviceBuffer<f32>,
    pub part_ms: DeviceBuffer<f32>,
    /// Each head's sink logit: −∞, a fold of nothing.
    pub sinks: DeviceBuffer<f32>,
    // The k-pool selector's: the step's live count, the indexer query and
    // head weights, the scores of the pools, and the list the attention
    // reads (its length the second word of `vis`).
    pub cnt: DeviceBuffer<u32>,
    pub qi: DeviceBuffer<f32>,
    pub wi: DeviceBuffer<f32>,
    pub scores: DeviceBuffer<f32>,
    pub list: DeviceBuffer<u32>,
    // The feed-forward blocks'.
    pub h: DeviceBuffer<f32>,
    pub sh_y: DeviceBuffer<f32>,
    pub rout: RouterOut,
    pub sel: DeviceBuffer<u32>,
    /// A layer without a selection bias in the file selects with none.
    pub no_bias: DeviceBuffer<f32>,
    // The step's words: its position; the attention's visible counts (no
    // window row, then the selector's list length); the delta rule's lane, 0.
    pub pos: DeviceBuffer<u32>,
    pub vis: DeviceBuffer<u32>,
    pub lane: DeviceBuffer<u32>,
}

impl Scratch {
    /// The step's buffers; the low-rank halves are `kda.head_dim` wide, as
    /// the header reader holds every KDA tensor to ik's exact dims.
    fn new(stream: &CudaStream, d: &Dims, ff: usize, ctx: usize) -> Result<Scratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let (n, c, v) = (d.embd, d.kda.channels(), d.kda.n_v * HEAD);
        let rows = d.heads;
        let segs = attn::segments(0, list_width(d.kept));
        Ok(Scratch {
            streams: [z(HC_STREAMS * n)?, z(HC_STREAMS * n)?],
            x: z(n)?,
            xn: z(n)?,
            out: z(n)?,
            fold: z(n)?,
            mixes: z(HC_MIX)?,
            hc: z(HC_MIX)?,
            hc_scratch: HcPreScratch::with_groups(stream, HC_STREAMS * n, 1)?,
            qkv: z(c)?,
            conv: z(c)?,
            fa: z(HEAD)?,
            ga: z(HEAD)?,
            beta_raw: z(d.kda.n_v)?,
            beta: z(d.kda.n_v)?,
            f: z(v)?,
            z: z(v)?,
            decay: z(v)?,
            o: z(v)?,
            gated: z(v)?,
            stack: z(d.stack())?,
            qr: z(d.q_lora)?,
            q: z(d.heads * d.head_k)?,
            qabs: z(rows * LATENT)?,
            att: z(rows * LATENT)?,
            av: z(d.heads * d.head_v)?,
            part_v: z(attn::partials_v_len(rows, segs))?,
            part_ms: z(attn::partials_ms_len(rows, segs))?,
            sinks: DeviceBuffer::from_host(stream, &vec![f32::NEG_INFINITY; d.heads])?,
            cnt: DeviceBuffer::zeroed(stream, 1)?,
            qi: z(kpool::HEADS * kpool::DIM)?,
            wi: z(kpool::HEADS)?,
            scores: z(pools_for(ctx))?,
            list: DeviceBuffer::zeroed(stream, list_width(d.kept))?,
            h: z(ff)?,
            sh_y: z(n)?,
            rout: RouterOut::new(stream)?,
            sel: DeviceBuffer::zeroed(stream, N_USED)?,
            no_bias: z(N_EXPERT)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            vis: DeviceBuffer::zeroed(stream, 2)?,
            lane: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    fn bytes(&self) -> usize {
        let f = [
            &self.streams[0],
            &self.streams[1],
            &self.x,
            &self.xn,
            &self.out,
            &self.fold,
            &self.mixes,
            &self.hc,
            &self.qkv,
            &self.conv,
            &self.fa,
            &self.ga,
            &self.beta_raw,
            &self.beta,
            &self.f,
            &self.z,
            &self.decay,
            &self.o,
            &self.gated,
            &self.stack,
            &self.qr,
            &self.q,
            &self.qabs,
            &self.att,
            &self.av,
            &self.part_v,
            &self.part_ms,
            &self.sinks,
            &self.qi,
            &self.wi,
            &self.scores,
            &self.h,
            &self.sh_y,
            &self.no_bias,
            &self.rout.logits,
            &self.rout.probs,
            &self.rout.weights,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.hc_scratch.device_bytes()
            + self.rout.ids.num_bytes()
            + self.sel.num_bytes()
            + self.pos.num_bytes()
            + self.vis.num_bytes()
            + self.cnt.num_bytes()
            + self.list.num_bytes()
            + self.lane.num_bytes()
    }
}

/// The token embedding as the host reads it: the file's q8_0 rows, one
/// dequantized per step and written four times, one copy a stream.
struct Embedding {
    file: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    row_bytes: usize,
    n_vocab: usize,
    row: Vec<f32>,
    streams: Vec<f32>,
}

impl Embedding {
    fn new(file: Arc<Split>, n_embd: usize) -> Result<Embedding, GpuError> {
        let name = names::token_embd();
        let refuse = |need: &'static str| GpuError::Tensor {
            what: WHAT,
            name: name.clone(),
            need,
        };
        let (shard, info) = file
            .find(&name)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(|| refuse("in the file"))?;
        if info.ty != GgmlType::Q8_0 || info.dims.first() != Some(&(n_embd as u64)) {
            return Err(refuse("q8_0 rows of embedding_length values"));
        }
        let n_vocab = info.dims.get(1).copied().unwrap_or(0) as usize;
        let row_bytes = n_embd / 32 * 34;
        if n_vocab == 0 || info.nbytes != (n_vocab * row_bytes) as u64 {
            return Err(refuse("a whole number of q8_0 rows"));
        }
        Ok(Embedding {
            file,
            shard,
            info,
            row_bytes,
            n_vocab,
            row: vec![0.0; n_embd],
            streams: vec![0.0; HC_STREAMS * n_embd],
        })
    }

    /// Row `token`, dequantized and repeated into the four streams; a token
    /// past the vocabulary is refused by name.
    fn fill(&mut self, token: u32) -> Result<(), GpuError> {
        let t = token as usize;
        if t >= self.n_vocab {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("token {token} is past the {} embedding rows", self.n_vocab),
            });
        }
        let data = self
            .file
            .shard(self.shard)
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "the embedding's shard",
            })?
            .data(&self.info)?;
        let src = &data[t * self.row_bytes..][..self.row_bytes];
        dequant_row(GgmlType::Q8_0, src, &mut self.row).map_err(model::ModelError::from)?;
        for s in self.streams.chunks_exact_mut(self.row.len()) {
            s.copy_from_slice(&self.row);
        }
        Ok(())
    }
}

/// One step's host values: its position.
#[derive(Clone, Copy, Debug)]
pub struct StepInput {
    pos: u32,
}

/// The GLM body. Field order is drop order: the host tier (its lock over
/// the host set, its page windows) before the buffers.
pub struct Body {
    hybrid: Hybrid<GlmHost>,
    layers: Range<usize>,
    cfg: Vec<LayerCfg>,
    /// Each layer's tensor names, made at load.
    names: Vec<LayerNames>,
    dims: Dims,
    k: Kernels,
    s: Scratch,
    stores: Vec<Store>,
    /// The slot map's card copy, the host run's rows.
    slots: DeviceTensor<u32>,
    /// The routed experts the slot map puts on the card.
    card: CardExperts,
    embd: Embedding,
    /// Each layer's streams after it, when a gate armed them.
    taps: Option<Vec<DeviceBuffer<f32>>>,
    /// Positions the stores hold: a step's position counts once its inputs
    /// are on the card, before its chain is enqueued or replayed. The next
    /// position is the model's (`GpuModel::pos`); the two differ only after
    /// a step failed past that point, and the next step is then refused.
    held: u32,
    /// A failure a gate planted for the next step ([`Body::plant`]).
    plant: Option<Plant>,
    /// Positions every store holds.
    ctx: usize,
    /// The most positions a latent layer attends whole
    /// (`place::dense_positions`): a prompt chunk within them attends every
    /// position, one past them through the selector.
    dense: usize,
    /// The KDA layers' stores on the host at chosen positions.
    ckpt: Checkpoints,
    /// How a prompt is fed, and the batch feed's buffers.
    prompt: prefill::PromptState,
}

/// Every KDA layer's state and conv ring, in layer order: the list the
/// checkpoints copy.
fn copied(stores: &mut [Store]) -> Vec<&mut DeviceBuffer<f32>> {
    stores
        .iter_mut()
        .filter_map(Store::copied)
        .flatten()
        .collect()
}

/// The parts of the body a walk writes, lent apart from its host tier.
pub(crate) struct Parts<'s> {
    pub k: &'s Kernels,
    pub d: &'s Dims,
    pub cfg: &'s [LayerCfg],
    pub names: &'s [LayerNames],
    pub s: &'s mut Scratch,
    pub stores: &'s mut [Store],
    pub slots: &'s DeviceTensor<u32>,
    pub card: &'s mut CardExperts,
    pub taps: Option<&'s mut [DeviceBuffer<f32>]>,
}

impl Parts<'_> {
    /// Layer `l`'s streams, `cur`, copied into its tap when a gate armed
    /// them.
    pub(crate) fn tap(&mut self, gpu: &Gpu, l: usize, cur: usize) -> Result<(), GpuError> {
        if let Some(taps) = self.taps.as_deref_mut() {
            let tap = taps.get_mut(l).ok_or(GpuError::State {
                what: WHAT,
                missing: "the layer's tap",
            })?;
            tap.copy_from_device_async(&self.s.streams[cur], gpu.stream())?;
        }
        Ok(())
    }
}

/// The resident weight `name`.
pub(crate) fn weight<'w>(w: &'w Weights, name: &str) -> Result<&'w DevWeight, GpuError> {
    w.get(name).ok_or_else(|| GpuError::Tensor {
        what: WHAT,
        name: name.to_string(),
        need: "a resident weight",
    })
}

/// The resident q8_0 weight `name`'s two planes.
pub(crate) fn q8<'w>(
    w: &'w Weights,
    name: &str,
) -> Result<(&'w DeviceTensor<u32>, &'w DeviceTensor<u16>), GpuError> {
    match weight(w, name)? {
        DevWeight::Q8_0 { qs, d, .. } => Ok((qs, d)),
        _ => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: "a q8_0 weight",
        }),
    }
}

/// The resident f32 tensor `name`.
pub(crate) fn f32t<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<f32>, GpuError> {
    match weight(w, name)? {
        DevWeight::F32 { w, .. } => Ok(w),
        _ => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: "an f32 tensor",
        }),
    }
}

/// The resident f32 vector `name`, as the buffer a kernel reads.
pub(crate) fn f32v<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceBuffer<f32>, GpuError> {
    f32t(w, name).map(DeviceTensor::buf)
}

/// `y = W · x` for the q8_0 weight `name` at one column.
pub(crate) fn gemv(
    gpu: &Gpu,
    w: &Weights,
    name: &str,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (qs, d) = q8(w, name)?;
    gpu.q8f32().enqueue_q8_0_gemv(gpu.stream(), qs, d, x, 1, y)
}

/// A shape the kernels have no geometry for, by name.
fn shape(detail: String) -> GpuError {
    GpuError::Shape { what: WHAT, detail }
}

/// The layer's SwiGLU limit, 0 (the plain combine) when it has none.
fn limit_of(act: Act) -> f32 {
    let Act::SwiGlu { limit } = act;
    limit.unwrap_or(0.0)
}

impl Body {
    /// Card `card` of `plan`, which `inputs` made, resident: the plan's card
    /// segments, the projections that read one input joined into one row
    /// stream each ([`Body::derive`]), and the body over them with the host
    /// tier over the file's routed layers, holding the load's host set as
    /// `host` asks. Refused by name: a plan with an expert tier card (before
    /// anything uploads), a plan of more than one card, a layer kind or a
    /// width no kernel here runs, routed layers that are not one run.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Glm5nextModel, GpuError> {
        refuse_expert_tiers(WHAT, plan.machine)?;
        let kinds: Vec<Layer> = inputs.spec.layers.iter().map(Layer::of).collect();
        GpuModel::load_placed(
            file,
            plan,
            card,
            host,
            |stream, _, layers, w| Body::derive(stream, &kinds, layers, w),
            |gpu, file, w, residency| {
                Body::load_placed(gpu, file, w, plan, inputs, card, host, residency)
            },
        )
    }

    /// The joins: a KDA layer's q, k and v projections into one row stream
    /// and its three conv tap sets into one, in the conv's channel order; a
    /// latent layer's four projections of the normed input. Each row keeps
    /// its file bits.
    fn derive(
        stream: &CudaStream,
        kinds: &[Layer],
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError> {
        for l in layers {
            let kind = kinds
                .get(l)
                .ok_or_else(|| shape(format!("layer {l} past the description")))?;
            match kind.mixer {
                MixerKind::DeltaRule => {
                    let (q, k, v) = (names::attn_q(l), names::attn_k(l), names::attn_v(l));
                    w.join_rows(stream, &[&q, &k, &v], names::attn_qkv(l))?;
                    let parts = ['q', 'k', 'v'].map(|p| names::ssm_conv1d(l, p));
                    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
                    w.join_rows(stream, &parts, names::ssm_conv1d_qkv(l))?;
                }
                MixerKind::Latent => {
                    let parts = [
                        names::attn_q_a(l),
                        names::attn_kv_a_mqa(l),
                        names::indexer_attn_k(l),
                        names::indexer_compressor_gate(l),
                    ];
                    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
                    w.join_rows(stream, &parts, names::attn_a_stack(l))?;
                }
                MixerKind::Gqa => {
                    return Err(shape(format!(
                        "layer {l}: a GQA mixer, which glm5next has none of"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The body of card `card`: its layers' programs and values from the
    /// plan's description, the stores at the plan's `ctx_max`, the step's
    /// buffers, the slot map the plan's routed segments make, the card
    /// experts it puts on the card and the host tier over the routed run.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card, file, weights, plan and inputs, and the host tier's residency and levers (rust-quality R8)"
    )]
    fn load_placed(
        gpu: &Gpu,
        file: Split,
        w: &Weights,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: HostResidency,
    ) -> Result<Body, GpuError> {
        let hp = &inputs.hp;
        let spec = &inputs.spec;
        let layers = plan
            .machine
            .cards
            .get(card)
            .map(|c| c.layers.clone())
            .ok_or_else(|| shape(format!("the plan has no card {card}")))?;
        if plan.machine.cards.len() != 1 || layers != (0..spec.layers.len()) {
            return Err(shape(format!(
                "card {card} runs layers {layers:?} of a plan over {} cards; the program runs \
                 every layer of the trunk (0..{}) on one card",
                plan.machine.cards.len(),
                spec.layers.len()
            )));
        }
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0 && c as u64 <= place::ORACLE_POSITIONS)
            .ok_or_else(|| {
                shape(format!(
                    "ctx_max {}: at least 1 and at most the {} positions a reference set \
                     checks the selector at",
                    plan.ctx_max,
                    place::ORACLE_POSITIONS
                ))
            })?;
        let dense = usize::try_from(place::dense_positions(hp))
            .map_err(|_| shape(format!("dense positions {}", place::dense_positions(hp))))?;
        let run = hosted(&spec.layers).map_err(|e| shape(e.to_string()))?;
        let dims = dims_of(inputs)?;
        let map = SlotMap::of_plan(plan, card, None, run.clone(), N_EXPERT)?;
        let cfg = spec
            .layers
            .iter()
            .enumerate()
            .map(|(l, s)| layer_cfg(l, s, &map))
            .collect::<Result<Vec<_>, _>>()?;
        let names = cfg
            .iter()
            .enumerate()
            .map(|(l, c)| LayerNames::of(l, c.kind))
            .collect::<Result<Vec<_>, _>>()?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let ff = cfg.iter().map(|c| c.ff).max().unwrap_or(0);
        let s = Scratch::new(stream, &dims, ff, ctx)?;
        let stores = cfg
            .iter()
            .map(|c| store(stream, c.kind, &dims, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let slots = DeviceTensor::upload(stream, &map.stage_view(), run.len(), N_EXPERT)?;
        let experts_on_card =
            CardExperts::new(gpu, w, &spec.layers, &map, dims.embd, hp.expert_ff)?;
        let boundary = Boundary::with_rows(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: dims.embd,
                n_used: N_USED,
            },
            1,
        )?;
        let file = Arc::new(file);
        let experts = GlmHost::build(Arc::clone(&file), hp, run.clone(), host.r8)?;
        let mut hybrid = Hybrid::new(boundary, map, experts, run.len())?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(residency);
        let embd = Embedding::new(file, dims.embd)?;
        if embd.n_vocab != hp.n_vocab {
            return Err(shape(format!(
                "the embedding has {} rows, the file's vocabulary {}",
                embd.n_vocab, hp.n_vocab
            )));
        }
        let lens: Vec<usize> = cfg
            .iter()
            .filter(|c| c.kind.mixer == MixerKind::DeltaRule)
            .flat_map(|_| [dims.kda.state_len(), dims.kda.ring_len()])
            .collect();
        let ckpt = Checkpoints::new(gpu.context(), lens, HOST_BUDGET, CHECKPOINT_EVERY)?;
        let lane = [0u32];
        let mut body = Body {
            hybrid,
            layers,
            cfg,
            names,
            dims,
            k: Kernels::load(gpu)?,
            s,
            stores,
            slots,
            card: experts_on_card,
            embd,
            taps: None,
            held: 0,
            plant: None,
            ctx,
            dense,
            ckpt,
            prompt: prefill::PromptState::new(),
        };
        body.s.lane.copy_from_host(stream, &lane)?;
        stream.synchronize()?;
        Ok(body)
    }

    /// Positions every store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// Each layer's programs, in layer order.
    #[must_use]
    pub fn kinds(&self) -> Vec<Layer> {
        self.cfg.iter().map(|c| c.kind).collect()
    }

    /// The layers the host tier serves.
    #[must_use]
    pub fn host_run(&self) -> Range<usize> {
        self.hybrid.slots().layers()
    }

    /// Device bytes of the layers' stores.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store::bytes).sum()
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<GlmHost> {
        &self.hybrid
    }

    /// The host tier, for a caller that attaches or marks its route trace.
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<GlmHost> {
        &mut self.hybrid
    }

    /// The slot map's card copy the handoff reads, a routed layer's row at
    /// the map's [`SlotMap::row_offset`].
    #[must_use]
    pub fn slot_copy(&self) -> &DeviceTensor<u32> {
        &self.slots
    }

    /// Arm (or disarm) the per-layer taps: after each layer the chain copies
    /// its streams into the layer's tap, which [`Body::taps`] reads back.
    /// Only through [`set_taps`], which drops the captured chains first: a
    /// capture holds the tap copies it was taken with.
    pub(crate) fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        self.taps = if on {
            let n = HC_STREAMS * self.dims.embd;
            Some(
                self.cfg
                    .iter()
                    .map(|_| DeviceBuffer::zeroed(gpu.stream(), n))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        Ok(())
    }

    /// Every layer's streams after the last step, `4 · n_embd` a layer:
    /// stream `s` of layer `l` at `l · 4n + s · n`. Blocking; refused when
    /// the taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let taps = self.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps (Body::set_taps)",
        })?;
        let mut out = Vec::with_capacity(taps.len() * HC_STREAMS * self.dims.embd);
        for t in taps {
            out.extend(t.to_host_vec(gpu.stream())?);
        }
        Ok(out)
    }

    /// The walk's parts, lent apart from the host tier.
    pub(crate) fn parts(&mut self) -> (Parts<'_>, &mut Hybrid<GlmHost>) {
        (
            Parts {
                k: &self.k,
                d: &self.dims,
                cfg: &self.cfg,
                names: &self.names,
                s: &mut self.s,
                stores: &mut self.stores,
                slots: &self.slots,
                card: &mut self.card,
                taps: self.taps.as_deref_mut(),
            },
            &mut self.hybrid,
        )
    }

    /// The longest prefix of at most `n` positions a cut of a model standing
    /// at `pos` keeps: every position, the empty model, or the nearest
    /// checkpoint at or below `n` ([`Body::kept`] says which).
    #[must_use]
    pub fn keep_point(&self, n: u32, pos: u32) -> u32 {
        self.kept(n, pos).at
    }

    /// What a cut to at most `n` positions of a model standing at `pos`
    /// keeps, and why: never past `pos`, and after a step that failed past
    /// its launch not `pos` either, since the stores hold one more.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        if self.held == pos {
            self.ckpt.kept(n, pos)
        } else {
            self.ckpt.kept(n.min(pos), self.held)
        }
    }

    /// The checkpoints: their positions, their slots, what they have done.
    #[must_use]
    pub fn checkpoints(&self) -> &Checkpoints {
        &self.ckpt
    }

    /// A checkpoint at `pos`, the model's position, after any waiting cut:
    /// the KDA layers' stores copied to a host slot, or nothing where one
    /// stands. Refused by name when the stores hold other positions (a step
    /// failed past its launch). Waits for the copies.
    pub fn checkpoint(&mut self, gpu: &Gpu, pos: u32) -> Result<Take, GpuError> {
        self.stores_at(pos)?;
        self.ckpt
            .take(gpu.stream(), pos, &mut copied(&mut self.stores))
    }

    /// Refused by name unless the stores hold `pos` positions: a step at
    /// `pos` that failed after its inputs were on the card has already run
    /// the KDA layers' recurrence over it, and running it again would apply
    /// the position twice.
    fn stores_at(&self, pos: u32) -> Result<(), GpuError> {
        match self.held {
            h if h == pos => Ok(()),
            h if h == pos.wrapping_add(1) => Err(shape(format!(
                "position {pos}: the step there failed after its chain was launched, so the \
                 recurrent stores hold it already; cut to a checkpoint (Body::kept) or reset"
            ))),
            h => Err(shape(format!(
                "position {pos}, where the stores hold {h} positions"
            ))),
        }
    }

    /// Plant `plant` for the next step: a gate's way to fail a step on
    /// either side of its launch without a fault ([`Plant`]).
    pub fn plant(&mut self, plant: Plant) {
        self.plant = Some(plant);
    }

    /// The planted failure at `at`, taken, as the step's error.
    fn planted(&mut self, at: Plant) -> Result<(), GpuError> {
        if self.plant == Some(at) {
            self.plant = None;
            return Err(GpuError::State {
                what: WHAT,
                missing: match at {
                    Plant::BeforeLaunch => "the planted failure before the launch",
                    Plant::AfterLaunch => "the planted failure after the launch",
                },
            });
        }
        Ok(())
    }
}

/// Where a gate plants a step's failure ([`Body::plant`]): in its refresh,
/// before any input reaches the card, or once its chain has run and its
/// host legs been served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plant {
    BeforeLaunch,
    AfterLaunch,
}

/// Arm (or disarm) `m`'s per-layer taps ([`Body::taps`] reads them back),
/// the model first set to eager steps: a captured chain holds the tap
/// copies it was taken with, so arming under one would leave the taps
/// unwritten and disarming would free buffers its replays write.
pub fn set_taps(m: &mut Glm5nextModel, on: bool) -> Result<(), GpuError> {
    m.set_mode(StepMode::Eager);
    let (gpu, _, body) = m.body_parts("glm5next set_taps")?;
    body.set_taps(gpu, on)
}

/// Feed `ids` from where `m` stands, one step a position, and take the
/// checkpoints the call's marks name ([`Checkpoints::marks`]): its start, the
/// multiples of [`CHECKPOINT_EVERY`] inside it, its end. The argmax after
/// the last id. Refused on a model a fault poisoned: no checkpoint copies
/// what a fault condemned.
pub fn prompt(m: &mut Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    const WHAT: &str = "glm5next prompt";
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let from = m.pos();
    let to = u32::try_from(ids.len())
        .ok()
        .and_then(|n| from.checked_add(n))
        .filter(|&to| to > from)
        .ok_or_else(|| shape(format!("a prompt of {} ids from {from}", ids.len())))?;
    let marks = m.body(WHAT)?.ckpt.marks(from, to);
    let mut argmax = None;
    let mut at = from;
    for mark in marks {
        if mark > at {
            let seg = &ids[(at - from) as usize..(mark - from) as usize];
            argmax = Some(m.step(seg)?);
            at = mark;
        }
        let pos = m.pos();
        let (gpu, _, body) = m.body_parts(WHAT)?;
        body.checkpoint(gpu, pos)?;
    }
    argmax.ok_or(GpuError::State {
        what: WHAT,
        missing: "a mark at the call's end",
    })
}

/// The file's widths and constants, each checked against the kernel that
/// fixes it.
fn dims_of(inputs: &PlanInputs) -> Result<Dims, GpuError> {
    let hp = &inputs.hp;
    let heads = hp.n_head;
    let checks = [
        ("hyper_connection.count", hp.hc.streams, HC_STREAMS),
        ("kda.head_dim", hp.kda_head_dim, HEAD),
        ("attention.kv_lora_rank", hp.kv_lora, LATENT),
        (
            "attention.kv_lora_rank (the attention's)",
            hp.kv_lora,
            attn::LATENT,
        ),
        (
            "attention.indexer.key_length (the index row's half)",
            2 * hp.indexer.head_dim,
            INDEX_ROW,
        ),
        (
            "attention.indexer.key_length (the pool key's)",
            hp.indexer.head_dim,
            INDEX_HEAD,
        ),
        (
            "attention.indexer.head_count",
            hp.indexer.n_head,
            kpool::HEADS,
        ),
        ("attention.indexer.kpool", hp.indexer.kpool, POOL),
        ("expert_count", hp.n_expert, N_EXPERT),
        ("expert_used_count", hp.n_used, N_USED),
        ("ssm.conv_kernel", hp.conv, bloomery_gpu::linear::CONV_TAPS),
    ];
    for (key, got, want) in checks {
        if got != want {
            return Err(shape(format!(
                "{key} is {got}; the kernels are built for {want}"
            )));
        }
    }
    if !(HC_STREAMS * hp.n_embd).is_multiple_of(HC_PIECE) {
        return Err(shape(format!(
            "the streams' {} values are not whole hc_pre pieces of {HC_PIECE}",
            HC_STREAMS * hp.n_embd
        )));
    }
    Ok(Dims {
        embd: hp.n_embd,
        heads,
        kda: LinearShape {
            n_k: heads,
            n_v: heads,
            map: KHeadMap::Tiled,
        },
        q_lora: hp.q_lora,
        head_k: hp.head_k,
        head_v: hp.head_v,
        index_d: hp.indexer.head_dim,
        rms_eps: hp.rms_eps,
        norm_eps: hp.norm_eps,
        kept: hp.indexer.top_k / hp.indexer.kpool,
        hc_eps: hp.hc.eps,
        hc_iters: u32::try_from(hp.hc.sinkhorn)
            .map_err(|_| shape(format!("sinkhorn_iterations {}", hp.hc.sinkhorn)))?,
        lb: hp.gate_lower_bound,
        scale: hp.weights_scale,
    })
}

/// Layer `l`'s programs from its description and the values its launches
/// take; a routed layer's row in the card copy is `map`'s
/// ([`SlotMap::row_offset`]).
fn layer_cfg(l: usize, s: &LayerSpec, map: &SlotMap) -> Result<LayerCfg, GpuError> {
    let kind = Layer::of(s);
    if kind.residual != ResidualKind::Hc {
        return Err(shape(format!(
            "layer {l}: a plain residual; every trunk block is wrapped in the streams"
        )));
    }
    if let Mixer::Gqa(_) = s.mixer {
        return Err(shape(format!(
            "layer {l}: a GQA mixer, which glm5next has none of"
        )));
    }
    let (limit, ff, bias) = match &s.ffn {
        Ffn::Dense { ff, act } => (limit_of(*act), *ff as usize, false),
        Ffn::Moe(m) => {
            let r = &m.router;
            if r.score != Score::Sigmoid || !r.norm || r.hash {
                return Err(shape(format!(
                    "layer {l}: a router scoring by {:?}, renormalizing {}, hashed {}; the \
                     routed block runs the sigmoid router that renormalizes its picks, unhashed",
                    r.score, r.norm, r.hash
                )));
            }
            let sh = m.shared.ok_or_else(|| {
                shape(format!(
                    "layer {l}: a routed block without its shared expert"
                ))
            })?;
            (limit_of(sh.act), sh.ff as usize, m.router.bias)
        }
    };
    let row_off = match kind.ffn {
        FfnKind::Moe => map
            .row_offset(l)
            .ok_or_else(|| shape(format!("layer {l}: a routed layer with no slot-map row")))?,
        FfnKind::Dense => 0,
    };
    Ok(LayerCfg {
        kind,
        limit,
        ff,
        bias,
        row_off,
    })
}

/// Layer `kind`'s store at `ctx` positions: a KDA layer's one-lane state and
/// its conv ring, a latent layer's latent and index rows and its pool plane.
fn store(stream: &CudaStream, kind: Layer, d: &Dims, ctx: usize) -> Result<Store, GpuError> {
    Ok(match kind.mixer {
        MixerKind::DeltaRule => Store::Kda {
            state: DeviceBuffer::zeroed(stream, d.kda.state_len())?,
            ring: DeviceBuffer::zeroed(stream, d.kda.ring_len())?,
        },
        MixerKind::Latent => Store::Latent {
            latent: DeviceTensor::zeroed(stream, ctx, LATENT)?,
            index: DeviceTensor::zeroed(stream, ctx, INDEX_ROW)?,
            pooled: DeviceTensor::zeroed(stream, pools_for(ctx), INDEX_HEAD)?,
        },
        MixerKind::Gqa => return Err(shape("a GQA store, which glm5next has none of".into())),
    })
}

impl ChainBody for Body {
    type Input = StepInput;
    type Host = Body;

    fn arch() -> Arch {
        Arch::Glm5next
    }

    /// The step at `pos`: its embedding row read from the file. A position
    /// past the stores, or one the stores do not stand at, is refused by
    /// name ([`Body::stores_at`]).
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(shape(format!(
                "a step at position {pos} in stores of {}",
                self.ctx
            )));
        }
        self.stores_at(pos)?;
        self.embd.fill(token)?;
        Ok(StepInput { pos })
    }

    /// The four stream copies of the row into the first stream buffer, the
    /// position, the visible counts and the live count: four host-to-device
    /// copies. Once
    /// they are sent the stores count the position: what follows launches.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        self.planted(Plant::BeforeLaunch)?;
        if self.ckpt.pending() {
            self.ckpt.apply(stream, &mut copied(&mut self.stores))?;
        }
        let p = input.pos;
        self.s.streams[0].copy_from_host(stream, &self.embd.streams)?;
        self.s.pos.copy_from_host(stream, &[p])?;
        self.s.vis.copy_from_host(stream, &[0, p + 1])?;
        self.s.cnt.copy_from_host(stream, &[p + 1])?;
        self.held = p + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let (parts, hybrid) = self.parts();
        program::walk_step(gpu, w, parts, hybrid, head)?;
        self.planted(Plant::AfterLaunch)
    }

    /// Every store and both stream buffers zeroed in place — a captured chain
    /// keeps their addresses — after the host tier's reset; every checkpoint
    /// dropped.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        self.hybrid.reset(stream)?;
        for s in &mut self.stores {
            s.zero(stream)?;
        }
        for s in &mut self.s.streams {
            s.zero_async(stream)?;
        }
        self.held = 0;
        self.plant = None;
        self.ckpt.clear();
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every RMS norm's, the head's
    /// included.
    fn head_eps(&self) -> f32 {
        self.dims.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.s.bytes()
            + self.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
            + self.card.bytes()
            + self.prompt.bytes()
            + self
                .taps
                .as_ref()
                .map_or(0, |t| t.iter().map(DeviceBuffer::num_bytes).sum())
    }

    fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    fn host(&mut self) -> Option<&mut Body> {
        Some(self)
    }
}

impl Rollback for Body {
    /// A cut to `pos`: nothing at the fed position; the empty model at 0;
    /// else the checkpoint at `pos`, copied back at the next step. Any other
    /// position is refused by name ([`Body::kept`] says what a cut keeps).
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.ckpt.cut(pos, self.held)?;
        self.held = pos;
        Ok(())
    }
}

impl HostServed for Body {
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)?;
        self.planted(Plant::AfterLaunch)
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.take_step_refusal()
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid.residency()
    }
}
