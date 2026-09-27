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
//! The routed experts run on the host tier: a routed layer's front ends at
//! the go, its shadow is the shared expert, its back the wait, the sum and
//! `hc_post` ([`crate::ffn`]). The step walks the layers with the runtime's
//! one-token walk ([`crate::program`]).
//!
//! The token's embedding row is read from the file on the host and written
//! into the first stream buffer four times, one copy before the chain; its
//! position and the attention's visible counts are two more words the
//! captured chain reads. Each latent layer caches every position it has
//! seen and attends all of them: the plan refuses a context past the
//! positions the indexer keeps whole (`place::dense_positions`), so the
//! selector never has to run. The next-token (MTP) layer is carried by the
//! file and not loaded: its tensors are `Role::Unused`, as ik loads and does
//! not run it.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::{
    Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap,
};
use bloomery_gpu::latent::{INDEX_ROW, LATENT, LatentKernels};
use bloomery_gpu::linear::{HEAD, KHeadMap, LinearKernels, LinearShape, PASS_ROWS};
use bloomery_gpu::model::{ChainBody, HostServed, StepKernels};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
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
use models::{Act, Ffn, LayerSpec, Mixer};
use runtime::layer::{FfnKind, Layer, MixerKind, ResidualKind, hosted};

use crate::host::GlmHost;
use crate::program;

/// What the body's errors name.
const WHAT: &str = "glm5next Body";

// The plan sizes each KDA layer's conv ring by its own pass width.
const _: () = assert!(place::PASS_ROWS == PASS_ROWS);

/// The GLM model: one card, the skeleton over this body.
pub type Glm5nextModel = GpuModel<Body>;

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
    /// The layer's row in the slot map's card copy, `n_expert` places a row.
    pub row_off: usize,
}

/// A layer's store: a KDA layer's recurrent state and conv ring, a latent
/// layer's latent rows and index rows, one row a position.
pub(crate) enum Store {
    Kda {
        state: DeviceBuffer<f32>,
        ring: DeviceBuffer<f32>,
    },
    Latent {
        latent: DeviceTensor<u16>,
        index: DeviceTensor<u16>,
    },
}

impl Store {
    fn bytes(&self) -> usize {
        match self {
            Store::Kda { state, ring } => state.num_bytes() + ring.num_bytes(),
            Store::Latent { latent, index } => latent.buf().num_bytes() + index.buf().num_bytes(),
        }
    }

    fn zero(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        match self {
            Store::Kda { state, ring } => {
                state.zero_async(stream)?;
                ring.zero_async(stream)?;
            }
            Store::Latent { latent, index } => {
                latent.buf_mut().zero_async(stream)?;
                index.buf_mut().zero_async(stream)?;
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
    // The feed-forward blocks'.
    pub h: DeviceBuffer<f32>,
    pub sh_y: DeviceBuffer<f32>,
    pub rout: RouterOut,
    pub sel: DeviceBuffer<u32>,
    /// A layer without a selection bias in the file selects with none.
    pub no_bias: DeviceBuffer<f32>,
    // The step's words: its position; the attention's visible counts (no
    // window row, then every cached position); the delta rule's lane, 0.
    pub pos: DeviceBuffer<u32>,
    pub vis: DeviceBuffer<u32>,
    pub lane: DeviceBuffer<u32>,
}

impl Scratch {
    fn new(
        stream: &CudaStream,
        d: &Dims,
        ff: usize,
        ranks: [usize; 2],
        ctx: usize,
    ) -> Result<Scratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let (n, c, v) = (d.embd, d.kda.channels(), d.kda.n_v * HEAD);
        let rows = d.heads;
        let segs = attn::segments(0, ctx);
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
            fa: z(ranks[0])?,
            ga: z(ranks[1])?,
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
            &self.h,
            &self.sh_y,
            &self.no_bias,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.hc_scratch.device_bytes()
            + self.sel.num_bytes()
            + self.pos.num_bytes()
            + self.vis.num_bytes()
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
    dims: Dims,
    k: Kernels,
    s: Scratch,
    stores: Vec<Store>,
    /// The slot map's card copy, the host run's rows.
    slots: DeviceTensor<u32>,
    embd: Embedding,
    /// Each layer's streams after it, when a gate armed them.
    taps: Option<Vec<DeviceBuffer<f32>>>,
    /// Positions fed since the load or the last reset.
    fed: u32,
    /// Positions every store holds.
    ctx: usize,
}

/// The parts of the body a walk writes, lent apart from its host tier.
pub(crate) struct Parts<'s> {
    pub k: &'s Kernels,
    pub d: &'s Dims,
    pub cfg: &'s [LayerCfg],
    pub s: &'s mut Scratch,
    pub stores: &'s mut [Store],
    pub slots: &'s DeviceTensor<u32>,
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
    /// `host` asks. Refused by name: a plan of more than one card, a layer
    /// kind or a width no kernel here runs, routed layers that are not one
    /// run.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Glm5nextModel, GpuError> {
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
    /// buffers, the slot map (every routed expert on the host) and the host
    /// tier over the routed run.
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
            .filter(|&c| c > 0 && c as u64 <= place::dense_positions(hp))
            .ok_or_else(|| {
                shape(format!(
                    "ctx_max {}: at least 1 and at most the {} positions the latent layers \
                     attend whole",
                    plan.ctx_max,
                    place::dense_positions(hp)
                ))
            })?;
        let run = hosted(&spec.layers).map_err(|e| shape(e.to_string()))?;
        let dims = dims_of(inputs)?;
        let cfg = spec
            .layers
            .iter()
            .enumerate()
            .map(|(l, s)| layer_cfg(l, s, &run, hp.n_expert))
            .collect::<Result<Vec<_>, _>>()?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let ff = cfg.iter().map(|c| c.ff).max().unwrap_or(0);
        let ranks = low_ranks(w, &cfg)?;
        let s = Scratch::new(stream, &dims, ff, ranks, ctx)?;
        let stores = cfg
            .iter()
            .map(|c| store(stream, c.kind, &dims, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let map = SlotMap::prefix(run.clone(), N_EXPERT, 0)?;
        let slots = DeviceTensor::upload(stream, map.as_slice(), run.len(), N_EXPERT)?;
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
        let lane = [0u32];
        let mut body = Body {
            hybrid,
            layers,
            cfg,
            dims,
            k: Kernels::load(gpu)?,
            s,
            stores,
            slots,
            embd,
            taps: None,
            fed: 0,
            ctx,
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

    /// Arm (or disarm) the per-layer taps: after each layer the chain copies
    /// its streams into the layer's tap, which [`Body::taps`] reads back. For
    /// an eager chain: a capture taken while they are armed records the
    /// copies.
    pub fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
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
                s: &mut self.s,
                stores: &mut self.stores,
                slots: &self.slots,
                taps: self.taps.as_deref_mut(),
            },
            &mut self.hybrid,
        )
    }

    /// Whether the positions from `n` on can be taken back: only the next
    /// one, since a KDA layer keeps one state and no history of it — or all
    /// of them, back to an empty model.
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        if n >= self.fed { self.fed } else { 0 }
    }
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
        hc_eps: hp.hc.eps,
        hc_iters: u32::try_from(hp.hc.sinkhorn)
            .map_err(|_| shape(format!("sinkhorn_iterations {}", hp.hc.sinkhorn)))?,
        lb: hp.gate_lower_bound,
        scale: hp.weights_scale,
    })
}

/// Layer `l`'s programs from its description and the values its launches
/// take; `run` is the host tier's layers.
fn layer_cfg(
    l: usize,
    s: &LayerSpec,
    run: &Range<usize>,
    n_expert: usize,
) -> Result<LayerCfg, GpuError> {
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
            let sh = m.shared.ok_or_else(|| {
                shape(format!(
                    "layer {l}: a routed block without its shared expert"
                ))
            })?;
            (limit_of(sh.act), sh.ff as usize, m.router.bias)
        }
    };
    let row_off = match kind.ffn {
        FfnKind::Moe => (l - run.start) * n_expert,
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

/// The low-rank projections' widths, `[ssm_f_a, ssm_g_a]` rows, the widest
/// over the KDA layers' resident weights (each layer's gemv checks its own
/// against the buffers at enqueue); 0 with no KDA layer.
fn low_ranks(w: &Weights, cfg: &[LayerCfg]) -> Result<[usize; 2], GpuError> {
    let mut ranks = [0usize; 2];
    for (l, c) in cfg.iter().enumerate() {
        if c.kind.mixer == MixerKind::DeltaRule {
            ranks[0] = ranks[0].max(q8(w, &names::ssm_f_a(l))?.1.rows());
            ranks[1] = ranks[1].max(q8(w, &names::ssm_g_a(l))?.1.rows());
        }
    }
    Ok(ranks)
}

/// Layer `kind`'s store at `ctx` positions: a KDA layer's one-lane state and
/// its conv ring, a latent layer's latent and index rows.
fn store(stream: &CudaStream, kind: Layer, d: &Dims, ctx: usize) -> Result<Store, GpuError> {
    Ok(match kind.mixer {
        MixerKind::DeltaRule => Store::Kda {
            state: DeviceBuffer::zeroed(stream, d.kda.state_len())?,
            ring: DeviceBuffer::zeroed(stream, d.kda.ring_len())?,
        },
        MixerKind::Latent => Store::Latent {
            latent: DeviceTensor::zeroed(stream, ctx, LATENT)?,
            index: DeviceTensor::zeroed(stream, ctx, INDEX_ROW)?,
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
    /// other than the next one, or past the stores, is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        if pos != self.fed || pos as usize >= self.ctx {
            return Err(shape(format!(
                "a step at position {pos} after {} positions, in stores of {}",
                self.fed, self.ctx
            )));
        }
        self.embd.fill(token)?;
        self.fed += 1;
        Ok(StepInput { pos })
    }

    /// The four stream copies of the row into the first stream buffer, the
    /// position and the visible counts: three host-to-device copies.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        let p = input.pos;
        self.s.streams[0].copy_from_host(stream, &self.embd.streams)?;
        self.s.pos.copy_from_host(stream, &[p])?;
        self.s.vis.copy_from_host(stream, &[0, p + 1])?;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let (parts, hybrid) = self.parts();
        program::walk_step(gpu, w, parts, hybrid, head)
    }

    /// Every store and both stream buffers zeroed in place — a captured chain
    /// keeps their addresses — after the host tier's reset.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        self.hybrid.reset(stream)?;
        for s in &mut self.stores {
            s.zero(stream)?;
        }
        for s in &mut self.s.streams {
            s.zero_async(stream)?;
        }
        self.fed = 0;
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
            + self.slots.buf().num_bytes()
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

impl HostServed for Body {
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.take_step_refusal()
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid.residency()
    }
}
