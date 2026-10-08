//! The MiMo-V2.6-Flash body: the load, the stores, the step's buffers and
//! [`Body`] behind `GpuModel`.
//!
//! All 48 layers are one card's chain: the dense layer 0 and 47 routed layers
//! whose experts all run on the host tier (the plan keeps none on a card).
//! Each layer is a GQA attention over a full plane of the stores' positions —
//! its window bounds what a query reads, not what the layer holds — and a
//! block; the step is [`crate::program`]'s walk, one token at a time (there is
//! no pass of several rows, no draft and no resident slot). The embedding is a
//! host row, dequantized per step and copied into `x` before the launch.
//!
//! What the load refuses, by name: a plan of other than one card with no
//! expert tier, a plan that puts a routed expert on a card, a description
//! whose layers the K192 kernels or the routed block do not run
//! ([`facts::attn_args`], [`facts::block_args`]), a resident weight of
//! other rows than the description's, a non-finite sink, stores whose bytes
//! are not the plan's, and a scratch past the plan's term.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::flash_gqa::{self, FlashGqaKernels, HEAD, HEAD_K192};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::handoff::HandoffKernels;
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::hybrid::{
    Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap,
};
use bloomery_gpu::model::{ChainBody, HostServed, StepMode};
use bloomery_gpu::rope_neox::{ROT_K192, RopeNeoxKernels};
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use bloomery_gpu_deepseek41::experts::ExpertKernels;
use bloomery_gpu_deepseek41::router::mimo2::{N_EXPERT, N_USED, RouterKernels, RouterOut};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{GgmlType, Split};
use model::arch::Arch;
use model::arch::mimo2::names;
use model::arch::mimo2::place::PlanInputs;
use model::arch::mimo2::program::{self as facts, AttnArgs, AttnBuilt, BlockArgs, RouterBuilt};
use model::embed::EmbedTable;
use model::placement::Plan;
use runtime::layer::{FfnKind, Layer, ResidualKind, hosted};

use crate::attn::{AttnNames, checked_sinks};
use crate::ffn::{FfnNames, routed_layers};
use crate::program;

/// What the body's errors name.
pub(crate) const WHAT: &str = "mimo2 Body";

/// The MiMo model: one card, the skeleton over this body.
pub type Mimo2Model = GpuModel<Body>;

/// The kernels the step launches besides `Gpu`'s own, loaded once.
pub(crate) struct Kernels {
    pub rope: RopeNeoxKernels,
    pub flash: FlashGqaKernels,
    pub experts: ExpertKernels,
    pub router: RouterKernels,
    pub handoff: HandoffKernels,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            rope: RopeNeoxKernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            handoff: HandoffKernels::load(ctx)?,
        })
    }
}

/// The file's widths and constants the launches take, read once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dims {
    pub embd: usize,
    /// Query heads, every layer's.
    pub heads: usize,
    pub rms_eps: f32,
    /// `expert_weights_scale`.
    pub scale: f32,
    /// The scores' scale, `1/√192`.
    pub score_scale: f32,
}

/// One layer's programs and the values its launches take.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerCfg {
    pub kind: Layer,
    pub attn: AttnArgs,
    /// The layer's rope table, an index into the body's tables.
    pub rope: usize,
    pub ffn: BlockArgs,
    /// The layer's row in the slot map's card copy, as a word offset
    /// ([`SlotMap::row_offset`]); 0 on a dense layer.
    pub row_off: usize,
}

/// A layer's tensor names, made at load.
pub(crate) struct LayerNames {
    pub attn: AttnNames,
    pub ffn: FfnNames,
}

/// A layer's cache: its key plane of `kv_heads · ctx` rows of 192 f16 and its
/// value plane of `kv_heads · ctx` rows of 128, as the f16 bits the append
/// writes.
pub(crate) struct Store {
    pub k: DeviceBuffer<u16>,
    pub v: DeviceBuffer<u16>,
}

impl Store {
    fn new(stream: &CudaStream, kv_heads: usize, ctx: usize) -> Result<Store, GpuError> {
        Ok(Store {
            k: DeviceBuffer::zeroed(stream, kv_heads * ctx * HEAD_K192)?,
            v: DeviceBuffer::zeroed(stream, kv_heads * ctx * HEAD)?,
        })
    }

    fn bytes(&self) -> usize {
        self.k.num_bytes() + self.v.num_bytes()
    }
}

/// One rope base's table on the card: `ROT_K192` f32 for each position below
/// the stores', the bits `RopeTable::push` makes.
pub(crate) struct RopeRows {
    pub theta: f32,
    pub table: DeviceBuffer<f32>,
}

impl RopeRows {
    /// Rows `0..ctx` of the table at base `theta`. Load-time only.
    fn new(stream: &CudaStream, theta: f32, ctx: usize) -> Result<RopeRows, GpuError> {
        let table = RopeTable::new(&RopeSpec::window(theta, ROT_K192))?;
        let mut host = Vec::with_capacity(ctx * ROT_K192);
        for pos in 0..ctx as u32 {
            table.push(pos, Direction::Forward, &mut host);
        }
        Ok(RopeRows {
            theta,
            table: DeviceBuffer::from_host(stream, &host)?,
        })
    }
}

/// Every buffer the step writes or reads besides the stores, one token's.
pub(crate) struct Scratch {
    /// The layer's input and output: the embedding writes it before the
    /// chain, and each layer's block writes it back (the last layer's output
    /// goes to the head's input instead).
    pub x: DeviceBuffer<f32>,
    /// The residual after the attention.
    pub x1: DeviceBuffer<f32>,
    /// A sub-layer's normed input, and its projection's output.
    pub xn: DeviceBuffer<f32>,
    pub out: DeviceBuffer<f32>,
    /// The fused q·k·v rows of the widest layer, the roped query heads, the
    /// flash's partials and its output rows.
    pub qkv: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub part_v: DeviceBuffer<f32>,
    pub part_ms: DeviceBuffer<f32>,
    pub y: DeviceBuffer<f32>,
    /// The dense block's gate·up·SwiGLU rows.
    pub h: DeviceBuffer<f32>,
    pub rout: RouterOut,
    pub sel: DeviceBuffer<u32>,
    /// The zero selection bias of a router the file gives none.
    pub no_bias: DeviceBuffer<f32>,
    /// The step's position, and its live key count (`pos + 1`).
    pub pos: DeviceBuffer<u32>,
    pub n_keys: DeviceBuffer<u32>,
}

impl Scratch {
    /// The buffers for a model of `d`, `qkv` rows of the widest layer and
    /// `ff` the widest dense block. Load-time only.
    fn new(stream: &CudaStream, d: &Dims, qkv: usize, ff: usize) -> Result<Scratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(Scratch {
            x: z(d.embd)?,
            x1: z(d.embd)?,
            xn: z(d.embd)?,
            out: z(d.embd)?,
            qkv: z(qkv)?,
            q: z(d.heads * HEAD_K192)?,
            part_v: z(flash_gqa::partials_v_len(1, d.heads))?,
            part_ms: z(flash_gqa::partials_ms_len(1, d.heads))?,
            y: z(d.heads * HEAD)?,
            h: z(ff.max(1))?,
            rout: RouterOut::new(stream)?,
            sel: DeviceBuffer::zeroed(stream, N_USED)?,
            no_bias: z(N_EXPERT)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    fn bytes(&self) -> usize {
        let f = [
            &self.x,
            &self.x1,
            &self.xn,
            &self.out,
            &self.qkv,
            &self.q,
            &self.part_v,
            &self.part_ms,
            &self.y,
            &self.h,
            &self.no_bias,
            &self.rout.logits,
            &self.rout.probs,
            &self.rout.weights,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.rout.ids.num_bytes()
            + self.sel.num_bytes()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
    }
}

/// The token embedding as the host reads it: the shared table's q8_0 rows,
/// one dequantized per step.
struct Embedding {
    table: EmbedTable,
    row: Vec<f32>,
}

impl Embedding {
    fn new(file: Arc<Split>, n_embd: usize) -> Result<Embedding, GpuError> {
        let name = names::token_embd();
        let table = EmbedTable::new(file, &name, n_embd)?;
        if table.ty() != GgmlType::Q8_0 {
            return Err(GpuError::Tensor {
                what: WHAT,
                name,
                need: "q8_0 rows of embedding_length values",
            });
        }
        Ok(Embedding {
            table,
            row: vec![0.0; n_embd],
        })
    }

    /// Row `token`, dequantized; a token past the vocabulary is refused by
    /// name.
    fn fill(&mut self, token: u32) -> Result<(), GpuError> {
        self.table.row_into(token, &mut self.row)?;
        Ok(())
    }
}

/// One step's host values: its position.
#[derive(Clone, Copy, Debug)]
pub struct StepInput {
    pos: u32,
}

/// The MiMo body. Field order is drop order: the host tier (its lock over
/// the host set and its page windows) before the buffers.
pub struct Body {
    hybrid: Hybrid<HostRun>,
    layers: Range<usize>,
    cfg: Vec<LayerCfg>,
    names: Vec<LayerNames>,
    dims: Dims,
    k: Kernels,
    s: Scratch,
    stores: Vec<Store>,
    /// One table a distinct rope base.
    ropes: Vec<RopeRows>,
    /// The slot map's card copy, the host run's rows: every entry the host
    /// mark. The host tier holds the host copy.
    slots: Arc<DeviceTensor<u32>>,
    embd: Embedding,
    /// Each layer's output after the last step, when a gate armed them.
    taps: Option<Vec<DeviceBuffer<f32>>>,
    /// Positions every store holds.
    ctx: usize,
}

/// The walk's parts, lent apart from the host tier.
pub(crate) struct Parts<'s> {
    pub k: &'s Kernels,
    pub d: &'s Dims,
    pub cfg: &'s [LayerCfg],
    pub names: &'s [LayerNames],
    pub s: &'s mut Scratch,
    pub stores: &'s mut [Store],
    pub ropes: &'s [RopeRows],
    pub slots: &'s DeviceTensor<u32>,
    pub taps: Option<&'s mut [DeviceBuffer<f32>]>,
    pub ctx: usize,
}

/// A shape the kernels have no geometry for, by name.
pub(crate) fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
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

/// The resident weight `name` holds `rows` rows of `k` values, in q8_0 when
/// `q8_0` and in f32 otherwise: a weight of other shapes would leave the
/// rows its launch reads unwritten, so the load refuses it by name.
fn expect_weight(
    w: &Weights,
    name: &str,
    q8_0: bool,
    (rows, k): (usize, usize),
) -> Result<(), GpuError> {
    let (is_q8, got_k) = match weight(w, name)? {
        DevWeight::Q8_0 { k, .. } => (true, *k),
        DevWeight::F32 { k, .. } => (false, *k),
        _ => (false, 0),
    };
    let got_rows = weight(w, name)?.rows();
    if is_q8 != q8_0 || got_rows != rows || got_k != k {
        return Err(shape(format!(
            "{name} holds {got_rows} rows of {got_k} {}; its launch reads {rows} rows of {k} {}",
            if is_q8 {
                "q8_0"
            } else {
                "values of another format"
            },
            if q8_0 { "q8_0" } else { "f32" },
        )));
    }
    Ok(())
}

/// Every weight layer `l` reads is resident in the rows its launches take.
fn check_layer(w: &Weights, c: &LayerCfg, n: &LayerNames, d: &Dims) -> Result<(), GpuError> {
    let a = c.attn;
    let qkv_rows = d.heads * HEAD_K192 + a.kv_heads * (HEAD_K192 + HEAD);
    expect_weight(w, &n.attn.norm, false, (1, d.embd))?;
    expect_weight(w, &n.attn.qkv, true, (qkv_rows, d.embd))?;
    expect_weight(w, &n.attn.o, true, (d.embd, d.heads * HEAD))?;
    if let Some(sinks) = &n.attn.sinks {
        expect_weight(w, sinks, false, (1, d.heads))?;
    }
    match &n.ffn {
        FfnNames::Dense {
            norm,
            gate,
            up,
            down,
        } => {
            expect_weight(w, norm, false, (1, d.embd))?;
            expect_weight(w, gate, true, (c.ffn.ff, d.embd))?;
            expect_weight(w, up, true, (c.ffn.ff, d.embd))?;
            expect_weight(w, down, true, (d.embd, c.ffn.ff))?;
        }
        FfnNames::Moe { norm, router, bias } => {
            expect_weight(w, norm, false, (1, d.embd))?;
            expect_weight(w, router, false, (N_EXPERT, d.embd))?;
            if let Some(bias) = bias {
                expect_weight(w, bias, false, (1, N_EXPERT))?;
            }
        }
    }
    Ok(())
}

/// Refused by name when the load's scratch, `made` device bytes
/// ([`Body::scratch_bytes`]), passes what card `card` of `plan` sets aside
/// for it (the card's `scratch_bytes` term): the plan counts those bytes by
/// that term alone, no formula of the body's, so the load holds them to it.
fn refuse_scratch_past(plan: &Plan<'_>, card: usize, made: usize) -> Result<(), GpuError> {
    let term = plan
        .machine
        .cards
        .get(card)
        .map(|c| c.scratch_bytes)
        .ok_or_else(|| shape(format!("the plan has no card {card}")))?;
    if made as u64 > term {
        return Err(shape(format!(
            "the load's scratch holds {made} device bytes (the step's buffers, the rope \
             tables, the host boundary, the slot map's copy); card {card}'s plan sets aside \
             {term} for it"
        )));
    }
    Ok(())
}

/// The file's widths and constants, each checked against the kernel that
/// fixes it.
fn dims_of(inputs: &PlanInputs) -> Result<Dims, GpuError> {
    let hp = &inputs.hp;
    let checks = [
        ("attention.key_length", hp.head_k, HEAD_K192),
        ("attention.value_length", hp.head_v, HEAD),
        ("rope.dimension_count", hp.rope_dims, ROT_K192),
        ("expert_count", hp.n_expert, N_EXPERT),
        ("expert_used_count", hp.n_used, N_USED),
    ];
    for (key, got, want) in checks {
        if got != want {
            return Err(shape(format!(
                "{key} is {got}; the kernels are built for {want}"
            )));
        }
    }
    if !hp.n_embd.is_multiple_of(32) {
        return Err(shape(format!(
            "embedding_length {} is not whole blocks of 32",
            hp.n_embd
        )));
    }
    Ok(Dims {
        embd: hp.n_embd,
        heads: hp.n_head,
        rms_eps: hp.rms_eps,
        scale: hp.weights_scale,
        score_scale: 1.0 / (HEAD_K192 as f32).sqrt(),
    })
}

/// Layer `l`'s programs and values from its description `s`: the attention's
/// ([`facts::attn_args`]) with its rope base's table in `thetas` (added
/// when new), the block's ([`facts::block_args`]). A layer that is not a
/// plain residual is refused
/// by name; a routed layer's row in the card copy is the caller's to fill.
fn layer_cfg(
    l: usize,
    s: &models::LayerSpec,
    d: &Dims,
    thetas: &mut Vec<f32>,
) -> Result<LayerCfg, GpuError> {
    let kind = Layer::of(s);
    if kind.residual != ResidualKind::Plain {
        return Err(shape(format!(
            "layer {l}: hyper-connection streams; the mimo2 chain is a plain residual"
        )));
    }
    let built = AttnBuilt {
        score_head: HEAD_K192,
        value_head: HEAD,
        rotated: ROT_K192,
        heads: d.heads,
    };
    let attn = facts::attn_args(l, s, built).map_err(|r| shape(r.to_string()))?;
    let router = RouterBuilt {
        n_expert: N_EXPERT,
        n_used: N_USED,
        scale: d.scale,
    };
    let ffn = facts::block_args(l, s, router).map_err(|r| shape(r.to_string()))?;
    let rope = match thetas
        .iter()
        .position(|t| t.to_bits() == attn.theta.to_bits())
    {
        Some(i) => i,
        None => {
            thetas.push(attn.theta);
            thetas.len() - 1
        }
    };
    Ok(LayerCfg {
        kind,
        attn,
        rope,
        ffn,
        row_off: 0,
    })
}

impl Body {
    /// Card `card` of `plan`, which `inputs` made, resident: the layers' weights
    /// the plan puts on the card, the stores at the plan's `ctx_max`, the host
    /// tier over every routed layer and the embedding as a host row; the load's
    /// slot map for the model's life. Refused by name (module doc): a plan of
    /// other than one card or with an expert tier, one that puts a routed
    /// expert on a card, a layer or a width no kernel here runs, routed layers
    /// that are not one run, stores whose bytes are not the plan's, and the
    /// machine's own refusals at the load.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Mimo2Model, GpuError> {
        GpuModel::load_placed(
            file,
            plan,
            card,
            host,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set| Body::load(gpu, Arc::new(file), w, (plan, inputs, card), host, set),
        )
    }

    /// The body of card `card`: its layers' programs and values from the
    /// plan's description, the stores at the plan's `ctx_max`, the step's
    /// buffers, the slot map the plan's routed segments make and the host tier
    /// over the routed run.
    fn load(
        gpu: &Gpu,
        file: Arc<Split>,
        w: &Weights,
        (plan, inputs, card): (&Plan<'_>, &PlanInputs, usize),
        host: HostCfg,
        residency: HostResidency,
    ) -> Result<Body, GpuError> {
        let (hp, spec) = (&inputs.hp, &inputs.spec);
        if plan.machine.cards.len() != 1 || !plan.machine.tiers.is_empty() {
            return Err(shape(format!(
                "a plan of {} cards and {} expert tiers; the program runs every layer on one \
                 card with no expert tier",
                plan.machine.cards.len(),
                plan.machine.tiers.len()
            )));
        }
        let layers = plan
            .machine
            .cards
            .get(card)
            .map(|c| c.layers.clone())
            .ok_or_else(|| shape(format!("the plan has no card {card}")))?;
        if layers != (0..spec.layers.len()) {
            return Err(shape(format!(
                "card {card} runs layers {layers:?}; the program runs every layer of the trunk \
                 (0..{})",
                spec.layers.len()
            )));
        }
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0 && u32::try_from(c).is_ok())
            .ok_or_else(|| {
                shape(format!(
                    "ctx_max {}: at least 1 and at most the positions a u32 counts",
                    plan.ctx_max
                ))
            })?;
        let dims = dims_of(inputs)?;
        let run = hosted(&spec.layers).map_err(|e| shape(e.to_string()))?;
        let map = SlotMap::of_plan(plan, card, None, run.clone(), N_EXPERT)?;
        for l in run.clone() {
            let (on_card, on_tier) = (map.on_card(l)?, map.on_tier(l)?);
            if on_card + on_tier > 0 {
                return Err(shape(format!(
                    "layer {l}: the plan puts {} routed experts on a card; this program runs \
                     every routed expert on the host tier",
                    on_card + on_tier
                )));
            }
        }
        let mut thetas = Vec::new();
        let mut cfg = Vec::with_capacity(spec.layers.len());
        for (l, s) in spec.layers.iter().enumerate() {
            let mut c = layer_cfg(l, s, &dims, &mut thetas)?;
            if c.kind.ffn == FfnKind::Moe {
                c.row_off = map.row_offset(l).ok_or_else(|| {
                    shape(format!("layer {l}: a routed layer with no slot-map row"))
                })?;
            }
            cfg.push(c);
        }
        let names: Vec<LayerNames> = cfg
            .iter()
            .enumerate()
            .map(|(l, c)| LayerNames {
                attn: AttnNames::of(l, &c.attn),
                ffn: FfnNames::of(l, c.kind.ffn, &c.ffn),
            })
            .collect();
        for (c, n) in cfg.iter().zip(&names) {
            check_layer(w, c, n, &dims)?;
        }
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        // MiMo fact: only the window layers carry sinks, one per query head, file constants.
        for (l, n) in names.iter().enumerate() {
            if let Some(name) = &n.attn.sinks {
                let sinks = f32v(w, name)?.to_host_vec(stream)?;
                checked_sinks(l, &sinks, dims.heads)?;
            }
        }
        let qkv = cfg
            .iter()
            .map(|c| dims.heads * HEAD_K192 + c.attn.kv_heads * (HEAD_K192 + HEAD))
            .max()
            .unwrap_or(0);
        let ff = cfg.iter().map(|c| c.ffn.ff).max().unwrap_or(0);
        let s = Scratch::new(stream, &dims, qkv, ff)?;
        let stores = cfg
            .iter()
            .map(|c| Store::new(stream, c.attn.kv_heads, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let held: usize = stores.iter().map(Store::bytes).sum();
        let planned = inputs.kv.bytes(layers.clone(), plan.ctx_max);
        if held as u64 != planned {
            return Err(shape(format!(
                "the stores hold {held} device bytes; the plan's layout counts {planned}"
            )));
        }
        let ropes = thetas
            .iter()
            .map(|&t| RopeRows::new(stream, t, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            run.len(),
            N_EXPERT,
        )?);
        let boundary = Boundary::new(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: dims.embd,
                n_used: N_USED,
            },
        )?;
        let experts = HostRun::build(
            Arc::clone(&file),
            run.start,
            host.r8,
            HostWidths {
                embd: hp.n_embd,
                ff: hp.expert_ff,
                n_used: hp.n_used,
            },
            |src| routed_layers(src, hp, run.clone()),
        )?;
        let mut hybrid = Hybrid::new(boundary, map, experts, run.len())?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(residency);
        let embd = Embedding::new(file, dims.embd)?;
        if embd.table.n_vocab() != hp.n_vocab {
            return Err(shape(format!(
                "the embedding has {} rows, the file's vocabulary {}",
                embd.table.n_vocab(),
                hp.n_vocab
            )));
        }
        let body = Body {
            hybrid,
            layers,
            cfg,
            names,
            dims,
            k: Kernels::load(gpu)?,
            s,
            stores,
            ropes,
            slots,
            embd,
            taps: None,
            ctx,
        };
        refuse_scratch_past(plan, card, body.scratch_bytes())?;
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
        let leg = |c: &LayerCfg| c.kind.host_leg();
        let first = self.cfg.iter().position(leg);
        let last = self.cfg.iter().rposition(leg);
        first.zip(last).map_or(0..0, |(f, l)| f..l + 1)
    }

    /// What each layer's attention launches take: its kv heads, window,
    /// sinks and value multiplier, and the base of the rope table the layer
    /// reads (not the description's: the table's own).
    #[must_use]
    pub fn attn_args(&self) -> Vec<AttnArgs> {
        self.cfg
            .iter()
            .map(|c| AttnArgs {
                theta: self.ropes[c.rope].theta,
                ..c.attn
            })
            .collect()
    }

    /// Device bytes of the stores.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store::bytes).sum()
    }

    /// Device bytes of the load's scratch, what the plan's card term
    /// `scratch_bytes` sets aside: the step's buffers, the rope tables, the
    /// host boundary and the slot map's card copy. A load past the term is
    /// refused by name.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self
                .ropes
                .iter()
                .map(|r| r.table.num_bytes())
                .sum::<usize>()
            + self.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    /// The host tier, for a gate's instrument or test seam.
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    /// Arm (or disarm) the per-layer taps: after each layer the chain copies
    /// its output into the layer's tap, which [`Body::taps`] reads back. Only
    /// through [`set_taps`], which drops the captured chains first: a capture
    /// holds the tap copies it was taken with.
    pub(crate) fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        self.taps = if on {
            Some(
                self.cfg
                    .iter()
                    .map(|_| DeviceBuffer::zeroed(gpu.stream(), self.dims.embd))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        Ok(())
    }

    /// Every layer's output after the last step, `n_embd` a layer: layer `l`
    /// at `l · n_embd`. Blocking; refused when the taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let taps = self.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps (set_taps)",
        })?;
        let mut out = Vec::with_capacity(taps.len() * self.dims.embd);
        for t in taps {
            out.extend(t.to_host_vec(gpu.stream())?);
        }
        Ok(out)
    }

    /// The walk's parts, lent apart from the host tier.
    pub(crate) fn parts(&mut self) -> (Parts<'_>, &mut Hybrid<HostRun>) {
        (
            Parts {
                k: &self.k,
                d: &self.dims,
                cfg: &self.cfg,
                names: &self.names,
                s: &mut self.s,
                stores: &mut self.stores,
                ropes: &self.ropes,
                slots: &self.slots,
                taps: self.taps.as_deref_mut(),
                ctx: self.ctx,
            },
            &mut self.hybrid,
        )
    }
}

/// Arm (or disarm) `m`'s per-layer taps ([`Body::taps`] reads them back), the
/// model first set to eager steps: a captured chain holds the tap copies it
/// was taken with, so arming under one would leave the taps unwritten and
/// disarming would free buffers its replays write.
pub fn set_taps(m: &mut Mimo2Model, on: bool) -> Result<(), GpuError> {
    m.set_mode(StepMode::Eager);
    let (gpu, _, body) = m.body_parts("mimo2 set_taps")?;
    body.set_taps(gpu, on)
}

impl ChainBody for Body {
    type Input = StepInput;
    type Host = Body;

    fn arch() -> Arch {
        Arch::Mimo2
    }

    /// The step at `pos`: its embedding row read from the file. A position
    /// past the stores or a token past the vocabulary is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(shape(format!(
                "a step at position {pos} in stores of {}",
                self.ctx
            )));
        }
        self.embd.fill(token)?;
        Ok(StepInput { pos })
    }

    /// The embedding row into `x`, the position and the live key count: three
    /// host-to-device copies.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        self.s.x.copy_from_host(stream, &self.embd.row)?;
        self.s.pos.copy_from_host(stream, &[input.pos])?;
        self.s.n_keys.copy_from_host(stream, &[input.pos + 1])?;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let (parts, hybrid) = self.parts();
        program::walk_step(gpu, w, parts, hybrid, head)
    }

    /// The host tier settled. The stores keep their rows: a position is
    /// written before a later one reads it, so a fresh prompt reads only what
    /// it wrote.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.hybrid.settle(gpu.stream())
    }

    /// `attention.layer_norm_rms_epsilon`: every RMS norm's, the head's
    /// included.
    fn head_eps(&self) -> f32 {
        self.dims.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.scratch_bytes()
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

    fn noted(&self, e: GpuError) -> GpuError {
        self.hybrid.noted(e)
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.take_step_refusal()
    }

    /// The host tier's refusal poison lifted ([`Hybrid::lift_refusal`]) — the
    /// settling a reset runs is [`Hybrid::settle`]'s ([`Body::reset`]).
    fn lift_refusal(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid.lift_refusal(stream)
    }

    /// The refusal the host tier is poisoned by now ([`Hybrid::refusal_poison`]).
    fn refusal_poison(&self) -> Option<Refusal> {
        self.hybrid.refusal_poison()
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid.residency()
    }
}
