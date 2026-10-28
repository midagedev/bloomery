//! The Solar-Open2 body: the load, each layer's store, the buffers the step
//! and the prompt batch run over, and [`Body`] behind `GpuModel`.
//!
//! The buffers are fixed here so the step ([`crate::step`]) and the prompt
//! batch ([`crate::prefill`]) are written against them at once: the step
//! reads [`Scratch`] (one row), the batch [`Batch`] (up to [`T_MAX`] rows);
//! both write the same [`Store`]s and the same tier shell.
//!
//! All 48 layers are one card's chain, each a mixer by its description (a
//! NoPE GQA layer with its out gate at every fourth layer from 0, a KDA layer
//! elsewhere) and a routed block with one shared expert, whose experts run on
//! the card where the plan puts them ([`crate::ffn::Routed`]), on the plan's
//! expert tier card ([`crate::tier`]) and on the host tier otherwise. A KDA layer's store is its recurrent state, its stamp and
//! its conv ring, one lane (the file has no next-token head), whatever the
//! context; a GQA layer's the K and V planes of every position. The
//! embedding is a host row, dequantized per step and copied into `x` before
//! the launch.
//!
//! The recurrent state has no history: a cut back to a position restores the
//! checkpoint taken there ([`Body::checkpoint`]), or the empty model at 0,
//! and is carried out at the next device call ([`Body::apply_cut`]).
//!
//! What the load refuses, by name: a plan of other than one stage card, more
//! expert tier cards than the host tier serves, a layer other than the two mixers over a routed block on a
//! plain residual, GQA or KDA widths the kernels have no geometry for, KDA
//! layers that disagree on their rule, a tensor whose type no site launches
//! or that is not resident in the rows its launch reads, stores whose bytes
//! are not the plan's, a step scratch past the plan's term, and a prompt batch past the
//! plan's reserve. The prompt batch's buffers ([`Body::prompt_bytes`]) are beside the scratch
//! term: the plan keeps [`PlanInputs::prompt_reserve`] of the card for them out of the expert
//! budget, and the load holds their made bytes to it.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::arch::hybrid::{Acts, GqaGate, GqaLayer, Site, wide_acts_at};
use bloomery_gpu::checkpoint::{self, Checkpoints};
use bloomery_gpu::flash_gqa::{GROUP, partials_ms_len, partials_v_len};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::leg_tier::{Hang, LegOpen};
use bloomery_gpu::host::refuse_tier_count;
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::served::{ResidencyParts, TierBody};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap::{PassReport, Residency};
use bloomery_gpu::host::swap_source::{
    DEADLINE, FileSwap, LIVE_DELAY, PlanStacks, ResidencyGlue, ResidencySpec, hosted_stacks,
};
use bloomery_gpu::host::tier::TierOpen;
use bloomery_gpu::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use bloomery_gpu::linear::mixer_kda::{KdaDecay, KdaDims, KdaRule};
use bloomery_gpu::linear::{CONV_TAPS, HEAD, KHeadMap, LinearShape};
use bloomery_gpu::model::{
    ChainBody, HostServed, MAX_PASS_ROWS, Rollback, StepMode, agreed, hosted_hybrid_tiered,
    refuse_reserve_past, refuse_scratch_past, stage_run,
};
use bloomery_gpu::site::{self, SiteTy, WideActs};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use bloomery_levers::{ChurnHosting, HostCfg};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::names;
use model::arch::solar2::host as solar_host;
use model::arch::solar2::place::{self, PlanInputs};
use model::embed::EmbedTable;
use model::placement::Plan;
use models::{DeltaKind, Gqa, KdaGate, LayerSpec, Mixer, ModelSpec, OutGate, RopeMode};
use runtime::layer::{FfnKind, Layer, MixerKind, ResidualKind, hosted};
use runtime::seqstate::{Kept, Take};

use bloomery_gpu_deepseek41::prompt_walk::{self, BatchBufs};

use crate::ffn::{CARD, Routed, RoutedDims};
use crate::prefill::Extra;
use crate::step::Kernels;
use crate::tier;

/// What the body's refusals name.
pub(crate) const WHAT: &str = "solar2 Body";

/// The prompt batch's rows at most: one [`Batch`] holds them.
pub const T_MAX: usize = 512;

// The plan's reserve for the batch is the formula of `place::prompt_reserve_bytes` at its rows,
// over the kernels' constants it copies.
const _: () = assert!(T_MAX == place::PROMPT_ROWS);
const _: () = assert!(place::FLASH_SEGMENTS == bloomery_gpu::flash_gqa::SEGMENTS);
const _: () = assert!(place::PARTIAL_ROWS == MAX_PASS_ROWS);
const _: () = assert!(place::GEMM_TILE_COLS == bloomery_gpu::gemm::GEMM_BN);
const _: () = assert!(place::GEMM32_STEP == bloomery_gpu::gemm::GEMM32_STEP);

/// The GQA word record: the first word is the position of the row (or of
/// the unit's first), the words from [`WORD_ROWS`] one a key head, the plane
/// row `h · ctx + position` the append lands at. The step's, the batch's and
/// a unit's records share this layout.
pub const WORD_POS: usize = 0;
/// See [`WORD_POS`].
pub const WORD_ROWS: usize = 1;

/// The rows in flight a load's host boundary and residency tally hold: the
/// step's one (the file has no draft to verify rows of).
pub const LOAD_ROWS: usize = 1;

/// The spacing of a prompt call's inner checkpoints; past the sequence's
/// budget the least recently used is evicted (`runtime::seqstate`).
pub const CHECKPOINT_EVERY: u32 = 512;

/// The Solar model: one card, the skeleton over this body.
pub type Solar2Model = GpuModel<Body>;

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
    /// Query heads and key heads of a GQA layer (64 and 8: group 8).
    pub heads: usize,
    pub kv_heads: usize,
    pub rms_eps: f32,
    /// The KDA mixer's widths and rule (`Softplus`, β scale 2).
    pub kda: KdaDims,
    /// Routed experts' and the shared expert's widths.
    pub expert_ff: usize,
    pub shared_ff: usize,
    pub ctx: usize,
}

/// A KDA layer's typed projections (Q4_K, Q5_0 on the low-rank halves).
pub struct KdaSites {
    /// q, k and v, each its own tensor's type (`attn_v` is Q6_K on some
    /// layers): the step writes them into the joined rows' three windows.
    pub qkv: [Site; 3],
    pub f_a: Site,
    pub g_a: Site,
    pub beta: Site,
    pub f_b: Site,
    pub g_b: Site,
    pub out: Site,
    /// The layer's f32 vectors by name: norm, conv (q|k|v joined), dt bias,
    /// `ssm_a`, gate norm.
    pub vecs: [String; 5],
}

impl KdaSites {
    /// Every site the layer launches, for the load's [`Acts::new`].
    #[must_use]
    pub fn sites(&self) -> [&Site; 9] {
        let [q, k, v] = &self.qkv;
        [
            q, k, v, &self.f_a, &self.g_a, &self.beta, &self.f_b, &self.g_b, &self.out,
        ]
    }
}

/// One layer's store and resident pieces, by its mixer kind.
#[allow(
    clippy::large_enum_variant,
    reason = "one store a layer in a vector of 48; boxing the KDA sites would put a pointer between the step and every site it launches (rust-quality R16)"
)]
pub enum Store {
    /// The recurrent state (64 heads × 128 × 128 f32), its stamp and the
    /// conv ring; one lane (the file has no NextN head).
    Kda {
        sites: KdaSites,
        state: DeviceBuffer<f32>,
        stamp: DeviceBuffer<u32>,
        ring: DeviceBuffer<f32>,
    },
    /// The GQA layer's sites, out gate and K/V planes (common type).
    Gqa(GqaLayer),
}

impl Store {
    /// Device bytes the store holds, the sites' weights not included.
    #[must_use]
    pub fn bytes(&self) -> usize {
        match self {
            Store::Kda {
                state, stamp, ring, ..
            } => state.num_bytes() + stamp.num_bytes() + ring.num_bytes(),
            Store::Gqa(g) => g.bytes(),
        }
    }

    /// Every site the layer launches.
    pub(crate) fn sites(&self) -> Vec<&Site> {
        match self {
            Store::Kda { sites, .. } => sites.sites().to_vec(),
            Store::Gqa(g) => g.sites(),
        }
    }
}

/// One layer's programs (`runtime::layer::Layer::of` its description) and
/// its routed block's card row.
#[derive(Clone, Copy, Debug)]
pub struct LayerCfg {
    pub kind: Layer,
    pub row_off: usize,
}

/// The step's one-row buffers: the streams, every mixer's intermediates
/// (lent to `mixer_kda::enqueue` and `hybrid::gqa_layer` as their `Bufs`),
/// the routed block's, the word record, the activation forms.
pub struct Scratch {
    pub x: DeviceBuffer<f32>,
    pub x1: DeviceBuffer<f32>,
    pub pos: DeviceBuffer<u32>,
    pub words: DeviceBuffer<u32>,
    pub n_keys: DeviceBuffer<u32>,
    /// The mixers' intermediates and the activation forms.
    pub mixers: MixerBufs,
}

impl Scratch {
    /// `rows` rows of every width the mixers and `sites` read. Load-time
    /// only.
    fn new<'s>(
        stream: &CudaStream,
        d: &Dims,
        rows: usize,
        sites: impl Iterator<Item = &'s Site>,
    ) -> Result<Scratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, rows * n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        Ok(Scratch {
            x: z(d.embd)?,
            x1: z(d.embd)?,
            pos: u(rows)?,
            words: u(WORD_ROWS + d.kv_heads)?,
            n_keys: u(rows)?,
            mixers: MixerBufs::new(stream, d, rows, sites)?,
        })
    }

    /// Device bytes of every buffer and activation form.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let f32s = [&self.x, &self.x1]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();
        let words = [&self.pos, &self.words, &self.n_keys]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();
        f32s + words + self.mixers.bytes()
    }
}

/// The mixers' buffers: the normed rows, every mixer's intermediates and the
/// activation forms, at `rows` rows. The step holds them in its [`Scratch`],
/// the prompt batch in its [`Batch`].
pub struct MixerBufs {
    pub xn: DeviceBuffer<f32>,
    /// KDA: qkv, fa, ga, beta_raw, f, z, conv, beta, decay, o, gated.
    pub kda: [DeviceBuffer<f32>; 11],
    /// GQA: q, k, v, g, attn, part_v, part_ms.
    pub gqa: [DeviceBuffer<f32>; 7],
    pub delta: DeviceBuffer<f32>,
    pub acts: Acts,
}

impl MixerBufs {
    /// `rows` rows of every width the mixers and `sites` read. Load-time
    /// only.
    fn new<'s>(
        stream: &CudaStream,
        d: &Dims,
        rows: usize,
        sites: impl Iterator<Item = &'s Site>,
    ) -> Result<MixerBufs, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, rows * n);
        let shape = d.kda.shape;
        let (channels, v, n_v) = (shape.channels(), shape.n_v * HEAD, shape.n_v);
        let (q_width, kv_width) = (d.heads * HEAD, d.kv_heads * HEAD);
        // The decode flash splits a row's keys; a batch's flash needs no partials.
        let attn_rows = rows.min(MAX_PASS_ROWS);
        Ok(MixerBufs {
            xn: z(d.embd)?,
            kda: [
                z(channels)?,
                z(HEAD)?,
                z(HEAD)?,
                z(n_v)?,
                z(v)?,
                z(v)?,
                z(channels)?,
                z(n_v)?,
                z(v)?,
                z(v)?,
                z(v)?,
            ],
            gqa: [
                z(q_width)?,
                z(kv_width)?,
                z(kv_width)?,
                z(q_width)?,
                z(q_width)?,
                DeviceBuffer::zeroed(stream, partials_v_len(attn_rows, d.heads))?,
                DeviceBuffer::zeroed(stream, partials_ms_len(attn_rows, d.heads))?,
            ],
            delta: z(d.embd)?,
            acts: Acts::new(stream, sites)?,
        })
    }

    /// Device bytes of every buffer and activation form.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let f32s = [&self.xn, &self.delta]
            .into_iter()
            .chain(&self.kda)
            .chain(&self.gqa)
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();
        f32s + self.acts.bytes()
    }
}

/// The prompt batch's buffers: the mixers' rows at [`T_MAX`] rows, and the
/// wide activation forms the GEMMs read.
pub struct Batch {
    pub rows: MixerBufs,
    pub wide: WideActs,
}

impl Batch {
    /// Device bytes of the rows and the wide forms.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.bytes() + self.wide.bytes()
    }
}

/// The token embedding as the host reads it, and the step's host values:
/// one dequantized row and the GQA word record.
pub(crate) struct HostIn {
    pub(crate) table: EmbedTable,
    row: Vec<f32>,
    words: Vec<u32>,
}

/// The Solar body. Field order is drop order: the shell, whose host tier
/// (its lock over the host set and its page windows) comes first, before the
/// buffers.
pub struct Body {
    /// The host tier, the residency glue and the taps.
    pub(crate) shell: TierShell<HostRun>,
    pub(crate) dims: Dims,
    pub(crate) cfg: Vec<LayerCfg>,
    pub(crate) stores: Vec<Store>,
    pub(crate) s: Scratch,
    pub(crate) batch: Batch,
    /// The prompt batch's units: the walker's own buffers.
    pub(crate) units: BatchBufs,
    pub(crate) slots: Arc<DeviceTensor<u32>>,
    /// The positions the stores hold: a step at `pos` needs `at == pos`, and
    /// it leaves `pos + 1`.
    pub(crate) at: u32,
    /// The mixers' kernels and the lane word.
    pub(crate) k: Kernels,
    /// The routed blocks, the card experts among them.
    pub(crate) ffn: Routed,
    /// The batch's buffers beyond [`Batch`].
    pub(crate) prefill: Extra,
    /// The step the last [`ChainBody::decode_input`] accepted: what
    /// `enqueue_chain` walks.
    cur: StepInput,
    pub(crate) embd: HostIn,
    /// The KDA layers' stores on the host at chosen positions.
    ckpt: Checkpoints,
    residency: Residency,
}

/// A shape the kernels have no geometry for, by name.
pub(crate) fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// Site types a K-quant projection of the file may be; the step's input
/// gemvs read the q8_1 form only these read.
const KQUANT: &[SiteTy] = &[SiteTy::Q3K, SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K];

/// Site types of the low-rank halves' second projections (K = 128: no
/// K-quant).
const LOW_RANK: &[SiteTy] = &[SiteTy::Q5_0, SiteTy::Q8_0, SiteTy::F32];

/// The joined conv taps of KDA layer `l`'s q, k and v, in the conv's channel
/// order: the one weight the mixer's conv reads, made at load.
fn conv_joined(l: usize) -> String {
    format!("derived.blk.{l}.ssm_conv1d_qkv")
}

/// The joins of the load: each KDA layer's three conv tap sets into one.
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
        if kind.mixer != MixerKind::DeltaRule {
            continue;
        }
        let parts = ['q', 'k', 'v'].map(|p| names::ssm_conv1d(l, p));
        let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
        w.join_rows(stream, &parts, conv_joined(l))?;
    }
    Ok(())
}

/// The KDA layer's rule from its description: the decay and the scale on
/// `σ(β)`, refused by name when no conv-prep entry runs the pair.
fn kda_rule(gate: KdaGate, beta_scale: f32) -> Result<KdaRule, GpuError> {
    let decay = match gate {
        KdaGate::Sigmoid { lower_bound } => KdaDecay::Sigmoid { lower_bound },
        KdaGate::Softplus => KdaDecay::Softplus,
    };
    KdaRule {
        decay,
        beta_scale,
        write_gate: false,
    }
    .check(WHAT)
}

/// A GQA layer's description held to what the kernels run: NoPE, no qk
/// norm, the out gate its own projection, every position attended, group
/// [`GROUP`] of [`HEAD`]-wide heads.
fn gqa_checked(l: usize, g: &Gqa) -> Result<(), GpuError> {
    let plain = g.rope.mode == RopeMode::None
        && !g.qk_norm
        && g.out_gate == OutGate::Proj
        && g.select.is_none()
        && g.window.is_none()
        && !g.sinks
        && g.value_scale.is_none();
    if !plain {
        return Err(shape(format!(
            "layer {l}: {g:?} is not a NoPE GQA layer with a projected out gate over every \
             position"
        )));
    }
    if g.head_dim as usize != HEAD || g.value_dim as usize != HEAD {
        return Err(shape(format!(
            "layer {l}: heads of {} keys and {} values; the kernels are built for {HEAD}",
            g.head_dim, g.value_dim
        )));
    }
    if g.group() as usize != GROUP {
        return Err(shape(format!(
            "layer {l}: {} query heads over {} key heads; the flash runs {GROUP} a key head",
            g.heads, g.kv_heads
        )));
    }
    Ok(())
}

/// The file's widths and constants, each checked against the kernel that
/// fixes it, from the description's layers.
fn dims_of(inputs: &PlanInputs, ctx: usize) -> Result<Dims, GpuError> {
    let (hp, spec) = (&inputs.hp, &inputs.spec);
    let checks = [
        ("attention.key_length", hp.head_dim, HEAD),
        ("kda.head_dim", hp.kda.head_dim, HEAD),
        ("ssm.conv_kernel", hp.kda.conv, CONV_TAPS),
    ];
    for (key, got, want) in checks {
        if got != want {
            return Err(shape(format!(
                "{key} is {got}; the kernels are built for {want}"
            )));
        }
    }
    if !hp.n_embd.is_multiple_of(256) {
        return Err(shape(format!(
            "embedding_length {} is not whole K-quant blocks of 256",
            hp.n_embd
        )));
    }
    let (mut kv, mut kda_heads, mut rule) = (None, None, None);
    for (l, s) in spec.layers.iter().enumerate() {
        match &s.mixer {
            Mixer::Gqa(g) => {
                gqa_checked(l, g)?;
                if g.heads as usize != hp.n_head {
                    return Err(shape(format!(
                        "layer {l}: {} query heads, the file's attention.head_count {}",
                        g.heads, hp.n_head
                    )));
                }
                agreed(WHAT, "the key head count", l, &mut kv, g.kv_heads as usize)?;
            }
            Mixer::DeltaRule(d) => {
                let DeltaKind::Kda { gate, beta_scale } = d.kind else {
                    return Err(shape(format!(
                        "layer {l}: {:?}; this program runs KDA layers",
                        d.kind
                    )));
                };
                if d.d as usize != HEAD || d.conv as usize != CONV_TAPS || d.k_heads != d.v_heads {
                    return Err(shape(format!(
                        "layer {l}: KDA of {} key and {} value heads of {} values and {} conv \
                         taps; the kernels are built for equal head counts of {HEAD} and \
                         {CONV_TAPS} taps",
                        d.k_heads, d.v_heads, d.d, d.conv
                    )));
                }
                agreed(
                    WHAT,
                    "the KDA head count",
                    l,
                    &mut kda_heads,
                    d.v_heads as usize,
                )?;
                agreed(
                    WHAT,
                    "the KDA rule",
                    l,
                    &mut rule,
                    kda_rule(gate, beta_scale)?,
                )?;
            }
            other => {
                return Err(shape(format!(
                    "layer {l}: a {other:?} mixer; this program runs GQA and KDA"
                )));
            }
        }
    }
    let kv_heads = kv.ok_or_else(|| shape("no GQA layer in the description"))?;
    let n_v = kda_heads.ok_or_else(|| shape("no KDA layer in the description"))?;
    let rule = rule.ok_or_else(|| shape("no KDA layer in the description"))?;
    Ok(Dims {
        embd: hp.n_embd,
        vocab: hp.n_vocab,
        heads: hp.n_head,
        kv_heads,
        rms_eps: hp.rms_eps,
        kda: KdaDims {
            embd: hp.n_embd,
            shape: LinearShape {
                n_k: n_v,
                n_v,
                map: KHeadMap::Tiled,
            },
            rms_eps: hp.rms_eps,
            rule,
        },
        expert_ff: hp.experts.ff,
        shared_ff: hp.experts.shared_ff(),
        ctx,
    })
}

/// The stacks of the layers `inputs` describes for the residency machine's
/// file source: each routed layer's gate, up and down, their types read from
/// its tensors. Refused by name: a layer that holds some of its three stacks
/// but not all, and a description whose routed layers are not one run.
fn stacks(inputs: &PlanInputs) -> Result<PlanStacks, GpuError> {
    let run = hosted(&inputs.spec.layers).map_err(|e| shape(format!("the routed layers: {e}")))?;
    hosted_stacks(&inputs.model, "solar2 stacks", (run.len(), false))
}

/// The f32s of each store one checkpoint copies, in the layers' order: every
/// KDA layer's recurrent state and conv ring. The one derivation of what a
/// checkpoint holds, read by the sequence's `Checkpoints` and by
/// [`checkpoint_price`].
fn checkpoint_lens(spec: &ModelSpec, d: &Dims) -> Vec<usize> {
    spec.layers
        .iter()
        .filter(|s| Layer::of(s).mixer == MixerKind::DeltaRule)
        .flat_map(|_| [d.kda.shape.state_len(), d.kda.shape.ring_len()])
        .collect()
}

/// What one checkpoint of a load of `inputs`' file costs on the host and how
/// a prompt call spaces them ([`CHECKPOINT_EVERY`]): the figure every
/// sequence's `Checkpoints` is made with and a plan's host reserve prices.
pub fn checkpoint_price(inputs: &PlanInputs) -> Result<runtime::seqstate::Price, GpuError> {
    let d = dims_of(inputs, 1)?;
    Ok(checkpoint::price(
        &checkpoint_lens(&inputs.spec, &d),
        CHECKPOINT_EVERY,
    ))
}

impl Body {
    /// Card `card` of `plan`, which `inputs` made, resident: the layers'
    /// weights the plan puts on the card, the stores at the plan's `ctx_max`,
    /// the host tier over every routed layer and the embedding as a host row;
    /// the load's slot map for the model's life ([`Residency::Off`]). Refused
    /// by name (module doc).
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Solar2Model, GpuError> {
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

    /// [`Body::open_placed`] by a plan that counts `slots` resident sequences
    /// ([`PlanInputs::plan_with_slots`]). The body holds one: more are refused
    /// by name, and a plan that counts another number of sequences is refused
    /// at the load.
    ///
    /// Under `residency` ([`ResidencySpec`]) the host set also holds each
    /// layer's churn pool ([`GpuModel::load_placed_with`]) and the body starts
    /// the machine over its slot map once its pieces are sized
    /// ([`HostServed::start_residency`]); [`Residency::Off`] keeps the load's
    /// slot map for the model's life. The prompt batch's buffers are made too,
    /// for one unit a group ([`crate::prefill::set_prefill_group`] grows them).
    pub fn open_placed_slots(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
        slots: NonZeroUsize,
    ) -> Result<Solar2Model, GpuError> {
        if slots.get() > 1 {
            return Err(shape(format!(
                "{slots} resident sequences: the program holds one sequence's stores and has no \
                 slots to add"
            )));
        }
        let tiers = TierOpen::of_machine(plan.machine)?;
        refuse_tier_count(WHAT, tiers.len())?;
        let kinds: Vec<Layer> = inputs.spec.layers.iter().map(Layer::of).collect();
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
            |stream, _, layers, w| derive(stream, &kinds, layers, w),
            |gpu, file, w, set, glue| {
                Body::load(
                    gpu,
                    file,
                    w,
                    (plan, inputs, card, &tiers),
                    host,
                    (set, glue, residency),
                )
            },
        )
    }

    /// The body of card `card`: its layers' programs and values from the
    /// plan's description, the stores at the plan's `ctx_max`, the step's
    /// buffers, the slot map the plan's routed segments make (the stage
    /// card's and the expert tier cards' `tiers`), the routed blocks and the
    /// host tier over the routed run with the tier cards hung under it.
    /// `glue` is the load's residency side, whose machine the body starts once
    /// its pieces are sized.
    fn load(
        gpu: &Gpu,
        file: &Arc<Split>,
        w: &Weights,
        (plan, inputs, card, tiers): (&Plan<'_>, &PlanInputs, usize, &[TierOpen]),
        host: HostCfg,
        (set, glue, residency): (HostResidency, ResidencyGlue, Residency),
    ) -> Result<Body, GpuError> {
        let (hp, spec) = (&inputs.hp, &inputs.spec);
        gpu.context().bind_to_thread()?;
        let (layers, ctx) = stage_run(WHAT, plan, card, spec.layers.len())?;
        let dims = dims_of(inputs, ctx)?;
        let run = hosted(&spec.layers).map_err(|e| shape(e.to_string()))?;
        let tier_cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
        let map =
            SlotMap::of_plan_tiers(plan, card, &tier_cards, run.clone(), hp.experts.n_expert)?;
        let mut lcfg = Vec::with_capacity(spec.layers.len());
        for (l, s) in spec.layers.iter().enumerate() {
            lcfg.push(layer_cfg(l, s, &map)?);
        }
        let stores = (0..spec.layers.len())
            .map(|l| store(gpu, file, w, inputs, &dims, l))
            .collect::<Result<Vec<_>, _>>()?;
        let held: u64 = stores.iter().map(|s| s.bytes() as u64).sum();
        let planned = inputs.kv.bytes(layers, plan.ctx_max);
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
        let s = Scratch::new(stream, &dims, 1, stores.iter().flat_map(Store::sites))?;
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
                batch_rows: T_MAX,
            },
            CARD,
        )?;
        let hang = Hang {
            tiers,
            family: &tier::NODE_FAMILY,
            open: LegOpen {
                n_expert: hp.experts.n_expert,
                layers: spec.layers.len(),
                hidden: dims.embd,
                n_used: hp.experts.n_used,
                rows: LOAD_ROWS,
            },
            cols: T_MAX,
            card_dontneed: host.card_dontneed,
            bytes: bloomery_gpu::model::CardBytes::File,
        };
        let mut hybrid = hosted_hybrid_tiered(
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
            Some(&hang),
            |src| solar_host::routed_layers(src, hp, spec, run.clone()),
        )?;
        let table = EmbedTable::new(Arc::clone(file), &names::token_embd(), dims.embd)?;
        if table.n_vocab() != hp.n_vocab {
            return Err(shape(format!(
                "the embedding has {} rows, the file's vocabulary {}",
                table.n_vocab(),
                hp.n_vocab
            )));
        }
        prompt_walk::prepare_host(gpu, &mut hybrid, T_MAX)?;
        let batch = new_batch(gpu, &dims, &stores)?;
        let extra = Extra::new(gpu, &dims, &stores, 1)?;
        let units = BatchBufs::new(gpu, dims.embd, T_MAX, 1)?;
        let ckpt = Checkpoints::sequence(
            gpu.context(),
            checkpoint_lens(spec, &dims),
            ctx,
            CHECKPOINT_EVERY,
        )?;
        let mut body = Body {
            shell: TierShell {
                hybrid,
                glue,
                taps: None,
                embd: dims.embd,
                what: WHAT,
            },
            cfg: lcfg,
            stores,
            s,
            batch,
            units,
            slots,
            at: 0,
            k: Kernels::load(gpu)?,
            ffn,
            prefill: extra,
            embd: HostIn {
                table,
                row: vec![0.0; dims.embd],
                words: vec![0; WORD_ROWS + dims.kv_heads],
            },
            cur: StepInput { token: 0, pos: 0 },
            ckpt,
            residency,
            dims,
        };
        refuse_scratch_past(
            WHAT,
            plan,
            card,
            body.scratch_bytes(),
            "the step's buffers, the routed blocks' buffers and card experts, the mixers' \
             kernels, the host boundary, the slot map's copy",
        )?;
        let reserve = inputs
            .prompt_batch_bytes(plan.machine)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        refuse_reserve_past(
            WHAT,
            "prompt batch buffers",
            reserve,
            body.prompt_bytes() as u64,
            "the rows and wide forms, the units and kernels, the routed blocks' rows",
        )?;
        // Last: the machine frees each layer's spare slots, and every piece
        // above sized itself from the load's map, capacity = live.
        body.start_residency(gpu)?;
        Ok(body)
    }

    /// Positions every GQA store holds.
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

    /// The residency the body was loaded with.
    #[must_use]
    pub fn residency(&self) -> Residency {
        self.residency
    }

    /// The positions the stores hold.
    #[must_use]
    pub fn held(&self) -> u32 {
        self.at
    }

    /// Device bytes of the stores: every KDA layer's state, stamp and conv
    /// ring, every GQA layer's planes.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store::bytes).sum()
    }

    /// The bytes of every store the `pos` written positions fill, each named
    /// by its layer, in layer order: a KDA layer's state, stamp and conv ring,
    /// a GQA layer's K and V planes over the rows `h · ctx + [0, pos)` of every
    /// key head. A gate's digest input; blocking.
    pub fn store_bytes_at(&self, gpu: &Gpu, pos: u32) -> Result<Vec<(String, Vec<u8>)>, GpuError> {
        let stream = gpu.stream();
        let f32s = |b: &DeviceBuffer<f32>| -> Result<Vec<u8>, GpuError> {
            Ok(b.to_host_vec(stream)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        };
        let (ctx, kv_heads, pos) = (self.dims.ctx, self.dims.kv_heads, pos as usize);
        if pos > ctx {
            return Err(shape(format!("store bytes at {pos} positions of {ctx}")));
        }
        let plane = |t: &DeviceTensor<u16>| -> Result<Vec<u8>, GpuError> {
            let host = t.buf().to_host_vec(stream)?;
            let mut out = Vec::with_capacity(kv_heads * pos * HEAD * 2);
            for h in 0..kv_heads {
                let rows = &host[h * ctx * HEAD..(h * ctx + pos) * HEAD];
                out.extend(rows.iter().flat_map(|v| v.to_le_bytes()));
            }
            Ok(out)
        };
        let mut out = Vec::new();
        for (l, store) in self.stores.iter().enumerate() {
            match store {
                Store::Kda {
                    state, stamp, ring, ..
                } => {
                    out.push((format!("layer {l} kda state"), f32s(state)?));
                    let stamp = stamp.to_host_vec(stream)?;
                    out.push((
                        format!("layer {l} kda stamp"),
                        stamp.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ));
                    out.push((format!("layer {l} kda ring"), f32s(ring)?));
                }
                Store::Gqa(g) => {
                    out.push((format!("layer {l} gqa k"), plane(&g.kc)?));
                    out.push((format!("layer {l} gqa v"), plane(&g.vc)?));
                }
            }
        }
        Ok(out)
    }

    /// Device bytes of the step's side of the card beside the stores and the
    /// weights: the step's buffers, the routed blocks' buffers and card
    /// experts, the kernels' own, the host boundary and the slot map's copy.
    /// The term the plan sets aside as scratch.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.ffn.bytes()
            + self.k.bytes()
            + self.shell.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
    }

    /// Device bytes of the prompt batch's buffers: the rows' and the wide
    /// forms ([`Batch`]), the units and kernels ([`Extra`]) and the routed
    /// blocks' rows. The plan keeps [`PlanInputs::prompt_reserve`] for them
    /// out of the expert budget; a load past it is refused by name.
    #[must_use]
    pub fn prompt_bytes(&self) -> usize {
        self.batch.bytes() + self.prefill.bytes() + self.units.bytes() + self.ffn.batch_bytes()
    }

    /// Every store's and buffer's bytes on the card, the armed taps
    /// included.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.store_bytes()
            + self.scratch_bytes()
            + self.prompt_bytes()
            + self
                .shell
                .taps
                .as_ref()
                .map_or(0, |t| t.iter().map(DeviceBuffer::num_bytes).sum())
    }

    /// The token embedding's row of `token`, dequantized into `row` (one row
    /// of `n_embd` values); a token past the vocabulary is refused by name.
    pub fn embed(&self, token: u32, row: &mut [f32]) -> Result<(), GpuError> {
        self.embd.table.row_into(token, row)?;
        Ok(())
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        self.shell.hybrid()
    }

    /// The host tier, for a gate's instrument or test seam.
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.shell.hybrid_mut()
    }

    /// Keep every residency boundary's report from now on, at most `passes`
    /// between two takes.
    pub fn log_residency(&mut self, passes: usize) {
        self.shell.log_residency(passes);
    }

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.shell.residency_source()
    }

    /// The residency boundaries' reports since the last take, in order, each
    /// with the kind of the pass it ended.
    pub fn take_residency_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.shell.take_residency_passes()
    }

    /// Arm (or disarm) the per-layer taps: after each layer the chain copies
    /// its output into the layer's tap. The caller drops its captured chains
    /// first: a capture holds the tap copies it was taken with.
    pub fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        self.shell.set_taps(gpu, self.cfg.len(), on)
    }

    /// Every layer's output after the last step, `n_embd` a layer. Blocking;
    /// refused when the taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        self.shell.taps(gpu)
    }

    /// Every KDA layer's state and conv ring, in layer order: the list the
    /// checkpoints copy.
    fn copied(stores: &mut [Store]) -> Vec<&mut DeviceBuffer<f32>> {
        stores
            .iter_mut()
            .filter_map(|s| match s {
                Store::Kda { state, ring, .. } => Some([state, ring]),
                Store::Gqa(_) => None,
            })
            .flatten()
            .collect()
    }

    /// Every KDA layer's stamp set to the position `at` its lane stands at.
    fn restamp(stream: &CudaStream, stores: &mut [Store], at: u32) -> Result<(), GpuError> {
        for s in stores {
            if let Store::Kda { stamp, .. } = s {
                stamp.copy_from_host(stream, &[at])?;
            }
        }
        Ok(())
    }

    /// The checkpoints: their positions, their slots, what they have done.
    #[must_use]
    pub fn checkpoints(&self) -> &Checkpoints {
        &self.ckpt
    }

    /// What a cut to at most `n` positions of a model standing at `pos`
    /// keeps, and why: never past `pos`, and after a step that failed past
    /// its launch not `pos` either, since the stores hold one more.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        if self.at == pos {
            self.ckpt.kept(n, pos)
        } else {
            self.ckpt.kept(n.min(pos), self.at)
        }
    }

    /// A checkpoint at `pos`, the model's position, after any waiting cut:
    /// the KDA layers' stores copied to a host slot, or nothing where one
    /// stands. Refused by name when the stores hold other positions. Waits for
    /// the copies.
    pub fn checkpoint(&mut self, gpu: &Gpu, pos: u32) -> Result<Take, GpuError> {
        self.held_at(pos)?;
        self.apply_cut(gpu.stream())?;
        self.ckpt
            .take(gpu.stream(), pos, &mut Body::copied(&mut self.stores))
    }

    /// The waiting cut carried out: each KDA store from its checkpoint's slot
    /// or zeroed, every stamp set to the position it leaves. Nothing when no
    /// cut waits. Waits for the copies; the first device call after a cut
    /// runs it (`refresh`, a prompt batch).
    pub fn apply_cut(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if !self.ckpt.pending() {
            return Ok(());
        }
        self.ckpt
            .apply(stream, &mut Body::copied(&mut self.stores))?;
        Body::restamp(stream, &mut self.stores, self.at)
    }

    /// A cut to `pos`: nothing at the held position; the empty model at 0;
    /// else the checkpoint at `pos`, copied back at the next device call. Any
    /// other position is refused by name ([`Body::kept`] says what a cut
    /// keeps).
    pub fn cut(&mut self, pos: u32) -> Result<(), GpuError> {
        self.ckpt.cut(pos, self.at)?;
        self.at = pos;
        Ok(())
    }

    /// Refused by name unless the stores hold `pos` positions: a step at
    /// `pos` that failed after its inputs were on the card has already run
    /// the KDA layers' recurrence over it, and running it again would apply
    /// the position twice.
    fn held_at(&self, pos: u32) -> Result<(), GpuError> {
        match self.at {
            h if h == pos => Ok(()),
            h if h == pos.wrapping_add(1) => Err(shape(format!(
                "a call at position {pos} after a step at {pos} that failed past its launch: \
                 the stores hold {h} positions, and running the KDA layers over {pos} again \
                 would apply it twice (cut back to a checkpoint first)"
            ))),
            h => Err(shape(format!(
                "a call at position {pos} on stores holding {h} positions"
            ))),
        }
    }
}

/// Arm (or disarm) `m`'s per-layer taps ([`Body::taps`] reads them back), the
/// model first set to eager steps: a captured chain holds the tap copies it
/// was taken with, so arming under one would leave the taps unwritten and
/// disarming would free buffers its replays write. A model set back to graph
/// steps afterwards captures its chain with the taps.
pub fn set_taps(m: &mut Solar2Model, on: bool) -> Result<(), GpuError> {
    m.set_mode(StepMode::Eager);
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.set_taps(gpu, on)
}

/// Layer `l`'s programs from its description `s`: a plain residual over a
/// routed block and one of the two mixers, its routed block's row in the card
/// copy `map`'s.
fn layer_cfg(l: usize, s: &LayerSpec, map: &SlotMap) -> Result<LayerCfg, GpuError> {
    let kind = Layer::of(s);
    if kind.residual != ResidualKind::Plain {
        return Err(shape(format!(
            "layer {l}: hyper-connection streams; the solar2 chain is a plain residual"
        )));
    }
    if !matches!(kind.mixer, MixerKind::Gqa | MixerKind::DeltaRule) {
        return Err(shape(format!(
            "layer {l}: a {:?} mixer; this program runs GQA and KDA",
            kind.mixer
        )));
    }
    if kind.ffn != FfnKind::Moe {
        return Err(shape(format!(
            "layer {l}: a dense block; this program runs routed blocks only"
        )));
    }
    let row_off = map
        .row_offset(l)
        .ok_or_else(|| shape(format!("layer {l}: a routed layer with no slot-map row")))?;
    Ok(LayerCfg { kind, row_off })
}

/// Layer `l`'s store and typed sites, by its mixer kind: the sites' types
/// from the file, checked against the resident weights.
fn store(
    gpu: &Gpu,
    file: &Split,
    w: &Weights,
    inputs: &PlanInputs,
    d: &Dims,
    l: usize,
) -> Result<Store, GpuError> {
    let stream = gpu.stream();
    let site = |name: String, rows_k: (usize, usize), allowed: &[SiteTy]| {
        Site::of_file((file, w), WHAT, name, rows_k, allowed)
    };
    let embd = d.embd;
    let norm = site::vector(w, WHAT, names::attn_norm(l), embd)?;
    if matches!(inputs.spec.layers[l].mixer, Mixer::Gqa(_)) {
        let (q_rows, kv_rows) = (d.heads * HEAD, d.kv_heads * HEAD);
        return Ok(Store::Gqa(GqaLayer {
            q: site(names::attn_q(l), (q_rows, embd), KQUANT)?,
            k: site(names::attn_k(l), (kv_rows, embd), KQUANT)?,
            v: site(names::attn_v(l), (kv_rows, embd), KQUANT)?,
            o: site(names::attn_output(l), (embd, q_rows), KQUANT)?,
            gate: GqaGate::Proj(site(
                model::arch::solar2::names::attn_gate(l),
                (q_rows, embd),
                KQUANT,
            )?),
            kc: DeviceTensor::zeroed(stream, d.kv_heads * d.ctx, HEAD)?,
            vc: DeviceTensor::zeroed(stream, d.kv_heads * d.ctx, HEAD)?,
            norm,
        }));
    }
    let lin = d.kda.shape;
    let (qk, v) = (lin.n_k * HEAD, lin.n_v * HEAD);
    let sites = KdaSites {
        qkv: [
            site(names::attn_q(l), (qk, embd), KQUANT)?,
            site(names::attn_k(l), (qk, embd), KQUANT)?,
            site(names::attn_v(l), (v, embd), KQUANT)?,
        ],
        f_a: site(names::ssm_f_a(l), (HEAD, embd), KQUANT)?,
        g_a: site(names::ssm_g_a(l), (HEAD, embd), KQUANT)?,
        beta: site(names::ssm_beta(l), (lin.n_v, embd), KQUANT)?,
        f_b: site(names::ssm_f_b(l), (v, HEAD), LOW_RANK)?,
        g_b: site(names::ssm_g_b(l), (v, HEAD), LOW_RANK)?,
        out: site(names::attn_output(l), (embd, v), KQUANT)?,
        vecs: [
            norm,
            site::vector(w, WHAT, conv_joined(l), lin.channels() * CONV_TAPS)?,
            site::vector(w, WHAT, names::ssm_dt_bias(l), v)?,
            site::vector(w, WHAT, names::ssm_a(l), lin.n_v)?,
            site::vector(w, WHAT, names::ssm_norm(l), HEAD)?,
        ],
    };
    // Lane 0 stands at position 0.
    Ok(Store::Kda {
        sites,
        state: DeviceBuffer::zeroed(stream, lin.state_len())?,
        stamp: DeviceBuffer::from_host(stream, &[0u32])?,
        ring: DeviceBuffer::zeroed(stream, lin.ring_len())?,
    })
}

/// The prompt batch's buffers: the step's at [`T_MAX`] rows and the wide
/// forms of the rows the sites that read `n_embd` values read. Load-time
/// only.
fn new_batch(gpu: &Gpu, d: &Dims, stores: &[Store]) -> Result<Batch, GpuError> {
    let sites = || stores.iter().flat_map(Store::sites);
    let rows = MixerBufs::new(gpu.stream(), d, T_MAX, sites())?;
    Ok(Batch {
        rows,
        wide: wide_acts_at(gpu.stream(), sites(), (T_MAX, d.embd))?,
    })
}

impl ChainBody for Body {
    type Input = StepInput;
    type Host = Body;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::SolarOpen2
    }

    /// The step at `pos`: its embedding row read from the file. A position
    /// past the stores, a token past the vocabulary, or a position the stores
    /// do not stand at is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        if pos as usize >= self.dims.ctx {
            return Err(shape(format!(
                "a step at position {pos} in stores of {}",
                self.dims.ctx
            )));
        }
        self.held_at(pos)?;
        self.embd.table.row_into(token, &mut self.embd.row)?;
        self.cur = StepInput { token, pos };
        Ok(self.cur)
    }

    /// A waiting cut carried out, then the embedding row into `x`, the
    /// position, the live key count and the GQA word record: four
    /// host-to-device copies. Once they are sent the stores count the
    /// position: what follows launches.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        let pos = input.pos;
        self.apply_cut(stream)?;
        let ctx = self.dims.ctx;
        let words = &mut self.embd.words;
        words[WORD_POS] = pos;
        for (h, word) in words[WORD_ROWS..].iter_mut().enumerate() {
            *word = h
                .checked_mul(ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    shape(format!(
                        "row {h} · {ctx} + {pos} does not fit a u32 row word"
                    ))
                })?;
        }
        self.s.x.copy_from_host(stream, &self.embd.row)?;
        self.s.pos.copy_from_host(stream, &[pos])?;
        self.s.n_keys.copy_from_host(stream, &[pos + 1])?;
        self.s.words.copy_from_host(stream, words)?;
        self.at = pos + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let input = self.cur;
        crate::step::enqueue_step(gpu, w, self, input, head)
    }

    /// The host tier settled, every KDA store zeroed in place (a captured
    /// chain keeps their addresses) with its lane at position 0, every
    /// checkpoint dropped. The GQA planes keep their rows: a position is
    /// written before a later one reads it.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        self.shell.hybrid.settle(stream)?;
        for s in &mut self.stores {
            if let Store::Kda { state, ring, .. } = s {
                state.zero_async(stream)?;
                ring.zero_async(stream)?;
            }
        }
        Body::restamp(stream, &mut self.stores, 0)?;
        self.ckpt.clear();
        self.at = 0;
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every RMS norm's, the head's
    /// included.
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
    /// The residency machine stops before any field frees card or pinned
    /// memory ([`Hybrid::stop_swap`]): the host tier is the first field and
    /// stops it in its own drop, and this holds that order as the machine's
    /// rule rather than the field list's.
    fn drop(&mut self) {
        self.shell.hybrid.stop_swap();
    }
}

impl Rollback for Body {
    /// A cut to `pos` ([`Body::cut`]); the card work waits for the next
    /// device call.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.cut(pos)
    }
}
