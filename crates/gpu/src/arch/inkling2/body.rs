//! The Inkling 2 `ChainBody`: one card holds the model, its stores one set a visit, and the step
//! walks [`models::visits`](model::arch::models::visits) — the stored layers before the loop's
//! core once, the core's passes one after another, the layers after it once — running each
//! visit's stored layer from the weights the layer's visits share, over the visit's own stores
//! (llama.cpp PR #32695's `src/models/inkling2.cpp`, the graph's layer loop).
//!
//! A visit, in the reference's order:
//!
//! 1. at the first core visit of a pass: on pass 0 the prelude's blocks and stream are kept; on a
//!    later pass the blocks the previous pass banked past its first become the message bank, its
//!    last output the bank's last row, and the blocks and stream go back to the prelude's;
//! 2. on a core visit of a pass past the first, the message mix: four queries over the bank,
//!    rows q, ffn, k, v, the first job of the attention mix's launch pair (`attn_res_step`);
//! 3. the attention input: the normed mix of the banked blocks and the stream (one query
//!    windowed, three windowless; layer 0 reads the stream), the q, k and v controllers on it as
//!    one launch pair (`loop_control::enqueue_attn`) where the messages run, then the input norms
//!    (a windowed visit's one norm, three times where the messages ran; a windowless visit's q,
//!    k and v norms);
//! 4. the projections of q, k, v, the relative rows, the sink logits and the output gate (and a
//!    windowless visit's conv taps), one launch with the input norms (`norm_proj`) where q, k, v,
//!    rel and gate are Q8_0 and the sinks and taps F32, the k and v short convs on the visit's
//!    ring, the epilogues
//!    and the f16 append (`qk_clip`), the rel-bias attention with the per-query sink
//!    (`rel_bias`), the sigmoid output gate, the output projection and its post conv;
//! 5. the block join: a visit of a block's first layer banks the stream and the attention output
//!    is the new stream; any other adds the two;
//! 6. the feed-forward block in the three parts a placed body attaches its host tier between:
//!    the front ([`Walk::ffn_front`]: the mix, the ffn controller where the messages run, the
//!    ffn norm and the router), the shadow ([`Walk::ffn_shadow`]: the card's slots, every expert
//!    resident) and the back ([`Walk::ffn_back`]: the post conv and the stream's sum).
//!
//! After the last visit the head's mix of the banked blocks and the stream is the output head's
//! input; the head softcaps the logits and writes the padded columns past the unpadded
//! vocabulary at [`PAD_LOGIT`], so the argmax reads the vocabulary alone.
//!
//! A layer of other site types, and a step under the imatrix collector (which reads each site's
//! normed input), runs the input norms and the projections a launch each. Every mix of the step
//! is the step mix's launch pair. A prompt runs as the rows pass ([`pass`]): the same walk over
//! up to [`place::PASS_ROWS`] rows a launch, the stores left where that many steps leave them.

use super::route::{self, N_USED, Route, RouteOut};
use super::state::{self, StepIn, VisitStore, WORD_POS, WORD_RING, WORD_TOKEN};
use crate::arch::hybrid::{Acts, DecodeInput, GqaCx, Site};
use crate::attn_res::{AttnResKernels, AttnResNormedArgs, MAX_NQ, MAX_VECS};
use crate::attn_res_step::{AttnResStepKernels, MixStepScratch};
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::draft_taps::DraftTaps;
use crate::flash_gqa::{FlashGqaKernels, window_segments};
use crate::folder::Model;
use crate::gemm::Gemm32Kernels;
use crate::head::Head;
use crate::host::StepLeg;
use crate::host::placed::PlacedHost;
use crate::host::swap::Residency;
use crate::host::swap_source::{DEADLINE, LIVE_DELAY, ResidencySpec, hosted_stacks_named};
use crate::kquant::Act;
use crate::linear::conv_res::{ConvResArgs, ConvResKernels};
use crate::linear::out_gate::{OutGateArgs, OutGateKernels};
use crate::loop_control::{LoopControlArgs, LoopControlKernels, MAX_MODULES, scratch_len};
use crate::model::{ChainBody, GpuModel, HostServed, block_count, refuse_scratch_past, stage_run};
use crate::norm_proj::{self, F32Site, NormProjArgs, NormProjKernels, Q8Site};
use crate::norm_proj_bf16::{self, Bf16Site, NormProjBf16Args, NormProjBf16Kernels};
use crate::q38::{EmbedIo, Q38Kernels};
use crate::qk_clip::{QkClipArgs, QkClipKernels};
use crate::rel_bias::{REL, RelBiasArgs, RelBiasKernels};
use crate::routed_block::RouteIn;
use crate::site::{self, KGemvKernels, SiteTy};
use crate::tensor::{PartedBuffer, Window, WindowMut, copy_row};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use bloomery_levers::{ChurnHosting, HostCfg};
use chain::linear::Ring;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{Split, TensorInfo};
use model::arch::Arch;
use model::arch::inkling2::hparams::{Dialect, Hparams, Loop};
use model::arch::inkling2::names::{self, Ctrl, CtrlPart, Inp};
use model::arch::inkling2::place::{self, Inputs};
use model::arch::models::{LoopSpec, OutGate, Visit, visits};
use model::placement::accepts::{self, Program};
use model::placement::kernels::DownAct;
use model::placement::{CardFormat, Plan};
use placed::{LOAD_ROWS, Open, Placed, StepWalk, placed_map};
use runtime::layer::hosted;
use std::ops::Range;
use std::sync::Arc;

#[path = "pass.rs"]
pub(super) mod pass;
#[path = "placed.rs"]
pub(super) mod placed;

const WHAT: &str = "inkling2::Body::load";
const STEP: &str = "inkling2 step";
const OPEN: &str = "inkling2 GpuModel::open";

/// The mixes' and the controllers' fixed epsilon (`inkling2.cpp`'s `mix` and `control`).
const MIX_EPS: f32 = 1e-6;
/// A head's values: the query, key and value rows'.
const HEAD: usize = 128;
/// The query heads a key head the rel-bias entries take: the Mini's three or the Small's six.
const GROUPS: [usize; 2] = [3, 6];
/// The types the dense sites launch: the files' Q8_0 projections, the K-quant ones of the dynamic
/// mixes (Q4_K, Q5_K, Q6_K) and their f32 and BF16 ones.
pub(super) const DENSE_TYPES: &[SiteTy] = &[
    SiteTy::Q8_0,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::F32,
    SiteTy::Bf16,
];
/// The logit the head writes at a padded column: far under any softcapped logit, finite, so the
/// argmax reads the unpadded vocabulary and the finite argmax's check holds.
pub const PAD_LOGIT: f32 = -1.0e30;
/// What a refusal of a gate or up stack names.
const GATE_UP_ROLE: &str = "a resident fused-w13 half of NVFP4 planes, MXFP8 planes, Q8_0 \
     blocks or an i-quant or K-quant stack the common card leg's gate·up arm reads";
/// What a refusal of a down stack names.
const DOWN_ROLE: &str = "a resident down stack of NVFP4 planes, MXFP8 planes, Q8_0 blocks or a \
     32-value block type the common card leg's down arm reads (Q2_0, IQ4_NL, Q5_1)";
/// A layer's attention projections a BF16 file tensor of the name loads in its stored words for
/// ([`dense_card`], which names the token embedding and the head beside them).
const RAW_SITES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_rel.weight",
    "attn_gate.weight",
    "attn_output.weight",
];

/// The stages [`Body::arm_taps`] keeps of every visit, in this order, each named as the reference
/// names that visit's node (`<stage>-<slot>`): the q, k and v rows and the relative rows after
/// their epilogue, the attention's output before its gate, the attention block's output after
/// its post conv, the feed-forward's input, the routed sum, the feed-forward's output after its
/// post conv and the visit's output. A stage is [`Body::tap_widths`] values wide.
pub const TAP_STAGES: [&str; 10] = [
    "Qcur",
    "Kcur",
    "Vcur",
    "Rcur",
    "kqv_out",
    "attn_out",
    "ffn_inp",
    "ffn_moe_out",
    "ffn_out",
    "l_out",
];

/// The message rows of the four-query message mix: the reference's `(attn, mlp)` and `(k, v)`
/// row pairs in that order, so the q, ffn, k and v messages.
const MSG_Q: usize = 0;
const MSG_FFN: usize = 1;
const MSG_K: usize = 2;
const MSG_V: usize = 3;
/// The message row of each attention module (q, k, v).
const MSG_ATTN: [usize; MAX_MODULES] = [MSG_Q, MSG_K, MSG_V];

/// A core layer's message controllers: each module's down, gain and forget rows (q, k, v, ffn),
/// and the four message queries of the layer's rows of the two model tables, staged at load in
/// the order [`MSG_Q`], [`MSG_FFN`], [`MSG_K`], [`MSG_V`].
struct Controls {
    w: [[String; 3]; 4],
    queries: DeviceBuffer<f32>,
}

/// One stored layer's resident pieces; its visits share them.
struct Layer {
    windowed: bool,
    q: Site,
    k: Site,
    v: Site,
    rel: Site,
    sink: Site,
    gate: Site,
    o: Site,
    /// A windowless layer's k and v conv-tap projections.
    taps: Option<[Site; 2]>,
    /// The q, k and v input norms: the windowed layer's one norm three times.
    norms: [String; 3],
    /// The attention mix's query rows and their count; layer 0 reads the stream.
    attn_query: Option<(String, usize)>,
    ffn_query: String,
    ffn_norm: String,
    q_clip: String,
    rel_clip: String,
    /// The bias table as the rel-bias entries read it, `[REL][extent]`.
    bias: DeviceBuffer<f32>,
    extent: usize,
    /// The k, v, attention-post and ffn-post conv filters.
    convs: [String; 4],
    router: String,
    probs_b: String,
    /// The routed stacks on the card (gate half, up half, down), `None` on a placed load: its
    /// experts are the host's ([`placed`]).
    stacks: Option<[String; 3]>,
    ctrl: Option<Controls>,
    /// The input norms and the projections run as one launch ([`norm_proj_fits`]).
    norm_proj: bool,
}

impl Layer {
    fn sites(&self) -> Vec<&Site> {
        let mut out = vec![
            &self.q, &self.k, &self.v, &self.rel, &self.sink, &self.gate, &self.o,
        ];
        if let Some(t) = &self.taps {
            out.extend(t.iter());
        }
        out
    }
}

/// The widths of a visit: the model's and the attention's, every layer's the same (refused at
/// load otherwise).
#[derive(Clone, Copy)]
struct Dims {
    n: usize,
    n_head: usize,
    n_kv: usize,
    taps: usize,
    window: usize,
    ff: usize,
    n_expert: usize,
    rank: usize,
    /// The conv ring's channels and the four convs' bases in it.
    ch: usize,
    bases: [usize; 4],
}

/// The kernels the step launches beside the `Gpu`'s own.
struct Kernels {
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash: FlashGqaKernels,
    q38: Q38Kernels,
    mix: AttnResKernels,
    step_mix: AttnResStepKernels,
    proj: NormProjKernels,
    proj_bf16: NormProjBf16Kernels,
    ctrl: LoopControlKernels,
    conv: ConvResKernels,
    clip: QkClipKernels,
    rel: RelBiasKernels,
    out_gate: OutGateKernels,
    router: Route,
    leg: CardLeg,
}

impl Kernels {
    fn load(gpu: &Gpu, n_expert: usize) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            q38: Q38Kernels::load(ctx)?,
            mix: AttnResKernels::load(ctx)?,
            step_mix: AttnResStepKernels::load(ctx)?,
            proj: NormProjKernels::load(ctx)?,
            proj_bf16: NormProjBf16Kernels::load(ctx)?,
            ctrl: LoopControlKernels::load(ctx)?,
            conv: ConvResKernels::load(ctx)?,
            clip: QkClipKernels::load(ctx)?,
            rel: RelBiasKernels::load(ctx)?,
            out_gate: OutGateKernels::load(ctx)?,
            router: Route::load(ctx, n_expert)?,
            leg: CardLeg::load(gpu)?,
        })
    }
}

/// The step's row buffers, one row each.
struct Scratch {
    emb: DeviceBuffer<f32>,
    /// The running stream.
    x: DeviceBuffer<f32>,
    /// The attention mix's rows (three on a windowless visit).
    mix: PartedBuffer<f32, 3>,
    /// The message mix's four rows.
    msg: PartedBuffer<f32, MAX_NQ>,
    /// The q, k and v controllers' outputs, and the controllers' down rows.
    ctl: [DeviceBuffer<f32>; MAX_MODULES],
    h: DeviceBuffer<f32>,
    /// The q, k and v inputs normed.
    xn: [DeviceBuffer<f32>; 3],
    q: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    k_raw: DeviceBuffer<f32>,
    v_raw: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    k_taps: DeviceBuffer<f32>,
    v_taps: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    fmix: DeviceBuffer<f32>,
    fctl: DeviceBuffer<f32>,
    fxn: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    /// The banked block inputs, `n` apart, and the message bank and its last row.
    banks: DeviceBuffer<f32>,
    msg_bank: DeviceBuffer<f32>,
    bank_last: DeviceBuffer<f32>,
    /// The prelude's stream, which every pass past the first restarts from.
    prelude: DeviceBuffer<f32>,
    /// The head's padded columns' constant.
    pad: DeviceBuffer<f32>,
    route: RouteOut,
    leg: LegBufs,
    /// The step mixes' statistics and tickets.
    mix_step: MixStepScratch,
}

impl Scratch {
    fn new(stream: &CudaStream, d: &Dims, pad: usize, router: &Route) -> Result<Scratch, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let n = d.n;
        let (q, kv, r) = (d.n_head * HEAD, d.n_kv * HEAD, d.n_head * REL);
        let segs = window_segments(d.window).max(window_segments(0));
        let leg = LegShape {
            cap: 1,
            slots_per_col: N_USED,
            n,
            ff: d.ff,
        };
        Ok(Scratch {
            emb: z(n)?,
            x: z(n)?,
            mix: PartedBuffer::zeroed(stream, [n; 3])?,
            msg: PartedBuffer::zeroed(stream, [n; MAX_NQ])?,
            ctl: [z(n)?, z(n)?, z(n)?],
            h: z(scratch_len(MAX_MODULES, 1, d.rank.max(1)))?,
            xn: [z(n)?, z(n)?, z(n)?],
            q: z(q)?,
            r: z(r)?,
            sinks: z(d.n_head)?,
            gate: z(q)?,
            k_raw: z(kv)?,
            v_raw: z(kv)?,
            k: z(kv)?,
            v: z(kv)?,
            k_taps: z(d.n_kv * d.taps)?,
            v_taps: z(d.n_kv * d.taps)?,
            attn: z(q)?,
            part_v: z(d.n_head * segs * HEAD)?,
            part_ms: z(d.n_head * segs * 2)?,
            gated: z(q)?,
            o: z(n)?,
            attn_out: z(n)?,
            ffn_inp: z(n)?,
            fmix: z(n)?,
            fctl: z(n)?,
            fxn: z(n)?,
            moe: z(n)?,
            ffn_out: z(n)?,
            banks: z(MAX_VECS * n)?,
            msg_bank: z(MAX_VECS * n)?,
            bank_last: z(n)?,
            prelude: z(n)?,
            pad: DeviceBuffer::from_host(
                stream,
                &vec![PAD_LOGIT; (pad * pass::rows::VERIFY_ROWS).max(1)],
            )?,
            route: router.new_out(stream)?,
            leg: LegBufs::for_down(stream, leg, DownAct::Q8Blocks32)?,
            mix_step: MixStepScratch::new(stream, n)?,
        })
    }

    fn bytes(&self) -> usize {
        let bufs = [
            &self.emb,
            &self.x,
            &self.h,
            &self.q,
            &self.r,
            &self.sinks,
            &self.gate,
            &self.k_raw,
            &self.v_raw,
            &self.k,
            &self.v,
            &self.k_taps,
            &self.v_taps,
            &self.attn,
            &self.part_v,
            &self.part_ms,
            &self.gated,
            &self.o,
            &self.attn_out,
            &self.ffn_inp,
            &self.fmix,
            &self.fctl,
            &self.fxn,
            &self.moe,
            &self.ffn_out,
            &self.banks,
            &self.msg_bank,
            &self.bank_last,
            &self.prelude,
            &self.pad,
        ];
        bufs.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.mix.whole().num_bytes()
            + self.msg.whole().num_bytes()
            + self
                .ctl
                .iter()
                .chain(&self.xn)
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.leg.bytes()
            + self.mix_step.bytes()
            + self.route.bytes()
    }
}

/// The Inkling 2 chain over one sequence on one card.
pub struct Body {
    hp: Hparams,
    d: Dims,
    visits: Vec<Visit>,
    layers: Vec<Layer>,
    stores: Vec<VisitStore>,
    k: Kernels,
    acts: Acts,
    s: Scratch,
    step_in: StepIn,
    /// The rows of the windowed visits' rings, 0 when none rings.
    ring: u32,
    /// Whether the visits of a pass past the first run their messages: always on a serving
    /// load; [`Body::set_messages`] turns them off for a gate's clause.
    messages: bool,
    /// Each visit's [`TAP_STAGES`] at the step's position, armed by [`Body::arm_taps`].
    taps: Option<DeviceBuffer<f32>>,
    /// A block draft's taps ([`GpuModel::set_draft_taps`]): the rows entering the visits it
    /// names, and the visit count for the final normed row.
    draft: Option<DraftTaps>,
    /// Each visit's routed ids and weights from the host, armed by [`Body::force_picks`].
    forced: Option<Forced>,
    /// The position the stores stand at: the next step's or pass's.
    pub(super) held: u32,
    /// One past the highest position any step or pass wrote since the last reset. A rollback
    /// leaves it where it is, so the depth of a later one is measured from it
    /// ([`runtime::stores::rollback_floor`]).
    end: u32,
    ctx: usize,
    /// The prompt pass's buffers and kernels ([`pass`]).
    pass: Option<pass::Pass>,
    /// A verify's own buffers beside the pass's ([`pass::rows`]).
    verify: Option<pass::rows::VerifyBufs>,
    /// The host tier and the host embedding of a placed load ([`placed`]); `None` on the
    /// whole-card load.
    placed: Option<Placed>,
}

impl GpuModel<Body> {
    /// Tap the rows entering `visits` (strictly rising, each at most the visit count; the visit
    /// count is the final normed row) from the next step on, one row a tap; empty stops tapping.
    /// The captured step is dropped first, so the next one records the copies. Refused by name
    /// on a placed load: its step walk carries no taps.
    pub fn set_draft_taps(&mut self, visits: &[usize]) -> Result<(), GpuError> {
        const WHAT_S: &str = "inkling2::set_draft_taps";
        self.drop_captures();
        let (gpu, _, body) = self.body_parts(WHAT_S)?;
        if body.placed.is_some() && !visits.is_empty() {
            return Err(GpuError::state(
                WHAT_S,
                "a whole-card load (a placed step carries no taps)",
            ));
        }
        body.draft = None;
        if visits.is_empty() {
            return Ok(());
        }
        let free = gpu.alloc_room()?;
        let mut t = DraftTaps::new(
            gpu.stream(),
            visits,
            body.visits.len() + 1,
            (body.d.n, 1),
            free,
        )?;
        t.plan(1);
        body.draft = Some(t);
        Ok(())
    }

    /// The last step's tapped row, the taps in the order they were named. Blocking; refused by
    /// name with no taps set.
    pub fn draft_taps(&mut self) -> Result<Vec<f32>, GpuError> {
        const WHAT_T: &str = "inkling2::draft_taps";
        let (gpu, _, body) = self.body_parts(WHAT_T)?;
        body.draft
            .as_ref()
            .ok_or(GpuError::state(WHAT_T, "taps set (set_draft_taps)"))?
            .read(gpu.stream(), 0, 1)
    }

    /// The whole model of `file` (a GGUF split, or a checkpoint folder's header and part source)
    /// resident on `gpu`, plus the output head: the dense sites and the f32 tables as the file holds
    /// them, a BF16 attention projection, token embedding and head in their stored words
    /// ([`dense_card`]), each routed layer's fused gate·up stack split at upload into its gate and
    /// up halves (a Q8_0 layer's stacks in the leg's blocks first), the visits' stores of `ctx`
    /// positions (a windowed visit's ring of its window), the step scratch. The model takes the
    /// file and closes it once the weights are resident.
    ///
    /// Refused by name: a file of llama.cpp master's loop dialect, a router other than 128
    /// experts and the top 8, layers whose widths differ, a group other than three query heads a
    /// key head, a file whose tensors and prompt pass ([`place::pass_bytes`]) exceed the card's
    /// free bytes.
    pub fn open(gpu: Gpu, file: impl Into<Model>, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let (file, origin) = file.into().into_whole();
        let n_layers = block_count(&file, OPEN)?;
        let hp = Hparams::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
        let reserve = place::pass_bytes(&pass::shape_of(&hp, &file)?, place::pass_rows(ctx));
        let (free, total) = gpu.mem_info()?;
        let bytes: u64 = file.iter_tensors().map(|(_, t)| t.nbytes).sum();
        if usize::try_from(bytes + reserve).map_or(true, |b| b > free) {
            return Err(GpuError::shape(
                OPEN,
                format!(
                    "the file's tensors are {bytes} B and the prompt pass {reserve} B; the card \
                     has {free} of {total} B free: the body holds the whole model on one card"
                ),
            ));
        }
        GpuModel::load_blocks_from(
            gpu,
            &file,
            (origin.bytes(), &dense_card),
            ctx,
            0..n_layers,
            true,
            |gpu, w| {
                let stacks = prepare_stacks(gpu.stream(), &file, w, &hp)?;
                Body::load(gpu, &file, w, (hp, ctx), (Some(stacks), None))
            },
        )
    }
}

impl GpuModel<Body> {
    /// Card `card` of `plan`, which `inputs` made, resident: the dense trunk, the stores at the
    /// plan's `ctx_max`, the host tier over every routed expert (the card leg runs no IQ stack,
    /// so the plan's card counts are 0, refused by name otherwise) and the embedding as a host
    /// row ([`placed`]). The load runs no residency machine. The prompt is stepped one id at a
    /// time. Refused by name: a plan of other than one stage card running every layer, a plan
    /// whose head is on no card, an expert tier card, a looped file, experts on the card, and
    /// scratch past the plan's term for it.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &Inputs,
        card: usize,
        host: HostCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        const PLAN: &str = "inkling2::Body::open_placed";
        let hp = &inputs.hp;
        let (_, ctx) = stage_run(PLAN, plan, card, hp.n_layer)?;
        plan.head_site(card).map_err(|e| GpuError::plan(PLAN, e))?;
        let run = hosted(&inputs.spec.layers)
            .map_err(|e| GpuError::shape(PLAN, format!("the routed layers: {e}")))?;
        let spec = ResidencySpec {
            lever: Residency::Off,
            pool: ChurnHosting::Hosted,
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k: hp.n_used[0],
            max_rows: LOAD_ROWS,
            stacks: Arc::new(hosted_stacks_named(
                &inputs.model,
                PLAN,
                (run.len(), true),
                fused_stack_names,
            )?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |_, _, _, _| Ok(()),
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
                    Body::load(gpu, file, w, (hp.clone(), ctx), (None, Some((open, card))))?;
                let p = body.placed.as_mut().ok_or(GpuError::shape(
                    PLAN,
                    "a placed load without its placed side",
                ))?;
                p.prepare_step_union()?;
                refuse_scratch_past(
                    PLAN,
                    plan,
                    card,
                    body.scratch_bytes(),
                    "the step's buffers, the host boundary, the slot map's copy",
                )?;
                let p = body.placed.as_mut().ok_or(GpuError::shape(
                    PLAN,
                    "a placed load without its placed side",
                ))?;
                p.start_residency(gpu)?;
                Ok(body)
            },
        )
    }
}

/// Layer `l`'s (gate, up, down) file names: the fused gate·up tensor twice (its two halves), the
/// down stack.
fn fused_stack_names(l: usize) -> [String; 3] {
    let fused = names::ffn_gate_up_exps(l);
    [fused.clone(), fused, names::ffn_down_exps(l)]
}

/// The card format of the file tensor `t`: a BF16 attention projection ([`RAW_SITES`]), token
/// embedding or head in its stored words (`Bf16Raw`, two bytes a weight, which the BF16 site,
/// `norm_proj_bf16`, embedding and head kernels read); every other tensor in its type's own (the
/// BF16 sink and conv projections widened to the F32 the fused launch reads).
fn dense_card(t: &TensorInfo) -> Option<CardFormat> {
    // A routed stack of a type with no default layout (the i-quants IQ3_XXS and IQ3_S, IQ4_NL, Q2_0,
    // MXFP4) loads in the file's blocks, the form the common leg reads every such stack in.
    if t.name.ends_with("_exps.weight") && CardFormat::of(t.ty).is_none() {
        return Some(accepts::leg_card_format(t.ty));
    }
    let raw = t.name == "token_embd.weight"
        || t.name == "output.weight"
        || t.name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(_, site)| RAW_SITES.contains(&site));
    (raw && t.ty == gguf::GgmlType::BF16).then_some(CardFormat::Bf16Raw)
}

/// `name`'s site of `rows` rows of `k`, its type the resident weight's (one of [`DENSE_TYPES`]).
fn site_of(w: &Weights, name: String, (rows, k): (usize, usize)) -> Result<Site, GpuError> {
    let ty = site::resident_site(w, WHAT, &name, (rows, k), DENSE_TYPES)?;
    Ok(Site::new(name, ty, k))
}

/// Whether `layer`'s input norms and projections run as one fused launch: q, k, v, rel and gate
/// all Q8_0 (`norm_proj`) or all BF16 words (`norm_proj_bf16`), the sinks and the taps F32, the
/// input a width the entry takes and every site's rows a multiple of its block's.
fn norm_proj_fits(layer: &Layer, d: &Dims) -> bool {
    let five = [&layer.q, &layer.k, &layer.v, &layer.rel, &layer.gate];
    let (max_k, block) = match five[0].ty {
        SiteTy::Q8_0 => (norm_proj::MAX_K, norm_proj::ROWS),
        SiteTy::Bf16 => (norm_proj_bf16::MAX_K, norm_proj_bf16::ROWS),
        _ => return false,
    };
    let same = five.iter().all(|s| s.ty == five[0].ty);
    let f32 = std::iter::once(&layer.sink)
        .chain(layer.taps.iter().flatten())
        .all(|s| s.ty == SiteTy::F32);
    let rows = [
        d.n_head * HEAD,
        d.n_kv * HEAD,
        d.n_head * REL,
        d.n_head,
        d.n_kv * d.taps,
    ];
    same && f32
        && d.n.is_multiple_of(32)
        && d.n <= max_k
        && rows.iter().all(|r| r.is_multiple_of(block))
}

/// Site `site`'s Q8_0 planes and its output `y`, for a `norm_proj` launch.
fn q8_site<'a>(
    w: &'a Weights,
    site: &Site,
    y: &'a mut DeviceBuffer<f32>,
) -> Result<Q8Site<'a>, GpuError> {
    let (qs, d) = w.q8_planes(STEP, &site.name)?;
    Ok(Q8Site { qs, d, y })
}

/// Site `site`'s BF16 words and its output `y`, for a `norm_proj_bf16` launch.
fn bf16_site<'a>(
    w: &'a Weights,
    site: &Site,
    y: &'a mut DeviceBuffer<f32>,
) -> Result<Bf16Site<'a>, GpuError> {
    Ok(Bf16Site {
        w: w.bf16(STEP, &site.name)?,
        y,
    })
}

/// `name`, an f32 vector of `len` values the weights hold, or a named refusal.
fn vector(w: &Weights, name: String, len: usize) -> Result<String, GpuError> {
    let got = w.f32_buf(WHAT, &name)?.len();
    if got != len {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} holds {got} values; the step reads {len}"),
        ));
    }
    Ok(name)
}

/// Layer `l`'s bias table `[extent][REL]` as the file lays it (`{REL, extent}` in ggml's order),
/// transposed to the rel-bias entries' `[REL][extent]`. Load-time only (one readback).
fn bias_table(
    stream: &CudaStream,
    w: &Weights,
    l: usize,
    extent: usize,
) -> Result<DeviceBuffer<f32>, GpuError> {
    let name = names::attn_rel_bias(l);
    let file = w.f32_buf(WHAT, &name)?.to_host_vec(stream)?;
    if file.len() != extent * REL {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name} holds {} values; the bias bank is {REL} x {extent}",
                file.len()
            ),
        ));
    }
    let mut t = vec![0.0f32; extent * REL];
    for (j, row) in file.as_chunks::<REL>().0.iter().enumerate() {
        for (dd, v) in row.iter().enumerate() {
            t[dd * extent + j] = *v;
        }
    }
    Ok(DeviceBuffer::from_host(stream, &t)?)
}

/// Core layer `l`'s controllers (the unsloth dialect's stems) and its four message queries,
/// staged from rows `2·(l − core_start)` and the next of the two model tables.
fn controls(
    stream: &CudaStream,
    w: &Weights,
    (l, lp, n): (usize, Loop, usize),
) -> Result<Controls, GpuError> {
    let module = |c: Ctrl| -> Result<[String; 3], GpuError> {
        let [d, g, f] = [CtrlPart::Down, CtrlPart::Gain, CtrlPart::Forget]
            .map(|p| names::loop_ctrl_pr(l, c, p));
        Ok([
            vector(w, d, lp.gate_rank * 2 * n)?,
            vector(w, g, n * lp.gate_rank)?,
            vector(w, f, n * lp.gate_rank)?,
        ])
    };
    let core = lp.core_end - lp.core_start;
    let res = vector(w, names::LOOP_RES_QUERY.to_owned(), 2 * core * n)?;
    let kv = vector(w, names::LOOP_KV_RES_QUERY.to_owned(), 2 * core * n)?;
    let row = 2 * (l - lp.core_start);
    let mut queries = DeviceBuffer::<f32>::zeroed(stream, MAX_NQ * n)?;
    for (at, (table, r)) in [(&res, row), (&res, row + 1), (&kv, row), (&kv, row + 1)]
        .into_iter()
        .enumerate()
    {
        let src = Window::<f32>::of(w.f32_buf(WHAT, table)?, r * n * 4, n)?;
        let mut dst = WindowMut::<f32>::of_mut(&mut queries, at * n * 4, n)?;
        copy_row(stream, &src, &mut dst, n)?;
    }
    Ok(Controls {
        w: [
            module(Ctrl::Attn)?,
            module(Ctrl::K)?,
            module(Ctrl::V)?,
            module(Ctrl::Ffn)?,
        ],
        queries,
    })
}

/// Each layer's routed stacks as the card leg reads them, the whole-card load's: a Q8_0 layer's
/// in the leg's blocks, then the fused gate·up split into its halves (the per-expert scale
/// shared, the split exact). The names of (gate half, up half, down) a layer, each checked
/// against the card leg's arms.
fn prepare_stacks(
    stream: &CudaStream,
    file: &Split,
    w: &mut Weights,
    hp: &Hparams,
) -> Result<Vec<[String; 3]>, GpuError> {
    let arms = accepts::leg_arms(Program::Inkling2Body)
        .ok_or_else(|| GpuError::shape(WHAT, "the inkling2 program names no card leg"))?;
    let ff = hp.expert_ff[0];
    let mut out = Vec::with_capacity(hp.n_layer);
    for l in 0..hp.n_layer {
        let fused = names::ffn_gate_up_exps(l);
        let down = format!("blk.{l}.ffn_down_exps.weight");
        let q8 = file
            .find(&fused)
            .is_some_and(|(_, t)| t.ty == gguf::GgmlType::Q8_0);
        if q8 {
            w.relayout(stream, file, &fused, CardFormat::KQuant)?;
            w.relayout(stream, file, &down, CardFormat::KQuant)?;
        }
        let (gate_half, up_half) = w.split_fused_w13(stream, &fused, ff)?;
        LegStacks::of_arms(
            w,
            [gate_half.as_str(), up_half.as_str(), down.as_str()],
            arms,
            (GATE_UP_ROLE, DOWN_ROLE),
        )?;
        out.push([gate_half, up_half, down]);
    }
    Ok(out)
}

impl Body {
    #[allow(
        clippy::too_many_lines,
        reason = "one load: the refusals, then each stored layer's resident pieces in file order"
    )]
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        (hp, ctx): (Hparams, usize),
        (stacks, placed): (Option<Vec<[String; 3]>>, Option<(Open<'_>, usize)>),
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        if hp.dialect != Dialect::Unsloth {
            return Err(GpuError::shape(
                WHAT,
                "a file of llama.cpp master's inkling2_loop dialect: the body reads the unsloth \
                 files' stems (blk.N.loop_{attn,k,v,mlp}_*)",
            ));
        }
        let n = hp.n_embd;
        let (n_head, n_kv) = (hp.n_head[0], hp.n_head_kv[0]);
        let same = |v: &[usize]| v.iter().all(|x| *x == v[0]);
        if !(same(&hp.n_head) && same(&hp.n_head_kv) && same(&hp.expert_ff) && same(&hp.n_used)) {
            return Err(GpuError::shape(
                WHAT,
                "layers of differing head counts, expert widths or picks: one step shape serves \
                 every visit",
            ));
        }
        let group_ok = n_kv > 0 && GROUPS.iter().any(|g| n_head == g * n_kv);
        if hp.head_dim != HEAD || !group_ok || hp.d_rel != REL {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "heads {n_head} over {n_kv} of {} with relative rows of {}: the rel-bias \
                     entries take {GROUPS:?} query heads a key head of {HEAD} and rows of {REL}",
                    hp.head_dim, hp.d_rel
                ),
            ));
        }
        if hp.n_used[0] != N_USED {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the top {} of {} experts: the body's routers take the top {N_USED}",
                    hp.n_used[0], hp.n_expert
                ),
            ));
        }
        let k = Kernels::load(gpu, hp.n_expert)?;
        let core = hp.loop_.filter(|lp| lp.passes > 1);
        let loop_spec = hp.loop_.map(|lp| LoopSpec {
            core_start: lp.core_start as u32,
            core_end: lp.core_end as u32,
            passes: lp.passes as u32,
            gate_rank: lp.gate_rank as u32,
        });
        let visits = visits(hp.n_layer, loop_spec);
        if placed.is_some() && (core.is_some() || visits.len() != hp.n_layer) {
            return Err(GpuError::shape(
                WHAT,
                "a looped file on a placed load: the placed walk's visit is its layer",
            ));
        }
        if visits.len() >= crate::fault::LAYER_HEAD as usize {
            return Err(GpuError::shape(
                WHAT,
                format!("{} visits: the fault word names fewer", visits.len()),
            ));
        }
        let most = place::banked_most(&hp);
        if most + 1 > MAX_VECS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{most} banked blocks and the stream: the mixes take at most {MAX_VECS} vectors"
                ),
            ));
        }
        let kvw = n_kv * HEAD;
        let d = Dims {
            n,
            n_head,
            n_kv,
            taps: hp.conv,
            window: hp.swa,
            ff: hp.expert_ff[0],
            n_expert: hp.n_expert,
            rank: hp.loop_.map_or(0, |lp| lp.gate_rank),
            ch: 2 * kvw + 2 * n,
            bases: [0, kvw, 2 * kvw, 2 * kvw + n],
        };
        let mut stacks = stacks.map(Vec::into_iter);
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let windowed = hp.windowed[l];
            let blk = |s: &str| format!("blk.{l}.{s}");
            let stacks = match stacks.as_mut() {
                None => None,
                Some(it) => Some(
                    it.next()
                        .ok_or_else(|| GpuError::shape(WHAT, "fewer stack triples than layers"))?,
                ),
            };
            let extent = if windowed { hp.swa } else { hp.extent };
            let norms = if windowed {
                let norm = vector(w, blk("attn_norm.weight"), n)?;
                [norm.clone(), norm.clone(), norm]
            } else {
                [Inp::NormQ, Inp::NormK, Inp::NormV].map(|p| names::inp(l, p))
            };
            for norm in &norms {
                vector(w, norm.clone(), n)?;
            }
            let taps = if windowed {
                None
            } else {
                Some([
                    site_of(w, names::inp(l, Inp::ConvK), (n_kv * d.taps, n))?,
                    site_of(w, names::inp(l, Inp::ConvV), (n_kv * d.taps, n))?,
                ])
            };
            let ctrl = core
                .filter(|lp| (lp.core_start..lp.core_end).contains(&l))
                .map(|lp| controls(stream, w, (l, lp, n)))
                .transpose()?;
            let mut layer = Layer {
                windowed,
                q: site_of(w, blk("attn_q.weight"), (n_head * HEAD, n))?,
                k: site_of(w, blk("attn_k.weight"), (kvw, n))?,
                v: site_of(w, blk("attn_v.weight"), (kvw, n))?,
                rel: site_of(w, names::attn_rel(l), (n_head * REL, n))?,
                sink: site_of(w, names::attn_sink_proj(l), (n_head, n))?,
                gate: site_of(w, blk("attn_gate.weight"), (n_head * HEAD, n))?,
                o: site_of(w, blk("attn_output.weight"), (n, n_head * HEAD))?,
                taps,
                norms,
                attn_query: if l == 0 {
                    None
                } else {
                    let nq = if windowed { 1 } else { 3 };
                    Some((vector(w, names::attn_res_query(l), nq * n)?, nq))
                },
                ffn_query: vector(w, names::ffn_res_query(l), n)?,
                ffn_norm: vector(w, blk("ffn_norm.weight"), n)?,
                q_clip: vector(w, names::attn_q_clip(l), n_head * HEAD)?,
                rel_clip: vector(w, names::attn_rel_clip(l), n_head * REL)?,
                bias: bias_table(stream, w, l, extent)?,
                extent,
                convs: [
                    vector(w, names::attn_k_conv(l), kvw * d.taps)?,
                    vector(w, names::attn_v_conv(l), kvw * d.taps)?,
                    vector(w, names::attn_post_conv(l), n * d.taps)?,
                    vector(w, names::ffn_post_conv(l), n * d.taps)?,
                ],
                router: blk("ffn_gate_inp.weight"),
                probs_b: vector(w, blk("exp_probs_b.bias"), hp.n_expert)?,
                stacks,
                ctrl,
                norm_proj: false,
            };
            layer.norm_proj = norm_proj_fits(&layer, &d);
            layers.push(layer);
            w.f32_tensor(WHAT, &layers[l].router)?;
        }
        vector(w, names::TOKEN_EMBD_NORM.to_owned(), n)?;
        vector(w, names::OUTPUT_RES_QUERY.to_owned(), n)?;
        let shapes = state::shapes(
            visits.iter().map(|v| hp.windowed[v.layer as usize]),
            (n_kv, HEAD),
            (hp.swa, ctx),
            (Ring::new(d.taps), d.ch),
        );
        let stores = shapes
            .into_iter()
            .map(|s| VisitStore::new(stream, s))
            .collect::<Result<Vec<_>, _>>()?;
        let ring = match stores.iter().find(|s| s.shape.ring) {
            Some(s) => {
                u32::try_from(s.shape.rows).map_err(|e| GpuError::shape(WHAT, e.to_string()))?
            }
            None => 0,
        };
        let acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        let pad = hp.n_vocab - hp.n_vocab_unpadded;
        // The prompt pass and the verify run on the card's own experts: a placed load has neither
        // (its prompt is stepped).
        let (pass, verify) = if placed.is_some() {
            (None, None)
        } else {
            (
                Some(pass::Pass::new(
                    gpu,
                    (&pass::shape_of(&hp, file)?, place::pass_rows(ctx)),
                    &layers,
                )?),
                Some(pass::rows::VerifyBufs::new(stream, &d)?),
            )
        };
        let placed = placed
            .map(|(open, card)| {
                let map = placed_map(&open, card, &hp)?;
                Placed::new(gpu, map, open)
            })
            .transpose()?;
        Ok(Body {
            pass,
            verify,
            s: Scratch::new(stream, &d, pad, &k.router)?,
            k,
            placed,
            step_in: StepIn::new(stream)?,
            ring,
            messages: true,
            taps: None,
            draft: None,
            forced: None,
            held: 0,
            end: 0,
            ctx,
            d,
            visits,
            layers,
            stores,
            acts,
            hp,
        })
    }

    /// The placed side of the load, `None` on the whole-card load ([`GpuModel::open_placed`]).
    #[must_use]
    pub fn placed(&self) -> Option<&Placed> {
        self.placed.as_ref()
    }

    /// The positions a rollback takes back, from the written end: every store is positional, so
    /// the least of the conv ring's reach and each windowed visit ring's slack
    /// ([`runtime::stores::rollback_reach`]).
    #[must_use]
    pub fn rollback_reach(&self) -> u32 {
        let window = self.hp.swa as u64;
        runtime::stores::rollback_reach(
            self.d.taps,
            self.stores
                .iter()
                .filter(|s| s.shape.ring)
                .map(|s| (s.shape.rows as u64, window)),
        )
    }

    /// The lowest position a rollback takes back to ([`runtime::stores::rollback_floor`] of the
    /// written end): nothing at all after a call that failed part-way, until a reset.
    #[must_use]
    pub fn rollback_floor(&self) -> u32 {
        if self.held == u32::MAX {
            return u32::MAX;
        }
        runtime::stores::rollback_floor(self.end, self.rollback_reach())
    }

    /// The visits the step walks, in run order: `(stored layer, pass)` each.
    #[must_use]
    pub fn visit_table(&self) -> Vec<(u32, u32)> {
        self.visits.iter().map(|v| (v.layer, v.pass)).collect()
    }

    /// The passes the core runs, 1 where the trunk has no loop.
    #[must_use]
    pub fn passes(&self) -> usize {
        self.hp.loop_.map_or(1, |lp| lp.passes)
    }

    /// Whether stored layer `l`'s input norms and projections run as one launch (`norm_proj`) on
    /// a step without the imatrix collector.
    #[must_use]
    pub fn norm_proj(&self, l: usize) -> bool {
        self.layers.get(l).is_some_and(|x| x.norm_proj)
    }

    /// The stored layers the visits run.
    #[must_use]
    pub fn stored_layers(&self) -> usize {
        self.layers.len()
    }

    /// Device bytes of the visits' stores.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(VisitStore::bytes).sum()
    }

    /// Every visit's stores as the card holds them, in run order: a gate's read of what a prompt
    /// left. Synchronizes `stream`.
    pub fn stores_to_host(&self, stream: &CudaStream) -> Result<Vec<state::StoreHost>, GpuError> {
        self.stores.iter().map(|s| s.to_host(stream)).collect()
    }

    /// Whether the visits of a pass past the first run their message mixes and controllers. Off,
    /// every such visit reads its sublayers' inputs as the controllers leave them when their down
    /// rows are zero (the input itself: `tanh(0) = 0`), so a gate holds the message path live. A
    /// captured chain is the mode it was captured in: set it before the capture.
    pub fn set_messages(&mut self, on: bool) {
        self.messages = on;
    }

    /// The values of each of [`TAP_STAGES`] a visit.
    #[must_use]
    pub fn tap_widths(&self) -> [usize; TAP_STAGES.len()] {
        tap_widths(&self.d)
    }

    /// Keep every visit's [`TAP_STAGES`] at each step's position from now on ([`Body::taps`]).
    /// Load-time work: before any capture.
    pub fn arm_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.placed.is_some() {
            return Err(GpuError::state(
                "inkling2::Body::arm_taps",
                "a whole-card load (a placed step carries no taps)",
            ));
        }
        let block: usize = self.tap_widths().iter().sum();
        self.taps = Some(DeviceBuffer::zeroed(
            gpu.stream(),
            self.visits.len() * block,
        )?);
        Ok(())
    }

    /// Stop keeping the visits' outputs ([`Body::arm_taps`]); a capture after it holds no tap's
    /// copy.
    pub fn disarm_taps(&mut self) {
        self.taps = None;
    }

    /// Every visit's [`TAP_STAGES`] at the last step's position, visit-major and the stages in
    /// order within a visit; refused by name when the taps are not armed. Synchronizes `stream`.
    pub fn taps(&self, stream: &CudaStream) -> Result<Vec<f32>, GpuError> {
        let t = self
            .taps
            .as_ref()
            .ok_or(GpuError::state(STEP, "the armed taps"))?;
        stream.synchronize()?;
        Ok(t.to_host_vec(stream)?)
    }

    /// Route every visit's feed-forward block by `ids` and `weights` (visit-major, the router's
    /// top count a visit, the weights as the router leaves them: normalized and scaled) instead
    /// of the router's picks, from the next step on, until [`Body::unforce_picks`]: a gate's
    /// teacher-forced chain. The first call allocates. Eager steps only: a capture reads the
    /// buffers it was captured over. Refused by name: other lengths, an id past the experts.
    pub fn force_picks(
        &mut self,
        gpu: &Gpu,
        (ids, weights): (&[u32], &[f32]),
    ) -> Result<(), GpuError> {
        const WHAT: &str = "inkling2::Body::force_picks";
        let len = self.visits.len() * N_USED;
        if ids.len() != len || weights.len() != len {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} ids and {} weights; the visits take {len} each",
                    ids.len(),
                    weights.len()
                ),
            ));
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.d.n_expert) {
            return Err(GpuError::shape(
                WHAT,
                format!("expert {bad} of {}", self.d.n_expert),
            ));
        }
        let stream = gpu.stream();
        let f = match &mut self.forced {
            Some(f) => f,
            slot @ None => slot.insert(Forced {
                ids: DeviceBuffer::zeroed(stream, len)?,
                weights: DeviceBuffer::zeroed(stream, len)?,
            }),
        };
        f.ids.copy_from_host(stream, ids)?;
        f.weights.copy_from_host(stream, weights)?;
        Ok(())
    }

    /// Route by the router's picks again ([`Body::force_picks`]).
    pub fn unforce_picks(&mut self) {
        self.forced = None;
    }

    fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.pass.as_ref().map_or(0, pass::Pass::bytes)
            + self
                .verify
                .as_ref()
                .map_or(0, pass::rows::VerifyBufs::bytes)
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
            + self.acts.bytes()
            + self.step_in.bytes()
            + self.k.kgemv.device_bytes()
            + self.taps.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.draft.as_ref().map_or(0, DraftTaps::bytes)
            + self
                .layers
                .iter()
                .map(|l| l.bias.num_bytes() + l.ctrl.as_ref().map_or(0, |c| c.queries.num_bytes()))
                .sum::<usize>()
    }

    fn walk(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let Body {
            hp,
            d,
            visits,
            layers,
            stores,
            k,
            acts,
            s,
            step_in,
            messages,
            taps,
            draft,
            forced,
            ctx,
            placed,
            ..
        } = self;
        let cx = GqaCx {
            what: STEP,
            gpu,
            w,
            kgemv: &k.kgemv,
            g32: &k.g32,
            flash: &k.flash,
            mma: false,
            out_gate: None,
            hidden: d.n,
            eps: hp.rms_eps,
            n_kv: d.n_kv,
            ctx: *ctx,
        };
        let mut walk = Walk {
            cx,
            k,
            acts,
            s,
            step_in,
            forced: forced.as_ref(),
            d: *d,
            eps: hp.rms_eps,
            clips: (hp.qk_clip, hp.rel_clip),
            act: Act::Situ {
                beta: hp.situ.gate_beta,
                linear_beta: hp.situ.value_beta,
            },
            route_scale: hp.weights_scale,
            mix_scale: 1.0 / (d.n as f32).sqrt(),
            block: hp.block,
            pushed: 0,
            prelude: 0,
            n_msg: 0,
            banked: false,
        };
        match placed.as_mut() {
            Some(Placed {
                host: PlacedHost { shell, side, .. },
                ..
            }) => {
                walk.embed_norm()?;
                let mut leg = StepLeg::new(gpu.stream(), &mut shell.hybrid);
                let mut step = StepWalk {
                    walk,
                    layers: &layers[..],
                    stores: &mut stores[..],
                    placed: side,
                };
                step.run(&mut leg)?;
                walk = step.walk;
            }
            None => {
                walk.embed()?;
                let core = hp.loop_.filter(|lp| lp.passes > 1);
                for (slot, v) in visits.iter().enumerate() {
                    let l = v.layer as usize;
                    let layer = &layers[l];
                    if let Some(lp) = core
                        && l == lp.core_start
                    {
                        walk.pass_edge(v.pass)?;
                    }
                    if let Some(t) = draft.as_mut()
                        && let Some(i) = t.slot(slot)
                    {
                        t.enter(gpu.stream(), i, &walk.s.x, 1)?;
                    }
                    let msg = *messages && v.pass > 0 && layer.ctrl.is_some();
                    walk.attn_half(slot, (l, layer), &mut stores[slot], msg)?;
                    walk.ffn_front(slot, layer, msg)?;
                    walk.ffn_shadow(slot, layer)?;
                    walk.ffn_back(slot, layer, &mut stores[slot])?;
                    if let Some(t) = taps.as_mut() {
                        walk.tap(t, slot)?;
                    }
                }
            }
        }
        walk.finish(
            visits.len(),
            head,
            (hp.softcap, hp.n_vocab_unpadded, hp.n_vocab),
        )?;
        if let Some(t) = draft.as_mut()
            && let Some(i) = t.slot(visits.len())
        {
            t.enter(gpu.stream(), i, head.normed(), 1)?;
        }
        Ok(())
    }
}

/// Teacher-forced picks: every visit's routed ids and their weights, visit-major, the router's
/// top count a visit.
struct Forced {
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

/// The step's borrowed parts and the walk's counters.
struct Walk<'a> {
    cx: GqaCx<'a>,
    k: &'a Kernels,
    acts: &'a mut Acts,
    s: &'a mut Scratch,
    step_in: &'a mut StepIn,
    forced: Option<&'a Forced>,
    d: Dims,
    eps: f32,
    /// The q/k and the relative rows' clip thresholds.
    clips: (f32, f32),
    act: Act,
    route_scale: f32,
    /// The mixes' score scale, `1 / √n`.
    mix_scale: f32,
    /// The layers a residual block spans.
    block: usize,
    /// Block inputs banked so far this step.
    pushed: usize,
    /// The prelude's banked blocks.
    prelude: usize,
    /// The message bank's rows past its last.
    n_msg: usize,
    /// Whether this visit banked the stream: its attention output is the feed-forward input.
    banked: bool,
}

/// `src`'s `rows` rows of `n` from row `from` into `dst` from row `to`, one copy; nothing for no
/// row.
fn copy_rows(
    stream: &CudaStream,
    (src, from): (&DeviceBuffer<f32>, usize),
    (dst, to): (&mut DeviceBuffer<f32>, usize),
    (rows, n): (usize, usize),
) -> Result<(), GpuError> {
    if rows == 0 {
        return Ok(());
    }
    let src = Window::<f32>::of(src, from * n * 4, rows * n)?;
    let mut dst = WindowMut::<f32>::of_mut(dst, to * n * 4, rows * n)?;
    copy_row(stream, &src, &mut dst, rows * n)
}

/// The values of each of [`TAP_STAGES`] a visit of widths `d`.
fn tap_widths(d: &Dims) -> [usize; TAP_STAGES.len()] {
    let (q, kv, r) = (d.n_head * HEAD, d.n_kv * HEAD, d.n_head * REL);
    [q, kv, kv, r, q, d.n, d.n, d.n, d.n, d.n]
}

/// The residual short conv's output scale: `1 / √(taps + 1)`, twice the taps for a dynamic
/// conv (`inkling2.cpp`'s `convolve`, f32).
fn conv_scale(taps: usize, dynamic: bool) -> f32 {
    let k = if dynamic { 2 * taps } else { taps };
    1.0 / ((k + 1) as f32).sqrt()
}

/// The attention input `j` (q, k, v) of a visit before its norm: the attention mix's row (row 0
/// of a windowed visit's one query), or the stream itself where the layer mixes none.
fn attn_input<'b>(
    (mix, x): (&'b PartedBuffer<f32, 3>, &'b DeviceBuffer<f32>),
    layer: &Layer,
    j: usize,
) -> &'b DeviceBuffer<f32> {
    match layer.attn_query {
        Some(_) => mix.part(if layer.windowed { 0 } else { j }),
        None => x,
    }
}

impl Walk<'_> {
    /// Visit `slot`'s [`TAP_STAGES`] into its block of `t`.
    fn tap(&mut self, t: &mut DeviceBuffer<f32>, slot: usize) -> Result<(), GpuError> {
        let stream = self.cx.gpu.stream();
        let s = &*self.s;
        let ffn_inp = if self.banked { &s.attn_out } else { &s.ffn_inp };
        let rows = [
            &s.q,
            &s.k,
            &s.v,
            &s.r,
            &s.attn,
            &s.attn_out,
            ffn_inp,
            &s.moe,
            &s.ffn_out,
            &s.x,
        ];
        let widths = tap_widths(&self.d);
        let block: usize = widths.iter().sum();
        let mut at = slot * block;
        for (src, w) in rows.into_iter().zip(widths) {
            let mut dst = WindowMut::<f32>::of_mut(t, at * 4, w)?;
            copy_row(stream, src, &mut dst, w)?;
            at += w;
        }
        Ok(())
    }

    /// The token's embedding row, normed by the embedding norm into the stream; the row's
    /// position and live key count beside it.
    fn embed(&mut self) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let StepIn {
            words, pos, n_keys, ..
        } = &mut *self.step_in;
        {
            let ids = Window::<u32>::of(words, WORD_TOKEN * 4, 1)?;
            let pos0 = Window::<u32>::of(words, WORD_POS * 4, 1)?;
            self.k.q38.enqueue_embed_lookup(
                gpu,
                STEP,
                (w, "token_embd.weight"),
                EmbedIo {
                    ids: &ids,
                    pos0: &pos0,
                    first: 0,
                    width: self.d.n,
                    y: &mut self.s.emb,
                    pos,
                    n_keys,
                },
            )?;
        }
        self.embed_norm()
    }

    /// The embedding row's norm into the stream: the step's first block input. A placed load's
    /// row arrives from the host ([`Placed::refresh`]), so it starts here.
    fn embed_norm(&mut self) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        gpu.elem().enqueue_rms_norm(
            gpu.stream(),
            &self.s.emb,
            w.f32_buf(STEP, names::TOKEN_EMBD_NORM)?,
            self.eps,
            self.d.n,
            1,
            &mut self.s.x,
        )
    }

    /// The first core visit of `pass`: pass 0 keeps the prelude's blocks and stream; a later pass
    /// takes the blocks the previous pass banked past its first as the message bank (copied out:
    /// this pass's banks land on their rows), the stream as the bank's last row, and restarts
    /// from the prelude.
    fn pass_edge(&mut self, pass: u32) -> Result<(), GpuError> {
        let stream = self.cx.gpu.stream();
        let n = self.d.n;
        let s = &mut *self.s;
        if pass == 0 {
            self.prelude = self.pushed;
            return copy_row(stream, &s.x, &mut s.prelude, n);
        }
        let first = self.prelude + 1;
        if self.pushed < first {
            return Err(GpuError::shape(
                STEP,
                format!(
                    "pass {pass} after {} banked blocks over a prelude of {}: a pass banks its \
                     first layer's input",
                    self.pushed, self.prelude
                ),
            ));
        }
        self.n_msg = self.pushed - first;
        copy_rows(
            stream,
            (&s.banks, first),
            (&mut s.msg_bank, 0),
            (self.n_msg, n),
        )?;
        copy_row(stream, &s.x, &mut s.bank_last, n)?;
        self.pushed = self.prelude;
        copy_row(stream, &s.prelude, &mut s.x, n)
    }

    /// The visit's attention half (module doc, steps 2 to 5), the stream joined.
    #[allow(
        clippy::too_many_lines,
        reason = "one sublayer in the reference's order: the messages, the inputs, the \
                  projections, the convs, the epilogue, the attention and the join"
    )]
    fn attn_half(
        &mut self,
        slot: usize,
        (l, layer): (usize, &Layer),
        store: &mut VisitStore,
        msg: bool,
    ) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let stream = gpu.stream();
        let d = self.d;
        let n = d.n;
        let fault = gpu.layer_sink(slot)?;
        let ctrl = match (msg, layer.ctrl.as_ref()) {
            (true, Some(c)) => Some(c),
            (true, None) => {
                return Err(GpuError::state(STEP, "a core layer's controllers"));
            }
            (false, _) => None,
        };
        let mix_args = |banks, n_banks, cur, queries, nq| AttnResNormedArgs {
            banks,
            n_banks,
            stride: n,
            cur,
            queries,
            nq,
            scale: self.mix_scale,
            eps: MIX_EPS,
            n,
            m: 1,
            fault,
        };
        {
            let Scratch {
                msg_bank,
                bank_last,
                banks,
                x,
                msg,
                mix,
                mix_step,
                ..
            } = &mut *self.s;
            // The message mix and the attention mix read nothing the other writes: one pair.
            let msgs = ctrl.map(|c| {
                (
                    mix_args(msg_bank, self.n_msg, bank_last, &c.queries, MAX_NQ),
                    msg.whole_mut(),
                )
            });
            let attn = match &layer.attn_query {
                Some((query, nq)) => Some((
                    mix_args(banks, self.pushed, x, w.f32_buf(STEP, query)?, *nq),
                    mix.whole_mut(),
                )),
                None => None,
            };
            let (first, second) = match (msgs, attn) {
                (Some(m), a) => (Some(m), a),
                (None, a) => (a, None),
            };
            if let Some(first) = first {
                self.k.step_mix.enqueue(stream, first, second, mix_step)?;
            }
        }
        let n_in = if layer.windowed && ctrl.is_none() {
            1
        } else {
            3
        };
        let fused = layer.norm_proj && crate::imatrix::collector().is_none();
        // The fused launch norms its own inputs.
        let norms = if fused { 0 } else { n_in };
        {
            let Scratch {
                mix,
                msg: msgs,
                x,
                ctl,
                h,
                xn,
                ..
            } = &mut *self.s;
            if let Some(c) = ctrl {
                let args = |j: usize| -> Result<LoopControlArgs<'_>, GpuError> {
                    let [down, gain, forget] = &c.w[j];
                    Ok(LoopControlArgs {
                        cur: attn_input((mix, x), layer, j),
                        msg: msgs.part(MSG_ATTN[j]),
                        down: w.f32_buf(STEP, down)?,
                        gain: w.f32_buf(STEP, gain)?,
                        forget: w.f32_buf(STEP, forget)?,
                        n,
                        rank: d.rank,
                        m: 1,
                        eps: MIX_EPS,
                        fault,
                    })
                };
                let [c0, c1, c2] = ctl;
                self.k.ctrl.enqueue_attn(
                    stream,
                    [args(0)?, args(1)?, args(2)?],
                    h,
                    [c0, c1, c2],
                )?;
            }
            for (j, out) in xn.iter_mut().enumerate().take(norms) {
                let src = if ctrl.is_some() {
                    &ctl[j]
                } else {
                    attn_input((mix, x), layer, j)
                };
                gpu.elem().enqueue_rms_norm(
                    stream,
                    src,
                    w.f32_buf(STEP, &layer.norms[j])?,
                    self.eps,
                    n,
                    1,
                    out,
                )?;
            }
        }
        // The projections of the normed inputs: q, the relative rows, the sinks and the gate read
        // the q input, k and its taps the k input, v and its taps the v input.
        let (ik, iv) = if n_in == 1 { (0, 0) } else { (1, 2) };
        if fused {
            let Scratch {
                mix,
                x,
                ctl,
                q,
                k_raw,
                v_raw,
                r,
                sinks,
                gate,
                k_taps,
                v_taps,
                ..
            } = &mut *self.s;
            let (mix, x, ctl) = (&*mix, &*x, &*ctl);
            // Inputs past the visit's count repeat input 0 and are not read.
            let at = |i: usize| if i < n_in { i } else { 0 };
            let src: [&DeviceBuffer<f32>; 3] = std::array::from_fn(|i| {
                if ctrl.is_some() {
                    &ctl[at(i)]
                } else {
                    attn_input((mix, x), layer, at(i))
                }
            });
            let gain = [
                w.f32_buf(STEP, &layer.norms[at(0)])?,
                w.f32_buf(STEP, &layer.norms[at(1)])?,
                w.f32_buf(STEP, &layer.norms[at(2)])?,
            ];
            let sink = w.f32_tensor(STEP, &layer.sink.name)?;
            // A windowed layer has no taps: the sink rows stand in, unread.
            let (kt, vt) = match &layer.taps {
                Some([tk, tv]) => (w.f32_tensor(STEP, &tk.name)?, w.f32_tensor(STEP, &tv.name)?),
                None => (sink, sink),
            };
            if layer.q.ty == SiteTy::Bf16 {
                self.k.proj_bf16.enqueue(
                    stream,
                    NormProjBf16Args {
                        x: src,
                        gain,
                        ik,
                        iv,
                        eps: self.eps,
                        k: n,
                        m: 1,
                        fault,
                        bf16: [
                            bf16_site(w, &layer.q, q)?,
                            bf16_site(w, &layer.k, k_raw)?,
                            bf16_site(w, &layer.v, v_raw)?,
                            bf16_site(w, &layer.rel, r)?,
                            bf16_site(w, &layer.gate, gate)?,
                        ],
                        sink: F32Site { w: sink, y: sinks },
                        kt: F32Site { w: kt, y: k_taps },
                        vt: F32Site { w: vt, y: v_taps },
                        taps: layer.taps.is_some(),
                    },
                )?;
            } else {
                self.k.proj.enqueue(
                    stream,
                    NormProjArgs {
                        x: src,
                        gain,
                        ik,
                        iv,
                        eps: self.eps,
                        k: n,
                        fault,
                        q8: [
                            q8_site(w, &layer.q, q)?,
                            q8_site(w, &layer.k, k_raw)?,
                            q8_site(w, &layer.v, v_raw)?,
                            q8_site(w, &layer.rel, r)?,
                            q8_site(w, &layer.gate, gate)?,
                        ],
                        sink: F32Site { w: sink, y: sinks },
                        kt: F32Site { w: kt, y: k_taps },
                        vt: F32Site { w: vt, y: v_taps },
                        taps: layer.taps.is_some(),
                    },
                )?;
            }
        } else {
            let (cx, acts, s) = (&self.cx, &mut *self.acts, &mut *self.s);
            layer.q.project(cx, acts, slot, &s.xn[0], &mut s.q)?;
            layer.k.project(cx, acts, slot, &s.xn[ik], &mut s.k_raw)?;
            layer.v.project(cx, acts, slot, &s.xn[iv], &mut s.v_raw)?;
            layer.rel.project(cx, acts, slot, &s.xn[0], &mut s.r)?;
            layer.sink.project(cx, acts, slot, &s.xn[0], &mut s.sinks)?;
            layer.gate.project(cx, acts, slot, &s.xn[0], &mut s.gate)?;
            if let Some([tk, tv]) = &layer.taps {
                tk.project(cx, acts, slot, &s.xn[ik], &mut s.k_taps)?;
                tv.project(cx, acts, slot, &s.xn[iv], &mut s.v_taps)?;
            }
        }
        // The k and v short convs on the visit's ring.
        let ring = Ring::new(d.taps);
        let (heads, dynamic) = if layer.windowed {
            (0, false)
        } else {
            (d.n_kv, true)
        };
        {
            let s = &mut *self.s;
            let kvw = d.n_kv * HEAD;
            for (j, (x, taps, y)) in [
                (&s.k_raw, &s.k_taps, &mut s.k),
                (&s.v_raw, &s.v_taps, &mut s.v),
            ]
            .into_iter()
            .enumerate()
            {
                self.k.conv.enqueue(
                    stream,
                    ConvResArgs {
                        ring,
                        d: kvw,
                        base: d.bases[j],
                        ch: d.ch,
                        heads,
                        m: 1,
                        x,
                        weights: w.f32_buf(STEP, &layer.convs[j])?,
                        dynamic: taps,
                        pos: &self.step_in.pos,
                        fault,
                        scale: conv_scale(d.taps, dynamic),
                        y,
                        state: &mut store.conv,
                    },
                )?;
            }
        }
        {
            let (s, step_in) = (&mut *self.s, &*self.step_in);
            let word = if store.shape.ring {
                WORD_RING
            } else {
                WORD_POS
            };
            let at = Window::<u32>::of(&step_in.words, word * 4, 1)?;
            self.k.clip.enqueue(
                stream,
                QkClipArgs {
                    q: &mut s.q,
                    k: &mut s.k,
                    v: &mut s.v,
                    r: &mut s.r,
                    gq: w.f32_buf(STEP, &layer.q_clip)?,
                    gr: w.f32_buf(STEP, &layer.rel_clip)?,
                    tau: if layer.windowed {
                        &step_in.one
                    } else {
                        &step_in.tau
                    },
                    pos: &at,
                    v_eps: self.eps,
                    qk_thresh: self.clips.0,
                    r_thresh: self.clips.1,
                    n_head: d.n_head,
                    n_kv: d.n_kv,
                    ctx: store.shape.rows,
                    m: 1,
                    fault,
                    cache_k: &mut store.kc,
                    cache_v: &mut store.vc,
                },
            )?;
            self.k.rel.enqueue(
                stream,
                RelBiasArgs {
                    q: &s.q,
                    kc: &store.kc,
                    vc: &store.vc,
                    n_keys: &step_in.n_keys,
                    r: &s.r,
                    bias: &layer.bias,
                    extent: layer.extent,
                    scale: 1.0 / (HEAD as f32).sqrt(),
                    n_kv: d.n_kv,
                    ctx: self.cx.ctx,
                    ring_rows: store.shape.ring.then_some(store.shape.rows),
                    m: 1,
                    window: if layer.windowed { d.window } else { 0 },
                    sinks: &s.sinks,
                    part_v: &mut s.part_v,
                    part_ms: &mut s.part_ms,
                    fault,
                    y: &mut s.attn,
                },
                d.n_head,
            )?;
            self.k.out_gate.enqueue_out_gate(
                stream,
                OutGateArgs {
                    kind: OutGate::Proj,
                    attn: &s.attn,
                    gate: &s.gate,
                    n_head: d.n_head,
                    m: 1,
                    fault,
                    y: &mut s.gated,
                },
            )?;
        }
        {
            let (cx, acts, s) = (&self.cx, &mut *self.acts, &mut *self.s);
            layer.o.project(cx, acts, slot, &s.gated, &mut s.o)?;
            self.k.conv.enqueue(
                stream,
                ConvResArgs {
                    ring,
                    d: n,
                    base: d.bases[2],
                    ch: d.ch,
                    heads: 0,
                    m: 1,
                    x: &s.o,
                    weights: w.f32_buf(STEP, &layer.convs[2])?,
                    dynamic: &s.k_taps,
                    pos: &self.step_in.pos,
                    fault,
                    scale: conv_scale(d.taps, false),
                    y: &mut s.attn_out,
                    state: &mut store.conv,
                },
            )?;
        }
        // The block join: a block's first layer banks the stream and its attention output is the
        // new stream; any other adds the two.
        let s = &mut *self.s;
        self.banked = l % self.block == 0;
        if self.banked {
            if self.pushed + 1 >= MAX_VECS {
                return Err(GpuError::shape(
                    STEP,
                    format!("a bank past {} blocks at visit {slot}", self.pushed),
                ));
            }
            let mut row = WindowMut::<f32>::of_mut(&mut s.banks, self.pushed * n * 4, n)?;
            copy_row(stream, &s.x, &mut row, n)?;
            self.pushed += 1;
            Ok(())
        } else {
            gpu.elem()
                .enqueue_add(stream, &s.x, &s.attn_out, n, &mut s.ffn_inp)
        }
    }

    /// The feed-forward block's front: the mix of the banked blocks and the joined stream, the
    /// ffn controller where the messages run, the ffn norm and the router.
    fn ffn_front(&mut self, slot: usize, layer: &Layer, msg: bool) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let stream = gpu.stream();
        let n = self.d.n;
        let fault = gpu.layer_sink(slot)?;
        let s = &mut *self.s;
        let cur = if self.banked { &s.attn_out } else { &s.ffn_inp };
        self.k.step_mix.enqueue(
            stream,
            (
                AttnResNormedArgs {
                    banks: &s.banks,
                    n_banks: self.pushed,
                    stride: n,
                    cur,
                    queries: w.f32_buf(STEP, &layer.ffn_query)?,
                    nq: 1,
                    scale: self.mix_scale,
                    eps: MIX_EPS,
                    n,
                    m: 1,
                    fault,
                },
                &mut s.fmix,
            ),
            None,
            &mut s.mix_step,
        )?;
        let normed_in = match (msg, layer.ctrl.as_ref()) {
            (true, Some(c)) => {
                let [down, gain, forget] = &c.w[3];
                self.k.ctrl.enqueue(
                    stream,
                    LoopControlArgs {
                        cur: &s.fmix,
                        msg: s.msg.part(MSG_FFN),
                        down: w.f32_buf(STEP, down)?,
                        gain: w.f32_buf(STEP, gain)?,
                        forget: w.f32_buf(STEP, forget)?,
                        n,
                        rank: self.d.rank,
                        m: 1,
                        eps: MIX_EPS,
                        fault,
                    },
                    &mut s.h,
                    &mut s.fctl,
                )?;
                &s.fctl
            }
            (true, None) => return Err(GpuError::state(STEP, "a core layer's controllers")),
            (false, _) => &s.fmix,
        };
        gpu.elem().enqueue_rms_norm(
            stream,
            normed_in,
            w.f32_buf(STEP, &layer.ffn_norm)?,
            self.eps,
            n,
            1,
            &mut s.fxn,
        )?;
        let rin = RouteIn {
            w: w.f32_tensor(STEP, &layer.router)?,
            bias: Some(w.f32_buf(STEP, &layer.probs_b)?),
            scale: self.route_scale,
            fault,
        };
        self.k.router.route(stream, &s.fxn, &rin, &mut s.route)
    }

    /// The feed-forward block's shadow: every pick on the card's slots (every expert resident,
    /// so a pick's id is its place), the slots' weighted sum.
    fn ffn_shadow(&mut self, slot: usize, layer: &Layer) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let arms = accepts::leg_arms(Program::Inkling2Body)
            .ok_or_else(|| GpuError::state(STEP, "the inkling2 program's card leg"))?;
        let names = layer
            .stacks
            .as_ref()
            .ok_or(GpuError::state(
                STEP,
                "the card's routed stacks (a whole-card load)",
            ))?
            .each_ref()
            .map(String::as_str);
        let stacks = LegStacks::of_arms(w, names, arms, (GATE_UP_ROLE, DOWN_ROLE))?;
        let s = &mut *self.s;
        let forced = self
            .forced
            .map(|f| -> Result<_, GpuError> {
                let at = slot * N_USED * 4;
                Ok((
                    Window::<u32>::of(&f.ids, at, N_USED)?,
                    Window::<f32>::of(&f.weights, at, N_USED)?,
                ))
            })
            .transpose()?;
        let (sel, weights): (&DeviceBuffer<u32>, &DeviceBuffer<f32>) = match &forced {
            Some((ids, ws)) => (ids, ws),
            None => (s.route.ids(), s.route.weights()),
        };
        let run = LegRun {
            layer: slot,
            stacks: &stacks,
            normed: &s.fxn,
            m: 1,
            sel,
            n_card: self.d.n_expert,
            rule: self.act,
        };
        self.k.leg.enqueue(
            gpu,
            &mut s.leg,
            &run,
            Some(LegSum {
                w: weights,
                acc: &mut s.moe,
            }),
        )
    }

    /// The feed-forward block's back: the post conv of the routed sum, the stream's sum.
    fn ffn_back(
        &mut self,
        slot: usize,
        layer: &Layer,
        store: &mut VisitStore,
    ) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let stream = gpu.stream();
        let d = self.d;
        let s = &mut *self.s;
        self.k.conv.enqueue(
            stream,
            ConvResArgs {
                ring: Ring::new(d.taps),
                d: d.n,
                base: d.bases[3],
                ch: d.ch,
                heads: 0,
                m: 1,
                x: &s.moe,
                weights: w.f32_buf(STEP, &layer.convs[3])?,
                dynamic: &s.k_taps,
                pos: &self.step_in.pos,
                fault: gpu.layer_sink(slot)?,
                scale: conv_scale(d.taps, false),
                y: &mut s.ffn_out,
                state: &mut store.conv,
            },
        )?;
        let ffn_inp = if self.banked { &s.attn_out } else { &s.ffn_inp };
        gpu.elem()
            .enqueue_add(stream, ffn_inp, &s.ffn_out, d.n, &mut s.x)
    }

    /// The head: the mix of the banked blocks and the stream into the head's input, then the
    /// head's norm, projection, softcap, the padded columns at [`PAD_LOGIT`] and the argmax.
    fn finish(
        &mut self,
        n_visits: usize,
        head: &mut Head,
        (cap, unpadded, padded): (f32, usize, usize),
    ) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let stream = gpu.stream();
        let n = self.d.n;
        let s = &mut *self.s;
        self.k.step_mix.enqueue(
            stream,
            (
                AttnResNormedArgs {
                    banks: &s.banks,
                    n_banks: self.pushed,
                    stride: n,
                    cur: &s.x,
                    queries: w.f32_buf(STEP, names::OUTPUT_RES_QUERY)?,
                    nq: 1,
                    scale: self.mix_scale,
                    eps: MIX_EPS,
                    n,
                    m: 1,
                    fault: gpu.layer_sink(n_visits)?,
                },
                head.input_mut(),
            ),
            None,
            &mut s.mix_step,
        )?;
        head_logits((gpu, w), head, &s.pad, (cap, unpadded, padded), 1)
    }
}

/// The head over its `m` input rows: the norm, the projection, the softcap at `cap`, the padded
/// columns past `unpadded` at [`PAD_LOGIT`] (`pad`'s, at least `m` rows of them), the argmax over
/// the `padded` columns. The rows lie column by column (`logits[v · m + r]`), so the padded
/// columns of every row are one tail.
fn head_logits(
    (gpu, w): (&Gpu, &Weights),
    head: &mut Head,
    pad: &DeviceBuffer<f32>,
    (cap, unpadded, padded): (f32, usize, usize),
    m: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let n_pad = (padded - unpadded) * m;
    head.enqueue_with_epilogue(gpu, w, |logits: &mut DeviceBuffer<f32>| {
        gpu.elem()
            .enqueue_softcap_rows(stream, logits, padded * m, cap)?;
        if n_pad > 0 {
            let mut tail = WindowMut::<f32>::of_mut(logits, unpadded * m * 4, n_pad)?;
            copy_row(stream, pad, &mut tail, n_pad)?;
        }
        Ok(())
    })
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = Placed;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::Inkling2
    }

    /// The step at `pos`; a position past the stores is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(GpuError::shape(
                "inkling2::Body::decode_input",
                format!("a step at position {pos} in stores of {}", self.ctx),
            ));
        }
        Ok(DecodeInput { token, pos })
    }

    /// The words and the global visits' factor: two copies. Refused by name, before either, off
    /// the position the stores stand at.
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
        let tau = state::tau_of(input.pos, self.hp.log_floor, self.hp.log_alpha);
        self.step_in
            .refresh(stream, (input.token, input.pos), (self.ring, tau))?;
        if let Some(p) = self.placed.as_mut() {
            p.refresh(
                stream,
                input.token,
                (
                    &mut self.s.emb,
                    &mut self.step_in.pos,
                    &mut self.step_in.n_keys,
                ),
                input.pos,
            )?;
        }
        self.held = input.pos + 1;
        self.end = self.end.max(self.held);
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.walk(gpu, w, head)
    }

    /// Every conv ring back to zero in place (a captured chain keeps their addresses). The K and
    /// V planes need nothing: no row at or past the live count is read, and every row below it is
    /// written by its own step first.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for store in &mut self.stores {
            store.clear(stream)?;
        }
        if let Some(p) = self.placed.as_mut() {
            p.reset(stream)?;
        }
        self.held = 0;
        self.end = 0;
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every norm's, the head's included.
    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes() + self.scratch_bytes()
    }

    /// The stored layers: a visit's stores are its own, its weights its layer's.
    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::conv_scale;

    /// The conv scales are the reference's `1 / √(taps + 1)` and `1 / √(2·taps + 1)`.
    #[test]
    fn the_conv_scales_count_the_taps() {
        assert_eq!(
            conv_scale(4, false).to_bits(),
            (1.0f32 / 5.0f32.sqrt()).to_bits()
        );
        assert_eq!(conv_scale(4, true).to_bits(), (1.0f32 / 3.0f32).to_bits());
    }
}
