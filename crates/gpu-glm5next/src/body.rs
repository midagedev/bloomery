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
//! selector at (`place::ORACLE_POSITIONS`). The next-token (MTP) layer is
//! carried by the file and loaded only by a NextN load
//! ([`Body::open_placed_nextn`], [`nextn`]): beside the chain, never in it,
//! its routed experts one more layer of the host tier's run; the plain load
//! leaves its tensors `Role::Unused`, as ik loads and does not run them.
//!
//! A cut behind the fed positions finds a KDA layer's state only in a
//! checkpoint ([`bloomery_gpu::checkpoint`]): each prompt call ([`prompt`])
//! copies every KDA layer's committed state and conv ring to the host at the
//! marks of `runtime::seqstate`, and a cut to one of them copies it back at
//! the next step; the latent rows are cut by position. A sequence state on
//! the host ([`seq`]) carries the latent rows, the KDA stores at the held
//! position and at the last checkpoint below it, and the NextN side, and
//! puts them back on an empty model.
//!
//! Each KDA layer's state has the load's stamped lanes (`linear::delta`'s
//! module doc, [`KdaLanes`]), the committed one named by one lane word every
//! KDA launch reads: one on a load whose sequences each run one row a pass,
//! two on one that verifies two rows ([`Body::open_placed_lanes`],
//! [`Body::open_placed_nextn`]), the plan counting the lanes the stores hold.
//! The lanes are a sequence's; the rows in flight are the load's: every load
//! holds [`LOAD_ROWS`] rows' buffers, the card experts' and the host
//! boundary's, whatever its lanes. The step and a prompt batch run in place
//! on the committed lane. A verify of two rows ([`pair`]) runs the step's
//! launches once a row, the rows one layer apart: row 0 in place, row 1 from
//! the lane row 0 wrote into the other; its commit keeps row 0 (the word
//! stays) or both (the word moves to row 1's lane) and copies nothing. A
//! pass of slots walks its rows a layer apart the same way, each row on its
//! own slot's sequence: on a plain load a row a slot, a plain step in place
//! on that sequence's committed lane; on a NextN load each slot's verify's
//! two, as that slot's own verify runs them, each slot then committed on its
//! own. The conv ring and the latent rows are indexed by position, so a row
//! taken back is written again by the next step there.
//!
//! The body serves resident sequence slots ([`Slots`],
//! `GpuModel::add_slots`): a sequence's stores, its lane word and lanes, its
//! checkpoints, the positions its stores and the arenas hold, the final
//! streams of each row of its own verify (a row a lane), and on a NextN load
//! the layer's store and row-0 copy travel together ([`GlmSlot`]), exchanged by pointer on a
//! select; the host tier and its residency, the weights, the heads and every
//! buffer a call writes before it reads are the load's. The plan counts the
//! sequences the load serves (`place::PlanInputs::plan_slots`,
//! `plan_nextn_slots`), the load holds its plan to them, and a sequence past
//! their count is refused by name.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::checkpoint::Checkpoints;
use bloomery_gpu::checkpoint::saved::Identity;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::refuse_tier_count;
use bloomery_gpu::host::swap::{BoundaryAt, PassReport, ResetReport, Residency};
use bloomery_gpu::host::swap_source::{FileSwap, ResidencyGlue, ResidencySpec};
use bloomery_gpu::host::tier::TierOpen;
use bloomery_gpu::hybrid::{
    Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap,
};
use bloomery_gpu::kpool::{self, KpoolKernels};
use bloomery_gpu::latent::{INDEX_HEAD, INDEX_ROW, LATENT, LatentKernels, POOL, pools_for};
use bloomery_gpu::linear::{self, HEAD, KHeadMap, LinearKernels, LinearShape, PASS_ROWS};
use bloomery_gpu::model::{
    ChainBody, HostServed, Rollback, SlotRange, Slots, StepKernels, StepMode,
};
use bloomery_gpu::qsa::{QsaKernels, list_width};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Branch, DeviceTensor, Gpu, GpuError, GpuModel, PartedBuffer};
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
use model::arch::glm5next::place::{self, KdaLanes, NextnInputs, NextnPlan, PlanInputs};
use model::placement::Plan;
use models::{Act, Ffn, LayerSpec, Mixer, Score};
use runtime::layer::{FfnKind, Layer, MixerKind, ResidualKind, hosted};
use runtime::seqstate::{HOST_BUDGET, Kept, Take};
use runtime::swaprule::KeptRows;

use crate::ffn::CardExperts;
use crate::host::GlmHost;
use crate::program;
use crate::swap;
use crate::tensors::LayerNames;
use crate::tier;

#[path = "prefill.rs"]
pub mod prefill;

#[path = "pair.rs"]
mod pair;

#[path = "nextn.rs"]
pub mod nextn;

#[path = "seq.rs"]
pub mod seq;

pub use pair::LOAD_ROWS;

/// What the body's errors name.
const WHAT: &str = "glm5next Body";

// The plan sizes each KDA layer's conv ring by its own pass width.
const _: () = assert!(place::PASS_ROWS == PASS_ROWS);

// The plan counts each KDA layer's state by the lanes the store holds.
const _: () = assert!(KdaLanes::MAX == LANES);

/// The GLM model: one card, the skeleton over this body.
pub type Glm5nextModel = GpuModel<Body>;

/// The spacing of a prompt call's inner checkpoints; past the host budget's
/// slots the oldest is evicted (`runtime::seqstate`).
pub const CHECKPOINT_EVERY: u32 = 512;

/// The most lanes of a KDA layer's state a load holds ([`KdaLanes`]): a
/// verify of up to this many rows keeps the state after each row in a lane of
/// its own.
pub const LANES: usize = 2;

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
        /// The load's lanes of the recurrent state.
        state: LaneState,
        /// The position each lane's state stands at (`linear::delta`).
        stamp: DeviceBuffer<u32>,
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
            Store::Kda { state, stamp, ring } => {
                state.whole().num_bytes() + stamp.num_bytes() + ring.num_bytes()
            }
            Store::Latent {
                latent,
                index,
                pooled,
            } => latent.buf().num_bytes() + index.buf().num_bytes() + pooled.buf().num_bytes(),
        }
    }

    /// A KDA layer's committed lane `lane` of its state and its conv ring,
    /// the stores a checkpoint copies; `None` for a latent layer. A lane past
    /// the load's is refused by name.
    fn copied(&mut self, lane: u32) -> Result<Option<[&mut DeviceBuffer<f32>; 2]>, GpuError> {
        match self {
            Store::Kda { state, ring, .. } => Ok(Some([state.part_mut(lane)?, ring])),
            Store::Latent { .. } => Ok(None),
        }
    }

    /// A KDA layer's stamps: lane `lane` at position `at`, every other lane
    /// never written. Synchronizes; never inside a capture.
    fn restamp(&mut self, stream: &CudaStream, lane: u32, at: u32) -> Result<(), GpuError> {
        if let Store::Kda { state, stamp, .. } = self {
            let lanes = state.lanes();
            stamp.copy_from_host(stream, &stamps(lane, at, lanes)?[..lanes.count()])?;
        }
        Ok(())
    }

    /// Every lane zeroed, lane 0 stamped at position 0 (a fresh store).
    fn zero(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        match self {
            Store::Kda { state, stamp, ring } => {
                let lanes = state.lanes();
                state.whole_mut().zero_async(stream)?;
                stamp.copy_from_host(stream, &stamps(0, 0, lanes)?[..lanes.count()])?;
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

/// A KDA layer's stamps with lane `lane` at position `at` and every other
/// lane never written ([`linear::delta::NEVER`]), the first `lanes` of them
/// the store's. A lane past `lanes` is refused by name.
fn stamps(lane: u32, at: u32, lanes: KdaLanes) -> Result<[u32; LANES], GpuError> {
    let mut st = [linear::delta::NEVER; LANES];
    let slot = st
        .get_mut(..lanes.count())
        .and_then(|s| s.get_mut(lane as usize))
        .ok_or_else(|| shape(format!("lane {lane} of a state of {} lanes", lanes.count())))?;
    *slot = at;
    Ok(st)
}

/// A KDA layer's recurrent state: the load's lanes, each a part of its own.
pub(crate) enum LaneState {
    One(DeviceBuffer<f32>),
    Two(PartedBuffer<f32, 2>),
}

impl LaneState {
    /// `lanes` zeroed lanes of `len` values each. Load-time only.
    fn zeroed(stream: &CudaStream, len: usize, lanes: KdaLanes) -> Result<LaneState, GpuError> {
        Ok(match lanes {
            KdaLanes::One => LaneState::One(DeviceBuffer::zeroed(stream, len)?),
            KdaLanes::Two => LaneState::Two(PartedBuffer::zeroed(stream, [len; 2])?),
        })
    }

    /// The lanes the state holds.
    pub(crate) fn lanes(&self) -> KdaLanes {
        match self {
            LaneState::One(_) => KdaLanes::One,
            LaneState::Two(_) => KdaLanes::Two,
        }
    }

    /// Every lane in order, the buffer a lanes launch reads and writes.
    pub(crate) fn whole(&self) -> &DeviceBuffer<f32> {
        match self {
            LaneState::One(b) => b,
            LaneState::Two(p) => p.whole(),
        }
    }

    /// Every lane in order, for a launch that writes them.
    pub(crate) fn whole_mut(&mut self) -> &mut DeviceBuffer<f32> {
        match self {
            LaneState::One(b) => b,
            LaneState::Two(p) => p.whole_mut(),
        }
    }

    /// Lane `lane`; one past the state's lanes is refused by name.
    pub(crate) fn part(&self, lane: u32) -> Result<&DeviceBuffer<f32>, GpuError> {
        match (self, lane) {
            (LaneState::One(b), 0) => Ok(b),
            (LaneState::Two(p), 0 | 1) => Ok(p.part(lane as usize)),
            _ => Err(self.past(lane)),
        }
    }

    /// Lane `lane`, for a copy that writes it alone; one past the state's
    /// lanes is refused by name.
    pub(crate) fn part_mut(&mut self, lane: u32) -> Result<&mut DeviceBuffer<f32>, GpuError> {
        let past = self.past(lane);
        match (self, lane) {
            (LaneState::One(b), 0) => Ok(b),
            (LaneState::Two(p), 0 | 1) => Ok(p.part_mut(lane as usize)),
            _ => Err(past),
        }
    }

    fn past(&self, lane: u32) -> GpuError {
        shape(format!(
            "lane {lane} of a state of {} lanes",
            self.lanes().count()
        ))
    }
}

/// The step's buffers for each of the [`LOAD_ROWS`] rows a pass runs — the
/// load's rows in flight, whatever its KDA lanes: the one-token step and a
/// prompt batch use row 0's, a verify row 0's and row 1's, a pass of slots
/// every row it lays — the live sequence's lane word every KDA launch reads,
/// and the host's side of its lanes: the committed lane and the verify
/// waiting for its commit ([`pair`]).
pub(crate) struct Scratch {
    /// Each row's buffers, row 0 the step's.
    pub rows: [RowScratch; LOAD_ROWS],
    /// The committed lane of every KDA layer's state; one word, since every
    /// layer commits the same rows.
    pub lane: DeviceBuffer<u32>,
    pub lanes: pair::Lanes,
}

impl Scratch {
    /// Each row's buffers ([`RowScratch::new`]) and the lane word at lane 0.
    /// Load-time only.
    fn new(stream: &CudaStream, d: &Dims, ff: usize, ctx: usize) -> Result<Scratch, GpuError> {
        let row = || RowScratch::new(stream, d, ff, ctx);
        Ok(Scratch {
            // One buffer set a row of the load's LOAD_ROWS; another count
            // fails to compile here.
            rows: [row()?, row()?, row()?, row()?],
            lane: DeviceBuffer::zeroed(stream, 1)?,
            lanes: pair::Lanes::default(),
        })
    }

    /// Row `r`'s buffers, to write; `None` past the load's rows.
    pub(crate) fn row_mut(&mut self, r: usize) -> Option<&mut RowScratch> {
        self.rows.get_mut(r)
    }

    /// Every row's buffers, row 0 first.
    fn rows(&self) -> impl Iterator<Item = &RowScratch> {
        self.rows.iter()
    }

    /// Every row's buffers, row 0 first, to write.
    fn rows_mut(&mut self) -> impl Iterator<Item = &mut RowScratch> {
        self.rows.iter_mut()
    }

    /// Exchange row `a`'s and row `b`'s final-streams buffers, two rows of
    /// the load, `a ≠ b`; pointer moves only.
    fn swap_streams(&mut self, a: usize, b: usize, fin: usize) -> Result<(), GpuError> {
        let Scratch { rows, .. } = self;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if lo == hi || hi >= rows.len() {
            return Err(shape(format!(
                "rows {a} and {b} of a pass on a load of {LOAD_ROWS}"
            )));
        }
        let (head, rest) = rows.split_at_mut(hi);
        let a = &mut head[lo].streams[fin];
        let b = &mut rest[0].streams[fin];
        std::mem::swap(a, b);
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.rows.iter().map(RowScratch::bytes).sum::<usize>() + self.lane.num_bytes()
    }
}

/// Every buffer one row of the step writes or reads besides the stores, one
/// token's.
pub(crate) struct RowScratch {
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
    // The row's words: its position; the attention's visible counts (no
    // window row, then the selector's list length).
    pub pos: DeviceBuffer<u32>,
    pub vis: DeviceBuffer<u32>,
}

impl RowScratch {
    /// One row's buffers; the low-rank halves are `kda.head_dim` wide, as
    /// the header reader holds every KDA tensor to ik's exact dims.
    fn new(stream: &CudaStream, d: &Dims, ff: usize, ctx: usize) -> Result<RowScratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let (n, c, v) = (d.embd, d.kda.channels(), d.kda.n_v * HEAD);
        let rows = d.heads;
        let segs = attn::segments(0, list_width(d.kept));
        Ok(RowScratch {
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
/// the host set, its page windows, its residency machine) before the
/// buffers.
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
    /// The slot map's card copy, the host run's rows. The host tier holds
    /// the host copy, and its residency machine, when it runs one, the
    /// second reference: the copy lives until both let go.
    slots: Arc<DeviceTensor<u32>>,
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
    /// The lanes of every KDA layer's state, fixed at load.
    lanes: KdaLanes,
    /// The positions each arena's rows hold, as the call that last wrote them
    /// left them ([`nextn::Wrote`]).
    wrote: nextn::Wrote,
    /// The most positions a latent layer attends whole
    /// (`place::dense_positions`): a prompt chunk within them attends every
    /// position, one past them through the selector.
    dense: usize,
    /// The KDA layers' stores on the host at chosen positions.
    ckpt: Checkpoints,
    /// How a prompt is fed, and the batch feed's buffers.
    prompt: prefill::PromptState,
    /// The load's residency side ([`ResidencyGlue`]): the machine's source
    /// and shape, the boundaries' log, the delegation to the host tier.
    residency_glue: ResidencyGlue,
    /// The residency the body was loaded with: `off` keeps the load's slot
    /// map for the model's life.
    residency: Residency,
    /// The next-token layer, on a NextN load ([`nextn`]).
    nextn: Option<Box<nextn::Nextn>>,
    /// The rows the last plan of a pass of several slots wrote, until its
    /// enqueue ([`SlotRows`]).
    planned: Option<pair::SlotsPlanned>,
    /// What a sequence state of this load belongs to ([`seq`]).
    who: Identity,
    /// The resident sequences the plan counted, and those made, the live one
    /// counted ([`Slots`]).
    slots_planned: usize,
    slots_made: usize,
}

/// Every KDA layer's committed lane `lane` of its state and its conv ring,
/// in layer order: the list the checkpoints copy. A lane past the load's is
/// refused by name.
fn copied(stores: &mut [Store], lane: u32) -> Result<Vec<&mut DeviceBuffer<f32>>, GpuError> {
    let mut out = Vec::new();
    for s in stores {
        if let Some(pair) = s.copied(lane)? {
            out.extend(pair);
        }
    }
    Ok(out)
}

/// One sequence's waiting cut carried out on its committed lane `lane`
/// ([`Checkpoints::apply`] of its checkpoints `ckpt` into its `stores`) and
/// every KDA layer's stamps set to the position it leaves (`held`): the
/// committed lane there, the other never written. Nothing when no cut
/// waits. Waits for the copies.
fn cut_into(
    stream: &CudaStream,
    ckpt: &mut Checkpoints,
    stores: &mut [Store],
    lane: u32,
    held: u32,
) -> Result<(), GpuError> {
    if !ckpt.pending() {
        return Ok(());
    }
    ckpt.apply(stream, &mut copied(stores, lane)?)?;
    for st in stores {
        st.restamp(stream, lane, held)?;
    }
    Ok(())
}

/// Refused by name unless a sequence whose stores hold `held` positions,
/// its lanes `lanes`, takes a call at `pos`: the stores hold `pos` positions
/// and no verify waits for its commit. A step at `pos` that failed after its
/// inputs were on the card has already run the KDA layers' recurrence over
/// it, and running it again would apply the position twice; a verify's rows
/// stand in lanes no launch reads until its commit names the one kept.
fn held_at(lanes: &pair::Lanes, held: u32, pos: u32) -> Result<(), GpuError> {
    lanes.refuse_if_waiting(pos)?;
    match held {
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

/// The parts of the body a walk writes, lent apart from its host tier.
pub(crate) struct Parts<'s> {
    pub k: &'s Kernels,
    pub d: &'s Dims,
    pub cfg: &'s [LayerCfg],
    pub names: &'s [LayerNames],
    /// Row `row`'s buffers, which the launches write ([`Parts::at_row`]),
    /// and the load's other rows' — each of `idle` the row `idle_at` names.
    pub s: &'s mut RowScratch,
    idle: [&'s mut RowScratch; LOAD_ROWS - 1],
    idle_at: [usize; LOAD_ROWS - 1],
    pub row: usize,
    /// The current row's sequence: the committed lane word every KDA launch
    /// reads, and its stores.
    pub lane: &'s DeviceBuffer<u32>,
    pub stores: &'s mut [Store],
    /// The rows' sequences ([`RowBind`]): which sequence's lane word and
    /// stores each row of the walk runs on.
    bind: RowBind<'s>,
    pub slots: &'s DeviceTensor<u32>,
    pub card: &'s mut CardExperts,
    pub taps: Option<&'s mut [DeviceBuffer<f32>]>,
}

/// One sequence's part of a walk: the lane word its KDA launches read and
/// its stores.
pub(crate) struct SeqParts<'s> {
    pub lane: &'s DeviceBuffer<u32>,
    pub stores: &'s mut [Store],
}

/// A walk's rows bound each to its sequence: the current row's, which
/// [`Parts`] holds, and the one alternate a pass holds beside it from its
/// `alt`, the first row of the second sequence's range. A pass of several
/// slots binds at most two sequences — the live and one parked, or two
/// parked with the live idle; every other walk (the step, a verify, a
/// prompt batch) runs one sequence's rows whole, with no alternate. The
/// sequences exchange their lane word and stores only at the boundary
/// ([`Parts::at_row`]), and a row numbers by its offset in its sequence's
/// rows of the pass ([`RowBind::base`]).
pub(crate) struct RowBind<'s> {
    /// The first row of the alternate's range; past every row when there is
    /// no alternate.
    alt: usize,
    other: Option<SeqParts<'s>>,
}

impl<'s> RowBind<'s> {
    /// The most sequences a walk binds: the current row's and the
    /// alternate.
    pub(crate) const SEQS: usize = 2;

    /// Every row the one sequence's: no exchange, a row its own offset.
    fn one() -> RowBind<'s> {
        RowBind {
            alt: usize::MAX,
            other: None,
        }
    }

    /// The rows of a pass of `rows` — each busy slot's range — with `other`
    /// the second sequence's part when the pass binds two, its rows from the
    /// second range's start on.
    fn of_pass(rows: &[SlotRange], other: Option<SeqParts<'s>>) -> RowBind<'s> {
        RowBind {
            alt: rows.get(1).map_or(usize::MAX, |r| r.rows.start),
            other,
        }
    }

    /// The alternate's part, when a move of the walk's row `from` → `to`
    /// crosses into its rows: only at that boundary do the sequences
    /// exchange.
    fn crossed(&mut self, from: usize, to: usize) -> Option<&mut SeqParts<'s>> {
        self.other
            .as_mut()
            .filter(|_| (from < self.alt) != (to < self.alt))
    }

    /// Row `row`'s offset in its sequence's rows of the pass: its row when
    /// the walk runs one sequence's rows (a verify's row 1 reads the
    /// committed lane and writes the next), its offset in its slot's range
    /// when it runs several sequences' (a plain pass's every row steps its
    /// own sequence in place, offset 0).
    fn base(&self, row: usize) -> usize {
        if self.other.is_some() && row >= self.alt {
            row - self.alt
        } else {
            row
        }
    }
}

impl<'s> Parts<'s> {
    /// The parts over `s`'s rows, at row 0, every row the live sequence's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the body's pieces a walk borrows apart, each named as the body names it (rust-quality R8)"
    )]
    pub(crate) fn of(
        k: &'s Kernels,
        d: &'s Dims,
        cfg: &'s [LayerCfg],
        names: &'s [LayerNames],
        s: &'s mut Scratch,
        stores: &'s mut [Store],
        slots: &'s DeviceTensor<u32>,
        card: &'s mut CardExperts,
        taps: Option<&'s mut [DeviceBuffer<f32>]>,
    ) -> Parts<'s> {
        let Scratch { rows, lane, .. } = s;
        let [s, a, b, c] = rows;
        Parts {
            k,
            d,
            cfg,
            names,
            s,
            idle: [a, b, c],
            idle_at: [1, 2, 3],
            row: 0,
            lane: &*lane,
            stores,
            bind: RowBind::one(),
            slots,
            card,
            taps,
        }
    }

    /// The launches after this write row `row`'s buffers, on its
    /// sequence's stores; a row past [`LOAD_ROWS`] is refused by name.
    pub(crate) fn at_row(&mut self, row: usize) -> Result<(), GpuError> {
        if row >= LOAD_ROWS {
            return Err(shape(format!(
                "row {row} of a pass on a load of {LOAD_ROWS} row buffers"
            )));
        }
        if row == self.row {
            return Ok(());
        }
        let at = self
            .idle_at
            .iter()
            .position(|&r| r == row)
            .ok_or_else(|| shape(format!("row {row}'s buffers, which no idle row holds")))?;
        std::mem::swap(&mut self.s, &mut self.idle[at]);
        let was = self.row;
        if let Some(SeqParts { lane, stores }) = self.bind.crossed(was, row) {
            std::mem::swap(&mut self.lane, lane);
            std::mem::swap(&mut self.stores, stores);
        }
        self.idle_at[at] = was;
        self.row = row;
        Ok(())
    }

    /// The current row's KDA row base: its row on a walk of one sequence (a
    /// verify's row 1 reads the committed lane and writes the next), its
    /// offset in its slot's range on a pass of several sequences', whose
    /// rows each step their own sequence in place on its committed lane
    /// ([`RowBind::base`]).
    pub(crate) fn base(&self) -> usize {
        self.bind.base(self.row)
    }

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

/// Refused by name unless `plan`'s card `card` counts the KV term of
/// `inputs`' layout at `lanes` KDA lanes for `slots` resident sequences
/// (`nextn` the next-token layer the load carries,
/// [`PlanInputs::kv_term`]): a plan made for other lanes or another count
/// than the load's would place its card experts over the stores' bytes, or
/// leave them unused. Zero slots is refused by name.
fn refuse_other_lanes(
    plan: &Plan<'_>,
    inputs: &PlanInputs,
    card: usize,
    (lanes, slots): (KdaLanes, usize),
    nextn: Option<&NextnInputs>,
) -> Result<(), GpuError> {
    let (layers, planned) = plan
        .machine
        .cards
        .get(card)
        .zip(plan.cards.get(card))
        .map(|(c, t)| (c.layers.clone(), t.kv_bytes))
        .ok_or_else(|| shape(format!("the plan has no card {card}")))?;
    let want = inputs
        .kv_term(layers, plan.ctx_max, lanes, nextn, slots)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    if planned != want {
        return Err(shape(format!(
            "the plan counts {planned} store bytes on card {card}; a load of {} KDA lanes and \
             {slots} resident sequences holds {want} (plan it with PlanInputs::plan_slots or \
             plan_nextn_slots at the load's lanes and sequences)",
            lanes.count()
        )));
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
            "the load's scratch holds {made} device bytes (the rows in flight, the card \
             experts' rows, the host boundary, the slot map's copy); card {card}'s plan sets \
             aside {term} for it"
        )));
    }
    Ok(())
}

/// Refused by name, before anything uploads, unless the expert tier cards
/// `tiers` of `plan` are ones this load hangs: at most the cards the host
/// tier serves ([`refuse_tier_count`]), and a slot map over the host run
/// (with the next-token layer `nextn` at its end on a NextN load) with every
/// expert on one device and no tier entry in that layer's row
/// ([`refuse_nextn_on_tier`]). A residency machine runs beside a tier as it
/// runs without one: it moves the stage card's experts alone and holds every
/// tier expert away ([`bloomery_gpu::host::swap`]).
fn refuse_tiers_before_upload(
    plan: &Plan<'_>,
    inputs: &PlanInputs,
    card: usize,
    tiers: &[TierOpen],
    nextn: Option<usize>,
) -> Result<(), GpuError> {
    if tiers.is_empty() {
        return Ok(());
    }
    refuse_tier_count(TIER_BEFORE_UPLOAD, tiers.len())?;
    let run = host_run(&inputs.spec.layers, nextn)?;
    let cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
    let map = SlotMap::of_plan_tiers(plan, card, &cards, run, N_EXPERT)
        .map_err(|e| GpuError::plan(TIER_BEFORE_UPLOAD, e))?;
    match nextn {
        Some(n) => refuse_nextn_on_tier(&map, n),
        None => Ok(()),
    }
}

/// Refused by name: a slot map whose row for the next-token layer `nextn`
/// sends an expert to a tier card. The draft's walk serves that layer's
/// routed experts through the host tier's batch port and runs no tier join,
/// and the port's serve leaves a tiered layer's upload to that join, so a
/// tier entry there would leave the walk's routed sum unwritten.
fn refuse_nextn_on_tier(map: &SlotMap, nextn: usize) -> Result<(), GpuError> {
    let on_tier = map.on_tier(nextn)?;
    if on_tier == 0 {
        return Ok(());
    }
    Err(GpuError::Shape {
        what: TIER_BEFORE_UPLOAD,
        detail: format!(
            "{NEXTN_ON_TIER}: the next-token layer {nextn}'s row sends {on_tier} experts to an \
             expert tier card; the draft's walk serves that layer on the host through the batch \
             port and runs no tier join"
        ),
    })
}

/// The words [`refuse_nextn_on_tier`]'s refusal opens with.
pub const NEXTN_ON_TIER: &str = "a NextN expert on the tier";

/// What a tiered load's checks before any upload are named as in their
/// refusals ([`Body::open_placed_lanes`]).
pub const TIER_BEFORE_UPLOAD: &str = "glm5next Body::open_placed_lanes (before upload)";

/// The host tier's run over the trunk `layers`: its routed layers, and on a
/// NextN load the next-token layer `nextn` after them — the slot map's
/// layers. Refused by name: a trunk with no routed run, and a next-token
/// layer that does not follow it.
fn host_run(layers: &[LayerSpec], nextn: Option<usize>) -> Result<Range<usize>, GpuError> {
    let trunk = hosted(layers).map_err(|e| shape(e.to_string()))?;
    match nextn {
        None => Ok(trunk),
        Some(n) if trunk.end == n => Ok(trunk.start..n + 1),
        Some(n) => Err(shape(format!(
            "the next-token layer {n} does not follow the host run {trunk:?}"
        ))),
    }
}

impl Body {
    /// Card `card` of `plan`, which `inputs` made, resident, the load's
    /// slot map for the model's life, one KDA lane ([`Body::open_placed_lanes`]
    /// under [`Residency::Off`] at [`KdaLanes::One`]).
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Glm5nextModel, GpuError> {
        Body::open_placed_lanes(
            file,
            plan,
            inputs,
            card,
            host,
            Residency::Off,
            KdaLanes::One,
        )
    }

    /// [`Body::open_placed`] under `residency`, one KDA lane
    /// ([`Body::open_placed_lanes`] at [`KdaLanes::One`]).
    pub fn open_placed_with(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
    ) -> Result<Glm5nextModel, GpuError> {
        Body::open_placed_lanes(file, plan, inputs, card, host, residency, KdaLanes::One)
    }

    /// [`Body::open_placed`] under `residency` ([`ResidencySpec`]) at `lanes`
    /// KDA lanes — two for a load that verifies two rows ([`Rows`]), one
    /// otherwise; the pass's rows in flight are [`LOAD_ROWS`] either way: the
    /// plan's host set also holds each layer's churn pool — the card's
    /// experts past the pinned ones — and the body keeps the load's
    /// [`ResidencyGlue`], whose machine it starts once its pieces are sized
    /// ([`HostServed::start_residency`]). The plan's expert tier card, when
    /// it names one, is hung under the host tier ([`TierOpen::of_machine`]):
    /// its routed segments load onto that card, the slot map sends their
    /// experts to it, and each layer that holds one runs as a tier layer
    /// ([`crate::tier`]); a residency machine beside it moves the stage
    /// card's experts alone. Refused by name: more tier cards than the host
    /// tier serves and a slot map that puts an expert on two devices (each
    /// before anything uploads), a
    /// plan of more than one stage card, a layer kind or a width no kernel
    /// here runs, routed layers that are not one run, and the machine's own
    /// refusals at the load (the churn pool past the host's headroom, a layer
    /// whose stacks the common file source cannot take), a plan made for
    /// other lanes ([`PlanInputs::plan_lanes`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's file, plan, inputs and card, the host tier's levers and residency, and the KDA lanes (rust-quality R8)"
    )]
    pub fn open_placed_lanes(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
        lanes: KdaLanes,
    ) -> Result<Glm5nextModel, GpuError> {
        Body::open_placed_slots(file, plan, inputs, card, host, residency, lanes, 1)
    }

    /// [`Body::open_placed_lanes`] for a load that serves `slots` resident
    /// sequences ([`Slots`]): the plan must have counted them
    /// ([`PlanInputs::plan_slots`]), and a sequence past the count is
    /// refused by name. Refused as [`Body::open_placed_lanes`] refuses, a
    /// plan made for another count among them.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's file, plan, inputs and card, the host tier's levers and residency, the KDA lanes and the resident sequences (rust-quality R8)"
    )]
    pub fn open_placed_slots(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
        lanes: KdaLanes,
        slots: usize,
    ) -> Result<Glm5nextModel, GpuError> {
        let tiers = TierOpen::of_machine(plan.machine);
        refuse_tiers_before_upload(plan, inputs, card, &tiers, None)?;
        refuse_other_lanes(plan, inputs, card, (lanes, slots), None)?;
        let kinds: Vec<Layer> = inputs.spec.layers.iter().map(Layer::of).collect();
        // The host tier's run — the layers the body's slot map holds, which
        // the machine's per-layer lists cover.
        let map_layers = host_run(&inputs.spec.layers, None)?.len();
        // The most rows one pass runs: the load's rows in flight.
        let spec = ResidencySpec {
            lever: residency,
            delay: swap::LIVE_DELAY,
            deadline: swap::DEADLINE,
            top_k: N_USED,
            max_rows: LOAD_ROWS,
            stacks: Arc::new(swap::Glm5Stacks::of(inputs, map_layers, None)?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |stream, _, layers, w| Body::derive(stream, &kinds, layers, w),
            |gpu, file, w, set, glue| {
                Body::load_placed(
                    gpu,
                    file,
                    w,
                    (plan, inputs, card, &tiers),
                    host,
                    set,
                    glue,
                    residency,
                    (lanes, slots),
                    None,
                )
            },
        )
    }

    /// [`Body::open_placed`] with the next-token layer beside the chain, the
    /// load's slot map for the model's life ([`Body::open_placed_nextn_with`]
    /// under [`Residency::Off`]).
    pub fn open_placed_nextn(
        file: Split,
        plan: &NextnPlan<'_>,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Glm5nextModel, GpuError> {
        Body::open_placed_nextn_with(file, plan, inputs, nextn, card, host, Residency::Off)
    }

    /// [`Body::open_placed`] with the next-token layer beside the chain under
    /// `residency`: card `card` of `plan`'s target plan resident as
    /// [`Body::open_placed_lanes`] makes it, the layer `nextn` describes
    /// resident as `plan`'s NextN plan places it ([`nextn::Nextn`]), and its
    /// routed experts served by the host tier as the run's last layer (the
    /// slot map's row for it every expert on the host), the host tier's batch
    /// port made for the walk's host leg. Those experts join the plan's host
    /// set ([`NextnPlan::host_runs`]), read in and locked with it as `host`
    /// asks, so no walk takes their first-touch reads. Its KDA state holds
    /// two lanes: the draft's rows are verified two at a time. Under
    /// `mid-p<P>-s<S>` the residency machine runs over the slot map's layers,
    /// the next-token layer's among them: the file source's parts span the
    /// card's trunk layers, so that layer has no card part and no card slot
    /// and the machine moves none of its experts, and no pass routes it (the
    /// draft's walks serve it through the batch port, which notes no id), so
    /// the machine lists it as unrouted; the host set also holds the
    /// churn pool, which the target plan's host headroom less the layer's
    /// host experts must take. The target plan's expert tier card, when it
    /// names one, is hung as [`Body::open_placed_lanes`] hangs it: it serves
    /// the target's steps, verifies and prompt batches, and the draft's walk
    /// stays on the stage card and the host. Refused as
    /// [`Body::open_placed_lanes`] refuses, and by name for a next-token
    /// layer that does not follow the host run and for a slot map that sends
    /// one of that layer's experts to a tier ([`refuse_nextn_on_tier`]).
    pub fn open_placed_nextn_with(
        file: Split,
        plan: &NextnPlan<'_>,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
    ) -> Result<Glm5nextModel, GpuError> {
        Body::open_placed_nextn_slots(file, plan, inputs, nextn, card, host, residency, 1)
    }

    /// [`Body::open_placed_nextn_with`] for a load that serves `slots`
    /// resident sequences ([`Slots`]): the plan must have counted them
    /// ([`PlanInputs::plan_nextn_slots`]: the trunk's stores and the
    /// next-token layer's store each for every sequence), and a sequence
    /// past the count is refused by name. Refused as
    /// [`Body::open_placed_nextn_with`] refuses, a plan made for another
    /// count among them.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's file, plans, inputs and card, the host tier's levers and residency, and the resident sequences (rust-quality R8)"
    )]
    pub fn open_placed_nextn_slots(
        file: Split,
        plan: &NextnPlan<'_>,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
        slots: usize,
    ) -> Result<Glm5nextModel, GpuError> {
        let target = &plan.plan;
        let tiers = TierOpen::of_machine(target.machine);
        refuse_tiers_before_upload(target, inputs, card, &tiers, Some(nextn.index))?;
        refuse_other_lanes(target, inputs, card, (KdaLanes::Two, slots), Some(nextn))?;
        let kinds: Vec<Layer> = inputs.spec.layers.iter().map(Layer::of).collect();
        // The slot map's layers: the trunk's routed run and the next-token
        // layer after it, which the machine's per-layer lists cover.
        let map_layers = host_run(&inputs.spec.layers, Some(nextn.index))?.len();
        let spec = ResidencySpec {
            lever: residency,
            delay: swap::LIVE_DELAY,
            deadline: swap::DEADLINE,
            top_k: N_USED,
            max_rows: LOAD_ROWS,
            stacks: Arc::new(swap::Glm5Stacks::of(inputs, map_layers, Some(nextn.index))?),
        };
        let hosted = plan.host_runs().map_err(|e| GpuError::plan(WHAT, e))?;
        GpuModel::load_placed_hosting(
            file,
            target,
            card,
            host,
            spec,
            (&hosted.0, hosted.1),
            |stream, _, layers, w| Body::derive(stream, &kinds, layers, w),
            |gpu, file, w, set, glue| {
                Body::load_placed(
                    gpu,
                    file,
                    w,
                    (target, inputs, card, &tiers),
                    host,
                    set,
                    glue,
                    residency,
                    (KdaLanes::Two, slots),
                    Some((plan, nextn)),
                )
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
    /// `glue` is the load's residency side ([`ResidencyGlue`]) for the
    /// `lever` the load ran under, whose machine the body starts once its
    /// pieces are sized. Each KDA layer's state holds `lanes` lanes, a
    /// sequence's; the step's rows, the card experts' and the host
    /// boundary's are the load's [`LOAD_ROWS`] rows in flight, whatever the
    /// lanes; stores whose bytes are not a sequence of the plan's are refused by
    /// name. The plan counted `seqs` resident sequences, the most the body
    /// serves ([`Slots`]). With `nextn` the next-token layer joins the host
    /// run's end and loads beside the chain ([`Body::open_placed_nextn`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card, file, weights, plan, inputs and tiers, the host tier's residency and levers, the residency glue, the KDA lanes and resident sequences, and the NextN plan (rust-quality R8)"
    )]
    fn load_placed(
        gpu: &Gpu,
        file: &Arc<Split>,
        w: &Weights,
        (plan, inputs, card, tiers): (&Plan<'_>, &PlanInputs, usize, &[TierOpen]),
        host: HostCfg,
        residency: HostResidency,
        glue: ResidencyGlue,
        lever: Residency,
        (lanes, seqs): (KdaLanes, usize),
        nextn: Option<(&NextnPlan<'_>, &NextnInputs)>,
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
        let run = host_run(&spec.layers, nextn.map(|(_, n)| n.index))?;
        let dims = dims_of(inputs)?;
        let tier_cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
        let map = SlotMap::of_plan_tiers(plan, card, &tier_cards, run.clone(), N_EXPERT)?;
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
            .map(|c| store(stream, c.kind, &dims, ctx, lanes))
            .collect::<Result<Vec<_>, _>>()?;
        let held_bytes: usize = stores.iter().map(Store::bytes).sum();
        let planned = inputs
            .kv
            .with_lanes(lanes)
            .bytes(layers.clone(), plan.ctx_max);
        if held_bytes as u64 != planned {
            return Err(shape(format!(
                "the stores hold {held_bytes} device bytes at {} KDA lanes; a sequence of the \
                 plan's counts {planned}",
                lanes.count()
            )));
        }
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            run.len(),
            N_EXPERT,
        )?);
        let experts_on_card = CardExperts::new(
            gpu,
            w,
            &spec.layers,
            &map,
            dims.embd,
            hp.expert_ff,
            LOAD_ROWS,
        )?;
        let boundary = Boundary::with_rows(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: dims.embd,
                n_used: N_USED,
            },
            LOAD_ROWS,
        )?;
        let file = Arc::clone(file);
        let experts = GlmHost::build(Arc::clone(&file), hp, run.clone(), host.r8)?;
        let tier_cards = tiers
            .iter()
            .enumerate()
            .map(|(i, t)| {
                tier::open_tier(
                    gpu,
                    &file,
                    plan,
                    t,
                    i,
                    &spec.layers,
                    &map,
                    [dims.embd, hp.expert_ff, LOAD_ROWS],
                    host.card_dontneed,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut hybrid = Hybrid::new(boundary, map, experts, run.len())?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(residency);
        if !tier_cards.is_empty() {
            let cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
            hybrid.attach_tiers(tier_cards, gpu)?;
            // The tier's batch staging is reserved on its card by the plan, so
            // it is made at load, not at the first prompt.
            hybrid.prepare_batch(gpu.context(), prefill::T_MAX.min(ctx))?;
            hybrid.check_tier_reserves(plan.machine, &cards)?;
        }
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
        let who = Identity {
            arch: "glm5next",
            file: embd
                .file
                .shard_path(0)
                .ok_or(GpuError::State {
                    what: WHAT,
                    missing: "the file's first shard",
                })?
                .to_path_buf(),
            card: gpu.device_name()?,
            ctx,
        };
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
            lanes,
            wrote: nextn::Wrote::default(),
            dense,
            ckpt,
            prompt: prefill::PromptState::new(),
            residency_glue: glue,
            residency: lever,
            nextn: None,
            planned: None,
            who,
            slots_planned: seqs,
            slots_made: 1,
        };
        body.s.lane.copy_from_host(stream, &lane)?;
        stream.synchronize()?;
        refuse_scratch_past(plan, card, body.scratch_bytes())?;
        if let Some((np, ni)) = nextn {
            let layer = nextn::Nextn::open(
                gpu,
                &body.embd.file,
                np,
                inputs,
                ni,
                w,
                &body.dims,
                (ctx, seqs),
                host.card_dontneed,
            )?;
            body.nextn = Some(Box::new(layer));
            body.prepare_port(gpu)?;
        }
        // Last: the machine frees each layer's spare slots, and every piece
        // above sized itself from the load's map, capacity = live.
        body.start_residency(gpu)?;
        Ok(body)
    }

    /// Positions every store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The lanes of every KDA layer's state: two on a load that verifies two
    /// rows, one otherwise.
    #[must_use]
    pub fn lanes(&self) -> KdaLanes {
        self.lanes
    }

    /// The rows whose final streams a sequence keeps ([`GlmSlot`]): its own
    /// verify's, a row a lane, the step's row 0 first — what the draft's
    /// walks read by position, and what the plan counts a sequence by
    /// (`place`'s bytes beside the stores). The load's other rows in flight
    /// hold no sequence's state past a call.
    pub(crate) fn seq_rows(&self) -> usize {
        self.lanes.count()
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

    /// Device bytes of the load's scratch, what the plan's card term
    /// `scratch_bytes` sets aside: the [`LOAD_ROWS`] rows in flight with
    /// the live sequence's lane word, the card experts' rows and tier side,
    /// the host boundary and the slot map's card copy. A load past the term
    /// is refused by name.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.card.bytes()
            + self.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
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

    /// Keep every residency boundary's report from now on, for
    /// [`Body::take_residency_passes`]: a binary that prints the `residency
    /// pass` records asks once, with the boundaries it takes at most between
    /// two takes, `passes`, so a pass logs its report without growing the
    /// log. Nothing is kept until then.
    pub fn log_residency(&mut self, passes: usize) {
        self.residency_glue.log_passes(passes);
    }

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.residency_glue.source()
    }

    /// The residency boundaries' reports since the last take, in order,
    /// each with the kind of the pass it ended; the log keeps its capacity.
    pub fn take_residency_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.residency_glue.take_passes()
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
            Parts::of(
                &self.k,
                &self.dims,
                &self.cfg,
                &self.names,
                &mut self.s,
                &mut self.stores,
                &self.slots,
                &mut self.card,
                self.taps.as_deref_mut(),
            ),
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
    /// With a verify waiting for its commit, a cut into its rows keeps them
    /// by the lanes' rule (`Rule`, [`pair`]).
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        if let Some(k) = self.s.lanes.kept(n, pos, self.held) {
            k
        } else if self.held == pos {
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
    /// the KDA layers' committed stores copied to a host slot, or nothing
    /// where one stands. Refused by name when the stores hold other
    /// positions (a step failed past its launch, a verify waits for its
    /// commit). Waits for the copies.
    pub fn checkpoint(&mut self, gpu: &Gpu, pos: u32) -> Result<Take, GpuError> {
        self.stores_at(pos)?;
        self.apply_cut(gpu.stream())?;
        let lane = self.s.lanes.committed();
        self.ckpt
            .take(gpu.stream(), pos, &mut copied(&mut self.stores, lane)?)
    }

    /// The live sequence's waiting cut carried out ([`cut_into`]).
    fn apply_cut(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let lane = self.s.lanes.committed();
        cut_into(stream, &mut self.ckpt, &mut self.stores, lane, self.held)
    }

    /// The live sequence's stores at `pos` ([`held_at`]).
    fn stores_at(&self, pos: u32) -> Result<(), GpuError> {
        held_at(&self.s.lanes, self.held, pos)
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
                    Plant::Group(_) => "the planted failure in a prompt group's walk",
                },
            });
        }
        Ok(())
    }
}

/// Where a gate plants a failure ([`Body::plant`]): a step's in its
/// refresh, before any input reaches the card, or once its chain has run and
/// its host legs been served; or a prompt group's walk at the front of unit
/// `u`'s first routed layer (`Group(u)`, taken by the next group that holds
/// that unit), after the units before it and the dense lead of every unit
/// have run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plant {
    BeforeLaunch,
    AfterLaunch,
    Group(usize),
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

/// Layer `kind`'s store at `ctx` positions: a KDA layer's state of `lanes`
/// lanes, lane 0 stamped at position 0, and its conv ring, a latent layer's
/// latent and index rows and its pool plane.
fn store(
    stream: &CudaStream,
    kind: Layer,
    d: &Dims,
    ctx: usize,
    lanes: KdaLanes,
) -> Result<Store, GpuError> {
    Ok(match kind.mixer {
        MixerKind::DeltaRule => Store::Kda {
            state: LaneState::zeroed(stream, d.kda.state_len(), lanes)?,
            stamp: DeviceBuffer::from_host(stream, &stamps(0, 0, lanes)?[..lanes.count()])?,
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
        self.refresh_row(stream, input, 0)
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
        for row in self.s.rows_mut() {
            for s in &mut row.streams {
                s.zero_async(stream)?;
            }
        }
        self.s.lane.copy_from_host(stream, &[0])?;
        self.s.lanes = pair::Lanes::default();
        self.wrote = nextn::Wrote::default();
        if let Some(n) = self.nextn.as_deref_mut() {
            n.forget();
        }
        self.held = 0;
        self.plant = None;
        self.planned = None;
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
            + self.scratch_bytes()
            + self.prompt.bytes()
            + self
                .taps
                .as_ref()
                .map_or(0, |t| t.iter().map(DeviceBuffer::num_bytes).sum())
            + self
                .nextn
                .as_deref()
                .map_or(0, |n| n.resident_bytes() + n.arena_bytes())
    }

    fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    fn host(&mut self) -> Option<&mut Body> {
        Some(self)
    }
}

impl Rollback for Body {
    /// A cut to `pos` ([`Body::cut`]).
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.cut(pos)
    }

    /// A verify's commit, or else a cut ([`Body::commit`]).
    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        self.commit(gpu, pos)
    }
}

/// One resident sequence of the load ([`Slots`], what a select exchanges):
/// its per-layer stores, the lane word every KDA launch reads and the host's
/// side of the lanes, the positions the stores hold, the KDA layers'
/// checkpoints, the positions each target arena's rows hold and the final
/// streams of each row of its own verify, a row a lane ([`Body::seq_rows`]:
/// the step's row and the verify's, which the draft's walks read by
/// position), and on a NextN load the layer's side
/// ([`nextn::NextnSeq`]). Every one is a buffer handle or a host value, so a
/// select moves pointers, and each slot's captured chains keep addressing
/// the buffers that were live when they were captured.
pub struct GlmSlot {
    stores: Vec<Store>,
    lane: DeviceBuffer<u32>,
    lanes: pair::Lanes,
    held: u32,
    ckpt: Checkpoints,
    wrote: nextn::Wrote,
    /// Each of [`Body::seq_rows`] rows' final streams, row 0's (the
    /// step's) first.
    rows: Vec<DeviceBuffer<f32>>,
    draft: Option<nextn::NextnSeq>,
}

impl Slots for Body {
    type Seq = GlmSlot;

    /// A sequence of the load's shape in the state the load leaves: zeroed
    /// stores with lane 0 stamped at position 0, the lane word at lane 0, no
    /// position held and no checkpoint, every arena holding no position, its
    /// rows zeroed, and on a NextN load the layer's side as
    /// [`nextn::Nextn::new_seq`] makes it. Refused by name past the
    /// sequences the plan counted ([`Body::open_placed_slots`]): a sequence
    /// the plan did not count would take card bytes the plan gave to
    /// experts. Load-time allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<GlmSlot, GpuError> {
        const WHAT_N: &str = "glm5next Body::new_seq";
        if self.slots_made >= self.slots_planned {
            return Err(GpuError::Shape {
                what: WHAT_N,
                detail: format!(
                    "sequence {} of a load whose plan counted {} resident sequences; plan and \
                     open it for as many as it serves (PlanInputs::plan_slots, \
                     Body::open_placed_slots)",
                    self.slots_made + 1,
                    self.slots_planned
                ),
            });
        }
        let stream = gpu.stream();
        let stores = self
            .cfg
            .iter()
            .map(|c| store(stream, c.kind, &self.dims, self.ctx, self.lanes))
            .collect::<Result<Vec<_>, _>>()?;
        let ckpt = Checkpoints::new(
            gpu.context(),
            self.ckpt.lens().to_vec(),
            HOST_BUDGET,
            CHECKPOINT_EVERY,
        )?;
        let wide = HC_STREAMS * self.dims.embd;
        let rows = (0..self.seq_rows())
            .map(|_| DeviceBuffer::zeroed(stream, wide))
            .collect::<Result<Vec<_>, _>>()?;
        let draft = self
            .nextn
            .as_deref()
            .map(|n| n.new_seq(stream))
            .transpose()?;
        let slot = GlmSlot {
            stores,
            lane: DeviceBuffer::zeroed(stream, 1)?,
            lanes: pair::Lanes::default(),
            held: 0,
            ckpt,
            wrote: nextn::Wrote::default(),
            rows,
            draft,
        };
        // Last, so a sequence whose allocation failed is not counted.
        self.slots_made += 1;
        Ok(slot)
    }

    /// Exchange the live sequence with `seq`: pointer moves only, nothing
    /// copied, captured or synchronized. The prompt batch's arena is the
    /// load's, so after the exchange its rows are neither side's: both
    /// records of it are emptied, and a walk that would read it is refused
    /// by name. Refused by name, moving nothing, while a verify waits for
    /// its commit (its rows stand in the live state's lanes), and for a
    /// sequence whose NextN side or rows do not match the load's.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut GlmSlot) -> Result<(), GpuError> {
        const WHAT_W: &str = "glm5next Body::swap_seq";
        self.s
            .lanes
            .refuse_if_waiting(self.held)
            .map_err(|e| GpuError::Shape {
                what: WHAT_W,
                detail: format!("a select: {e}"),
            })?;
        if self.nextn.is_some() != seq.draft.is_some() {
            return Err(GpuError::State {
                what: WHAT_W,
                missing: "the NextN side of every sequence (a NextN load serves sequences that \
                          each carry one)",
            });
        }
        let rows = self.seq_rows();
        if seq.rows.len() != rows {
            return Err(GpuError::Shape {
                what: WHAT_W,
                detail: format!(
                    "a sequence of {} rows' streams on a load whose sequences keep {rows}",
                    seq.rows.len()
                ),
            });
        }
        std::mem::swap(&mut self.stores, &mut seq.stores);
        std::mem::swap(&mut self.s.lane, &mut seq.lane);
        std::mem::swap(&mut self.s.lanes, &mut seq.lanes);
        std::mem::swap(&mut self.held, &mut seq.held);
        std::mem::swap(&mut self.ckpt, &mut seq.ckpt);
        std::mem::swap(&mut self.wrote, &mut seq.wrote);
        self.wrote.prefill = nextn::Held::NONE;
        seq.wrote.prefill = nextn::Held::NONE;
        let fin = program::final_streams(self.cfg.len());
        for (row, own) in self.s.rows_mut().take(rows).zip(seq.rows.iter_mut()) {
            std::mem::swap(&mut row.streams[fin], own);
        }
        if let (Some(n), Some(d)) = (self.nextn.as_deref_mut(), seq.draft.as_mut()) {
            n.swap_seq(d);
        }
        Ok(())
    }

    /// Device bytes one sequence holds: its stores, its lane word, its
    /// rows' final streams ([`Body::seq_rows`]) and, on a NextN load, the
    /// layer's side — the terms a plan counts a sequence by
    /// (`place::PlanInputs::seq_terms`, one sequence's `SeqTerms::bytes`),
    /// which a gate holds equal. The other rows in flight are the load's.
    fn seq_bytes(&self) -> usize {
        let fin = program::final_streams(self.cfg.len());
        self.store_bytes()
            + self.s.lane.num_bytes()
            + self
                .s
                .rows()
                .take(self.seq_rows())
                .map(|r| r.streams[fin].num_bytes())
                .sum::<usize>()
            + self.nextn.as_deref().map_or(0, nextn::Nextn::seq_bytes)
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

    /// The host tier's residency boundary at `at` ([`Hybrid::swap_at`]), its
    /// report logged when a binary asked ([`Body::log_residency`]).
    fn at_boundary(&mut self, stream: &CudaStream, at: BoundaryAt) -> Result<(), GpuError> {
        self.residency_glue
            .at_boundary(&mut self.hybrid, stream, at)
    }

    fn keep_rows(&mut self, kept: KeptRows, kind: PassKind) -> Result<(), GpuError> {
        self.residency_glue.keep_rows(&mut self.hybrid, kept, kind)
    }

    fn residency_reset(&mut self, stream: &CudaStream) -> Result<Option<ResetReport>, GpuError> {
        self.residency_glue.reset(&mut self.hybrid, stream)
    }

    fn stop_residency(&mut self) {
        self.residency_glue.stop(&mut self.hybrid);
    }

    fn start_residency(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.residency_glue.start(
            &mut self.hybrid,
            gpu.context(),
            gpu.stream(),
            Arc::clone(&self.slots),
        )
    }
}

impl Drop for Body {
    /// The residency machine stops before any field frees card or pinned
    /// memory ([`Hybrid::stop_swap`]): the host tier is the first field and
    /// stops it in its own drop, and this holds that order as the machine's
    /// rule rather than the field list's — a field that waits on the card
    /// (a synchronize, a free) dropping ahead of the tier would hold that
    /// wait until the machine's own drop, which comes later.
    fn drop(&mut self) {
        self.hybrid.stop_swap();
    }
}

/// The steps feed under the residency machine, refused by name: each prompt
/// id would end a decode pass the residency rule counts, where the prompt
/// call keeps 0 rows (V4.1's rule, `gpu_deepseek41::body`'s
/// `refuse_steps_under_residency`, at its load there).
pub(crate) fn refuse_steps_under_residency(m: &Glm5nextModel) -> Result<(), GpuError> {
    if m.body(WHAT)?.residency != Residency::Off {
        return Err(GpuError::Shape {
            what: "BLOOMERY_PREFILL",
            detail: "steps beside BLOOMERY_RESIDENCY: each prompt id would end a decode pass \
                     the residency rule counts, where the prompt call keeps 0 rows; use \
                     BLOOMERY_PREFILL=batch or BLOOMERY_RESIDENCY=off"
                .to_string(),
        });
    }
    Ok(())
}
