//! The prompt batch: a prompt of `P` ids fed in batches of at most
//! [`T_MAX`] positions ([`prefill`]), which leaves the model where `P`
//! decode steps over the same ids leave it in everything a later step
//! reads: every KDA layer's state and conv ring, every latent layer's latent
//! and index rows and pool plane, the checkpoints, and the last position's
//! logits. A batch of at most [`CHUNK`] positions writes the steps' bits; one
//! of [`GEMM_FROM`] or more runs its mixers' Q8_0 projections on the GEMM
//! ([`GemmFront`]: a KDA mixer's input projections but the decay's pair and
//! its output projection, a latent mixer's joined projection), whose
//! activations are quantized to 32-value int8 blocks, so it writes the
//! GEMM's bits there. Either way a token's bits are a function of the ids
//! before it and of which side of [`GEMM_FROM`] each batch they ran in fell,
//! not of the batches' cut otherwise: the GEMM's output for a column is that
//! of any other column count, and every other launch writes a column what
//! the step's one-column launch writes.
//!
//! A call is cut at its checkpoint marks
//! ([`bloomery_gpu::checkpoint::Checkpoints::marks`]: its start, every
//! multiple of [`CHECKPOINT_EVERY`](super::CHECKPOINT_EVERY) inside it, its
//! end), each run
//! between two marks into batches of at most [`T_MAX`] positions
//! ([`call_batches`]), and each mark's checkpoint is taken where the steps
//! take it, so a cut keeps the same points after either feed. The call's
//! batches run in groups of [`set_prefill_group`]'s size (`place::groups`: a
//! lone last batch joins the group before it); a group is one walk of the
//! runtime's layer schedule at the point `(G, T, Batch)`
//! ([`runtime::sched::walk`]), one unit a batch, through the host tier's
//! batch port ([`BatchLeg`]): layer by layer, each unit's front enqueued
//! ahead of the previous unit's host serve when the group holds two or more.
//! The walk serves only the layers with a host leg
//! ([`LayerProgram::host_leg`]: the dense lead downloads nothing). Per layer:
//! - the front: the mixer sub-layer whole and the feed-forward sub-layer's
//!   input, then the dense block and its `hc_post`, or the routed block's
//!   norm, its router over the batch's rows and the download of the rows,
//!   ids and weights to the host;
//! - the shadow (a routed layer): the card experts, where the slot map puts
//!   any, and the shared expert, while the host serves every token of the
//!   batch in one union call ([`GlmHost`]'s) and uploads the sums;
//! - the back: the host's sums plus the shadow's, and `hc_post`.
//!
//! The head runs after the last layer of the batch that holds the call's last
//! position, for that position alone.
//!
//! A group's units share every buffer one item writes and the same or the
//! next part of that item reads; what a unit's back, or its later layers,
//! read after the next unit's front ran is the unit's own (`UnitBufs`: the
//! streams, the feed-forward mix, the route's weights and tier places, the
//! positions). That holds only behind a dense prefix: a group of two or
//! more is refused by name on a load with a dense layer past a routed one.
//! Under layer-major order the units' positions still run in order through
//! every recurrent and cached store, so a group writes what its batches
//! alone write, bit for bit. A mark inside a group exists, per KDA layer,
//! only between that layer's fronts of the two units around it: its take is
//! opened before the group, gets each KDA layer's stores right after the
//! front of the unit that ends on it, and is sealed once the group's fault
//! word is read, or abandoned with the group.
//!
//! Every launch but the GEMM's writes, per token, what the step's one-token
//! launch writes, and what carries state from a position to the next (the
//! conv ring, the delta rule's state, the latent rows an attention reads)
//! runs in position order:
//! - one launch over the batch's `T` rows where the kernel takes any count:
//!   the RMS norms, `hc_pre` (in its token groups), the fold and `hc_post`,
//!   the KDA conv and prep, the delta step, the gated norm, the latent and
//!   index appends, the pool keys the batch's tokens complete, the router's
//!   two launches, the card places, the sums;
//! - the GEMM's projections over the batch's `T` rows from [`GEMM_FROM`];
//! - chunks of up to [`CHUNK`] tokens where it takes at most that many: the
//!   q8_0 gemvs (`q8_0_gemv_mcol`, `q8_0_gemv_heads_mcol`) the GEMM does
//!   not take, the
//!   gate·up·SwiGLU (`ds41_shexp_gate_up_q8_0_mcol`), the card experts'
//!   launches, the k-pool selector, and the attention, each token over the
//!   positions at and before its own — the batch's own rows and pools
//!   written before the first chunk attends.
//!
//! A latent layer's chunk attends as the step does at its tokens'
//! positions: while its last token sees at most the positions the indexer
//! keeps whole (`place::dense_positions`), every position at and before
//! each token's, which is the list the selector gives there; past them the
//! selector runs over the chunk's tokens (`crate::mla::select`) and the
//! attention reads the positions each token's list names, a token of the
//! chunk still within them listing every one of its positions.
//!
//! A latent layer's joined projection runs as its two row ranges — the
//! query's low rank, then the latent, the index key and the pool gate — so
//! the query's norm reads whole rows; each row's dot is the joined launch's.
//!
//! A call that fails is taken back to where it found the model, through the
//! checkpoint of its start, unless a fault poisoned it. The fault word is
//! read at the end of every group, before a mark's checkpoint can copy or
//! seal a state the fault condemned.

use std::mem::ManuallyDrop;
use std::ops::Range;
use std::time::Instant;

use bloomery_gpu::checkpoint::{Checkpoints, Pending};
use bloomery_gpu::fault::read_cards;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::kpool;
use bloomery_gpu::latent::{IndexKeyArgs, LATENT, LatentAppendArgs, Rows, pools_for};
use bloomery_gpu::linear::conv::KdaConvArgs;
use bloomery_gpu::linear::delta::{DeltaArgs, DeltaLanesArgs, KdaLanesArgs};
use bloomery_gpu::linear::norm_gate::NormGateArgs;
use bloomery_gpu::prompt_timing::{Mark, PromptStats, PromptTiming, nanos};
use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvHeadsMcolArgs, Q8_0GemvMcolArgs};
use bloomery_gpu::qsa::list_width;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, DeviceTensor, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::attn::{self, AttnArgs, SelectedRows};
use bloomery_gpu_deepseek41::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_STREAMS, HcPostArgs, HcPreScratch, HcQ8Params, HcQ8PreArgs,
};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::DeviceBuffer;
use gguf::GgmlType;
use gguf::quant::dequant_row;
use model::arch::glm5next::names::Sub;
use model::arch::glm5next::place;
use model::moe::UNION_MAX_COLS;
use runtime::layer::{FfnKind, MixerKind};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use super::nextn::GlmArena;
use super::{
    Body, Dims, Embedding, Glm5nextModel, Parts, Store, copied, f32t, f32v, prompt, q8, shape,
    weight,
};
use crate::ffn::{self, CardRows};
use crate::gemm::{FrontShape, GemmFront, KdaInNames, KdaInRows, LatentInRows};
use crate::host::GlmHost;
use crate::mla::{self, Select};
use crate::tensors::{FfnNames, MixerNames, other_kind};
use bloomery_gpu_deepseek41::chain::ffn::{CardAccTier, Places};

/// What the prompt batch's errors name.
const WHAT: &str = "glm5next prefill";

/// Positions one batch runs at most: the host union's columns.
pub const T_MAX: usize = UNION_MAX_COLS;

/// Tokens one chunk runs at most: the m-column kernels', and `hc_pre`'s
/// token group.
pub const CHUNK: usize = COL_GROUP;
const _: () = assert!(CHUNK == HC_MAX_TOKENS);

/// The fewest tokens a batch runs its GEMM projections over
/// ([`GemmFront`]: a KDA mixer's input and output projections but its decay
/// pair, a latent mixer's joined projection): a batch of at most a chunk
/// runs them as the step's gemv does, so it writes the step's bits.
pub const GEMM_FROM: usize = CHUNK + 1;

/// How a prompt is fed: in batches ([`prefill`]) or one decode step per id
/// ([`prompt`]) — the same-binary arm, which is the decode step and not a
/// second implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Batch,
    Steps,
}

impl PrefillMode {
    /// The mode [`PrefillMode::name`] names; `None` for any other word.
    #[must_use]
    pub fn from_name(name: &str) -> Option<PrefillMode> {
        [PrefillMode::Batch, PrefillMode::Steps]
            .into_iter()
            .find(|m| m.name() == name)
    }

    /// The name a `load` line prints and `--prefill` takes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PrefillMode::Batch => "batch",
            PrefillMode::Steps => "steps",
        }
    }
}

/// The body's prompt feed: its mode, the batches a group runs, the batch's
/// buffers once a batch feed made them, the unit a sink reads, whether a
/// call runs, and the batch walks' timing when a caller armed it.
pub(crate) struct PromptState {
    mode: PrefillMode,
    /// Batches a group runs layer by layer ([`set_prefill_group`]): one until
    /// set.
    group: usize,
    batch: Option<Box<Batch>>,
    /// The unit whose final streams [`PromptState::final_streams`] hands out:
    /// the one a sink is called for.
    sink: usize,
    /// A prompt call by batches runs: its group is not changed under it.
    in_call: bool,
    /// The batch walks' timing ([`set_prompt_stats`]); unarmed, a walk
    /// records nothing and waits for nothing beyond its own serves.
    timing: Option<PromptTiming>,
    /// The batch walks' route taps ([`set_prompt_route_taps`]); unarmed, a
    /// walk copies nothing.
    routes: Option<RouteTaps>,
    /// The batch walks' planted routes ([`plant_prompt_routes`]).
    plant_routes: Option<RoutePlant>,
}

impl PromptState {
    /// The steps feed with no buffers: what a load starts as.
    pub(crate) fn new() -> PromptState {
        PromptState {
            mode: PrefillMode::Steps,
            group: 1,
            batch: None,
            sink: 0,
            in_call: false,
            timing: None,
            routes: None,
            plant_routes: None,
        }
    }

    /// Device bytes of the batch's buffers, 0 before they are made, and of
    /// the route taps while armed.
    pub(crate) fn bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.bytes())
            + self.routes.as_ref().map_or(0, RouteTaps::bytes)
            + self.plant_routes.as_ref().map_or(0, |p| {
                p.layers
                    .iter()
                    .flatten()
                    .map(TapPlanes::bytes)
                    .sum::<usize>()
            })
    }

    /// The stream buffer `fin` of the sink's unit's rows, where a group's
    /// walk leaves every row's final streams, and the rows it holds; `None`
    /// before the batch's buffers are made.
    pub(crate) fn final_streams(&self, fin: usize) -> Option<(&DeviceBuffer<f32>, usize)> {
        let b = self.batch.as_deref()?;
        Some((b.units.get(self.sink)?.streams.get(fin)?, b.bufs.cap))
    }
}

/// The batch walks' route taps, for the gates: each routed layer's router
/// scores ([`N_EXPERT`] a position, unbiased), picks ([`N_USED`] a position,
/// in slot order) and their weights at every position below `rows` a batch
/// ran, copied by the walk right after the router; `None` for a dense layer.
struct RouteTaps {
    rows: usize,
    layers: Vec<Option<TapPlanes>>,
    /// The positions whose rows the batch calls since the arming wrote, in
    /// one run: a call that starts inside or at the end of it extends it,
    /// any other starts it again.
    fed: Range<usize>,
}

/// One routed layer's route planes for `rows` positions: the scores (a
/// tap's only), the picks and their weights.
struct TapPlanes {
    probs: Option<DeviceBuffer<f32>>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

impl TapPlanes {
    fn bytes(&self) -> usize {
        self.probs.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.ids.num_bytes()
            + self.weights.num_bytes()
    }
}

impl RouteTaps {
    fn bytes(&self) -> usize {
        self.layers.iter().flatten().map(TapPlanes::bytes).sum()
    }
}

/// The batch walks' planted routes, for the gates: each routed layer's picks
/// and weights at every position below `rows`, written over the router's
/// own right after it (and after the taps' copy), so every expert the
/// batch runs and every weight it sums with are the plant's.
struct RoutePlant {
    rows: usize,
    layers: Vec<Option<TapPlanes>>,
}

/// One routed layer's taps read back ([`prompt_route_taps`]): the router's
/// scores, [`N_EXPERT`] a position, its picks and their weights, [`N_USED`]
/// a position, position-major from position 0.
#[derive(Clone, Debug)]
pub struct RouteTapRows {
    pub probs: Vec<f32>,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Arm the batch walks' route taps for positions below `rows`, or disarm
/// them with 0: each routed layer's scores, picks and weights at every
/// position a batch runs, which [`prompt_route_taps`] reads back. A batch
/// call past `rows` is refused by name before it runs. Load-time
/// allocation, for the gates; refused inside a call.
pub fn set_prompt_route_taps(m: &mut Glm5nextModel, rows: usize) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    if body.prompt.in_call {
        return Err(shape(format!(
            "route taps of {rows} rows set inside a prompt call"
        )));
    }
    body.prompt.routes = None;
    if rows == 0 {
        return Ok(());
    }
    let stream = gpu.stream();
    let layers = body
        .cfg
        .iter()
        .map(|c| {
            if !c.kind.host_leg() {
                return Ok(None);
            }
            Ok(Some(TapPlanes {
                probs: Some(DeviceBuffer::zeroed(stream, rows * N_EXPERT)?),
                ids: DeviceBuffer::zeroed(stream, rows * N_USED)?,
                weights: DeviceBuffer::zeroed(stream, rows * N_USED)?,
            }))
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
    body.prompt.routes = Some(RouteTaps {
        rows,
        layers,
        fed: 0..0,
    });
    Ok(())
}

/// Each layer's route taps for positions `0 .. n` ([`set_prompt_route_taps`]),
/// `None` for a dense layer. Blocking. Refused by name while unarmed and for
/// positions the batch calls since the arming did not write in one run from
/// position 0.
pub fn prompt_route_taps(
    m: &mut Glm5nextModel,
    n: usize,
) -> Result<Vec<Option<RouteTapRows>>, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let taps = body.prompt.routes.as_ref().ok_or(GpuError::State {
        what: WHAT,
        missing: "armed route taps (set_prompt_route_taps)",
    })?;
    if taps.fed.start != 0 || n > taps.fed.end {
        return Err(shape(format!(
            "route taps of positions 0..{n}: the batch calls since the arming wrote {:?}",
            taps.fed
        )));
    }
    let stream = gpu.stream();
    taps.layers
        .iter()
        .map(|t| {
            let Some(t) = t else {
                return Ok(None);
            };
            let probs = t.probs.as_ref().ok_or(GpuError::State {
                what: WHAT,
                missing: "a route tap's scores",
            })?;
            Ok(Some(RouteTapRows {
                probs: span(WHAT, probs, 0, n * N_EXPERT)?.to_host_vec(stream)?,
                ids: span(WHAT, &t.ids, 0, n * N_USED)?.to_host_vec(stream)?,
                weights: span(WHAT, &t.weights, 0, n * N_USED)?.to_host_vec(stream)?,
            }))
        })
        .collect()
}

/// Plant `routes` (each layer's, as [`prompt_route_taps`] reads them, for
/// positions `0 .. n`) into the batch walks, or take the plant back with
/// `None`: from the next call each routed layer's picks and weights at a
/// position below `n` are the plant's, written over the router's own. A
/// batch call past `n` is refused by name before it runs, and so is a plant
/// whose layers are not the load's routed ones or whose rows are not `n`
/// positions. Load-time allocation, for the gates; refused inside a call.
pub fn plant_prompt_routes(
    m: &mut Glm5nextModel,
    routes: Option<(&[Option<RouteTapRows>], usize)>,
) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    if body.prompt.in_call {
        return Err(shape("a route plant set inside a prompt call".to_string()));
    }
    body.prompt.plant_routes = None;
    let Some((routes, n)) = routes else {
        return Ok(());
    };
    if routes.len() != body.cfg.len() || n == 0 {
        return Err(shape(format!(
            "a route plant of {} layers and {n} positions on a load of {} layers",
            routes.len(),
            body.cfg.len()
        )));
    }
    let stream = gpu.stream();
    let layers = body
        .cfg
        .iter()
        .zip(routes)
        .enumerate()
        .map(|(l, (c, r))| match (c.kind.host_leg(), r) {
            (false, None) => Ok(None),
            (true, Some(r)) if r.ids.len() == n * N_USED && r.weights.len() == n * N_USED => {
                Ok(Some(TapPlanes {
                    probs: None,
                    ids: DeviceBuffer::from_host(stream, &r.ids)?,
                    weights: DeviceBuffer::from_host(stream, &r.weights)?,
                }))
            }
            _ => Err(shape(format!(
                "a route plant's layer {l}: {} picks and {} weights, the load's layer {}",
                r.as_ref().map_or(0, |r| r.ids.len()),
                r.as_ref().map_or(0, |r| r.weights.len()),
                if c.kind.host_leg() {
                    format!("routed, {n} positions of {N_USED}")
                } else {
                    "dense".to_string()
                }
            ))),
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
    stream.synchronize()?;
    body.prompt.plant_routes = Some(RoutePlant { rows: n, layers });
    Ok(())
}

/// A prompt call's tap ([`prompt_with`]): called after each unit the call
/// runs — a batch, or a step — with the arena its rows' final streams sit in,
/// the unit's first position and its rows, before the next unit overwrites
/// them.
pub type GlmPromptSink<'a> =
    dyn FnMut(&mut Glm5nextModel, GlmArena, u32, usize) -> Result<(), GpuError> + 'a;

/// The batch's buffers: the walk's shared ones, one set a unit of a group,
/// and the host sums the batch port uploads (lent to the port apart from
/// them).
struct Batch {
    bufs: Bufs,
    units: Vec<UnitBufs>,
    hsum: DeviceBuffer<f32>,
}

impl Batch {
    fn bytes(&self) -> usize {
        self.bufs.bytes()
            + self.units.iter().map(UnitBufs::bytes).sum::<usize>()
            + self.hsum.num_bytes()
    }
}

/// A group's unit's own buffers, for up to `cap` tokens: what the back of a
/// layer-batch, or a later layer of the same batch, reads after the next
/// unit's front has run (`runtime::sched`'s batch order: the shadow of item
/// `x`, the front of `x + 1`, the serve and the back of `x`), so the next
/// unit's front must not write it.
struct UnitBufs {
    /// The embedding rows on the host, four copies a token, before their
    /// upload.
    rows: Vec<f32>,
    /// Each token's position, its live count (the position plus one), and
    /// the attention's visible counts (no window row, then every position at
    /// and before its own), on the host.
    pos_host: Vec<u32>,
    cnt_host: Vec<u32>,
    vis_host: Vec<u32>,
    /// The four streams per token, ping-ponged as the step's, and which one
    /// the next sub-layer reads.
    streams: [DeviceBuffer<f32>; 2],
    cur: usize,
    /// The feed-forward sub-layer's mix, which its `hc_post` reads in the
    /// back.
    hc: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    cnt: DeviceBuffer<u32>,
    vis: DeviceBuffer<u32>,
    /// The route's weights, and with a tier each slot's tier place: a tiered
    /// layer's card sum reads them in the back.
    weights: DeviceBuffer<f32>,
    tsel: Option<DeviceBuffer<u32>>,
}

impl UnitBufs {
    /// One unit's buffers for up to `cap` tokens of `n` values, the tier
    /// places when `tiered`: [`unit_bytes`]' bytes, refused by name
    /// otherwise. Load-time or first-prompt only.
    fn new(gpu: &Gpu, n: usize, cap: usize, tiered: bool) -> Result<UnitBufs, GpuError> {
        let stream = gpu.stream();
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
        let zu = |len: usize| DeviceBuffer::<u32>::zeroed(stream, len);
        let ub = UnitBufs {
            rows: vec![0.0; cap * HC_STREAMS * n],
            pos_host: vec![0; cap],
            cnt_host: vec![0; cap],
            vis_host: vec![0; 2 * cap],
            streams: [z(cap * HC_STREAMS * n)?, z(cap * HC_STREAMS * n)?],
            cur: 0,
            hc: z(cap * HC_MIX)?,
            pos: zu(cap)?,
            cnt: zu(cap)?,
            vis: zu(2 * cap)?,
            weights: z(cap * N_USED)?,
            tsel: if tiered {
                Some(zu(cap * N_USED)?)
            } else {
                None
            },
        };
        let want = unit_bytes(n, cap, tiered);
        if ub.bytes() != want {
            return Err(shape(format!(
                "a prompt group's unit takes {} B; unit_bytes counts {want}",
                ub.bytes()
            )));
        }
        Ok(ub)
    }

    fn bytes(&self) -> usize {
        [&self.streams[0], &self.streams[1], &self.hc, &self.weights]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + [&self.pos, &self.cnt, &self.vis]
                .iter()
                .map(|b| b.num_bytes())
                .sum::<usize>()
            + self.tsel.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

/// Card bytes of one unit's buffers ([`UnitBufs`]) for up to `cap` tokens of
/// `n` values, the tier places when `tiered`: per token the two stream
/// buffers, the feed-forward mix, the position, the live and the two visible
/// counts, the route's weights and the tier places, four bytes a value.
fn unit_bytes(n: usize, cap: usize, tiered: bool) -> usize {
    4 * cap * (2 * HC_STREAMS * n + HC_MIX + 4 + N_USED + if tiered { N_USED } else { 0 })
}

/// Every buffer a group's units share besides the stores, for up to `cap`
/// tokens where a launch takes the batch whole and [`CHUNK`] where it takes
/// a chunk: each is written and read inside one part of one item, or by an
/// item's shadow and read by its back with no front between them, so the
/// units use it one after another in stream order.
struct Bufs {
    cap: usize,
    x: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    /// The fold `hc_post` writes beside the streams: never read.
    fold: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    hc_scratch: HcPreScratch,
    // A KDA mixer's, per token.
    qkv: DeviceBuffer<f32>,
    conv: DeviceBuffer<f32>,
    fa: DeviceBuffer<f32>,
    ga: DeviceBuffer<f32>,
    beta_raw: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    f: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    decay: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    // A latent mixer's: the projections per token, the heads per chunk.
    qa: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qabs: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    // The k-pool selector's, per chunk: the indexer query and head weights
    // (a token a column), the pools' scores and the lists.
    qi: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    list: DeviceBuffer<u32>,
    // The feed-forward blocks': the SwiGLU rows per chunk, the rest per
    // token.
    h: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    sh_y: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    sel: DeviceBuffer<u32>,
    // The card experts', per chunk: the rows' q8_1 form, the gate·up rows a
    // slot, per chunk width the q8_1 of its slots' columns, the downs — on a
    // load with an expert tier the whole batch's, which a tiered layer's
    // card sum reads beside the tier's rows after the serve.
    act_x: Q8Act,
    card_h: DeviceBuffer<f32>,
    act_h: Vec<Q8Act>,
    card_down: DeviceBuffer<f32>,
    /// The GEMM projections of a batch of [`GEMM_FROM`] tokens or more.
    front: GemmFront,
}

impl Bufs {
    /// The shared buffers for batches of up to `cap` tokens of `d`'s widths,
    /// `ff` the widest dense or shared-expert width, `expert_ff` a routed
    /// expert's, over stores of `ctx` positions, the card downs a batch's
    /// when `tiered` (a load with an expert tier). The attention's partials
    /// cover `ctx` keys: a chunk selects only on stores past the dense
    /// positions, which are a list's width, so they cover a list too.
    /// Load-time or first-prompt only.
    fn new(
        gpu: &Gpu,
        d: &Dims,
        [ff, expert_ff]: [usize; 2],
        ctx: usize,
        cap: usize,
        tiered: bool,
    ) -> Result<Bufs, GpuError> {
        let stream = gpu.stream();
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
        let zu = |len: usize| DeviceBuffer::<u32>::zeroed(stream, len);
        let (n, ch, v, nv) = (
            d.embd,
            d.kda.channels(),
            d.kda.n_v * bloomery_gpu::linear::HEAD,
            d.kda.n_v,
        );
        let head = bloomery_gpu::linear::HEAD;
        let rows = CHUNK * d.heads;
        let segs = attn::segments(0, ctx);
        let groups = cap.div_ceil(CHUNK);
        Ok(Bufs {
            cap,
            x: z(cap * n)?,
            xn: z(cap * n)?,
            out: z(cap * n)?,
            fold: z(cap * n)?,
            mixes: z(cap * HC_MIX)?,
            hc_scratch: HcPreScratch::with_groups(stream, HC_STREAMS * n, groups)?,
            qkv: z(cap * ch)?,
            conv: z(cap * ch)?,
            fa: z(cap * head)?,
            ga: z(cap * head)?,
            beta_raw: z(cap * nv)?,
            beta: z(cap * nv)?,
            f: z(cap * v)?,
            z: z(cap * v)?,
            decay: z(cap * v)?,
            o: z(cap * v)?,
            gated: z(cap * v)?,
            qa: z(cap * d.q_lora)?,
            kv: z(cap * kv_width(d))?,
            qr: z(cap * d.q_lora)?,
            q: z(CHUNK * d.heads * d.head_k)?,
            qabs: z(rows * LATENT)?,
            att: z(rows * LATENT)?,
            av: z(CHUNK * d.heads * d.head_v)?,
            part_v: z(attn::partials_v_len(rows, segs))?,
            part_ms: z(attn::partials_ms_len(rows, segs))?,
            qi: z(CHUNK * kpool::HEADS * kpool::DIM)?,
            wi: z(CHUNK * kpool::HEADS)?,
            scores: z(CHUNK * pools_for(ctx))?,
            list: zu(CHUNK * list_width(d.kept))?,
            h: z(CHUNK * ff)?,
            normed: z(cap * n)?,
            sh_y: z(cap * n)?,
            acc: z(cap * n)?,
            pre: z(cap * n)?,
            probs: z(cap * N_EXPERT)?,
            ids: zu(cap * N_USED)?,
            sel: zu(cap * N_USED)?,
            act_x: Q8Act::with_k(stream, CHUNK, n)?,
            card_h: z(CHUNK * N_USED * expert_ff)?,
            act_h: (1..=CHUNK)
                .map(|c| Q8Act::with_slots(stream, c * N_USED, expert_ff))
                .collect::<Result<_, _>>()?,
            card_down: z(if tiered { cap } else { CHUNK } * N_USED * n)?,
            front: GemmFront::open(
                gpu,
                FrontShape {
                    cols: cap,
                    embd: n,
                    low: head,
                    gated: v,
                },
            )?,
        })
    }

    fn bytes(&self) -> usize {
        let f = [
            &self.x,
            &self.xn,
            &self.out,
            &self.fold,
            &self.mixes,
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
            &self.qa,
            &self.kv,
            &self.qr,
            &self.q,
            &self.qabs,
            &self.att,
            &self.av,
            &self.part_v,
            &self.part_ms,
            &self.qi,
            &self.wi,
            &self.scores,
            &self.h,
            &self.normed,
            &self.sh_y,
            &self.acc,
            &self.pre,
            &self.probs,
            &self.card_h,
            &self.card_down,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>()
            + [&self.ids, &self.sel, &self.list]
                .iter()
                .map(|b| b.num_bytes())
                .sum::<usize>()
            + self.hc_scratch.device_bytes()
            + self.act_x.device_bytes()
            + self.act_h.iter().map(Q8Act::device_bytes).sum::<usize>()
            + self.front.bytes()
    }
}

/// The joined projection's rows after the query's low rank: the latent, the
/// index key and the pool gate.
fn kv_width(d: &Dims) -> usize {
    LATENT + 2 * d.index_d
}

/// The chunks of a batch of `t` tokens: runs of [`CHUNK`] from its first,
/// the last one shorter; `(first token, tokens)` each.
fn chunks(t: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..t).step_by(CHUNK).map(move |c0| (c0, CHUNK.min(t - c0)))
}

/// The batches of a call from `from` to `to` whose checkpoint marks are
/// `marks` ([`bloomery_gpu::checkpoint::Checkpoints::marks`]): each run between two marks cut into
/// `⌈len / T_MAX⌉` batches of near-equal size, the first ones a position
/// longer. Each layer reads every host expert its batch's tokens route to
/// once, so a short last batch would pay that read for few tokens.
#[must_use]
pub fn call_batches(from: u32, to: u32, marks: &[u32]) -> Vec<Range<u32>> {
    let mut out = Vec::new();
    let mut at = from;
    for &mark in marks.iter().filter(|&&k| k > from && k <= to) {
        let len = (mark - at) as usize;
        let k = len.div_ceil(T_MAX);
        let mut p = at;
        for j in 0..k {
            let n = (len / k + usize::from(j < len % k)) as u32;
            out.push(p..p + n);
            p += n;
        }
        at = mark;
    }
    out
}

/// The batches a call of `n` ids from the model's position runs
/// ([`call_batches`] over the body's marks): what a `time prompt` record
/// counts as its passes.
pub fn batches_of(m: &Glm5nextModel, n: usize) -> Result<Vec<Range<u32>>, GpuError> {
    let from = m.pos();
    let to = end_of(from, n)?;
    let marks = m.body(WHAT)?.ckpt.marks(from, to);
    Ok(call_batches(from, to, &marks))
}

/// The call's end, `from + n`, refused by name for no id or past `u32`.
fn end_of(from: u32, n: usize) -> Result<u32, GpuError> {
    u32::try_from(n)
        .ok()
        .and_then(|n| from.checked_add(n))
        .filter(|&to| to > from)
        .ok_or_else(|| shape(format!("a prompt of {n} ids from {from}")))
}

/// The call of `ids` from the model's position, refused by name before
/// anything runs: on a poisoned model, for no id, and past the positions the
/// stores hold — the model then stands where it stood, every store as it
/// was. Returns the call's end.
fn check_call(m: &Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let from = m.pos();
    let to = end_of(from, ids.len())?;
    let ctx = m.body(WHAT)?.ctx;
    if to as usize > ctx {
        return Err(shape(format!(
            "a prompt of {} ids from position {from} ends at {to}, past the {ctx} positions the \
             stores hold (the load's ctx)",
            ids.len()
        )));
    }
    Ok(to)
}

/// Feed `ids` from where `m` stands by the body's mode ([`set_prefill`]) and
/// return the argmax after the last: [`prefill`] or the steps ([`prompt`]),
/// each call refused by name before anything runs when it would pass the
/// stores' positions, so neither feed stops part of the way there. The batch
/// call is one residency pass ([`call`]); the steps feed is refused by name
/// while a residency machine runs ([`refuse_steps_under_residency`]).
pub fn feed(m: &mut Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    check_call(m, ids)?;
    match m.body(WHAT)?.prompt.mode {
        PrefillMode::Batch => call(m, |m| prefill(m, ids)),
        PrefillMode::Steps => {
            super::refuse_steps_under_residency(m)?;
            prompt(m, ids)
        }
    }
}

/// A prompt call `run` of `m` as one residency pass: its boundary before it
/// and none inside ([`GpuModel::pass_boundary`]), so the slot map is one map
/// for the whole call, and 0 rows kept after it, whatever it returned — the
/// residency rule counts decode rows only, and the batch service notes no
/// id. Nothing more on a load with no residency machine.
fn call<T>(
    m: &mut Glm5nextModel,
    run: impl FnOnce(&mut Glm5nextModel) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    m.pass_boundary()?;
    let r = run(m);
    let kept = m.keep_rows(0, bloomery_gpu::host::PassKind::Prompt);
    // The call's own error first: a keep refused after a failed call is its echo.
    let v = r?;
    kept?;
    Ok(v)
}

/// Feed `ids` from where `m` stands by `mode` and return the argmax after
/// the last, `sink` called after each unit the call runs ([`GlmPromptSink`]):
/// each batch of [`prefill`] ([`GlmArena::Prefill`]), or each step
/// ([`GlmArena::Step`]), one a position. Refused by name before anything runs
/// as [`feed`] refuses. `None` is [`feed`]'s call by `mode`. A sink's error
/// ends the call with it, its positions taken back as a failed call's are.
pub fn prompt_with(
    m: &mut Glm5nextModel,
    ids: &[u32],
    mode: PrefillMode,
    sink: Option<&mut GlmPromptSink<'_>>,
) -> Result<u32, GpuError> {
    check_call(m, ids)?;
    match (mode, sink) {
        (PrefillMode::Batch, sink) => call(m, |m| prefill_units(m, ids, sink)),
        (PrefillMode::Steps, None) => {
            super::refuse_steps_under_residency(m)?;
            prompt(m, ids)
        }
        (PrefillMode::Steps, Some(sink)) => {
            super::refuse_steps_under_residency(m)?;
            steps_with(m, ids, sink)
        }
    }
}

/// [`prompt`]'s steps one position each, `sink` called after each with the
/// step's arena, and its checkpoints at the call's marks.
fn steps_with(
    m: &mut Glm5nextModel,
    ids: &[u32],
    sink: &mut GlmPromptSink<'_>,
) -> Result<u32, GpuError> {
    let from = m.pos();
    let to = end_of(from, ids.len())?;
    let marks = m.body(WHAT)?.ckpt.marks(from, to);
    let mut argmax = None;
    let mut at = from;
    for mark in marks {
        while at < mark {
            let id = ids[(at - from) as usize];
            let r = m
                .step(&[id])
                .and_then(|t| sink(m, GlmArena::Step, at, 1).map(|()| t));
            match r {
                Ok(t) => argmax = Some(t),
                Err(e) => return Err(take_back(m, from, e)),
            }
            at += 1;
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

/// The body's feed mode.
pub fn prefill_mode(m: &Glm5nextModel) -> Result<PrefillMode, GpuError> {
    Ok(m.body(WHAT)?.prompt.mode)
}

/// Set `m`'s feed to `mode`; the batch feed's buffers, the host tier's batch
/// sets and the host union's slabs are made here, once, so that a timed
/// prompt allocates nothing. Returns whether it made any.
pub fn set_prefill(m: &mut Glm5nextModel, mode: PrefillMode) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.prompt.mode = mode;
    if mode == PrefillMode::Steps || body.prompt.batch.is_some() {
        return Ok(false);
    }
    body.make_batch(gpu)?;
    Ok(true)
}

/// The batches a prompt group of `m` runs layer by layer: `g` consecutive
/// batches of a call walked as one group ([`place::groups`]: a lone last
/// batch joins the group before it), each layer-batch's front enqueued
/// ahead of the previous one's host serve; 1 runs each batch alone, and
/// every `g` writes the same bits. Refused by name inside a call, for a `g`
/// outside 1 to the lever's most, and for a `g` of 2 or more on a load whose
/// dense layers do not all come before its routed ones. When the batch's
/// buffers are made and hold fewer units than `g` needs
/// ([`place::group_sets`]), the units it lacks and the timing's marks are
/// made here, between calls. Returns whether it made any.
pub fn set_prefill_group(m: &mut Glm5nextModel, g: usize) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    if body.prompt.in_call {
        return Err(shape(format!(
            "a prompt group of {g} batches set inside a prompt call"
        )));
    }
    if !(1..=GROUP_MAX).contains(&g) {
        return Err(GpuError::Shape {
            what: "BLOOMERY_PREFILL_GROUP",
            detail: format!("{g} batches, where a group holds 1 to {GROUP_MAX}"),
        });
    }
    refuse_dense_after_routed(&body.cfg, g)?;
    body.prompt.group = g;
    let made = body.grow_units(gpu)?;
    Ok(made)
}

/// The batches a prompt group of `m` runs ([`set_prefill_group`]).
pub fn prefill_group(m: &Glm5nextModel) -> Result<usize, GpuError> {
    Ok(m.body(WHAT)?.prompt.group)
}

/// Arm (or disarm) the batch walks' timing ([`PromptTiming`]): one event a
/// part boundary a (layer, unit), the serves' host times and the call's host
/// walls, which [`take_prompt_stats`] reads. Load-time allocation; unarmed,
/// a walk records no event, keeps no time and waits for nothing beyond its
/// own serves.
pub fn set_prompt_stats(m: &mut Glm5nextModel, on: bool) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.prompt.timing = if on {
        Some(PromptTiming::new(
            gpu.context(),
            body.cfg.len(),
            place::group_sets(body.prompt.group),
        )?)
    } else {
        None
    };
    Ok(())
}

/// The prompt's batch-walk stats, taken — the next prompt starts clean.
/// `Ok(None)` with the timing unarmed or no batch walked (a prompt by steps
/// keeps none).
pub fn take_prompt_stats(m: &mut Glm5nextModel) -> Result<Option<PromptStats>, GpuError> {
    Ok(m.body_parts(WHAT)?
        .2
        .prompt
        .timing
        .as_mut()
        .and_then(PromptTiming::take))
}

/// A made prompt batch's device bytes ([`prompt_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptBytes {
    /// The buffers every unit of a group shares.
    pub shared: usize,
    /// One unit's own buffers.
    pub unit: usize,
    /// The units made: as many as the group holds batches.
    pub units: usize,
    /// The host sums' buffer.
    pub hsum: usize,
    /// The card's free device bytes after them: what a group's units past
    /// the first came out of, the plan reserving none for them.
    pub free: usize,
}

/// `m`'s prompt batch's device bytes; `None` before a batch feed made them.
pub fn prompt_bytes(m: &Glm5nextModel) -> Result<Option<PromptBytes>, GpuError> {
    let Some(b) = m.body(WHAT)?.prompt.batch.as_deref() else {
        return Ok(None);
    };
    let (free, _) = m.gpu().mem_info()?;
    Ok(Some(PromptBytes {
        shared: b.bufs.bytes(),
        unit: b.units.first().map_or(0, UnitBufs::bytes),
        units: b.units.len(),
        hsum: b.hsum.num_bytes(),
        free,
    }))
}

/// The largest `BLOOMERY_PREFILL_GROUP`: the lever registry's.
const GROUP_MAX: usize = bloomery_levers::PREFILL_GROUP_MAX as usize;
const _: () = assert!(GROUP_MAX as u64 == bloomery_levers::PREFILL_GROUP_MAX);

/// Refused by name for a group of 2 or more on a load with a dense layer
/// past a routed one: a dense front writes the shared buffers the previous
/// routed item's back still reads (the shared-buffer rule holds only for a
/// dense prefix).
fn refuse_dense_after_routed(cfg: &[super::LayerCfg], g: usize) -> Result<(), GpuError> {
    if g < 2 {
        return Ok(());
    }
    let first_routed = cfg.iter().position(|c| c.kind.host_leg());
    let late_dense = first_routed.and_then(|r| (r..cfg.len()).find(|&l| !cfg[l].kind.host_leg()));
    match late_dense {
        Some(l) => Err(shape(format!(
            "a prompt group of {g} batches on a load whose dense layer {l} comes after a routed \
             one: a group's units share buffers only behind a dense prefix"
        ))),
        None => Ok(()),
    }
}

/// Feed `ids` from where `m` stands in batches (module doc) and return the
/// argmax after the last. Refused by name before anything runs: on a
/// poisoned model, past the stores' positions, with the taps armed (a tap
/// holds one token's streams, and a batch runs many). The batch's buffers
/// are made by the first call when [`set_prefill`] has not made them.
pub fn prefill(m: &mut Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    prefill_units(m, ids, None)
}

/// [`prefill`], `sink` called after each batch, in batch order, once the
/// group that holds it has run ([`prompt_with`]).
fn prefill_units(
    m: &mut Glm5nextModel,
    ids: &[u32],
    sink: Option<&mut GlmPromptSink<'_>>,
) -> Result<u32, GpuError> {
    let to = check_call(m, ids)?;
    let from = m.pos();
    {
        let (gpu, _, body) = m.body_parts(WHAT)?;
        if body.taps.is_some() {
            return Err(shape(
                "a prompt batch with the taps armed: a tap holds one step's streams".to_string(),
            ));
        }
        if let Some(taps) = body.prompt.routes.as_ref()
            && to as usize > taps.rows
        {
            return Err(shape(format!(
                "a prompt batch from {from} to {to} with route taps of {} rows",
                taps.rows
            )));
        }
        if let Some(plant) = body.prompt.plant_routes.as_ref()
            && to as usize > plant.rows
        {
            return Err(shape(format!(
                "a prompt batch from {from} to {to} with routes planted for {} positions",
                plant.rows
            )));
        }
        if body.prompt.batch.is_none() {
            body.make_batch(gpu)?;
        }
        body.prompt.in_call = true;
    }
    let r = call_groups(m, ids, from, to, sink);
    if let Ok(body) = m.body_parts(WHAT).map(|(_, _, b)| b) {
        body.prompt.in_call = false;
        if let Some(taps) = body.prompt.routes.as_mut() {
            let (from, to) = (from as usize, to as usize);
            taps.fed = match &r {
                Err(_) => 0..0,
                Ok(_) if taps.fed.start <= from && from <= taps.fed.end => taps.fed.start..to,
                Ok(_) => from..to,
            };
        }
    }
    r
}

/// [`prefill_units`]'s groups: the checkpoint at the call's start, then per
/// group its inner marks' takes opened, its walk ([`Body::enqueue_group`]),
/// the takes sealed once its fault word is read, the sink per unit, and the
/// checkpoint at its end when that is a mark. A group that fails abandons
/// its open takes and takes the call back.
fn call_groups(
    m: &mut Glm5nextModel,
    ids: &[u32],
    from: u32,
    to: u32,
    mut sink: Option<&mut GlmPromptSink<'_>>,
) -> Result<u32, GpuError> {
    let marks = m.body(WHAT)?.ckpt.marks(from, to);
    let batches = call_batches(from, to, &marks);
    let group = m.body(WHAT)?.prompt.group;
    if marks.first() == Some(&from) {
        let (gpu, _, body) = m.body_parts(WHAT)?;
        body.timed_checkpoint(gpu, from)?;
    }
    let mut argmax = None;
    for gr in place::groups(batches.len(), group) {
        let runs = &batches[gr];
        let (Some(first), Some(end)) = (runs.first(), runs.last()) else {
            continue;
        };
        let (start, end) = (first.start, end.end);
        let seg = &ids[(start - from) as usize..(end - from) as usize];
        let last = end == to;
        let mut takes = match open_takes(m, runs, &marks) {
            Ok(t) => t,
            Err(e) => return Err(take_back(m, from, e)),
        };
        let ran = m.run_rows(seg.len(), WHAT, |gpu, w, body, head, pos| {
            body.enqueue_group(gpu, w, head, seg, pos, runs, last, &mut takes)
        });
        let sealed = match ran {
            Ok(t) => seal_takes(m, &mut takes).map(|()| t),
            Err(e) => {
                abandon_takes(m, &mut takes);
                Err(e)
            }
        };
        let ran = sealed.and_then(|t| {
            let Some(s) = sink.as_deref_mut() else {
                return Ok(t);
            };
            for (u, run) in runs.iter().enumerate() {
                let (_, _, body) = m.body_parts(WHAT)?;
                body.prompt.sink = u;
                body.wrote.prefill = super::nextn::Held::at(run.start, run.end - run.start);
                s(
                    m,
                    GlmArena::Prefill,
                    run.start,
                    (run.end - run.start) as usize,
                )?;
            }
            Ok(t)
        });
        match ran {
            Ok(t) => argmax = t.or(argmax),
            Err(e) => return Err(take_back(m, from, e)),
        }
        if marks.contains(&end) {
            let (gpu, _, body) = m.body_parts(WHAT)?;
            body.timed_checkpoint(gpu, end)?;
        }
    }
    argmax.ok_or(GpuError::State {
        what: WHAT,
        missing: "the head of the batch that holds the call's last position",
    })
}

/// The open takes of a group of `runs`: one a unit whose end is one of the
/// call's `marks` inside the group (the group's own end is taken after it,
/// as the steps take it), in unit order; `None` for a unit that ends on no
/// mark, or on a mark where a checkpoint stands. Opened at the model's
/// position after any waiting cut ([`bloomery_gpu::checkpoint::Checkpoints::open_take`]);
/// a refusal abandons the ones opened before it.
fn open_takes(
    m: &mut Glm5nextModel,
    runs: &[Range<u32>],
    marks: &[u32],
) -> Result<Vec<Option<Pending>>, GpuError> {
    let pos = m.pos();
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let mut takes: Vec<Option<Pending>> = Vec::with_capacity(runs.len());
    let inner = runs.len().saturating_sub(1);
    let r = (|| {
        if runs[..inner].iter().any(|r| marks.contains(&r.end)) {
            body.stores_at(pos)?;
            body.apply_cut(gpu.stream())?;
        }
        for run in &runs[..inner] {
            takes.push(if marks.contains(&run.end) {
                let lane = body.s.lanes.committed();
                body.ckpt
                    .open_take(gpu.stream(), run.end, &mut copied(&mut body.stores, lane)?)?
            } else {
                None
            });
        }
        takes.push(None);
        Ok(())
    })();
    if let Err(e) = r {
        for p in takes.into_iter().flatten() {
            // The refusal is the error; an abandon's own failure is its echo.
            let _ = body.ckpt.abandon(gpu.stream(), p);
        }
        return Err(e);
    }
    Ok(takes)
}

/// Every open take of a group that ran sealed ([`Checkpoints::seal`]),
/// waited for under the timing's checkpoint wall; the first refusal is the
/// group's error, the rest abandoned.
fn seal_takes(m: &mut Glm5nextModel, takes: &mut [Option<Pending>]) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let t0 = body.prompt.timing.as_ref().map(|_| Instant::now());
    let mut r = Ok(());
    for p in takes.iter_mut().filter_map(Option::take) {
        r = match r {
            Ok(()) => body.ckpt.seal(gpu.stream(), p),
            Err(e) => {
                let _ = body.ckpt.abandon(gpu.stream(), p);
                Err(e)
            }
        };
    }
    if let (Some(t0), Some(t)) = (t0, body.prompt.timing.as_mut()) {
        t.add_ckpt(nanos(t0.elapsed()));
    }
    r
}

/// Every open take of a group that failed abandoned
/// ([`Checkpoints::abandon`]): the points from before the call stand, and no
/// checkpoint holds the failed group's state. The group's error is the
/// call's; an abandon's own failure is its echo.
fn abandon_takes(m: &mut Glm5nextModel, takes: &mut [Option<Pending>]) {
    let Ok((gpu, _, body)) = m.body_parts(WHAT) else {
        return;
    };
    for p in takes.iter_mut().filter_map(Option::take) {
        let _ = body.ckpt.abandon(gpu.stream(), p);
    }
}

/// A call from `from` that failed with `e`: its positions taken back through
/// the checkpoint of its start, so the model stands where the call found it.
/// A fault poisons the model and is returned as it came: nothing runs on it
/// until a reset.
fn take_back(m: &mut Glm5nextModel, from: u32, e: GpuError) -> GpuError {
    if matches!(e, GpuError::Fault { .. }) || m.poisoned().is_some() {
        return e;
    }
    let kept = match m.body(WHAT) {
        Ok(body) => body.keep_point(from, m.pos()),
        Err(b) => return b,
    };
    match m.rollback(kept) {
        Ok(()) if kept == from => e,
        Ok(()) => shape(format!(
            "a prompt call from position {from} failed ({e}); its positions are taken back and \
             the model stands at {kept}, the cut the checkpoints grant"
        )),
        Err(r) => shape(format!(
            "a prompt call from position {from} failed ({e}), and taking it back failed too ({r})"
        )),
    }
}

/// One store's bits as a digest ([`store_digests`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreDigest {
    pub layer: usize,
    /// `state`, `ring`, `latent`, `index` or `pooled`.
    pub what: &'static str,
    /// FNV-1a over the store's words, in order.
    pub fnv: u64,
}

/// Every layer's stores read back and digested, in layer order: a KDA
/// layer's committed lane of its state and its conv ring, a latent layer's
/// latent and index rows and its pool plane (all the rows the stores hold,
/// fed or not). Blocking.
pub fn store_digests(m: &mut Glm5nextModel) -> Result<Vec<StoreDigest>, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let stream = gpu.stream();
    let lane = body.s.lanes.committed();
    let mut out = Vec::new();
    for (layer, s) in body.stores.iter().enumerate() {
        match s {
            Store::Kda { state, ring, .. } => {
                for (what, b) in [("state", state.part(lane)?), ("ring", ring)] {
                    let words = b.to_host_vec(stream)?;
                    out.push(StoreDigest {
                        layer,
                        what,
                        fnv: fnv(words.iter().map(|v| u64::from(v.to_bits()))),
                    });
                }
            }
            Store::Latent {
                latent,
                index,
                pooled,
            } => {
                for (what, b) in [("latent", latent), ("index", index), ("pooled", pooled)] {
                    let words = b.buf().to_host_vec(stream)?;
                    out.push(StoreDigest {
                        layer,
                        what,
                        fnv: fnv(words.iter().map(|&v| u64::from(v))),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// One store's live values ([`store_rows`]).
#[derive(Clone, Debug)]
pub struct StoreRows {
    pub layer: usize,
    /// `state`, `ring`, `latent`, `index` or `pooled`.
    pub what: &'static str,
    /// The values, widened to f32 from the f16 of a latent layer's rows.
    pub values: Vec<f32>,
}

/// Every layer's live stores read back as values, in layer order: a KDA
/// layer's committed lane of its state and its conv ring whole, a latent
/// layer's latent and index rows of the first `live` positions and the pools
/// they complete ([`bloomery_gpu::latent::POOL`] positions a pool). Refused
/// by name for `live` past the stores' rows. Blocking.
pub fn store_rows(m: &mut Glm5nextModel, live: usize) -> Result<Vec<StoreRows>, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let stream = gpu.stream();
    let lane = body.s.lanes.committed();
    let half = |w: Vec<u16>| -> Vec<f32> { w.into_iter().map(gguf::quant::half_to_f32).collect() };
    let mut out = Vec::new();
    for (layer, s) in body.stores.iter().enumerate() {
        match s {
            Store::Kda { state, ring, .. } => {
                for (what, b) in [("state", state.part(lane)?), ("ring", ring)] {
                    out.push(StoreRows {
                        layer,
                        what,
                        values: b.to_host_vec(stream)?,
                    });
                }
            }
            Store::Latent {
                latent,
                index,
                pooled,
            } => {
                let pools = live / bloomery_gpu::latent::POOL;
                for (what, t, rows) in [
                    ("latent", latent, live),
                    ("index", index, live),
                    ("pooled", pooled, pools),
                ] {
                    if rows > t.rows() {
                        return Err(shape(format!(
                            "layer {layer}'s {what} rows 0..{rows} past the {} it holds",
                            t.rows()
                        )));
                    }
                    let words = span(WHAT, t.buf(), 0, rows * t.cols())?.to_host_vec(stream)?;
                    out.push(StoreRows {
                        layer,
                        what,
                        values: half(words),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// FNV-1a over `words`, each word's eight little-endian bytes.
fn fnv(words: impl Iterator<Item = u64>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325_u64, |h, w| {
        w.to_le_bytes().iter().fold(h, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

/// Which cached positions a chunk's tokens attend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keys {
    /// Every one at and before each token's position.
    Dense,
    /// The ones each token's selector list names.
    Selected,
}

/// The keys the chunk at `positions` attends over index rows `rows`: every
/// cached position while the chunk ends within `dense`, the positions the
/// indexer keeps whole (the step's list there is every position, in order);
/// the selector's lists past them. A chunk past the rows is refused by name.
fn prompt_keys(rows: usize, positions: Range<usize>, dense: usize) -> Result<Keys, GpuError> {
    if positions.end > rows {
        return Err(shape(format!(
            "a prompt chunk at positions {positions:?} past the {rows} index rows"
        )));
    }
    Ok(if positions.end <= dense {
        Keys::Dense
    } else {
        Keys::Selected
    })
}

impl Body {
    /// The host tier's batch sets and the host union's slabs for batches of
    /// the batch feed's size, made once, without the batch's buffers: what a
    /// NextN walk's host leg serves through.
    pub(super) fn prepare_port(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let cap = T_MAX.min(self.ctx);
        self.hybrid.prepare_batch(gpu.context(), cap)?;
        self.hybrid.host_mut().prepare_union(cap)
    }

    /// The batch feed's buffers — the shared ones and a set for each unit a
    /// group of the body's size holds ([`place::group_sets`]) — the host
    /// tier's batch sets for as many tokens and the host union's slabs, made
    /// once. Refused by name for a group of 2 or more on a load with a dense
    /// layer past a routed one.
    fn make_batch(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.prompt.batch.is_some() {
            return Ok(());
        }
        refuse_dense_after_routed(&self.cfg, self.prompt.group)?;
        let ff = self.cfg.iter().map(|c| c.ff).max().unwrap_or(0);
        let expert_ff = self.card.ff();
        let cap = T_MAX.min(self.ctx);
        let tiered = self.card.tier().is_some();
        let bufs = Bufs::new(gpu, &self.dims, [ff, expert_ff], self.ctx, cap, tiered)?;
        self.refuse_unreserved(gpu, place::group_sets(self.prompt.group) - 1)?;
        let units = (0..place::group_sets(self.prompt.group))
            .map(|_| UnitBufs::new(gpu, self.dims.embd, cap, tiered))
            .collect::<Result<Vec<_>, _>>()?;
        let hsum = DeviceBuffer::zeroed(gpu.stream(), cap * self.dims.embd)?;
        let batch = Batch { bufs, units, hsum };
        self.hybrid.prepare_batch(gpu.context(), cap)?;
        self.hybrid.host_mut().prepare_union(cap)?;
        gpu.stream().synchronize()?;
        self.prompt.batch = Some(Box::new(batch));
        Ok(())
    }

    /// Refused by name when `units` more units of a group ([`unit_bytes`]
    /// each) pass the card's free device bytes: the plan reserves nothing
    /// for a group's units past the first, so they come out of what the load
    /// left free, its margin included, and a group that does not fit is
    /// refused here, not by the driver.
    fn refuse_unreserved(&self, gpu: &Gpu, units: usize) -> Result<(), GpuError> {
        if units == 0 {
            return Ok(());
        }
        let cap = T_MAX.min(self.ctx);
        let extra = units * unit_bytes(self.dims.embd, cap, self.card.tier().is_some());
        let (free, _) = gpu.mem_info()?;
        if extra <= free {
            return Ok(());
        }
        Err(shape(format!(
            "a prompt group of {} batches: {units} more units take {extra} B, and the card has \
             {free} B free; the plan reserves none for a group's units past the first",
            self.prompt.group
        )))
    }

    /// The units the body's group needs ([`place::group_sets`]) that the
    /// batch's buffers lack, made, and the timing's marks remade for as
    /// many: between calls only. Nothing before the buffers are made, or
    /// when they hold enough. Returns whether it made any.
    fn grow_units(&mut self, gpu: &Gpu) -> Result<bool, GpuError> {
        let need = place::group_sets(self.prompt.group);
        let more = self
            .prompt
            .batch
            .as_deref()
            .map_or(0, |b| need.saturating_sub(b.units.len()));
        self.refuse_unreserved(gpu, more)?;
        let tiered = self.card.tier().is_some();
        if let Some(b) = self.prompt.batch.as_deref_mut()
            && more > 0
        {
            for _ in 0..more {
                b.units
                    .push(UnitBufs::new(gpu, self.dims.embd, b.bufs.cap, tiered)?);
            }
            gpu.stream().synchronize()?;
        }
        if self
            .prompt
            .timing
            .as_ref()
            .is_some_and(|t| t.units() < need)
        {
            self.prompt.timing = Some(PromptTiming::new(gpu.context(), self.cfg.len(), need)?);
        }
        Ok(more > 0)
    }

    /// [`Body::checkpoint`] at `pos`, its wall the timing's checkpoint wait
    /// when armed.
    fn timed_checkpoint(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let t0 = self.prompt.timing.as_ref().map(|_| Instant::now());
        let r = self.checkpoint(gpu, pos).map(|_| ());
        if let (Some(t0), Some(t)) = (t0, self.prompt.timing.as_mut()) {
            t.add_ckpt(nanos(t0.elapsed()));
        }
        r
    }

    /// Enqueue one group of `ids` from position `pos`, its batches `runs`
    /// (consecutive, from `pos`): each unit's rows and positions on the card,
    /// one walk `(units, T, Batch)` over every layer, and the head after the
    /// last unit when `last` — the pass [`bloomery_gpu::GpuModel::run_rows`]
    /// runs, returning whether the head was enqueued. Inside the walk each
    /// KDA layer's stores are copied into the open take of the mark its unit
    /// ends on (`takes`, one a unit) right after that unit's front. An inner
    /// group waits for its launches and reads the fault word once, so a
    /// fault ends the call before the mark after it; the last group's word
    /// rides the head's readback.
    #[allow(
        clippy::too_many_arguments,
        reason = "one pass's inputs: the model's parts run_rows lends, the group's ids, \
                  batches and end flag, and its open takes"
    )]
    fn enqueue_group(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        ids: &[u32],
        pos: u32,
        runs: &[Range<u32>],
        last: bool,
        takes: &mut [Option<Pending>],
    ) -> Result<bool, GpuError> {
        self.stores_at(pos)?;
        let n = self.dims.embd;
        let layers = self.cfg.len();
        let lane = self.s.lanes.committed();
        let store_ix = kda_store_index(&self.stores);
        let plant = match self.plant {
            Some(super::Plant::Group(u)) if u < runs.len() => {
                self.plant = None;
                Some(u)
            }
            _ => None,
        };
        let Body {
            hybrid,
            cfg,
            names,
            dims,
            k,
            s,
            stores,
            slots,
            card,
            embd,
            held,
            wrote,
            dense,
            prompt,
            ckpt,
            ..
        } = self;
        let PromptState {
            batch,
            timing,
            sink,
            routes,
            plant_routes,
            ..
        } = prompt;
        let Batch { bufs, units, hsum } = batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers (set_prefill)",
        })?;
        let g = runs.len();
        let ts: Vec<usize> = runs.iter().map(|r| (r.end - r.start) as usize).collect();
        if g == 0 || g > units.len() || takes.len() != g {
            return Err(shape(format!(
                "a group of {g} batches ({} takes) on buffers of {} units",
                takes.len(),
                units.len()
            )));
        }
        let cap = bufs.cap;
        if ts.iter().any(|&t| t == 0 || t > cap)
            || runs.first().map(|r| r.start) != Some(pos)
            || runs.windows(2).any(|w| w[0].end != w[1].start)
            || ts.iter().sum::<usize>() != ids.len()
        {
            return Err(shape(format!(
                "a group of batches {runs:?} from {pos} over {} ids: 1 to {cap} positions a \
                 batch, consecutive from the group's position",
                ids.len()
            )));
        }
        let stream = gpu.stream();
        let mut at = 0;
        for (ub, &t) in units.iter_mut().zip(&ts) {
            let p0 = pos + at as u32;
            let rows = &mut ub.rows[..t * HC_STREAMS * n];
            for (&token, r) in ids[at..at + t]
                .iter()
                .zip(rows.chunks_exact_mut(HC_STREAMS * n))
            {
                fill_row(embd, token, r)?;
            }
            for i in 0..t {
                let p = p0 + i as u32;
                ub.pos_host[i] = p;
                ub.cnt_host[i] = p + 1;
                ub.vis_host[2 * i] = 0;
                ub.vis_host[2 * i + 1] = p + 1;
            }
            span_mut(WHAT, &mut ub.streams[0], 0, t * HC_STREAMS * n)?
                .copy_from_host(stream, &ub.rows[..t * HC_STREAMS * n])?;
            span_mut(WHAT, &mut ub.pos, 0, t)?.copy_from_host(stream, &ub.pos_host[..t])?;
            span_mut(WHAT, &mut ub.cnt, 0, t)?.copy_from_host(stream, &ub.cnt_host[..t])?;
            span_mut(WHAT, &mut ub.vis, 0, 2 * t)?.copy_from_host(stream, &ub.vis_host[..2 * t])?;
            ub.cur = 0;
            at += t;
        }
        let end = runs.last().map_or(pos, |r| r.end);
        *held = end;
        let lastrun = runs.last().map_or(pos..pos, Clone::clone);
        wrote.prefill = super::nextn::Held::at(lastrun.start, lastrun.end - lastrun.start);
        *sink = g - 1;
        let mut prog = PromptProgram {
            gpu,
            w,
            p: Parts::of(k, dims, cfg, names, s, stores, slots, card, None),
            b: bufs,
            u: &mut units[..g],
            t: &ts,
            dense: *dense,
            takes: Takes {
                ckpt,
                open: takes,
                lane,
                store: &store_ix,
            },
            plant,
            routes: routes.as_mut(),
            planted: plant_routes.as_ref(),
        };
        let o = Overlap {
            units: g,
            cols: ts.iter().copied().max().unwrap_or(1),
            port: PortKind::Batch,
        };
        let mut leg = BatchLeg::new(stream, hybrid, hsum, cap);
        leg.set_unit_cols(&ts);
        leg.set_timer(timing.as_mut());
        leg.begin_walk(g)?;
        sched::walk(o, layers, &mut leg, &mut prog)?;
        leg.end_walk()?;
        if last {
            prog.head(head)?;
        }
        let t0 = timing.as_ref().map(|_| Instant::now());
        let read = read_fault(gpu, hybrid, last);
        if let (Some(t0), Some(t)) = (t0, timing.as_mut()) {
            t.add_fault(nanos(t0.elapsed()));
        }
        read
    }
}

/// The fault word after a group's walk: a fault the group raised on the
/// expert tier is the call's error, as the stage card's is (the first layer
/// wins); `None` without a tier. The last group's stage word rides the
/// head's readback ([`bloomery_gpu::GpuModel::run_rows`]), so it returns
/// that the head was enqueued; an inner group reads both words here.
fn read_fault(
    gpu: &Gpu,
    hybrid: &mut bloomery_gpu::hybrid::Hybrid<GlmHost>,
    last: bool,
) -> Result<bool, GpuError> {
    let tier = hybrid.tier_fault()?;
    if last {
        return match tier {
            Some(t) => Err(GpuError::fault(
                WHAT,
                read_cards(&[gpu.fault()?, Some(t)]).unwrap_or(t),
            )),
            None => Ok(true),
        };
    }
    match read_cards(&[gpu.fault()?, tier]) {
        Some(fault) => Err(GpuError::fault(WHAT, fault)),
        None => Ok(false),
    }
}

/// Each layer's first store in the checkpoints' list (`super::copied`: two
/// a KDA layer, in layer order); `None` for a latent layer.
fn kda_store_index(stores: &[Store]) -> Vec<Option<usize>> {
    let mut next = 0;
    stores
        .iter()
        .map(|s| match s {
            Store::Kda { .. } => {
                next += 2;
                Some(next - 2)
            }
            Store::Latent { .. } => None,
        })
        .collect()
}

/// Token `token`'s embedding row into `out`, four copies, one a stream; a
/// token past the vocabulary is refused by name.
fn fill_row(e: &Embedding, token: u32, out: &mut [f32]) -> Result<(), GpuError> {
    let t = token as usize;
    if t >= e.n_vocab {
        return Err(shape(format!(
            "token {token} is past the {} embedding rows",
            e.n_vocab
        )));
    }
    let data = e
        .file
        .shard(e.shard)
        .ok_or(GpuError::State {
            what: WHAT,
            missing: "the embedding's shard",
        })?
        .data(&e.info)?;
    let src = &data[t * e.row_bytes..][..e.row_bytes];
    let (first, rest) = out.split_at_mut(e.row.len());
    dequant_row(GgmlType::Q8_0, src, first).map_err(model::ModelError::from)?;
    for s in rest.chunks_exact_mut(first.len()) {
        s.copy_from_slice(first);
    }
    Ok(())
}

/// `y = W · x` for the q8_0 weight `name` over `c` token columns of `x`,
/// token-major into `y` (`q8_0_gemv_mcol`: each column the one-column
/// gemv's bits).
fn mcol(
    gpu: &Gpu,
    w: &Weights,
    name: &str,
    x: &DeviceBuffer<f32>,
    c: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (qs, d) = q8(w, name)?;
    gpu.q8f32().enqueue_q8_0_gemv_mcol(
        gpu.stream(),
        Q8_0GemvMcolArgs {
            qs,
            d,
            x,
            m: c,
            out: GemvOut::TokenMajor,
            y,
        },
    )
}

/// Rows `rows` of a resident q8_0 weight as a weight of their own: both
/// planes' windows, given back when it drops.
struct RowWindow {
    qs: ManuallyDrop<DeviceTensor<u32>>,
    d: ManuallyDrop<DeviceTensor<u16>>,
}

impl RowWindow {
    /// Rows `rows` of the planes `qs` and `d`, refused by name unless they
    /// are a non-empty run inside both.
    fn of(
        qs: &DeviceTensor<u32>,
        d: &DeviceTensor<u16>,
        rows: Range<usize>,
    ) -> Result<RowWindow, GpuError> {
        if rows.is_empty() || rows.end > qs.rows() || qs.rows() != d.rows() {
            return Err(shape(format!(
                "rows {rows:?} of a q8_0 weight of {} (scales {})",
                qs.rows(),
                d.rows()
            )));
        }
        let (n, r0) = (rows.len(), rows.start);
        let at_q = (r0 * qs.cols() * size_of::<u32>()) as u64;
        let at_d = (r0 * d.cols() * size_of::<u16>()) as u64;
        // SAFETY: rows r0 .. r0 + n lie inside both planes (checked above),
        // each a whole row from its first element, so aligned for its type;
        // the planes are resident weights, in place and alive for the load,
        // and the windows are given back when this drops, within the walk
        // that made them.
        let (qs, d) = unsafe {
            (
                DeviceTensor::window(
                    qs.buf().cu_deviceptr() + at_q,
                    n,
                    qs.cols(),
                    qs.buf().context(),
                ),
                DeviceTensor::window(
                    d.buf().cu_deviceptr() + at_d,
                    n,
                    d.cols(),
                    d.buf().context(),
                ),
            )
        };
        Ok(RowWindow { qs, d })
    }
}

impl Drop for RowWindow {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and handed straight back
        // wrapped, so nothing drops the memory it does not own.
        let (qs, d) = unsafe {
            (
                ManuallyDrop::take(&mut self.qs),
                ManuallyDrop::take(&mut self.d),
            )
        };
        DeviceTensor::release(ManuallyDrop::new(qs));
        DeviceTensor::release(ManuallyDrop::new(d));
    }
}

/// One group's walk: the body's parts, the shared buffers, each unit's own
/// and its tokens, the positions the latent layers attend whole, the group's
/// open takes, and the unit a gate's plant fails the walk at.
struct PromptProgram<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    p: Parts<'a>,
    b: &'a mut Bufs,
    u: &'a mut [UnitBufs],
    t: &'a [usize],
    dense: usize,
    takes: Takes<'a>,
    /// [`super::Plant::Group`]'s unit: its first routed front fails.
    plant: Option<usize>,
    /// The armed route taps, which each router's rows are copied into.
    routes: Option<&'a mut RouteTaps>,
    /// The planted routes, written over each router's picks and weights.
    planted: Option<&'a RoutePlant>,
}

/// A group's open takes, one a unit ([`Body::enqueue_group`]), with what
/// their copies need: the checkpoints, the committed lane, each layer's first
/// store in the checkpoints' list.
struct Takes<'a> {
    ckpt: &'a mut Checkpoints,
    open: &'a mut [Option<Pending>],
    lane: u32,
    store: &'a [Option<usize>],
}

impl PromptProgram<'_> {
    /// Sub-layer `sub` of layer `l`'s input into `x` for unit `u`: its own
    /// mix of the unit's streams over its tokens (`hc_pre_q8_0` in its token
    /// groups) and their fold by it.
    fn hc_in(&mut self, l: usize, u: usize, sub: Sub) -> Result<(), GpuError> {
        let (gpu, w, t) = (self.gpu, self.w, self.t[u]);
        let d = *self.p.d;
        let n = self
            .p
            .names
            .get(l)
            .ok_or_else(|| shape(format!("layer {l} past the {} named", self.p.names.len())))?
            .hc(sub);
        let (qs, dd) = q8(w, &n.fn_)?;
        let params = HcQ8Params {
            qs,
            d: dd,
            scale: f32v(w, &n.scale)?,
            base: f32v(w, &n.base)?,
            eps: d.hc_eps,
            iters: d.hc_iters,
        };
        let b = &mut *self.b;
        let ub = &mut self.u[u];
        let streams = &ub.streams[ub.cur];
        let hc = &self.p.k.hc;
        hc.enqueue_pre_q8_0(
            gpu.stream(),
            &HcQ8PreArgs {
                params: &params,
                x: streams,
                tokens: t,
                rms_eps: d.rms_eps,
            },
            t.min(CHUNK),
            &mut b.hc_scratch,
            &mut b.mixes,
            &mut ub.hc,
        )?;
        hc.enqueue_fold(gpu.stream(), streams, &ub.hc, d.embd, t, &mut b.x)
    }

    /// The sub-layer's output `out` into unit `u`'s other stream buffer by
    /// its mix (`hc_post`), residual from the current one.
    fn hc_out(&mut self, u: usize) -> Result<(), GpuError> {
        let t = self.t[u];
        let b = &mut *self.b;
        let ub = &mut self.u[u];
        let [s0, s1] = &mut ub.streams;
        let (res, next) = if ub.cur == 0 { (&*s0, s1) } else { (&*s1, s0) };
        self.p.k.hc.enqueue_post(
            self.gpu.stream(),
            &HcPostArgs {
                x: &b.out,
                res,
                hc: &ub.hc,
                n_embd: self.p.d.embd,
                tokens: t,
            },
            next,
            &mut b.fold,
        )?;
        ub.cur ^= 1;
        Ok(())
    }

    /// Layer `l`'s KDA mixer over unit `u`'s batch (`kda::kda`'s launches,
    /// the projections by chunk and the conv, the delta step and the gated
    /// norm over every token in position order).
    fn kda(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill kda";
        let (gpu, w, t) = (self.gpu, self.w, self.t[u]);
        let stream = gpu.stream();
        let d = *self.p.d;
        let fault = gpu.layer_sink(l)?;
        let Some(MixerNames::Kda(nm)) = self.p.names.get(l).map(|n| &n.mixer) else {
            return Err(other_kind(W, l));
        };
        let Some(Store::Kda { state, stamp, ring }) = self.p.stores.get_mut(l) else {
            return Err(GpuError::State {
                what: W,
                missing: "the layer's KDA store",
            });
        };
        let b = &mut *self.b;
        let ub = &self.u[u];
        let head = bloomery_gpu::linear::HEAD;
        let (n, ch, nv) = (d.embd, d.kda.channels(), d.kda.n_v);
        let v = nv * head;
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.x,
            f32v(w, &nm.norm)?,
            d.rms_eps,
            n,
            t,
            &mut b.xn,
        )?;
        let gemm = t >= GEMM_FROM;
        if gemm {
            b.front.kda_in(
                gpu,
                w,
                l,
                t,
                &b.xn,
                KdaInNames {
                    qkv: &nm.qkv,
                    g_a: &nm.g_a,
                    beta: &nm.beta,
                    g_b: &nm.g_b,
                },
                KdaInRows {
                    qkv: &mut b.qkv,
                    ga: &mut b.ga,
                    beta_raw: &mut b.beta_raw,
                    z: &mut b.z,
                },
            )?;
        }
        for (c0, c) in chunks(t) {
            let xs = span(W, &b.xn, c0 * n, c * n)?;
            if !gemm {
                mcol(
                    gpu,
                    w,
                    &nm.qkv,
                    &xs,
                    c,
                    &mut *span_mut(W, &mut b.qkv, c0 * ch, c * ch)?,
                )?;
            }
            mcol(
                gpu,
                w,
                &nm.f_a,
                &xs,
                c,
                &mut *span_mut(W, &mut b.fa, c0 * head, c * head)?,
            )?;
            if !gemm {
                mcol(
                    gpu,
                    w,
                    &nm.g_a,
                    &xs,
                    c,
                    &mut *span_mut(W, &mut b.ga, c0 * head, c * head)?,
                )?;
                mcol(
                    gpu,
                    w,
                    &nm.beta,
                    &xs,
                    c,
                    &mut *span_mut(W, &mut b.beta_raw, c0 * nv, c * nv)?,
                )?;
            }
            let fa = span(W, &b.fa, c0 * head, c * head)?;
            mcol(
                gpu,
                w,
                &nm.f_b,
                &fa,
                c,
                &mut *span_mut(W, &mut b.f, c0 * v, c * v)?,
            )?;
            if !gemm {
                let ga = span(W, &b.ga, c0 * head, c * head)?;
                mcol(
                    gpu,
                    w,
                    &nm.g_b,
                    &ga,
                    c,
                    &mut *span_mut(W, &mut b.z, c0 * v, c * v)?,
                )?;
            }
        }
        let lin = &self.p.k.linear;
        lin.conv.enqueue_kda_conv_prep(
            stream,
            KdaConvArgs {
                x: &b.qkv,
                b_raw: &b.beta_raw,
                f: &b.f,
                w: f32v(w, &nm.conv)?,
                dt_bias: f32v(w, &nm.dt_bias)?,
                ssm_a: f32v(w, &nm.a)?,
                pos: &ub.pos,
                shape: d.kda,
                lb: d.lb,
                eps: d.rms_eps,
                m: t,
                fault,
                y: &mut b.conv,
                beta: &mut b.beta,
                decay: &mut b.decay,
                ring,
            },
        )?;
        lin.delta.enqueue_kda_delta_lanes(
            stream,
            KdaLanesArgs {
                lanes: DeltaLanesArgs {
                    delta: DeltaArgs {
                        qkv: &b.conv,
                        beta: &b.beta,
                        decay: &b.decay,
                        lane: self.p.lane,
                        lane_at: 0,
                        lanes: state.lanes().count(),
                        shape: d.kda,
                        m: t,
                        fault,
                        o: &mut b.o,
                        state: state.whole_mut(),
                    },
                    each: false,
                    pos: &ub.pos,
                    stamp,
                },
                row: 0,
            },
        )?;
        lin.norm_gate.enqueue_norm_gate_sigmoid(
            stream,
            NormGateArgs {
                o: &b.o,
                z: &b.z,
                w: f32v(w, &nm.gate_norm)?,
                eps: d.rms_eps,
                n_v: nv,
                m: t,
                fault,
                y: &mut b.gated,
            },
        )?;
        if gemm {
            return b.front.kda_out(gpu, w, l, t, &b.gated, &nm.out, &mut b.out);
        }
        for (c0, c) in chunks(t) {
            let gs = span(W, &b.gated, c0 * v, c * v)?;
            mcol(
                gpu,
                w,
                &nm.out,
                &gs,
                c,
                &mut *span_mut(W, &mut b.out, c0 * n, c * n)?,
            )?;
        }
        Ok(())
    }

    /// Layer `l`'s latent mixer over unit `u`'s batch (`mla::mla`'s
    /// launches): the norm, the joined projection by its two row ranges and
    /// chunk, the query's norm, every token's latent and index rows appended
    /// at its position and the pools the tokens complete, then chunk by chunk
    /// the heads, the selector past the dense positions ([`prompt_keys`]),
    /// the attention and the output projection.
    fn mla(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill mla";
        let (gpu, w, t) = (self.gpu, self.w, self.t[u]);
        let stream = gpu.stream();
        let d = *self.p.d;
        let fault = gpu.layer_sink(l)?;
        let Some(MixerNames::Latent(nm)) = self.p.names.get(l).map(|n| &n.mixer) else {
            return Err(other_kind(W, l));
        };
        let Some(Store::Latent {
            latent,
            index,
            pooled,
        }) = self.p.stores.get_mut(l)
        else {
            return Err(GpuError::State {
                what: W,
                missing: "the layer's latent store",
            });
        };
        let b = &mut *self.b;
        let ub = &mut self.u[u];
        let s = &*self.p.s;
        let (n, ql, kvw) = (d.embd, d.q_lora, kv_width(&d));
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.x,
            f32v(w, &nm.norm)?,
            d.rms_eps,
            n,
            t,
            &mut b.xn,
        )?;
        if t >= GEMM_FROM {
            b.front.latent_in(
                gpu,
                w,
                l,
                t,
                &b.xn,
                (&nm.stack, ql, kvw),
                LatentInRows {
                    qa: &mut b.qa,
                    kv: &mut b.kv,
                },
            )?;
        } else {
            let (qs, dd) = q8(w, &nm.stack)?;
            let qa_rows = RowWindow::of(qs, dd, 0..ql)?;
            let kv_rows = RowWindow::of(qs, dd, ql..ql + kvw)?;
            for (c0, c) in chunks(t) {
                let xs = span(W, &b.xn, c0 * n, c * n)?;
                for (win, y, width) in [(&qa_rows, &mut b.qa, ql), (&kv_rows, &mut b.kv, kvw)] {
                    gpu.q8f32().enqueue_q8_0_gemv_mcol(
                        stream,
                        Q8_0GemvMcolArgs {
                            qs: &win.qs,
                            d: &win.d,
                            x: &xs,
                            m: c,
                            out: GemvOut::TokenMajor,
                            y: &mut *span_mut(W, y, c0 * width, c * width)?,
                        },
                    )?;
                }
            }
        }
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.qa,
            f32v(w, &nm.q_a_norm)?,
            d.rms_eps,
            ql,
            t,
            &mut b.qr,
        )?;
        let lat = &self.p.k.latent;
        lat.enqueue_latent_append(
            stream,
            LatentAppendArgs {
                rows: Rows {
                    x: &b.kv,
                    stride: kvw,
                    m: t,
                },
                off: 0,
                gain: f32v(w, &nm.kv_a_norm)?,
                pos: &ub.pos,
                eps: d.rms_eps,
                fault,
                cache: latent,
            },
        )?;
        lat.enqueue_index_key_append(
            stream,
            IndexKeyArgs {
                rows: Rows {
                    x: &b.kv,
                    stride: kvw,
                    m: t,
                },
                k_off: LATENT,
                g_off: LATENT + d.index_d,
                w: f32v(w, &nm.index_norm)?,
                b: f32v(w, &nm.index_norm_bias)?,
                pos: &ub.pos,
                eps: d.norm_eps,
                fault,
                cache: index,
            },
        )?;
        mla::pool(
            stream, w, self.p.k, &nm.sel, index, &ub.cnt, t, fault, pooled,
        )?;
        let first = ub.pos_host[0] as usize;
        let rows = index.rows();
        let kept = d.kept;
        let (qs_kb, d_kb) = q8(w, &nm.k_b)?;
        let (qs_vb, d_vb) = q8(w, &nm.v_b)?;
        // SAFETY: a view of no rows at the address of the layer's own latent
        // cache, a live allocation aligned for u16 that outlives the view; the
        // attention reads no window row through it, and the view is released
        // below before the cache can drop.
        let window = unsafe {
            DeviceTensor::<u16>::window(latent.buf().cu_deviceptr(), 0, LATENT, gpu.context())
        };
        let r = (|| -> Result<(), GpuError> {
            for (c0, c) in chunks(t) {
                let qr = span(W, &b.qr, c0 * ql, c * ql)?;
                mcol(gpu, w, &nm.q_b, &qr, c, &mut b.q)?;
                gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                    stream,
                    Q8_0GemvHeadsMcolArgs {
                        qs: qs_kb,
                        d: d_kb,
                        x: &b.q,
                        rows_per_head: LATENT,
                        x_head_stride: d.head_k,
                        y_head_stride: LATENT,
                        y_off: 0,
                        m: c,
                        x_col_stride: d.heads * d.head_k,
                        y_col_stride: d.heads * LATENT,
                        y: &mut b.qabs,
                    },
                )?;
                let keys = prompt_keys(rows, first + c0..first + c0 + c, self.dense)?;
                if keys == Keys::Selected {
                    let xn = span(W, &b.xn, c0 * n, c * n)?;
                    let cnt = span(W, &ub.cnt, c0, c)?;
                    let mut vis = span_mut(W, &mut ub.vis, 2 * c0, 2 * c)?;
                    mla::select(
                        gpu,
                        w,
                        self.p.k,
                        &nm.sel,
                        Select {
                            stream,
                            m: c,
                            xn: &xn,
                            qr: &qr,
                            cnt: &cnt,
                            index,
                            pooled,
                            qi: &mut b.qi,
                            wi: &mut b.wi,
                            scores: &mut b.scores,
                            list: &mut b.list,
                            vis: &mut vis,
                            kept,
                            fault,
                        },
                    )?;
                }
                let vis = span(W, &ub.vis, 2 * c0, 2 * c)?;
                self.p.k.attn.enqueue(
                    stream,
                    AttnArgs {
                        q: &b.qabs,
                        window: &window,
                        compressed: Some(&*latent),
                        selected: (keys == Keys::Selected).then_some(SelectedRows {
                            rows: &b.list,
                            stride: list_width(kept),
                        }),
                        vis: &vis,
                        sinks: &s.sinks,
                        scale: 1.0 / (d.head_k as f32).sqrt(),
                        tokens: c,
                        heads: d.heads,
                        part_v: &mut b.part_v,
                        part_ms: &mut b.part_ms,
                        y: &mut b.att,
                        fault,
                    },
                )?;
                gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                    stream,
                    Q8_0GemvHeadsMcolArgs {
                        qs: qs_vb,
                        d: d_vb,
                        x: &b.att,
                        rows_per_head: d.head_v,
                        x_head_stride: LATENT,
                        y_head_stride: d.head_v,
                        y_off: 0,
                        m: c,
                        x_col_stride: d.heads * LATENT,
                        y_col_stride: d.heads * d.head_v,
                        y: &mut b.av,
                    },
                )?;
                mcol(
                    gpu,
                    w,
                    &nm.out,
                    &b.av,
                    c,
                    &mut *span_mut(W, &mut b.out, c0 * n, c * n)?,
                )?;
            }
            Ok(())
        })();
        DeviceTensor::release(window);
        r
    }

    /// Layer `l`'s dense block over unit `u`'s batch: the norm, then by
    /// chunk the gate·up·SwiGLU and the down projection.
    fn dense(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill dense";
        let (gpu, w, t) = (self.gpu, self.w, self.t[u]);
        let stream = gpu.stream();
        let (d, c) = (*self.p.d, self.p.cfg[l]);
        let Some(FfnNames::Dense {
            norm,
            gate,
            up,
            down,
        }) = self.p.names.get(l).map(|n| &n.ffn)
        else {
            return Err(other_kind(W, l));
        };
        let b = &mut *self.b;
        let n = d.embd;
        gpu.elem()
            .enqueue_rms_norm(stream, &b.x, f32v(w, norm)?, d.rms_eps, n, t, &mut b.xn)?;
        let (g, u) = (weight(w, gate)?, weight(w, up)?);
        for (c0, cn) in chunks(t) {
            let xs = span(W, &b.xn, c0 * n, cn * n)?;
            self.p
                .k
                .experts
                .enqueue_shexp_gate_up_mcol(stream, g, u, &xs, cn, c.limit, &mut b.h)?;
            mcol(
                gpu,
                w,
                down,
                &b.h,
                cn,
                &mut *span_mut(W, &mut b.out, c0 * n, cn * n)?,
            )?;
        }
        Ok(())
    }

    /// Layer `l`'s routed block up to its host leg for `at`'s unit: the norm
    /// into `normed`, the router over every token (the scores, then each
    /// token's picks), the timer's [`Mark::FrontEnd`], and the download of
    /// the rows, weights and ids to the host.
    fn route(&mut self, port: &mut BatchLeg<'_, GlmHost>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (gpu, w, t) = (self.gpu, self.w, self.t[at.unit]);
        let stream = gpu.stream();
        let (d, c) = (*self.p.d, self.p.cfg[l]);
        let fault = gpu.layer_sink(l)?;
        let nm = ffn::moe_names(&self.p, l)?;
        let b = &mut *self.b;
        let ub = &mut self.u[at.unit];
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.x,
            f32v(self.w, nm.norm)?,
            d.rms_eps,
            d.embd,
            t,
            &mut b.normed,
        )?;
        let bias = if c.bias {
            f32v(w, nm.bias)?
        } else {
            &self.p.s.no_bias
        };
        self.p.k.router.enqueue_router_rows(
            stream,
            f32t(w, nm.router)?,
            &b.normed,
            bias,
            d.scale,
            t,
            &mut b.probs,
            &mut b.ids,
            &mut ub.weights,
            fault,
        )?;
        let p0 = ub.pos_host[0] as usize;
        if let Some(taps) = self.routes.as_deref_mut() {
            let Some(Some(TapPlanes {
                probs: Some(probs),
                ids,
                weights,
            })) = taps.layers.get_mut(l)
            else {
                return Err(GpuError::State {
                    what: WHAT,
                    missing: "a routed layer's route taps",
                });
            };
            span_mut(WHAT, probs, p0 * N_EXPERT, t * N_EXPERT)?
                .copy_from_device_async(&*span(WHAT, &b.probs, 0, t * N_EXPERT)?, stream)?;
            span_mut(WHAT, ids, p0 * N_USED, t * N_USED)?
                .copy_from_device_async(&*span(WHAT, &b.ids, 0, t * N_USED)?, stream)?;
            span_mut(WHAT, weights, p0 * N_USED, t * N_USED)?
                .copy_from_device_async(&*span(WHAT, &ub.weights, 0, t * N_USED)?, stream)?;
        }
        if let Some(plant) = self.planted {
            let Some(Some(TapPlanes { ids, weights, .. })) = plant.layers.get(l) else {
                return Err(GpuError::State {
                    what: WHAT,
                    missing: "a routed layer's planted routes",
                });
            };
            span_mut(WHAT, &mut b.ids, 0, t * N_USED)?
                .copy_from_device_async(&*span(WHAT, ids, p0 * N_USED, t * N_USED)?, stream)?;
            span_mut(WHAT, &mut ub.weights, 0, t * N_USED)?
                .copy_from_device_async(&*span(WHAT, weights, p0 * N_USED, t * N_USED)?, stream)?;
        }
        port.mark(at, Mark::FrontEnd as usize)?;
        let key = port.key(at);
        let Some(side) = self.p.card.tier().filter(|t| t.k(l) > 0) else {
            return port
                .hybrid()
                .enqueue_download(stream, [&b.normed, &ub.weights], &b.ids, key);
        };
        // A tiered layer's route carries each slot's tier place too, which the
        // tier's service reads.
        let tsel = ub.tsel.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier places of a batch made with a tier",
        })?;
        self.p.card.batch().enqueue_places(
            stream,
            &Places {
                ids: &b.ids,
                n: t * N_USED,
                map: side.places(),
                row_off: c.row_off,
                n_expert: N_EXPERT,
            },
            fault,
            &mut *tsel,
        )?;
        port.hybrid().enqueue_download_tiered(
            stream,
            [&b.normed, &ub.weights],
            &b.ids,
            &[&*tsel],
            key,
        )
    }

    /// `at`'s card experts, where the slot map puts any, and its shared
    /// expert by chunk, under its host leg; the card sum plus the shared
    /// expert's output after them.
    fn shadow_rows(&mut self, at: At) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill shadow";
        let l = at.layer;
        let (gpu, w, t) = (self.gpu, self.w, self.t[at.unit]);
        let stream = gpu.stream();
        let c = self.p.cfg[l];
        let n = self.p.d.embd;
        let card = self.p.card.has(l);
        let tiered = self.p.card.tier_k(l) > 0;
        let b = &mut *self.b;
        let ub = &self.u[at.unit];
        if card {
            ffn::card_rows(
                gpu,
                w,
                &mut self.p,
                l,
                CardRows {
                    normed: &b.normed,
                    ids: &b.ids,
                    weights: &ub.weights,
                    t,
                    sel: &mut b.sel,
                    act_x: &mut b.act_x,
                    h: &mut b.card_h,
                    act_h: &mut b.act_h,
                    down: &mut b.card_down,
                    acc: &mut b.acc,
                    tiered,
                },
            )?;
        }
        let nm = ffn::moe_names(&self.p, l)?;
        let (g, u) = (weight(w, nm.sh_gate)?, weight(w, nm.sh_up)?);
        for (c0, cn) in chunks(t) {
            let xs = span(W, &b.normed, c0 * n, cn * n)?;
            self.p
                .k
                .experts
                .enqueue_shexp_gate_up_mcol(stream, g, u, &xs, cn, c.limit, &mut b.h)?;
            mcol(
                gpu,
                w,
                nm.sh_down,
                &b.h,
                cn,
                &mut *span_mut(W, &mut b.sh_y, c0 * n, cn * n)?,
            )?;
        }
        if card && !tiered {
            gpu.elem()
                .enqueue_add(stream, &b.acc, &b.sh_y, t * n, &mut b.pre)?;
        }
        Ok(())
    }

    /// A tiered layer's card sum over the batch, once the walk's serve has
    /// served the block and the tier's rows have landed
    /// ([`BatchLeg::join_tiered`], which uploads the host sums after it):
    /// every token's slots in slot order from the stage card's downs or the
    /// tier's rows (`ds41_ffn_card_acc_8_tier`), then the shared expert's
    /// output added — the shadow's two launches of a one-card layer.
    fn join_tier(&mut self, port: &mut BatchLeg<'_, GlmHost>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (gpu, t) = (self.gpu, self.t[at.unit]);
        let n = self.p.d.embd;
        let n_card = self.p.card.n_card(l);
        let n_tier = self.p.card.tier_k(l);
        let batch = self.p.card.batch();
        let b = &mut *self.b;
        let ub = &self.u[at.unit];
        let tsel = ub.tsel.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier places of a batch made with a tier",
        })?;
        let (down, w, sel, acc) = (&b.card_down, &ub.weights, &b.sel, &mut b.acc);
        port.join_tiered(at, |stream, trows| {
            let a = CardAccTier {
                down,
                trows,
                w,
                sel,
                tsel,
                n,
                m: t,
                n_card,
                n_tier,
                n_used: N_USED,
            };
            batch.enqueue_card_acc_tier(stream, &a, acc)
        })?;
        gpu.elem()
            .enqueue_add(gpu.stream(), &b.acc, &b.sh_y, t * n, &mut b.pre)
    }

    /// After the last layer: the group's last unit's last token's streams'
    /// mean into the head's input, and the head.
    fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let n = self.p.d.embd;
        let (Some(ub), Some(&t)) = (self.u.last(), self.t.last()) else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a unit of the group for the head",
            });
        };
        let last = span(
            WHAT,
            &ub.streams[ub.cur],
            (t - 1) * HC_STREAMS * n,
            HC_STREAMS * n,
        )?;
        self.p
            .k
            .hc
            .enqueue_mean(self.gpu.stream(), &last, n, 0, head.input_mut())?;
        head.enqueue(self.gpu, self.w)
    }
}

impl PromptProgram<'_> {
    /// `at`'s KDA layer's two stores copied into the open take of the mark
    /// its unit ends on, where one is open: right after its front, while the
    /// stores hold that mark's state — the next unit's front moves them on.
    /// Nothing for a latent layer or a unit that ends on no open take.
    fn take_stores(&mut self, at: At) -> Result<(), GpuError> {
        let Some(i) = self.takes.store.get(at.layer).copied().flatten() else {
            return Ok(());
        };
        let Some(pending) = self.takes.open.get_mut(at.unit).and_then(Option::as_mut) else {
            return Ok(());
        };
        let Some(store) = self.p.stores.get_mut(at.layer) else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the layer's store for its take",
            });
        };
        let Some([state, ring]) = store.copied(self.takes.lane)? else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a KDA layer's stores for its take",
            });
        };
        let stream = self.gpu.stream();
        self.takes.ckpt.copy(stream, pending, i, state)?;
        self.takes.ckpt.copy(stream, pending, i + 1, ring)
    }
}

impl<'a> LayerProgram for PromptProgram<'a> {
    type Port = BatchLeg<'a, GlmHost>;

    /// A routed layer's; the dense lead has none.
    fn host_leg(&self, at: At) -> bool {
        self.p.cfg.get(at.layer).is_some_and(|c| c.kind.host_leg())
    }

    /// The mixer sub-layer, then the block's input and the dense block with
    /// its `hc_post`, or the routed block up to its download, the routed
    /// layer's parts marked for the walk's timer; then a KDA layer's stores
    /// into the open take of the mark its unit ends on.
    fn front(&mut self, port: &mut BatchLeg<'a, GlmHost>, at: At) -> Result<(), GpuError> {
        let (l, u) = (at.layer, at.unit);
        let kind = self.p.cfg[l].kind;
        let routed = kind.host_leg();
        if routed && self.plant == Some(u) {
            self.plant = None;
            return Err(GpuError::State {
                what: WHAT,
                missing: "the planted failure in a prompt group's walk",
            });
        }
        let enq = if routed { port.part_start() } else { None };
        if routed {
            port.mark(at, Mark::Front as usize)?;
        }
        let r = (|| {
            self.hc_in(l, u, Sub::Attn)?;
            match kind.mixer {
                MixerKind::DeltaRule => self.kda(l, u)?,
                MixerKind::Latent => self.mla(l, u)?,
                MixerKind::Gqa => {
                    return Err(shape(format!(
                        "layer {l}: a GQA mixer, which glm5next has none of"
                    )));
                }
            }
            self.hc_out(u)?;
            self.hc_in(l, u, Sub::Ffn)?;
            match kind.ffn {
                FfnKind::Dense => {
                    self.dense(l, u)?;
                    self.hc_out(u)
                }
                FfnKind::Moe => {
                    self.route(port, at)?;
                    port.mark(at, Mark::Down as usize)
                }
            }
        })();
        if routed {
            port.part_end(at, enq);
        }
        r?;
        if kind.mixer == MixerKind::DeltaRule {
            self.take_stores(at)?;
        }
        Ok(())
    }

    /// A routed layer's card experts and shared expert under its host leg.
    fn shadow(&mut self, port: &mut BatchLeg<'a, GlmHost>, at: At) -> Result<(), GpuError> {
        if !self.p.cfg[at.layer].kind.host_leg() {
            return Ok(());
        }
        let enq = port.part_start();
        let r = self
            .shadow_rows(at)
            .and_then(|()| port.mark(at, Mark::Shadow as usize));
        port.part_end(at, enq);
        r
    }

    /// A routed layer's host sums (the walk's serve uploaded them) plus the
    /// shadow's, and `hc_post`.
    fn back(&mut self, port: &mut BatchLeg<'a, GlmHost>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.p.cfg[l].kind.host_leg() {
            return Ok(());
        }
        let enq = port.part_start();
        let r = (|| {
            if self.p.card.tier_k(l) > 0 {
                self.join_tier(port, at)?;
            }
            let n = self.p.d.embd;
            let t = self.t[at.unit];
            let b = &mut *self.b;
            let shadowed = if self.p.card.sums(l) { &b.pre } else { &b.sh_y };
            self.gpu.elem().enqueue_add(
                self.gpu.stream(),
                port.hsum(),
                shadowed,
                t * n,
                &mut b.out,
            )?;
            self.hc_out(at.unit)?;
            port.mark(at, Mark::Back as usize)
        })();
        port.part_end(at, enq);
        r
    }
}
