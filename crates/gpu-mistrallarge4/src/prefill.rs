//! The prompt batch: a prompt of `P` ids fed in units of at most [`T_MAX`] positions
//! ([`prefill`]), which leaves the model where `P` decode steps over the same ids leave it in
//! everything a later step reads: every owner layer's latent rows and every full indexer layer's
//! index keys at those positions, and the last position's logits.
//!
//! The walk is the common owner's ([`prompt_walk`]): the units of a group, the host staging of
//! their rows, the routed block over the unit's slots and the dense lead's block over its rows,
//! the head with the file's softcap as data (`BatchBufs::set_softcap`) and the fault word. This
//! file holds what Mistral Large 4's architecture decides, the attention over a unit's rows:
//! - every projection one GEMM a site over the unit's rows ([`project_wide`]), the query's rope
//!   and an owner's fused key norm, rope and append at each row's own position and rope row;
//! - a full layer's lightning indexer over the rows: the rows' index keys appended first, then
//!   each row's causal top-k over the rows up to its own, [`IDX_CHUNK`] rows a launch (the
//!   scores scratch's rows), into the unit's list; the launch's rows' index queries and head
//!   weights, which the projections write a row a token, transposed first into the score pass's
//!   token per column (the step's one row is both layouts); a shared layer reads that list as it
//!   stands, the unit's latest full layer's; a list a gate forces ([`Body::force_selection`])
//!   replaces the call's last row's pick in that list;
//! - the absorbed queries and the SSMax pairs at each row's own position;
//! - the attention over each row's picked rows of the layer's cache — its own on an owner, its
//!   source's on a reader, which appends nothing — the selected latent flash in place on a roped
//!   layer and the selected 512-wide walk on a NoPE one, [`ATTN_ROWS`] rows a launch (the
//!   partials' rows), each followed by its rows' value up-projection;
//! - the output projection over the rows and the residual.
//!
//! A group's units share every buffer one front writes and the same front reads (the rows, the
//! wide forms, the partials, the indexer's scores); what a unit's later layers read after the
//! next unit's front ran is the unit's own: its record (each row's live count, selected count,
//! walk pair, rope row and its bits) and its list.

use bloomery_gpu::arch::hybrid::{Site, WideCx, project_wide, wide_acts_at};
use bloomery_gpu::flash::{FlashGeom, FlashMergeSelArgs, FlashSelArgs, segments_for};
use bloomery_gpu::gemm::{GemmKernels, GemmRoute};
use bloomery_gpu::head::Head;
use bloomery_gpu::kquant::SelDown;
use bloomery_gpu::latent::{LatentAppendArgs, Rows as LatentRows};
use bloomery_gpu::model::{GpuModel, refuse_reserve_past};
use bloomery_gpu::prompt_timing::PromptStats;
use bloomery_gpu::q8f32::Q8_0GemvHeadsMcolArgs;
use bloomery_gpu::rope_table::{Direction, RopeTable};
use bloomery_gpu::site::{WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, FaultSink, Gpu, GpuError, Pitched, Q8Act, dtod_2d};
use bloomery_gpu_deepseek41::attn::{self, AttnArgs, SelectedRows};
use bloomery_gpu_deepseek41::index_key::IndexKeyInputArgs;
use bloomery_gpu_deepseek41::indexer::{IndexerArgs, IndexerScratch};
use bloomery_gpu_deepseek41::prompt_walk::{self, Batch, BatchAttention, UnitRows};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::mistrallarge4::place;
use runtime::layer::{FfnKind, Layer as LayerKind};

use crate::body::{Body, Forced, Kernels, LayerStore, Shape, kq_scale, stack, up_sel};
use crate::features::FeatureRows;
use crate::tensors::{self, Layer, VUp};

/// One selection-tapped layer of a prompt batch: the layer, its kept rows and their scores.
pub type SelectionRows = (usize, Vec<u32>, Vec<f32>);

/// What this file's refusals name.
const WHAT: &str = "mistrallarge4 prefill";

/// The most positions a unit holds: the plan's prompt rows, one host union call's columns.
pub const T_MAX: usize = place::PROMPT_ROWS;
const _: () = assert!(T_MAX <= model::moe::UNION_MAX_COLS);

/// Rows an indexer launch scores ([`place::IDX_CHUNK`]): the scores scratch's rows, each over the
/// context. The scratch is sized by the context, so the plan's scratch term holds it; the rows'
/// queries and weights a token per column are the prompt reserve's.
pub const IDX_CHUNK: usize = place::IDX_CHUNK;
// A unit's rows are whole chunks, so a chunk's rows past the unit's stay inside its buffers.
const _: () = assert!(T_MAX.is_multiple_of(IDX_CHUNK));

/// Rows an attention launch reads ([`place::ATTN_ROWS`]): the partials' rows.
pub const ATTN_ROWS: usize = place::ATTN_ROWS;

/// Columns of one per-head q8_0 launch: the launcher's most.
const MCOL: usize = 8;
// A sub-block's value up-projection is one launch.
const _: () = assert!(ATTN_ROWS <= MCOL);

fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

fn missing(what: &'static str) -> GpuError {
    GpuError::State {
        what: WHAT,
        missing: what,
    }
}

/// The partial segments a query row of the attention takes: the wider of the roped layers'
/// selected flash over a list's width and the NoPE layers' walk over the one-row ring and the
/// list. Refused by name where the kernels' count is not the plan's ([`place::attn_segments`]),
/// whose reserve holds the partials.
fn attn_segments(sh: &Shape) -> Result<usize, GpuError> {
    let segs = segments_for(sh.top_k).max(attn::segments(1, sh.top_k));
    let planned = place::attn_segments(sh.top_k as u64);
    if segs as u64 != planned {
        return Err(shape(format!(
            "the attention cuts a row into {segs} segments over a list of {}, the plan's \
             reserve {planned}",
            sh.top_k
        )));
    }
    Ok(segs)
}

/// The rows every front writes and reads within itself, shared by a group's units: the
/// attention's intermediates over [`T_MAX`] rows (a sub-block's [`ATTN_ROWS`] for the attention
/// output and its partials), the NoPE walk's sinks and its one-row ring.
struct Rows {
    xn: DeviceBuffer<f32>,
    x_ln: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    /// The absorbed queries, a roped layer's flash rows `[q_rope | q_nope2]`.
    q_cur: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    kv_s: DeviceBuffer<f32>,
    kvr: DeviceBuffer<f32>,
    qi: DeviceBuffer<f32>,
    qi_moved: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    kidx: DeviceBuffer<f32>,
    kidx_moved: DeviceBuffer<f32>,
    /// An indexer launch's [`IDX_CHUNK`] rows of index queries and of head weights a token per
    /// column, `[idx_q × IDX_CHUNK]` and `[idx_heads × IDX_CHUNK]`: the score pass's layout.
    qi_cols: DeviceBuffer<f32>,
    wi_cols: DeviceBuffer<f32>,
    /// A sub-block's heads' latent outputs, and their q8_1 form with the ids that make each
    /// head its own expert of a `_sel` value stack.
    att: DeviceBuffer<f32>,
    v_act: Q8Act,
    v_ids: DeviceBuffer<u32>,
    /// The heads' values over the unit's rows, the output projection's input.
    av: DeviceBuffer<f32>,
    /// The output projection, the residual's delta.
    r: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// Each head's sink logit, −∞: the NoPE walk has none.
    sinks: DeviceBuffer<f32>,
    /// The NoPE walk's window ring: one row it never reads (every row's window count is 0).
    ring: DeviceTensor<u16>,
}

impl Rows {
    /// The rows for `sh`'s widths. Load-time only.
    fn new(stream: &CudaStream, sh: &Shape) -> Result<Rows, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let t = T_MAX;
        let width = sh.latent + sh.rope;
        let (qk, idx_q, hv) = (
            sh.heads * sh.head_k,
            sh.idx_heads * sh.idx_d,
            sh.heads * sh.head_v,
        );
        let q_rows = ATTN_ROWS * sh.heads;
        let segs = attn_segments(sh)?;
        let ids: Vec<u32> = (0..ATTN_ROWS).flat_map(|_| 0..sh.heads as u32).collect();
        let sinks: Vec<f32> = std::iter::repeat_n(f32::NEG_INFINITY, sh.heads).collect();
        Ok(Rows {
            xn: z(t * sh.hidden)?,
            x_ln: z(t * sh.hidden)?,
            q: z(t * qk)?,
            q_rope: z(t * qk)?,
            q_cur: z(t * sh.heads * width)?,
            kv_a: z(t * width)?,
            kv_s: z(t * width)?,
            kvr: z(t * width)?,
            qi: z(t * idx_q)?,
            qi_moved: z(t * idx_q)?,
            wi: z(t * sh.idx_heads)?,
            kidx: z(t * sh.idx_d)?,
            kidx_moved: z(t * sh.idx_d)?,
            qi_cols: z(IDX_CHUNK * idx_q)?,
            wi_cols: z(IDX_CHUNK * sh.idx_heads)?,
            att: z(q_rows * sh.latent)?,
            v_act: Q8Act::with_slots(stream, q_rows, sh.latent)?,
            v_ids: DeviceBuffer::from_host(stream, &ids)?,
            av: z(t * hv)?,
            r: z(t * sh.hidden)?,
            part_v: z(q_rows * segs * sh.latent)?,
            part_ms: z(q_rows * segs * 2)?,
            sinks: DeviceBuffer::from_host(stream, &sinks)?,
            ring: DeviceTensor::zeroed(stream, 1, sh.latent)?,
        })
    }

    /// Device bytes of the rows.
    fn bytes(&self) -> usize {
        [
            &self.xn,
            &self.x_ln,
            &self.q,
            &self.q_rope,
            &self.q_cur,
            &self.kv_a,
            &self.kv_s,
            &self.kvr,
            &self.qi,
            &self.qi_moved,
            &self.wi,
            &self.kidx,
            &self.kidx_moved,
            &self.qi_cols,
            &self.wi_cols,
            &self.att,
            &self.av,
            &self.r,
            &self.part_v,
            &self.part_ms,
            &self.sinks,
        ]
        .into_iter()
        .map(DeviceBuffer::num_bytes)
        .sum::<usize>()
            + self.v_act.device_bytes()
            + self.v_ids.num_bytes()
            + self.ring.buf().num_bytes()
    }
}

/// A unit's own record, read by its later layers after the next unit's front ran: each row's
/// live count `pos + 1` (the indexer's, [`T_MAX`] words and the list's width after them), its
/// selected count `min(pos + 1, top_k)` (the flash's), its NoPE walk pair `(0, selected)`, and
/// its rope row and the row's bits (the score pass's table), and the position one past its last
/// row (the unit that holds the call's last row is the one a forced list reaches).
struct Record {
    end: u32,
    ints: DeviceBuffer<u32>,
    counts: DeviceBuffer<u32>,
    vis: DeviceBuffer<u32>,
    cs: DeviceBuffer<f32>,
    words: DeviceBuffer<u32>,
}

impl Record {
    /// A unit's record for `sh`'s widths. Load-time only.
    fn new(stream: &CudaStream, sh: &Shape) -> Result<Record, GpuError> {
        let t = T_MAX;
        Ok(Record {
            end: 0,
            ints: DeviceBuffer::zeroed(stream, t + 1)?,
            counts: DeviceBuffer::zeroed(stream, t)?,
            vis: DeviceBuffer::zeroed(stream, 2 * t)?,
            cs: DeviceBuffer::zeroed(stream, t * sh.rope)?,
            words: DeviceBuffer::zeroed(stream, t * sh.rope)?,
        })
    }

    fn bytes(&self) -> usize {
        self.ints.num_bytes()
            + self.counts.num_bytes()
            + self.vis.num_bytes()
            + self.cs.num_bytes()
            + self.words.num_bytes()
    }
}

/// The host side of a record, one for every unit: each copy completes on the host before the
/// staging returns.
struct HostRecord {
    ints: Vec<u32>,
    counts: Vec<u32>,
    vis: Vec<u32>,
    cs: Vec<f32>,
    words: Vec<u32>,
}

/// A full layer's selection at the call's last row, copied as the walk leaves it (a gate's
/// instrument): the row's list and its scores (the scratch's row, valid below its live count
/// when that passes the list's width).
struct SelTap {
    list: DeviceBuffer<u32>,
    scores: DeviceBuffer<f32>,
}

/// What the prompt batch holds beyond the walk's units: the shared rows, every unit's record and
/// list, the wide forms the projections read, the GEMM's kernels and its one-expert table, the
/// indexer's scores scratch, and the gate's selection taps when armed. Made once, so that a timed
/// prompt allocates nothing.
pub struct Prefill {
    rows: Rows,
    records: Vec<Record>,
    /// Each unit's list, `[T_MAX × top_k]`: its latest full layer's, which the shared layers after
    /// it read. A unit's own, for a group's next unit writes its list between this unit's layers.
    lists: Vec<DeviceBuffer<u32>>,
    host: HostRecord,
    /// The wide forms at the model width, and at each other width a site reads.
    wide: WideActs,
    arms: Vec<(usize, WideActs)>,
    gemm: GemmKernels,
    route: GemmRoute,
    /// The scores of [`IDX_CHUNK`] rows over the context: sized by the context, so the plan's
    /// scratch term holds it beside the step's own scratches.
    scratch: IndexerScratch,
    /// Each full layer's selection at the call's last row, when a gate armed them.
    sel: Option<Vec<Option<SelTap>>>,
}

impl Prefill {
    /// The batch's buffers for a group of one unit over `layers`' sites at `sh`'s widths.
    /// Load-time only.
    pub(crate) fn new(gpu: &Gpu, sh: &Shape, layers: &[Layer]) -> Result<Prefill, GpuError> {
        let stream = gpu.stream();
        let sites: Vec<&Site> = layers.iter().flat_map(tensors::sites).collect();
        let wide = wide_acts_at(stream, sites.iter().copied(), (T_MAX, sh.hidden))?;
        let mut widths: Vec<usize> = sites
            .iter()
            .map(|s| s.k())
            .filter(|&k| k != sh.hidden)
            .collect();
        widths.sort_unstable();
        widths.dedup();
        let arms = widths
            .into_iter()
            .map(|k| Ok((k, wide_acts_at(stream, sites.iter().copied(), (T_MAX, k))?)))
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Prefill {
            rows: Rows::new(stream, sh)?,
            records: vec![Record::new(stream, sh)?],
            lists: vec![DeviceBuffer::zeroed(stream, T_MAX * sh.top_k)?],
            host: HostRecord {
                ints: vec![0; T_MAX + 1],
                counts: vec![0; T_MAX],
                vis: vec![0; 2 * T_MAX],
                cs: Vec::with_capacity(T_MAX * sh.rope),
                words: Vec::with_capacity(T_MAX * sh.rope),
            },
            wide,
            arms,
            gemm: GemmKernels::load(gpu.context())?,
            route: GemmRoute::new(stream, T_MAX, 1)?,
            scratch: IndexerScratch::new(stream, IDX_CHUNK, sh.ctx)?,
            sel: None,
        })
    }

    /// Set the group to `group` units a walk: the records and lists it lacks are made here,
    /// between calls. Returns whether it made any.
    fn set_group(&mut self, gpu: &Gpu, sh: &Shape, group: usize) -> Result<bool, GpuError> {
        let sets = prompt_walk::group_sets(group);
        let have = self.records.len();
        for _ in have..sets {
            self.records.push(Record::new(gpu.stream(), sh)?);
            self.lists
                .push(DeviceBuffer::zeroed(gpu.stream(), T_MAX * sh.top_k)?);
        }
        Ok(sets > have)
    }

    /// The device bytes the plan's prompt reserve holds: the rows, the records, the wide forms
    /// and the one-expert table.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.bytes()
            + self.records.iter().map(Record::bytes).sum::<usize>()
            + self
                .lists
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.wide.bytes()
            + self.arms.iter().map(|a| a.1.bytes()).sum::<usize>()
            + self.route.bytes()
    }

    /// The device bytes the plan's scratch term holds: the indexer's scores scratch, sized by
    /// the context.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.device_bytes()
    }

    /// The device bytes of the gate's selection taps, when armed.
    #[must_use]
    pub fn tap_bytes(&self) -> usize {
        self.sel
            .iter()
            .flatten()
            .flatten()
            .map(|s| s.list.num_bytes() + s.scores.num_bytes())
            .sum()
    }
}

/// Set the prompt group to `g` units a walk: a group's units run their layers together, so the
/// card work of one unit overlaps the previous one's host serve; 1 runs each unit alone, and
/// every `g` writes the same bits. Refused by name for a `g` outside 1 to the lever's most (the
/// walk's check, before anything is made) and for buffers past the plan's prompt reserve, which
/// holds the units of the group the plan was priced for ([`place::configure_group`]). The
/// units, records and lists it lacks are made here, between calls. Returns whether it made any.
pub fn set_prefill_group(m: &mut GpuModel<Body>, g: usize) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let sh = body.sh;
    let units = body.units.set_group(gpu, sh.hidden, g)?;
    let records = body.prefill.set_group(gpu, &sh, g)?;
    body.check_prompt_reserve(g)?;
    // A feature tap's prompt buffer holds the group's rows, and is outside the plan's prompt
    // reserve (the draft's card room is the tap's).
    if let Some(f) = body.feat.as_mut() {
        f.set_group(gpu, g)?;
    }
    Ok(units || records)
}

/// Arm (or disarm) the prompt walks' timing ([`prompt_walk::BatchBufs::set_stats`]): one event a
/// part boundary a (layer, unit), the serves' host times and the call's fault wall, which
/// [`take_prompt_stats`] reads. Unarmed, a walk records nothing and waits for nothing beyond its
/// own serves.
pub fn set_prompt_stats(m: &mut GpuModel<Body>, on: bool) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let layers = body.layers.len();
    body.units.set_stats(gpu, layers, on)
}

/// The prompt's walk stats, taken: the next prompt starts clean. `Ok(None)` with the timing
/// unarmed or no batch walked.
pub fn take_prompt_stats(m: &mut GpuModel<Body>) -> Result<Option<PromptStats>, GpuError> {
    Ok(m.body_parts(WHAT)?.2.units.take_stats())
}

/// Arm or disarm the prompt batch's taps (a gate's instrument): each layer's output at the
/// call's last position (the shell's, which the walk writes) and each full layer's list and
/// scores at the call's last row. Made here, between calls.
pub fn set_prefill_taps(m: &mut GpuModel<Body>, on: bool) -> Result<(), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let sh = body.sh;
    body.shell.set_taps(gpu, sh.n_layer, on)?;
    let p = &mut body.prefill;
    if !on {
        p.sel = None;
        return Ok(());
    }
    let stream = gpu.stream();
    p.sel = Some(
        body.layers
            .iter()
            .map(|a| {
                a.kind
                    .full
                    .then(|| {
                        Ok::<_, GpuError>(SelTap {
                            list: DeviceBuffer::zeroed(stream, sh.top_k)?,
                            scores: DeviceBuffer::zeroed(stream, sh.ctx)?,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(())
}

/// Feed `ids` from position `at` in units of up to [`T_MAX`] positions and enqueue the head
/// after the last: the pass `GpuModel::run_rows` runs, returning whether the head was enqueued
/// (always, on success). Refused by name before anything runs: for no id and past the stores'
/// positions. A group waits for its launches and reads the fault word once at its end, so a
/// fault ends the call before the next group; the last group's word rides the head's readback.
pub fn prefill(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
    ids: &[u32],
    at: u32,
) -> Result<bool, GpuError> {
    prefill_with(gpu, w, body, head, ids, at, None)
}

/// [`prefill`] with the feature rows of the call's last `window` positions handed to the sink
/// after each group of units that holds any of them ([`FeatureRows`]): the units, launches and
/// bits are the plain call's, plus the tap's copies. Refused by name, before anything runs,
/// without a feature tap ([`crate::features::attach_features`]). The step's tap rows are not
/// the call's: a read of them after it is refused by name.
pub fn prefill_with(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
    ids: &[u32],
    at: u32,
    features: Option<FeatureRows<'_, '_>>,
) -> Result<bool, GpuError> {
    let Body {
        sh,
        layers,
        cfg,
        stores,
        k,
        table,
        ffn,
        shell,
        embed,
        units,
        prefill,
        forced,
        feat,
        ..
    } = body;
    let sh = *sh;
    let n_layer = cfg.len();
    let call_end = u32::try_from(ids.len())
        .ok()
        .and_then(|n| at.checked_add(n))
        .ok_or_else(|| shape(format!("{} ids at {at} pass a position word", ids.len())))?;
    if let Some(f) = feat.as_mut() {
        f.step.plan(0);
    }
    let mut reader = match (features, feat.as_mut()) {
        (None, _) => None,
        (Some(call), Some(f)) => {
            let taps = f
                .batch
                .as_mut()
                .ok_or_else(|| missing("the feature tap's prompt buffer"))?;
            Some(call.reader(taps, (at, ids.len()))?)
        }
        (Some(_), None) => return Err(missing("a feature tap (attach_features)")),
    };
    let Prefill {
        rows,
        records,
        lists,
        host,
        wide,
        arms,
        gemm,
        route,
        scratch,
        sel,
    } = prefill;
    let mut attention = Attention {
        gpu,
        w,
        sh,
        layers,
        cfg,
        stores,
        k,
        table,
        rows,
        records,
        lists,
        host,
        wide,
        arms,
        gemm,
        route,
        scratch,
        sel: sel.as_deref_mut(),
        forced: forced.as_ref(),
        call_end,
    };
    let mut batch = Batch {
        gpu,
        w,
        bufs: units,
        attention: &mut attention,
        ffn,
        hybrid: &mut shell.hybrid,
        embed,
        head,
        taps: shell.taps.as_deref_mut(),
        embd: sh.hidden,
        ctx: sh.ctx,
        layers: n_layer,
    };
    let mut at = at;
    match reader.as_mut() {
        Some(reader) => batch.prompt_entering(ids, &mut at, reader),
        None => batch.prompt(ids, &mut at),
    }
}

/// The attention over a unit's rows: what Mistral Large 4's architecture decides, as the walk's
/// attention.
struct Attention<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    sh: Shape,
    layers: &'a [Layer],
    cfg: &'a [LayerKind],
    stores: &'a mut [LayerStore],
    k: &'a Kernels,
    table: &'a RopeTable,
    rows: &'a mut Rows,
    records: &'a mut [Record],
    lists: &'a mut [DeviceBuffer<u32>],
    host: &'a mut HostRecord,
    wide: &'a mut WideActs,
    arms: &'a mut [(usize, WideActs)],
    gemm: &'a GemmKernels,
    route: &'a mut GemmRoute,
    scratch: &'a mut IndexerScratch,
    sel: Option<&'a mut [Option<SelTap>]>,
    /// A gate's forced list, and the position one past the call's last row, which names the unit
    /// whose last row it stands for.
    forced: Option<&'a Forced>,
    call_end: u32,
}

impl BatchAttention for Attention<'_> {
    /// The unit's record: each row's live count, selected count, walk pair and rope row at its
    /// position `at + t`; the rows past `m` count 0, so an indexer launch's rows past the unit
    /// score and pick nothing.
    fn stage(&mut self, u: usize, at: u32, m: usize) -> Result<(), GpuError> {
        let sh = self.sh;
        if m == 0 || m > T_MAX {
            return Err(shape(format!("a unit of {m} rows: 1 to {T_MAX}")));
        }
        let n_records = self.records.len();
        let end = u32::try_from(m)
            .ok()
            .and_then(|m| at.checked_add(m))
            .ok_or_else(|| shape(format!("a unit of {m} rows at {at} passes a position word")))?;
        let rec = self
            .records
            .get_mut(u)
            .ok_or_else(|| shape(format!("unit {u} past the {n_records} records")))?;
        rec.end = end;
        let h = &mut *self.host;
        let top_k = u32::try_from(sh.top_k)
            .map_err(|_| shape(format!("a list of {} does not fit a word", sh.top_k)))?;
        h.cs.clear();
        for (t, p) in (at..).take(T_MAX).enumerate() {
            // A causal row sees itself: its live count is at least one.
            let live = if t < m { p + 1 } else { 0 };
            h.ints[t] = live;
            h.counts[t] = live.min(top_k);
            h.vis[2 * t] = 0;
            h.vis[2 * t + 1] = live.min(top_k);
            if t < m {
                self.table.push(p, Direction::Forward, &mut h.cs);
            }
        }
        h.ints[T_MAX] = top_k;
        if h.cs.len() != m * sh.rope {
            return Err(shape(format!(
                "the rope rows hold {} values, the unit's {m} rows {}",
                h.cs.len(),
                m * sh.rope
            )));
        }
        h.words.clear();
        h.words.extend(h.cs.iter().map(|f| f.to_bits()));
        let stream = self.gpu.stream();
        rec.ints.copy_from_host(stream, &h.ints)?;
        rec.counts.copy_from_host(stream, &h.counts)?;
        rec.vis.copy_from_host(stream, &h.vis)?;
        span_mut(WHAT, &mut rec.cs, 0, m * sh.rope)?.copy_from_host(stream, &h.cs)?;
        span_mut(WHAT, &mut rec.words, 0, m * sh.rope)?.copy_from_host(stream, &h.words)?;
        Ok(())
    }

    /// Layer `l`'s attention over unit `u`'s `m` rows, then the residual into `x1`.
    fn attend(&mut self, l: usize, u: usize, m: usize, rows: UnitRows<'_>) -> Result<(), GpuError> {
        self.gemm.enqueue_route_dense(
            self.gpu.stream(),
            m,
            self.route,
            self.gpu.unlabelled_sink(),
        )?;
        self.layer(l, u, m, rows)
    }

    /// The dense lead's block, a routed block on every other layer.
    fn ffn(&self, l: usize) -> FfnKind {
        self.cfg[l].ffn
    }
}

impl<'a> Attention<'a> {
    /// Layer `l` over unit `u`'s `m` rows (the step's stages 2–8 over the rows): `x` in, the
    /// residual `x1` out.
    #[allow(
        clippy::too_many_lines,
        reason = "the layer's stages in their order: the projections, the key path, the indexer, the absorption, the attention by sub-block and the output"
    )]
    fn layer(&mut self, l: usize, u: usize, m: usize, rows: UnitRows<'_>) -> Result<(), GpuError> {
        let (gpu, w, sh) = (self.gpu, self.w, self.sh);
        let (layers, k, gemm): (&'a [Layer], &'a Kernels, &'a GemmKernels) =
            (self.layers, self.k, self.gemm);
        let a = &layers[l];
        let stream = gpu.stream();
        let elem = gpu.elem();
        let fault = gpu.layer_sink(l)?;
        let rope = a.kind.rope;
        let pe = sh.pe(rope);
        let width = sh.latent + pe;
        let (qk, idx_q) = (sh.heads * sh.head_k, sh.idx_heads * sh.idx_d);
        let rec = self
            .records
            .get(u)
            .ok_or_else(|| shape(format!("unit {u} has no record")))?;
        let (forced, call_end) = (self.forced, self.call_end);
        let unit_list = self
            .lists
            .get_mut(u)
            .ok_or_else(|| shape(format!("unit {u} has no list")))?;
        let b = &mut *self.rows;
        let cx = WideCx {
            what: WHAT,
            gpu,
            w,
            wk: WideKernels {
                gemm,
                g32: Some(&k.g32),
            },
            m,
        };
        // 2. the norm over the rows, and 3–5. every site that reads them in one quantization:
        //    the query, an owner's latent, a full layer's index key and head weights.
        let gain = w.f32_buf(WHAT, &a.attn_norm)?;
        elem.enqueue_rms_norm(stream, rows.x, gain, sh.eps, sh.hidden, m, &mut b.xn)?;
        let owner = if a.kind.cache == l {
            Some(
                a.kv_a
                    .as_ref()
                    .ok_or_else(|| missing("an owner's kv_a site"))?,
            )
        } else {
            None
        };
        let idx = if a.kind.full {
            Some(
                a.idx
                    .as_ref()
                    .ok_or_else(|| missing("a full layer's indexer"))?,
            )
        } else {
            None
        };
        {
            let forms = (&mut *self.wide, &*self.route);
            let x = &b.xn;
            match (owner, idx) {
                (Some(kv), Some(i)) => project_wide(
                    &cx,
                    forms,
                    l,
                    x,
                    &mut [
                        (&a.q, qk, &mut b.q),
                        (kv, width, &mut b.kv_a),
                        (&i.k, sh.idx_d, &mut b.kidx),
                        (&i.proj, sh.idx_heads, &mut b.wi),
                    ],
                ),
                (Some(kv), None) => project_wide(
                    &cx,
                    forms,
                    l,
                    x,
                    &mut [(&a.q, qk, &mut b.q), (kv, width, &mut b.kv_a)],
                ),
                (None, Some(i)) => project_wide(
                    &cx,
                    forms,
                    l,
                    x,
                    &mut [
                        (&a.q, qk, &mut b.q),
                        (&i.k, sh.idx_d, &mut b.kidx),
                        (&i.proj, sh.idx_heads, &mut b.wi),
                    ],
                ),
                (None, None) => project_wide(&cx, forms, l, x, &mut [(&a.q, qk, &mut b.q)]),
            }?;
        }
        // The rope of the query's columns, each row at its own rope row.
        if rope {
            elem.enqueue_rope(
                stream,
                &b.q,
                &rec.cs,
                sh.rope,
                (qk / sh.rope) as u32,
                m,
                &mut b.q_rope,
            )?;
        }
        // 4. an owner's key path: the fused norm, rope and f16 append of each row at its own
        //    position — the bare latent's append on a NoPE owner, nothing on a reader.
        if let Some(_kv) = owner {
            let norm_name = a
                .kv_a_norm
                .as_ref()
                .ok_or_else(|| missing("an owner's kv_a norm"))?;
            let norm = w.f32_buf(WHAT, norm_name)?;
            let cache = self.stores[l]
                .latent
                .as_mut()
                .ok_or_else(|| missing("the layer's latent store"))?;
            if rope {
                k.fused.enqueue_kv_norm_rope_append_rows(
                    stream,
                    &b.kv_a,
                    norm,
                    &rec.cs,
                    rows.pos,
                    sh.eps,
                    sh.latent,
                    sh.rope,
                    m,
                    &mut b.kv_s,
                    &mut b.kvr,
                    cache,
                    fault,
                )?;
            } else {
                k.latent.enqueue_latent_append(
                    stream,
                    LatentAppendArgs {
                        rows: LatentRows {
                            x: &b.kv_a,
                            stride: sh.latent,
                            m,
                        },
                        off: 0,
                        gain: norm,
                        pos: rows.pos,
                        eps: sh.eps,
                        fault,
                        cache,
                    },
                )?;
            }
        }
        // 5. a full layer's lightning indexer: the rows' keys first, then every row's causal
        //    top-k into the unit's list.
        if let Some(i) = idx {
            elem.enqueue_ln_rows(
                stream,
                &b.xn,
                w.f32_buf(WHAT, &i.q_norm_w)?,
                w.f32_buf(WHAT, &i.q_norm_b)?,
                sh.idx_eps,
                sh.hidden,
                m,
                fault,
                &mut b.x_ln,
            )?;
            project_wide(
                &cx,
                (&mut *self.wide, &*self.route),
                l,
                &b.x_ln,
                &mut [(&i.q_b, idx_q, &mut b.qi)],
            )?;
            // A roped index head moves its leading rope values to the kernel's tail; a NoPE
            // head's projections stay as written.
            if rope {
                rope_to_tail(
                    stream,
                    (&b.qi, &mut b.qi_moved),
                    (sh.idx_d, sh.rope),
                    m * sh.idx_heads,
                )?;
                rope_to_tail(stream, (&b.kidx, &mut b.kidx_moved), (sh.idx_d, sh.rope), m)?;
            }
            elem.enqueue_ssmax_rows(
                stream,
                if rope { &mut b.qi_moved } else { &mut b.qi },
                rows.pos,
                w.f32_buf(WHAT, &i.ssmax_w)?,
                w.f32_buf(WHAT, &i.ssmax_b)?,
                sh.idx_heads,
                sh.idx_d,
                m,
                fault,
            )?;
            let st = self.stores[l]
                .idx
                .as_mut()
                .ok_or_else(|| missing("a full layer's index store"))?;
            k.index_key.enqueue_index_key_input(
                stream,
                IndexKeyInputArgs {
                    x: if rope { &b.kidx_moved } else { &b.kidx },
                    gain: &i.k_norm_w,
                    bias: &i.k_norm_b,
                    pos: rows.pos,
                    cs: &rec.cs,
                    eps: sh.idx_eps,
                    n_dims: pe,
                    m,
                    cache: &mut st.keys,
                    fault,
                },
            )?;
            let keys = &st.keys;
            let indexer = if rope {
                &k.indexer_rope
            } else {
                &k.indexer_nope
            };
            let q_all = if rope { &b.qi_moved } else { &b.qi };
            for c0 in (0..m).step_by(IDX_CHUNK) {
                // The score pass reads a token per column, `q[(h·idx_d + v)·tokens + t]`, and
                // the projections write a token per row: the chunk's `IDX_CHUNK` rows of `n`
                // values transposed into `n` rows of `IDX_CHUNK` (the transpose's `rows` are the
                // chunk's tokens here).
                let q = span(WHAT, q_all, c0 * idx_q, IDX_CHUNK * idx_q)?;
                k.transpose
                    .enqueue(stream, &q, IDX_CHUNK, idx_q, &mut b.qi_cols)?;
                let wv = span(WHAT, &b.wi, c0 * sh.idx_heads, IDX_CHUNK * sh.idx_heads)?;
                k.transpose
                    .enqueue(stream, &wv, IDX_CHUNK, sh.idx_heads, &mut b.wi_cols)?;
                let mut list = span_mut(WHAT, unit_list, c0 * sh.top_k, IDX_CHUNK * sh.top_k)?;
                let mut args = IndexerArgs {
                    q: &b.qi_cols,
                    w: &b.wi_cols,
                    ints: &rec.ints,
                    n_vis_at: c0,
                    top_k_at: T_MAX,
                    tables: &rec.words,
                    rope_at: c0 * sh.rope,
                    rope_stride: sh.rope,
                    keys,
                    tokens: IDX_CHUNK,
                    scratch: &mut *self.scratch,
                    list: &mut list,
                    stride: sh.top_k,
                };
                indexer.enqueue_score(stream, &mut args)?;
                indexer.enqueue_topk(stream, &mut args, &rec.ints, c0, T_MAX)?;
            }
            // A gate's forced list stands for the call's last row's pick, in the unit that
            // holds that row.
            if let Some(f) = forced.filter(|f| f.layer == l && rec.end == call_end) {
                let mut row = span_mut(WHAT, unit_list, (m - 1) * sh.top_k, sh.top_k)?;
                row.copy_from_device_async(&f.list, stream)?;
            }
            if let Some(tap) = self
                .sel
                .as_deref_mut()
                .and_then(|s| s.get_mut(l))
                .and_then(Option::as_mut)
            {
                // The unit's last row sits in its last chunk, whose scores the scratch holds.
                let last = m - 1;
                let rows_k = keys.rows();
                let list = span(WHAT, unit_list, last * sh.top_k, sh.top_k)?;
                let scores = span(
                    WHAT,
                    &self.scratch.scores,
                    (last % IDX_CHUNK) * rows_k,
                    rows_k,
                )?;
                tap.list.copy_from_device_async(&list, stream)?;
                tap.scores.copy_from_device_async(&scores, stream)?;
            }
        }
        // 6. the absorbed queries, eight rows a launch, each the step's own bits; a roped layer's
        //    flash rows take their rope spans; then the SSMax at each row's position.
        let (qs, dd) = w.q8_planes(WHAT, &a.k_b)?;
        let q_row = sh.heads * width;
        for c0 in (0..m).step_by(MCOL) {
            let n = MCOL.min(m - c0);
            let x = span(WHAT, &b.q, c0 * qk, n * qk)?;
            let mut y = span_mut(WHAT, &mut b.q_cur, c0 * q_row, n * q_row)?;
            gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                stream,
                Q8_0GemvHeadsMcolArgs {
                    qs,
                    d: dd,
                    x: &x,
                    rows_per_head: sh.latent,
                    x_head_stride: sh.head_k,
                    y_head_stride: width,
                    y_off: pe,
                    m: n,
                    x_col_stride: qk,
                    y_col_stride: q_row,
                    y: &mut y,
                },
            )?;
        }
        if rope {
            // Each head's roped tail into the first `rope` values of its flash row.
            let f = std::mem::size_of::<f32>();
            dtod_2d(
                stream,
                (&b.q_rope, (sh.head_k - sh.rope) * f),
                (&mut b.q_cur, 0),
                Pitched {
                    width: sh.rope * f,
                    rows: m * sh.heads,
                    dev_pitch: sh.head_k * f,
                    host_pitch: width * f,
                },
            )?;
        }
        elem.enqueue_ssmax_rows(
            stream,
            &mut b.q_cur,
            rows.pos,
            w.f32_buf(WHAT, &a.ssmax_w)?,
            w.f32_buf(WHAT, &a.ssmax_b)?,
            sh.heads,
            width,
            m,
            fault,
        )?;
        // 7–8. the attention over each row's picked rows of the layer's cache, a sub-block of
        //    rows a launch, and the sub-block's values up-projected.
        let cache = self.stores[a.kind.cache]
            .latent
            .as_ref()
            .ok_or_else(|| missing("the cache layer's latent store"))?;
        let hv = sh.heads * sh.head_v;
        for s0 in (0..m).step_by(ATTN_ROWS) {
            let n = ATTN_ROWS.min(m - s0);
            let q = span(WHAT, &b.q_cur, s0 * q_row, n * q_row)?;
            let list = span(WHAT, unit_list, s0 * sh.top_k, n * sh.top_k)?;
            if rope {
                let counts = span(WHAT, &rec.counts, s0, n)?;
                k.flash.enqueue_flash_latent_mma_sel(
                    stream,
                    FlashSelArgs {
                        q: &q,
                        kv: cache,
                        list: &list,
                        counts: &counts,
                        stride: sh.top_k,
                        kq_scale: kq_scale(sh),
                        geom: FlashGeom {
                            tokens: n,
                            heads: sh.heads,
                            rope_dims: sh.rope,
                            latent_dims: sh.latent,
                        },
                        part_v: &mut b.part_v,
                        part_ms: &mut b.part_ms,
                        fault,
                    },
                )?;
                k.flash.enqueue_flash_merge_sel(
                    stream,
                    FlashMergeSelArgs {
                        counts: &counts,
                        stride: sh.top_k,
                        tokens: n,
                        heads: sh.heads,
                        latent_dims: sh.latent,
                        part_v: &b.part_v,
                        part_ms: &b.part_ms,
                        y: &mut b.att,
                    },
                )?;
            } else {
                let vis = span(WHAT, &rec.vis, 2 * s0, 2 * n)?;
                k.attn.enqueue(
                    stream,
                    AttnArgs {
                        q: &q,
                        window: &b.ring,
                        compressed: Some(cache),
                        selected: Some(SelectedRows {
                            rows: &list,
                            stride: sh.top_k,
                        }),
                        vis: &vis,
                        sinks: &b.sinks,
                        scale: kq_scale(sh),
                        tokens: n,
                        heads: sh.heads,
                        part_v: &mut b.part_v,
                        part_ms: &mut b.part_ms,
                        y: &mut b.att,
                        fault,
                    },
                )?;
            }
            let mut y = span_mut(WHAT, &mut b.av, s0 * hv, n * hv)?;
            values_up(
                (gpu, w, k, sh),
                (a, l, n),
                (&b.att, &mut b.v_act, &b.v_ids),
                fault,
                &mut y,
            )?;
        }
        // 8. the output projection over the rows, then the residual.
        let forms = if a.o.k() == sh.hidden {
            &mut *self.wide
        } else {
            arm_at(&mut *self.arms, a.o.k())?
        };
        project_wide(
            &cx,
            (forms, &*self.route),
            l,
            &b.av,
            &mut [(&a.o, sh.hidden, &mut b.r)],
        )?;
        elem.enqueue_add(stream, rows.x, &b.r, m * sh.hidden, rows.x1)
    }
}

/// A sub-block's `n` rows of heads' latent outputs `att` up-projected to their values into `y`
/// (`n · heads · head_v`): the file's Q8_0 planes through the per-head gemv over the rows, or a
/// `_sel` stack whose experts are the heads, each column of the q8_1 form its head's.
fn values_up(
    (gpu, w, k, sh): (&Gpu, &Weights, &Kernels, Shape),
    (a, l, n): (&Layer, usize, usize),
    (att, v_act, v_ids): (&DeviceBuffer<f32>, &mut Q8Act, &DeviceBuffer<u32>),
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    match &a.v_b {
        VUp::Planes(name) => {
            let (qs, dd) = w.q8_planes(WHAT, name)?;
            gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                stream,
                Q8_0GemvHeadsMcolArgs {
                    qs,
                    d: dd,
                    x: att,
                    rows_per_head: sh.head_v,
                    x_head_stride: sh.latent,
                    y_head_stride: sh.head_v,
                    y_off: 0,
                    m: n,
                    x_col_stride: sh.heads * sh.latent,
                    y_col_stride: sh.heads * sh.head_v,
                    y,
                },
            )
        }
        VUp::Sel { name, ty, family } => {
            gpu.enqueue_quantize_q8_1_layer(att, v_act, l)?;
            let down = SelDown {
                w: stack(w, name)?,
                act: v_act,
                sel: v_ids,
                n_slots: n * sh.heads,
                rows_per_expert: sh.head_v,
            };
            up_sel(gpu, k, (*family, *ty), &down, fault, y)
        }
    }
}

/// `rows` heads of `head` values of `src`, each head's leading `rope` values moved to its tail
/// into `dst`: the step's move of the index head's `[rope | nope]` file order into the kernel's
/// `[nope | rope]`, as two strided copies.
fn rope_to_tail(
    stream: &CudaStream,
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    (head, rope): (usize, usize),
    rows: usize,
) -> Result<(), GpuError> {
    let f = std::mem::size_of::<f32>();
    let nope = head - rope;
    let pitch = head * f;
    dtod_2d(
        stream,
        (src, rope * f),
        (&mut *dst, 0),
        Pitched {
            width: nope * f,
            rows,
            dev_pitch: pitch,
            host_pitch: pitch,
        },
    )?;
    dtod_2d(
        stream,
        (src, 0),
        (dst, nope * f),
        Pitched {
            width: rope * f,
            rows,
            dev_pitch: pitch,
            host_pitch: pitch,
        },
    )
}

/// The wide forms of input width `k` beside the model width's.
fn arm_at(arms: &mut [(usize, WideActs)], k: usize) -> Result<&mut WideActs, GpuError> {
    arms.iter_mut()
        .find(|a| a.0 == k)
        .map(|a| &mut a.1)
        .ok_or_else(|| shape(format!("no wide forms of width {k}")))
}

impl Body {
    /// The prompt batch's buffers the plan's prompt reserve holds: the attention's rows,
    /// records and lists, the walk's units and the routed block's batch buffers.
    #[must_use]
    pub fn prompt_bytes(&self) -> usize {
        self.prefill.bytes() + self.units.bytes() + self.ffn.batch_bytes()
    }

    /// Refused by name when the prompt batch's buffers, made for a prompt group of `group`, pass
    /// the plan's prompt reserve. The reserve holds the units of the group the plan was priced for
    /// ([`place::configure_group`]); a group of more units than that names its group in the
    /// refusal.
    pub(crate) fn check_prompt_reserve(&self, group: usize) -> Result<(), GpuError> {
        let sets = prompt_walk::group_sets(group);
        let what = "attention rows, unit records and lists, common walk units and shared routed \
                    batch buffers";
        let parts = if sets > 1 {
            format!(
                "{what}, at a prompt group of {group}: {sets} units, which \
                 `place::configure_group` prices before the plan"
            )
        } else {
            what.to_owned()
        };
        refuse_reserve_past(
            WHAT,
            "prompt batch buffers",
            self.prompt_reserve,
            self.prompt_bytes() as u64,
            &parts,
        )
    }

    /// Every layer's output at the last prompt call's last position, `n_layer · hidden` values
    /// in layer order (a gate's instrument; it synchronizes the stream). Refused by name with
    /// the taps unarmed ([`set_prefill_taps`]).
    pub fn prefill_taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        self.shell.taps(gpu)
    }

    /// The device bytes of the prompt batch's taps, when armed: the shell's layer outputs and
    /// the selection taps.
    #[must_use]
    pub fn prefill_tap_bytes(&self) -> usize {
        self.shell
            .taps
            .iter()
            .flatten()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self.prefill.tap_bytes()
    }

    /// Each full layer's list and scores at the last prompt call's last row, in layer order: the
    /// layer, its `top_k` list entries and its scores over the context (a gate's instrument; it
    /// synchronizes the stream). Refused by name with the taps unarmed.
    pub fn prefill_selection(&self, gpu: &Gpu) -> Result<Vec<SelectionRows>, GpuError> {
        let sel = self
            .prefill
            .sel
            .as_ref()
            .ok_or_else(|| missing("the prompt batch's selection taps (set_prefill_taps)"))?;
        let stream = gpu.stream();
        sel.iter()
            .enumerate()
            .filter_map(|(l, s)| s.as_ref().map(|s| (l, s)))
            .map(|(l, s)| {
                Ok((
                    l,
                    s.list.to_host_vec(stream)?,
                    s.scores.to_host_vec(stream)?,
                ))
            })
            .collect()
    }

    /// Every store's name and its bytes over the first `pos` positions, in layer order: an
    /// owner's latent rows and a full layer's index keys (a gate's instrument; it synchronizes
    /// the stream).
    pub fn store_rows(&self, gpu: &Gpu, pos: usize) -> Result<Vec<(String, Vec<u8>)>, GpuError> {
        if pos > self.sh.ctx {
            return Err(shape(format!(
                "{pos} positions past the stores' {}",
                self.sh.ctx
            )));
        }
        let stream = gpu.stream();
        let rows = |t: &DeviceTensor<u16>| -> Result<Vec<u8>, GpuError> {
            let all = t.buf().to_host_vec(stream)?;
            Ok(all[..pos * t.cols()]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        };
        let mut out = Vec::new();
        for (l, st) in self.stores.iter().enumerate() {
            if let Some(latent) = st.latent.as_ref() {
                out.push((format!("blk.{l}.latent"), rows(latent)?));
            }
            if let Some(idx) = st.idx.as_ref() {
                out.push((format!("blk.{l}.index_keys"), rows(&idx.keys)?));
            }
        }
        Ok(out)
    }
}
