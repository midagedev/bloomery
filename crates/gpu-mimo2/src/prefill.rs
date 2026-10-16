//! The prompt batch: a prompt of `P` ids fed in batches of at most [`T_MAX`]
//! positions ([`prefill`]), which leaves the model where `P` decode steps
//! over the same ids leave it in everything a later step reads: every layer's
//! key and value rows at those positions, and the last position's logits.
//! Every launch writes, per token, what the step's one-token launch writes
//! (the projections by chunks of [`COL_GROUP`] columns, the rope-and-append,
//! the norms, the router's two launches and the adds over the batch's rows),
//! the flash is each row's own over the positions at and before it, and the
//! host serves each routed block's tokens in one union call whose column is
//! the step's output; so a batch writes the steps' bits, whatever the cut.
//!
//! A call is cut by [`runtime::prompt::batches`] into batches of at most
//! [`T_MAX`] positions, and its batches run in groups of
//! [`set_prefill_group`]'s size ([`runtime::prompt::groups`]); a group is one
//! walk of the runtime's layer schedule at the point `(G, T, Batch)`
//! ([`runtime::sched::walk`]), one unit a batch, through the host tier's
//! batch port ([`BatchLeg`]). The body holds no checkpoint, so a call has no
//! marks: it is cut at its end alone. Per layer:
//! - the front: the attention sub-layer over the batch's rows, then either
//!   the dense block whole, or the routed block's norm, its router over the
//!   rows and the download of the rows, ids and weights to the host;
//! - the shadow: none — the routed block has no shared expert and no card
//!   expert;
//! - the back: the host's sums (the walk's serve uploaded them) plus the
//!   residual after the attention.
//!
//! The head runs after the last layer of the batch that holds the call's last
//! position, for that position alone.
//!
//! A window layer runs the flash over the batch's rows in one launch, a full
//! layer in chunks of [`FLASH_ROWS`]: the segment partials a launch holds
//! grow with its rows and a full layer cuts a row's head into
//! [`bloomery_gpu::flash_gqa::SEGMENTS`] segments, a window layer into the
//! window's own. Every row's key and value rows are appended before the first
//! chunk attends, and a row reads only the positions at and before its own
//! (its live count).
//!
//! A group's units share every buffer one item writes and the same or the
//! next part of that item reads; what a unit's back, or its later layers,
//! read after the next unit's front ran is the unit's own (`UnitBufs`: the
//! layer's input and output rows, the residual after the attention, the
//! positions). That holds only behind a dense prefix: a group of two or more
//! is refused by name on a load with a dense layer past a routed one.
//!
//! A call that fails stands the model where it found it: its positions are
//! taken back (the stores keep every row, and a row past the model's position
//! is dead), unless a fault poisoned the model. The fault word is read at the
//! end of every group.

use std::ops::Range;

use bloomery_gpu::flash_gqa::{GqaK192Args, HEAD, HEAD_K192, window_segments};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::read_fault;
use bloomery_gpu::rope_neox::K192Args;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, Gpu, GpuError};
use bloomery_gpu_deepseek41::router::mimo2::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::DeviceBuffer;
use model::moe::UNION_MAX_COLS;
use runtime::layer::FfnKind;
pub use runtime::prompt::PrefillMode;
use runtime::prompt::{
    GROUP_LEVER, call_batches, call_end, check_group, chunks, group_sets, groups,
    refuse_dense_after_routed,
};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use super::{Body, Dims, Kernels, LayerCfg, LayerNames, Mimo2Model, RopeBase, Store, WHAT, shape};
use crate::ffn::FfnNames;

/// Positions one batch runs at most: the host union's columns.
pub const T_MAX: usize = UNION_MAX_COLS;

/// Rows one flash launch of a full layer runs at most.
pub const FLASH_ROWS: usize = 16;

/// The largest `BLOOMERY_PREFILL_GROUP`: the lever registry's.
const GROUP_MAX: usize = bloomery_levers::PREFILL_GROUP_MAX as usize;
const _: () = assert!(GROUP_MAX as u64 == bloomery_levers::PREFILL_GROUP_MAX);

/// The body's prompt feed: its mode, the batches a group runs, the batch's
/// buffers once a batch feed made them, and whether a call runs.
pub(crate) struct PromptState {
    mode: PrefillMode,
    /// Batches a group runs layer by layer ([`set_prefill_group`]): one until
    /// set.
    group: usize,
    batch: Option<Box<Batch>>,
    /// A prompt call by batches runs: its group is not changed under it.
    in_call: bool,
}

impl PromptState {
    /// The steps feed with no buffers: what a load starts as.
    pub(crate) fn new() -> PromptState {
        PromptState {
            mode: PrefillMode::Steps,
            group: 1,
            batch: None,
            in_call: false,
        }
    }

    /// Device bytes of the batch's buffers, 0 before they are made.
    pub(crate) fn bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.bytes())
    }
}

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
    /// The embedding rows on the host before their upload, and each token's
    /// position and live count (the position plus one).
    rows: Vec<f32>,
    pos_host: Vec<u32>,
    keys_host: Vec<u32>,
    /// The layer's input rows: the embedding for the first layer, each
    /// layer's block output after it.
    x: DeviceBuffer<f32>,
    /// The residual after the attention, which the layer's block and back
    /// read.
    x1: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
}

impl UnitBufs {
    /// One unit's buffers for up to `cap` tokens of `n` values. Load-time or
    /// first-prompt only.
    fn new(gpu: &Gpu, n: usize, cap: usize) -> Result<UnitBufs, GpuError> {
        let stream = gpu.stream();
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
        let zu = |len: usize| DeviceBuffer::<u32>::zeroed(stream, len);
        Ok(UnitBufs {
            rows: vec![0.0; cap * n],
            pos_host: vec![0; cap],
            keys_host: vec![0; cap],
            x: z(cap * n)?,
            x1: z(cap * n)?,
            pos: zu(cap)?,
            n_keys: zu(cap)?,
        })
    }

    fn bytes(&self) -> usize {
        self.x.num_bytes() + self.x1.num_bytes() + self.pos.num_bytes() + self.n_keys.num_bytes()
    }

    /// Card bytes of one unit's buffers for up to `cap` tokens of `n` values:
    /// what [`UnitBufs::new`] allocates, counted before it does.
    fn planned(n: usize, cap: usize) -> usize {
        size_of::<f32>() * 2 * cap * n + size_of::<u32>() * 2 * cap
    }
}

/// Every buffer a group's units share besides the host sums, in values: the
/// one place the sizes are read, by the allocation and by the bytes the
/// refusal counts. Each buffer is written and read inside one part of one
/// item, or by an item's front and read by its back with no front between
/// them, so the units use it one after another in stream order.
struct Lens {
    /// The attention's normed input and projection output, the routed block's
    /// normed rows and its router's scores, ids and weights, per token.
    xn: usize,
    out: usize,
    normed: usize,
    probs: usize,
    ids: usize,
    weights: usize,
    /// The fused q·k·v rows of the widest layer, the roped query heads and
    /// the flash's output rows, per token.
    qkv: usize,
    q: usize,
    y: usize,
    /// The dense block's gate·up·SwiGLU rows of one chunk of columns.
    h: usize,
    /// The flash's partials at the widest launch any layer makes.
    part_v: usize,
    part_ms: usize,
}

impl Lens {
    /// The lengths for batches of up to `cap` tokens of `d`'s widths over the
    /// layers `cfg`.
    fn of(d: &Dims, cfg: &[LayerCfg], cap: usize) -> Lens {
        let qkv = cfg
            .iter()
            .map(|c| d.heads * HEAD_K192 + c.attn.kv_heads * (HEAD_K192 + HEAD))
            .max()
            .unwrap_or(0);
        let ff = cfg.iter().map(|c| c.ffn.ff).max().unwrap_or(0);
        let launch = |per_seg: usize| {
            cfg.iter()
                .map(|c| {
                    let a = c.attn;
                    flash_rows(a.window, cap) * d.heads * window_segments(a.window) * per_seg
                })
                .max()
                .unwrap_or(0)
        };
        Lens {
            xn: cap * d.embd,
            out: cap * d.embd,
            normed: cap * d.embd,
            probs: cap * N_EXPERT,
            ids: cap * N_USED,
            weights: cap * N_USED,
            qkv: cap * qkv,
            q: cap * d.heads * HEAD_K192,
            y: cap * d.heads * HEAD,
            h: COL_GROUP * ff.max(1),
            part_v: launch(HEAD),
            part_ms: launch(2),
        }
    }

    /// The device bytes of the buffers these lengths make (every value is
    /// four bytes).
    fn bytes(&self) -> usize {
        size_of::<f32>()
            * (self.xn
                + self.out
                + self.normed
                + self.probs
                + self.ids
                + self.weights
                + self.qkv
                + self.q
                + self.y
                + self.h
                + self.part_v
                + self.part_ms)
    }
}

/// The rows one flash launch of a layer with `window` runs of a batch of `t`
/// tokens: the whole batch on a window layer, [`FLASH_ROWS`] on a full one.
fn flash_rows(window: usize, t: usize) -> usize {
    if window > 0 { t } else { FLASH_ROWS.min(t) }
}

/// The shared buffers.
struct Bufs {
    cap: usize,
    xn: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
}

impl Bufs {
    /// The shared buffers for batches of up to `cap` tokens. Load-time or
    /// first-prompt only.
    fn new(gpu: &Gpu, l: &Lens, cap: usize) -> Result<Bufs, GpuError> {
        let stream = gpu.stream();
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
        Ok(Bufs {
            cap,
            xn: z(l.xn)?,
            out: z(l.out)?,
            normed: z(l.normed)?,
            probs: z(l.probs)?,
            ids: DeviceBuffer::zeroed(stream, l.ids)?,
            weights: z(l.weights)?,
            qkv: z(l.qkv)?,
            q: z(l.q)?,
            y: z(l.y)?,
            h: z(l.h)?,
            part_v: z(l.part_v)?,
            part_ms: z(l.part_ms)?,
        })
    }

    fn bytes(&self) -> usize {
        let f = [
            &self.xn,
            &self.out,
            &self.normed,
            &self.probs,
            &self.weights,
            &self.qkv,
            &self.q,
            &self.y,
            &self.h,
            &self.part_v,
            &self.part_ms,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>() + self.ids.num_bytes()
    }
}

/// Positions a batch buffer holds: the batch's most, or the stores' when
/// fewer.
fn cap_of(ctx: usize) -> usize {
    T_MAX.min(ctx)
}

/// The call of `ids` from the model's position, refused by name before
/// anything runs: on a poisoned model, for no id, and past the positions the
/// stores hold — the model then stands where it stood, every store as it
/// was. Returns the call's end.
fn check_call(m: &Mimo2Model, ids: &[u32]) -> Result<u32, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let ctx = m.body(WHAT)?.ctx;
    call_end(m.pos(), ids.len(), ctx).map_err(|r| shape(r.to_string()))
}

/// Feed `ids` from where `m` stands by the body's mode ([`set_prefill`]) and
/// return the argmax after the last: [`prefill`] or the steps, each call
/// refused by name before anything runs when it would pass the stores'
/// positions, so neither feed stops part of the way there.
pub fn feed(m: &mut Mimo2Model, ids: &[u32]) -> Result<u32, GpuError> {
    check_call(m, ids)?;
    match m.body(WHAT)?.prompt.mode {
        PrefillMode::Batch => prefill(m, ids),
        PrefillMode::Steps => m.step(ids),
    }
}

/// The body's feed mode.
pub fn prefill_mode(m: &Mimo2Model) -> Result<PrefillMode, GpuError> {
    Ok(m.body(WHAT)?.prompt.mode)
}

/// Set `m`'s feed to `mode`; the batch feed's buffers, the host tier's batch
/// sets and the host union's slabs are made here, once, so that a timed
/// prompt allocates nothing. Returns whether it made any.
pub fn set_prefill(m: &mut Mimo2Model, mode: PrefillMode) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.prompt.mode = mode;
    if mode == PrefillMode::Steps || body.prompt.batch.is_some() {
        return Ok(false);
    }
    body.make_batch(gpu)?;
    Ok(true)
}

/// The batches a prompt group of `m` runs layer by layer: `g` consecutive
/// batches of a call walked as one group ([`groups`]: a lone last batch joins
/// the group before it), each layer-batch's front enqueued ahead of the
/// previous one's host serve; 1 runs each batch alone, and every `g` writes
/// the same bits. Refused by name inside a call, for a `g` outside 1 to the
/// lever's most, and for a `g` of 2 or more on a load whose dense layers do
/// not all come before its routed ones. When the batch's buffers are made and
/// hold fewer units than `g` needs ([`group_sets`]), the units it lacks are
/// made here, between calls. Returns whether it made any.
pub fn set_prefill_group(m: &mut Mimo2Model, g: usize) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    if body.prompt.in_call {
        return Err(shape(format!(
            "a prompt group of {g} batches set inside a prompt call"
        )));
    }
    check_group(g, GROUP_MAX).map_err(|r| GpuError::Shape {
        what: GROUP_LEVER,
        detail: r.to_string(),
    })?;
    body.refuse_late_dense(g)?;
    body.prompt.group = g;
    body.grow_units(gpu)
}

/// A made prompt batch's device bytes ([`prompt_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptBytes {
    /// All the batch's buffers: the shared ones, the units' and the host
    /// sums'.
    pub total: usize,
    /// The units made: as many as the group holds batches.
    pub units: usize,
    /// The card's free device bytes after them, which the load's plan
    /// reserved none of them out of.
    pub free: usize,
}

/// `m`'s prompt batch's device bytes; `None` before a batch feed made them.
pub fn prompt_bytes(m: &Mimo2Model) -> Result<Option<PromptBytes>, GpuError> {
    let Some(b) = m.body(WHAT)?.prompt.batch.as_deref() else {
        return Ok(None);
    };
    let (free, _) = m.gpu().mem_info()?;
    Ok(Some(PromptBytes {
        total: b.bytes(),
        units: b.units.len(),
        free,
    }))
}

/// The batches a prompt group of `m` runs ([`set_prefill_group`]).
pub fn prefill_group(m: &Mimo2Model) -> Result<usize, GpuError> {
    Ok(m.body(WHAT)?.prompt.group)
}

/// Feed `ids` from where `m` stands in batches (module doc) and return the
/// argmax after the last. Refused by name before anything runs: on a
/// poisoned model, past the stores' positions, with the taps armed (a tap
/// holds one step's layer outputs, and a batch runs many). The batch's
/// buffers are made by the first call when [`set_prefill`] has not made
/// them.
pub fn prefill(m: &mut Mimo2Model, ids: &[u32]) -> Result<u32, GpuError> {
    let to = check_call(m, ids)?;
    let from = m.pos();
    {
        let (gpu, _, body) = m.body_parts(WHAT)?;
        if body.taps.is_some() {
            return Err(shape(
                "a prompt batch with the taps armed: a tap holds one step's layer outputs"
                    .to_string(),
            ));
        }
        body.make_batch(gpu)?;
        body.prompt.in_call = true;
    }
    let r = call_groups(m, ids, from, to);
    if let Ok((_, _, body)) = m.body_parts(WHAT) {
        body.prompt.in_call = false;
    }
    r
}

/// [`prefill`]'s groups: per group its walk ([`Body::enqueue_group`]) through
/// the model's pass. A group that fails takes the call back.
fn call_groups(m: &mut Mimo2Model, ids: &[u32], from: u32, to: u32) -> Result<u32, GpuError> {
    let batches = call_batches(from, to, &[to], T_MAX);
    let group = m.body(WHAT)?.prompt.group;
    let mut argmax = None;
    for gr in groups(batches.len(), group) {
        let runs = &batches[gr];
        let (Some(first), Some(end)) = (runs.first(), runs.last()) else {
            continue;
        };
        let seg = &ids[(first.start - from) as usize..(end.end - from) as usize];
        let last = end.end == to;
        let ran = m.run_rows(seg.len(), WHAT, |gpu, w, body, head, pos| {
            body.enqueue_group(gpu, w, head, seg, pos, runs, last)
        });
        match ran {
            Ok(t) => argmax = t.or(argmax),
            Err(e) => return Err(take_back(m, from, e)),
        }
    }
    argmax.ok_or(GpuError::State {
        what: WHAT,
        missing: "the head of the batch that holds the call's last position",
    })
}

/// A call from `from` that failed with `e`: its positions taken back, so the
/// model stands where the call found it. A fault poisons the model and is
/// returned as it came: nothing runs on it until a reset.
fn take_back(m: &mut Mimo2Model, from: u32, e: GpuError) -> GpuError {
    if matches!(e, GpuError::Fault { .. }) || m.poisoned().is_some() {
        return e;
    }
    match m.rollback(from) {
        Ok(()) => e,
        Err(r) => shape(format!(
            "a prompt call from position {from} failed ({e}), and taking it back failed too ({r})"
        )),
    }
}

impl Body {
    /// Refused by name for a group of `g` batches on a load with a dense
    /// layer past a routed one ([`refuse_dense_after_routed`]).
    fn refuse_late_dense(&self, g: usize) -> Result<(), GpuError> {
        refuse_dense_after_routed(self.cfg.iter().map(|c| c.kind.host_leg()), g)
            .map_err(|r| shape(r.to_string()))
    }

    /// Refused by name when `units` more units of `cap` tokens and `shared`
    /// more bytes pass the card's free device bytes: the load's plan reserves
    /// none for the prompt batch, so its buffers come out of what the load
    /// left free, and a batch that does not fit is refused here, not by the
    /// driver.
    fn refuse_unreserved(&self, gpu: &Gpu, units: usize, shared: usize) -> Result<(), GpuError> {
        let cap = cap_of(self.ctx);
        let need = shared + units * UnitBufs::planned(self.dims.embd, cap);
        if need == 0 {
            return Ok(());
        }
        let (free, _) = gpu.mem_info()?;
        if need <= free {
            return Ok(());
        }
        Err(shape(format!(
            "a prompt batch of {cap} positions, a group of {} batches: {need} B of buffers, and \
             the card has {free} B free; the plan reserves none for the prompt batch",
            self.prompt.group
        )))
    }

    /// The batch feed's buffers — the shared ones and a set for each unit a
    /// group of the body's size holds ([`group_sets`]) — the host tier's
    /// batch sets for as many tokens and the host union's slabs, made once.
    /// Refused by name for a group of 2 or more on a load with a dense layer
    /// past a routed one, and for buffers past the card's free bytes.
    fn make_batch(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.prompt.batch.is_some() {
            return Ok(());
        }
        self.refuse_late_dense(self.prompt.group)?;
        let cap = cap_of(self.ctx);
        let n = self.dims.embd;
        let lens = Lens::of(&self.dims, &self.cfg, cap);
        let sets = group_sets(self.prompt.group);
        let shared = lens.bytes() + size_of::<f32>() * cap * n;
        let planned = shared + sets * UnitBufs::planned(n, cap);
        self.refuse_unreserved(gpu, sets, shared)?;
        let bufs = Bufs::new(gpu, &lens, cap)?;
        let units = (0..sets)
            .map(|_| UnitBufs::new(gpu, n, cap))
            .collect::<Result<Vec<_>, _>>()?;
        let hsum = DeviceBuffer::zeroed(gpu.stream(), cap * n)?;
        let batch = Batch { bufs, units, hsum };
        if batch.bytes() != planned {
            return Err(shape(format!(
                "the prompt batch takes {} B; the sizes the refusal counts add to {planned}",
                batch.bytes()
            )));
        }
        self.hybrid.prepare_batch(gpu.context(), cap)?;
        self.hybrid.host_mut().prepare_union(cap)?;
        gpu.stream().synchronize()?;
        self.prompt.batch = Some(Box::new(batch));
        Ok(())
    }

    /// The units the body's group needs ([`group_sets`]) that the batch's
    /// buffers lack, made: between calls only. Nothing before the buffers are
    /// made, or when they hold enough. Returns whether it made any.
    fn grow_units(&mut self, gpu: &Gpu) -> Result<bool, GpuError> {
        let need = group_sets(self.prompt.group);
        let more = self
            .prompt
            .batch
            .as_deref()
            .map_or(0, |b| need.saturating_sub(b.units.len()));
        self.refuse_unreserved(gpu, more, 0)?;
        let n = self.dims.embd;
        if let Some(b) = self.prompt.batch.as_deref_mut()
            && more > 0
        {
            for _ in 0..more {
                b.units.push(UnitBufs::new(gpu, n, b.bufs.cap)?);
            }
            gpu.stream().synchronize()?;
        }
        Ok(more > 0)
    }

    /// Enqueue one group of `ids` from position `pos`, its batches `runs`
    /// (consecutive, from `pos`): each unit's rows and positions on the card,
    /// one walk `(units, T, Batch)` over every layer, and the head after the
    /// last unit when `last` — the pass [`bloomery_gpu::GpuModel::run_rows`]
    /// runs, returning whether the head was enqueued. A group waits for its
    /// launches and reads the fault word once at its end, so a fault ends the
    /// call before the next group; the last group's word rides the head's
    /// readback.
    #[allow(
        clippy::too_many_arguments,
        reason = "one pass's inputs: the model's parts run_rows lends, the group's ids, \
                  batches and end flag"
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
    ) -> Result<bool, GpuError> {
        let n = self.dims.embd;
        let layers = self.cfg.len();
        let Body {
            hybrid,
            cfg,
            names,
            dims,
            k,
            s,
            stores,
            ropes,
            embd,
            prompt,
            ctx,
            ..
        } = self;
        let Batch { bufs, units, hsum } = prompt.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers (set_prefill)",
        })?;
        let g = runs.len();
        let ts: Vec<usize> = runs.iter().map(|r| (r.end - r.start) as usize).collect();
        if g == 0 || g > units.len() {
            return Err(shape(format!(
                "a group of {g} batches on buffers of {} units",
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
            for (&token, row) in ids[at..at + t]
                .iter()
                .zip(ub.rows[..t * n].chunks_exact_mut(n))
            {
                embd.fill_into(token, row)?;
            }
            for i in 0..t {
                ub.pos_host[i] = p0 + i as u32;
                ub.keys_host[i] = p0 + i as u32 + 1;
            }
            span_mut(WHAT, &mut ub.x, 0, t * n)?.copy_from_host(stream, &ub.rows[..t * n])?;
            span_mut(WHAT, &mut ub.pos, 0, t)?.copy_from_host(stream, &ub.pos_host[..t])?;
            span_mut(WHAT, &mut ub.n_keys, 0, t)?.copy_from_host(stream, &ub.keys_host[..t])?;
            at += t;
        }
        let mut prog = PromptProgram {
            gpu,
            w,
            k,
            d: dims,
            cfg,
            names,
            stores,
            ropes,
            no_bias: &s.no_bias,
            ctx: *ctx,
            b: bufs,
            u: &mut units[..g],
            t: &ts,
        };
        let o = Overlap {
            units: g,
            cols: ts.iter().copied().max().unwrap_or(1),
            port: PortKind::Batch,
        };
        let mut leg = BatchLeg::new(stream, hybrid, hsum, cap);
        leg.set_unit_cols(&ts);
        sched::walk(o, layers, &mut leg, &mut prog)?;
        if last {
            prog.head(head)?;
        }
        read_fault(WHAT, gpu, hybrid, last)
    }
}

/// One group's walk: the body's parts, the shared buffers, each unit's own
/// and its tokens.
struct PromptProgram<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    d: &'a Dims,
    cfg: &'a [LayerCfg],
    names: &'a [LayerNames],
    stores: &'a mut [Store],
    ropes: &'a [RopeBase],
    /// The zero selection bias of a router the file gives none.
    no_bias: &'a DeviceBuffer<f32>,
    ctx: usize,
    b: &'a mut Bufs,
    u: &'a mut [UnitBufs],
    t: &'a [usize],
}

impl PromptProgram<'_> {
    /// Layer `l`'s attention over unit `u`'s batch (`attn::attention`'s
    /// launches over the batch's rows): `x` in, the residual `x1` out. The
    /// projections run by chunks of [`COL_GROUP`] columns, the rope-and-append
    /// and the add over the whole batch, the flash as the module doc says.
    fn attention(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        const W: &str = "mimo2 prefill attention";
        let (gpu, w, t) = (self.gpu, self.w, self.t[u]);
        let stream = gpu.stream();
        let (d, c) = (*self.d, self.cfg[l]);
        let a = c.attn;
        let n = &self.names[l].attn;
        let fault = gpu.layer_sink(l)?;
        let (b, ub, store) = (&mut *self.b, &mut self.u[u], &mut self.stores[l]);
        let width = d.heads * HEAD_K192 + a.kv_heads * (HEAD_K192 + HEAD);
        gpu.elem().enqueue_rms_norm(
            stream,
            &ub.x,
            w.f32_buf(WHAT, &n.norm)?,
            d.rms_eps,
            d.embd,
            t,
            &mut b.xn,
        )?;
        for (c0, cn) in chunks(t, COL_GROUP) {
            let xs = span(W, &b.xn, c0 * d.embd, cn * d.embd)?;
            let mut ys = span_mut(W, &mut b.qkv, c0 * width, cn * width)?;
            w.q8_gemv_mcol(gpu, WHAT, &n.qkv, &xs, cn, &mut ys)?;
        }
        self.k.rope.enqueue_neox_append_k192(
            stream,
            K192Args {
                qkv: &mut b.qkv,
                q: &mut b.q,
                table: &self.ropes[c.rope].rows.table,
                pos: &ub.pos,
                v_scale: a.v_scale,
                n_head: d.heads,
                n_kv: a.kv_heads,
                ctx: self.ctx,
                m: t,
                fault,
                cache_k: &mut store.k,
                cache_v: &mut store.v,
            },
        )?;
        let sinks = n
            .sinks
            .as_deref()
            .map(|name| w.f32_buf(WHAT, name))
            .transpose()?;
        let (q_row, y_row) = (d.heads * HEAD_K192, d.heads * HEAD);
        for (c0, cn) in chunks(t, flash_rows(a.window, t)) {
            let q = span(W, &b.q, c0 * q_row, cn * q_row)?;
            let keys = span(W, &ub.n_keys, c0, cn)?;
            let mut y = span_mut(W, &mut b.y, c0 * y_row, cn * y_row)?;
            self.k.flash.enqueue_pass_k192(
                stream,
                GqaK192Args {
                    q: &q,
                    kc: &store.k,
                    vc: &store.v,
                    n_keys: &keys,
                    scale: d.score_scale,
                    n_kv: a.kv_heads,
                    ctx: self.ctx,
                    m: cn,
                    window: a.window,
                    sinks,
                    part_v: &mut b.part_v,
                    part_ms: &mut b.part_ms,
                    fault,
                    y: &mut y,
                },
                d.heads,
            )?;
        }
        for (c0, cn) in chunks(t, COL_GROUP) {
            let ys = span(W, &b.y, c0 * y_row, cn * y_row)?;
            let mut out = span_mut(W, &mut b.out, c0 * d.embd, cn * d.embd)?;
            w.q8_gemv_mcol(gpu, WHAT, &n.o, &ys, cn, &mut out)?;
        }
        gpu.elem()
            .enqueue_add(stream, &ub.x, &b.out, t * d.embd, &mut ub.x1)
    }

    /// Layer `l`'s dense block over unit `u`'s batch (`ffn::dense`'s launches
    /// over the batch's rows): the norm over every row, the gate·up·SwiGLU and
    /// the down projection by chunks of [`COL_GROUP`] columns, and the add of
    /// the residual into `x`.
    fn dense(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        const W: &str = "mimo2 prefill dense";
        let (gpu, w, t) = (self.gpu, self.w, self.t[u]);
        let stream = gpu.stream();
        let (d, c) = (*self.d, self.cfg[l]);
        let FfnNames::Dense {
            norm,
            gate,
            up,
            down,
        } = &self.names[l].ffn
        else {
            return Err(other_kind(l));
        };
        let (b, ub) = (&mut *self.b, &mut self.u[u]);
        gpu.elem().enqueue_rms_norm(
            stream,
            &ub.x1,
            w.f32_buf(WHAT, norm)?,
            d.rms_eps,
            d.embd,
            t,
            &mut b.xn,
        )?;
        let (gate, up) = (w.resident(WHAT, gate)?, w.resident(WHAT, up)?);
        for (c0, cn) in chunks(t, COL_GROUP) {
            let xs = span(W, &b.xn, c0 * d.embd, cn * d.embd)?;
            self.k.experts.enqueue_shexp_gate_up_mcol(
                stream,
                gate,
                up,
                &xs,
                cn,
                c.ffn.limit,
                &mut b.h,
            )?;
            let mut out = span_mut(W, &mut b.out, c0 * d.embd, cn * d.embd)?;
            w.q8_gemv_mcol(gpu, WHAT, down, &b.h, cn, &mut out)?;
        }
        gpu.elem()
            .enqueue_add(stream, &ub.x1, &b.out, t * d.embd, &mut ub.x)
    }

    /// Layer `l`'s routed block up to its host leg for `at`'s unit
    /// (`ffn::front`'s launches over the batch's rows): the norm into
    /// `normed`, the router over every row (the scores, then each row's
    /// picks), and the download of the rows, weights and ids to the host.
    fn route(&mut self, port: &mut BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (gpu, w, t) = (self.gpu, self.w, self.t[at.unit]);
        let stream = gpu.stream();
        let d = *self.d;
        let fault = gpu.layer_sink(l)?;
        let FfnNames::Moe { norm, router, bias } = &self.names[l].ffn else {
            return Err(other_kind(l));
        };
        let (b, ub) = (&mut *self.b, &self.u[at.unit]);
        gpu.elem().enqueue_rms_norm(
            stream,
            &ub.x1,
            w.f32_buf(WHAT, norm)?,
            d.rms_eps,
            d.embd,
            t,
            &mut b.normed,
        )?;
        let bias = match bias {
            Some(name) => w.f32_buf(WHAT, name)?,
            None => self.no_bias,
        };
        self.k.router.enqueue_router_rows(
            stream,
            w.f32_tensor(WHAT, router)?,
            &b.normed,
            bias,
            d.scale,
            t,
            &mut b.probs,
            &mut b.ids,
            &mut b.weights,
            fault,
        )?;
        let key = port.key(at);
        port.hybrid()
            .enqueue_download(stream, [&b.normed, &b.weights], &b.ids, key)
    }

    /// After the last layer: the group's last unit's last token's output
    /// row into the head's input, and the head.
    fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let n = self.d.embd;
        let (Some(ub), Some(&t)) = (self.u.last(), self.t.last()) else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a unit of the group for the head",
            });
        };
        let row = span(WHAT, &ub.x, (t - 1) * n, n)?;
        head.input_mut()
            .copy_from_device_async(&row, self.gpu.stream())?;
        head.enqueue(self.gpu, self.w)
    }
}

/// `l`'s names as the other kind's, by name.
fn other_kind(l: usize) -> GpuError {
    shape(format!(
        "layer {l}: the names of another block kind than the prompt batch runs"
    ))
}

impl<'a> LayerProgram for PromptProgram<'a> {
    type Port = BatchLeg<'a, HostRun>;

    /// A routed layer's; the dense lead has none.
    fn host_leg(&self, at: At) -> bool {
        self.cfg.get(at.layer).is_some_and(|c| c.kind.host_leg())
    }

    /// The attention sub-layer, then the dense block whole, or the routed
    /// block up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, u) = (at.layer, at.unit);
        self.attention(l, u)?;
        match self.cfg[l].kind.ffn {
            FfnKind::Dense => self.dense(l, u),
            FfnKind::Moe => self.route(port, at),
        }
    }

    /// A routed layer's host sums (the walk's serve uploaded them) plus the
    /// residual after the attention, into the unit's `x`.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.cfg[at.layer].kind.host_leg() {
            return Ok(());
        }
        let t = self.t[at.unit];
        let ub = &mut self.u[at.unit];
        self.gpu.elem().enqueue_add(
            self.gpu.stream(),
            port.hsum(),
            &ub.x1,
            t * self.d.embd,
            &mut ub.x,
        )
    }
}
