//! K-EXAONE's MTP layer on the target's card ([`Draft`]): its weights, its ring store and its walk's
//! buffers, opened once at load beside the target ([`crate::FlintBody::open_placed_slots_drafted`]), and its
//! program ([`walk`], [`chain`], [`prompt_with`]).
//!
//! The program is the DeepSeek-style NextN block the checkpoint's `nextn.*` tensors name: row `t` is
//! the token at position `pos0 + t` beside the target's final hidden row at `pos0 + t - 1`, the hidden
//! row normed by the target's `output_norm`. Every launch is an entry the target's chain already has:
//! - the store part, every row: the token's embedding row from the file's table on the host, the
//!   hidden rows' `output_norm`, each row's `[enorm(e) | hnorm(h)]` (the MTP input pack,
//!   `bloomery_gpu::mtp`, one stream), `nextn.eh_proj` over the rows (the by-name gemv), then the
//!   layer's own head128 sliding-window attention ([`Binding::enqueue`]) over its own ring store,
//!   which appends the rows at their positions; [`WalkMode::Store`] stops here, a later row's
//!   attention reads nothing else of a row;
//! - the full part, every row (the last row's is the one read): the layer's dense SwiGLU block
//!   ([`Routed::dense_rows`], registered beside the target's dense lead by [`Routed::add_dense`]),
//!   `nextn.shared_head_norm` as the head's gain and the target's `output.weight` projection and
//!   argmax ([`Head::with_gain`]).
//!
//! The walk runs launch by launch ([`WalkMode::Graph`] is refused by name). The store is by position:
//! one walk may start at or below the end of the last walk of the sequence ([`Draft::held`]); a reset
//! and a cut behind it move that end back.
//!
//! A walk that reads the target's rows reads them by position: each target arena records the
//! positions its rows hold ([`Wrote`]) — the step, the verify and the prompt unit that last wrote it,
//! less what a cut took back — and a walk from `pos0` reads rows that hold positions `pos0 - 1 ..`
//! of it, or is refused by name. The arenas are the draft's own buffers: the step's and the verify's
//! rows are copied out of the heads' inputs at the end of the walk that wrote them, the prompt unit's
//! out of the walk's last layer ([`PromptTap`]).

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::attention_h128::{Binding, Cache, Kernels, Scratch};
use bloomery_gpu::draft::{
    DraftFeed, DraftHead, Feed, Held, Hidden, WalkBounds, WalkMode, refuse_own, refuse_rows,
};
use bloomery_gpu::head::{Head, HeadNorm};
use bloomery_gpu::host::served::{prompt_pass, take_back};
use bloomery_gpu::model::{ChainBody, Rollback};
use bloomery_gpu::mtp::{MtpInputArgs, MtpKernels};
use bloomery_gpu::rope_table::RopeRows;
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::site::{ByName, ByNameKernels, Out};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, Gpu, GpuError, GpuModel};
use bloomery_gpu_deepseek41::prompt_walk::Entering;
use bloomery_gpu_deepseek41::shared_routed::Routed;
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::names;
use model::embed::EmbedTable;
use models::LayerSpec;
use runtime::prompt::CallEnd;

use crate::body::{FlintBody, PASS_ROWS};
use crate::prefill::T_MAX;

/// What the draft's errors name.
const WHAT: &str = "flint MTP";

/// The program's [`WalkBounds::WALK_ROWS`] and [`WalkBounds::STORE_ROWS`]: the m-column kernels' width.
pub const WALK_ROWS: usize = COL_GROUP;

impl WalkBounds for Draft {
    const WALK_ROWS: usize = WALK_ROWS;
    /// No walk is captured.
    const GRAPH_ROWS: usize = 0;
    const STORE_ROWS: usize = WALK_ROWS;
}

/// A target arena whose final hidden rows a walk reads: where a step, a verify and a prompt unit leave
/// their rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlintArena {
    /// The step's one row.
    Step,
    /// A verify's two rows, row `r` the pass's row `r`.
    Pair,
    /// A prompt unit's rows, row `i` the unit's position `i`.
    Prefill,
}

/// Each target arena's [`Held`]: the draft's record of where the step, the verify and the prompt unit
/// left their rows.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Wrote {
    pub(crate) step: Held,
    pub(crate) pair: Held,
    pub(crate) prefill: Held,
}

impl Wrote {
    fn of(&self, walk: FlintArena) -> Held {
        match walk {
            FlintArena::Step => self.step,
            FlintArena::Pair => self.pair,
            FlintArena::Prefill => self.prefill,
        }
    }
}

/// The layer's tensor names.
struct Names {
    eh: String,
    enorm: String,
    hnorm: String,
    head_norm: String,
}

/// The MTP layer's weights, store, buffers and head.
pub struct Draft {
    w: Weights,
    names: Names,
    /// Whether the file holds the layer's own `shared_head_norm`; else the head's gain is the target's
    /// `output_norm`.
    own_gain: bool,
    binding: Binding,
    cache: Cache,
    scratch: Scratch,
    by_name: ByName,
    by_name_k: ByNameKernels,
    input: MtpKernels,
    layer: usize,
    hidden: usize,
    eps: f32,
    ctx: usize,
    /// The walk's rows: embeddings, the hidden rows normed, the packed inputs, the projection's output,
    /// the attention's and the block's.
    e: DeviceBuffer<f32>,
    hn: DeviceBuffer<f32>,
    packed: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    x1: DeviceBuffer<f32>,
    x2: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    row: Vec<f32>,
    /// The target arenas' rows, the draft's own copies.
    step: DeviceBuffer<f32>,
    pair: DeviceBuffer<f32>,
    prefill: DeviceBuffer<f32>,
    wrote: Wrote,
    held: usize,
    head: Head,
}

/// A shape the draft refuses, by name.
fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

fn zeroed(stream: &CudaStream, n: usize) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::zeroed(stream, n)?)
}

impl Draft {
    /// The layer `layer` (the first past the trunk) loaded beside the target: its tensors resident in
    /// the draft's own weights, its dense block registered with `ffn`, its store and buffers made, its
    /// head opened over the target's `output.weight`. Load-time only. Refused by name: a tensor the
    /// layer lacks or reads in a form the program does not, and a layer of another width.
    pub(crate) fn load<R: RouterEntry>(
        gpu: &Gpu,
        (file, tw): (&Arc<Split>, &Weights),
        ffn: &mut Routed<R>,
        (spec, layer): (&LayerSpec, usize),
        (hidden, eps, ctx): (usize, f32, usize),
    ) -> Result<Draft, GpuError> {
        let stream = gpu.stream();
        let prefix = names::blk(layer, "");
        let w = Weights::load_where(stream, file, |name| name.starts_with(&prefix))?;
        let binding = Binding::load(file, &w, layer, spec, hidden, eps, ctx)?;
        ffn.add_dense((file, &w), layer, spec)?;
        let names = Names {
            eh: names::blk(layer, "nextn.eh_proj.weight"),
            enorm: names::blk(layer, "nextn.enorm.weight"),
            hnorm: names::blk(layer, "nextn.hnorm.weight"),
            head_norm: names::blk(layer, "nextn.shared_head_norm.weight"),
        };
        let want = [2 * hidden as u64, hidden as u64];
        let dims = file.find(&names.eh).map(|(_, t)| t.dims.as_slice());
        if dims != Some(&want[..]) {
            return Err(shape(format!(
                "{}: dims {dims:?}; the input projection takes [enorm(e) | hnorm(h)], {want:?}",
                names.eh
            )));
        }
        for gain in [&names.enorm, &names.hnorm] {
            let len = w.f32_buf(WHAT, gain)?.len();
            if len != hidden {
                return Err(shape(format!(
                    "{gain}: {len} values, the model is {hidden} wide"
                )));
            }
        }
        let own_gain = w.get(&names.head_norm).is_some();
        let head = if own_gain {
            Head::with_gain(gpu, (&w, &names.head_norm), tw, eps, 1, HeadNorm::Rms)?
        } else {
            Head::with_gain(gpu, (tw, &names::output_norm()), tw, eps, 1, HeadNorm::Rms)?
        };
        let cache = Cache::new(gpu, &binding)?;
        let scratch = Scratch::new(gpu, std::slice::from_ref(&binding), WALK_ROWS)?;
        let by_name = ByName::for_sites(gpu, &w, [names.eh.as_str()], WALK_ROWS)?;
        let rows = WALK_ROWS * hidden;
        Ok(Draft {
            names,
            own_gain,
            by_name,
            by_name_k: ByNameKernels::load(gpu)?,
            input: MtpKernels::load(gpu.context())?,
            layer,
            hidden,
            eps,
            ctx,
            e: zeroed(stream, rows)?,
            hn: zeroed(stream, rows)?,
            packed: zeroed(stream, 2 * rows)?,
            x: zeroed(stream, rows)?,
            x1: zeroed(stream, rows)?,
            x2: zeroed(stream, rows)?,
            pos: DeviceBuffer::zeroed(stream, WALK_ROWS)?,
            n_keys: DeviceBuffer::zeroed(stream, WALK_ROWS)?,
            row: vec![0.0; hidden],
            step: zeroed(stream, hidden)?,
            pair: zeroed(stream, PASS_ROWS * hidden)?,
            prefill: zeroed(stream, T_MAX * hidden)?,
            wrote: Wrote::default(),
            held: 0,
            head,
            binding,
            cache,
            scratch,
            w,
        })
    }

    /// Device bytes the draft holds: its weights, store, buffers, arenas and head.
    pub fn bytes(&self) -> usize {
        self.w.resident_bytes()
            + self.cache.bytes()
            + self.scratch.bytes()
            + self.by_name.bytes()
            + self.head.resident_bytes()
            + [
                &self.e,
                &self.hn,
                &self.packed,
                &self.x,
                &self.x1,
                &self.x2,
                &self.step,
                &self.pair,
                &self.prefill,
            ]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
    }

    /// Positions the draft's store holds: a walk may start at or below it, never past it.
    pub fn held(&self) -> usize {
        self.held
    }

    /// The values of a hidden row.
    pub fn hidden_width(&self) -> usize {
        self.hidden
    }

    /// The sequence's draft state back to empty: the store's end and the arenas' records.
    pub(crate) fn reset(&mut self) {
        self.held = 0;
        self.wrote = Wrote::default();
    }

    /// Every arena's positions from `pos` on taken back, and the store's end with them, past the
    /// entry at `pos` itself: it reads the target's hidden row at `pos - 1` and the token at `pos`,
    /// which a call that continues the sequence at `pos` over the same id keeps valid, and the
    /// window's join takes it as held (`runtime::mtp::MtpDraft::rejoin_through`). A walk from `pos`
    /// rewrites it.
    pub(crate) fn cut(&mut self, pos: u32) {
        self.held = self.held.min(pos as usize + 1);
        for h in [
            &mut self.wrote.step,
            &mut self.wrote.pair,
            &mut self.wrote.prefill,
        ] {
            h.cut(pos);
        }
    }

    /// Whether a cut of `depth` positions leaves the keys the store's next walk reads: the layer's
    /// own ring, as [`FlintBody::cut_keeps`] holds the target's.
    pub(crate) fn cut_keeps(&self, depth: u64) -> bool {
        runtime::stores::cut_depth_keeps(
            depth,
            self.binding.capacity() as u64,
            self.ctx as u64,
            self.binding.window() as u64,
        )
    }

    /// The step at `pos` is about to leave its row in the step arena.
    pub(crate) fn stepping(&mut self, pos: u32) {
        self.wrote.step = Held::at(pos, 1);
    }

    /// The verify pass at `pos` is about to leave its rows in the pair arena.
    pub(crate) fn passing(&mut self, pos: u32) {
        self.wrote.pair = Held::at(pos, PASS_ROWS as u32);
    }

    /// The prompt unit's reader ([`PromptTap`]) for a call from position `from` over `layers` layers.
    pub(crate) fn tap(&mut self, layers: usize, from: u32) -> PromptTap<'_> {
        PromptTap {
            arena: &mut self.prefill,
            hidden: self.hidden,
            layers,
            from,
            runs: Vec::new(),
        }
    }

    /// The prompt unit of `rows` rows from `at` has left its rows in the prefill arena.
    pub(crate) fn wrote_prefill(&mut self, at: u32, rows: usize) {
        self.wrote.prefill = Held::at(at, rows as u32);
    }

    /// The walk's last layer's rows, copied out of the heads' inputs: the one row of a step, the two of
    /// a verify. Enqueued behind the walk, on its stream.
    pub(crate) fn keep_rows(&mut self, gpu: &Gpu, heads: &[Head]) -> Result<(), GpuError> {
        let n = self.hidden;
        let stream = gpu.stream();
        match heads {
            [one] => self.step.copy_from_device_async(one.input(), stream)?,
            [a, b] => {
                for (r, head) in [a, b].into_iter().enumerate() {
                    span_mut(WHAT, &mut self.pair, r * n, n)?
                        .copy_from_device_async(&*span(WHAT, head.input(), 0, n)?, stream)?;
                }
            }
            _ => {
                return Err(shape(format!(
                    "a walk of {} rows: the draft keeps one or two",
                    heads.len()
                )));
            }
        }
        Ok(())
    }

    /// The hidden rows of `feed` (`m` of them) normed into `self.hn`: copied from the host as they
    /// come, or the arena's rows by `output_norm`. Refused by name: a host slice of other than `m · n`
    /// values, rows past the arena, a walk at position 0 and rows that hold other positions.
    fn gather(
        &mut self,
        gpu: &Gpu,
        tw: &Weights,
        feed: &Feed<'_, FlintArena>,
    ) -> Result<(), GpuError> {
        let (n, m) = (self.hidden, feed.tokens.len());
        let stream = gpu.stream();
        match feed.hidden {
            Hidden::Host(v) => {
                if v.len() != m * n {
                    return Err(shape(format!(
                        "{} hidden values for {m} rows of {n}",
                        v.len()
                    )));
                }
                span_mut(WHAT, &mut self.hn, 0, m * n)?.copy_from_host(stream, v)?;
            }
            Hidden::Target { walk, first } => {
                let (arena, rows) = match walk {
                    FlintArena::Step => (&self.step, 1),
                    FlintArena::Pair => (&self.pair, PASS_ROWS),
                    FlintArena::Prefill => (&self.prefill, T_MAX),
                };
                self.wrote
                    .of(walk)
                    .check_read(WHAT, walk, rows, (first, m), feed.pos0)?;
                gpu.elem().enqueue_rms_norm(
                    stream,
                    &*span(WHAT, arena, first * n, m * n)?,
                    tw.f32_buf(WHAT, &names::output_norm())?,
                    self.eps,
                    n,
                    m,
                    &mut self.hn,
                )?;
            }
        }
        Ok(())
    }

    /// One walk of `feed` through the layer in `mode`: the store part always, the block and the head's
    /// enqueue past it unless the mode is [`WalkMode::Store`]. No readback.
    fn run<R: RouterEntry>(
        &mut self,
        gpu: &Gpu,
        tw: &Weights,
        (k, rope, ffn, embed): (&Kernels, &RopeRows, &mut Routed<R>, &EmbedTable),
        feed: Feed<'_, FlintArena>,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        let (n, m, pos0) = (self.hidden, feed.tokens.len(), feed.pos0);
        refuse_rows::<Draft>(WHAT, mode, m)?;
        if mode == WalkMode::Graph {
            return Err(shape(
                "a captured walk: the draft's walks run launch by launch",
            ));
        }
        if feed.media.is_some() {
            return Err(shape(
                "a walk over media rows: this program's rows are token embeddings",
            ));
        }
        if pos0 as usize > self.held || pos0 as usize + m > self.ctx {
            return Err(shape(format!(
                "a walk of {m} rows from position {pos0}; the store holds {} positions and the \
                 context is {}",
                self.held, self.ctx
            )));
        }
        let stream = gpu.stream();
        let first = self.held.min(pos0 as usize);
        self.held = first;
        for (t, &token) in feed.tokens.iter().enumerate() {
            embed.row_into(token, &mut self.row)?;
            span_mut(WHAT, &mut self.e, t * n, n)?.copy_from_host(stream, &self.row)?;
        }
        self.gather(gpu, tw, &feed)?;
        let at: Vec<u32> = (0..m as u32).map(|t| pos0 + t).collect();
        let keys: Vec<u32> = at.iter().map(|p| p + 1).collect();
        span_mut(WHAT, &mut self.pos, 0, m)?.copy_from_host(stream, &at)?;
        span_mut(WHAT, &mut self.n_keys, 0, m)?.copy_from_host(stream, &keys)?;
        self.input.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &self.e,
                h: &self.hn,
                enorm: self.w.f32_buf(WHAT, &self.names.enorm)?,
                hnorm: self.w.f32_buf(WHAT, &self.names.hnorm)?,
                hidden: n,
                eps: self.eps,
                m,
                streams: 1,
                fault: gpu.layer_sink(self.layer)?,
                out: &mut self.packed,
            },
        )?;
        self.by_name.gemv(
            gpu,
            (Some(&self.by_name_k), None),
            (&self.w, &self.names.eh),
            self.layer,
            &self.packed,
            m,
            Out::whole(&mut self.x),
        )?;
        // SAFETY: the rows' positions are consecutive from `pos0` and each count is its position + 1;
        // the store holds every position below `pos0` (the walk starts at or below `held`).
        unsafe {
            self.binding.enqueue(
                gpu,
                &self.w,
                k,
                &mut self.scratch,
                &mut self.cache,
                &self.x,
                &mut self.x1,
                &self.pos,
                &self.n_keys,
                m,
                rope,
            )?;
        }
        if mode != WalkMode::Store {
            ffn.dense_rows((gpu, &self.w), (self.layer, m), (&self.x1, &mut self.x2))?;
            self.head
                .input_mut()
                .copy_from_device_async(&*span(WHAT, &self.x2, (m - 1) * n, n)?, stream)?;
            let gain = if self.own_gain { &self.w } else { tw };
            self.head.enqueue_over(gpu, gain, tw)?;
        }
        self.held = pos0 as usize + m;
        Ok(())
    }
}

/// What a walk refuses before any launch: a row-list head (the load opens the full head alone), the
/// draft's own row (a proposal holds one id, so there is none) and a load without the layer.
fn check_feed(
    head: DraftHead,
    feed: DraftFeed<'_, FlintArena>,
) -> Result<Feed<'_, FlintArena>, GpuError> {
    head.check_opened(WHAT, false)?;
    feed.target_rows(WHAT)
}

/// One walk of `feed` into `head` in `mode`, no readback ([`Draft::run`]). A fault the walk's own
/// calls meet is its error and poisons the model; one they do not read stays on the card's fault word,
/// which the next readback names.
pub fn walk<R: RouterEntry>(
    m: &mut GpuModel<FlintBody<R>>,
    feed: DraftFeed<'_, FlintArena>,
    head: DraftHead,
    mode: WalkMode,
) -> Result<(), GpuError>
where
    FlintBody<R>: ChainBody,
{
    let feed = check_feed(head, feed)?;
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let (gpu, tw, body) = m.body_parts(WHAT)?;
    let FlintBody {
        draft,
        k,
        rope,
        ffn,
        embed,
        ..
    } = body;
    let d = draft.as_mut().ok_or_else(|| {
        shape("a walk on a load without the MTP layer (open_placed_slots_drafted loads it)")
    })?;
    let r = d.run(gpu, tw, (k, rope, ffn, embed), feed, mode);
    m.note_fault(WHAT, r)
}

/// One window's chain, one readback: `refresh` walked eager through the head, its last row's
/// prediction the proposal's one id, written to `out[0]`; the count, 1. `own` walks past it, a store
/// walk's mode and media rows are refused by name.
pub fn chain<R: RouterEntry>(
    m: &mut GpuModel<FlintBody<R>>,
    refresh: DraftFeed<'_, FlintArena>,
    own: usize,
    head: DraftHead,
    mode: WalkMode,
    out: &mut [u32],
) -> Result<usize, GpuError>
where
    FlintBody<R>: ChainBody,
{
    let refresh = check_feed(head, refresh)?;
    if own != 0 {
        return Err(refuse_own(WHAT, &format!("a chain of {own} own walks")));
    }
    if mode == WalkMode::Store {
        return Err(shape("a chain in the store walk's mode: it reads the head"));
    }
    let place = out
        .first_mut()
        .ok_or_else(|| shape("a proposal of 1 id into 0 places"))?;
    walk(m, DraftFeed::Rows(refresh), head, mode)?;
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let d = body
        .draft
        .as_ref()
        .ok_or_else(|| shape("a chain on a load without the MTP layer"))?;
    let read = d.head.tokens(gpu);
    let tokens = m.note_fault(WHAT, read)?;
    *place = *tokens
        .first()
        .ok_or_else(|| shape("a head readback of no token"))?;
    Ok(1)
}

/// The draft of `m`, refused by name on a load without it.
fn drafted<R: RouterEntry>(m: &GpuModel<FlintBody<R>>) -> Result<&Draft, GpuError>
where
    FlintBody<R>: ChainBody,
{
    m.body(WHAT)?.draft.as_ref().ok_or_else(|| {
        shape("the MTP layer on a load without it (open_placed_slots_drafted loads it)")
    })
}

/// The head the load opened for the draft: the full vocabulary.
pub fn draft_head<R: RouterEntry>(m: &GpuModel<FlintBody<R>>) -> Result<DraftHead, GpuError>
where
    FlintBody<R>: ChainBody,
{
    drafted(m).map(|_| DraftHead::Full)
}

/// [`Draft::hidden_width`].
pub fn hidden_width<R: RouterEntry>(m: &GpuModel<FlintBody<R>>) -> Result<usize, GpuError>
where
    FlintBody<R>: ChainBody,
{
    drafted(m).map(Draft::hidden_width)
}

/// [`Draft::held`].
pub fn held<R: RouterEntry>(m: &GpuModel<FlintBody<R>>) -> Result<usize, GpuError>
where
    FlintBody<R>: ChainBody,
{
    drafted(m).map(Draft::held)
}

/// Device bytes of the draft (0 on a load without it).
pub fn draft_bytes<R: RouterEntry>(m: &GpuModel<FlintBody<R>>) -> Result<usize, GpuError>
where
    FlintBody<R>: ChainBody,
{
    Ok(m.body(WHAT)?.draft.as_ref().map_or(0, Draft::bytes))
}

/// The prompt unit's row `first`, at position `pos - 1`, kept as the step arena's row.
pub fn retain_prompt_row<R: RouterEntry>(
    m: &mut GpuModel<FlintBody<R>>,
    walk: FlintArena,
    first: usize,
    pos: u32,
) -> Result<(), GpuError>
where
    FlintBody<R>: ChainBody,
{
    if walk != FlintArena::Prefill {
        return Err(shape(format!("a shared prompt arena, got {walk:?}")));
    }
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let d = body
        .draft
        .as_mut()
        .ok_or_else(|| shape("a prompt row kept on a load without the MTP layer"))?;
    d.wrote
        .prefill
        .check_read(WHAT, walk, T_MAX, (first, 1), pos)?;
    let n = d.hidden;
    span_mut(WHAT, &mut d.step, 0, n)?
        .copy_from_device_async(&*span(WHAT, &d.prefill, first * n, n)?, gpu.stream())?;
    d.wrote.step = Held::at(pos - 1, 1);
    Ok(())
}

/// A prompt unit's reader ([`Entering`]): the rows the unit's last layer leaves, copied into the
/// draft's prefill arena at the unit's row offset.
pub(crate) struct PromptTap<'a> {
    pub(crate) arena: &'a mut DeviceBuffer<f32>,
    pub(crate) hidden: usize,
    pub(crate) layers: usize,
    pub(crate) from: u32,
    pub(crate) runs: Vec<Range<u32>>,
}

impl Entering for PromptTap<'_> {
    fn group(&mut self, runs: &[Range<u32>]) -> Result<(), GpuError> {
        self.runs = runs.to_vec();
        Ok(())
    }

    fn enter(
        &mut self,
        stream: &CudaStream,
        l: usize,
        u: usize,
        rows: (&DeviceBuffer<f32>, usize),
    ) -> Result<(), GpuError> {
        if l != self.layers {
            return Ok(());
        }
        let n = self.hidden;
        let (buf, t) = rows;
        let at = self
            .runs
            .get(u)
            .map(|r| (r.start - self.from) as usize)
            .ok_or_else(|| shape(format!("unit {u} of a group of {}", self.runs.len())))?;
        span_mut(WHAT, self.arena, at * n, t * n)?
            .copy_from_device_async(&*span(WHAT, buf, 0, t * n)?, stream)
            .map_err(Into::into)
    }

    fn done(&mut self, _stream: &CudaStream) -> Result<(), GpuError> {
        Ok(())
    }
}

/// A prompt call's tap ([`prompt_with`]): called after each unit the call runs with the arena its rows'
/// final hidden rows sit in, the unit's first position and its rows.
pub type UnitSink<'a, R> =
    dyn FnMut(&mut GpuModel<FlintBody<R>>, FlintArena, u32, usize) -> Result<(), GpuError> + 'a;

/// The prompt of `ids` from the model's position as units of [`T_MAX`] rows, `sink` called after each
/// with the draft's prefill arena, the argmax after the last when `end` runs the head. Refused by name
/// before anything runs: a load without the layer, [`CallEnd::Close`] (the program takes no checkpoint
/// inside a unit) and a prompt the plain call refuses. A sink's error ends the call with it, its
/// positions taken back as a failed call's are.
pub fn prompt_with<R: RouterEntry>(
    m: &mut GpuModel<FlintBody<R>>,
    ids: &[u32],
    end: CallEnd,
    mut sink: Option<&mut UnitSink<'_, R>>,
) -> Result<Option<u32>, GpuError>
where
    FlintBody<R>: ChainBody + Rollback,
{
    drafted(m)?;
    if end == CallEnd::Close {
        return Err(shape(
            "a closing call: the program takes no checkpoint inside a prompt unit",
        ));
    }
    let from = m.pos();
    {
        let body = m.body(WHAT)?;
        runtime::prompt::call_end(from, ids.len(), body.ctx).map_err(|e| shape(e.to_string()))?;
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= body.embed.n_vocab()) {
            return Err(shape(format!(
                "prompt token {id} past vocabulary {}",
                body.embed.n_vocab()
            )));
        }
    }
    let run = prompt_pass(m, |m| {
        let mut argmax = None;
        let mut at = from;
        for chunk in ids.chunks(T_MAX) {
            let token = m.run_rows(chunk.len(), WHAT, |gpu, w, body, head, pos| {
                crate::prefill::prefill_draft(gpu, w, body, head, chunk, pos)
            })?;
            if let Some(sink) = sink.as_deref_mut() {
                sink(m, FlintArena::Prefill, at, chunk.len())?;
            }
            argmax = token;
            at += chunk.len() as u32;
        }
        Ok(argmax)
    });
    match run {
        Ok(token) => Ok(token.filter(|_| end.head())),
        Err(e) => Err(take_back(m, from, e, |_| Ok(from), "cut", shape)),
    }
}
