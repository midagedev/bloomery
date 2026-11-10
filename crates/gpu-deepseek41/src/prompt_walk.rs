//! The prompt batch's walker: the units of a group, the walk of the layers over them, the head and the
//! fault word. The family gives the walk its layers' attention ([`BatchAttention`]); the walk holds the
//! rest, which no family's architecture decides: the unit split, the host staging of the rows, the
//! routed block over the unit's slots, the logits of the last row and the fault word.
//!
//! A call is cut by [`runtime::prompt::call_batches`] into units of at most `t_max` positions, and its
//! units run in groups of `group` units ([`runtime::prompt::groups`]); a group is one walk
//! of the runtime's layer schedule at the point `(g, T, Batch)` ([`runtime::sched::walk`]) through the
//! host tier's batch port ([`BatchLeg`]). Per layer and unit:
//! - the front: the layer's attention (the family's), then its feed-forward: the routed block's norm
//!   and router over the unit's rows and the download of the rows, ids and weights to the host (a
//!   routed layer), or the dense block over the unit's rows (a dense layer);
//! - the shadow: the shared expert and the card experts, under the host leg (a routed layer);
//! - the back: the host sums (the walk's serve uploaded them) joined with the shadow's, into the unit's
//!   `x` (a routed layer).
//!
//! The head runs after the last layer of the last unit, for its last position alone. A group waits for
//! its launches and reads the fault word once at its end, so a fault ends the call before the next
//! group; the last group's word rides the head's readback.
//!
//! The walk owns the call's host streaming ([`PromptStream`], the `BLOOMERY_XSTREAM` lever), which a
//! family sets with its own facts as data ([`BatchBufs::set_xstream`], [`StreamFacts`]) and no stream
//! code of its own. Under `admit` a call of enough ids opens on the residency machine; each walk's
//! first unit picks at a routed layer's shadow, after the shared expert and before the card route, and
//! the walk's later units read its pick ([`PromptStream::unit_pick`]); the card route waits for the
//! pick's landings and runs over the moved map ([`Routed::card_rows`]), the serve under it; the
//! last unit's back owes the layer's return in a returning call's last walk, and the walk's end
//! returns its last layer ([`PromptStream::end_walk`]). The call's end keeps the placement, or returns
//! it before a decode history. Under `split` the family's [`SplitFacts`] start the expert stream at
//! the first setting; a layer's pick then streams the host experts the rule sends to the card into a
//! half of the stream's ring, the unit's card route runs their slots over the half beside the card's
//! stacks into the one card sum ([`Routed::card_rows`]), and the host's sum leaves them out. On a
//! layer the expert tier holds experts of the card route leaves the stream's union places for the
//! back, whose tier join sums the stage card's, the streamed and the tier's slots in slot order by
//! the count the stream hands it ([`PromptStream::unit_join`], [`Routed::join_rows`]).
//!
//! Armed ([`BatchBufs::set_stats`]), the walk's leg keeps a card mark at each part boundary of a
//! routed layer, the parts' enqueue walls and the serves' host times and expert counts, which
//! [`BatchBufs::take_stats`] hands out as the common [`PromptStats`]; unarmed, nothing records.
//!
//! A call may carry media ([`BatchBufs::set_media`]): the staging puts a span position's row, widened
//! exactly from bf16, in place of its token's embedding, and the walk after it is the text walk's. The
//! call takes the rows it was given whatever it returns, and ends with a named error when it staged
//! fewer rows than it held.

use std::ops::Range;
use std::time::{Duration, Instant};

use bloomery_gpu::head::Head;
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::host::costs::ExpertShape;
use bloomery_gpu::host::prompt_stream::PromptStream;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::read_fault;
use bloomery_gpu::host::xstream::{XCfg, XMode};
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::media::{CallRows, MediaSpan, widen_bf16};
use bloomery_gpu::prompt_timing::{Mark, PromptStats, PromptTiming, nanos};
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::embed::EmbedTable;
use runtime::layer::FfnKind;
use runtime::prompt::{call_batches, call_end, call_groups, heat_call};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::shared_routed::Routed;
use crate::span::{span, span_mut};

pub use runtime::prompt::{GROUP_LEVER, check_group, group_sets};

/// What this file's refusals name.
const WHAT: &str = "prompt batch";

/// The largest `BLOOMERY_PREFILL_GROUP`: the lever registry's.
pub const GROUP_MAX: usize = bloomery_levers::PREFILL_GROUP_MAX as usize;
const _: () = assert!(GROUP_MAX as u64 == bloomery_levers::PREFILL_GROUP_MAX);

/// A shape the walk refuses, by name.
fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// A unit's rows: the layer's input and its output, the positions and the live key counts.
pub struct UnitBuf {
    x: DeviceBuffer<f32>,
    x1: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
}

impl UnitBuf {
    /// The buffers of a unit of at most `rows` positions. Load-time only.
    fn new(gpu: &Gpu, embd: usize, rows: usize) -> Result<Self, GpuError> {
        let stream = gpu.stream();
        Ok(Self {
            x: DeviceBuffer::zeroed(stream, rows * embd)?,
            x1: DeviceBuffer::zeroed(stream, rows * embd)?,
            pos: DeviceBuffer::zeroed(stream, rows)?,
            n_keys: DeviceBuffer::zeroed(stream, rows)?,
        })
    }

    /// The device bytes of the buffers.
    fn bytes(&self) -> usize {
        self.x.num_bytes() + self.x1.num_bytes() + self.pos.num_bytes() + self.n_keys.num_bytes()
    }
}

/// The unit's rows as the family's attention borrows them: the layer's input, the residual after the
/// attention, the positions and the live key counts.
pub struct UnitRows<'a> {
    /// The unit's rows: the layer's input, and its output after the feed-forward.
    pub x: &'a DeviceBuffer<f32>,
    /// The residual after the attention: `x + attention(rms_norm(x))`.
    pub x1: &'a mut DeviceBuffer<f32>,
    /// The unit's positions, one a row.
    pub pos: &'a DeviceBuffer<u32>,
    /// The unit's live key counts, the position plus one, one a row.
    pub n_keys: &'a DeviceBuffer<u32>,
}

/// What a family gives the prompt batch's walk: the attention of every layer over a unit's rows, and the
/// kind of every layer's feed-forward. Both families implement it the same way: the attention over the
/// unit's rows, and the feed-forward's kind as the layer's description names it.
pub trait BatchAttention {
    /// The family's own staging of unit `u`: the records its attention reads beside the unit's rows
    /// (a GQA layer's word record, say). The walk calls it once a unit, before the layers run.
    fn stage(&mut self, _u: usize, _at: u32, _m: usize) -> Result<(), GpuError> {
        Ok(())
    }

    /// Layer `l`'s attention over unit `u`'s `m` rows: the unit's `x` in, the residual `x1` out.
    fn attend(&mut self, l: usize, u: usize, m: usize, rows: UnitRows<'_>) -> Result<(), GpuError>;

    /// Layer `l`'s feed-forward: the routed block (the shared expert and the card experts under the
    /// host leg) or a dense block of its own.
    fn ffn(&self, l: usize) -> FfnKind;
}

/// A unit's host staging: the embedding rows before their upload, the positions and the live key counts.
struct Stage {
    rows: Vec<f32>,
    pos: Vec<u32>,
    keys: Vec<u32>,
}

impl Stage {
    /// The staging of a unit of at most `rows` positions.
    fn new(embd: usize, rows: usize) -> Self {
        Self {
            rows: vec![0.0; rows * embd],
            pos: vec![0; rows],
            keys: vec![0; rows],
        }
    }

    /// The unit `run`'s positions and live key counts (the position plus one).
    fn set(&mut self, run: &Range<u32>) {
        for ((pos, keys), p) in self
            .pos
            .iter_mut()
            .zip(self.keys.iter_mut())
            .zip(run.clone())
        {
            *pos = p;
            *keys = p + 1;
        }
    }
}

/// The rows of a call's media spans ([`BatchBufs::set_media`]): each span's rows widened to f32 and
/// held by absolute position, the rows the staging puts in place of their positions' token embeddings.
/// The call that takes them counts the rows it staged.
struct CallMedia {
    /// The call's first position: the spans' positions count from it.
    first: u32,
    /// A row's width.
    n_embd: usize,
    /// The spans, ascending and disjoint: a span's positions and its rows, `n_embd` values a position.
    spans: Vec<(Range<u32>, Vec<f32>)>,
    /// The rows the spans hold.
    given: usize,
    /// The rows staged so far.
    staged: usize,
}

impl CallMedia {
    /// The rows of `spans` for a call from position `first`, each row of `n_embd` values widened.
    /// Refused by name by [`CallRows::new`]: a span that is not whole rows, spans that are not
    /// ascending and disjoint, positions past u32; and a row of no width.
    fn new(first: u32, n_embd: usize, spans: &[MediaSpan<'_>]) -> Result<CallMedia, GpuError> {
        if n_embd == 0 {
            return Err(shape("media rows of a width of 0 values"));
        }
        let rows = CallRows::new(first, n_embd, spans.iter().map(|s| (&s.at, s.rows)))
            .map_err(|refusal| shape(refusal.to_string()))?;
        let at = |i: usize| u32::try_from(i).ok().and_then(|i| first.checked_add(i));
        let mut held = Vec::with_capacity(spans.len());
        let mut given = 0;
        for s in spans {
            let (Some(start), Some(end)) = (at(s.at.start), at(s.at.end)) else {
                return Err(shape(format!(
                    "a media span at {:?} of a call from position {first}: its positions pass u32",
                    s.at
                )));
            };
            let mut widened = vec![0.0f32; s.at.len() * n_embd];
            for (p, out) in (start..end).zip(widened.chunks_exact_mut(n_embd)) {
                let row = rows
                    .row(p)
                    .ok_or_else(|| shape(format!("position {p} of a media span holds no row")))?;
                widen_bf16(row, out);
            }
            given += s.at.len();
            held.push((start..end, widened));
        }
        Ok(CallMedia {
            first,
            n_embd,
            spans: held,
            given,
            staged: 0,
        })
    }

    /// Position `p`'s row, counted as staged; `None` off the spans.
    fn take(&mut self, p: u32) -> Option<&[f32]> {
        let k = self.spans.partition_point(|(at, _)| at.end <= p);
        let (at, rows) = self.spans.get(k).filter(|(at, _)| at.start <= p)?;
        let i = usize::try_from(p - at.start).ok()? * self.n_embd;
        let row = rows.get(i..i + self.n_embd)?;
        self.staged += 1;
        Some(row)
    }

    /// Refused by name before the call runs: rows set for another first position than the call's
    /// `from`, a width other than the model's `embd`, or a span ending past the call's end `to`.
    fn check_call(&self, from: u32, to: u32, embd: usize) -> Result<(), GpuError> {
        if self.first != from {
            return Err(shape(format!(
                "media rows set for a call from position {}, and the call starts at {from}",
                self.first
            )));
        }
        if self.n_embd != embd {
            return Err(shape(format!(
                "media rows of {} values a position, and the model's rows are {embd}",
                self.n_embd
            )));
        }
        match self.spans.last() {
            Some((at, _)) if at.end > to => Err(shape(format!(
                "a media span ending at position {} in a call that ends at {to}",
                at.end
            ))),
            _ => Ok(()),
        }
    }

    /// Refused by name when the call staged fewer rows than the spans hold: a position the staging
    /// fed a token's embedding is a position that lost its media row.
    fn check_staged(&self) -> Result<(), GpuError> {
        if self.staged == self.given {
            return Ok(());
        }
        Err(shape(format!(
            "a call that staged {} of its {} media rows",
            self.staged, self.given
        )))
    }
}

/// A reader of the rows that enter each layer of a prompt call's units (a draft's taps), handed to
/// [`Batch::prompt_entering`]. The walk tells it each group of units before the walk
/// ([`Entering::group`]), shows it each unit's rows as they enter each layer and, past the last
/// layer, as that layer left them ([`Entering::enter`]), and tells it when the group's launches are
/// enqueued ([`Entering::done`]). It enqueues only on the stream it is given, and never changes a
/// row.
pub trait Entering {
    /// The group about to be walked holds the consecutive units `runs`, one position range each.
    fn group(&mut self, runs: &[Range<u32>]) -> Result<(), GpuError>;

    /// Unit `u`'s `rows` (the buffer and how many of its rows are the unit's) enter layer `l`; for
    /// `l` the layer count, they are the last layer's output. Called in stream order at the point
    /// the walk reaches the layer.
    fn enter(
        &mut self,
        stream: &CudaStream,
        l: usize,
        u: usize,
        rows: (&DeviceBuffer<f32>, usize),
    ) -> Result<(), GpuError>;

    /// The group's launches are enqueued and its fault word read: the rows it laid in may be read
    /// (a blocking read on `stream` waits for them).
    fn done(&mut self, stream: &CudaStream) -> Result<(), GpuError>;
}

/// The reader of a call that taps nothing.
struct NoEntering;

impl Entering for NoEntering {
    fn group(&mut self, _runs: &[Range<u32>]) -> Result<(), GpuError> {
        Ok(())
    }

    fn enter(
        &mut self,
        _stream: &CudaStream,
        _l: usize,
        _u: usize,
        _rows: (&DeviceBuffer<f32>, usize),
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn done(&mut self, _stream: &CudaStream) -> Result<(), GpuError> {
        Ok(())
    }
}

/// What a family hands the walk's host streaming ([`BatchBufs::set_xstream`]), its own facts as data.
#[derive(Clone, Copy, Debug)]
pub struct StreamFacts {
    /// The least count a pick admits on a load whose lane never probed: a call of fewer ids opens no
    /// streaming.
    pub floor: u32,
    /// The card tail rate the family's records priced for its routed kind [µs per card pick]; 0 where
    /// no record priced it (the walk's pick then balances without it).
    pub tail_us_per_card_pick: f64,
    /// The expert stream's facts under `split`; `None` for a family that hands none (`split` is then
    /// refused by name).
    pub split: Option<SplitFacts>,
}

/// What a family hands the walk's expert stream under `split` ([`BatchBufs::set_xstream`]), its own
/// facts as data: the bound on every host wait of the stream. The stream prices one routed expert on
/// this device and host from its widths (the card route's), the router entry's picks a column and the
/// walk's units, which each read a streamed expert's weights once.
#[derive(Clone, Copy, Debug)]
pub struct SplitFacts {
    pub deadline: Duration,
}

/// The prompt batch's buffers: the units of a group, the host staging of their rows and the host sums
/// the batch port uploads, and the calls' host streaming. Made once, so that a timed prompt allocates
/// nothing.
pub struct BatchBufs {
    group: usize,
    t_max: usize,
    units: Vec<UnitBuf>,
    stages: Vec<Stage>,
    hsum: DeviceBuffer<f32>,
    /// The calls' host streaming: off until the family sets it ([`BatchBufs::set_xstream`]).
    stream: PromptStream,
    /// The family's facts a streaming call opens with, once set.
    facts: Option<StreamFacts>,
    /// The walks' timing and the layers it was made for ([`BatchBufs::set_stats`]); unarmed, a walk
    /// records nothing and waits for nothing beyond its own serves.
    timing: Option<(PromptTiming, usize)>,
    /// The head's logit softcap, as the family's data ([`BatchBufs::set_softcap`]); `None` puts no
    /// launch between the projection and the argmax.
    softcap: Option<f32>,
    /// The card bytes the stream's ring buffers may take, the plan's ring term less its pass
    /// ([`BatchBufs::set_ring_cap`]); `None` leaves the ring what the card's margin has past the
    /// keep-free.
    ring_cap: Option<u64>,
    /// The media rows of the next call ([`BatchBufs::set_media`]); the call takes them.
    media: Option<CallMedia>,
}

impl BatchBufs {
    /// The buffers for a group of `group` units of at most `t_max` positions. Refused by name for a
    /// group outside the lever's range. Load-time only.
    pub fn new(gpu: &Gpu, embd: usize, t_max: usize, group: usize) -> Result<Self, GpuError> {
        check_group(group, GROUP_MAX).map_err(|r| GpuError::Shape {
            what: GROUP_LEVER,
            detail: r.to_string(),
        })?;
        let stream = gpu.stream();
        let sets = group_sets(group);
        let expected = model::placement::kernels::ffn::PromptWalk {
            hidden: embd as u64,
            sets: sets as u64,
        }
        .bytes(t_max as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| GpuError::shape("prompt batch buffers", "buffer geometry overflows"))?;
        let buffers = Self {
            group,
            t_max,
            units: (0..sets)
                .map(|_| UnitBuf::new(gpu, embd, t_max))
                .collect::<Result<_, _>>()?,
            stages: (0..sets).map(|_| Stage::new(embd, t_max)).collect(),
            hsum: DeviceBuffer::zeroed(stream, t_max * embd)?,
            stream: PromptStream::default(),
            facts: None,
            timing: None,
            softcap: None,
            ring_cap: None,
            media: None,
        };
        if buffers.bytes() != expected {
            return Err(GpuError::shape(
                "prompt batch buffers",
                format!("allocated {} bytes, planned {expected}", buffers.bytes()),
            ));
        }
        Ok(buffers)
    }

    /// What the next prompt calls move (`BLOOMERY_XSTREAM`), behind the guards every family shares
    /// ([`PromptStream::check_mode`]: `hybrid`'s residency machine among them), with the family's
    /// `facts`. `split` starts the expert stream on `gpu` at its first setting
    /// ([`PromptStream::start_split`], from `facts.split` and the router entry `R`'s counts), and
    /// then the ring pass's GEMM buffers of `ffn`'s card route for the ring's half
    /// ([`Routed::arm_ring`]); the stream then stays for the load's life, idle under `admit` and
    /// `off`. A load with an expert tier streams as one without (the module doc's tier join).
    /// Refused by name: `split` with no [`SplitFacts`], the guards' refusals and the stream's own at
    /// its start (the card's room for the ring among them).
    pub fn set_xstream<R: RouterEntry>(
        &mut self,
        gpu: &Gpu,
        (hybrid, ffn): (&mut Hybrid<HostRun>, &mut Routed<R>),
        mode: XMode,
        facts: StreamFacts,
    ) -> Result<(), GpuError> {
        self.stream.check_mode(hybrid, mode)?;
        if mode == XMode::Split {
            let split = facts.split.ok_or_else(|| {
                GpuError::shape(
                    bloomery_levers::XSTREAM,
                    "split on a family that hands the walk no expert stream facts (SplitFacts)",
                )
            })?;
            let started = PromptStream::start_split(hybrid, gpu, || {
                let (hidden, ff) = ffn.expert_widths();
                let shape = ExpertShape {
                    hidden,
                    ff,
                    top_k: R::N_USED,
                    units: self.group,
                };
                XCfg::new(shape, R::N_EXPERT, split.deadline, self.ring_cap)
            })?;
            if let Some(half) = started {
                ffn.arm_ring(gpu, half)?;
            }
        }
        self.stream.set_mode(mode);
        self.facts = Some(facts);
        Ok(())
    }

    /// The card bytes the expert stream's ring buffers may take (`XCfg::room_cap`): the plan's ring
    /// term less its pass, for a family whose plan counts a ring. `None` leaves the ring what the
    /// card's margin has past the keep-free.
    pub fn set_ring_cap(&mut self, cap: Option<u64>) {
        self.ring_cap = cap;
    }

    /// The head's logit softcap: `cap · tanh(logit / cap)` over the last row's logits between the
    /// projection and the argmax, the step's own launch (`enqueue_softcap_rows`), for a family whose
    /// file names a cap; 0 for none. Refused by name: a negative or non-finite cap.
    pub fn set_softcap(&mut self, cap: f32) -> Result<(), GpuError> {
        if !cap.is_finite() || cap < 0.0 {
            return Err(shape(format!(
                "a logit softcap of {cap}: 0 for none, else a finite bound above 0"
            )));
        }
        self.softcap = (cap > 0.0).then_some(cap);
        Ok(())
    }

    /// The media rows of the next prompt call, which starts at position `first`: `spans` hold their
    /// rows of `n_embd` bf16 values at positions of the call's ids. The call's staging puts a span
    /// position's row, widened exactly, in place of its token's embedding; the call takes the rows
    /// whatever it returns ([`Batch::prompt`]) and ends with a named error when it staged fewer than
    /// it held. No span sets nothing. Refused by name: rows still held from a pass that never reached
    /// the walk ([`BatchBufs::clear_media`]), and spans that are not whole rows, ascending and disjoint
    /// ([`CallRows::new`]).
    pub fn set_media(
        &mut self,
        first: u32,
        n_embd: usize,
        spans: &[MediaSpan<'_>],
    ) -> Result<(), GpuError> {
        if self.media.is_some() {
            return Err(shape(
                "media rows set beside rows a call has not taken: one call's rows at a time",
            ));
        }
        if !spans.is_empty() {
            self.media = Some(CallMedia::new(first, n_embd, spans)?);
        }
        Ok(())
    }

    /// Drops media rows no call took (a pass refused before its walk), so none stands in for a later
    /// call's embeddings.
    pub fn clear_media(&mut self) {
        self.media = None;
    }

    /// The calls' host streaming: its load line word ([`PromptStream::word`]).
    #[must_use]
    pub fn stream(&self) -> &PromptStream {
        &self.stream
    }

    /// The calls' host streaming to change between calls: the last call's records
    /// ([`PromptStream::take_pick_records`]) and the return's twin
    /// ([`PromptStream::set_return_at_end`]).
    pub fn stream_mut(&mut self) -> &mut PromptStream {
        &mut self.stream
    }

    /// The units a walk's group runs ([`BatchBufs::set_group`]).
    #[must_use]
    pub fn group(&self) -> usize {
        self.group
    }

    /// Set the group to `group` units a walk: the units it lacks are made here, between calls. Returns
    /// whether it made any.
    pub fn set_group(&mut self, gpu: &Gpu, embd: usize, group: usize) -> Result<bool, GpuError> {
        check_group(group, GROUP_MAX).map_err(|r| GpuError::Shape {
            what: GROUP_LEVER,
            detail: r.to_string(),
        })?;
        let sets = group_sets(group);
        let have = self.units.len();
        for _ in have..sets {
            self.units.push(UnitBuf::new(gpu, embd, self.t_max)?);
            self.stages.push(Stage::new(embd, self.t_max));
        }
        self.group = group;
        if let Some((t, layers)) = self.timing.as_mut()
            && t.units() < sets
        {
            *t = PromptTiming::new(gpu.context(), *layers, sets)?;
        }
        Ok(sets > have)
    }

    /// Arm (or disarm) the walks' timing ([`PromptTiming`]) for walks of `layers` layers: one event
    /// a part boundary a (layer, unit), the serves' host times and the call's fault wall, which
    /// [`BatchBufs::take_stats`] reads. Load-time allocation; unarmed, a walk records no event and
    /// waits for nothing beyond its own serves.
    pub fn set_stats(&mut self, gpu: &Gpu, layers: usize, on: bool) -> Result<(), GpuError> {
        self.timing = if on {
            let t = PromptTiming::new(gpu.context(), layers, self.units.len())?;
            Some((t, layers))
        } else {
            None
        };
        Ok(())
    }

    /// The prompt's walk stats, taken: the next prompt starts clean. `None` with the timing unarmed
    /// or no walk run.
    pub fn take_stats(&mut self) -> Option<PromptStats> {
        self.timing.as_mut().and_then(|(t, _)| t.take())
    }

    /// The device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.units.iter().map(UnitBuf::bytes).sum::<usize>() + self.hsum.num_bytes()
    }
}

/// The host tier's batch sets and the host union's slabs for units of `t_max` columns, made once so
/// that a timed prompt allocates nothing. Load-time only.
pub fn prepare_host(gpu: &Gpu, hybrid: &mut Hybrid<HostRun>, t_max: usize) -> Result<(), GpuError> {
    hybrid.prepare_batch(gpu.context(), t_max)?;
    hybrid.host_mut().prepare_union(t_max)
}

/// The parts the prompt batch's walk holds: the family's attention, the routed block, the host tier, the
/// head and the layers' taps.
pub struct Batch<'a, R: RouterEntry, A: BatchAttention> {
    /// The card.
    pub gpu: &'a Gpu,
    /// The resident weights.
    pub w: &'a Weights,
    /// The units of a group and the host sums.
    pub bufs: &'a mut BatchBufs,
    /// The family's attention.
    pub attention: &'a mut A,
    /// The routed block over the unit's slots.
    pub ffn: &'a mut Routed<R>,
    /// The host tier.
    pub hybrid: &'a mut Hybrid<HostRun>,
    /// The token embedding.
    pub embed: &'a EmbedTable,
    /// The head.
    pub head: &'a mut Head,
    /// Each layer's tap, when a gate armed them: the layer's output at the call's last position.
    pub taps: Option<&'a mut [DeviceBuffer<f32>]>,
    /// The model's width.
    pub embd: usize,
    /// The positions the stores hold.
    pub ctx: usize,
    /// The layers' count.
    pub layers: usize,
}

impl<R: RouterEntry, A: BatchAttention> Batch<'_, R, A> {
    /// Feed `ids` from position `at` in units of at most `t_max` positions and enqueue the head after
    /// the last: the groups of [`call_groups`] run one after the other, and `at` is raised to the call's end.
    /// Refused by name before anything runs: for no id and past the stores' positions. With the taps
    /// armed, each layer's output at the call's last position is copied to its tap. Under host
    /// streaming the groups run inside the call's streaming ([`PromptStream::begin`] and
    /// [`PromptStream::end`]): a call that continues a sequence holding decode heat returns each layer to
    /// the set it started with, a call at position 0 keeps its placement (`HostTier::call_history`). The media rows set for the call
    /// ([`BatchBufs::set_media`]) are taken here whatever the call returns; refused by name before anything runs: rows set for
    /// another first position or width than the call's, or a span past its end, and after it a call
    /// that staged fewer rows than it held.
    pub fn prompt(&mut self, ids: &[u32], at: &mut u32) -> Result<bool, GpuError> {
        self.prompt_entering(ids, at, &mut NoEntering)
    }

    /// [`Batch::prompt`] with `entering` shown the rows that enter each layer of each unit
    /// ([`Entering`]): the call's walks, launches and bits are the plain call's.
    pub fn prompt_entering(
        &mut self,
        ids: &[u32],
        at: &mut u32,
        entering: &mut dyn Entering,
    ) -> Result<bool, GpuError> {
        let from = *at;
        // The call's media rows are its own: taken here, so none stays for a later call.
        let mut media = self.bufs.media.take();
        let to = call_end(from, ids.len(), self.ctx).map_err(|r| shape(r.to_string()))?;
        if let Some(m) = &media {
            m.check_call(from, to, self.embd)?;
        }
        let history = self.hybrid.call_history(from);
        heat_call(
            self,
            history,
            |w, returns| w.stream_begin(ids.len(), to, returns),
            |w| w.enqueue_call(ids, (from, to), &mut media, &mut *entering),
            |w, kept, _| w.bufs.stream.end(&mut *w.hybrid, w.gpu.stream(), kept),
        )?;
        if let Some(m) = &media {
            m.check_staged()?;
        }
        *at = (*at).max(to);
        Ok(true)
    }

    /// Open the call's streaming over `n` ids ending at `end` ([`PromptStream::begin`]), returning to
    /// its start when `returns`, with the family's facts; nothing while the lever is off. Refused by
    /// name: a lever set with no family facts.
    fn stream_begin(&mut self, n: usize, end: u32, returns: bool) -> Result<(), GpuError> {
        let bufs = &mut *self.bufs;
        if bufs.stream.mode() == XMode::Off {
            return Ok(());
        }
        let facts = bufs.facts.ok_or(GpuError::State {
            what: WHAT,
            missing: "the family's stream facts (BatchBufs::set_xstream)",
        })?;
        bufs.stream.begin(
            &mut *self.hybrid,
            self.gpu.stream(),
            true,
            n,
            end as usize,
            returns,
            facts.floor,
            facts.tail_us_per_card_pick,
            self.layers,
        )
    }

    /// The call's units from `from` to `to` over `ids`, the groups of [`call_groups`] one after the other.
    fn enqueue_call(
        &mut self,
        ids: &[u32],
        (from, to): (u32, u32),
        media: &mut Option<CallMedia>,
        entering: &mut dyn Entering,
    ) -> Result<(), GpuError> {
        let units = call_batches(from, to, &[to], self.bufs.t_max);
        for gr in call_groups(&units, from, to, self.bufs.group) {
            let runs = &units[gr.units];
            self.enqueue_group(&ids[gr.ids], runs, gr.last, (media, &mut *entering))?;
        }
        Ok(())
    }

    /// One group of `ids` over the consecutive units `runs`: each unit's rows and positions on the card,
    /// one walk `(units, T, Batch)` over every layer, and the head after the last unit when `last`.
    fn enqueue_group(
        &mut self,
        ids: &[u32],
        runs: &[Range<u32>],
        last: bool,
        (media, entering): (&mut Option<CallMedia>, &mut dyn Entering),
    ) -> Result<(), GpuError> {
        let Batch {
            gpu,
            w,
            bufs,
            attention,
            ffn,
            hybrid,
            embed,
            head,
            taps,
            embd,
            layers,
            ..
        } = self;
        let (gpu, w) = (*gpu, *w);
        let bufs = &mut **bufs;
        let attention = &mut **attention;
        let ffn = &mut **ffn;
        let hybrid = &mut **hybrid;
        let embed = *embed;
        let head = &mut **head;
        let (embd, layers) = (*embd, *layers);
        let g = runs.len();
        let ts: Vec<usize> = runs.iter().map(|r| (r.end - r.start) as usize).collect();
        if g == 0
            || g > bufs.units.len()
            || ts.iter().any(|&t| t == 0 || t > bufs.t_max)
            || runs.windows(2).any(|w| w[0].end != w[1].start)
            || ts.iter().sum::<usize>() != ids.len()
        {
            return Err(shape(format!(
                "a group of units {runs:?} over {} ids on buffers of {} units: 1 to {} positions a \
                 unit, consecutive",
                ids.len(),
                bufs.units.len(),
                bufs.t_max
            )));
        }
        let stream = gpu.stream();
        let n = embd;
        let mut from = 0;
        for (u, ((unit, stage), (run, &t))) in bufs
            .units
            .iter_mut()
            .zip(bufs.stages.iter_mut())
            .zip(runs.iter().zip(&ts))
            .enumerate()
        {
            let rows = ids[from..from + t]
                .iter()
                .zip(stage.rows[..t * n].chunks_exact_mut(n));
            for (p, (&token, row)) in (run.start..).zip(rows) {
                match media.as_mut().and_then(|m| m.take(p)) {
                    Some(held) => row.copy_from_slice(held),
                    None => embed.row_into(token, row)?,
                }
            }
            stage.set(run);
            span_mut(WHAT, &mut unit.x, 0, t * n)?.copy_from_host(stream, &stage.rows[..t * n])?;
            span_mut(WHAT, &mut unit.pos, 0, t)?.copy_from_host(stream, &stage.pos[..t])?;
            span_mut(WHAT, &mut unit.n_keys, 0, t)?.copy_from_host(stream, &stage.keys[..t])?;
            attention.stage(u, run.start, t)?;
            from += t;
        }
        entering.group(runs)?;
        let mut units: Vec<Unit<'_>> = bufs
            .units
            .iter_mut()
            .map(|u| Unit {
                x: &mut u.x,
                x1: &mut u.x1,
                pos: &u.pos,
                n_keys: &u.n_keys,
            })
            .collect();
        units.truncate(g);
        let mut walk = Walk {
            gpu,
            w,
            ffn,
            attention,
            units,
            ts: &ts,
            embd: n,
            taps: taps.as_deref_mut(),
            stream: &mut bufs.stream,
            walk_end: runs[g - 1].end as usize,
            softcap: bufs.softcap,
            entering: &mut *entering,
        };
        let o = Overlap {
            units: g,
            cols: ts.iter().copied().max().unwrap_or(1),
            port: PortKind::Batch,
        };
        let mut leg = BatchLeg::new(stream, hybrid, &mut bufs.hsum, bufs.t_max);
        leg.set_unit_cols(&ts);
        leg.set_timer(bufs.timing.as_mut().map(|(t, _)| t));
        leg.begin_walk(g)?;
        sched::walk(o, layers, &mut leg, &mut walk)?;
        leg.end_walk()?;
        walk.entered_last(layers)?;
        if last {
            walk.head(head)?;
        }
        walk.stream.end_walk(leg.hybrid(), stream)?;
        walk.stream.walk_done();
        let t0 = bufs.timing.as_ref().map(|_| Instant::now());
        let read = read_fault(WHAT, gpu, hybrid, last);
        if let (Some(t0), Some((t, _))) = (t0, bufs.timing.as_mut()) {
            t.add_fault(nanos(t0.elapsed()));
        }
        read?;
        entering.done(stream)
    }
}

/// A unit's rows as the walk borrows them.
struct Unit<'a> {
    x: &'a mut DeviceBuffer<f32>,
    x1: &'a mut DeviceBuffer<f32>,
    pos: &'a DeviceBuffer<u32>,
    n_keys: &'a DeviceBuffer<u32>,
}

impl Unit<'_> {
    /// The unit's rows as the family's attention borrows them.
    fn rows(&mut self) -> UnitRows<'_> {
        UnitRows {
            x: &*self.x,
            x1: &mut *self.x1,
            pos: self.pos,
            n_keys: self.n_keys,
        }
    }
}

/// One group's walk: the family's attention, the routed block, each unit's own rows and the call's
/// host streaming, the group's last position its walk's end.
struct Walk<'a, R: RouterEntry, A: BatchAttention> {
    gpu: &'a Gpu,
    w: &'a Weights,
    ffn: &'a mut Routed<R>,
    attention: &'a mut A,
    units: Vec<Unit<'a>>,
    ts: &'a [usize],
    embd: usize,
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
    stream: &'a mut PromptStream,
    walk_end: usize,
    /// The head's logit softcap ([`BatchBufs::set_softcap`]).
    softcap: Option<f32>,
    /// The reader of the rows entering each layer ([`Entering`]).
    entering: &'a mut dyn Entering,
}

impl<R: RouterEntry, A: BatchAttention> Walk<'_, R, A> {
    /// After the last layer: the last unit's last row into the head's input, and the head, with the
    /// family's softcap between its projection and its argmax when the family set one.
    fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let n = self.embd;
        let (Some(unit), Some(&t)) = (self.units.last(), self.ts.last()) else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a unit of the group for the head",
            });
        };
        let row = span(WHAT, &*unit.x, (t - 1) * n, n)?;
        head.input_mut()
            .copy_from_device_async(&row, self.gpu.stream())?;
        let gpu = self.gpu;
        let Some(cap) = self.softcap else {
            return head.enqueue(gpu, self.w);
        };
        let n = head.n_vocab() * head.m();
        head.enqueue_with_epilogue(gpu, self.w, |logits| {
            gpu.elem()
                .enqueue_softcap_rows(gpu.stream(), logits, n, cap)
        })
    }

    /// The group's units' rows past the last layer, shown to the reader as the layer count's
    /// entering rows ([`Entering::enter`]).
    fn entered_last(&mut self, layers: usize) -> Result<(), GpuError> {
        let stream = self.gpu.stream();
        for (u, (unit, &t)) in self.units.iter().zip(self.ts).enumerate() {
            self.entering.enter(stream, layers, u, (&*unit.x, t))?;
        }
        Ok(())
    }

    /// Layer `l`'s output at the call's last position, copied into its tap: the group's last unit's last
    /// row, and a later group, if any, overwrites what this one copied.
    fn tap(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        if u + 1 != self.ts.len() {
            return Ok(());
        }
        let Some(taps) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        let tap = taps.get_mut(l).ok_or(GpuError::State {
            what: WHAT,
            missing: "the layer's tap",
        })?;
        let n = self.embd;
        let t = self.ts[u];
        let row = span(WHAT, &*self.units[u].x, (t - 1) * n, n)?;
        Ok(tap.copy_from_device_async(&row, self.gpu.stream())?)
    }
}

impl<'a, R: RouterEntry, A: BatchAttention> LayerProgram for Walk<'a, R, A> {
    type Port = BatchLeg<'a, HostRun>;

    /// A routed layer's feed-forward has a host leg; a dense layer's runs whole on the card.
    fn host_leg(&self, at: At) -> bool {
        self.attention.ffn(at.layer) == FfnKind::Moe
    }

    /// The attention, then the feed-forward up to its download. A routed layer's card marks
    /// (front, front's end, downloads) and enqueue wall go to the walk's timer when it has one.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, u) = (at.layer, at.unit);
        let t = self.ts[u];
        self.entering
            .enter(self.gpu.stream(), l, u, (&*self.units[u].x, t))?;
        let routed = self.attention.ffn(l) == FfnKind::Moe;
        let enq = if routed { port.part_start() } else { None };
        if routed {
            port.mark(at, Mark::Front as usize)?;
        }
        let rows = self.units[u].rows();
        self.attention.attend(l, u, t, rows)?;
        if routed {
            let gw = (self.gpu, self.w);
            self.ffn.front_rows(gw, (l, t), &*self.units[u].x1)?;
            port.mark(at, Mark::FrontEnd as usize)?;
            let key = port.key(at);
            self.ffn.download_rows(self.gpu, port.hybrid(), (key, t))?;
            port.mark(at, Mark::Down as usize)?;
            port.part_end(at, enq);
            Ok(())
        } else {
            // A dense layer's block runs whole on the card: its rows go into the unit's `x`.
            let unit = &mut self.units[u];
            let (x1, x) = (&*unit.x1, &mut *unit.x);
            self.ffn.dense_rows((self.gpu, self.w), (l, t), (x1, x))?;
            self.tap(l, u)
        }
    }

    /// The shared expert and the card experts under the host leg. Under host streaming the shared
    /// expert first, so the card runs it while the host counts and picks; then the layer's pick (the
    /// first unit's; a later unit reads the one it held, [`PromptStream::unit_pick`]), the card route
    /// behind the pick's landings over the moved map, and the unit's read of the layer's slots
    /// ([`PromptStream::unit_read`]). Under `split` the layer's ring view joins the card route: the
    /// streamed experts' slots run over its half into the card sum, or on a tier layer into the rows
    /// the back's join sums ([`Routed::card_rows`]), and the stream keeps their count for the back.
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.attention.ffn(at.layer) != FfnKind::Moe {
            return Ok(());
        }
        let enq = port.part_start();
        let gw = (self.gpu, self.w);
        let lm = (at.layer, self.ts[at.unit]);
        if !self.stream.streaming() {
            self.ffn.shadow_rows(gw, lm, &*self.units[at.unit].x1)?;
            port.mark(at, Mark::Shadow as usize)?;
            port.part_end(at, enq);
            return Ok(());
        }
        self.ffn.shared_rows(gw, lm, &*self.units[at.unit].x1)?;
        let (units, ffn) = (self.ts.len(), &*self.ffn);
        let (ring, land) =
            self.stream
                .unit_pick(port, at, (self.ts[0], units), self.walk_end, |l| {
                    ffn.card_has(l)
                })?;
        self.ffn.card_rows(gw, lm, (&land, ring.as_ref()))?;
        let (ffn, joined) = (&*self.ffn, ring.as_ref().map(|r| r.n));
        self.stream
            .unit_read(port, (at, units), (land, joined), |l| ffn.card_has(l))?;
        port.mark(at, Mark::Shadow as usize)?;
        port.part_end(at, enq);
        Ok(())
    }

    /// The host sums joined with the shadow's, into the unit's `x`, and the layer's tap. A layer the
    /// expert tier holds experts of is joined first: the card sum over the stage card's, the
    /// streamed (the stream's join of the unit, [`PromptStream::unit_join`]) and the tier's slots,
    /// once the host leg has landed the tier's rows, which also uploads the host sums. Any other
    /// layer's card route summed its streamed slots in the shadow.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.attention.ffn(at.layer) != FfnKind::Moe {
            return Ok(());
        }
        let enq = port.part_start();
        let t = self.ts[at.unit];
        let (gpu, l) = (self.gpu, at.layer);
        let streamed = self.stream.unit_join(at)?.unwrap_or(0);
        if self.ffn.tier_k(l) > 0 {
            let ffn = &mut *self.ffn;
            port.join_tiered(at, |_, trows| ffn.join_rows(gpu, (l, t), (trows, streamed)))?;
        }
        let unit = &mut self.units[at.unit];
        self.ffn
            .back_rows(gpu, (l, t), (port.hsum(), &mut *unit.x))?;
        port.mark(at, Mark::Back as usize)?;
        port.part_end(at, enq);
        let card = self.ffn.card_has(l);
        self.stream
            .unit_back((at, self.ts.len()), self.walk_end, card);
        self.tap(at.layer, at.unit)
    }
}

#[cfg(test)]
mod tests {
    use super::CallMedia;
    use bloomery_gpu::media::MediaSpan;

    /// The bf16 bits of a value whose low 16 bits are zero, so the widening gives it back.
    fn bf16(x: f32) -> u16 {
        u16::try_from(x.to_bits() >> 16).expect("the high half of a word")
    }

    /// Rows of width 3 whose value is `base` plus the value's index.
    fn rows(base: f32, n: usize) -> Vec<u16> {
        (0..n * 3).map(|i| bf16(base + i as f32)).collect()
    }

    /// The span rows of a call from position 10: positions 11 and 12, then 15.
    fn spans<'a>(first: &'a [u16], second: &'a [u16]) -> [MediaSpan<'a>; 2] {
        [
            MediaSpan {
                at: 1..3,
                rows: first,
            },
            MediaSpan {
                at: 5..6,
                rows: second,
            },
        ]
    }

    /// A position inside a span takes that span's row by its absolute position, a position off the
    /// spans takes none, and the rows staged are counted.
    #[test]
    fn a_span_position_takes_its_widened_row_by_absolute_position() {
        let (a, b) = (rows(1.0, 2), rows(100.0, 1));
        let mut m = CallMedia::new(10, 3, &spans(&a, &b)).expect("whole rows in order");
        assert_eq!(m.take(10), None);
        assert_eq!(m.take(11), Some(&[1.0f32, 2.0, 3.0][..]));
        assert_eq!(m.take(12), Some(&[4.0f32, 5.0, 6.0][..]));
        assert_eq!(m.take(13), None);
        assert_eq!(m.take(14), None);
        assert_eq!(m.take(15), Some(&[100.0f32, 101.0, 102.0][..]));
        assert_eq!(m.take(16), None);
        assert_eq!(m.staged, 3);
        assert!(m.check_staged().is_ok());
    }

    /// A call that fed one span position a token's embedding staged fewer rows than the spans hold,
    /// and ends with a name.
    #[test]
    fn a_call_that_stages_fewer_rows_than_it_holds_is_refused_by_name() {
        let (a, b) = (rows(1.0, 2), rows(100.0, 1));
        let mut m = CallMedia::new(10, 3, &spans(&a, &b)).expect("whole rows in order");
        for p in [11, 15] {
            assert!(m.take(p).is_some());
        }
        let refusal = m
            .check_staged()
            .expect_err("a row left unstaged")
            .to_string();
        assert!(
            refusal.contains("staged 2 of its 3 media rows"),
            "{refusal}"
        );
    }

    /// Rows set for another first position or width than the call's, and a span past the call's end,
    /// are refused before the walk runs.
    #[test]
    fn rows_that_do_not_fit_the_call_are_refused_by_name() {
        let (a, b) = (rows(1.0, 2), rows(100.0, 1));
        let m = CallMedia::new(10, 3, &spans(&a, &b)).expect("whole rows in order");
        assert!(m.check_call(10, 16, 3).is_ok());
        assert!(
            m.check_call(11, 17, 3)
                .is_err_and(|e| e.to_string().contains("from position 10"))
        );
        assert!(
            m.check_call(10, 16, 4)
                .is_err_and(|e| e.to_string().contains("3 values"))
        );
        assert!(
            m.check_call(10, 15, 3)
                .is_err_and(|e| e.to_string().contains("ending at position 16"))
        );
    }

    /// A span that is not whole rows, spans that overlap and a row of no width are refused by name.
    #[test]
    fn spans_that_are_not_whole_ordered_rows_are_refused_by_name() {
        let (a, b) = (rows(1.0, 2), rows(100.0, 1));
        let short = &a[..5];
        let ragged = [MediaSpan {
            at: 1..3,
            rows: short,
        }];
        assert!(CallMedia::new(10, 3, &ragged).is_err());
        let twice = [
            MediaSpan {
                at: 1..3,
                rows: &a[..],
            },
            MediaSpan {
                at: 2..3,
                rows: &b[..],
            },
        ];
        assert!(CallMedia::new(10, 3, &twice).is_err());
        assert!(CallMedia::new(10, 0, &spans(&a, &b)).is_err());
    }
}
