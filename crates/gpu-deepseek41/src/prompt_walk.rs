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
//! stacks into the one card sum ([`Routed::card_rows`]), and the host's sum leaves them out.
//!
//! Armed ([`BatchBufs::set_stats`]), the walk's leg keeps a card mark at each part boundary of a
//! routed layer, the parts' enqueue walls and the serves' host times and expert counts, which
//! [`BatchBufs::take_stats`] hands out as the common [`PromptStats`]; unarmed, nothing records.

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
use bloomery_gpu::prompt_timing::{Mark, PromptStats, PromptTiming, nanos};
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use model::embed::EmbedTable;
use runtime::layer::FfnKind;
use runtime::prompt::{call_batches, call_end, groups};
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
    /// ([`PromptStream::check_mode`]: `tiered`, whether the load hangs an expert tier, and `hybrid`'s
    /// residency machine among them), with the family's `facts`. `split` starts the expert stream on
    /// `gpu` at its first setting ([`PromptStream::start_split`], from `facts.split` and the router
    /// entry `R`'s counts), and then the ring pass's GEMM buffers of `ffn`'s card route for the
    /// ring's half ([`Routed::arm_ring`]); the stream then stays for the load's life, idle under
    /// `admit` and `off`. Refused by name: `split` with no [`SplitFacts`], the guards' refusals and
    /// the stream's own at its start (the card's room for the ring among them).
    pub fn set_xstream<R: RouterEntry>(
        &mut self,
        gpu: &Gpu,
        (hybrid, ffn): (&mut Hybrid<HostRun>, &mut Routed<R>),
        (mode, tiered): (XMode, bool),
        facts: StreamFacts,
    ) -> Result<(), GpuError> {
        self.stream.check_mode(hybrid, (mode, tiered))?;
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
                XCfg::new(shape, R::N_EXPERT, split.deadline, None)
            })?;
            if let Some(half) = started {
                ffn.arm_ring(gpu, half)?;
            }
        }
        self.stream.set_mode(mode);
        self.facts = Some(facts);
        Ok(())
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
    /// the last: the groups of [`groups`] run one after the other, and `at` is raised to the call's end.
    /// Refused by name before anything runs: for no id and past the stores' positions. With the taps
    /// armed, each layer's output at the call's last position is copied to its tap. Under host
    /// streaming the groups run inside the call's streaming ([`PromptStream::begin`] and
    /// [`PromptStream::end`]): before a decode history the call returns each layer to the set it
    /// started with, else its placement stays.
    pub fn prompt(&mut self, ids: &[u32], at: &mut u32) -> Result<bool, GpuError> {
        let from = *at;
        let to = call_end(from, ids.len(), self.ctx).map_err(|r| shape(r.to_string()))?;
        // Read once: the call keeps no row, so the history does not move until its end.
        let history = self.hybrid.call_history();
        let r = self
            .stream_begin(ids.len(), to, history)
            .and_then(|()| self.enqueue_call(ids, from, to));
        // The call's own error first: an end refused after a failed call is its echo.
        let ended =
            self.bufs
                .stream
                .end(&mut *self.hybrid, self.gpu.stream(), r.is_ok() && !history);
        r?;
        ended?;
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

    /// The call's units from `from` to `to` over `ids`, the groups of [`groups`] one after the other.
    fn enqueue_call(&mut self, ids: &[u32], from: u32, to: u32) -> Result<(), GpuError> {
        let units = call_batches(from, to, &[to], self.bufs.t_max);
        for gr in groups(units.len(), self.bufs.group) {
            let runs = &units[gr];
            let (Some(first), Some(end)) = (runs.first(), runs.last()) else {
                continue;
            };
            let seg = &ids[(first.start - from) as usize..(end.end - from) as usize];
            self.enqueue_group(seg, runs, end.end == to)?;
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
            for (&token, row) in ids[from..from + t]
                .iter()
                .zip(stage.rows[..t * n].chunks_exact_mut(n))
            {
                embed.row_into(token, row)?;
            }
            stage.set(run);
            span_mut(WHAT, &mut unit.x, 0, t * n)?.copy_from_host(stream, &stage.rows[..t * n])?;
            span_mut(WHAT, &mut unit.pos, 0, t)?.copy_from_host(stream, &stage.pos[..t])?;
            span_mut(WHAT, &mut unit.n_keys, 0, t)?.copy_from_host(stream, &stage.keys[..t])?;
            attention.stage(u, run.start, t)?;
            from += t;
        }
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
        Ok(())
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
}

impl<R: RouterEntry, A: BatchAttention> Walk<'_, R, A> {
    /// After the last layer: the last unit's last row into the head's input, and the head.
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
        head.enqueue(self.gpu, self.w)
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
    /// streamed experts' slots run over its half into the card sum ([`Routed::card_rows`]).
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
        let (ffn, streams) = (&*self.ffn, ring.is_some());
        self.stream
            .unit_read(port, (at, units), (land, streams), |l| ffn.card_has(l))?;
        port.mark(at, Mark::Shadow as usize)?;
        port.part_end(at, enq);
        Ok(())
    }

    /// The host sums joined with the shadow's, into the unit's `x`, and the layer's tap. A layer the
    /// expert tier holds experts of is joined first: the card sum over the stage card's and the
    /// tier's slots, once the host leg has landed the tier's rows, which also uploads the host sums.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.attention.ffn(at.layer) != FfnKind::Moe {
            return Ok(());
        }
        let enq = port.part_start();
        let t = self.ts[at.unit];
        let (gpu, l) = (self.gpu, at.layer);
        if self.ffn.tier_k(l) > 0 {
            let ffn = &mut *self.ffn;
            port.join_tiered(at, |_, trows| ffn.join_rows(gpu, (l, t), trows))?;
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
