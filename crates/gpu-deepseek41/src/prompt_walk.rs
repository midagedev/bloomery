//! The prompt batch's walker: the units of a group, the walk of the layers over them, the head and the
//! fault word. The family gives the walk its layers' attention ([`BatchAttention`]); the walk holds the
//! rest, which no family's architecture decides: the unit split, the host staging of the rows, the
//! routed block over the unit's slots, the logits of the last row and the fault word.
//!
//! A call is cut by [`runtime::prompt::call_batches`] into units of at most `t_max` positions, and its
//! units run in groups of [`BatchBufs::group`] units ([`runtime::prompt::groups`]); a group is one walk
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

use std::ops::Range;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::read_fault;
use bloomery_gpu::hybrid::Hybrid;
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

/// The prompt batch's buffers: the units of a group, the host staging of their rows and the host sums
/// the batch port uploads. Made once, so that a timed prompt allocates nothing.
pub struct BatchBufs {
    group: usize,
    t_max: usize,
    units: Vec<UnitBuf>,
    stages: Vec<Stage>,
    hsum: DeviceBuffer<f32>,
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
        Ok(Self {
            group,
            t_max,
            units: (0..sets)
                .map(|_| UnitBuf::new(gpu, embd, t_max))
                .collect::<Result<_, _>>()?,
            stages: (0..sets).map(|_| Stage::new(embd, t_max)).collect(),
            hsum: DeviceBuffer::zeroed(stream, t_max * embd)?,
        })
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
        Ok(sets > have)
    }

    /// The units a group holds.
    #[must_use]
    pub fn group(&self) -> usize {
        self.group
    }

    /// The most positions a unit holds.
    #[must_use]
    pub fn t_max(&self) -> usize {
        self.t_max
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
    /// armed, each layer's output at the call's last position is copied to its tap.
    pub fn prompt(&mut self, ids: &[u32], at: &mut u32) -> Result<bool, GpuError> {
        let from = *at;
        let to = call_end(from, ids.len(), self.ctx).map_err(|r| shape(r.to_string()))?;
        let units = call_batches(from, to, &[to], self.bufs.t_max);
        for gr in groups(units.len(), self.bufs.group) {
            let runs = &units[gr];
            let (Some(first), Some(end)) = (runs.first(), runs.last()) else {
                continue;
            };
            let seg = &ids[(first.start - from) as usize..(end.end - from) as usize];
            self.enqueue_group(seg, runs, end.end == to)?;
        }
        *at = (*at).max(to);
        Ok(true)
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
        };
        let o = Overlap {
            units: g,
            cols: ts.iter().copied().max().unwrap_or(1),
            port: PortKind::Batch,
        };
        let mut leg = BatchLeg::new(stream, hybrid, &mut bufs.hsum, bufs.t_max);
        leg.set_unit_cols(&ts);
        sched::walk(o, layers, &mut leg, &mut walk)?;
        if last {
            walk.head(head)?;
        }
        read_fault(WHAT, gpu, hybrid, last)?;
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

/// One group's walk: the family's attention, the routed block and each unit's own rows.
struct Walk<'a, R: RouterEntry, A: BatchAttention> {
    gpu: &'a Gpu,
    w: &'a Weights,
    ffn: &'a mut Routed<R>,
    attention: &'a mut A,
    units: Vec<Unit<'a>>,
    ts: &'a [usize],
    embd: usize,
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
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

    /// The attention, then the feed-forward up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, u) = (at.layer, at.unit);
        let t = self.ts[u];
        let rows = self.units[u].rows();
        self.attention.attend(l, u, t, rows)?;
        if self.attention.ffn(l) == FfnKind::Moe {
            let gw = (self.gpu, self.w);
            self.ffn.front_rows(gw, (l, t), &*self.units[u].x1)?;
            let key = port.key(at);
            self.ffn.download(port.hybrid(), self.gpu.stream(), key)
        } else {
            // A dense layer's block runs whole on the card: its rows go into the unit's `x`.
            let unit = &mut self.units[u];
            let (x1, x) = (&*unit.x1, &mut *unit.x);
            self.ffn.dense_rows((self.gpu, self.w), (l, t), (x1, x))?;
            self.tap(l, u)
        }
    }

    /// The shared expert and the card experts under the host leg.
    fn shadow(&mut self, _port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.attention.ffn(at.layer) != FfnKind::Moe {
            return Ok(());
        }
        let gw = (self.gpu, self.w);
        let lm = (at.layer, self.ts[at.unit]);
        self.ffn.shadow_rows(gw, lm, &*self.units[at.unit].x1)
    }

    /// The host sums joined with the shadow's, into the unit's `x`, and the layer's tap.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.attention.ffn(at.layer) != FfnKind::Moe {
            return Ok(());
        }
        let t = self.ts[at.unit];
        let unit = &mut self.units[at.unit];
        self.ffn
            .back_rows(self.gpu, (at.layer, t), (port.hsum(), &mut *unit.x))?;
        self.tap(at.layer, at.unit)
    }
}
