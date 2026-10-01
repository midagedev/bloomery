//! A prompt's batch walks timed: [`PromptTiming`], the [`LegTimer`] a batch
//! walk sets on its leg ([`crate::host::BatchLeg::set_timer`]) — one for
//! every architecture whose prompt walks its batches through the host tier's
//! batch port (Qwen3.8's ubatch walk, GLM-5.3's group of batches). Armed, it
//! keeps one card event a [`Mark`] a (layer, unit), the serves' host times as
//! one row a layer-batch, the parts' enqueue walls and the prompt's host
//! walls outside the walks; unarmed (no timer set), nothing records and
//! nothing waits.

use crate::GpuError;
use crate::host::{LegTimer, ServeNote};
use cuda_core::{CudaContext, CudaEvent, CudaStream};
use runtime::sched::At;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What the timing refuses names.
const WHAT: &str = "prompt batch walk timing";

/// A card mark of a layer-batch's stream, in the order the walk enqueues
/// them: the front's first launch, the front's last launch before the route's
/// downloads, the downloads' end (the card route, on a card layer, and the
/// shared expert's launches follow), the shared expert's last launch, the
/// sums' upload, the gated sum.
#[derive(Clone, Copy)]
pub enum Mark {
    Front = 0,
    FrontEnd = 1,
    Down = 2,
    Shadow = 3,
    Upload = 4,
    Back = 5,
}

/// The marks one layer-batch's card time reads, one a part boundary.
const MARKS: usize = 6;

/// `d` in whole nanoseconds, saturating.
#[must_use]
pub fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// One layer-batch of a walk in progress: the serve's host time and the
/// union's per-expert column counts. The card times and the parts' enqueue
/// walls are read once the walk's last mark is waited for.
struct Served {
    unit: usize,
    layer: usize,
    cols: usize,
    slots: u64,
    experts: usize,
    m_max: usize,
    m_hot: usize,
    cols_hot: usize,
    m_sq: u64,
    wait_ns: u64,
    union_ns: u64,
    serve_ns: u64,
}

/// One layer-batch of a prompt's batch walks under the armed timing: its
/// columns, listed host slots and their per-expert counts (the listed
/// experts, the widest one's columns, the hot ones past an L2-sized
/// activation set and their columns, and `Σ m²` over the listed experts), its
/// serve's host time (the wait on the route's copies; the union call's wall,
/// which carries the routing scan and the plan build in front of it; the
/// serve's whole wall, the upload's enqueue in it), the host wall its parts
/// took to enqueue, and its card time by part (the front's launches, the
/// route's downloads, the shared expert under the union, the sums' upload,
/// the gated sum).
///
/// The fields are public for the record a caller renders.
pub struct PromptLb {
    /// The prompt's batch, from 0: the batches the walks before ran, plus
    /// the unit within its walk.
    pub b: u64,
    pub layer: usize,
    /// The unit's columns.
    pub cols: usize,
    /// Host slots the union listed.
    pub slots: u64,
    /// Distinct experts the layer-batch's columns listed.
    pub experts: usize,
    /// The most columns one listed expert took.
    pub m_max: usize,
    /// Listed experts past [`HOT_COLS`](crate::host::batch::HOT_COLS)
    /// columns, and the columns they took.
    pub m_hot: usize,
    pub cols_hot: usize,
    /// `Σ m²` over the listed experts.
    pub m_sq: u64,
    pub wait_ns: u64,
    pub union_ns: u64,
    pub serve_ns: u64,
    /// The layer's front, shared expert and gated sum as the host enqueued
    /// them.
    pub enqueue_ns: u64,
    pub front_ms: f64,
    pub down_ms: f64,
    pub shadow_ms: f64,
    pub upload_ms: f64,
    pub back_ms: f64,
}

/// A prompt's batch walks under the armed timing: the batches and
/// layer-batches that ran, the host prologue the walks' plans took, the
/// walks' whole wall, the host walls the architecture's call spent outside
/// the walks reading the fault word and waiting for its checkpoints (0 where
/// it notes none), and one row a layer-batch.
///
/// The fields are public for the record a caller renders.
pub struct PromptStats {
    /// Batches walked, a unit each.
    pub ubatches: u64,
    pub prologue_ns: u64,
    pub walk_ns: u64,
    pub fault_ns: u64,
    pub ckpt_ns: u64,
    pub rows: Vec<PromptLb>,
}

/// A prompt's batch-walk timing, allocated once a caller arms it: one event
/// a mark a (layer, unit), the walk in progress's rows, the walks that
/// completed, and the prompt's host walls.
pub struct PromptTiming {
    marks: Vec<CudaEvent>,
    /// The units a walk may hold, which the marks were made for.
    units: usize,
    /// The walk in progress's units.
    walk_units: usize,
    rows: Vec<Served>,
    done: Vec<PromptLb>,
    /// The host wall the parts took to enqueue, a (layer, unit) each.
    enqueue_ns: Vec<u64>,
    batches: u64,
    prologue_ns: u64,
    walk_ns: u64,
    fault_ns: u64,
    ckpt_ns: u64,
    /// The walk in progress's wall, from its first enqueue.
    walk_t0: Option<Instant>,
    /// The walk in progress's last recorded mark: one stream, so the last
    /// the card reaches.
    last: Option<usize>,
}

impl PromptTiming {
    /// Timing for walks of `layers` layers and up to `units` units over one
    /// stream's context. Load-time only; no unit is refused by name.
    pub fn new(
        ctx: &Arc<CudaContext>,
        layers: usize,
        units: usize,
    ) -> Result<PromptTiming, GpuError> {
        if units == 0 {
            return Err(GpuError::shape(WHAT, "timing for walks of no unit"));
        }
        // Timed events: `new_event(None)` makes a sync-only event, whose
        // `elapsed_ms` the driver refuses.
        let marks = (0..layers * units * MARKS)
            .map(|_| ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PromptTiming {
            marks,
            units,
            walk_units: 0,
            rows: Vec::new(),
            done: Vec::new(),
            enqueue_ns: vec![0; layers * units],
            batches: 0,
            prologue_ns: 0,
            walk_ns: 0,
            fault_ns: 0,
            ckpt_ns: 0,
            walk_t0: None,
            last: None,
        })
    }

    /// The units a walk may hold.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// A walk's plan's host wall added to the prompt's prologue.
    pub fn add_prologue(&mut self, ns: u64) {
        self.prologue_ns += ns;
    }

    /// A fault word's read, its host wall added to the prompt's.
    pub fn add_fault(&mut self, ns: u64) {
        self.fault_ns += ns;
    }

    /// A checkpoint's wait, its host wall added to the prompt's.
    pub fn add_ckpt(&mut self, ns: u64) {
        self.ckpt_ns += ns;
    }

    /// The prompt's stats so far, taken — every wall zeroed and the rows
    /// moved — or `None` before a walk completed.
    pub fn take(&mut self) -> Option<PromptStats> {
        if self.batches == 0 {
            return None;
        }
        Some(PromptStats {
            ubatches: std::mem::take(&mut self.batches),
            prologue_ns: std::mem::take(&mut self.prologue_ns),
            walk_ns: std::mem::take(&mut self.walk_ns),
            fault_ns: std::mem::take(&mut self.fault_ns),
            ckpt_ns: std::mem::take(&mut self.ckpt_ns),
            rows: std::mem::take(&mut self.done),
        })
    }

    /// The index of `at`'s (layer, unit), refused by name past the marks.
    fn slot(&self, at: At) -> Result<usize, GpuError> {
        let i = at.layer * self.units + at.unit;
        if at.unit >= self.units || i >= self.enqueue_ns.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a mark of layer {} unit {}: the timing was made for {} layers of {} units",
                    at.layer,
                    at.unit,
                    self.enqueue_ns.len() / self.units,
                    self.units
                ),
            ));
        }
        Ok(i)
    }

    /// Mark `site` of `at` on the stream, in [`Mark`]'s layout.
    fn record(&mut self, stream: &CudaStream, at: At, site: usize) -> Result<(), GpuError> {
        let i = self.slot(at)? * MARKS + site;
        let Some(mark) = self.marks.get(i).filter(|_| site < MARKS) else {
            return Err(GpuError::shape(
                WHAT,
                format!("mark site {site}; a layer-batch has {MARKS}"),
            ));
        };
        mark.record(stream)?;
        self.last = Some(i);
        Ok(())
    }

    /// The walk over: its card marks read into its rows — the last mark
    /// waited for first, so the reads see every event — and its rows filed
    /// under their batch, its wall added to the prompt's.
    fn close(&mut self) -> Result<(), GpuError> {
        let wall = self.walk_t0.take().map(|t| nanos(t.elapsed())).unwrap_or(0);
        if let Some(last) = self.last.take() {
            self.marks[last].synchronize()?;
        }
        let rows = std::mem::take(&mut self.rows);
        for row in rows {
            let at = At {
                unit: row.unit,
                layer: row.layer,
            };
            let i = self.slot(at)?;
            let Some(m) = self.marks.get(i * MARKS..).and_then(|m| m.get(..MARKS)) else {
                return Err(GpuError::state(
                    WHAT,
                    "a card mark for every layer of the walk",
                ));
            };
            let span = |a: Mark, b: Mark| -> Result<f64, GpuError> {
                Ok(f64::from(m[a as usize].elapsed_ms(&m[b as usize])?))
            };
            self.done.push(PromptLb {
                b: self.batches + row.unit as u64,
                layer: row.layer,
                cols: row.cols,
                slots: row.slots,
                experts: row.experts,
                m_max: row.m_max,
                m_hot: row.m_hot,
                cols_hot: row.cols_hot,
                m_sq: row.m_sq,
                wait_ns: row.wait_ns,
                union_ns: row.union_ns,
                serve_ns: row.serve_ns,
                enqueue_ns: self.enqueue_ns[i],
                front_ms: span(Mark::Front, Mark::FrontEnd)?,
                down_ms: span(Mark::FrontEnd, Mark::Down)?,
                shadow_ms: span(Mark::Down, Mark::Shadow)?,
                upload_ms: span(Mark::Shadow, Mark::Upload)?,
                back_ms: span(Mark::Upload, Mark::Back)?,
            });
        }
        self.enqueue_ns.fill(0);
        self.batches += self.walk_units as u64;
        self.walk_ns += wall;
        Ok(())
    }
}

/// The batch walk's timing behind the batch leg's [`LegTimer`]: the serve's
/// host times and walls as its rows, the parts' card marks and enqueue walls,
/// the walk's own wall.
impl LegTimer for PromptTiming {
    /// The sums' upload ([`Mark::Upload`]).
    fn upload_mark(&mut self, stream: &CudaStream, at: At) -> Result<(), GpuError> {
        self.record(stream, at, Mark::Upload as usize)
    }

    /// The serve's row, in serve order for [`PromptTiming::close`].
    fn served(&mut self, note: ServeNote) {
        self.rows.push(Served {
            unit: note.unit,
            layer: note.layer,
            cols: note.cols,
            slots: note.slots,
            experts: note.union_cols.experts,
            m_max: note.union_cols.m_max,
            m_hot: note.union_cols.m_hot,
            cols_hot: note.union_cols.cols_hot,
            m_sq: note.union_cols.m_sq,
            wait_ns: note.times.wait_ns,
            union_ns: note.union_ns,
            serve_ns: note.serve_ns,
        });
    }

    /// A part boundary's mark, in [`Mark`]'s layout.
    fn mark(&mut self, stream: &CudaStream, at: At, site: usize) -> Result<(), GpuError> {
        self.record(stream, at, site)
    }

    /// A part of `at`'s host enqueue wall.
    fn note_part(&mut self, at: At, ns: u64) {
        if let Ok(i) = self.slot(at) {
            self.enqueue_ns[i] += ns;
        }
    }

    /// A walk of `units` units begins: its rows taken back, its wall
    /// started. A walk that fails leaves no rows: the next one clears them.
    /// Refused by name past the units the marks were made for.
    fn begin_walk(&mut self, units: usize) -> Result<(), GpuError> {
        if units == 0 || units > self.units {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a walk of {units} units; the timing was made for 1..={}",
                    self.units
                ),
            ));
        }
        self.rows.clear();
        self.enqueue_ns.fill(0);
        self.walk_units = units;
        self.last = None;
        self.walk_t0 = Some(Instant::now());
        Ok(())
    }

    /// A walk ends: its card marks read and its rows filed. Blocking — the
    /// stream's tail, the last upload and gated sum, is waited for.
    fn end_walk(&mut self) -> Result<(), GpuError> {
        self.close()
    }
}
