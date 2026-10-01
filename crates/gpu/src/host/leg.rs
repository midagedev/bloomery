//! The host legs of a walk as the schedules see them: [`StepLeg`], the
//! [`Port`] of kind [`PortKind::Step`] over a tier's step port, and
//! [`BatchLeg`], the [`Port`] of kind [`PortKind::Batch`] over its batch port —
//! one of each for every architecture whose layer program hands its routed
//! experts to the host tier. A program reaches the tier through the leg
//! ([`StepLeg::hybrid`], [`BatchLeg::hybrid`]): a step walk for its go, its
//! wait and its [`HostTier::row_enqueued`]; an eager batch walk for its
//! download, its serve and its upload. A batch walk that wants its exchange
//! and its parts timed sets a [`LegTimer`] on its leg
//! ([`BatchLeg::set_timer`]) — the architecture's adapter; without one the
//! leg runs the exchange alone.
//! A batch walk whose layer's card sum reads the tier's rows (GLM) leaves
//! the upload to its join ([`BatchLeg::join_tiered`]): the serve does not
//! upload a tier layer, and the join settles the tiers, enqueues the sum over
//! their rows, then the upload.

use cuda_core::{CudaStream, DeviceBuffer};
use runtime::sched::{At, Overlap, Port, PortKind, Refused};

use super::batch::{BatchKey, ServeTimes, UnionCols};
use super::{HostExperts, HostTier};
use crate::GpuError;
use std::time::{Duration, Instant};

/// What a walk the leg refuses names.
const WHAT: &str = "host StepLeg (the host tier's step port)";
/// What a walk the batch leg refuses names.
const WHAT_BATCH: &str = "host BatchLeg (the host tier's batch port)";

/// A step walk's host leg: the stream its chain is enqueued on and the tier
/// that serves the chain's routed experts.
pub struct StepLeg<'a, H: HostExperts> {
    stream: &'a CudaStream,
    hybrid: &'a mut HostTier<H>,
}

impl<'a, H: HostExperts> StepLeg<'a, H> {
    /// The leg of a walk enqueued on `stream` whose host legs `hybrid` serves.
    pub fn new(stream: &'a CudaStream, hybrid: &'a mut HostTier<H>) -> StepLeg<'a, H> {
        StepLeg { stream, hybrid }
    }

    /// The tier, for a layer program's go, wait and enqueue notice.
    pub fn hybrid(&mut self) -> &mut HostTier<H> {
        self.hybrid
    }

    /// The stream the walk is enqueued on.
    pub fn stream(&self) -> &'a CudaStream {
        self.stream
    }
}

impl<H: HostExperts> Port for StepLeg<'_, H> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Step;

    /// Opens the tier's step port on the point's rows
    /// ([`HostTier::open_step`]): one row of one column is the step, two rows
    /// the pair pass; any other point is refused by name there.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        self.hybrid.open_step(self.stream, o.units, o.cols)
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT,
            detail: why.to_string(),
        }
    }
}

/// One serve of a batch walk's host exchange, timed, for the [`LegTimer`]
/// that asked for it: which layer over how many columns of the open walk, the
/// slots its host experts listed and their per-expert counts, the wall of
/// their union call, the serve's own host times and its whole wall (the
/// upload's enqueue in it).
pub struct ServeNote {
    /// The served layer.
    pub layer: usize,
    /// The open walk's columns.
    pub cols: usize,
    /// Host slots the union listed.
    pub slots: u64,
    /// The union's per-expert column counts.
    pub union_cols: UnionCols,
    /// The union call's wall.
    pub union_ns: u64,
    /// The serve's host times ([`HostTier::serve_key`]).
    pub times: ServeTimes,
    /// The serve's whole wall.
    pub serve_ns: u64,
}

/// The timed points of a batch walk, an architecture's optional adapter a
/// walk sets on its leg ([`BatchLeg::set_timer`]). The leg itself calls the
/// exchange's two points — [`LegTimer::upload_mark`] once the sums' upload is
/// enqueued, [`LegTimer::served`] with the serve's times and walls — and a
/// layer program reaches the walk's and its parts' points through the leg it
/// is handed ([`BatchLeg::begin_walk`], [`BatchLeg::mark`],
/// [`BatchLeg::part_end`], [`BatchLeg::end_walk`]), so no program borrows an
/// adapter of its own. With no timer set nothing records and nothing waits.
pub trait LegTimer {
    /// The mark of `layer`'s sums' upload on `stream`.
    fn upload_mark(&mut self, stream: &CudaStream, layer: usize) -> Result<(), GpuError>;

    /// One serve exchanged: `note`'s times and walls.
    fn served(&mut self, note: ServeNote);

    /// The mark of `layer`'s part `site` on `stream`, in the architecture's
    /// own layout of sites.
    fn mark(&mut self, stream: &CudaStream, layer: usize, site: usize) -> Result<(), GpuError>;

    /// `ns` of layer `layer`'s host enqueue wall, a part of it.
    fn note_part(&mut self, layer: usize, ns: u64);

    /// A walk begins; whatever the adapter kept of an earlier one is its own
    /// to clear.
    fn begin_walk(&mut self);

    /// A walk ends; the adapter may read its marks, which can wait for the
    /// stream's tail.
    fn end_walk(&mut self) -> Result<(), GpuError>;
}

/// `d` in whole nanoseconds, saturating.
fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// An eager batch walk's host leg: the stream its layers are enqueued on, the
/// tier whose batch port serves their routed experts, the card buffer each
/// layer's host sums land in, and the most columns a unit may hold (the
/// port's sets and the host's union were made for them). Unit `u` of layer
/// `l` over `cols` columns is the key `(l, u, 0 .. cols)` ([`BatchLeg::key`]):
/// the program's download names it, the walk's serve serves it and uploads
/// its sums.
pub struct BatchLeg<'a, H: HostExperts> {
    stream: &'a CudaStream,
    hybrid: &'a mut HostTier<H>,
    hsum: &'a mut DeviceBuffer<f32>,
    cap: usize,
    /// The open walk's columns.
    cols: usize,
    /// The walk's timing, when it set one ([`BatchLeg::set_timer`]).
    timer: Option<&'a mut dyn LegTimer>,
}

impl<'a, H: HostExperts> BatchLeg<'a, H> {
    /// The leg of a walk enqueued on `stream` whose host legs `hybrid`'s batch
    /// port serves into `hsum`, for units of up to `cap` columns
    /// ([`HostTier::prepare_batch`] made its sets for `cap`).
    pub fn new(
        stream: &'a CudaStream,
        hybrid: &'a mut HostTier<H>,
        hsum: &'a mut DeviceBuffer<f32>,
        cap: usize,
    ) -> BatchLeg<'a, H> {
        BatchLeg {
            stream,
            hybrid,
            hsum,
            cap,
            cols: 0,
            timer: None,
        }
    }

    /// The walk's timing, `timer`'s ([`LegTimer`]): its serves timed, its
    /// parts marked — or none, which runs the exchange alone.
    pub fn set_timer<T: LegTimer + 'a>(&mut self, timer: Option<&'a mut T>) {
        self.timer = timer.map(|t| t as &mut dyn LegTimer);
    }

    /// The tier, for a layer program's download.
    pub fn hybrid(&mut self) -> &mut HostTier<H> {
        self.hybrid
    }

    /// The stream the walk is enqueued on.
    pub fn stream(&self) -> &'a CudaStream {
        self.stream
    }

    /// The host sums the last upload sent, `cols` rows of the model width.
    pub fn hsum(&self) -> &DeviceBuffer<f32> {
        self.hsum
    }

    /// `at`'s exchange key over the open walk's columns.
    #[must_use]
    pub fn key(&self, at: At) -> BatchKey {
        BatchKey {
            layer: at.layer,
            set: at.unit,
            at: 0,
            u: self.cols,
        }
    }

    /// Mark `site` of `layer`'s part on the stream, in the architecture's own
    /// layout of sites ([`LegTimer::mark`]); nothing with no timer.
    pub fn mark(&mut self, layer: usize, site: usize) -> Result<(), GpuError> {
        match self.timer.as_deref_mut() {
            Some(t) => t.mark(self.stream, layer, site),
            None => Ok(()),
        }
    }

    /// The start of a part of layer `layer`'s host enqueue wall, which
    /// [`BatchLeg::part_end`] closes; `None` with no timer.
    pub fn part_start(&self) -> Option<Instant> {
        self.timer.as_deref().map(|_| Instant::now())
    }

    /// The wall of a part started at `t` added to layer `layer`'s enqueue
    /// ([`LegTimer::note_part`]); nothing with no timer or no start.
    pub fn part_end(&mut self, layer: usize, t: Option<Instant>) {
        if let (Some(t), Some(w)) = (t, self.timer.as_deref_mut()) {
            w.note_part(layer, nanos(t.elapsed()));
        }
    }

    /// A walk begins ([`LegTimer::begin_walk`]); nothing with no timer.
    pub fn begin_walk(&mut self) {
        if let Some(t) = self.timer.as_deref_mut() {
            t.begin_walk();
        }
    }

    /// A walk ends ([`LegTimer::end_walk`]); nothing with no timer.
    pub fn end_walk(&mut self) -> Result<(), GpuError> {
        match self.timer.as_deref_mut() {
            Some(t) => t.end_walk(),
            None => Ok(()),
        }
    }

    /// Whether `layer` is a tiered layer: one an expert tier holds experts
    /// of, whose serve leaves the upload to [`BatchLeg::join_tiered`].
    fn tiered(&self, layer: usize) -> Result<bool, GpuError> {
        if self.hybrid.tiers().is_empty() {
            return Ok(false);
        }
        Ok(self.hybrid.on_tier(layer)? > 0)
    }

    /// The join of `at`'s tiered layer after its serve: the expert tier's
    /// rows of the block, once the host has seen its service complete under
    /// the go deadline ([`HostTier::tier_rows_of`]: a tier late past it is
    /// lost), handed to `join` with the stream — the program enqueues the
    /// card sum that reads them — then the upload of the host sums into the
    /// leg's `hsum`, marked for the walk's timer when it has one. The rows
    /// stay readable on the stream until the set's next download. Refused by
    /// name for a layer no tier holds an expert of, and past the one tier
    /// card the host tier serves.
    pub fn join_tiered(
        &mut self,
        at: At,
        join: impl FnOnce(&CudaStream, &DeviceBuffer<f32>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        if !self.tiered(at.layer)? {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "a tiered join of layer {}, which no expert tier holds an expert of",
                    at.layer
                ),
            });
        }
        super::refuse_tier_count(WHAT_BATCH, self.hybrid.tiers().len())?;
        let key = self.key(at);
        let rows = self.hybrid.tier_rows_of(key, 0)?;
        join(self.stream, rows)?;
        self.hybrid.enqueue_upload(self.stream, self.hsum, key)?;
        match self.timer.as_deref_mut() {
            Some(t) => t.upload_mark(self.stream, at.layer),
            None => Ok(()),
        }
    }
}

impl<H: HostExperts> Port for BatchLeg<'_, H> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Batch;

    /// One unit of `1..=cap` columns, both sets of the port free
    /// ([`HostTier::begin_group`]: the walk is eager, and its caller leaves
    /// nothing of an earlier walk in flight); any other point is refused by
    /// name.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        if o.units != 1 || !(1..=self.cap).contains(&o.cols) {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "{} units of {} columns; the batch leg walks one unit of 1..={} columns",
                    o.units, o.cols, self.cap
                ),
            });
        }
        self.cols = o.cols;
        self.hybrid.begin_group()
    }

    /// Wait for `at`'s download, serve its host experts in one union call and
    /// enqueue the upload of their sums into the leg's `hsum`, the serve
    /// timed — its per-expert column counts ([`HostTier::served_union_cols`])
    /// read between the serve and the upload — and the upload marked for the
    /// walk's timer when it has one. A tiered layer's upload waits for its
    /// tier's rows: the program's back calls [`BatchLeg::join_tiered`].
    fn serve(&mut self, at: At) -> Result<(), GpuError> {
        let key = self.key(at);
        let tiered = self.tiered(at.layer)?;
        let Some(t) = self.timer.as_deref_mut() else {
            self.hybrid.serve_key(key)?;
            if tiered {
                return Ok(());
            }
            return self.hybrid.enqueue_upload(self.stream, self.hsum, key);
        };
        let t0 = Instant::now();
        let before = self.hybrid.stats();
        let times = self.hybrid.serve_key(key)?;
        let union_cols = self.hybrid.served_union_cols(key)?;
        if !tiered {
            self.hybrid.enqueue_upload(self.stream, self.hsum, key)?;
            t.upload_mark(self.stream, at.layer)?;
        }
        let after = self.hybrid.stats();
        t.served(ServeNote {
            layer: at.layer,
            cols: self.cols,
            slots: after
                .batch_host_slots
                .saturating_sub(before.batch_host_slots),
            union_cols,
            union_ns: after.batch_ns.saturating_sub(before.batch_ns),
            times,
            serve_ns: nanos(t0.elapsed()),
        });
        Ok(())
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT_BATCH,
            detail: why.to_string(),
        }
    }
}
