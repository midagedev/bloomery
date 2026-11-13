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
//! A batch walk may hold several units (a prompt group's batches): the
//! port's two exchange sets are taken in turn by its downloads, serves and
//! uploads, a serve taking the oldest download not served, so the walk's
//! order — the next item's front ahead of this item's serve, or under
//! [`BatchOrder::Fronts`] each unit's front ahead of the serves — keeps at
//! most two layer-batches in flight, and the leg refuses by name a walk in
//! the fronts order of more than two units. The leg's one `hsum` holds one
//! item's sums at a time: item `x + 1`'s upload is enqueued after item `x`'s
//! back read them (`runtime::sched`'s batch orders put `x`'s back before
//! `x + 1`'s serve and, for a tiered layer, before `x + 1`'s join). Units
//! that sit side by side in one block of rows ([`BatchLeg::set_unit_rows`])
//! name their own rows: a unit's copies, serve and upload touch tokens
//! `at .. u` of the block and of `hsum` alone, so its sums never overlap
//! another unit's.
//! A batch walk whose layer's card sum reads the tier's rows (GLM) leaves
//! the upload to its join ([`BatchLeg::join_tiered`]): the serve does not
//! upload a tier layer, and the join settles the tiers, enqueues the sum over
//! their rows, then the upload.

use cuda_core::{CudaStream, DeviceBuffer};
use runtime::sched::{At, BatchOrder, Overlap, Port, PortKind, Refused};

use super::batch::{BatchKey, ServeTimes, UnionCols};
use super::{HostExperts, HostTier};
use crate::GpuError;
use std::ops::Range;
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
/// that asked for it: which layer of which unit over how many columns of the open walk, the
/// slots its host experts listed and their per-expert counts, the wall of
/// their union call, the serve's own host times and its whole wall (the
/// upload's enqueue in it).
pub struct ServeNote {
    /// The served layer.
    pub layer: usize,
    /// The served unit of the walk.
    pub unit: usize,
    /// The served unit's columns.
    pub cols: usize,
    /// Host slots the union listed.
    pub slots: u64,
    /// Host slots the columns listed whose expert the expert stream's
    /// exclusion set left out.
    pub excluded: u64,
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
    /// The mark of `at`'s sums' upload on `stream`.
    fn upload_mark(&mut self, stream: &CudaStream, at: At) -> Result<(), GpuError>;

    /// One serve exchanged: `note`'s times and walls.
    fn served(&mut self, note: ServeNote);

    /// The mark of `at`'s part `site` on `stream`, in the architecture's
    /// own layout of sites.
    fn mark(&mut self, stream: &CudaStream, at: At, site: usize) -> Result<(), GpuError>;

    /// `ns` of `at`'s host enqueue wall, a part of it.
    fn note_part(&mut self, at: At, ns: u64);

    /// A walk of `units` units begins; whatever the adapter kept of an
    /// earlier one is its own to clear.
    fn begin_walk(&mut self, units: usize) -> Result<(), GpuError>;

    /// A walk ends; the adapter may read its marks, which can wait for the
    /// stream's tail.
    fn end_walk(&mut self) -> Result<(), GpuError>;
}

/// `at`'s exchange key over the block rows `rows`: the unit the batch's label
/// within its group ([`BatchKey::set`]), its tokens `rows`.
fn batch_key(at: At, rows: Range<usize>) -> BatchKey {
    BatchKey {
        layer: at.layer,
        set: at.unit,
        at: rows.start,
        u: rows.end,
    }
}

/// Where each unit of a batch walk sits in the block its layers' routing and
/// sums are held in: the walk's columns, and a unit's own columns or its own
/// rows when the walk set them. A unit's rows are `0 .. cols` of its own
/// block unless the walk set rows, which place the units side by side in one
/// block.
#[derive(Clone, Copy, Default)]
struct UnitShape<'a> {
    /// The open walk's columns.
    cols: usize,
    /// Each unit's columns, when the walk set them.
    unit_cols: Option<&'a [usize]>,
    /// Each unit's rows of the block, when the walk set them.
    unit_rows: Option<&'a [Range<usize>]>,
}

impl UnitShape<'_> {
    /// Unit `unit`'s rows: its own when the walk set them, else `0 ..` its
    /// columns (its own when the walk set them, else the walk's).
    fn rows_of(&self, unit: usize) -> Range<usize> {
        if let Some(r) = self.unit_rows.and_then(|r| r.get(unit)) {
            return r.clone();
        }
        let cols = self
            .unit_cols
            .and_then(|c| c.get(unit).copied())
            .unwrap_or(self.cols);
        0..cols
    }

    /// `at`'s exchange key over its unit's rows.
    fn key(&self, at: At) -> BatchKey {
        batch_key(at, self.rows_of(at.unit))
    }
}

/// What [`BatchLeg`]'s [`Port::open_ordered`] refuses by name: no unit, a
/// walk's columns outside `1..=cap`, units' own columns (`shape.unit_cols`)
/// that are not one entry a unit, each in `1..=` the walk's columns, units'
/// own rows (`shape.unit_rows`) that are not one non-empty range a unit,
/// ascending and not overlapping, each no wider than the walk's columns and
/// ending within `cap` rows (the port's sets and `hsum` hold the block's
/// rows), both lists set, and the fronts order ([`BatchOrder::Fronts`]) for
/// more than two units (the port has two exchange sets).
fn refuse_open(
    o: Overlap,
    order: BatchOrder,
    cap: usize,
    shape: &UnitShape<'_>,
) -> Result<(), GpuError> {
    if o.units == 0 || !(1..=cap).contains(&o.cols) {
        return Err(GpuError::Shape {
            what: WHAT_BATCH,
            detail: format!(
                "{} units of {} columns; the batch leg walks one unit or more of 1..={cap} \
                 columns",
                o.units, o.cols
            ),
        });
    }
    if order == BatchOrder::Fronts && o.units > 2 {
        return Err(GpuError::Shape {
            what: WHAT_BATCH,
            detail: format!(
                "a walk of {} units in the {order:?} order; the port has two exchange sets, \
                 which hold the fronts of two units at most",
                o.units
            ),
        });
    }
    if shape.unit_cols.is_some() && shape.unit_rows.is_some() {
        return Err(GpuError::Shape {
            what: WHAT_BATCH,
            detail: "units' columns and units' rows both set; a walk names its units' shape \
                     once"
                .into(),
        });
    }
    if let Some(c) = shape.unit_cols
        && (c.len() != o.units || c.iter().any(|&n| !(1..=o.cols).contains(&n)))
    {
        return Err(GpuError::Shape {
            what: WHAT_BATCH,
            detail: format!(
                "units of {c:?} columns for a walk of {} units of up to {} columns",
                o.units, o.cols
            ),
        });
    }
    if let Some(r) = shape.unit_rows
        && (r.len() != o.units
            || r.iter()
                .any(|r| r.is_empty() || r.len() > o.cols || r.end > cap)
            || r.windows(2).any(|w| w[0].end > w[1].start))
    {
        return Err(GpuError::Shape {
            what: WHAT_BATCH,
            detail: format!(
                "units of rows {r:?} for a walk of {} units of up to {} columns over {cap} rows; \
                 the rows of a unit are one non-empty range each, ascending and apart",
                o.units, o.cols
            ),
        });
    }
    Ok(())
}

/// `d` in whole nanoseconds, saturating.
fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// An eager batch walk's host leg: the stream its layers are enqueued on, the
/// tier whose batch port serves their routed experts, the card buffer each
/// layer's host sums land in, and the most columns a unit may hold (the
/// port's sets and the host's union were made for them). Unit `u` of layer
/// `l` over its `cols` columns is the key `(l, u, 0 .. cols)`
/// ([`BatchLeg::key`]): the program's download names it, the walk's serve
/// serves it and uploads its sums. A unit's columns are the walk's unless
/// the walk set each unit's own ([`BatchLeg::set_unit_cols`]: a group's last
/// batch may be shorter). Units that share one block of rows name their own
/// rows instead ([`BatchLeg::set_unit_rows`]): the key `(l, u, r.start ..
/// r.end)`.
pub struct BatchLeg<'a, H: HostExperts> {
    stream: &'a CudaStream,
    hybrid: &'a mut HostTier<H>,
    hsum: &'a mut DeviceBuffer<f32>,
    cap: usize,
    /// The open walk's columns.
    cols: usize,
    /// Each unit's columns, when the walk set them.
    unit_cols: Option<&'a [usize]>,
    /// Each unit's rows of the block, when the walk set them: the leg's own
    /// copy, so a walk that cuts its rows as it starts can name them.
    unit_rows: Option<Vec<Range<usize>>>,
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
            unit_cols: None,
            unit_rows: None,
            timer: None,
        }
    }

    /// The walk's timing, `timer`'s ([`LegTimer`]): its serves timed, its
    /// parts marked — or none, which runs the exchange alone.
    pub fn set_timer<T: LegTimer + 'a>(&mut self, timer: Option<&'a mut T>) {
        self.timer = timer.map(|t| t as &mut dyn LegTimer);
    }

    /// Each unit's own columns, unit `u` the `u`-th: the next walk's
    /// [`Port::open_ordered`] refuses by name a list that is not one entry a
    /// unit, each in `1..=` the walk's columns.
    pub fn set_unit_cols(&mut self, cols: &'a [usize]) {
        self.unit_cols = Some(cols);
    }

    /// Each unit's own rows of the block the walk's routing and sums are held
    /// in, unit `u` the `u`-th: unit `u`'s key names tokens `rows[u]`, so its
    /// download, serve and upload copy those rows of the block and of the
    /// leg's `hsum` and no other. The next walk's [`Port::open_ordered`]
    /// refuses by name a list that is not one non-empty range a unit,
    /// ascending and not overlapping, each no wider than the walk's columns
    /// and ending within the leg's `cap` rows, and a walk that set the units'
    /// columns as well. The leg keeps its own copy of the list.
    pub fn set_unit_rows(&mut self, rows: &[Range<usize>]) {
        self.unit_rows = Some(rows.to_vec());
    }

    /// The units' columns or rows the walk set, and the open walk's columns.
    fn shape(&self) -> UnitShape<'_> {
        UnitShape {
            cols: self.cols,
            unit_cols: self.unit_cols,
            unit_rows: self.unit_rows.as_deref(),
        }
    }

    /// The tier, for a layer program's download.
    pub fn hybrid(&mut self) -> &mut HostTier<H> {
        self.hybrid
    }

    /// The stream the walk is enqueued on.
    pub fn stream(&self) -> &'a CudaStream {
        self.stream
    }

    /// The host sums the last upload sent: its unit's rows of the model
    /// width, at the unit's own row offsets in the buffer.
    pub fn hsum(&self) -> &DeviceBuffer<f32> {
        self.hsum
    }

    /// `at`'s exchange key over its unit's rows: the unit is the batch's
    /// label within its group, and the port picks its exchange set in turn.
    #[must_use]
    pub fn key(&self, at: At) -> BatchKey {
        self.shape().key(at)
    }

    /// Mark `site` of `at`'s part on the stream, in the architecture's own
    /// layout of sites ([`LegTimer::mark`]); nothing with no timer.
    pub fn mark(&mut self, at: At, site: usize) -> Result<(), GpuError> {
        match self.timer.as_deref_mut() {
            Some(t) => t.mark(self.stream, at, site),
            None => Ok(()),
        }
    }

    /// The start of a part of a layer-batch's host enqueue wall, which
    /// [`BatchLeg::part_end`] closes; `None` with no timer.
    pub fn part_start(&self) -> Option<Instant> {
        self.timer.as_deref().map(|_| Instant::now())
    }

    /// The wall of a part started at `t` added to `at`'s enqueue
    /// ([`LegTimer::note_part`]); nothing with no timer or no start.
    pub fn part_end(&mut self, at: At, t: Option<Instant>) {
        if let (Some(t), Some(w)) = (t, self.timer.as_deref_mut()) {
            w.note_part(at, nanos(t.elapsed()));
        }
    }

    /// A walk of `units` units begins ([`LegTimer::begin_walk`]); nothing
    /// with no timer.
    pub fn begin_walk(&mut self, units: usize) -> Result<(), GpuError> {
        match self.timer.as_deref_mut() {
            Some(t) => t.begin_walk(units),
            None => Ok(()),
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
    /// of for the batch ([`HostTier::batch_slots`]: a tier on a node with no
    /// batch leg holds none, the host serves its experts), whose serve leaves
    /// the upload to [`BatchLeg::join_tiered`].
    fn tiered(&self, layer: usize) -> Result<bool, GpuError> {
        if self.hybrid.tiers().is_empty() {
            return Ok(false);
        }
        Ok(self.hybrid.batch_slots()?.on_tier(layer)? > 0)
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
            Some(t) => t.upload_mark(self.stream, at),
            None => Ok(()),
        }
    }
}

impl<H: HostExperts> Port for BatchLeg<'_, H> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Batch;

    /// [`Port::open_ordered`] in [`BatchOrder::Ahead`]'s order.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        self.open_ordered(o, BatchOrder::Ahead)
    }

    /// One unit or more of `1..=cap` columns — each unit's own columns or its
    /// own rows of the block, when the walk set them — in either batch order,
    /// the fronts order for two units at most, both sets of the port free
    /// ([`HostTier::begin_group`], once a walk: the walk is eager, and its
    /// caller leaves nothing of an earlier walk in flight); any other point
    /// is refused by name.
    fn open_ordered(&mut self, o: Overlap, order: BatchOrder) -> Result<(), GpuError> {
        refuse_open(o, order, self.cap, &self.shape())?;
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
            t.upload_mark(self.stream, at)?;
        }
        let after = self.hybrid.stats();
        t.served(ServeNote {
            layer: at.layer,
            unit: at.unit,
            cols: key.u - key.at,
            slots: after
                .batch_host_slots
                .saturating_sub(before.batch_host_slots),
            excluded: after
                .batch_excluded_slots
                .saturating_sub(before.batch_excluded_slots),
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

#[cfg(test)]
mod tests {
    use super::super::batch::BatchPort;
    use super::*;
    use runtime::sched::{self, LayerProgram};

    /// The batch port's exchange under a walk's order, keyed as the leg keys
    /// it ([`UnitShape`], the leg's own keying) and opened as the leg opens
    /// ([`refuse_open`]): each front downloads its item, each serve serves and
    /// uploads it, each back reads the sums the upload sent.
    struct Exchange<'a> {
        stream: &'a CudaStream,
        port: &'a mut BatchPort,
        hsum: &'a mut DeviceBuffer<f32>,
        cap: usize,
        n_embd: usize,
        shape: UnitShape<'a>,
        /// Each serve's activations as the port handed them: the item, and
        /// how many values.
        served: Vec<(At, usize)>,
    }

    impl Port for Exchange<'_> {
        type Error = GpuError;
        const KIND: PortKind = PortKind::Batch;

        fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
            self.open_ordered(o, BatchOrder::Ahead)
        }

        fn open_ordered(&mut self, o: Overlap, order: BatchOrder) -> Result<(), GpuError> {
            refuse_open(o, order, self.cap, &self.shape)?;
            self.shape.cols = o.cols;
            self.port.begin();
            Ok(())
        }

        fn serve(&mut self, at: At) -> Result<(), GpuError> {
            let key = self.shape.key(at);
            let mut values = 0;
            self.port.serve(key, |layer, x, _, _, sums| {
                values = x.data().len();
                // Each sum is the row's own value plus the layer: the back
                // reads which item's sums it got.
                for (s, v) in sums.iter_mut().zip(x.data()) {
                    *s = v + layer as f32;
                }
                Ok(())
            })?;
            self.served.push((at, values));
            self.port.upload(self.stream, self.hsum, key)
        }

        fn refused(why: Refused) -> GpuError {
            GpuError::shape(WHAT_BATCH, why.to_string())
        }
    }

    /// The blocks the units' rows are read from (unit `u` reads
    /// `x[x_of[u]]`, whose rows hold `unit + 1` everywhere), and the sums
    /// each back read, from its unit's rows of `hsum`. `FRONTS` walks in
    /// [`BatchOrder::Fronts`]'s order.
    struct Items<'a, const FRONTS: bool> {
        x: Vec<DeviceBuffer<f32>>,
        x_of: Vec<usize>,
        w: DeviceBuffer<f32>,
        ids: DeviceBuffer<u32>,
        backs: Vec<(At, Vec<f32>)>,
        port: std::marker::PhantomData<Exchange<'a>>,
    }

    impl<'a, const FRONTS: bool> LayerProgram for Items<'a, FRONTS> {
        type Port = Exchange<'a>;
        const BATCH_ORDER: BatchOrder = if FRONTS {
            BatchOrder::Fronts
        } else {
            BatchOrder::Ahead
        };

        fn front(&mut self, port: &mut Exchange<'a>, at: At) -> Result<(), GpuError> {
            let key = port.shape.key(at);
            let x = &self.x[self.x_of[at.unit]];
            port.port
                .download(port.stream, [x, &self.w], &self.ids, key)
        }

        fn back(&mut self, port: &mut Exchange<'a>, at: At) -> Result<(), GpuError> {
            let (n, rows) = (port.n_embd, port.shape.rows_of(at.unit));
            let got = port.hsum.to_host_vec(port.stream)?;
            self.backs
                .push((at, got[rows.start * n..rows.end * n].to_vec()));
            Ok(())
        }
    }

    fn walk(units: usize, cols: usize) -> Overlap {
        Overlap {
            units,
            cols,
            port: PortKind::Batch,
        }
    }

    /// The leg opens a walk of one unit or more, each unit's own columns
    /// in `1..=` the walk's, and refuses by name no unit, columns outside
    /// `1..=cap`, and a units' list of another length or with an entry
    /// outside the walk's columns.
    #[test]
    fn open_takes_one_unit_or_more_and_refuses_the_rest() {
        let ahead = BatchOrder::Ahead;
        let none = UnitShape::default();
        for units in [1, 2, 3, 8] {
            refuse_open(walk(units, 512), ahead, 512, &none)
                .unwrap_or_else(|e| panic!("a walk of {units} units: {e}"));
        }
        let cols = |c| UnitShape {
            unit_cols: Some(c),
            ..UnitShape::default()
        };
        refuse_open(walk(3, 512), ahead, 512, &cols(&[512, 512, 1]))
            .expect("three units, the last of one column");
        let refused = [
            (walk(0, 512), none, "0 units of 512 columns"),
            (walk(2, 0), none, "2 units of 0 columns"),
            (walk(2, 513), none, "2 units of 513 columns"),
            (
                walk(3, 512),
                cols(&[512, 512]),
                "units of [512, 512] columns",
            ),
            (walk(2, 512), cols(&[512, 0]), "units of [512, 0] columns"),
            (
                walk(2, 256),
                cols(&[256, 512]),
                "units of [256, 512] columns",
            ),
        ];
        for (o, shape, want) in refused {
            let e = refuse_open(o, ahead, 512, &shape).expect_err(want);
            assert!(e.to_string().contains(want), "{want}: {e}");
        }
    }

    /// Units that share one block name their rows: the leg opens a walk of
    /// units whose rows are one non-empty range each, ascending, apart,
    /// within the walk's columns and the leg's rows, and refuses by name a
    /// list of another length, an empty range, a range past the leg's rows,
    /// ranges that overlap or descend, a unit wider than the walk's columns,
    /// and a walk that set the units' columns too.
    #[test]
    fn open_takes_unit_rows_and_refuses_the_rest() {
        let ahead = BatchOrder::Ahead;
        let rows = |r| UnitShape {
            unit_rows: Some(r),
            ..UnitShape::default()
        };
        refuse_open(walk(2, 256), ahead, 512, &rows(&[0..256, 256..512]))
            .expect("two halves of the block");
        refuse_open(walk(2, 3), ahead, 5, &rows(&[0..3, 3..5])).expect("an odd block");
        let whole = 0..512;
        refuse_open(
            walk(1, 512),
            ahead,
            512,
            &rows(std::slice::from_ref(&whole)),
        )
        .expect("one unit of the block");
        let both = UnitShape {
            unit_cols: Some(&[256, 256]),
            unit_rows: Some(&[0..256, 256..512]),
            ..UnitShape::default()
        };
        let refused = [
            (walk(3, 256), rows(&[0..256, 256..512]), "units of rows"),
            (walk(2, 256), rows(&[0..256, 256..256]), "units of rows"),
            (walk(2, 256), rows(&[0..256, 300..513]), "units of rows"),
            (walk(2, 256), rows(&[0..256, 255..400]), "units of rows"),
            (walk(2, 256), rows(&[256..512, 0..256]), "units of rows"),
            (walk(2, 200), rows(&[0..256, 256..400]), "units of rows"),
            (
                walk(2, 256),
                both,
                "units' columns and units' rows both set",
            ),
        ];
        for (o, shape, want) in refused {
            let e = refuse_open(o, ahead, 512, &shape).expect_err(want);
            assert!(e.to_string().contains(want), "{want}: {e}");
        }
    }

    /// The fronts order holds the fronts of its units in flight beside each
    /// other: one unit or two fit the port's two sets, three are refused by
    /// name; the other order takes any group.
    #[test]
    fn the_fronts_order_is_refused_past_two_units() {
        let none = UnitShape::default();
        for units in [1, 2] {
            refuse_open(walk(units, 512), BatchOrder::Fronts, 512, &none)
                .unwrap_or_else(|e| panic!("{units} units in the fronts order: {e}"));
        }
        for units in [3, 8] {
            let e = refuse_open(walk(units, 512), BatchOrder::Fronts, 512, &none)
                .expect_err("the fronts order of more than two units");
            let want = format!("{units} units in the Fronts order");
            assert!(e.to_string().contains(&want), "{want}: {e}");
            refuse_open(walk(units, 512), BatchOrder::Ahead, 512, &none)
                .unwrap_or_else(|e| panic!("{units} units ahead: {e}"));
        }
    }

    /// A unit's key is its own rows of the block: the walk's columns from row
    /// 0 unless the walk set the unit's columns (from row 0) or its rows (at
    /// their offsets), and the unit and layer are the item's.
    #[test]
    fn a_units_key_is_its_own_rows() {
        let at = |unit, layer| At { unit, layer };
        let key = |layer, set, at, u| BatchKey { layer, set, at, u };
        let plain = UnitShape {
            cols: 4,
            ..UnitShape::default()
        };
        assert_eq!(plain.key(at(1, 3)), key(3, 1, 0, 4));
        let narrow = UnitShape {
            cols: 4,
            unit_cols: Some(&[4, 3]),
            ..UnitShape::default()
        };
        assert_eq!(narrow.key(at(0, 0)), key(0, 0, 0, 4));
        assert_eq!(narrow.key(at(1, 2)), key(2, 1, 0, 3));
        let split = UnitShape {
            cols: 3,
            unit_rows: Some(&[0..3, 3..5]),
            ..UnitShape::default()
        };
        assert_eq!(split.key(at(0, 7)), key(7, 0, 0, 3));
        assert_eq!(split.key(at(1, 7)), key(7, 1, 3, 5));
    }

    /// A three-unit group walked across a layer boundary — unit 2 of layer
    /// 0 then unit 0 of layer 1, which an odd group puts in one set in turn
    /// — through the port's two sets: every serve takes the oldest download
    /// not served (the port refuses any other by name), no download meets a
    /// set still held, and each back reads its own item's sums. A third
    /// download while two sets are in flight is refused by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_three_units_across_a_layer_take_the_sets_in_turn() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let (n_embd, cap) = (2, 4);
        let mut port = BatchPort::new(&ctx, n_embd, 1, cap).expect("a batch port");
        let mut hsum = DeviceBuffer::<f32>::zeroed(&stream, cap * n_embd).expect("hsum");
        let x = |u: usize| {
            DeviceBuffer::from_host(&stream, &vec![(u + 1) as f32; cap * n_embd]).expect("x")
        };
        let cols = [4, 4, 3];
        let (backs, served, items_x, w, ids) = {
            let mut items = Items::<false> {
                x: vec![x(0), x(1), x(2)],
                x_of: vec![0, 1, 2],
                w: DeviceBuffer::from_host(&stream, &[1.0f32; 4]).expect("w"),
                ids: DeviceBuffer::from_host(&stream, &[0u32; 4]).expect("ids"),
                backs: Vec::new(),
                port: std::marker::PhantomData,
            };
            let mut ex = Exchange {
                stream: &stream,
                port: &mut port,
                hsum: &mut hsum,
                cap,
                n_embd,
                shape: UnitShape {
                    unit_cols: Some(&cols),
                    ..UnitShape::default()
                },
                served: Vec::new(),
            };
            sched::walk(walk(3, 4), 2, &mut ex, &mut items)
                .expect("a three-unit walk over two layers");
            (items.backs, ex.served, items.x, items.w, items.ids)
        };
        let want: Vec<(At, Vec<f32>)> = (0..2)
            .flat_map(|layer| (0..3).map(move |unit| At { unit, layer }))
            .map(|at| {
                let v = (at.unit + 1 + at.layer) as f32;
                (at, vec![v; n_embd * cols[at.unit]])
            })
            .collect();
        assert_eq!(backs, want);
        assert!(
            served
                .iter()
                .all(|&(at, values)| values == n_embd * cols[at.unit]),
            "a serve's activations are its unit's columns: {served:?}"
        );
        let key = |unit| BatchKey {
            layer: 0,
            set: unit,
            at: 0,
            u: 4,
        };
        port.begin();
        for (unit, x) in items_x.iter().take(2).enumerate() {
            port.download(&stream, [x, &w], &ids, key(unit))
                .expect("a download into a free set");
        }
        let third = port
            .download(&stream, [&items_x[2], &w], &ids, key(2))
            .expect_err("a third download while two sets are in flight");
        assert!(
            third
                .to_string()
                .contains("both sets hold a layer not uploaded yet"),
            "{third}"
        );
        stream.synchronize().expect("the stream");
    }

    /// Two units side by side in one block of five rows — unit 0 the first
    /// three, unit 1 the last two — walked in the fronts order over two
    /// layers through the port's two sets: every serve takes its unit's rows
    /// of the block and no others, each unit's sums land in its own rows of
    /// `hsum`, and each back reads its own unit's sums. The block's rows
    /// hold `unit + 1`, so a key at another unit's rows reads the other
    /// unit's sums.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_two_units_at_row_offsets_keep_their_sums_apart() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let (n_embd, cap) = (2, 5);
        let mut port = BatchPort::new(&ctx, n_embd, 1, cap).expect("a batch port");
        let mut hsum = DeviceBuffer::<f32>::zeroed(&stream, cap * n_embd).expect("hsum");
        let rows = [0..3, 3..5];
        let block: Vec<f32> = rows
            .iter()
            .enumerate()
            .flat_map(|(unit, r)| vec![(unit + 1) as f32; n_embd * r.len()])
            .collect();
        let mut items = Items::<true> {
            x: vec![DeviceBuffer::from_host(&stream, &block).expect("the block")],
            x_of: vec![0, 0],
            w: DeviceBuffer::from_host(&stream, &[1.0f32; 5]).expect("w"),
            ids: DeviceBuffer::from_host(&stream, &[0u32; 5]).expect("ids"),
            backs: Vec::new(),
            port: std::marker::PhantomData,
        };
        let mut ex = Exchange {
            stream: &stream,
            port: &mut port,
            hsum: &mut hsum,
            cap,
            n_embd,
            shape: UnitShape {
                unit_rows: Some(&rows),
                ..UnitShape::default()
            },
            served: Vec::new(),
        };
        sched::walk(walk(2, 3), 2, &mut ex, &mut items)
            .expect("a two-unit walk over two layers in the fronts order");
        let want: Vec<(At, Vec<f32>)> = sched::order_in(walk(2, 3), 2, BatchOrder::Fronts)
            .filter_map(|i| match i {
                sched::Item::Back(at) => Some(at),
                _ => None,
            })
            .map(|at| {
                let v = (at.unit + 1 + at.layer) as f32;
                (at, vec![v; n_embd * rows[at.unit].len()])
            })
            .collect();
        assert_eq!(items.backs.len(), 4);
        assert_eq!(items.backs, want);
        assert!(
            ex.served
                .iter()
                .all(|&(at, values)| values == n_embd * rows[at.unit].len()),
            "a serve's activations are its unit's rows: {:?}",
            ex.served
        );
        stream.synchronize().expect("the stream");
    }
}
