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
//! order — the next item's front ahead of this item's serve — keeps at most
//! two layer-batches in flight. The leg's one `hsum` holds one item's sums
//! at a time: item `x + 1`'s upload is enqueued after item `x`'s back read
//! them (`runtime::sched`'s batch order puts `x`'s back before `x + 1`'s
//! serve and, for a tiered layer, before `x + 1`'s join).
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

/// `at`'s exchange key over `cols` columns: the unit the batch's label
/// within its group ([`BatchKey::set`]), its tokens `0 .. cols`.
fn batch_key(at: At, cols: usize) -> BatchKey {
    BatchKey {
        layer: at.layer,
        set: at.unit,
        at: 0,
        u: cols,
    }
}

/// What [`BatchLeg`]'s [`Port::open`] refuses by name: no unit, a walk's
/// columns outside `1..=cap`, or units' own columns (`unit_cols`) that are not
/// one entry a unit, each in `1..=` the walk's columns.
fn refuse_open(o: Overlap, cap: usize, unit_cols: Option<&[usize]>) -> Result<(), GpuError> {
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
    if let Some(c) = unit_cols
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
/// batch may be shorter).
pub struct BatchLeg<'a, H: HostExperts> {
    stream: &'a CudaStream,
    hybrid: &'a mut HostTier<H>,
    hsum: &'a mut DeviceBuffer<f32>,
    cap: usize,
    /// The open walk's columns.
    cols: usize,
    /// Each unit's columns, when the walk set them.
    unit_cols: Option<&'a [usize]>,
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
            timer: None,
        }
    }

    /// The walk's timing, `timer`'s ([`LegTimer`]): its serves timed, its
    /// parts marked — or none, which runs the exchange alone.
    pub fn set_timer<T: LegTimer + 'a>(&mut self, timer: Option<&'a mut T>) {
        self.timer = timer.map(|t| t as &mut dyn LegTimer);
    }

    /// Each unit's own columns, unit `u` the `u`-th: the next walk's
    /// [`Port::open`] refuses by name a list that is not one entry a unit,
    /// each in `1..=` the walk's columns.
    pub fn set_unit_cols(&mut self, cols: &'a [usize]) {
        self.unit_cols = Some(cols);
    }

    /// Unit `unit`'s columns: its own when the walk set them, else the
    /// walk's.
    fn cols_of(&self, unit: usize) -> usize {
        self.unit_cols
            .and_then(|c| c.get(unit).copied())
            .unwrap_or(self.cols)
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

    /// `at`'s exchange key over its unit's columns: the unit is the batch's
    /// label within its group, and the port picks its exchange set in turn.
    #[must_use]
    pub fn key(&self, at: At) -> BatchKey {
        batch_key(at, self.cols_of(at.unit))
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
            Some(t) => t.upload_mark(self.stream, at),
            None => Ok(()),
        }
    }
}

impl<H: HostExperts> Port for BatchLeg<'_, H> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Batch;

    /// One unit or more of `1..=cap` columns — each unit's own columns, when
    /// the walk set them, one a unit in `1..=` the walk's — both sets of the
    /// port free ([`HostTier::begin_group`], once a walk: the walk is eager,
    /// and its caller leaves nothing of an earlier walk in flight); any other
    /// point is refused by name.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        refuse_open(o, self.cap, self.unit_cols)?;
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
            cols: key.u,
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

#[cfg(test)]
mod tests {
    use super::super::batch::BatchPort;
    use super::*;
    use runtime::sched::{self, LayerProgram};

    /// The batch port's exchange under a walk's order, keyed as the leg keys
    /// it: each front downloads its item, each serve serves and uploads it,
    /// each back reads the sums the upload sent.
    struct Exchange<'a> {
        stream: &'a CudaStream,
        port: &'a mut BatchPort,
        hsum: &'a mut DeviceBuffer<f32>,
        cols: &'a [usize],
    }

    impl Port for Exchange<'_> {
        type Error = GpuError;
        const KIND: PortKind = PortKind::Batch;

        fn open(&mut self, _: Overlap) -> Result<(), GpuError> {
            self.port.begin();
            Ok(())
        }

        fn serve(&mut self, at: At) -> Result<(), GpuError> {
            let key = batch_key(at, self.cols[at.unit]);
            self.port.serve(key, |layer, x, _, _, sums| {
                // Each sum is the row's own value plus the layer: the back
                // reads which item's sums it got.
                for (s, v) in sums.iter_mut().zip(x.data()) {
                    *s = v + layer as f32;
                }
                Ok(())
            })?;
            self.port.upload(self.stream, self.hsum, key)
        }

        fn refused(why: Refused) -> GpuError {
            GpuError::shape(WHAT_BATCH, why.to_string())
        }
    }

    /// Each item's rows, `unit + 1` everywhere, and the sums each back read.
    struct Items<'a> {
        x: [DeviceBuffer<f32>; 3],
        w: DeviceBuffer<f32>,
        ids: DeviceBuffer<u32>,
        backs: Vec<(At, Vec<f32>)>,
        port: std::marker::PhantomData<Exchange<'a>>,
    }

    impl<'a> LayerProgram for Items<'a> {
        type Port = Exchange<'a>;

        fn front(&mut self, port: &mut Exchange<'a>, at: At) -> Result<(), GpuError> {
            let key = batch_key(at, port.cols[at.unit]);
            port.port
                .download(port.stream, [&self.x[at.unit], &self.w], &self.ids, key)
        }

        fn back(&mut self, port: &mut Exchange<'a>, at: At) -> Result<(), GpuError> {
            let n = 2 * port.cols[at.unit];
            let got = port.hsum.to_host_vec(port.stream)?;
            self.backs.push((at, got[..n].to_vec()));
            Ok(())
        }
    }

    /// The leg opens a walk of one unit or more, each unit's own columns
    /// in `1..=` the walk's, and refuses by name no unit, columns outside
    /// `1..=cap`, and a units' list of another length or with an entry
    /// outside the walk's columns.
    #[test]
    fn open_takes_one_unit_or_more_and_refuses_the_rest() {
        let walk = |units, cols| Overlap {
            units,
            cols,
            port: PortKind::Batch,
        };
        for units in [1, 2, 3, 8] {
            refuse_open(walk(units, 512), 512, None)
                .unwrap_or_else(|e| panic!("a walk of {units} units: {e}"));
        }
        refuse_open(walk(3, 512), 512, Some(&[512, 512, 1]))
            .expect("three units, the last of one column");
        let refused = [
            (walk(0, 512), None, "0 units of 512 columns"),
            (walk(2, 0), None, "2 units of 0 columns"),
            (walk(2, 513), None, "2 units of 513 columns"),
            (
                walk(3, 512),
                Some(&[512, 512][..]),
                "units of [512, 512] columns",
            ),
            (
                walk(2, 512),
                Some(&[512, 0][..]),
                "units of [512, 0] columns",
            ),
            (
                walk(2, 256),
                Some(&[256, 512][..]),
                "units of [256, 512] columns",
            ),
        ];
        for (o, cols, want) in refused {
            let e = refuse_open(o, 512, cols).expect_err(want);
            assert!(e.to_string().contains(want), "{want}: {e}");
        }
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
        let (backs, items_x, w, ids) = {
            let mut items = Items {
                x: [x(0), x(1), x(2)],
                w: DeviceBuffer::from_host(&stream, &[1.0f32; 4]).expect("w"),
                ids: DeviceBuffer::from_host(&stream, &[0u32; 4]).expect("ids"),
                backs: Vec::new(),
                port: std::marker::PhantomData,
            };
            let mut ex = Exchange {
                stream: &stream,
                port: &mut port,
                hsum: &mut hsum,
                cols: &cols,
            };
            let o = Overlap {
                units: 3,
                cols: 4,
                port: PortKind::Batch,
            };
            sched::walk(o, 2, &mut ex, &mut items).expect("a three-unit walk over two layers");
            (items.backs, items.x, items.w, items.ids)
        };
        let want: Vec<(At, Vec<f32>)> = (0..2)
            .flat_map(|layer| (0..3).map(move |unit| At { unit, layer }))
            .map(|at| {
                let v = (at.unit + 1 + at.layer) as f32;
                (at, vec![v; n_embd * cols[at.unit]])
            })
            .collect();
        assert_eq!(backs, want);
        port.begin();
        for (unit, x) in items_x.iter().take(2).enumerate() {
            port.download(&stream, [x, &w], &ids, batch_key(At { unit, layer: 0 }, 4))
                .expect("a download into a free set");
        }
        let third = port
            .download(
                &stream,
                [&items_x[2], &w],
                &ids,
                batch_key(At { unit: 2, layer: 0 }, 4),
            )
            .expect_err("a third download while two sets are in flight");
        assert!(
            third
                .to_string()
                .contains("both sets hold a layer not uploaded yet"),
            "{third}"
        );
        stream.synchronize().expect("the stream");
    }
}
