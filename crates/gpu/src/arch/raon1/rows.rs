//! Raon-1's passes of several rows, each a walk of the body over the rows
//! the staging holds ([`Run::Rows`]), and the taps a block draft reads:
//! - a verify ([`Rows`]): the current id and the drafted ones at the next
//!   positions, each row's RDA state in a lane of its own, every row into
//!   one head of the pass's rows; [`Rollback`] commits the rows kept
//!   ([`super::state`]);
//! - a prompt call ([`GpuModel::prompt_rows`]): wide units of up to
//!   [`WIDE_ROWS`] rows ([`super::wide`]) while at least [`WIDE_FROM`] rows
//!   remain and the file runs wide ([`Body::wide_ok`]), then the rest's ids
//!   staged on the card [`GROUP`] at a time, each
//!   chunk of up to [`PASS_ROWS`] of them one walk in place, the head on the
//!   call's last row alone;
//! - the taps ([`GpuModel::set_draft_taps`]): the streams entering the
//!   layers a draft names, laid in by every walk, one row a position.
//!
//! The position the stores stand at ([`RowState::held`]) is the body's: a
//! step, a pass or a prompt call anywhere else is refused by name, since an
//! RDA layer's state holds its position alone.

use super::body::{Body, Run, Tail};
use super::mhc::STREAMS;
use super::state::{Lanes, PASS_ROWS, Pass};
use super::wide::{WIDE_FROM, WIDE_ROWS};
use crate::draft_taps::{DraftTaps, TapRowsSink};
use crate::head::Head;
use crate::hybrid::Chain;
use crate::model::{GpuModel, Rollback, RowHeads, Rows};
use crate::tensor::Window;
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};

const WHAT: &str = "raon1 rows";

/// The rows one staging copy holds: a prompt call stages its ids, and the
/// taps hand their rows back, once a group.
pub(super) const GROUP: usize = 64;

/// The rows a pass walks, staged on the card by one copy: `cap` rows of
/// ids (the token once a stream), positions and live key counts, one buffer
/// in that order; and the embedding's own per-row outputs, which no layer
/// reads.
pub(super) struct RowsIn {
    cap: usize,
    host: Vec<u32>,
    buf: DeviceBuffer<u32>,
    epos: DeviceBuffer<u32>,
    ekeys: DeviceBuffer<u32>,
}

/// One walk's windows of the staging ([`RowsIn::views`]).
pub(super) struct RowViews<'a> {
    /// The rows' ids, [`STREAMS`] a row.
    pub(super) ids: Window<'a, u32>,
    /// The rows' positions and live key counts, one a row.
    pub(super) pos: Window<'a, u32>,
    pub(super) keys: Window<'a, u32>,
    /// What the embedding writes of its rows' positions, a stream a row.
    pub(super) epos: &'a mut DeviceBuffer<u32>,
    pub(super) ekeys: &'a mut DeviceBuffer<u32>,
}

impl RowsIn {
    pub(super) fn new(stream: &CudaStream, cap: usize) -> Result<RowsIn, GpuError> {
        let len = (STREAMS + 2) * cap;
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        Ok(RowsIn {
            cap,
            host: vec![0; len],
            buf: u(len)?,
            epos: u(STREAMS * cap)?,
            ekeys: u(STREAMS * cap)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        [&self.buf, &self.epos, &self.ekeys]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum()
    }

    /// `ids` at positions `pos, pos + 1, …` into the staging's first rows:
    /// one copy, which waits for the walks before it. Refused by name for
    /// no ids, more than the staging holds, or a live count past u32.
    pub(super) fn stage(
        &mut self,
        stream: &CudaStream,
        ids: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        let c = self.cap;
        if ids.is_empty() || ids.len() > c {
            return Err(GpuError::shape(
                WHAT,
                format!("{} ids staged in {c} rows", ids.len()),
            ));
        }
        let last = launch_u32(WHAT, "rows", ids.len())?;
        pos.checked_add(last).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("{last} rows from position {pos}: the live count passes u32"),
            )
        })?;
        for (r, (&id, at)) in ids.iter().zip(pos..).enumerate() {
            self.host[STREAMS * r..STREAMS * (r + 1)].fill(id);
            self.host[STREAMS * c + r] = at;
            self.host[(STREAMS + 1) * c + r] = at + 1;
        }
        Ok(self.buf.copy_from_host(stream, &self.host)?)
    }

    /// The windows of `m` staged rows from row `at`; refused by name past
    /// the staging.
    pub(super) fn views(&mut self, at: usize, m: usize) -> Result<RowViews<'_>, GpuError> {
        let c = self.cap;
        if m == 0 || at + m > c {
            return Err(GpuError::shape(
                WHAT,
                format!("rows {at}..{} of a staging of {c}", at + m),
            ));
        }
        let f = size_of::<u32>();
        let RowsIn {
            buf, epos, ekeys, ..
        } = self;
        let buf: &DeviceBuffer<u32> = buf;
        Ok(RowViews {
            ids: Window::of(buf, STREAMS * at * f, STREAMS * m)?,
            pos: Window::of(buf, (STREAMS * c + at) * f, m)?,
            keys: Window::of(buf, ((STREAMS + 1) * c + at) * f, m)?,
            epos,
            ekeys,
        })
    }
}

/// The body's side of its passes: the staging, the RDA stores' lane word,
/// the taps when a draft set them, and the position every store stands at.
pub(super) struct RowState {
    pub(super) rin: RowsIn,
    pub(super) lanes: Lanes,
    pub(super) taps: Option<DraftTaps>,
    /// The position the next row runs at: where every store stands once the
    /// walks planned so far have run.
    pub(super) held: u32,
}

impl RowState {
    /// The staging of [`GROUP`] rows and the word of `lanes` lanes (1, or
    /// [`PASS_ROWS`] on a load that verifies), no taps, at position 0.
    /// Load-time only.
    pub(super) fn new(stream: &CudaStream, lanes: usize) -> Result<RowState, GpuError> {
        Ok(RowState {
            rin: RowsIn::new(stream, GROUP)?,
            lanes: Lanes::new(stream, lanes)?,
            taps: None,
            held: 0,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.rin.bytes()
            + self.lanes.word().num_bytes()
            + self.taps.as_ref().map_or(0, DraftTaps::bytes)
    }

    /// Refused by name unless the stores stand at `pos`.
    fn stands_at(&self, pos: u32, what: &str) -> Result<(), GpuError> {
        if pos != self.held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{what} at position {pos}; the stores stand at {} (a call that failed \
                     part-way: reset)",
                    self.held
                ),
            ));
        }
        Ok(())
    }

    /// A step at `pos`: refused by name while a verify waits or off the
    /// position the stores stand at; the taps lay in its one row.
    pub(super) fn step_at(&mut self, pos: u32) -> Result<(), GpuError> {
        self.stands_at(pos, "a step")?;
        self.lanes.open(Pass::InPlace, pos, 1)?;
        if let Some(t) = self.taps.as_mut() {
            t.plan(1);
        }
        Ok(())
    }

    /// Back to position 0, the lane word at lane 0 and nothing waiting,
    /// beside the stores' own reset.
    pub(super) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.lanes.reset(stream)?;
        self.held = 0;
        Ok(())
    }
}

impl Rows for Body {
    const MAX_ROWS: usize = PASS_ROWS;
    /// No host tier serves a Raon-1 replay; the pass is the chain's
    /// one-token layout at `m` rows.
    const CHAIN: Chain = Chain::Step;
    /// The head's rows are independent: one read of `output` for the pass.
    const HEADS: RowHeads = RowHeads::One;

    /// The pass's ids and positions staged, the verify opened on the lane
    /// word ([`Lanes::open`]), the taps planned at its rows. Refused by name
    /// off the position the stores stand at, while a verify waits, and on a
    /// load of one lane ([`GpuModel::open_verifying`] holds the lanes).
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        let r = &mut self.rows;
        r.stands_at(pos, "a verify")?;
        r.lanes.open(Pass::Verify, pos, tokens.len())?;
        if let Err(e) = r.rin.stage(stream, tokens, pos) {
            r.lanes.abandon();
            return Err(e);
        }
        if let Some(t) = r.taps.as_mut() {
            t.plan(tokens.len());
        }
        r.held = pos + launch_u32(WHAT, "rows", tokens.len())?;
        Ok(())
    }

    /// The walk of the staged rows into `heads[0]`, a head of the pass's
    /// rows, each row's RDA state in a lane of its own. Refused by name for
    /// a head of fewer than 2 rows or more than the load's lanes.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        let head = heads
            .first_mut()
            .ok_or(GpuError::state(WHAT, "the pass's head of its rows"))?;
        let (m, lanes) = (head.m(), self.rows.lanes.count());
        if !(2..=lanes).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a verify of {m} rows on a load of {lanes} lanes an RDA store: 2 to the lanes \
                     (GpuModel::open_verifying holds {PASS_ROWS})"
                ),
            ));
        }
        let run = Run::Rows {
            at: 0,
            m,
            pass: Pass::Verify,
        };
        self.walk(gpu, w, run, Tail::Head(head))
    }
}

impl Rollback for Body {
    /// Without the card only the position the stores stand at, nothing
    /// waiting: a commit moves the lane word on the card
    /// ([`Rollback::rollback_on`]).
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        let r = &self.rows;
        if r.lanes.waiting().is_some() || pos != r.held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "back to position {pos} from {} without the card: a commit moves the lane \
                     word (GpuModel::rollback)",
                    r.held
                ),
            ));
        }
        Ok(())
    }

    /// The verify waiting kept to `pos`: its first `pos − pos0` rows, 1 to
    /// its rows ([`Lanes::commit`], which copies the lane word when it
    /// moves). With nothing waiting, the position the stores stand at takes
    /// nothing back; any other is refused by name: an RDA state holds its
    /// position alone, so a cut keeps it or nothing (a reset).
    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let r = &mut self.rows;
        match r.lanes.waiting() {
            Some((pos0, rows)) => {
                let keep = pos.checked_sub(pos0).ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!(
                            "back to position {pos}, before the verify of {rows} rows at {pos0}"
                        ),
                    )
                })?;
                r.held = r.lanes.commit(gpu.stream(), keep as usize)?;
                Ok(())
            }
            None if pos == r.held => Ok(()),
            None => Err(GpuError::shape(
                WHAT,
                format!(
                    "back to position {pos} from {}: an RDA state holds its position alone (a \
                     cut keeps it or resets)",
                    r.held
                ),
            )),
        }
    }
}

impl Body {
    /// A prompt call's walks from `pos` ([`GpuModel::prompt_rows`]): wide
    /// units of up to [`WIDE_ROWS`] rows ([`super::wide`]) as long as at
    /// least [`WIDE_FROM`] rows remain and the file runs wide
    /// ([`Body::wide_ok`]; a file that does not takes the rest's path for
    /// every row), then the rest with its ids staged a
    /// group at a time, each chunk of up to [`PASS_ROWS`] a walk in place
    /// (every row the step's bits). The head is on the call's last row; with
    /// `tapped`, the taps of the call's last `window` rows are handed to its
    /// sink after each unit or group that holds any. Returns `true`: the
    /// call's last walk enqueues the head.
    fn prompt_walk(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        (ids, pos): (&[u32], u32),
        mut tapped: Option<(usize, &mut TapRowsSink<'_>)>,
    ) -> Result<bool, GpuError> {
        let stream = gpu.stream();
        let n = ids.len();
        self.rows.stands_at(pos, "a prompt call")?;
        self.rows.lanes.open(Pass::InPlace, pos, n)?;
        if tapped.is_some() && self.rows.taps.is_none() {
            return Err(GpuError::state(WHAT, "taps set (set_draft_taps)"));
        }
        let from = tapped
            .as_ref()
            .map_or(n, |(window, _)| n - (*window).min(n));
        let mut done = 0;
        while self.wide_ok() && n - done >= WIDE_FROM {
            let m = WIDE_ROWS.min(n - done);
            let at = pos + launch_u32(WHAT, "row", done)?;
            self.wide.rin.stage(stream, &ids[done..done + m], at)?;
            if let Some(t) = self.rows.taps.as_mut() {
                t.plan(m);
            }
            let tail = if done + m == n {
                Tail::Last(&mut *head)
            } else {
                Tail::Open
            };
            let run = Run::Rows {
                at: 0,
                m,
                pass: Pass::InPlace,
            };
            self.walk(gpu, w, run, tail)?;
            self.rows.held = pos + launch_u32(WHAT, "row", done + m)?;
            self.hand_over(gpu, tapped.as_mut(), (pos, from), (done, m))?;
            done += m;
        }
        for (g, group) in ids[done..].chunks(GROUP).enumerate() {
            let g0 = done + g * GROUP;
            let gpos = pos + launch_u32(WHAT, "row", g0)?;
            self.rows.rin.stage(stream, group, gpos)?;
            if let Some(t) = self.rows.taps.as_mut() {
                t.plan(group.len());
            }
            for (c, chunk) in group.chunks(PASS_ROWS).enumerate() {
                let (at, m) = (c * PASS_ROWS, chunk.len());
                let tail = if g0 + at + m == n {
                    Tail::Last(&mut *head)
                } else {
                    Tail::Open
                };
                let run = Run::Rows {
                    at,
                    m,
                    pass: Pass::InPlace,
                };
                self.walk(gpu, w, run, tail)?;
                self.rows.held = gpos + launch_u32(WHAT, "row", at + m)?;
            }
            self.hand_over(gpu, tapped.as_mut(), (pos, from), (g0, group.len()))?;
        }
        Ok(true)
    }

    /// The tapped rows of the call's rows `first .. first + rows` that lie
    /// from `from` on, handed to the sink as `(position, rows)`; the call
    /// began at position `pos`. Nothing for an untapped call or a unit wholly
    /// before `from`.
    fn hand_over(
        &self,
        gpu: &Gpu,
        tapped: Option<&mut (usize, &mut TapRowsSink<'_>)>,
        (pos, from): (u32, usize),
        (first, rows): (usize, usize),
    ) -> Result<(), GpuError> {
        let Some((_, sink)) = tapped else {
            return Ok(());
        };
        let (lo, hi) = (from.max(first), first + rows);
        if lo >= hi {
            return Ok(());
        }
        let taps = self
            .rows
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "taps set (set_draft_taps)"))?;
        let read = taps.read(gpu.stream(), lo - first, hi - lo)?;
        sink(pos + launch_u32(WHAT, "row", lo)?, &read)
    }
}

impl GpuModel<Body> {
    /// Run `ids` from the model's position and return the argmax after the
    /// last: units of up to [`WIDE_ROWS`] ids as GEMMs (the wide arm: a row
    /// is the step's row to the error of the GEMM's sums, not its bits) while
    /// at least [`WIDE_FROM`] remain, the rest as passes of up to
    /// [`PASS_ROWS`] ids whose rows are the steps' bits; a call of up to
    /// [`PASS_ROWS`] ids is those passes alone. The head is on the call's
    /// last row alone (its fault word read there). Refused by name with no
    /// ids, and while a verify waits.
    pub fn prompt_rows(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        self.prompt_call(ids, None)
    }

    /// [`GpuModel::prompt_rows`] with the tapped rows of the call's last
    /// `window` positions (all of them when the call is shorter) handed to
    /// `sink` as `(first position, rows)`, one [`GpuModel::draft_tap_width`]
    /// row a position, after each unit or group that holds any of them — in
    /// order, no position twice or missed. The walks are the plain
    /// call's, so its stores and argmax are bit for bit. Refused by name with
    /// no taps set; a sink that fails ends the call with its error.
    pub fn prompt_rows_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: &mut TapRowsSink<'_>,
    ) -> Result<u32, GpuError> {
        self.prompt_call(ids, Some((window, sink)))
    }

    fn prompt_call(
        &mut self,
        ids: &[u32],
        tapped: Option<(usize, &mut TapRowsSink<'_>)>,
    ) -> Result<u32, GpuError> {
        const WHAT_P: &str = "raon1 GpuModel::prompt_rows";
        self.run_rows(ids.len(), WHAT_P, |gpu, w, body, head, pos| {
            body.prompt_walk(gpu, w, head, (ids, pos), tapped)
        })?
        .ok_or(GpuError::state(
            WHAT_P,
            "the head's token of the call's last row",
        ))
    }

    /// Tap the rows entering `layers` (strictly rising, each below the
    /// layer count; empty stops tapping) from the next walk on: every walk —
    /// a step, a verify's pass, each prompt chunk or wide unit — lays them
    /// into a buffer of [`WIDE_ROWS`] rows a layer ([`DraftTaps`]), allocated here against the
    /// card's free bytes with [`FIT_RESERVE`](crate::model::fit::FIT_RESERVE)
    /// left. The captured chains are dropped first (a capture records the
    /// copies into the buffer of its time), and the next graph step or pass
    /// recaptures with the copies. Taps already at `layers` stay as they are,
    /// captures with them. A refused call leaves taps off. Load-time
    /// allocation.
    pub fn set_draft_taps(&mut self, layers: &[usize]) -> Result<(), GpuError> {
        const WHAT_S: &str = "raon1 GpuModel::set_draft_taps";
        let (_, _, body) = self.body_parts(WHAT_S)?;
        if body
            .rows
            .taps
            .as_ref()
            .is_some_and(|t| t.layers() == layers)
        {
            return Ok(());
        }
        self.drop_captures();
        let (gpu, _, body) = self.body_parts(WHAT_S)?;
        body.rows.taps = None;
        if layers.is_empty() {
            return Ok(());
        }
        let (n_layer, hidden) = body.tap_shape();
        let free = gpu.alloc_room()?;
        body.rows.taps = Some(DraftTaps::new(
            gpu.stream(),
            layers,
            n_layer,
            (hidden, WIDE_ROWS),
            free,
        )?);
        Ok(())
    }

    /// Floats one position's tapped row holds: the tapped layers times the
    /// streams' width; 0 with no taps set.
    #[must_use]
    pub fn draft_tap_width(&self) -> usize {
        self.body(WHAT)
            .map_or(0, |b| b.rows.taps.as_ref().map_or(0, DraftTaps::width))
    }

    /// The first `rows` rows of the last walk — a step's one, a verify's
    /// `m` — one [`GpuModel::draft_tap_width`] row each, a position's taps
    /// in the order the layers were named, four streams a layer (the rows
    /// `DsparkBody::append_as` takes). Blocking. Refused by name with no taps
    /// set, and for more rows than the last walk laid in.
    pub fn draft_taps(&mut self, rows: usize) -> Result<Vec<f32>, GpuError> {
        let (gpu, _, body) = self.body_parts(WHAT)?;
        body.rows
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "taps set (set_draft_taps)"))?
            .read(gpu.stream(), 0, rows)
    }
}
