//! A recurrent model's sequence state on the host ([`SeqState`]): what a
//! server's prompt cache holds for a session it switches away from, and puts
//! back when the session returns.
//!
//! A state of `n` positions holds three parts, each copied once between the
//! card and the host:
//! - the positional stores' rows of positions `0..n` (the K and V planes,
//!   the keys a selector scores, every f16 store indexed by position), as the
//!   body lists them ([`Spans`]): rows past `n` are never read before a later
//!   position writes them, so they are not carried;
//! - the recurrent stores (the body's checkpoint list,
//!   [`Checkpoints::lens`]) at `n`, read from the card;
//! - the recurrent stores at each point it carries, read from the
//!   checkpoints' host slots ([`Checkpoints::read_point`]): a later request
//!   that shares only a shorter prefix (a resend with the reasoning taken
//!   out of the last turn) cuts to one of them.
//!
//! Each position the state carries has a host value of the body's own `T`
//! beside it (a history the card does not hold), handed back with it.
//!
//! A positional store moves as one 2-D copy ([`crate::tensor::dtoh_2d`]): its
//! planes are the rows of the copy, the card's `stride` apart and the host's
//! `len` apart, so a store of four planes is one call, not four, and a state
//! takes one stream synchronize, not one a plane.
//!
//! A resume ([`SeqState::load`]) puts every part back: the positional rows
//! and the recurrent stores at `n` on the card, each carried point into the
//! checkpoints ([`Checkpoints::adopt`]), which then hold those points and no
//! other. [`SeqState::load_prefix`] puts back only the first `k` positions the
//! state can be cut to (its [`SeqState::keep_point`]): the rows of those
//! positions, the recurrent stores at `k` and the points up to it, moving
//! exactly what a load and a cut to `k` would leave and none of the rows past
//! it. Both are refused by name, nothing copied, for a state of another
//! model, card or context ([`Identity`]), of another store layout, or onto
//! checkpoints that hold a point.
//!
//! A save the request path does not wait for ([`SeqState::save_async`]) copies
//! on a stream of its own ([`SaveLane`]) into pinned staging ([`Staging`])
//! after everything the engine stream ran so far, and returns a [`Pending`];
//! the state is read only once the copy has landed ([`Pending::land`]). The
//! rows the copy reads are not cut, reset or replaced before then.
//!
//! The host memory of a state taken by [`SeqState::save`] is pageable: each
//! copy runs on the engine stream, one synchronize at the end.

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, sys};
use runtime::savemap::{Per, Run, Strip, plan_prefix, plan_whole};

use super::Checkpoints;
use super::staging::{NoRoom, Staged, Staging};
use crate::GpuError;
use crate::tensor::{Pitched, dtoh_2d, htod_2d};

const WHAT: &str = "sequence state";

/// What a state belongs to: a resume onto anything else is refused by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// The architecture's name.
    pub arch: &'static str,
    /// The model file's first shard.
    pub file: PathBuf,
    /// The card the stores live on.
    pub card: String,
    /// The positions every store was sized for.
    pub ctx: usize,
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} on {} at {} positions",
            self.arch,
            self.file.display(),
            self.card,
            self.ctx
        )
    }
}

/// The rows of one positional f16 store a state carries, under one borrow
/// of it: `planes` runs of `len` values, the first at value 0 and each
/// `stride` values past the one before.
pub struct Spans<'a> {
    buf: &'a mut DeviceBuffer<u16>,
    run: Run,
}

impl<'a> Spans<'a> {
    /// The runs of `buf` (the type's doc), of no growth rule: only the whole
    /// runs are ever copied, so a state with such a store cannot be put back
    /// as a prefix ([`SeqState::load_prefix`]). Refused by name when one run
    /// would leave the buffer or overlap the next.
    pub fn planes(
        buf: &'a mut DeviceBuffer<u16>,
        planes: usize,
        stride: usize,
        len: usize,
    ) -> Result<Spans<'a>, GpuError> {
        Spans::of(
            buf,
            Run {
                planes,
                stride,
                len,
                per: Per::Whole,
            },
        )
    }

    /// The runs of `n` positions of a store that holds `values` values a
    /// position in each plane: a prefix of `k` positions is the first
    /// `k · values` of each plane.
    pub fn by_position(
        buf: &'a mut DeviceBuffer<u16>,
        planes: usize,
        stride: usize,
        n: usize,
        values: usize,
    ) -> Result<Spans<'a>, GpuError> {
        let per = Per::Position { values };
        let len = per.len_at(n).map_err(|e| shape(e.to_string()))?;
        Spans::of(
            buf,
            Run {
                planes,
                stride,
                len,
                per,
            },
        )
    }

    /// The runs of `n` positions of a store that holds `values` values a pool
    /// of `every` positions in each plane, the last pool partial: a prefix of
    /// `k` positions is the first `ceil(k / every) · values` of each plane.
    pub fn by_pool(
        buf: &'a mut DeviceBuffer<u16>,
        planes: usize,
        stride: usize,
        n: usize,
        pool: (usize, usize),
    ) -> Result<Spans<'a>, GpuError> {
        let (every, values) = pool;
        let per = Per::Pool { every, values };
        let len = per.len_at(n).map_err(|e| shape(e.to_string()))?;
        Spans::of(
            buf,
            Run {
                planes,
                stride,
                len,
                per,
            },
        )
    }

    fn of(buf: &'a mut DeviceBuffer<u16>, run: Run) -> Result<Spans<'a>, GpuError> {
        let end = run
            .planes
            .checked_sub(1)
            .map_or(Some(0), |p| p.checked_mul(run.stride)?.checked_add(run.len));
        if end.is_none_or(|e| e > buf.len()) || (run.planes > 1 && run.stride < run.len) {
            return Err(shape(format!(
                "{} planes of {} values {} apart in a store of {}",
                run.planes,
                run.len,
                run.stride,
                buf.len()
            )));
        }
        Ok(Spans { buf, run })
    }

    /// The values of each run, in order.
    fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.run.planes).map(|_| self.run.len)
    }
}

/// The stores a state is read from or put back into, lent for one call:
/// the positional rows the body lists, and the recurrent stores in the
/// checkpoints' order ([`Checkpoints::lens`]).
pub struct Lent<'s, 'p, 'r> {
    pub positional: &'s mut [Spans<'p>],
    pub recurrent: &'s mut [&'r mut DeviceBuffer<f32>],
}

impl Lent<'_, '_, '_> {
    fn spans(&self) -> Vec<usize> {
        self.positional.iter().flat_map(Spans::sizes).collect()
    }

    fn lens(&self) -> Vec<usize> {
        self.recurrent.iter().map(|s| s.len()).collect()
    }

    fn runs(&self) -> Vec<Run> {
        self.positional.iter().map(|s| s.run).collect()
    }
}

/// The host values a resume hands back: the current position's, and each
/// carried point's with its position.
pub type Hosts<'s, T> = (&'s T, &'s [(u32, T)]);

/// What a resume put back ([`SeqState::load_prefix`]): where the model stands
/// now, the host values to stand there with, and what the copies moved.
pub struct Resumed<'s, T> {
    /// The positions the model holds: the state's, or the point a prefix
    /// was cut to.
    pub at: u32,
    /// The host value of the position `at`.
    pub current: &'s T,
    /// The points the checkpoints adopted, ascending, each at or below `at`
    /// (the one at `at` itself, when `at` is a point), with their host values.
    pub points: &'s [(u32, T)],
    /// The bytes the copies moved to the card, and the calls that moved them.
    pub bytes: u64,
    pub calls: u64,
}

/// The host arrays of a state: pageable, taken on the engine stream, or an
/// extent of pinned staging a copy stream filled.
enum Host {
    Owned { rows: Vec<u16>, recurrent: Vec<f32> },
    Staged(Staged),
}

impl Host {
    fn rows(&self) -> &[u16] {
        match self {
            Host::Owned { rows, .. } => rows,
            Host::Staged(s) => s.rows(),
        }
    }

    fn recurrent(&self) -> &[f32] {
        match self {
            Host::Owned { recurrent, .. } => recurrent,
            Host::Staged(s) => s.recurrent(),
        }
    }
}

/// A sequence state of `positions` positions on the host (the module doc),
/// with the body's host value `T` beside the current position and each
/// carried point.
pub struct SeqState<T> {
    who: Identity,
    positions: u32,
    /// The positional rows, in the order the body lists them.
    host: Host,
    /// The positional stores' runs, in the order the body lists them.
    runs: Vec<Run>,
    /// f32s of each recurrent store.
    lens: Vec<usize>,
    current: T,
    /// Ascending, each below `positions`.
    points: Vec<(u32, T)>,
    /// What the copy cost, set once the host arrays hold the whole state: at
    /// once for a state taken on the engine stream, by [`Pending::land`] for
    /// one a copy stream fills. A resume reads the arrays only after.
    landed: OnceLock<SaveCost>,
}

/// The copy stream saves run on and the pinned staging their states are made
/// in. One per model: made at load, beside the engine stream.
pub struct SaveLane {
    stream: Arc<CudaStream>,
    staging: Staging,
}

impl SaveLane {
    /// A lane on `ctx`'s device with `limit` pinned bytes of states, all
    /// pinned now ([`Staging::pinned`]). Load-time only: it pins, and makes a
    /// stream, where `ctx` already has its first.
    pub fn new(ctx: &Arc<CudaContext>, limit: u64) -> Result<SaveLane, GpuError> {
        Ok(SaveLane {
            stream: ctx.new_stream()?,
            staging: Staging::pinned(ctx, limit)?,
        })
    }

    /// The pinned staging the lane's states are made in.
    #[must_use]
    pub fn staging(&self) -> &Staging {
        &self.staging
    }
}

/// The two streams of a save the request path does not wait for: the lane's
/// copy stream, which waits for what `engine` ran so far, and the engine's.
#[derive(Clone, Copy)]
pub struct Streams<'a> {
    pub lane: &'a SaveLane,
    pub engine: &'a CudaStream,
}

/// What a save's copy cost ([`Pending::land`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SaveCost {
    /// The copy's time on the copy stream, from its first byte to its last:
    /// the device copies and the points' host copy after them.
    pub copy_ms: f64,
    /// The copy calls it took: one a positional store, one a recurrent store.
    pub calls: u64,
    /// The ms the staging pool spent pinning memory during the save: 0, the
    /// pool pins at load ([`Staging::pinned`]) and a save only borrows.
    pub alloc_ms: f64,
    /// The bytes copied off the card.
    pub bytes: u64,
}

/// A state whose copy runs on the copy stream ([`SeqState::save_async`]): its
/// positions, bytes and points are known at once, its rows and stores are read
/// only after [`Pending::land`]. The state is shared ([`Pending::shared`]): a
/// holder reads it through `&self` once it has landed, and a resume of it
/// before then is refused by name.
pub struct Pending<T> {
    state: Arc<SeqState<T>>,
    start: CudaEvent,
    end: Arc<CudaEvent>,
    calls: u64,
    alloc_ms: f64,
}

impl<T> Pending<T> {
    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.state.positions
    }

    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.bytes()
    }

    /// The positions of the points it carries, ascending.
    #[must_use]
    pub fn points(&self) -> Vec<u32> {
        self.state.points()
    }

    /// [`SeqState::keep_point`].
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        self.state.keep_point(n)
    }

    /// Whether the copy has landed; never waits.
    pub fn ready(&self) -> Result<bool, GpuError> {
        Ok(self.state.landed.get().is_some() || self.end.query()?)
    }

    /// Waits for the copy to land and lets the state's bytes be read; what it
    /// cost. The same answer on every call, from any holder of the state.
    pub fn land(&self) -> Result<SaveCost, GpuError> {
        if let Some(&cost) = self.state.landed.get() {
            return Ok(cost);
        }
        self.end.synchronize()?;
        let cost = SaveCost {
            copy_ms: f64::from(self.start.elapsed_ms(&self.end)?),
            calls: self.calls,
            alloc_ms: self.alloc_ms,
            bytes: self.state.bytes() as u64,
        };
        Ok(*self.state.landed.get_or_init(|| cost))
    }

    /// The state, whose copy may still run: a holder of it waits for the
    /// copy ([`Pending::land`]) before a resume reads it.
    #[must_use]
    pub fn shared(&self) -> Arc<SeqState<T>> {
        Arc::clone(&self.state)
    }

    /// The state once it has landed; `None` while the copy may still run.
    #[must_use]
    pub fn state(&self) -> Option<&SeqState<T>> {
        self.state.landed.get().is_some().then_some(&*self.state)
    }

    /// Waits for the copy and hands the state over with its cost.
    pub fn into_state(self) -> Result<(Arc<SeqState<T>>, SaveCost), GpuError> {
        let cost = self.land()?;
        Ok((self.state, cost))
    }
}

/// The copies that read `lent`'s stores into the host arrays, enqueued on
/// `stream`: one 2-D copy a strip of `strips` into the rows at `rows`, and one
/// a recurrent store into the recurrent array at `rec`, in the stores' order.
/// Returns the calls enqueued.
///
/// # Safety
///
/// `rows .. rows + rows_len` and `rec .. rec + rec_len` must be writable
/// memory (pinned, or pageable under a synchronize of `stream`) that nothing
/// reads, writes or frees until the copies have completed.
unsafe fn read_into(
    stream: &CudaStream,
    lent: &Lent<'_, '_, '_>,
    strips: &[Strip],
    (rows, rows_len): (*mut u8, usize),
    (rec, rec_len): (*mut u8, usize),
) -> Result<u64, GpuError> {
    let mut calls = 0;
    for s in strips {
        let (shape, at) = pitched(s);
        // SAFETY: the host span `at ..` lies in the rows (checked by
        // `dtoh_2d` against the length given); the memory's contract is this
        // fn's caller's.
        unsafe {
            dtoh_2d(
                stream,
                &*lent.positional[s.run].buf,
                0,
                rows.add(at),
                rows_len - at,
                shape,
            )?;
        }
        calls += 1;
    }
    let mut off = 0;
    for store in lent.recurrent.iter() {
        let bytes = store.len() * size_of::<f32>();
        let one = Pitched {
            width: bytes,
            rows: 1,
            dev_pitch: bytes,
            host_pitch: bytes,
        };
        // SAFETY: as above for the recurrent array.
        unsafe { dtoh_2d(stream, &**store, 0, rec.add(off), rec_len - off, one)? };
        off += bytes;
        calls += 1;
    }
    Ok(calls)
}

/// The copies that put `strips` of the rows at `rows`, and the recurrent
/// stores of `lent` from `rec` (`stores` bytes of each, in order), onto the
/// card, enqueued on `stream`. Returns the calls enqueued.
///
/// # Safety
///
/// `rows .. rows + rows_len` and `rec .. rec + rec_len` must be readable
/// memory that nothing writes or frees until the copies have completed (a
/// synchronize of `stream`).
unsafe fn put_from(
    stream: &CudaStream,
    lent: &mut Lent<'_, '_, '_>,
    strips: &[Strip],
    (rows, rows_len): (*const u8, usize),
    (rec, rec_len): (*const u8, usize),
) -> Result<u64, GpuError> {
    let mut calls = 0;
    for s in strips {
        let (shape, at) = pitched(s);
        // SAFETY: the host span lies in the rows (checked by `htod_2d`); the
        // memory's contract is this fn's caller's.
        unsafe {
            htod_2d(
                stream,
                &mut *lent.positional[s.run].buf,
                0,
                rows.add(at),
                rows_len - at,
                shape,
            )?;
        }
        calls += 1;
    }
    let mut off = 0;
    for store in lent.recurrent.iter_mut() {
        let bytes = store.len() * size_of::<f32>();
        let one = Pitched {
            width: bytes,
            rows: 1,
            dev_pitch: bytes,
            host_pitch: bytes,
        };
        // SAFETY: as above for the recurrent array.
        unsafe { htod_2d(stream, &mut **store, 0, rec.add(off), rec_len - off, one)? };
        off += bytes;
        calls += 1;
    }
    Ok(calls)
}

/// A strip as the bytes of its 2-D copy, and the byte its host side starts at.
fn pitched(s: &Strip) -> (Pitched, usize) {
    let z = size_of::<u16>();
    (
        Pitched {
            width: s.width * z,
            rows: s.rows,
            dev_pitch: s.dev_pitch * z,
            host_pitch: s.host_pitch * z,
        },
        s.host_off * z,
    )
}

impl<T> SeqState<T> {
    /// The state of `who`'s model standing at `positions`: `lent`'s
    /// positional rows, its recurrent stores (the checkpoints' list, which
    /// hold `positions`), and the stores of each of `points` from `ckpt`,
    /// each position with its host value. Every copy is ordered after what
    /// the stream ran before. Refused by name when the points are not
    /// ascending from above 0 and below `positions`, `ckpt` holds no such
    /// point, or the stores are not the checkpoints'.
    pub fn save(
        stream: &CudaStream,
        who: Identity,
        positions: u32,
        lent: &Lent<'_, '_, '_>,
        current: T,
        points: Vec<(u32, T)>,
        ckpt: &Checkpoints,
    ) -> Result<SeqState<T>, GpuError> {
        let lens = Self::checked(lent, positions, &points, ckpt)?;
        let runs = lent.runs();
        let strips = plan_whole(&runs).map_err(|e| shape(e.to_string()))?;
        let one: usize = lens.iter().sum();
        let mut rows = vec![0u16; runs.iter().map(Run::values).sum()];
        let mut recurrent = vec![0f32; one * (1 + points.len())];
        let (rows_at, rows_len) = (rows.as_mut_ptr().cast::<u8>(), rows.len() * 2);
        let rec_at = recurrent.as_mut_ptr().cast::<u8>();
        // SAFETY: both vectors are neither read nor freed before the
        // synchronize below, which every path takes, whether or not every
        // copy was enqueued; `rec_len` is the current stores' bytes, the
        // first `one` values of the array.
        let sent = unsafe {
            read_into(
                stream,
                lent,
                &strips,
                (rows_at, rows_len),
                (rec_at, one * 4),
            )
        };
        let synced = stream.synchronize();
        sent?;
        synced?;
        for (i, &(at, _)) in points.iter().enumerate() {
            let from = one * (1 + i);
            ckpt.read_point(at, &mut recurrent[from..from + one])?;
        }
        Ok(SeqState {
            who,
            positions,
            host: Host::Owned { rows, recurrent },
            runs,
            lens,
            current,
            points,
            landed: OnceLock::from(SaveCost::default()),
        })
    }

    /// [`SeqState::save`] whose copies run on `lane`'s stream into pinned
    /// staging, after everything `engine` ran so far, and whose call returns
    /// once they are enqueued: the state is a [`Pending`]. What the call
    /// itself does is one copy call a store and one host copy for the points
    /// (`ckpt`'s slots into the staging, queued on the lane's stream after the
    /// device copies: [`Checkpoints::enqueue_points`]), nothing waited for.
    /// The rows it reads are not cut, reset or replaced until
    /// [`Pending::land`], and `ckpt` refuses a take, an open take or an adopt
    /// until then.
    /// [`NoRoom`], nothing enqueued, when the staging holds no free extent
    /// for the state ([`Staging::lend`]): the caller makes room or saves with
    /// [`SeqState::save`].
    pub fn save_async(
        streams: Streams<'_>,
        who: Identity,
        positions: u32,
        lent: &Lent<'_, '_, '_>,
        current: T,
        points: Vec<(u32, T)>,
        ckpt: &Checkpoints,
    ) -> Result<Result<Pending<T>, NoRoom>, GpuError> {
        let lens = Self::checked(lent, positions, &points, ckpt)?;
        let runs = lent.runs();
        let strips = plan_whole(&runs).map_err(|e| shape(e.to_string()))?;
        let one: usize = lens.iter().sum();
        let rows_n: usize = runs.iter().map(Run::values).sum();
        let Streams { lane, engine } = streams;
        let pinned = lane.staging.stats().grown_ms;
        let mut staged = match lane.staging.lend(rows_n, one * (1 + points.len()))? {
            Ok(staged) => staged,
            Err(short) => return Ok(Err(short)),
        };
        // What the lend pinned: 0, the pool pins at load.
        let alloc_ms = lane.staging.stats().grown_ms - pinned;
        let ctx = engine.context();
        let timed = Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let ready = engine.record_event(None)?;
        lane.stream.wait(&ready)?;
        let start = ctx.new_event(timed)?;
        start.record(&lane.stream)?;
        let (rows_at, rows_len) = (staged.rows_ptr(), staged.rows_bytes());
        let rec_at = staged.recurrent_ptr();
        // SAFETY: the extent is the staging's, lent to this state alone; it is
        // neither read nor returned before the copy lands — the end event is
        // held with it below, and every failure path synchronizes the lane's
        // stream first.
        let sent = unsafe {
            read_into(
                &lane.stream,
                lent,
                &strips,
                (rows_at, rows_len),
                (rec_at, one * 4),
            )
        };
        let calls = match sent {
            Ok(n) => n,
            Err(e) => {
                lane.stream.synchronize()?;
                return Err(e);
            }
        };
        if !points.is_empty() {
            let ats: Vec<u32> = points.iter().map(|&(at, _)| at).collect();
            // SAFETY: the points' region of the extent is this state's alone,
            // read or freed only after the end event below has completed (it
            // is held with the extent); the checkpoints refuse a write to the
            // slots until then ([`Checkpoints::guard_points`]) and wait for
            // the context before freeing them.
            let queued = unsafe {
                ckpt.enqueue_points(
                    &lane.stream,
                    &ats,
                    rec_at.add(one * 4).cast::<f32>(),
                    one * points.len(),
                )
            };
            if let Err(e) = queued {
                lane.stream.synchronize()?;
                return Err(e);
            }
        }
        let end = ctx.new_event(timed)?;
        end.record(&lane.stream)?;
        let end = Arc::new(end);
        staged.hold(Arc::clone(&end));
        ckpt.guard_points(Arc::clone(&end));
        Ok(Ok(Pending {
            state: Arc::new(SeqState {
                who,
                positions,
                host: Host::Staged(staged),
                runs,
                lens,
                current,
                points,
                landed: OnceLock::new(),
            }),
            start,
            end,
            calls,
            alloc_ms,
        }))
    }

    /// The recurrent stores' lengths, after the checks both saves share:
    /// refused by name when the stores are not the checkpoints', or the points
    /// do not ascend from above 0 and below `positions`.
    fn checked(
        lent: &Lent<'_, '_, '_>,
        positions: u32,
        points: &[(u32, T)],
        ckpt: &Checkpoints,
    ) -> Result<Vec<usize>, GpuError> {
        let lens = lent.lens();
        if lens != ckpt.lens() {
            return Err(shape(format!(
                "recurrent stores of {lens:?} f32s; the checkpoints copy {:?}",
                ckpt.lens()
            )));
        }
        check_points(positions, points.iter().map(|&(p, _)| p))?;
        Ok(lens)
    }

    /// The state put back on `who`'s model: `lent`'s positional rows and
    /// recurrent stores from the host, each carried point adopted by `ckpt`;
    /// the host values of the current position and of each point returned.
    /// Refused by name, nothing copied, when `who` is another model's, the
    /// runs or the stores are another layout, or `ckpt` holds a point.
    pub fn load(
        &self,
        stream: &CudaStream,
        who: &Identity,
        lent: &mut Lent<'_, '_, '_>,
        ckpt: &mut Checkpoints,
    ) -> Result<Hosts<'_, T>, GpuError> {
        let r = self.resume(stream, who, lent, ckpt, None)?;
        Ok((r.current, r.points))
    }

    /// The prefix of the state a model can be cut to at most `n` put back on
    /// `who`'s model: `k` = [`SeqState::keep_point`] of `n` positions of every
    /// positional store's rows (no row of a position past `k` moves), the
    /// recurrent stores as they stood at `k`, each carried point up to `k`
    /// adopted by `ckpt`. A model that stands at `k` after it is what a load
    /// and a cut to `k` leave, for the stores and the checkpoints alike.
    /// Refused by name, nothing copied, as [`SeqState::load`] refuses, and for
    /// a prefix of no positions (the model is reset instead) and for a state
    /// with a positional store of no growth rule ([`Spans::planes`]) put back
    /// short of whole.
    pub fn load_prefix(
        &self,
        stream: &CudaStream,
        who: &Identity,
        lent: &mut Lent<'_, '_, '_>,
        ckpt: &mut Checkpoints,
        n: u32,
    ) -> Result<Resumed<'_, T>, GpuError> {
        self.resume(stream, who, lent, ckpt, Some(n))
    }

    fn resume(
        &self,
        stream: &CudaStream,
        who: &Identity,
        lent: &mut Lent<'_, '_, '_>,
        ckpt: &mut Checkpoints,
        prefix: Option<u32>,
    ) -> Result<Resumed<'_, T>, GpuError> {
        if *who != self.who {
            return Err(shape(format!("a state of {} put back on {who}", self.who)));
        }
        if self.landed.get().is_none() {
            return Err(shape(
                "a state whose copy is still running put back: wait for the save to land first",
            ));
        }
        let spans = lent.spans();
        let mine: Vec<usize> = self
            .runs
            .iter()
            .flat_map(|r| (0..r.planes).map(|_| r.len))
            .collect();
        if spans != mine {
            return Err(shape(format!(
                "a state of {} positional runs ({} values) put back into {} ({} values)",
                mine.len(),
                mine.iter().sum::<usize>(),
                spans.len(),
                spans.iter().sum::<usize>()
            )));
        }
        let lens = lent.lens();
        if lens != self.lens || lens != ckpt.lens() {
            return Err(shape(format!(
                "a state of recurrent stores of {:?} f32s put back into {lens:?} (the \
                 checkpoints copy {:?})",
                self.lens,
                ckpt.lens()
            )));
        }
        if !ckpt.positions().is_empty() || ckpt.pending() {
            return Err(shape(format!(
                "a state put back beside checkpoints at {:?}: resume onto an empty model",
                ckpt.positions()
            )));
        }
        let at = prefix.map_or(self.positions, |n| self.keep_point(n));
        if at == 0 {
            return Err(shape(format!(
                "a prefix of 0 positions (asked {prefix:?}): reset the model instead"
            )));
        }
        let whole = at == self.positions;
        let strips = if whole {
            plan_whole(&self.runs)
        } else {
            plan_prefix(&self.runs, at as usize)
        }
        .map_err(|e| shape(e.to_string()))?;
        // The recurrent stores as they stood at `at`: the state's own, or the
        // point's.
        let one: usize = self.lens.iter().sum();
        let (from, current) = if whole {
            (0, &self.current)
        } else {
            let i = self
                .points
                .iter()
                .position(|&(p, _)| p == at)
                .expect("keep_point names the state's position or a point");
            (one * (1 + i), &self.points[i].1)
        };
        let rows = self.host.rows();
        let recurrent = &self.host.recurrent()[from..from + one];
        // SAFETY: the host arrays are the state's own, which no copy writes
        // (a state is read only once its save has landed) and which outlive
        // the call; the synchronize below ends every copy first, on every
        // path.
        let sent = unsafe {
            put_from(
                stream,
                lent,
                &strips,
                (rows.as_ptr().cast(), rows.len() * 2),
                (recurrent.as_ptr().cast(), one * 4),
            )
        };
        let synced = stream.synchronize();
        let calls = sent?;
        synced?;
        let bytes = strips.iter().map(Strip::values).sum::<usize>() * size_of::<u16>()
            + one * size_of::<f32>();
        let adopted = self.points.partition_point(|&(p, _)| p <= at);
        for (i, &(p, _)) in self.points[..adopted].iter().enumerate() {
            let from = one * (1 + i);
            ckpt.adopt(p, &self.host.recurrent()[from..from + one])?;
        }
        Ok(Resumed {
            at,
            current,
            points: &self.points[..adopted],
            bytes: bytes as u64,
            calls,
        })
    }

    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.positions
    }

    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        size_of_val(self.host.rows()) + size_of_val(self.host.recurrent())
    }

    /// The positions of the points it carries, ascending.
    #[must_use]
    pub fn points(&self) -> Vec<u32> {
        self.points.iter().map(|&(p, _)| p).collect()
    }

    /// The longest prefix of at most `n` positions a model holds right after
    /// the state is put back: every position when `n` reaches them, else the
    /// nearest carried point at or below `n`, else none.
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        if n >= self.positions {
            return self.positions;
        }
        self.points
            .iter()
            .rev()
            .map(|&(p, _)| p)
            .find(|&p| p <= n)
            .unwrap_or(0)
    }
}

/// Refused by name unless `points` ascend, each above 0 and below
/// `positions`.
fn check_points(positions: u32, points: impl Iterator<Item = u32>) -> Result<(), GpuError> {
    let mut last = 0;
    for p in points {
        if p <= last || p >= positions {
            return Err(shape(format!(
                "a point at {p} after {last} in a state of {positions} positions: the points \
                 ascend from above 0 and lie below the positions"
            )));
        }
        last = p;
    }
    Ok(())
}

fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::hold;

    /// A state of 5 positions carries rows 0..5 of a two-plane store, the
    /// recurrent stores at 5 and the point at 3; put back on an empty model it
    /// restores the rows and the stores, the checkpoints hold exactly the
    /// point, and the host values come back. A state put back on another
    /// model's identity is refused by name, nothing copied.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_state_goes_and_comes_back() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let who = Identity {
            arch: "test",
            file: PathBuf::from("/x"),
            card: "card".to_owned(),
            ctx: 8,
        };
        let mut c = Checkpoints::new(&ctx, vec![4], 1 << 20, 512).expect("checkpoints");
        // Two planes of 8 positions, a row of 2 values.
        let plane: Vec<u16> = (0..32).collect();
        let mut kv = DeviceBuffer::from_host(&stream, &plane).expect("a store");
        let mut rec = DeviceBuffer::from_host(&stream, &[3.0f32; 4]).expect("a store");
        assert!(matches!(
            c.take(&stream, 3, &mut [&mut rec]).expect("a take at 3"),
            runtime::seqstate::Take::Copy { .. }
        ));
        rec.copy_from_host(&stream, &[5.0; 4]).expect("rec");
        let s = {
            let mut spans = [Spans::planes(&mut kv, 2, 16, 10).expect("spans")];
            let lent = Lent {
                positional: &mut spans,
                recurrent: &mut [&mut rec],
            };
            SeqState::save(&stream, who.clone(), 5, &lent, 'c', vec![(3, 'p')], &c).expect("a save")
        };
        assert_eq!(s.bytes(), 2 * 10 * 2 + 2 * 16);
        assert_eq!(
            (s.keep_point(4), s.keep_point(2), s.keep_point(9)),
            (3, 0, 5)
        );
        let mut d = Checkpoints::new(&ctx, vec![4], 1 << 20, 512).expect("checkpoints");
        kv.copy_from_host(&stream, &[9u16; 32]).expect("kv");
        rec.copy_from_host(&stream, &[0.0; 4]).expect("rec");
        let other = Identity {
            ctx: 16,
            ..who.clone()
        };
        {
            let mut spans = [Spans::planes(&mut kv, 2, 16, 10).expect("spans")];
            let mut lent = Lent {
                positional: &mut spans,
                recurrent: &mut [&mut rec],
            };
            let err = s
                .load(&stream, &other, &mut lent, &mut d)
                .expect_err("another model's identity refused");
            assert!(
                err.to_string()
                    .contains("put back on test /x on card at 16"),
                "{err}"
            );
        }
        assert_eq!(rec.to_host_vec(&stream).expect("rec"), [0.0; 4]);
        let (cur, pts) = {
            let mut spans = [Spans::planes(&mut kv, 2, 16, 10).expect("spans")];
            let mut lent = Lent {
                positional: &mut spans,
                recurrent: &mut [&mut rec],
            };
            let (cur, pts) = s.load(&stream, &who, &mut lent, &mut d).expect("a load");
            (*cur, pts.to_vec())
        };
        assert_eq!((cur, pts), ('c', vec![(3, 'p')]));
        let back = kv.to_host_vec(&stream).expect("kv");
        assert_eq!(&back[..10], &plane[..10]);
        assert_eq!(&back[16..26], &plane[16..26]);
        assert_eq!(back[10], 9, "a row past the positions is not carried");
        assert_eq!(rec.to_host_vec(&stream).expect("rec"), [5.0; 4]);
        assert_eq!(d.positions(), [3]);
        d.cut(3, 5).expect("a cut to the adopted point");
        d.apply(&stream, &mut [&mut rec]).expect("its restore");
        assert_eq!(rec.to_host_vec(&stream).expect("rec"), [3.0; 4]);
    }

    // ----------------------------------------------------- the synthetic model

    const POS: u32 = 40;
    const CTX: usize = 64;
    /// Values of a K or V row a position, planes (heads), the raw keys' row,
    /// the pool's positions and the values of a pooled key.
    const H: usize = 8;
    const PLANES: usize = 4;
    const RAW: usize = 4;
    const POOL: usize = 8;
    const PV: usize = 2;
    const POISON: u16 = 0xBEEF;
    const LENS: [usize; 2] = [4, 3];

    fn who() -> Identity {
        Identity {
            arch: "test",
            file: PathBuf::from("/x"),
            card: "card".to_owned(),
            ctx: CTX,
        }
    }

    /// One model's buffers: K and V of four planes, the raw keys, the pooled
    /// keys, two recurrent stores and the checkpoints.
    struct Model {
        k: DeviceBuffer<u16>,
        v: DeviceBuffer<u16>,
        raw: DeviceBuffer<u16>,
        pooled: DeviceBuffer<u16>,
        rec: [DeviceBuffer<f32>; 2],
        ckpt: Checkpoints,
    }

    impl Model {
        /// Every row `fill`: `Some` patterns (each value names its buffer and
        /// place), `None` the poison.
        fn new(ctx: &Arc<CudaContext>, stream: &CudaStream, pattern: bool) -> Model {
            let mk = |id: u16, n: usize| {
                let host: Vec<u16> = (0..n)
                    .map(|i| {
                        if pattern {
                            (id << 12) | i as u16
                        } else {
                            POISON
                        }
                    })
                    .collect();
                DeviceBuffer::from_host(stream, &host).expect("a store")
            };
            Model {
                k: mk(1, PLANES * CTX * H),
                v: mk(2, PLANES * CTX * H),
                raw: mk(3, CTX * RAW),
                pooled: mk(4, CTX / POOL * PV),
                rec: [
                    DeviceBuffer::from_host(stream, &[0.0f32; LENS[0]]).expect("rec"),
                    DeviceBuffer::from_host(stream, &[0.0f32; LENS[1]]).expect("rec"),
                ],
                ckpt: Checkpoints::new(ctx, LENS.to_vec(), 1 << 20, 512).expect("checkpoints"),
            }
        }

        fn fill_rec(&mut self, stream: &CudaStream, v: f32) {
            self.rec[0]
                .copy_from_host(stream, &[v; LENS[0]])
                .expect("rec");
            self.rec[1]
                .copy_from_host(stream, &[v; LENS[1]])
                .expect("rec");
        }
    }

    /// Runs `f` with `m`'s stores lent.
    fn lend<R>(
        m: &mut Model,
        n: usize,
        f: impl FnOnce(&mut Lent<'_, '_, '_>, &mut Checkpoints) -> R,
    ) -> R {
        let Model {
            k,
            v,
            raw,
            pooled,
            rec,
            ckpt,
        } = m;
        let stride = CTX * H;
        let mut spans = [
            Spans::by_position(k, PLANES, stride, n, H).expect("spans"),
            Spans::by_position(v, PLANES, stride, n, H).expect("spans"),
            Spans::by_position(raw, 1, 0, n, RAW).expect("spans"),
            Spans::by_pool(pooled, 1, 0, n, (POOL, PV)).expect("spans"),
        ];
        let [r0, r1] = rec;
        let mut lent = Lent {
            positional: &mut spans,
            recurrent: &mut [r0, r1],
        };
        f(&mut lent, ckpt)
    }

    /// A source model standing at [`POS`]: the pattern in every row, the
    /// recurrent stores 3.0 now, 2.0 at the point 32 and 1.0 at 16.
    fn source(ctx: &Arc<CudaContext>, stream: &CudaStream) -> Model {
        let mut m = Model::new(ctx, stream, true);
        for (at, v) in [(16, 1.0), (32, 2.0)] {
            m.fill_rec(stream, v);
            let Model { rec, ckpt, .. } = &mut m;
            let [r0, r1] = rec;
            ckpt.take(stream, at, &mut [r0, r1]).expect("a take");
        }
        m.fill_rec(stream, 3.0);
        m
    }

    /// The first `n` positions' values of every store of `m`, in order.
    fn rows_below(m: &Model, stream: &CudaStream, n: usize) -> Vec<u16> {
        let mut out = Vec::new();
        for buf in [&m.k, &m.v] {
            let host = buf.to_host_vec(stream).expect("rows");
            for p in 0..PLANES {
                out.extend_from_slice(&host[p * CTX * H..p * CTX * H + n * H]);
            }
        }
        out.extend_from_slice(&m.raw.to_host_vec(stream).expect("raw")[..n * RAW]);
        out.extend_from_slice(
            &m.pooled.to_host_vec(stream).expect("pooled")[..n.div_ceil(POOL) * PV],
        );
        out
    }

    /// The bytes of a resume of `n` positions: worked out from the layout, not
    /// from the code under test.
    fn bytes_of(n: usize) -> u64 {
        let rows = 2 * PLANES * n * H + n * RAW + n.div_ceil(POOL) * PV;
        (rows * 2 + (LENS[0] + LENS[1]) * 4) as u64
    }

    /// `s` as a sync save and as a save off the request path, each from a
    /// source model; the second one landed.
    fn both_saves(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        lane: &SaveLane,
    ) -> (SeqState<char>, Arc<SeqState<char>>, SaveCost) {
        let mut src = source(ctx, stream);
        let points = vec![(16, 'p'), (32, 'q')];
        let sync = lend(&mut src, POS as usize, |lent, ckpt| {
            SeqState::save(stream, who(), POS, lent, 'c', points.clone(), ckpt).expect("a save")
        });
        let pending = lend(&mut src, POS as usize, |lent, ckpt| {
            let streams = Streams {
                lane,
                engine: stream,
            };
            SeqState::save_async(streams, who(), POS, lent, 'c', points.clone(), ckpt)
                .expect("a save off the path")
                .expect("staging room")
        });
        let (state, cost) = pending.into_state().expect("a landed copy");
        (sync, state, cost)
    }

    /// A save off the request path is the sync save's state, bit for bit, in
    /// one copy call a store (the four stores and the two recurrent ones,
    /// where plane by plane takes 12), and the pool lends the second save the
    /// memory the first one's state gave back.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_an_async_save_is_the_sync_saves_state_in_one_call_a_store() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let lane = SaveLane::new(&ctx, 8 << 20).expect("a lane");
        let (sync, state, cost) = both_saves(&ctx, &stream, &lane);
        assert_eq!(sync.host.rows(), state.host.rows());
        assert_eq!(sync.host.recurrent(), state.host.recurrent());
        assert_eq!(sync.bytes(), state.bytes());
        assert_eq!((cost.calls, cost.bytes), (6, sync.bytes() as u64));
        let by_plane = (2 * PLANES + 1 + 1) + 2;
        assert!(
            cost.calls < by_plane as u64,
            "{} calls of {by_plane}",
            cost.calls
        );
        assert_eq!(
            cost.alloc_ms, 0.0,
            "a save pins nothing: the pool pinned at load"
        );
        drop(state);
        let (_, state2, cost2) = both_saves(&ctx, &stream, &lane);
        assert_eq!(cost2.alloc_ms, 0.0, "the pool had the memory");
        assert_eq!(lane.staging().stats().grown, 1, "one slab for both");
        assert_eq!(state2.host.rows(), sync.host.rows());
    }

    /// A prefix of `n` positions leaves the card as a load and a cut to its
    /// point leave it — every row below the point, the recurrent stores, the
    /// points the checkpoints hold, the host values — and moves the bytes of
    /// those positions only: a row past the point is still the poison.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_prefix_load_is_a_load_and_a_cut_and_moves_only_its_positions() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let lane = SaveLane::new(&ctx, 8 << 20).expect("a lane");
        let (state, _, _) = both_saves(&ctx, &stream, &lane);
        // Asked 20 keeps the point 16; asked 35 the point 32; asked 40 all.
        for (asked, k, rec, cur, points) in [
            (20u32, 16u32, 1.0f32, 'p', vec![(16u32, 'p')]),
            (35, 32, 2.0, 'q', vec![(16, 'p'), (32, 'q')]),
            (40, 40, 3.0, 'c', vec![(16, 'p'), (32, 'q')]),
        ] {
            assert_eq!(state.keep_point(asked), k);
            let mut cut = Model::new(&ctx, &stream, false);
            lend(&mut cut, POS as usize, |lent, ckpt| {
                state.load(&stream, &who(), lent, ckpt).expect("a load");
                if k < POS {
                    ckpt.cut(k, POS).expect("a cut");
                    let [r0, r1] = &mut lent.recurrent else {
                        unreachable!()
                    };
                    ckpt.apply(&stream, &mut [r0, r1]).expect("its restore");
                }
            });
            let mut pre = Model::new(&ctx, &stream, false);
            let got = lend(&mut pre, POS as usize, |lent, ckpt| {
                let r = state
                    .load_prefix(&stream, &who(), lent, ckpt, asked)
                    .expect("a prefix load");
                (r.at, *r.current, r.points.to_vec(), r.bytes, r.calls)
            });
            assert_eq!((got.0, got.1, got.2), (k, cur, points), "asked {asked}");
            assert_eq!(got.3, bytes_of(k as usize), "the bytes of {k} positions");
            assert_eq!(got.4, 6, "one call a store");
            let n = k as usize;
            assert_eq!(
                rows_below(&pre, &stream, n),
                rows_below(&cut, &stream, n),
                "rows below {k}"
            );
            assert_eq!(
                rows_below(&pre, &stream, n),
                rows_below(&source(&ctx, &stream), &stream, n)
            );
            for i in 0..2 {
                let (a, b) = (
                    pre.rec[i].to_host_vec(&stream).expect("rec"),
                    cut.rec[i].to_host_vec(&stream).expect("rec"),
                );
                assert_eq!(a, b, "store {i} at {k}");
                assert_eq!(a, vec![rec; LENS[i]]);
            }
            assert_eq!(pre.ckpt.positions(), cut.ckpt.positions(), "points at {k}");
            // Nothing past the point moved: the K plane's row k is the poison.
            if k < POS {
                let kb = pre.k.to_host_vec(&stream).expect("k");
                for p in 0..PLANES {
                    assert_eq!(
                        kb[p * CTX * H + n * H],
                        POISON,
                        "plane {p} row {k} uploaded"
                    );
                }
                let pooled = pre.pooled.to_host_vec(&stream).expect("pooled");
                assert_eq!(
                    pooled[n.div_ceil(POOL) * PV],
                    POISON,
                    "a pool past {k} uploaded"
                );
            }
        }
    }

    /// Refused by name, nothing copied: a prefix of no positions, a state
    /// whose store has no growth rule put back short, and a state that does
    /// not fit the staging pool.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_prefix_load_and_a_save_refuse_by_name() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let lane = SaveLane::new(&ctx, 8 << 20).expect("a lane");
        let (state, _, _) = both_saves(&ctx, &stream, &lane);
        let mut m = Model::new(&ctx, &stream, false);
        let err = lend(&mut m, POS as usize, |lent, ckpt| {
            state
                .load_prefix(&stream, &who(), lent, ckpt, 10)
                .err()
                .expect("a prefix below the first point")
        });
        assert!(err.to_string().contains("a prefix of 0 positions"), "{err}");
        assert_eq!(
            rows_below(&m, &stream, 1),
            rows_below(&Model::new(&ctx, &stream, false), &stream, 1)
        );
        // A pool of 1 MiB cannot stage 64 MiB: no room, nothing enqueued.
        let small = SaveLane::new(&ctx, 1 << 20).expect("a lane");
        let mut src = source(&ctx, &stream);
        let short = lend(&mut src, POS as usize, |lent, ckpt| {
            let streams = Streams {
                lane: &small,
                engine: &stream,
            };
            SeqState::save_async(streams, who(), POS, lent, 'c', vec![(16, 'p')], ckpt)
                .expect("a save")
                .err()
        });
        let short = short.expect("a state past the pool is no save off the path");
        assert!(
            short.need > short.held && !short.fragmented(),
            "a state over the whole pool is full, not fragmented: {short:?}"
        );
    }

    /// A Qwen3-30B-shaped state (48 layers' K and V: 96 stores of 4 planes of
    /// 128 values a position, 4096 of 8192 positions) begins its copy in a
    /// call that returns while the copy still runs, takes one copy call a
    /// store, and moves the bytes of those positions; what the copy and the
    /// call cost, alone and beside the same bytes going the other way, is
    /// printed (an observation, not a timing: no lease holds the machine).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_qwen3_shaped_state_begins_in_a_call_and_copies_in_one_call_a_store() {
        use std::time::Instant;
        const STORES: usize = 96;
        const HEAD: usize = 128;
        const MAX: usize = 8192;
        const AT: usize = 4096;
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let lane = SaveLane::new(&ctx, 1 << 30).expect("a lane");
        let stores = || -> Vec<DeviceBuffer<u16>> {
            (0..STORES)
                .map(|_| DeviceBuffer::zeroed(&stream, PLANES * MAX * HEAD).expect("a store"))
                .collect()
        };
        fn spans(bufs: &mut [DeviceBuffer<u16>]) -> Vec<Spans<'_>> {
            bufs.iter_mut()
                .map(|b| Spans::by_position(b, PLANES, MAX * HEAD, AT, HEAD).expect("spans"))
                .collect()
        }
        let mut from = stores();
        let mut from_spans = spans(&mut from);
        let mut rec = DeviceBuffer::<f32>::zeroed(&stream, 4).expect("a recurrent store");
        let ckpt = Checkpoints::new(&ctx, vec![4], 1 << 20, 512).expect("checkpoints");
        let lent = Lent {
            positional: &mut from_spans,
            recurrent: &mut [&mut rec],
        };
        let at = |lent: &Lent<'_, '_, '_>| {
            let t = Instant::now();
            let streams = Streams {
                lane: &lane,
                engine: &stream,
            };
            let p = SeqState::save_async(streams, who(), AT as u32, lent, 'c', Vec::new(), &ckpt)
                .expect("a save off the path")
                .expect("staging room");
            (p, t.elapsed().as_secs_f64() * 1e3)
        };
        // The engine stream is held shut: the save's copies wait behind it, so
        // they are in flight when the call returns.
        let hold = hold::Hold::on(&stream).expect("a hold");
        let (pending, begin_ms) = at(&lent);
        let ready = pending.ready();
        hold.open();
        let in_flight = !ready.expect("a query");
        let (state, alone) = pending.into_state().expect("a landed copy");
        let want = (STORES * PLANES * AT * HEAD * 2 + 4 * 4) as u64;
        assert!(
            in_flight,
            "the copy had landed when its call returned ({begin_ms:.3} ms)"
        );
        assert_eq!((alone.calls, alone.bytes), (STORES as u64 + 1, want));
        assert_eq!(state.bytes() as u64, want);
        // The same bytes going back, alone, then with a save running beside it.
        let mut to = stores();
        let mut to_spans = spans(&mut to);
        let mut to_rec = DeviceBuffer::<f32>::zeroed(&stream, 4).expect("a recurrent store");
        let mut to_ckpt = Checkpoints::new(&ctx, vec![4], 1 << 20, 512).expect("checkpoints");
        let mut onto = Lent {
            positional: &mut to_spans,
            recurrent: &mut [&mut to_rec],
        };
        let up = |onto: &mut Lent<'_, '_, '_>, ckpt: &mut Checkpoints| {
            let t = Instant::now();
            state.load(&stream, &who(), onto, ckpt).expect("a load");
            t.elapsed().as_secs_f64() * 1e3
        };
        let up_alone = up(&mut onto, &mut to_ckpt);
        let (beside, _) = at(&lent);
        let up_beside = up(&mut onto, &mut to_ckpt);
        let (beside_state, down_beside) = beside.into_state().expect("a landed copy");
        // The pool holds the memory now: a save asks it for nothing, and its
        // call still returns while the copy runs.
        drop(beside_state);
        let hold = hold::Hold::on(&stream).expect("a hold");
        let (warm, begin_warm) = at(&lent);
        let warm_ready = warm.ready();
        hold.open();
        let warm_in_flight = !warm_ready.expect("a query");
        let (_, warm_cost) = warm.into_state().expect("a landed copy");
        assert!(
            warm_in_flight,
            "the copy had landed when its call returned ({begin_warm:.3} ms)"
        );
        assert_eq!(warm_cost.alloc_ms, 0.0, "the pool lent the memory it held");
        let gb = |ms: f64| want as f64 / ms / 1e6;
        eprintln!(
            "qwen3-shaped state {want} bytes, {} calls (plane by plane {}): begin {begin_ms:.3} ms \
             first, {begin_warm:.3} ms warm (pinned at load {:.1} ms); d2h alone {:.2} ms ({:.1} GB/s), h2d alone {up_alone:.2} ms ({:.1} GB/s); \
             together d2h {:.2} ms ({:.1} GB/s), h2d {up_beside:.2} ms ({:.1} GB/s)",
            alone.calls,
            STORES * PLANES + 1,
            lane.staging().stats().grown_ms,
            alone.copy_ms,
            gb(alone.copy_ms),
            gb(up_alone),
            down_beside.copy_ms,
            gb(down_beside.copy_ms),
            gb(up_beside),
        );
    }
    /// A save's host copy of a point runs on the lane, not in its call. With
    /// the engine stream held shut, the call returns while the point's bytes
    /// are not yet in the extent (a copy on the calling thread has written
    /// them by then), and within a bound derived from its enqueues: the one
    /// strip, the one recurrent store, the host function, three events and a
    /// stream wait are eight driver calls of tens of microseconds at most, so
    /// the bound is a few ms, under what a core takes to copy 180 MiB even at
    /// its fastest. Until the save lands the checkpoints refuse a write to the
    /// slot by name; after, the bytes are whole and the slot is free.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_points_host_copy_runs_on_the_lane_and_its_slot_waits_for_it() {
        use std::time::Instant;
        const ONE: usize = 45 << 20;
        const AT: usize = 16;
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let lane = SaveLane::new(&ctx, 1 << 30).expect("a lane");
        let mut pos = DeviceBuffer::<u16>::zeroed(&stream, PLANES * 64 * 8).expect("a store");
        let mut spans = [Spans::by_position(&mut pos, PLANES, 64 * 8, AT, 8).expect("spans")];
        let mut rec = DeviceBuffer::<f32>::zeroed(&stream, ONE).expect("a recurrent store");
        let mut ckpt =
            Checkpoints::new(&ctx, vec![ONE], 2 * 4 * ONE as u64, 512).expect("checkpoints");
        ckpt.adopt(8, &vec![1.0f32; ONE]).expect("a point");
        let lent = Lent {
            positional: &mut spans,
            recurrent: &mut [&mut rec],
        };
        let mut nothing = Lent {
            positional: &mut [],
            recurrent: &mut [],
        };
        let mut empty =
            Checkpoints::new(&ctx, vec![ONE], 4 * ONE as u64, 512).expect("checkpoints");
        let hold = hold::Hold::on(&stream).expect("a hold");
        let t = Instant::now();
        let streams = Streams {
            lane: &lane,
            engine: &stream,
        };
        let pending =
            SeqState::save_async(streams, who(), AT as u32, &lent, 'c', vec![(8, 'p')], &ckpt)
                .expect("a save off the path")
                .expect("staging room");
        let begin_ms = t.elapsed().as_secs_f64() * 1e3;
        // What the held stream shows is read first and judged after it opens:
        // a failed assert with the stream shut would drop `pending`, whose
        // extent waits for a copy that cannot run.
        let Host::Staged(staged) = &pending.state.host else {
            hold.open();
            panic!("an async save is staged");
        };
        let point_still_zero = staged.recurrent()[ONE..].iter().all(|&v| v == 0.0);
        let ready = pending.ready();
        let refused = ckpt.adopt(12, &vec![2.0f32; ONE]);
        // A holder of the shared state is refused by name until the copy lands.
        let shared = pending.shared();
        let early = shared.load(&stream, &who(), &mut nothing, &mut empty);
        hold.open();
        assert!(begin_ms < 4.0, "the call took {begin_ms:.3} ms");
        assert!(
            point_still_zero,
            "the point's bytes were in the extent when the call returned ({begin_ms:.3} ms)"
        );
        assert!(!ready.expect("a query"), "the copy had landed");
        assert!(
            matches!(&refused, Err(GpuError::Shape { detail, .. }) if detail.contains("save in flight")),
            "a slot written under the point copy: {:?}",
            refused.err().map(|e| e.to_string())
        );
        assert!(
            matches!(&early, Err(GpuError::Shape { detail, .. }) if detail.contains("still running")),
            "a state resumed while its copy ran: {:?}",
            early.err().map(|e| e.to_string())
        );
        let (state, cost) = pending.into_state().expect("a landed copy");
        assert_eq!(
            shared.landed.get().copied(),
            Some(cost),
            "the holder sees the landing"
        );
        assert!(
            state.host.recurrent()[ONE..].iter().all(|&v| v == 1.0),
            "the point's bytes did not arrive whole"
        );
        ckpt.adopt(12, &vec![2.0f32; ONE])
            .expect("the slot is free once the save landed");
        eprintln!(
            "point save: begin {begin_ms:.3} ms, copy {:.2} ms",
            cost.copy_ms
        );
    }
}
