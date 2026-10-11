//! Host copies of a body's recurrent stores at chosen positions: the device
//! half of `runtime::seqstate`. A body hands in its copied stores
//! (`runtime::state::copied`) as one list, always in the same order, and
//! this owns the pinned host slots, the ledger's rules and the copies.
//!
//! Every copy runs on the engine stream and is waited for before the call
//! returns: the device is busy with it either way, and no pinned slot is
//! ever read, reused or freed under a copy in flight. A cut is planned when
//! the model is cut and carried out at the body's next device call
//! ([`Checkpoints::apply`]), since a cut has no stream; a take carries out
//! a waiting cut first.
//!
//! A prompt group's inner marks are taken store by store: a take opened
//! before the group ([`Checkpoints::open_take`]) gets each store's copy
//! enqueued where that store holds the mark's state ([`Checkpoints::copy`]),
//! and is sealed once the group's walk is waited for and found sound
//! ([`Checkpoints::seal`]) or abandoned ([`Checkpoints::abandon`]). Its
//! copies are in flight between the open and the seal or abandon, and its
//! slot is the open take's alone until then.
//!
//! A slot is made (pinned and zeroed) the first time a checkpoint needs it,
//! and kept for the load. Each slot records the position its last finished
//! copy holds, apart from the ledger: a restore reads a slot only as that
//! position ([`seqstate::restorable`]), so no ledger point can hand back a
//! slot no copy filled.
//!
//! A sequence state on the host ([`saved`]) carries the stores of some of
//! these points away and hands them back: [`Checkpoints::read_point`] copies
//! a point's slot out, [`Checkpoints::adopt`] makes a point of host bytes.
//! A save off the request path has the copy-out run on its copy stream
//! ([`Checkpoints::enqueue_points`]): the slots it reads are not written
//! ([`Checkpoints::take`], [`Checkpoints::open_take`], [`Checkpoints::adopt`])
//! or freed before it lands, and a write that reaches one is refused by name.

pub mod saved;
pub mod staging;

use std::ffi::c_void;
use std::sync::{Arc, Mutex, PoisonError};

use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer, sys};
use runtime::seqstate::{self, Cut, Kept, Ledger, Price, Stats, Take};

use crate::GpuError;
use crate::graph::cu;

const WHAT: &str = "checkpoint";

/// One slot: a pinned buffer per store, and the position its last finished
/// copy holds — none when made, and none again after a copy that failed.
struct Slot {
    bufs: Vec<PinnedHostBuffer<f32>>,
    holds: Option<u32>,
}

/// A take opened inside a prompt group ([`Checkpoints::open_take`]): the
/// mark it is of, its slot, and which stores' copies are enqueued. Sealed or
/// abandoned, never dropped: a take neither seals is a slot holding nothing
/// that no restore reads.
#[must_use]
#[derive(Debug)]
pub struct Pending {
    at: u32,
    slot: usize,
    sent: Vec<bool>,
}

impl Pending {
    /// The mark the take is of.
    #[must_use]
    pub fn at(&self) -> u32 {
        self.at
    }
}

/// One model's checkpoints: the ledger, the stores' lengths, the slots.
pub struct Checkpoints {
    ledger: Ledger,
    /// f32s of each copied store, in the body's order.
    lens: Vec<usize>,
    slots: Vec<Slot>,
    /// The ledger's points and the slot each fills, ascending by position:
    /// the ledger's own map, kept beside it from what its calls return.
    points: Vec<(u32, usize)>,
    /// Takes opened and neither sealed nor abandoned: copies may be in
    /// flight into their slots.
    open: usize,
    /// The spacing of a prompt call's inner checkpoints (`seqstate::marks`).
    every: u32,
    /// The ends of the saves whose copy of a point's slots may still run
    /// ([`Checkpoints::enqueue_points`]): no slot is written while one has
    /// not completed.
    reading: Mutex<Vec<Arc<CudaEvent>>>,
    ctx: Arc<CudaContext>,
}

/// The host bytes one checkpoint of stores of `lens` f32s copies.
#[must_use]
pub fn point_bytes(lens: &[usize]) -> u64 {
    lens.iter().map(|&n| (n * size_of::<f32>()) as u64).sum()
}

/// The price of checkpoints of stores of `lens` f32s taken inside a prompt
/// call every `every` positions: what a family's body makes its sequences'
/// `Checkpoints` with ([`Checkpoints::sequence`]) and what its plan reserves
/// the host for.
#[must_use]
pub fn price(lens: &[usize], every: u32) -> Price {
    Price {
        point_bytes: point_bytes(lens),
        every,
    }
}

/// [`Price::budget`] of `price` at `ctx` positions, refused by name as the
/// rule refuses: the one figure a sequence's `Checkpoints` is made with and
/// a plan reserves a sequence.
pub fn budget(price: Price, ctx: u64) -> Result<u64, GpuError> {
    price.budget(ctx).map_err(|e| GpuError::Shape {
        what: WHAT,
        detail: format!("the budget at {ctx} positions: {e}"),
    })
}

impl Checkpoints {
    /// One sequence's checkpoints of stores of `lens` f32s in a load of
    /// `ctx` positions, taken inside a prompt call every `every` positions:
    /// held to the sequence's budget ([`budget`] of its [`price`]), refused
    /// by name when the ledger made is not exactly that many slots. The one
    /// constructor: no body names a budget of its own.
    pub fn sequence(
        cuda: &Arc<CudaContext>,
        lens: Vec<usize>,
        ctx: usize,
        every: u32,
    ) -> Result<Checkpoints, GpuError> {
        let positions = u64::try_from(ctx).map_err(|_| GpuError::Shape {
            what: WHAT,
            detail: format!("a context of {ctx} positions passes u64"),
        })?;
        let granted = budget(price(&lens, every), positions)?;
        let ckpt = Checkpoints::new(cuda, lens, granted, every)?;
        seqstate::check_ledger(granted, ckpt.capacity(), ckpt.bytes()).map_err(|e| {
            GpuError::Shape {
                what: WHAT,
                detail: format!("at {ctx} positions: {e}"),
            }
        })?;
        Ok(ckpt)
    }

    /// Checkpoints of stores of `lens` f32s, as many as `budget` bytes of
    /// host slots hold, taken inside a prompt call every `every` positions.
    /// Refused by name when not one fits.
    fn new(
        ctx: &Arc<CudaContext>,
        lens: Vec<usize>,
        budget: u64,
        every: u32,
    ) -> Result<Checkpoints, GpuError> {
        let cap = seqstate::capacity(budget, point_bytes(&lens)).map_err(refused)?;
        Ok(Checkpoints {
            ledger: Ledger::new(cap).map_err(refused)?,
            lens,
            slots: Vec::new(),
            points: Vec::new(),
            open: 0,
            every,
            reading: Mutex::new(Vec::new()),
            ctx: Arc::clone(ctx),
        })
    }

    /// f32s of each copied store, in the body's order.
    #[must_use]
    pub fn lens(&self) -> &[usize] {
        &self.lens
    }

    /// Bytes one checkpoint copies.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        point_bytes(&self.lens)
    }

    /// Checkpoints the budget holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ledger.capacity()
    }

    /// Slots made so far.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn stats(&self) -> Stats {
        self.ledger.stats()
    }

    /// The checkpoints' positions, ascending.
    #[must_use]
    pub fn positions(&self) -> Vec<u32> {
        self.ledger.positions()
    }

    /// What a cut to at most `n` keeps of the `held` positions, and why.
    #[must_use]
    pub fn kept(&self, n: u32, held: u32) -> Kept {
        self.ledger.keep(n, held)
    }

    /// The positions a prompt call from `from` to `to` takes a checkpoint at.
    #[must_use]
    pub fn marks(&self, from: u32, to: u32) -> Vec<u32> {
        seqstate::marks(from, to, self.every)
    }

    /// Whether a cut waits for [`Checkpoints::apply`].
    #[must_use]
    pub fn pending(&self) -> bool {
        self.ledger.pending().is_some()
    }

    /// Plan a cut to `n` of the `held` positions; refused by name unless a
    /// checkpoint, the held position or 0 is at `n`.
    pub fn cut(&mut self, n: u32, held: u32) -> Result<(), GpuError> {
        let cut = self.ledger.cut(n, held).map_err(refused)?;
        if cut != Cut::Stay {
            self.points.retain(|&(p, _)| p <= n);
        }
        Ok(())
    }

    /// Carry out the waiting cut on `stores`: each from its slot, or each
    /// zeroed. Waits for the copies. Nothing when no cut waits.
    pub fn apply(
        &mut self,
        stream: &CudaStream,
        stores: &mut [&mut DeviceBuffer<f32>],
    ) -> Result<(), GpuError> {
        let Some(cut) = self.ledger.pending() else {
            return Ok(());
        };
        self.check(stores)?;
        match cut {
            Cut::Stay => {}
            Cut::Empty => {
                for s in stores.iter_mut() {
                    s.zero_async(stream)?;
                }
            }
            Cut::Restore { slot, at } => {
                let src = self.slots.get(slot).ok_or(GpuError::State {
                    what: WHAT,
                    missing: "the restored checkpoint's slot",
                })?;
                seqstate::restorable(slot, src.holds, at).map_err(refused)?;
                let mut sent = Ok(());
                for (s, h) in stores.iter_mut().zip(&src.bufs) {
                    // SAFETY: the stream is synchronized below before this
                    // returns, whether or not every copy was enqueued, and
                    // `h` is neither written nor freed before then: the
                    // slots are only touched through `&mut self`.
                    sent = unsafe { s.copy_from_pinned_host_async(stream, h) };
                    if sent.is_err() {
                        break;
                    }
                }
                let synced = stream.synchronize();
                sent?;
                synced?;
            }
        }
        stream.synchronize()?;
        self.ledger.applied();
        Ok(())
    }

    /// A checkpoint of `stores` at `at`, the position the model stands at,
    /// after any waiting cut: a copy into a free, new or evicted slot, or
    /// nothing where one stands or at 0. Waits for the copies.
    pub fn take(
        &mut self,
        stream: &CudaStream,
        at: u32,
        stores: &mut [&mut DeviceBuffer<f32>],
    ) -> Result<Take, GpuError> {
        self.unread("a take")?;
        self.apply(stream, stores)?;
        self.check(stores)?;
        let take = self.ledger.take(at).map_err(refused)?;
        if let Take::Copy { slot, new, evicted } = take {
            self.make_slot(slot, new)?;
            self.took(at, slot, evicted);
            if let Err(e) = self.copy_into(stream, slot, at, stores) {
                // The slot holds part of a copy: no point may name it.
                self.clear();
                return Err(e);
            }
        }
        Ok(take)
    }

    /// Slot `slot` made (pinned and zeroed) when the ledger names it `new`:
    /// next to the made ones.
    fn make_slot(&mut self, slot: usize, new: bool) -> Result<(), GpuError> {
        if !new {
            return Ok(());
        }
        if slot != self.slots.len() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the ledger's new slot next to the made ones",
            });
        }
        let bufs = self
            .lens
            .iter()
            .map(|&n| PinnedHostBuffer::<f32>::zeroed(&self.ctx, n))
            .collect::<Result<Vec<_>, _>>()?;
        self.slots.push(Slot { bufs, holds: None });
        Ok(())
    }

    /// A take of the mark `at` opened ahead of the stores reaching it, after
    /// any waiting cut on `stores` (which hold the model's position): the
    /// ledger's point at `at` and its slot, which holds nothing until the
    /// take is sealed. `None` where one stands or at 0 (nothing to copy).
    /// Refused by name as [`Checkpoints::take`] refuses.
    pub fn open_take(
        &mut self,
        stream: &CudaStream,
        at: u32,
        stores: &mut [&mut DeviceBuffer<f32>],
    ) -> Result<Option<Pending>, GpuError> {
        self.unread("an open take")?;
        self.apply(stream, stores)?;
        self.check(stores)?;
        let Take::Copy { slot, new, evicted } = self.ledger.take(at).map_err(refused)? else {
            return Ok(None);
        };
        if let Err(e) = self.make_slot(slot, new) {
            // The point names a slot that was not made: no point may name it.
            self.clear();
            return Err(e);
        }
        self.took(at, slot, evicted);
        self.slots[slot].holds = None;
        self.open += 1;
        Ok(Some(Pending {
            at,
            slot,
            sent: vec![false; self.lens.len()],
        }))
    }

    /// Enqueue store `i`'s copy, `store` holding the open take's mark, into
    /// its slot. Refused by name for a store past the list, of another
    /// length, or copied already.
    pub fn copy(
        &mut self,
        stream: &CudaStream,
        pending: &mut Pending,
        i: usize,
        store: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let len = self.lens.get(i).copied();
        if len != Some(store.len()) || pending.sent.get(i) != Some(&false) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "store {i}'s copy of {} f32s into the take at {}: the checkpoints copy {}                      stores of {:?}, and store {i} is {}",
                    store.len(),
                    pending.at,
                    self.lens.len(),
                    self.lens,
                    if pending.sent.get(i) == Some(&true) {
                        "copied already"
                    } else {
                        "not one of them or of another length"
                    }
                ),
            });
        }
        let dst = self
            .slots
            .get_mut(pending.slot)
            .and_then(|s| s.bufs.get_mut(i))
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "the open take's slot",
            })?;
        // SAFETY: `dst` is the open take's slot's, which nothing else reads,
        // writes or frees until the take is sealed or abandoned — both
        // synchronize the stream first — or the checkpoints drop, which waits
        // for the context while a take is open.
        unsafe { store.copy_to_pinned_host_async(stream, dst)? };
        pending.sent[i] = true;
        Ok(())
    }

    /// The open take sealed: its copies waited for, its slot holding its
    /// mark. A take a later open evicted (a budget of fewer slots than the
    /// group's marks) is sealed as nothing, as the steps' take would have
    /// been evicted there. Refused by name, the take given back, when a store
    /// was not copied.
    pub fn seal(&mut self, stream: &CudaStream, pending: Pending) -> Result<(), GpuError> {
        let synced = stream.synchronize();
        self.open = self.open.saturating_sub(1);
        let Pending { at, slot, sent } = pending;
        if !self.points.contains(&(at, slot)) {
            synced?;
            return Ok(());
        }
        if let Err(e) = synced {
            self.forget(at)?;
            return Err(e.into());
        }
        if let Some(i) = sent.iter().position(|s| !s) {
            self.forget(at)?;
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("the take at {at} sealed with store {i} not copied"),
            });
        }
        self.slots[slot].holds = Some(at);
        Ok(())
    }

    /// The open take given back: its copies waited for, its point forgotten
    /// ([`Ledger::forget`]) and every other point left as it stands.
    pub fn abandon(&mut self, stream: &CudaStream, pending: Pending) -> Result<(), GpuError> {
        let synced = stream.synchronize();
        self.open = self.open.saturating_sub(1);
        if self.points.contains(&(pending.at, pending.slot)) {
            self.forget(pending.at)?;
        }
        synced?;
        Ok(())
    }

    /// The point at `at` given back to the ledger ([`Ledger::forget`]) and
    /// out of the points beside it; every other point stands.
    fn forget(&mut self, at: u32) -> Result<(), GpuError> {
        self.ledger.forget(at).map_err(refused)?;
        self.points.retain(|&(p, _)| p != at);
        Ok(())
    }

    /// The ledger's new point at `at` in `slot`, which the point at
    /// `evicted` gave up when there was one.
    fn took(&mut self, at: u32, slot: usize, evicted: Option<u32>) {
        if let Some(e) = evicted {
            self.points.retain(|&(p, _)| p != e);
        }
        let i = self.points.partition_point(|&(p, _)| p < at);
        self.points.insert(i, (at, slot));
    }

    /// The slot of the point at `at`, which its last finished copy holds;
    /// refused by name when no point stands there or its slot holds another
    /// position.
    fn slot_of(&self, at: u32) -> Result<&Slot, GpuError> {
        let slot = self
            .points
            .iter()
            .find(|&&(p, _)| p == at)
            .map(|&(_, s)| s)
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "no checkpoint at position {at}; they stand at {:?}",
                    self.positions()
                ),
            })?;
        let src = self.slots.get(slot).ok_or(GpuError::State {
            what: WHAT,
            missing: "the point's slot",
        })?;
        seqstate::restorable(slot, src.holds, at).map_err(refused)?;
        Ok(src)
    }

    /// The point at `at`'s stores into `dst`, one after another in the
    /// body's order ([`Checkpoints::lens`] summed); refused by name when no
    /// point stands there, its slot holds another position, or `dst` is
    /// another length. Host memory only.
    pub fn read_point(&self, at: u32, dst: &mut [f32]) -> Result<(), GpuError> {
        let src = self.slot_of(at)?;
        self.fits(dst.len())?;
        let mut off = 0;
        for b in &src.bufs {
            dst[off..off + b.len()].copy_from_slice(b.as_slice());
            off += b.len();
        }
        Ok(())
    }

    /// The points at `ats`' stores copied after the work `stream` holds, into
    /// `dst` (`ats.len()` checkpoints' f32s, each in [`Checkpoints::read_point`]'s
    /// layout, one after another): a host copy the driver runs in the stream's
    /// order, so the call returns once it is queued. Refused by name, nothing
    /// queued, as [`Checkpoints::read_point`] refuses. Follow it with
    /// [`Checkpoints::guard_points`] on the event that ends the copy.
    ///
    /// # Safety
    ///
    /// `dst .. dst + dst_len` must be writable pinned memory that nothing
    /// reads, writes or frees until the copy has completed, and `self` must
    /// outlive the copy or be waited for first (its drop does).
    pub unsafe fn enqueue_points(
        &self,
        stream: &CudaStream,
        ats: &[u32],
        dst: *mut f32,
        dst_len: usize,
    ) -> Result<(), GpuError> {
        let one: usize = self.lens.iter().sum();
        if dst_len != one * ats.len() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{dst_len} host f32s for {} checkpoints of {one}", ats.len()),
            });
        }
        let mut parts: Vec<(usize, usize, usize)> = Vec::new();
        for (i, &at) in ats.iter().enumerate() {
            let src = self.slot_of(at)?;
            let mut off = one * i;
            for b in &src.bufs {
                // SAFETY: `off + b.len()` is within `dst_len` (the lens sum to `one`).
                let to = unsafe { dst.add(off) };
                parts.push((b.as_slice().as_ptr() as usize, to as usize, b.len() * 4));
                off += b.len();
            }
        }
        let data = Box::into_raw(Box::new(parts)).cast::<c_void>();
        // SAFETY: `data` is the box `copy_points` takes back, once; the
        // addresses in it are the slots' pinned buffers (not written or freed
        // while a copy reads them: `unread`, `Drop`) and the caller's `dst`.
        let rc = unsafe { sys::cuLaunchHostFunc(stream.cu_stream(), Some(copy_points), data) };
        if rc != sys::cudaError_enum_CUDA_SUCCESS {
            // SAFETY: the launch failed, so `copy_points` never takes it.
            drop(unsafe { Box::from_raw(data.cast::<Vec<(usize, usize, usize)>>()) });
        }
        cu(rc, "cuLaunchHostFunc")
    }

    /// `end` marks the end of the copy [`Checkpoints::enqueue_points`] queued:
    /// no slot is written before it has completed.
    pub fn guard_points(&self, end: Arc<CudaEvent>) {
        self.reading
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(end);
    }

    /// Refused by name, nothing changed, while a copy of a point's slots
    /// ([`Checkpoints::enqueue_points`]) has not completed: `what` would write
    /// a slot under it, and the caller waits for the save first
    /// ([`saved::Pending::land`]).
    fn unread(&self, what: &str) -> Result<(), GpuError> {
        let mut reading = self.reading.lock().unwrap_or_else(PoisonError::into_inner);
        let mut live = Vec::new();
        let mut failed = None;
        for e in reading.drain(..) {
            match e.query() {
                Ok(true) => {}
                Ok(false) => live.push(e),
                Err(err) => {
                    failed.get_or_insert(err);
                    live.push(e);
                }
            }
        }
        *reading = live;
        if let Some(err) = failed {
            return Err(err.into());
        }
        if reading.is_empty() {
            return Ok(());
        }
        Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "save in flight: {what} would write a slot a saved state's point copy still \
                 reads; wait for the save to land first"
            ),
        })
    }

    /// A point at `at` whose stores are `src` ([`Checkpoints::read_point`]'s
    /// layout): a slot the ledger gives it, filled from the host. The model
    /// is cut without a copy: `at` lies at or above every point, and no cut
    /// waits. Refused by name otherwise, at 0, where a point stands, and when
    /// `src` is another length; a slot left half written names no point.
    /// Host memory only.
    pub fn adopt(&mut self, at: u32, src: &[f32]) -> Result<(), GpuError> {
        self.unread("an adopt")?;
        self.fits(src.len())?;
        let take = self.ledger.take(at).map_err(refused)?;
        let Take::Copy { slot, new, evicted } = take else {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a point adopted at {at}, where the ledger takes {take:?}"),
            });
        };
        if new {
            if slot != self.slots.len() {
                self.clear();
                return Err(GpuError::State {
                    what: WHAT,
                    missing: "the ledger's new slot next to the made ones",
                });
            }
            let bufs = match self
                .lens
                .iter()
                .map(|&n| PinnedHostBuffer::<f32>::zeroed(&self.ctx, n))
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(b) => b,
                Err(e) => {
                    self.clear();
                    return Err(e.into());
                }
            };
            self.slots.push(Slot { bufs, holds: None });
        }
        self.took(at, slot, evicted);
        let Some(dst) = self.slots.get_mut(slot) else {
            self.clear();
            return Err(GpuError::State {
                what: WHAT,
                missing: "the adopted point's slot",
            });
        };
        let mut off = 0;
        for b in &mut dst.bufs {
            let n = b.len();
            b.as_mut_slice().copy_from_slice(&src[off..off + n]);
            off += n;
        }
        dst.holds = Some(at);
        Ok(())
    }

    /// Refused by name unless `len` f32s are one checkpoint's.
    fn fits(&self, len: usize) -> Result<(), GpuError> {
        let want: usize = self.lens.iter().sum();
        if len == want {
            Ok(())
        } else {
            Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{len} host f32s for a checkpoint of {want}"),
            })
        }
    }

    /// `stores`, which hold position `at`, into slot `slot`'s buffers; waits
    /// for the copies. The slot holds `at` once every copy has finished, and
    /// nothing from the first enqueue on until then.
    fn copy_into(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        at: u32,
        stores: &[&mut DeviceBuffer<f32>],
    ) -> Result<(), GpuError> {
        let dst = self.slots.get_mut(slot).ok_or(GpuError::State {
            what: WHAT,
            missing: "the checkpoint's slot",
        })?;
        dst.holds = None;
        let mut sent = Ok(());
        for (s, h) in stores.iter().zip(dst.bufs.iter_mut()) {
            // SAFETY: the stream is synchronized below before this returns,
            // whether or not every copy was enqueued, and `h` is neither
            // read nor freed before then: the slots are only touched through
            // `&mut self`.
            sent = unsafe { s.copy_to_pinned_host_async(stream, h) };
            if sent.is_err() {
                break;
            }
        }
        let synced = stream.synchronize();
        sent?;
        synced?;
        dst.holds = Some(at);
        Ok(())
    }

    /// Every checkpoint dropped and no cut waiting: the model is empty. The
    /// slots stay made.
    pub fn clear(&mut self) {
        self.ledger.clear();
        self.points.clear();
    }

    /// Refused by name unless `stores` are the stores of the lengths this
    /// was made for, in order.
    fn check(&self, stores: &[&mut DeviceBuffer<f32>]) -> Result<(), GpuError> {
        let lens: Vec<usize> = stores.iter().map(|s| s.len()).collect();
        if lens != self.lens {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} stores of {lens:?} f32s; the checkpoints copy {} of {:?}",
                    lens.len(),
                    self.lens.len(),
                    self.lens
                ),
            });
        }
        Ok(())
    }
}

impl Drop for Checkpoints {
    /// With a take open, its copies may still write its slot, and a save's
    /// point copy may still read them: the context is waited for before the
    /// slots are freed, under the device's capture
    /// lock ([`crate::capsync::ctx_sync_in_drop`], which a drop path calls
    /// because it must not panic).
    fn drop(&mut self) {
        let reading = !self
            .reading
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty();
        if self.open > 0 || reading {
            let _ = crate::capsync::ctx_sync_in_drop(&self.ctx, "Checkpoints::drop");
        }
    }
}

/// The host copy [`Checkpoints::enqueue_points`] queues: `(from, to, bytes)`
/// each, as addresses. Runs on the driver's thread in the stream's order and
/// makes no driver call.
unsafe extern "C" fn copy_points(data: *mut c_void) {
    // SAFETY: the box `enqueue_points` leaked for this one call.
    let parts = unsafe { Box::from_raw(data.cast::<Vec<(usize, usize, usize)>>()) };
    for &(from, to, bytes) in parts.iter() {
        // SAFETY: both ranges are pinned memory `enqueue_points`' contract
        // keeps unwritten and unfreed until this copy has completed, and they
        // are different buffers.
        unsafe { std::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, bytes) };
    }
}

/// A stream held shut for a test: what is enqueued on it after [`hold::Hold::on`]
/// does not start until the test opens the hold, so a copy behind it is in
/// flight for as long as the test says, whatever else the device or the
/// driver is doing.
#[cfg(test)]
pub(crate) mod hold {
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use cuda_core::{CudaStream, sys};

    use crate::GpuError;
    use crate::graph::cu;

    /// Runs on the driver's thread in the stream's order, and returns once
    /// the flag is up.
    unsafe extern "C" fn wait_open(flag: *mut c_void) {
        // SAFETY: the `Arc` count `Hold::on` handed this call.
        let flag = unsafe { Arc::from_raw(flag.cast::<AtomicBool>()) };
        while !flag.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// The hold; dropping it opens it, so a failed assert never leaves the
    /// stream shut. While it is shut the test must not wait for anything
    /// enqueued behind it: no land, no event or stream sync, no extent drop.
    ///
    /// No other test may run beside it: a device-wide sync in a parallel test
    /// (`cuMemFree` is one) waits for the shut stream while holding the
    /// driver's lock that this test's next launch needs, and neither returns.
    /// So a test that holds is named `hw_held_…`, the lib's parallel pass
    /// skips that prefix, and a second pass runs it under `RUST_TEST_THREADS=1`
    /// (`just gate-gpu-lib`); [`Hold::on`] refuses by name anywhere else.
    pub struct Hold(Arc<AtomicBool>);

    impl Hold {
        /// Shuts `stream` after the work it holds so far. Panics by name
        /// unless the tests run one at a time (see [`Hold`]).
        pub fn on(stream: &CudaStream) -> Result<Hold, GpuError> {
            assert_eq!(
                std::env::var("RUST_TEST_THREADS").as_deref(),
                Ok("1"),
                "a held stream deadlocks a parallel test's device sync: name the test hw_held_… so \
                 gate-gpu-lib runs it in its RUST_TEST_THREADS=1 pass"
            );
            let flag = Arc::new(AtomicBool::new(false));
            let data = Arc::into_raw(Arc::clone(&flag)).cast_mut().cast::<c_void>();
            // SAFETY: `data` is a live `Arc` count that `wait_open` takes.
            let rc = unsafe { sys::cuLaunchHostFunc(stream.cu_stream(), Some(wait_open), data) };
            if rc != sys::cudaError_enum_CUDA_SUCCESS {
                // SAFETY: the launch failed, so `wait_open` never takes it.
                drop(unsafe { Arc::from_raw(data.cast::<AtomicBool>()) });
            }
            cu(rc, "cuLaunchHostFunc")?;
            Ok(Hold(flag))
        }

        /// Opens the hold: the stream runs on.
        pub fn open(&self) {
            self.0.store(true, Ordering::Release);
        }
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            self.open();
        }
    }
}

fn refused(e: seqstate::CheckpointError) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sequence's checkpoints hold the points one prompt call over its
    /// context takes: stores of 7 f32s spaced 512 apart make 1 point at 512
    /// positions, 8 at 4,096 (the marks 512 to 3,584 and the end), each of
    /// `point_bytes`, the ledger exactly the budget's slots; a context past
    /// the u32 positions is refused by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_sequence_holds_the_budget_of_its_context() {
        let (ctx, _stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let lens = vec![4, 3];
        for (positions, points) in [(512usize, 1usize), (4_096, 8)] {
            let c = Checkpoints::sequence(&ctx, lens.clone(), positions, 512).expect("checkpoints");
            assert_eq!(c.capacity(), points, "{positions} positions");
            assert_eq!(c.bytes(), point_bytes(&lens));
            let planned = budget(price(&lens, 512), positions as u64).expect("a budget");
            assert_eq!(planned, points as u64 * c.bytes(), "{positions} positions");
        }
        let Err(e) = Checkpoints::sequence(&ctx, lens, usize::MAX, 512) else {
            panic!("a context past the u32 positions is refused");
        };
        assert!(e.to_string().contains("passes the u32 positions"), "{e}");
    }

    /// A restore reads a slot only as the position its last finished copy
    /// holds: a ledger point planted on a slot made and never copied is
    /// refused by name, and the stores keep what they held; a slot a take
    /// filled restores.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_restore_refuses_a_slot_no_copy_filled() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let mut c = Checkpoints::new(&ctx, vec![4, 3], 1 << 20, 512).expect("checkpoints");
        let mut a = DeviceBuffer::from_host(&stream, &[1.0f32; 4]).expect("a store");
        let mut b = DeviceBuffer::from_host(&stream, &[2.0f32; 3]).expect("a store");
        let take = c
            .take(&stream, 5, &mut [&mut a, &mut b])
            .expect("a take at 5");
        assert!(matches!(take, Take::Copy { slot: 0, .. }), "{take:?}");
        // The plant: the ledger names a new slot at 9, the slot is made as a
        // take makes it, and no copy runs.
        let planted = c.ledger.take(9).expect("the ledger's point at 9");
        assert!(
            matches!(
                planted,
                Take::Copy {
                    slot: 1,
                    new: true,
                    ..
                }
            ),
            "{planted:?}"
        );
        let bufs = c
            .lens
            .iter()
            .map(|&n| PinnedHostBuffer::<f32>::zeroed(&ctx, n))
            .collect::<Result<Vec<_>, _>>()
            .expect("the slot's buffers");
        c.slots.push(Slot { bufs, holds: None });
        c.cut(9, 10).expect("a cut to the planted point");
        let err = c
            .apply(&stream, &mut [&mut a, &mut b])
            .expect_err("a restore of a slot no copy filled");
        assert!(
            err.to_string()
                .contains("a restore of position 9 from slot 1, which no copy has finished"),
            "{err}"
        );
        assert_eq!(a.to_host_vec(&stream).expect("a"), [1.0; 4]);
        c.clear();
        a.copy_from_host(&stream, &[0.0; 4]).expect("a");
        let take = c
            .take(&stream, 5, &mut [&mut a, &mut b])
            .expect("a take at 5");
        assert!(matches!(take, Take::Copy { .. }), "{take:?}");
        a.copy_from_host(&stream, &[3.0; 4]).expect("a");
        c.cut(5, 6).expect("a cut to 5");
        c.apply(&stream, &mut [&mut a, &mut b])
            .expect("a restore of the copied slot");
        assert_eq!(a.to_host_vec(&stream).expect("a"), [0.0; 4]);
    }

    /// A group's take store by store: open, one copy a store, seal leaves
    /// the slot holding the mark, and a restore of it gives the stores the
    /// copied bits; a seal with a store not copied is refused by name and
    /// leaves no point; an abandon gives back its own point alone — on an
    /// empty ledger the ledger is empty again, every earlier point stands.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_open_copy_seal_and_abandon() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let mut c = Checkpoints::new(&ctx, vec![4, 3], 1 << 20, 512).expect("checkpoints");
        let mut a = DeviceBuffer::from_host(&stream, &[1.0f32; 4]).expect("a store");
        let mut b = DeviceBuffer::from_host(&stream, &[2.0f32; 3]).expect("a store");
        let p = c
            .open_take(&stream, 9, &mut [&mut a, &mut b])
            .expect("an open take at 9")
            .expect("a copy at 9");
        assert_eq!(p.at(), 9);
        c.abandon(&stream, p).expect("an abandon");
        assert!(c.positions().is_empty(), "{:?}", c.positions());
        let mut p = c
            .open_take(&stream, 5, &mut [&mut a, &mut b])
            .expect("an open take at 5")
            .expect("a copy at 5");
        let slot = p.slot;
        c.copy(&stream, &mut p, 0, &a).expect("store 0's copy");
        let again = c.copy(&stream, &mut p, 0, &a).expect_err("store 0 twice");
        assert!(again.to_string().contains("copied already"), "{again}");
        c.copy(&stream, &mut p, 1, &b).expect("store 1's copy");
        a.copy_from_host(&stream, &[7.0; 4]).expect("a");
        c.seal(&stream, p).expect("a seal");
        assert_eq!(c.slots[slot].holds, Some(5));
        assert_eq!(c.positions(), vec![5]);
        let mut p = c
            .open_take(&stream, 9, &mut [&mut a, &mut b])
            .expect("an open take at 9")
            .expect("a copy at 9");
        c.copy(&stream, &mut p, 0, &a).expect("store 0's copy");
        let short = c
            .seal(&stream, p)
            .expect_err("a seal with store 1 not copied");
        assert!(
            short
                .to_string()
                .contains("the take at 9 sealed with store 1 not copied"),
            "{short}"
        );
        assert_eq!(c.positions(), vec![5]);
        let p = c
            .open_take(&stream, 9, &mut [&mut a, &mut b])
            .expect("an open take at 9")
            .expect("a copy at 9");
        c.abandon(&stream, p).expect("an abandon");
        assert_eq!(c.positions(), vec![5]);
        c.cut(5, 9).expect("a cut to 5");
        c.apply(&stream, &mut [&mut a, &mut b])
            .expect("a restore of the sealed take");
        assert_eq!(a.to_host_vec(&stream).expect("a"), [1.0; 4]);
        assert_eq!(b.to_host_vec(&stream).expect("b"), [2.0; 3]);
    }
}
