//! Host copies of a model's recurrent stores at chosen positions: the Metal
//! device half of [`runtime::seqstate`]. A body hands in its copied stores
//! as one list, always in the same order, and this owns the ledger's rules,
//! the stores' lengths and the copies.
//!
//! The caller holds the stream idle across every take and restore: no token
//! runs ahead, and the last step has settled. Every buffer is shared, so a
//! copy is the host's between those pages while no device work runs.
//!
//! A slot is one shared buffer per store, made when a take first asks for
//! it, so a short chat pins only the points it took. The slot records the
//! position its last finished copy holds, apart from the ledger: a restore
//! reads a slot only as that position ([`runtime::seqstate::restorable`]).
//! A copy that fails leaves the slot holding nothing a restore will read,
//! and the point is forgotten.

use std::fmt;

use bloomery_device::{DeviceError, Elems};
use objc2_metal::MTLDevice;
use runtime::seqstate::{self, Cut, Kept, Ledger, Price, Take};

use crate::buffer::MetalBuffer;
use crate::handle::Handle;

const WHAT: &str = "checkpoint";

/// One slot: a shared buffer per store, and the position its last finished
/// copy holds — none when made, and none again after a copy that failed.
struct Slot {
    bufs: Vec<MetalBuffer<f32>>,
    holds: Option<u32>,
}

/// A store a restore writes: its buffer and the element its copy starts at.
pub(crate) struct Restore<'a> {
    pub(crate) buf: &'a mut MetalBuffer<f32>,
    pub(crate) at: Elems,
}

/// One model's checkpoints: the ledger, the stores' lengths, the slots.
pub(crate) struct Checkpoints {
    device: Handle<dyn MTLDevice>,
    ledger: Ledger,
    /// f32s of each copied store, in the body's order.
    lens: Vec<usize>,
    slots: Vec<Slot>,
    every: u32,
    point_bytes: u64,
}

/// A copy or a plan this half refuses, by name.
#[derive(Debug)]
pub(crate) enum Error {
    /// The ledger's own refusal.
    Rule(seqstate::CheckpointError),
    /// A store list that is not the one the checkpoints were made for, or a
    /// slot the ledger named that this half does not hold.
    Shape { detail: String },
    /// A buffer the device did not make, or a copy it refused.
    Device(DeviceError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Rule(e) => write!(f, "{e}"),
            Error::Shape { detail } => write!(f, "{WHAT}: {detail}"),
            Error::Device(e) => write!(f, "{WHAT}: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// The bytes one checkpoint of stores of `lens` f32s copies.
fn point_bytes(lens: &[usize]) -> Result<u64, Error> {
    let width = u64::try_from(size_of::<f32>()).map_err(|_| Error::Shape {
        detail: "the size of an f32 does not fit a byte count".to_string(),
    })?;
    let mut bytes = 0u64;
    for &n in lens {
        let n = u64::try_from(n).map_err(|_| Error::Shape {
            detail: format!("a store of {n} f32s passes a byte count"),
        })?;
        let row = n.checked_mul(width).ok_or_else(|| Error::Shape {
            detail: format!("a store of {n} f32s overflows a byte count"),
        })?;
        bytes = bytes.checked_add(row).ok_or_else(|| Error::Shape {
            detail: "the checkpoint's stores overflow a byte count".to_string(),
        })?;
    }
    Ok(bytes)
}

fn shape(detail: impl Into<String>) -> Error {
    Error::Shape {
        detail: detail.into(),
    }
}

impl Checkpoints {
    /// Checkpoints of stores of `lens` f32s in a load of `ctx` positions,
    /// taken inside a prompt call every `every` positions. The slot count is
    /// what [`Price::budget`] grants at `ctx`; the buffers are made later,
    /// one slot at a time. Refused by name when the budget holds none.
    pub(crate) fn open(
        device: Handle<dyn MTLDevice>,
        lens: Vec<usize>,
        ctx: u64,
        every: u32,
    ) -> Result<Checkpoints, Error> {
        if lens.is_empty() || lens.contains(&0) {
            return Err(shape(
                "a checkpoint copies no store: a delta layer's lane and ring are required",
            ));
        }
        let point_bytes = point_bytes(&lens)?;
        let budget = Price { point_bytes, every }
            .budget(ctx)
            .map_err(Error::Rule)?;
        let cap = seqstate::capacity(budget, point_bytes).map_err(Error::Rule)?;
        let ledger = Ledger::new(cap, every).map_err(Error::Rule)?;
        seqstate::check_ledger(budget, ledger.capacity(), point_bytes).map_err(Error::Rule)?;
        Ok(Checkpoints {
            device,
            ledger,
            lens,
            slots: Vec::new(),
            every,
            point_bytes,
        })
    }

    /// Bytes one checkpoint copies.
    #[must_use]
    pub(crate) fn point_bytes(&self) -> u64 {
        self.point_bytes
    }

    /// Checkpoints the budget holds.
    #[must_use]
    pub(crate) fn capacity(&self) -> usize {
        self.ledger.capacity()
    }

    /// The spacing of a prompt call's inner checkpoints.
    #[must_use]
    pub(crate) fn every(&self) -> u32 {
        self.every
    }

    /// The checkpoints' positions, ascending.
    #[must_use]
    pub(crate) fn positions(&self) -> Vec<u32> {
        self.ledger.positions()
    }

    /// The positions a prompt call from `from` to `to` takes a checkpoint at.
    #[must_use]
    pub(crate) fn marks(&self, from: u32, to: u32) -> Vec<u32> {
        seqstate::marks(from, to, self.every)
    }

    /// The longest prefix of at most `n` positions a cut keeps of the `held`
    /// the model stands at.
    #[must_use]
    pub(crate) fn kept(&self, n: u32, held: u32) -> Kept {
        self.ledger.keep(n, held)
    }

    /// A checkpoint at `at`, the position the model stands at, from `src` in
    /// the body's order. A length that is not the lens is refused by name
    /// before the ledger moves. A slot the device does not make is forgotten,
    /// so nothing names it.
    pub(crate) fn take(&mut self, at: u32, src: &[&[f32]]) -> Result<(), Error> {
        self.check_lengths(src.len(), src.iter().map(|row| row.len()))?;
        match self.ledger.take(at).map_err(Error::Rule)? {
            Take::Empty | Take::Held => Ok(()),
            Take::Copy { slot, .. } => {
                if let Err(e) = self.copy_into(slot, at, src) {
                    if let Err(forgotten) = self.ledger.forget(at) {
                        return Err(shape(format!(
                            "{e}; and the point at {at} was not forgotten: {forgotten}"
                        )));
                    }
                    return Err(e);
                }
                Ok(())
            }
        }
    }

    /// Plan a cut to `n` of the `held` positions and carry it out into `dst`,
    /// one store in the body's order, each copied once from the slot's pages
    /// into the store's buffer at its start. A store list that is not the
    /// lens, or a store whose buffer does not hold its length from its start,
    /// is refused before the ledger moves. [`Cut::Stay`] writes nothing; a
    /// write the device refuses partway is [`Error::Device`], the stores
    /// before it already written.
    pub(crate) fn cut(&mut self, n: u32, held: u32, dst: &mut [Restore<'_>]) -> Result<(), Error> {
        if dst.len() != self.lens.len() {
            return Err(shape(format!(
                "a restore of {} stores; the checkpoint holds {}",
                dst.len(),
                self.lens.len()
            )));
        }
        for (i, (d, &want)) in dst.iter().zip(&self.lens).enumerate() {
            let room = d.buf.pages().len().get().saturating_sub(d.at.get());
            if room < want {
                return Err(shape(format!(
                    "store {i} holds {room} f32s from its start; the checkpoint copies {want}"
                )));
            }
        }
        match self.ledger.cut(n, held).map_err(Error::Rule)? {
            Cut::Stay => Ok(()),
            Cut::Empty => {
                for (d, &want) in dst.iter_mut().zip(&self.lens) {
                    d.buf
                        .write_at(d.at, &vec![0.0; want])
                        .map_err(Error::Device)?;
                }
                self.ledger.applied();
                Ok(())
            }
            Cut::Restore { slot, at } => {
                self.restore(slot, at, dst)?;
                self.ledger.applied();
                Ok(())
            }
        }
    }

    /// Every point dropped and no cut waiting.
    pub(crate) fn clear(&mut self) {
        self.ledger.clear();
    }

    fn check_lengths(
        &self,
        n_stores: usize,
        lengths: impl Iterator<Item = usize>,
    ) -> Result<(), Error> {
        if n_stores != self.lens.len() {
            return Err(shape(format!(
                "a copy of {n_stores} stores; the checkpoint holds {}",
                self.lens.len()
            )));
        }
        for (i, (got, &want)) in lengths.zip(&self.lens).enumerate() {
            if got != want {
                return Err(shape(format!(
                    "store {i} is {got} f32s; the checkpoint holds {want}"
                )));
            }
        }
        Ok(())
    }

    /// The slot a take named. An index one past the slots made is allocated;
    /// a gap is refused, so a failed allocation is asked for again rather
    /// than read.
    fn ensure_slot(&mut self, slot: usize) -> Result<(), Error> {
        if slot < self.slots.len() {
            return Ok(());
        }
        if slot > self.slots.len() {
            return Err(shape(format!(
                "a checkpoint asked for slot {slot} with {} made",
                self.slots.len()
            )));
        }
        let mut bufs = Vec::with_capacity(self.lens.len());
        for &n in &self.lens {
            bufs.push(MetalBuffer::new(&self.device, Elems(n)).map_err(Error::Device)?);
        }
        self.slots.push(Slot { bufs, holds: None });
        Ok(())
    }

    fn copy_into(&mut self, slot: usize, at: u32, src: &[&[f32]]) -> Result<(), Error> {
        self.ensure_slot(slot)?;
        let kept = &mut self.slots[slot];
        // A copy that stops midway holds no position a restore may read.
        kept.holds = None;
        for (buf, row) in kept.bufs.iter_mut().zip(src) {
            buf.write(row).map_err(Error::Device)?;
        }
        kept.holds = Some(at);
        Ok(())
    }

    fn restore(&self, slot: usize, at: u32, dst: &mut [Restore<'_>]) -> Result<(), Error> {
        let kept = self
            .slots
            .get(slot)
            .ok_or_else(|| shape(format!("a restore from slot {slot}, which was never made")))?;
        seqstate::restorable(slot, kept.holds, at).map_err(Error::Rule)?;
        for (buf, d) in kept.bufs.iter().zip(dst.iter_mut()) {
            // SAFETY: no device work binds a slot's buffer, and the host
            // writes one only through `&mut self` (a take's copy), so nothing
            // writes these pages while `&self` reads them here.
            let src = unsafe { buf.pages().as_slice() };
            d.buf.write_at(d.at, src).map_err(Error::Device)?;
        }
        Ok(())
    }
}
