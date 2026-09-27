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
//! A slot is made (pinned and zeroed) the first time a checkpoint needs it,
//! and kept for the load.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, PinnedHostBuffer};
use runtime::seqstate::{self, Cut, Kept, Ledger, Stats, Take};

use crate::GpuError;

const WHAT: &str = "checkpoint";

/// One model's checkpoints: the ledger, the stores' lengths, the slots.
pub struct Checkpoints {
    ledger: Ledger,
    /// f32s of each copied store, in the body's order.
    lens: Vec<usize>,
    /// Per slot, one pinned buffer per store.
    slots: Vec<Vec<PinnedHostBuffer<f32>>>,
    /// The spacing of a prompt call's inner checkpoints (`seqstate::marks`).
    every: u32,
    ctx: Arc<CudaContext>,
}

impl Checkpoints {
    /// Checkpoints of stores of `lens` f32s, as many as `budget` bytes of
    /// host slots hold, taken inside a prompt call every `every` positions.
    /// Refused by name when not one fits.
    pub fn new(
        ctx: &Arc<CudaContext>,
        lens: Vec<usize>,
        budget: u64,
        every: u32,
    ) -> Result<Checkpoints, GpuError> {
        let bytes = lens.iter().map(|&n| (n * size_of::<f32>()) as u64).sum();
        let cap = seqstate::capacity(budget, bytes).map_err(refused)?;
        Ok(Checkpoints {
            ledger: Ledger::new(cap).map_err(refused)?,
            lens,
            slots: Vec::new(),
            every,
            ctx: Arc::clone(ctx),
        })
    }

    /// Bytes one checkpoint copies.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.lens
            .iter()
            .map(|&n| (n * size_of::<f32>()) as u64)
            .sum()
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
        self.ledger.cut(n, held).map(|_| ()).map_err(refused)
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
            Cut::Restore { slot } => {
                let src = self.slots.get(slot).ok_or(GpuError::State {
                    what: WHAT,
                    missing: "the restored checkpoint's slot",
                })?;
                let mut sent = Ok(());
                for (s, h) in stores.iter_mut().zip(src) {
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
        self.apply(stream, stores)?;
        self.check(stores)?;
        let take = self.ledger.take(at).map_err(refused)?;
        if let Take::Copy { slot, new, .. } = take {
            if new {
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
                self.slots.push(bufs);
            }
            if let Err(e) = self.copy_into(stream, slot, stores) {
                // The slot holds part of a copy: no point may name it.
                self.ledger.clear();
                return Err(e);
            }
        }
        Ok(take)
    }

    /// `stores` into slot `slot`'s buffers; waits for the copies.
    fn copy_into(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        stores: &[&mut DeviceBuffer<f32>],
    ) -> Result<(), GpuError> {
        let dst = self.slots.get_mut(slot).ok_or(GpuError::State {
            what: WHAT,
            missing: "the checkpoint's slot",
        })?;
        let mut sent = Ok(());
        for (s, h) in stores.iter().zip(dst.iter_mut()) {
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
        Ok(synced?)
    }

    /// Every checkpoint dropped and no cut waiting: the model is empty. The
    /// slots stay made.
    pub fn clear(&mut self) {
        self.ledger.clear();
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

fn refused(e: seqstate::CheckpointError) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: e.to_string(),
    }
}
