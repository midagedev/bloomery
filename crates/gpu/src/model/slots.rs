//! Resident sequence slots: the common owner of N sequences inside one
//! [`GpuModel`] ([`GpuModel::add_slots`]). A body that implements [`Slots`]
//! parks its per-sequence state with the model, and the model switches
//! slots by pointer exchange — no device copy, no allocation, no capture, no
//! synchronization.
//!
//! Why exchange: a CUDA graph replays the buffer addresses it recorded, so a
//! slot's captured chains must stay with that slot's buffers. The model
//! keeps one graph cache ([`Graphs`]) a slot, and every capture the body
//! records over its sequence's stores travels inside its [`Slots::Seq`] —
//! exchanged by the body's own [`Slots::swap_seq`], so a select costs no
//! capture and never replays one slot's chain over another's buffers.
//!
//! Where each slot's state lives: slot 0's home is the live fields (the
//! load's own sequence, where every enqueue already reads it); each further
//! slot's home is its [`ParkedSlot`] entry, and the entry of the slot that
//! is live holds slot 0's state. A switch to or from slot 0 is one
//! exchange; a switch between two parked slots is two (the live slot's
//! state home first, the target's out) — pointer swaps either way.

use super::{ChainBody, GpuModel, Graphs};
use crate::{Gpu, GpuError};

/// A [`ChainBody`] that can hold several sequences at once: N resident
/// sequences inside one model over one set of weights, switched by pointer
/// exchange. The body keeps the live sequence where its own enqueues read
/// it; the model parks the others ([`GpuModel::add_slots`]) and every call
/// acts on the selected one ([`GpuModel::select_slot`]). Model-wide state
/// stays one: the fault word, the residency machine and host tier, the
/// heads and the weights.
pub trait Slots: ChainBody {
    /// Everything one sequence owns on the card and host: the stores a step
    /// or a prompt writes and a later call reads, and any body-internal
    /// capture that records those stores' addresses.
    type Seq: Send;

    /// A sequence in the state the load leaves ([`ChainBody::reset`]'s
    /// contract), of the same shape (`ctx_max`) as the live one. Load-time
    /// allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Self::Seq, GpuError>;

    /// Exchange the live sequence with `seq`: pointer moves only for a body
    /// whose sequences are device buffers. Fallible and handed the card for
    /// a body that must drain work in flight before its exchange.
    fn swap_seq(&mut self, gpu: &Gpu, seq: &mut Self::Seq) -> Result<(), GpuError>;

    /// Device bytes one sequence holds: what [`GpuModel::resident_bytes`]
    /// grows by per added slot.
    fn seq_bytes(&self) -> usize;
}

/// A parked sequence as the model holds it: type-erased only as far as the
/// parking needs, and given back to the body's exchange whole — the blanket
/// impl below is the only `ParkedSeq` a `B::Seq` ever becomes, so a box
/// holds exactly the sequence [`Slots::new_seq`] made.
pub(super) trait ParkedSeq<B: ChainBody>: Send {
    /// Exchange this parked sequence with `body`'s live one
    /// ([`Slots::swap_seq`]).
    fn exchange(&mut self, gpu: &Gpu, body: &mut B) -> Result<(), GpuError>;
}

impl<B: Slots> ParkedSeq<B> for B::Seq {
    fn exchange(&mut self, gpu: &Gpu, body: &mut B) -> Result<(), GpuError> {
        body.swap_seq(gpu, self)
    }
}

/// One parked slot's state in the model's parking lot
/// ([`GpuModel::parked`]): the slot's captured chains, the position it stood
/// at, its device bytes and its sequence. The captures are declared first —
/// fields drop in declaration order, and a capture must be destroyed while
/// every buffer it addresses is alive.
pub(super) struct ParkedSlot<B: ChainBody> {
    pub(super) graphs: Graphs,
    pub(super) pos: u32,
    pub(super) bytes: usize,
    pub(super) seq: Box<dyn ParkedSeq<B>>,
}

impl<B: Slots> GpuModel<B>
where
    // A parked sequence outlives the call that parked it: it sits in the
    // model's lot until the model drops, so it can borrow nothing.
    B::Seq: 'static,
{
    /// The model serves `n` slots from now on: `n − slots()` new sequences
    /// allocated through [`Slots::new_seq`], each parked empty at position 0
    /// with no capture. The live slot keeps its state. `n == 0` and
    /// `n < slots()` are refused by name (slots are never removed);
    /// callable any time.
    pub fn add_slots(&mut self, n: usize) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::add_slots";
        let held = self.parked.len() + 1;
        if n == 0 {
            return Err(GpuError::shape(WHAT, "a slot count of at least 1"));
        }
        if n < held {
            return Err(GpuError::shape(
                WHAT,
                format!("{n} slots of a model that already serves {held}; slots are never removed"),
            ));
        }
        let bytes = self.body.seq_bytes();
        for _ in held..n {
            let seq = self.body.new_seq(&self.gpu)?;
            self.parked.push(ParkedSlot {
                graphs: Graphs::new(),
                pos: 0,
                bytes,
                seq: Box::new(seq),
            });
        }
        Ok(())
    }

    /// The slots the model serves: 1 until [`GpuModel::add_slots`] grows it.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.parked.len() + 1
    }

    /// The slot every later call acts on: 0 until a
    /// [`GpuModel::select_slot`] moves it.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Make `slot` the one every later call acts on, exchanging the live
    /// slot's sequence ([`Slots::swap_seq`]), position and capture cache
    /// with its parked state. Selecting the live slot is a no-op; out of
    /// range is refused by name. Pointer moves only: nothing allocated,
    /// captured, copied or synchronized.
    pub fn select_slot(&mut self, slot: usize) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::select_slot";
        let n = self.parked.len() + 1;
        if slot >= n {
            return Err(GpuError::shape(
                WHAT,
                format!("slot {slot} of a model that serves 0..{n}"),
            ));
        }
        if slot == self.selected {
            return Ok(());
        }
        // The homes rule (module doc): bring the live slot's state home —
        // slot 0's into the live fields — then park it in the target's
        // entry. One of the two exchanges runs on every switch.
        // A failed exchange leaves the live fields whole: `selected` names
        // the slot they hold after each step, so a refusal mid-switch leaves
        // slot 0 (or the old slot) live and named, never a mix.
        self.one_pass = None;
        if self.selected != 0 {
            self.swap_parked(self.selected - 1)?;
            self.selected = 0;
        }
        if slot != 0 {
            self.swap_parked(slot - 1)?;
            self.selected = slot;
        }
        Ok(())
    }

    /// Exchange the live slot's state with parked entry `i`'s: the sequence
    /// first (the one fallible part, [`Slots::swap_seq`]), then the capture
    /// cache and the position — so a refused exchange moves nothing.
    fn swap_parked(&mut self, i: usize) -> Result<(), GpuError> {
        let GpuModel {
            graphs,
            parked,
            body,
            gpu,
            pos,
            ..
        } = self;
        let p = &mut parked[i];
        // Called by name: method syntax probes the blanket impl's `B::Seq`
        // against the box itself and fails there instead of dereferencing.
        ParkedSeq::exchange(p.seq.as_mut(), gpu, body.as_mut())?;
        std::mem::swap(graphs, &mut p.graphs);
        std::mem::swap(pos, &mut p.pos);
        Ok(())
    }
}
