//! The Metal fault word: the pages the kernels raise.
//!
//! The word is one shared buffer of [`FAULT_WORDS`] u32s: word 0 is the
//! first-layer word, and word `1 + layer` is that layer's site mask. A
//! kernel raises it with an atomic minimum on word 0 and an atomic OR on
//! its own layer's mask — the same two operations the contract's host model
//! does. The host reads it through the contract's one decoding rule
//! ([`Fault::read_from`]) and clears it, as atomic words, after the stream it
//! is asked against has finished its work, or, with no wait of its own, on
//! the proof a settle of the stream returned ([`Settled`]).

use core::sync::atomic::Ordering;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use bloomery_device::fault::{FAULT_NONE, FAULT_WORDS, Fault, FaultWord};
use bloomery_device::{DeviceError, Elems, Stream};

use crate::buffer::MetalBuffer;
use crate::encode::Binding;
use crate::stream::{MetalStream, Settled};

/// The fault word of a Metal device.
pub struct MetalFaultWord {
    buffer: MetalBuffer<u32>,
}

impl MetalFaultWord {
    /// A fault word on the device, clean.
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, DeviceError> {
        let word = Self {
            buffer: MetalBuffer::new(device, Elems(FAULT_WORDS))?,
        };
        // No stream has seen the buffer yet, so nothing is in flight.
        word.write_clean()?;
        Ok(word)
    }

    /// The word as a kernel binds it.
    #[must_use]
    pub fn binding(&self) -> Binding<'_> {
        Binding::whole(self.buffer.metal())
    }

    /// Every word, once `s`'s work so far is done: what the gates compare with
    /// the host model of the word.
    pub fn words(&self, s: &MetalStream) -> Result<Vec<u32>, DeviceError> {
        s.sync()?;
        Ok(self.buffer.pages().to_vec())
    }

    /// The fault the word holds, read on `settled`: the stream's proof that
    /// the recording which raises into this word is done, so the read waits
    /// for nothing. The caller pairs the word with that recording; work
    /// still running may raise into another word, not this one.
    pub fn read_settled(&self, settled: &Settled) -> Result<Option<Fault>, DeviceError> {
        let _ = settled;
        let pages = self.buffer.pages();
        Fault::read_from(|i| pages.load(i, Ordering::Relaxed))
    }

    /// The word clean again, on `settled`, under the pairing
    /// [`MetalFaultWord::read_settled`] reads under: no work still running
    /// raises into this word.
    pub fn clear_settled(&self, settled: &Settled) -> Result<(), DeviceError> {
        let _ = settled;
        self.write_clean()
    }

    /// Word 0 clean and every mask empty.
    fn write_clean(&self) -> Result<(), DeviceError> {
        let pages = self.buffer.pages();
        pages.store(0, FAULT_NONE, Ordering::Relaxed)?;
        for i in 1..FAULT_WORDS {
            pages.store(i, 0, Ordering::Relaxed)?;
        }
        Ok(())
    }
}

impl FaultWord<MetalStream> for MetalFaultWord {
    fn read(&self, s: &MetalStream) -> Result<Option<Fault>, DeviceError> {
        s.sync()?;
        let pages = self.buffer.pages();
        Fault::read_from(|i| pages.load(i, Ordering::Relaxed))
    }

    fn clear(&self, s: &MetalStream) -> Result<(), DeviceError> {
        s.sync()?;
        self.write_clean()
    }
}
