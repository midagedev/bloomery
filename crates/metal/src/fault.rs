//! The Metal fault word: the pages the kernels raise.
//!
//! The word is one shared buffer of [`FAULT_WORDS`] u32s: word 0 is the
//! first-layer word, and word `1 + layer` is that layer's site mask. A
//! kernel raises it with an atomic minimum on word 0 and an atomic OR on
//! its own layer's mask — the same two operations the contract's host model
//! does. The host reads and clears it as atomic words, after the stream it is
//! asked against has finished its work.

use core::sync::atomic::Ordering;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use bloomery_device::fault::{FAULT_NONE, FAULT_WORDS, Fault, FaultWord, LAYER_NONE, mask_index};
use bloomery_device::{DeviceError, Elems, Stream};

use crate::buffer::MetalBuffer;
use crate::encode::Binding;
use crate::stream::MetalStream;

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
        word.write_clean();
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
        Ok(self
            .buffer
            .pages()
            .atomics()
            .iter()
            .map(|w| w.load(Ordering::Relaxed))
            .collect())
    }

    /// Word 0 clean and every mask empty.
    fn write_clean(&self) {
        let words = self.buffer.pages().atomics();
        words[0].store(FAULT_NONE, Ordering::Relaxed);
        for mask in &words[1..] {
            mask.store(0, Ordering::Relaxed);
        }
    }
}

impl FaultWord<MetalStream> for MetalFaultWord {
    fn read(&self, s: &MetalStream) -> Result<Option<Fault>, DeviceError> {
        s.sync()?;
        let words = self.buffer.pages().atomics();
        let first = words[0].load(Ordering::Relaxed);
        // A word naming a layer past the last has no mask; it reads as a
        // fault with no sites, as the host model does.
        let sites = if first == FAULT_NONE || first >> 8 > LAYER_NONE {
            0
        } else {
            words[mask_index(first >> 8)].load(Ordering::Relaxed)
        };
        Ok(Fault::from_words(first, sites))
    }

    fn clear(&self, s: &MetalStream) -> Result<(), DeviceError> {
        s.sync()?;
        self.write_clean();
        Ok(())
    }
}
