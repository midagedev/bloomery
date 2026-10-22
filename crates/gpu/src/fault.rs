//! The card's fault word: how a kernel that cannot panic refuses an input.
//!
//! A kernel that meets input it has no defined answer for — a non-finite
//! activation in a quantizer, a router that cannot fill its six slots, an
//! index past the stream it gathers from — keeps its memory accesses defined
//! and raises a fault instead of writing a plausible value and moving on.
//!
//! The fault lives in one allocation per [`crate::Gpu`]: word 0 is the
//! *first-layer word*, [`FAULT_NONE`] while clean, and word `1 + layer` is
//! layer `layer`'s *site mask*, 0 while clean. A raise stores `(layer << 8) |
//! site` into word 0 with an atomic minimum and ORs `1 << site` into its own
//! layer's mask. However many threads raise, word 0 ends up naming the
//! earliest layer and the smallest site code raised there, and that layer's
//! mask every site raised in it: deterministic, and a pair a gate can pin.
//! The mask is per layer because a non-finite value raised in one layer is
//! met again by the quantizers of every later one; only the first layer's
//! sites say where it started. Site codes do not follow step order, so the
//! host prints a mask in the architecture's step order
//! ([`crate::arch::step_order`]).
//!
//! The pair is sticky. The head's argmax copies the word and the first
//! layer's mask next to the token it writes ([`crate::elem`]'s
//! `argmax_fault`), so the step's one readback carries both with no launch
//! and no copy of its own; nothing on the card clears them. A prompt of
//! several tokens is several bodies before one readback, and a fault in the
//! first must still be there at the last. The host clears them
//! ([`crate::Gpu::clear_fault`]) when the state they condemned is gone: a
//! model's `reset`, or a gate that has asserted them.
//!
//! The layout, the site codes, the decoding and the host model of a raise
//! have one owner for every backend, `bloomery_device::fault`, which this
//! module re-exports; a kernel reads its constants and [`mask_index`] from
//! there too. What is the card's own stays here: the sink a kernel raises
//! through, the quantizers' finite test, and the reader of a `Gpu`'s allocation;
//! the step order of each architecture is the arch registry's
//! ([`crate::arch::step_order`]).
//!
//! A kernel gets the allocation as a [`FaultSink`] launch argument: its
//! device address and the layer the launch belongs to, one by-value
//! parameter. The site is the kernel's own constant. Only
//! [`crate::Gpu::fault_sink`] makes a sink, from the allocation it owns, so a
//! sink always addresses a live fault of that `Gpu`'s context.

pub use bloomery_device::fault::*;

use cuda_device::ptx_asm;

/// A kernel's handle on the fault: the allocation's device address and the
/// layer of the launch. `Copy`, one by-value launch parameter. The allocation
/// must outlive every launch and captured graph that carries the sink.
/// [`crate::Gpu::fault_sink`] uses the allocation that `Gpu` owns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FaultSink {
    word: *mut u32,
    layer: u32,
}

impl FaultSink {
    /// The sink over the word at device address `word`, for launches of
    /// `layer`.
    ///
    /// # Safety
    ///
    /// `word` addresses [`FAULT_WORDS`] live device `u32`s of the context
    /// the sink's launches run in, `layer <= LAYER_NONE`, and the
    /// allocation stays alive while any launch or captured graph that
    /// carries the sink can run.
    pub unsafe fn new(word: *mut u32, layer: u32) -> FaultSink {
        FaultSink { word, layer }
    }

    /// Raise a fault at `site` for this sink's layer: the first-layer word
    /// becomes the minimum of itself and `(layer << LAYER_SHIFT) | site`, and the
    /// layer's mask takes bit `site`. Device code; any number of threads may
    /// raise.
    ///
    /// Relaxed device-scope `red.global.min` and `red.global.or`, spelled in
    /// PTX: the pointer arrives inside a by-value parameter, so the compiler
    /// sees a generic address, and its atomics on a generic address branch on
    /// `isspacep.local` into a thread-private load/store twin — a local path
    /// in every raising kernel that the fault, a global allocation, never
    /// takes.
    #[inline(always)]
    pub fn raise(self, site: FaultSite) {
        let code = (self.layer << LAYER_SHIFT) | site as u32;
        let bit = 1u32 << site as u32;
        let at = 4 * mask_index(self.layer) as u64;
        // SAFETY: `word` addresses FAULT_WORDS live u32s of global memory by
        // the constructor's contract (a `DeviceBuffer` of the context) and
        // `layer <= LAYER_NONE`, so `word + at` is the layer's mask inside
        // it; the generic-to-global conversion is exact; every access to the
        // allocation on the card is these reductions or the argmax's
        // volatile copies.
        unsafe {
            ptx_asm!(
                "{ .reg .b64 g;\n\t\
                   cvta.to.global.u64 g, %0;\n\t\
                   red.relaxed.gpu.global.min.u32 [g], %1;\n\t\
                   add.u64 g, g, %2;\n\t\
                   red.relaxed.gpu.global.or.b32 [g], %3; }",
                in("l") self.word,
                in("r") code,
                in("l") at,
                in("r") bit,
                clobber("memory")
            );
        }
    }

    /// The first-layer word as it stands. Device code: the argmax's copy
    /// into its readback buffer, after every kernel of the step before it.
    #[inline(always)]
    pub fn read(self) -> u32 {
        // SAFETY: `word` addresses a live device u32 by the constructor's
        // contract. Volatile: the value must come from memory on every
        // launch and replay.
        unsafe { core::ptr::read_volatile(self.word) }
    }

    /// The site mask of the layer `word` (a [`FaultSink::read`] value)
    /// names, 0 for [`FAULT_NONE`] (and for a word no raise can store, which
    /// still reads back as a fault). Device code: the argmax's second copy.
    #[inline(always)]
    pub fn read_sites(self, word: u32) -> u32 {
        if word == FAULT_NONE || word >> LAYER_SHIFT > LAYER_NONE {
            return 0;
        }
        // SAFETY: `word >> LAYER_SHIFT <= LAYER_NONE` for every word a raise
        // stores, so the mask index is below FAULT_WORDS, inside the
        // allocation the constructor's contract names. Volatile as in `read`.
        unsafe { core::ptr::read_volatile(self.word.add(mask_index(word >> LAYER_SHIFT))) }
    }
}

/// Whether all four values are finite — the quantizers' test, one per lane
/// quad.
#[inline(always)]
#[must_use]
pub fn quad_finite(v: [f32; 4]) -> bool {
    // `&`, not `&&`: four tests and no branch on the default path.
    v[0].is_finite() & v[1].is_finite() & v[2].is_finite() & v[3].is_finite()
}

/// `into.len()` words of the fault allocation `word` from word `first`, by a
/// synchronous copy that waits on no stream of the engine: the caller orders
/// it ([`read`]). A range past the allocation is refused by name.
pub fn read_words(
    word: &cuda_core::DeviceBuffer<u32>,
    first: usize,
    into: &mut [u32],
) -> Result<(), crate::GpuError> {
    use cuda_core::sys;
    let past = first
        .checked_add(into.len())
        .filter(|&end| end <= word.len());
    if past.is_none() {
        return Err(crate::GpuError::shape(
            "fault::read_words",
            format!(
                "words {first}..+{} pass the allocation's {} words",
                into.len(),
                word.len()
            ),
        ));
    }
    word.context().bind_to_thread()?;
    // SAFETY: the source range lies inside the allocation (checked above);
    // the destination is `into`, `4 * into.len()` bytes that outlive the
    // synchronous copy; the context is current on this thread (bound above).
    let rc = unsafe {
        sys::cuMemcpyDtoH_v2(
            into.as_mut_ptr().cast(),
            word.cu_deviceptr() + 4 * first as u64,
            std::mem::size_of_val(into),
        )
    };
    crate::graph::cu(rc, "cuMemcpyDtoH_v2 (fault word)")
}

/// What the fault `word` (a `Gpu`'s fault allocation, [`FAULT_WORDS`] u32s or
/// more) holds now: the first-layer word and that layer's site mask, each read
/// by [`read_words`] and decoded by the device side's one rule
/// ([`Fault::read_from`]). The one reader of the word: the caller orders it —
/// the engine's reader synchronizes its stream first ([`crate::Gpu::fault`]),
/// a batch service runs after the event its handoffs came down at, so every
/// kernel that wrote them has finished.
pub fn read(word: &cuda_core::DeviceBuffer<u32>) -> Result<Option<Fault>, crate::GpuError> {
    Fault::read_from(|i: usize| -> Result<u32, crate::GpuError> {
        let mut v = [FAULT_NONE];
        read_words(word, i, &mut v)?;
        Ok(v[0])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's errors that carry a fault name its cause once: the device
    /// side's `Display` is pinned in `bloomery_device::fault`, this is the
    /// wrapping.
    #[test]
    fn an_engine_error_names_the_cause_once() {
        let f = Fault::at(7, FaultSite::ExpertId);
        let cause = FaultSite::ExpertId.cause();
        for e in [
            crate::GpuError::fault("step", f),
            crate::GpuError::Poisoned {
                what: "step",
                fault: f,
            },
        ] {
            let text = e.to_string();
            assert_eq!(text.matches(cause).count(), 1, "{text}");
        }
    }
}
