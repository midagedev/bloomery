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
//! host prints a mask in the architecture's step order ([`step_order`]).
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
//! through, the quantizers' finite test, the reader of a `Gpu`'s allocation
//! and the map from an architecture to its step order.
//!
//! A kernel gets the allocation as a [`FaultSink`] launch argument: its
//! device address and the layer the launch belongs to, one by-value
//! parameter. The site is the kernel's own constant. Only
//! [`crate::Gpu::fault_sink`] makes a sink, from the allocation it owns, so a
//! sink always addresses a live fault of that `Gpu`'s context.

pub use bloomery_device::fault::*;

use cuda_device::ptx_asm;

/// The step orders a fault's mask prints in: the device side's tables, and
/// the map from an architecture to its own.
pub mod step_order {
    pub use bloomery_device::fault::step_order::*;

    use bloomery_device::fault::StepOrder;

    /// The order of `arch`'s step, named as the architecture's `Debug` form.
    #[must_use]
    pub fn of(arch: model::arch::Arch) -> StepOrder {
        use model::arch::Arch;
        let (name, sites) = match arch {
            Arch::Deepseek2 => ("Deepseek2", DEEPSEEK2),
            Arch::Deepseek41 => ("Deepseek41", DEEPSEEK41),
            Arch::Qwen3moe => ("Qwen3moe", QWEN3MOE),
            Arch::Qwen35moe => ("Qwen35moe", QWEN35MOE),
            Arch::Glm5next => ("Glm5next", GLM5NEXT),
            Arch::Mimo2 => ("Mimo2", MIMO2),
            Arch::SolarOpen2 => ("SolarOpen2", SOLAR2),
            Arch::Raon1 => ("Raon1", SOLAR2),
            Arch::MiniMaxM3 => ("MiniMaxM3", MINIMAXM3),
            // No program runs nemotron_h. The listing is MiMo's so a fault
            // word still decodes; it is not a step this architecture runs.
            Arch::NemotronH => ("NemotronH", NEMOTRON_H),
            Arch::NemotronHMoe => ("NemotronHMoe", NEMOTRON_H_MOE),
            Arch::GemmaEmbedding => ("GemmaEmbedding", GEMMA_ENCODER),
            Arch::GemmaEmbedding2 => ("GemmaEmbedding2", GEMMA_ENCODER),
        };
        StepOrder { arch: name, sites }
    }
}

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
    /// becomes the minimum of itself and `(layer << 8) | site`, and the
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
        let code = (self.layer << 8) | site as u32;
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
        if word == FAULT_NONE || word >> 8 > LAYER_NONE {
            return 0;
        }
        // SAFETY: `word >> 8 <= LAYER_NONE` for every word a raise stores,
        // so the mask index is below FAULT_WORDS, inside the allocation the
        // constructor's contract names. Volatile as in `read`.
        unsafe { core::ptr::read_volatile(self.word.add(mask_index(word >> 8))) }
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

/// What the fault `word` (a `Gpu`'s fault allocation, [`FAULT_WORDS`] u32s or
/// more) holds now: the first-layer word and that layer's site mask, each read
/// by a synchronous copy that waits on no stream of the engine (they are all
/// non-blocking), and decoded by the device side's one rule
/// ([`Fault::read_from`]). The one reader of the word: the caller orders it —
/// the engine's reader synchronizes its stream first ([`crate::Gpu::fault`]),
/// a batch service runs after the event its handoffs came down at, so every
/// kernel that wrote them has finished.
pub fn read(word: &cuda_core::DeviceBuffer<u32>) -> Result<Option<Fault>, crate::GpuError> {
    use cuda_core::sys;
    word.context().bind_to_thread()?;
    Fault::read_from(|i: usize| -> Result<u32, crate::GpuError> {
        let mut v = FAULT_NONE;
        // SAFETY: word `i` lies inside the allocation (`read_from` asks for
        // word 0 or the mask index of a layer at most LAYER_NONE, below
        // FAULT_WORDS, which every holder of the word checked its length
        // against — `module_fault_word` and the `Gpu`'s own allocation); the
        // destination is `v`, four bytes that outlive the synchronous copy;
        // the context is current on this thread (bound above).
        let rc = unsafe {
            sys::cuMemcpyDtoH_v2(
                (&raw mut v).cast(),
                word.cu_deviceptr() + 4 * i as u64,
                std::mem::size_of::<u32>(),
            )
        };
        crate::graph::cu(rc, "cuMemcpyDtoH_v2 (fault word)")?;
        Ok(v)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::arch::Arch;

    /// Every architecture; a new one is a compile error here until it is
    /// listed (and [`step_order::of`] gives it a table).
    fn every_arch() -> [Arch; 13] {
        let all = [
            Arch::Deepseek2,
            Arch::Deepseek41,
            Arch::Qwen3moe,
            Arch::Qwen35moe,
            Arch::Glm5next,
            Arch::Mimo2,
            Arch::SolarOpen2,
            Arch::MiniMaxM3,
            Arch::Raon1,
            Arch::NemotronH,
            Arch::NemotronHMoe,
            Arch::GemmaEmbedding,
            Arch::GemmaEmbedding2,
        ];
        for a in all {
            match a {
                Arch::Deepseek2
                | Arch::Deepseek41
                | Arch::Qwen3moe
                | Arch::Qwen35moe
                | Arch::Glm5next
                | Arch::Mimo2
                | Arch::SolarOpen2
                | Arch::MiniMaxM3
                | Arch::Raon1
                | Arch::NemotronH
                | Arch::NemotronHMoe
                | Arch::GemmaEmbedding
                | Arch::GemmaEmbedding2 => {}
            }
        }
        all
    }

    #[test]
    fn every_step_order_names_every_site_exactly_once() {
        for arch in every_arch() {
            let order = step_order::of(arch).sites;
            assert_eq!(
                order.len(),
                FaultSite::ALL.len(),
                "{arch:?}'s step order has {} sites, FaultSite has {}",
                order.len(),
                FaultSite::ALL.len()
            );
            for site in FaultSite::ALL {
                let n = order.iter().filter(|s| *s == site).count();
                assert_eq!(n, 1, "{arch:?}'s step order names {site:?} {n} times");
            }
        }
    }

    /// A fault's print names the architecture as its `Debug` form does.
    #[test]
    fn every_step_order_is_named_as_its_architecture() {
        for arch in every_arch() {
            assert_eq!(step_order::of(arch).arch, format!("{arch:?}"));
        }
    }

    /// MiMo-V2.6-Flash's order opens with the six sites its step raises, in
    /// the order its launches meet them: the host's token check, the append's
    /// position, the flash's key count, the router, the handoff, the head.
    #[test]
    fn the_mimo2_order_leads_with_the_sites_its_step_raises() {
        assert_eq!(
            step_order::of(Arch::Mimo2).sites[..6],
            [
                FaultSite::TokenId,
                FaultSite::CachePos,
                FaultSite::KeyCount,
                FaultSite::Router,
                FaultSite::ExpertId,
                FaultSite::Logit,
            ]
        );
    }

    /// Solar-Open2's order opens with the fourteen sites its step raises, in
    /// the order a layer's launches meet them: the host's token check, the
    /// norm, the quantizers, the KDA conv, delta step (lane word, then stamp)
    /// and gated norm, the GQA append and flash, the router, the handoff, the
    /// head. The out gate's product is the norm's site, standing first.
    #[test]
    fn the_solar2_order_leads_with_the_sites_its_step_raises() {
        assert_eq!(
            step_order::of(Arch::SolarOpen2).sites[..14],
            [
                FaultSite::TokenId,
                FaultSite::F32Product,
                FaultSite::QuantColumn,
                FaultSite::Q5Quant,
                FaultSite::LinearConv,
                FaultSite::LinearDelta,
                FaultSite::DeltaLane,
                FaultSite::DeltaStamp,
                FaultSite::LinearGate,
                FaultSite::CachePos,
                FaultSite::KeyCount,
                FaultSite::Router,
                FaultSite::ExpertId,
                FaultSite::Logit,
            ]
        );
    }

    #[test]
    fn a_fault_prints_its_mask_in_step_order_and_says_when_it_cannot() {
        let f = Fault::of_sites(3, &[FaultSite::KeyCount, FaultSite::QuantColumn]);
        assert_eq!((f.code, f.sites), (1, (1 << 1) | (1 << 10)));
        assert_eq!(Fault::from_words(f.word(), f.sites), Some(f));
        assert_eq!(Fault::from_words(FAULT_NONE, 0), None);
        let unknown = f.to_string();
        assert!(
            unknown.contains("code order") && unknown.contains("quant_column, key_count"),
            "{unknown}"
        );
        let qwen = f.in_order(step_order::of(Arch::Qwen3moe));
        assert_eq!(qwen, f, "the step order is not part of equality");
        let shown = qwen.to_string();
        assert!(
            shown.contains("Qwen3moe step order") && shown.contains("key_count, quant_column"),
            "{shown}"
        );
        // A mask bit past every site prints as its code: the bit after the highest site's.
        let past = FaultSite::ALL.iter().map(|&s| s as u32).max().unwrap_or(0) + 1;
        let odd = Fault::from_words((3 << 8) | 1, (1 << 1) | (1 << past)).unwrap();
        assert!(odd.to_string().contains(&format!("code {past}")), "{odd}");
    }

    #[test]
    fn the_cause_prints_once_in_an_error_and_not_in_the_plain_form() {
        let f = Fault::at(7, FaultSite::ExpertId);
        let cause = FaultSite::ExpertId.cause();
        let plain = f.to_string();
        assert!(
            plain.contains("site expert_id") && !plain.contains(cause),
            "{plain}"
        );
        assert_eq!(format!("{f:#}").matches(cause).count(), 1, "{f:#}");
        let line = format!("word \"{f}\" (want \"{f}\")");
        assert!(!line.contains(cause), "{line}");
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
