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
//! and the step order of each architecture, with the map to it.
//!
//! A kernel gets the allocation as a [`FaultSink`] launch argument: its
//! device address and the layer the launch belongs to, one by-value
//! parameter. The site is the kernel's own constant. Only
//! [`crate::Gpu::fault_sink`] makes a sink, from the allocation it owns, so a
//! sink always addresses a live fault of that `Gpu`'s context.

pub use bloomery_device::fault::*;

use cuda_device::ptx_asm;

/// The order a layer's step meets the fault sites in, per architecture —
/// the order a site mask prints in — and the map from an architecture to its
/// own. Each table names every site exactly once, at its *first* occurrence on
/// the decode path: a site a step meets twice (the norm and the q8_1
/// quantizer run before attention and again around the experts) stands where
/// it is first met, and the sites an architecture never raises follow the
/// ones it does, so no bit of a mask goes unprinted. The embedding's
/// [`FaultSite::TokenId`] comes first: it is raised before any layer (as an
/// unlabelled launch).
pub mod step_order {
    use bloomery_device::fault::FaultSite;
    use bloomery_device::fault::FaultSite::{
        AttnCount, AttnSel, CachePos, CacheValue, CandMask, DeltaLane, DeltaStamp, ExpertId,
        F32Product, HcMix, HcQuant, KeyCount, KvQuant, LinearConv, LinearDelta, LinearGate, Logit,
        Mamba2Scan, NormQuant, Ple, PoolSelect, Q5Quant, QuantColumn, Router, TokenId,
    };
    use bloomery_device::fault::StepOrder;

    /// DeepSeek-V2-Lite (`arch::deepseek2`): the fused norm and quantizer at
    /// the layer's entry, the key path's cache append, the attention
    /// output's q8_1 quantizer (in the flash launch or its own), the router,
    /// the card's slot list and the routed `_sel` kernels, the down
    /// projection's 32-value quantizer.
    pub const DEEPSEEK2: &[FaultSite] = &[
        TokenId,
        NormQuant,
        CachePos,
        KvQuant,
        QuantColumn,
        Router,
        ExpertId,
        Q5Quant,
        AttnCount,
        AttnSel,
        KeyCount,
        HcQuant,
        LinearConv,
        LinearDelta,
        LinearGate,
        DeltaLane,
        CacheValue,
        PoolSelect,
        Ple,
        HcMix,
        F32Product,
        DeltaStamp,
        Logit,
        CandMask,
        Mamba2Scan,
    ];
    /// DeepSeek-V4.1 (`gpu-deepseek41`): HC_PRE's in-register quantizer, the
    /// attention norm, the projections' quantizer, the attention's visible
    /// counts and selected rows, then the MoE sub-layer's router and the
    /// routed experts (the handoff's id check among them).
    pub const DEEPSEEK41: &[FaultSite] = &[
        TokenId,
        HcQuant,
        NormQuant,
        QuantColumn,
        CandMask,
        AttnCount,
        AttnSel,
        Router,
        ExpertId,
        KeyCount,
        Q5Quant,
        CachePos,
        KvQuant,
        LinearConv,
        LinearDelta,
        LinearGate,
        DeltaLane,
        CacheValue,
        PoolSelect,
        Ple,
        HcMix,
        F32Product,
        DeltaStamp,
        Logit,
        Mamba2Scan,
    ];
    /// Qwen3-MoE (`arch::qwen3moe`): the fused norm and quantizer at the
    /// layer's entry, the cache append's position, the flash's key count,
    /// the attention output's quantizer, the router's fused norm, the routed
    /// experts. On the ubatch path the grouped GEMM's route (`ExpertId`) is
    /// enqueued before the router's norm; the table follows the decode path.
    pub const QWEN3MOE: &[FaultSite] = &[
        TokenId,
        NormQuant,
        CachePos,
        KvQuant,
        KeyCount,
        QuantColumn,
        Router,
        ExpertId,
        AttnCount,
        AttnSel,
        Q5Quant,
        HcQuant,
        LinearConv,
        LinearDelta,
        LinearGate,
        DeltaLane,
        CacheValue,
        PoolSelect,
        Ple,
        HcMix,
        F32Product,
        DeltaStamp,
        Logit,
        CandMask,
        Mamba2Scan,
    ];

    /// Qwen3.6-35B-A3B and Qwen3.8-Flash-Next (`arch::qwen3moe`'s `Body35`
    /// and `Body38`; both are `Arch::Qwen35moe`, so one order serves both).
    /// Qwen3.6's two layer kinds read in their launch order: the fused norm
    /// and quantizer at the layer's entry; an attention layer's cache append
    /// and flash key count, or a delta layer's conv, delta step (its lane
    /// word right after it) and gated norm; the output projection's
    /// quantizer; then the router's fused norm and the routed experts.
    /// Qwen3.8's sites stand beside them: its PLE site before its layer's
    /// mix, its gated-residual mix before the rest of the layer, its f32
    /// products before the output projection's quantizer. That is not every
    /// Qwen3.8 layer's launch order; among the differences, a delta layer's
    /// first [`F32Product`] is the shared expert's sum, raised after
    /// [`Router`] and [`ExpertId`], and a selecting layer raises
    /// [`PoolSelect`] (its raw key append first) before its out gate's
    /// [`F32Product`]. The mask names every site raised either way; only the
    /// order it prints them in differs.
    /// A q8_0 head's logits read after every layer's sites.
    pub const QWEN35MOE: &[FaultSite] = &[
        TokenId,
        Ple,
        HcMix,
        NormQuant,
        CachePos,
        KvQuant,
        KeyCount,
        LinearConv,
        LinearDelta,
        DeltaLane,
        DeltaStamp,
        LinearGate,
        F32Product,
        QuantColumn,
        Router,
        ExpertId,
        Logit,
        AttnCount,
        AttnSel,
        Q5Quant,
        HcQuant,
        PoolSelect,
        CacheValue,
        CandMask,
        Mamba2Scan,
    ];

    /// GLM-5.3-Flash (`gpu-glm5next`): a KDA layer's conv, delta step (its
    /// lane word right after it) and gated norm; a latent layer's two cache
    /// appends and its attention's visible count; then the router and the
    /// handoff's id check; the q8_0 head's logits last. The token id is checked on the host before the
    /// step (the embedding is a host row), and the hyper-connection and
    /// projection launches read f32 rows with nothing to raise; those sites
    /// follow the ones the step raises.
    pub const GLM5NEXT: &[FaultSite] = &[
        TokenId,
        LinearConv,
        LinearDelta,
        DeltaLane,
        LinearGate,
        CachePos,
        CacheValue,
        KvQuant,
        AttnCount,
        Router,
        ExpertId,
        Logit,
        HcQuant,
        NormQuant,
        QuantColumn,
        AttnSel,
        KeyCount,
        Q5Quant,
        PoolSelect,
        Ple,
        HcMix,
        F32Product,
        DeltaStamp,
        CandMask,
        Mamba2Scan,
    ];

    /// No program runs `nemotron_h`. The order is [`MIMO2`]'s so a fault word
    /// still decodes; it is not a step this architecture runs.
    pub const NEMOTRON_H: &[FaultSite] = MIMO2;

    /// The Gemma encoders have no step order of their own. The order is
    /// [`MIMO2`]'s so a fault word still decodes; it is not the order the
    /// `gemma-embedding2` encoder's pass raises its sites in.
    pub const GEMMA_ENCODER: &[FaultSite] = MIMO2;

    /// No program runs `nemotron_h_moe`. The order is [`MIMO2`]'s so a fault
    /// word still decodes; it is not a step this architecture runs.
    pub const NEMOTRON_H_MOE: &[FaultSite] = MIMO2;

    /// MiMo-V2.6-Flash (`gpu-mimo2`): the token id checked on the host before
    /// the step; a layer's cache append (position) and its flash (visible
    /// key count); the router and the handoff's id check; the q8_0 head's
    /// logits last. The projections and the norms read f32 rows with
    /// nothing to raise; those sites follow the ones the step raises.
    pub const MIMO2: &[FaultSite] = &[
        TokenId,
        CachePos,
        KeyCount,
        Router,
        ExpertId,
        Logit,
        QuantColumn,
        NormQuant,
        Q5Quant,
        HcQuant,
        AttnSel,
        AttnCount,
        LinearConv,
        LinearDelta,
        LinearGate,
        DeltaLane,
        CacheValue,
        PoolSelect,
        Ple,
        HcMix,
        F32Product,
        DeltaStamp,
        CandMask,
        KvQuant,
        Mamba2Scan,
    ];

    /// MiniMax-M3 (`gpu-minimaxm3`): the token id checked on the host before the step; a layer's cache append
    /// (position), the sparse layers' block pick (the selector's counts and the ids it keeps) and the flash's
    /// visible key count; the router and the handoff's id check; the q8_0 head's logits last. The projections and
    /// the norms read f32 rows with nothing to raise; the sites the step never raises follow in [`MIMO2`]'s order.
    pub const MINIMAXM3: &[FaultSite] = &[
        TokenId,
        CachePos,
        PoolSelect,
        KeyCount,
        Router,
        ExpertId,
        Logit,
        QuantColumn,
        NormQuant,
        Q5Quant,
        HcQuant,
        AttnSel,
        AttnCount,
        LinearConv,
        LinearDelta,
        LinearGate,
        DeltaLane,
        CacheValue,
        Ple,
        HcMix,
        F32Product,
        DeltaStamp,
        CandMask,
        KvQuant,
        Mamba2Scan,
    ];

    /// Solar-Open2 (`gpu-solar2`): the token id checked on the host before
    /// the step; a layer's norm and the q8_1 quantizer of its normed row, the
    /// Q5_0 halves' 32-value quantizer, a KDA layer's conv, delta step (its
    /// lane word and stamp right after it) and gated norm, a GQA layer's
    /// cache append (position), flash (visible key count) and out gate's
    /// product (the norm's [`F32Product`] already stands first); then the
    /// router and the handoff's id check, and the head's logits last. The
    /// order of the sites the step raises is the order a KDA layer's launches
    /// meet them, then the sites only a GQA layer meets; the sites the step
    /// never raises follow. Raon-1 (`arch::raon1`) runs the same order: an RDA
    /// layer launches a KDA layer's sites, and a latent layer's cache append
    /// and visible count stand where a GQA layer's do.
    pub const SOLAR2: &[FaultSite] = &[
        TokenId,
        F32Product,
        QuantColumn,
        Q5Quant,
        LinearConv,
        LinearDelta,
        DeltaLane,
        DeltaStamp,
        LinearGate,
        CachePos,
        KeyCount,
        Router,
        ExpertId,
        Logit,
        NormQuant,
        HcQuant,
        AttnCount,
        AttnSel,
        CacheValue,
        KvQuant,
        PoolSelect,
        Ple,
        HcMix,
        CandMask,
        Mamba2Scan,
    ];

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
