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
//! A kernel gets the allocation as a [`FaultSink`] launch argument: its
//! device address and the layer the launch belongs to, one by-value
//! parameter. The site is the kernel's own constant. Only
//! [`crate::Gpu::fault_sink`] makes a sink, from the allocation it owns, so a
//! sink always addresses a live fault of that `Gpu`'s context.

use cuda_device::ptx_asm;

/// The fault word's value while no fault has been raised.
pub const FAULT_NONE: u32 = u32::MAX;

/// The layer a launch outside every layer raises with: the output head.
pub const LAYER_HEAD: u32 = 0xFFFE;

/// The layer a launch raises with when its caller names none — a gate's
/// direct launch, or an architecture whose chain does not label its
/// quantizers.
pub const LAYER_NONE: u32 = 0xFFFF;

/// The fault allocation's length in u32s that carry meaning: the first-layer
/// word, then one site mask for every layer up to [`LAYER_NONE`].
pub const FAULT_WORDS: usize = 1 + LAYER_NONE as usize + 1;

/// The word index of layer `layer`'s site mask in the fault allocation.
#[inline(always)]
#[must_use]
pub const fn mask_index(layer: u32) -> usize {
    1 + layer as usize
}

/// Where a fault was raised: the kernel family that refused its input. The
/// codes are fixed — a gate pins the word they make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FaultSite {
    /// The q8_1 activation quantizer (`q8_1_quant_vals`: `q3k_quantize_q8_1`
    /// and every entry that runs its body) met a non-finite value.
    QuantColumn = 1,
    /// `norm_quant` normalized a column to a non-finite value, or to a zero
    /// scale.
    NormQuant = 2,
    /// The 32-value q8 quantizer of the Q5 path met a non-finite value.
    Q5Quant = 3,
    /// `ds41_hc_pre`'s in-register q8_1 activation met a non-finite value.
    HcQuant = 4,
    /// A router met a non-finite logit, found fewer finite candidates than
    /// slots, or produced a non-finite weight.
    Router = 5,
    /// A selected-attention list named a row past the compressed stream.
    AttnSel = 6,
    /// A grouped GEMM's route table met an expert id past its stack.
    ExpertId = 7,
    /// An attention launch was handed a visible count past the rows of its
    /// key source: the window ring's, or the compressed stream's.
    AttnCount = 8,
    /// A token id that selects a table row — an embedding row, the DSpark
    /// Markov head's previous token — lies past the table's rows.
    TokenId = 9,
    /// A prefill flash row's live key count was zero or past the cache.
    KeyCount = 10,
    /// A cache append was handed a position at or past the cache's rows.
    CachePos = 11,
}

// A site is one bit of a u32 mask.
const _: () = assert!((FaultSite::CachePos as u32) < 32);

impl FaultSite {
    /// Every site, in code order.
    pub const ALL: &[FaultSite] = &[
        FaultSite::QuantColumn,
        FaultSite::NormQuant,
        FaultSite::Q5Quant,
        FaultSite::HcQuant,
        FaultSite::Router,
        FaultSite::AttnSel,
        FaultSite::ExpertId,
        FaultSite::AttnCount,
        FaultSite::TokenId,
        FaultSite::KeyCount,
        FaultSite::CachePos,
    ];

    /// The site's name as an error prints it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            FaultSite::QuantColumn => "quant_column",
            FaultSite::NormQuant => "norm_quant",
            FaultSite::Q5Quant => "q5_quant",
            FaultSite::HcQuant => "hc_quant",
            FaultSite::Router => "router",
            FaultSite::AttnSel => "attn_sel",
            FaultSite::ExpertId => "expert_id",
            FaultSite::AttnCount => "attn_count",
            FaultSite::TokenId => "token_id",
            FaultSite::KeyCount => "key_count",
            FaultSite::CachePos => "cache_pos",
        }
    }

    /// What the site refused, for the error's text.
    #[must_use]
    pub fn cause(self) -> &'static str {
        match self {
            FaultSite::QuantColumn
            | FaultSite::NormQuant
            | FaultSite::Q5Quant
            | FaultSite::HcQuant => "non-finite activation",
            FaultSite::Router => {
                "a non-finite router logit, fewer finite candidates than slots, or a non-finite weight"
            }
            FaultSite::AttnSel => "a selected row past the compressed stream",
            FaultSite::ExpertId => "a routed expert id past the stack's expert count",
            FaultSite::AttnCount => "a visible key count past the rows of its source",
            FaultSite::TokenId => "a token id past the table's rows",
            FaultSite::KeyCount => "a flash row's live key count of zero or past the cache",
            FaultSite::CachePos => "a cache append position at or past the cache's rows",
        }
    }
}

/// The order a layer's step meets the fault sites in, per architecture —
/// the order a site mask prints in. Each table names every site exactly
/// once, at its *first* occurrence on the decode path: a site a step meets
/// twice (the norm and the q8_1 quantizer run before attention and again
/// around the experts) stands where it is first met, and the sites an
/// architecture never raises follow the ones it does, so no bit of a mask
/// goes unprinted. The embedding's [`FaultSite::TokenId`] comes first: it is
/// raised before any layer (as an unlabelled launch).
pub mod step_order {
    use super::FaultSite;
    use super::FaultSite::{
        AttnCount, AttnSel, CachePos, ExpertId, HcQuant, KeyCount, NormQuant, Q5Quant, QuantColumn,
        Router, TokenId,
    };

    /// DeepSeek-V2-Lite (`arch::deepseek2`): the fused norm and quantizer at
    /// the layer's entry, the key path's cache append, the attention
    /// output's q8_1 quantizer (in the flash launch or its own), the router,
    /// the card's slot list and the routed `_sel` kernels, the down
    /// projection's 32-value quantizer.
    pub const DEEPSEEK2: &[FaultSite] = &[
        TokenId,
        NormQuant,
        CachePos,
        QuantColumn,
        Router,
        ExpertId,
        Q5Quant,
        AttnCount,
        AttnSel,
        KeyCount,
        HcQuant,
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
        AttnCount,
        AttnSel,
        Router,
        ExpertId,
        KeyCount,
        Q5Quant,
        CachePos,
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
        KeyCount,
        QuantColumn,
        Router,
        ExpertId,
        AttnCount,
        AttnSel,
        Q5Quant,
        HcQuant,
    ];

    /// The order of `arch`'s step.
    #[must_use]
    pub fn of(arch: model::arch::Arch) -> &'static [FaultSite] {
        match arch {
            model::arch::Arch::Deepseek2 => DEEPSEEK2,
            model::arch::Arch::Deepseek41 => DEEPSEEK41,
            model::arch::Arch::Qwen3moe => QWEN3MOE,
        }
    }
}

/// A raised fault, decoded from the pair the card holds: the first layer a
/// launch raised in, the smallest site code raised there, and the mask of
/// every site raised there (`1 << code` per site). A code that is no
/// [`FaultSite`] is kept as it came, so a corrupt word still reads as a
/// fault rather than as clean. Equality is on those three; the step order a
/// fault prints its mask in ([`Fault::in_arch`]) is presentation.
#[derive(Clone, Copy, Debug)]
pub struct Fault {
    /// The raising launch's layer: a model layer, [`LAYER_HEAD`] or
    /// [`LAYER_NONE`].
    pub layer: u32,
    /// The raw site code (`FaultSite as u32` for every known site): the
    /// smallest raised in `layer`.
    pub code: u32,
    /// Every site raised in `layer`, bit `code` per site.
    pub sites: u32,
    /// The architecture whose step order the mask prints in, once a model
    /// has named it.
    arch: Option<model::arch::Arch>,
}

impl PartialEq for Fault {
    fn eq(&self, other: &Fault) -> bool {
        (self.layer, self.code, self.sites) == (other.layer, other.code, other.sites)
    }
}

impl Eq for Fault {}

impl Fault {
    /// The fault a first-layer word and that layer's mask hold, `None` for
    /// [`FAULT_NONE`].
    #[must_use]
    pub fn from_words(word: u32, sites: u32) -> Option<Fault> {
        (word != FAULT_NONE).then_some(Fault {
            layer: word >> 8,
            code: word & 0xFF,
            sites,
            arch: None,
        })
    }

    /// The fault of `site` raised alone in `layer`: what a gate that plants
    /// one fault expects.
    #[must_use]
    pub fn at(layer: u32, site: FaultSite) -> Fault {
        Fault::of_sites(layer, &[site])
    }

    /// The fault of every site in `sites` raised in `layer` and nowhere
    /// earlier. An empty list is no fault a card can hold; it reads as code
    /// 0 with an empty mask.
    #[must_use]
    pub fn of_sites(layer: u32, sites: &[FaultSite]) -> Fault {
        Fault {
            layer,
            code: sites.iter().map(|s| *s as u32).min().unwrap_or(0),
            sites: sites.iter().fold(0, |m, s| m | (1 << *s as u32)),
            arch: None,
        }
    }

    /// The first-layer word this fault is stored as.
    #[must_use]
    pub fn word(self) -> u32 {
        (self.layer << 8) | self.code
    }

    /// This fault, printing its mask in `arch`'s step order.
    #[must_use]
    pub fn in_arch(self, arch: model::arch::Arch) -> Fault {
        Fault {
            arch: Some(arch),
            ..self
        }
    }

    /// The site, when the code is a known one.
    #[must_use]
    pub fn site(self) -> Option<FaultSite> {
        FaultSite::ALL
            .iter()
            .copied()
            .find(|s| *s as u32 == self.code)
    }

    /// The mask's sites in `order`, then any bit `order` does not name, as
    /// its raw code: nothing the card raised is left out.
    #[must_use]
    pub fn sites_in(self, order: &[FaultSite]) -> Vec<Result<FaultSite, u32>> {
        let mut out: Vec<Result<FaultSite, u32>> = order
            .iter()
            .filter(|s| self.sites & (1 << **s as u32) != 0)
            .map(|s| Ok(*s))
            .collect();
        for bit in 0..32 {
            if self.sites & (1 << bit) != 0 && !order.iter().any(|s| *s as u32 == bit) {
                out.push(Err(bit));
            }
        }
        out
    }
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let where_ = match self.layer {
            LAYER_HEAD => "the output head".to_string(),
            LAYER_NONE => "an unlabelled launch".to_string(),
            l => format!("layer {l}"),
        };
        write!(f, "{where_}")?;
        match self.site() {
            Some(s) => write!(f, " site {} ({})", s.name(), s.cause())?,
            None => write!(f, " site code {} (unknown)", self.code)?,
        }
        let (order, named) = match self.arch {
            Some(a) => (step_order::of(a), format!("{a:?} step order")),
            None => (
                FaultSite::ALL,
                "architecture unknown, so in code order, not step order".to_string(),
            ),
        };
        let names: Vec<String> = self
            .sites_in(order)
            .into_iter()
            .map(|s| match s {
                Ok(s) => s.name().to_string(),
                Err(code) => format!("code {code}"),
            })
            .collect();
        write!(f, "; sites raised in {where_} ({named}): ")?;
        if names.is_empty() {
            write!(f, "none recorded (mask 0)")
        } else {
            write!(f, "{}", names.join(", "))
        }
    }
}

/// A kernel's handle on the fault: the allocation's device address and the
/// layer of the launch. `Copy`, one by-value launch parameter; made only by
/// [`crate::Gpu::fault_sink`] from the allocation that `Gpu` owns, which
/// outlives every launch on its stream and every graph captured there (a
/// graph drops before the buffers it addresses).
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
    pub(crate) unsafe fn new(word: *mut u32, layer: u32) -> FaultSink {
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

#[cfg(test)]
mod tests {
    use super::*;
    use model::arch::Arch;

    /// Every architecture; a new one is a compile error here until it is
    /// listed (and [`step_order::of`] gives it a table).
    fn every_arch() -> [Arch; 3] {
        let all = [Arch::Deepseek2, Arch::Deepseek41, Arch::Qwen3moe];
        for a in all {
            match a {
                Arch::Deepseek2 | Arch::Deepseek41 | Arch::Qwen3moe => {}
            }
        }
        all
    }

    #[test]
    fn every_step_order_names_every_site_exactly_once() {
        for arch in every_arch() {
            let order = step_order::of(arch);
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
        let qwen = f.in_arch(Arch::Qwen3moe);
        assert_eq!(qwen, f, "the step order is not part of equality");
        let shown = qwen.to_string();
        assert!(shown.contains("key_count, quant_column"), "{shown}");
        let odd = Fault::from_words((3 << 8) | 1, (1 << 1) | (1 << 20)).unwrap();
        assert!(odd.to_string().contains("code 20"), "{odd}");
    }
}
