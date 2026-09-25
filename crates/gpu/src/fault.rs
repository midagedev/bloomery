//! The card's fault word: how a kernel that cannot panic refuses an input.
//!
//! A kernel that meets input it has no defined answer for — a non-finite
//! activation in a quantizer, a router that cannot fill its six slots, an
//! index past the stream it gathers from — keeps its memory accesses defined
//! and raises a fault instead of writing a plausible value and moving on.
//! The fault is one `u32` per [`crate::Gpu`], [`FAULT_NONE`] while clean. A
//! raise stores `(layer << 8) | site` with an atomic minimum, so however many
//! threads raise, the word ends up naming the earliest `(layer, site)` in step
//! order: deterministic, and the same word a gate can pin.
//!
//! The word is sticky. The head's argmax copies it next to the token it
//! writes ([`crate::elem`]'s `argmax_fault`), so the step's one readback
//! carries it with no launch and no copy of its own; nothing on the card
//! clears it. A prompt of several tokens is several bodies before one
//! readback, and a fault in the first must still be in the word at the last.
//! The host clears it ([`crate::Gpu::clear_fault`]) when the state it
//! condemned is gone: a model's `reset`, or a gate that has asserted it.
//!
//! A kernel gets the word as a [`FaultSink`] launch argument: its device
//! address and the layer the launch belongs to, one by-value parameter. The
//! site is the kernel's own constant. Only [`crate::Gpu::fault_sink`] makes a
//! sink, from the word it owns, so a sink always addresses a live word of
//! that `Gpu`'s context.

use cuda_device::ptx_asm;

/// The fault word's value while no fault has been raised.
pub const FAULT_NONE: u32 = u32::MAX;

/// The layer a launch outside every layer raises with: the output head.
pub const LAYER_HEAD: u32 = 0xFFFE;

/// The layer a launch raises with when its caller names none — a gate's
/// direct launch, or an architecture whose chain does not label its
/// quantizers.
pub const LAYER_NONE: u32 = 0xFFFF;

/// Where a fault was raised: the kernel family that refused its input. The
/// codes are fixed — a gate pins the word they make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FaultSite {
    /// The q8_1 activation quantizer (`q8_1_quant_block`: `q3k_quantize_q8_1`
    /// and every entry that runs its body) met a non-finite value.
    QuantColumn = 1,
    /// `norm_quant` normalized a column to a non-finite value, or to a zero
    /// scale.
    NormQuant = 2,
    /// The 32-value q8 quantizer of the Q5 path met a non-finite value.
    Q5Quant = 3,
    /// `ds41_hc_pre`'s in-register q8_1 activation met a non-finite value.
    HcQuant = 4,
    /// A router found fewer finite candidates than slots, or produced a
    /// non-finite weight.
    Router = 5,
    /// A selected-attention list named a row past the compressed stream.
    AttnSel = 6,
    /// A grouped GEMM's route table met an expert id past its stack.
    ExpertId = 7,
    /// A token id that selects a table row — an embedding row, the DSpark
    /// Markov head's previous token — lies past the table's rows.
    TokenId = 9,
    /// A prefill flash row's live key count was zero or past the cache.
    KeyCount = 10,
}

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
        FaultSite::TokenId,
        FaultSite::KeyCount,
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
            FaultSite::TokenId => "token_id",
            FaultSite::KeyCount => "key_count",
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
                "fewer finite router candidates than slots, or a non-finite weight"
            }
            FaultSite::AttnSel => "a selected row past the compressed stream",
            FaultSite::ExpertId => "a routed expert id past the stack's expert count",
            FaultSite::TokenId => "a token id past the table's rows",
            FaultSite::KeyCount => "a flash row's live key count of zero or past the cache",
        }
    }
}

/// A raised fault, decoded from the word: the layer the launch belonged to
/// and the site code. A code that is no [`FaultSite`] is kept as it came, so
/// a corrupt word still reads as a fault rather than as clean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    /// The raising launch's layer: a model layer, [`LAYER_HEAD`] or
    /// [`LAYER_NONE`].
    pub layer: u32,
    /// The raw site code (`FaultSite as u32` for every known site).
    pub code: u32,
}

impl Fault {
    /// The fault a word holds, `None` for [`FAULT_NONE`].
    #[must_use]
    pub fn from_word(word: u32) -> Option<Fault> {
        (word != FAULT_NONE).then_some(Fault {
            layer: word >> 8,
            code: word & 0xFF,
        })
    }

    /// The word this fault is stored as.
    #[must_use]
    pub fn word(self) -> u32 {
        (self.layer << 8) | self.code
    }

    /// The site, when the code is a known one.
    #[must_use]
    pub fn site(self) -> Option<FaultSite> {
        FaultSite::ALL
            .iter()
            .copied()
            .find(|s| *s as u32 == self.code)
    }
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.layer {
            LAYER_HEAD => write!(f, "the output head")?,
            LAYER_NONE => write!(f, "an unlabelled launch")?,
            l => write!(f, "layer {l}")?,
        }
        match self.site() {
            Some(s) => write!(f, " site {} ({})", s.name(), s.cause()),
            None => write!(f, " site code {} (unknown)", self.code),
        }
    }
}

/// A kernel's handle on the fault word: the word's device address and the
/// layer of the launch. `Copy`, one by-value launch parameter; made only by
/// [`crate::Gpu::fault_sink`] from the word that `Gpu` owns, which outlives
/// every launch on its stream and every graph captured there (a graph drops
/// before the buffers it addresses).
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
    /// `word` addresses one live device `u32` of the context the sink's
    /// launches run in, and stays allocated while any launch or captured
    /// graph that carries the sink can run.
    pub(crate) unsafe fn new(word: *mut u32, layer: u32) -> FaultSink {
        FaultSink { word, layer }
    }

    /// Raise a fault at `site` for this sink's layer: the word becomes the
    /// minimum of itself and `(layer << 8) | site`. Device code; any number
    /// of threads may raise.
    ///
    /// A relaxed device-scope `red.global.min`, spelled in PTX: the word's
    /// pointer arrives inside a by-value parameter, so the compiler sees a
    /// generic address, and its atomics on a generic address branch on
    /// `isspacep.local` into a thread-private load/store twin — a local path
    /// in every raising kernel that the word, a global allocation, never takes.
    #[inline(always)]
    pub fn raise(self, site: FaultSite) {
        let code = (self.layer << 8) | site as u32;
        // SAFETY: `word` addresses a live u32 of global memory by the
        // constructor's contract (a `DeviceBuffer` of the context), so the
        // generic-to-global conversion is exact; every access to it on the
        // card is this reduction or the argmax's volatile copy.
        unsafe {
            ptx_asm!(
                "{ .reg .b64 g;\n\t\
                   cvta.to.global.u64 g, %0;\n\t\
                   red.relaxed.gpu.global.min.u32 [g], %1; }",
                in("l") self.word,
                in("r") code,
                clobber("memory")
            );
        }
    }

    /// The word as it stands. Device code: the argmax's copy into its
    /// readback buffer, after every kernel of the step before it.
    #[inline(always)]
    pub fn read(self) -> u32 {
        // SAFETY: `word` addresses a live device u32 by the constructor's
        // contract. Volatile: the value must come from memory on every
        // launch and replay.
        unsafe { core::ptr::read_volatile(self.word) }
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
