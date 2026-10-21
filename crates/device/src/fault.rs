//! The fault word: how a kernel that cannot panic refuses an input.
//!
//! A kernel that meets input it has no defined answer for — a non-finite
//! activation, an index past the rows it gathers from — keeps its memory
//! accesses defined and raises a fault instead of writing a plausible value
//! and moving on.
//!
//! The word lives in one allocation: word 0 is the *first-layer word*,
//! [`FAULT_NONE`] while clean, and word `1 + layer` is layer `layer`'s *site
//! mask*, 0 while clean. A raise stores `(layer << 8) | site` into word 0
//! with an atomic minimum and ORs `1 << site` into its own layer's mask.
//! However many threads raise, word 0 ends up naming the earliest layer and
//! the smallest site code raised there, and that layer's mask every site
//! raised in it: deterministic, and a pair a gate can pin. The mask is per
//! layer because a non-finite value raised in one layer is met again by the
//! quantizers of every later one; only the first layer's sites say where it
//! started.
//!
//! The pair is sticky until the host clears it: a model that has been
//! condemned stays condemned until its state is gone.

use crate::DeviceError;

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

/// The mask bit of one site.
#[inline(always)]
#[must_use]
pub const fn site_mask(site: FaultSite) -> u32 {
    1u32 << (site as u32)
}

/// Where a fault was raised: the kernel family that refused its input. The
/// codes are fixed — a gate pins the word they make.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum FaultSite {
    /// The q8_1 activation quantizer met a non-finite value.
    QuantColumn = 1,
    /// A normalizing kernel met a non-finite value, or a zero scale.
    NormQuant = 2,
    /// The 32-value q8 quantizer of the Q5 path met a non-finite value.
    Q5Quant = 3,
    /// A hyper-connection's in-register q8_1 activation met a non-finite
    /// value.
    HcQuant = 4,
    /// A router met a non-finite logit, found fewer finite candidates than
    /// slots, or produced a non-finite weight.
    Router = 5,
    /// A selected-attention list named a row past the compressed stream.
    AttnSel = 6,
    /// A grouped GEMM's route table met an expert id past its stack.
    ExpertId = 7,
    /// An attention launch was handed a visible count past the rows of its
    /// key source.
    AttnCount = 8,
    /// A token id that selects a table row lies past the table's rows.
    TokenId = 9,
    /// A prefill flash row's live key count was zero or past the cache.
    KeyCount = 10,
    /// A cache append was handed a position at or past the cache's rows.
    CachePos = 11,
    /// A linear-attention conv and prep met a non-finite input or a token
    /// position other than the call's own.
    LinearConv = 12,
    /// A linear-attention delta step produced a non-finite output.
    LinearDelta = 13,
    /// A linear-attention gated norm met a non-finite input or a mean square
    /// that is not finite.
    LinearGate = 14,
    /// A linear-attention delta step read a lane word at or past the state's
    /// lanes.
    DeltaLane = 15,
    /// A cache append stored a value that is not finite.
    CacheValue = 16,
    /// A token-pool selector met a key or a score that is not finite, or a
    /// selected token past the cache.
    PoolSelect = 18,
    /// A PLE site's gate or conv met a value that is not finite.
    Ple = 19,
    /// A gated-residual hyper-connection met a value that is not finite.
    HcMix = 20,
    /// An f32 product kernel met a value that is not finite.
    F32Product = 21,
    /// A head whose logits no quantizer checked met a logit that is not
    /// finite.
    Logit = 22,
    /// A linear-attention delta step read a lane whose stamp is not the
    /// position of the call's first row.
    DeltaStamp = 23,
    /// A two-level candidate selection met a count past its rows or a list
    /// that is not in order.
    CandMask = 26,
    /// A q8_0 cache append met a value that is not finite.
    KvQuant = 27,
    /// A Mamba-2 decode step produced a value that is not finite.
    Mamba2Scan = 28,
}

// A site is one bit of a u32 mask, and no site's code is 0, so a raise never
// stores word 0 with an empty mask: that pair is an unwritten seed, which reads
// as a state error instead of decoding as a fault.
const _: () = {
    let mut i = 0;
    while i < FaultSite::ALL.len() {
        let code = FaultSite::ALL[i] as u32;
        assert!(code >= 1, "a FaultSite code is 0");
        assert!(code < 32, "a FaultSite code is not one bit of a u32 mask");
        i += 1;
    }
};

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
        FaultSite::LinearConv,
        FaultSite::LinearDelta,
        FaultSite::LinearGate,
        FaultSite::DeltaLane,
        FaultSite::CacheValue,
        FaultSite::PoolSelect,
        FaultSite::Ple,
        FaultSite::HcMix,
        FaultSite::F32Product,
        FaultSite::Logit,
        FaultSite::DeltaStamp,
        FaultSite::CandMask,
        FaultSite::KvQuant,
        FaultSite::Mamba2Scan,
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
            FaultSite::LinearConv => "linear_conv",
            FaultSite::LinearDelta => "linear_delta",
            FaultSite::LinearGate => "linear_gate",
            FaultSite::DeltaLane => "delta_lane",
            FaultSite::CacheValue => "cache_value",
            FaultSite::PoolSelect => "pool_select",
            FaultSite::Ple => "ple",
            FaultSite::HcMix => "hc_mix",
            FaultSite::F32Product => "f32_product",
            FaultSite::Logit => "logit",
            FaultSite::DeltaStamp => "delta_stamp",
            FaultSite::CandMask => "cand_mask",
            FaultSite::KvQuant => "kv_quant",
            FaultSite::Mamba2Scan => "mamba2_scan",
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
            FaultSite::LinearConv => {
                "a non-finite conv input, a conv sum, query/key norm, beta or decay not finite, \
                 or a token position other than pos[0] + t"
            }
            FaultSite::LinearDelta => {
                "a non-finite delta-step output: a non-finite input or a state no longer finite"
            }
            FaultSite::LinearGate => "a non-finite gated-norm input or mean square",
            FaultSite::DeltaLane => "a delta-step lane word at or past the state's lanes",
            FaultSite::CacheValue => {
                "a cache row value not finite: a non-finite input, a non-finite mean square or \
                 variance, or a value past f16's range"
            }
            FaultSite::PoolSelect => {
                "a raw or pooled indexer key or a selector score not finite, or a selected token \
                 past the cache"
            }
            FaultSite::Ple => {
                "a PLE key, stream, gated value or conv input not finite (a value, its sum of \
                 squares or the gate's dot), or a token position other than pos[0] + t"
            }
            FaultSite::HcMix => {
                "a hyper-connection stream sum of squares, partial dot, bottleneck value, combine \
                 weight or mixed value not finite"
            }
            FaultSite::F32Product => {
                "an attention output, its gate, a shared expert output, its weight, the host sum, \
                 an RMS sum of squares, audio channel variance, attention score or GLU input not finite, \
                 or a product or sum that is not"
            }
            FaultSite::Logit => "a head logit not finite",
            FaultSite::DeltaStamp => {
                "a delta-step lane whose stamp is not the position of the call's first row: a lane \
                 never written, or one a commit did not leave there"
            }
            FaultSite::CandMask => {
                "a candidate count past its score rows, a source or moved score not finite, a kept \
                 list not strictly ascending or not ending at its pin, or a list entry past the \
                 candidate rows"
            }
            FaultSite::KvQuant => "a q8_0 cache row value not finite",
            FaultSite::Mamba2Scan => {
                "a Mamba-2 step output not finite: a non-finite input, a scan state no longer \
                 finite, or a grouped norm whose mean square is not finite"
            }
        }
    }
}

/// A raised fault, decoded from the pair the device holds: the first layer a
/// launch raised in, the smallest site code raised there, and the mask of
/// every site raised there (`1 << code` per site). A code that is no
/// [`FaultSite`] is kept as it came, so a corrupt word still reads as a
/// fault rather than as clean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    /// The raising launch's layer: a model layer, [`LAYER_HEAD`] or
    /// [`LAYER_NONE`].
    pub layer: u32,
    /// The raw site code (`FaultSite as u32` for every known site): the
    /// smallest raised in `layer`.
    pub code: u32,
    /// Every site raised in `layer`, bit `code` per site.
    pub sites: u32,
}

impl core::fmt::Display for Fault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.site() {
            Some(site) => write!(
                f,
                "layer {} raised {} ({})",
                self.layer,
                site.name(),
                site.cause()
            ),
            None => write!(f, "layer {} raised site code {}", self.layer, self.code),
        }
    }
}

impl Fault {
    /// The fault a first-layer word and that layer's mask hold, `None` for
    /// [`FAULT_NONE`].
    #[must_use]
    pub fn from_words(word: u32, sites: u32) -> Option<Self> {
        (word != FAULT_NONE).then_some(Self {
            layer: word >> 8,
            code: word & 0xFF,
            sites,
        })
    }

    /// The fault the whole word holds: word 0 names the layer, and that
    /// layer's mask word carries the sites; `None` while clean. A layer past
    /// [`LAYER_NONE`] has no mask word and reads as no sites. A `words` shorter
    /// than [`FAULT_WORDS`] is refused by name.
    pub fn decode(words: &[u32]) -> Result<Option<Self>, DeviceError> {
        check_words(words.len())?;
        let first = words[0];
        let sites = if first == FAULT_NONE || first >> 8 > LAYER_NONE {
            0
        } else {
            words[mask_index(first >> 8)]
        };
        Ok(Self::from_words(first, sites))
    }

    /// The fault of `site` raised alone in `layer`: what a gate that plants
    /// one fault expects.
    #[must_use]
    pub fn at(layer: u32, site: FaultSite) -> Self {
        Self::of_sites(layer, &[site])
    }

    /// The fault of every site in `sites` raised in `layer` and nowhere
    /// earlier.
    #[must_use]
    pub fn of_sites(layer: u32, sites: &[FaultSite]) -> Self {
        Self {
            layer,
            code: sites.iter().map(|s| *s as u32).min().unwrap_or(0),
            sites: sites.iter().fold(0, |m, s| m | site_mask(*s)),
        }
    }

    /// The first-layer word this fault is stored as.
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

/// The refusal of a fault allocation shorter than the word's own length.
fn check_words(len: usize) -> Result<(), DeviceError> {
    if len < FAULT_WORDS {
        return Err(DeviceError::Range {
            what: "fault word",
            detail: format!("{len} words are fewer than the word's {FAULT_WORDS}"),
        });
    }
    Ok(())
}

/// The word's rule, on the host: a raise stores `(layer << 8) | site` into
/// word 0 with a minimum and ORs `1 << site` into its own layer's mask. A
/// layer past [`LAYER_NONE`] has no mask word: word 0 names the fault and no
/// mask is written, as the device's kernel does. A `words` shorter than
/// [`FAULT_WORDS`] is refused by name.
///
/// This is the model the gates check the device against: the device does the
/// same two operations atomically, in the kernel.
pub fn raise(words: &mut [u32], layer: u32, site: FaultSite) -> Result<(), DeviceError> {
    check_words(words.len())?;
    let word = (layer << 8) | (site as u32);
    words[0] = words[0].min(word);
    if layer <= LAYER_NONE {
        words[mask_index(layer)] |= site_mask(site);
    }
    Ok(())
}

/// The fault word of a device, read and cleared by the host against a
/// stream `S`: the stream the kernels that raise it were enqueued on.
pub trait FaultWord<S>: Send + Sync {
    /// The fault the word holds once `s`'s work so far is done, `None` while
    /// clean.
    fn read(&self, s: &S) -> Result<Option<Fault>, DeviceError>;

    /// The word is clean again, ordered on `s`: the raises of `s`'s work so
    /// far are done before the clear, and the work enqueued after it starts
    /// from a clean word. Work on another stream is the caller's to order.
    fn clear(&self, s: &S) -> Result<(), DeviceError>;
}
