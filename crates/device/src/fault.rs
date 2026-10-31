//! The fault word: how a kernel that cannot panic refuses an input. The one
//! owner of its layout, its site codes and its decoding, for every backend.
//!
//! A kernel that meets input it has no defined answer for — a non-finite
//! activation in a quantizer, a router that cannot fill its slots, an index
//! past the stream it gathers from — keeps its memory accesses defined and
//! raises a fault instead of writing a plausible value and moving on.
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
//! started. Site codes do not follow step order, so the host prints a mask in
//! the architecture's step order ([`StepOrder`], [`Fault::in_order`]); the
//! tables are the engine's, one per architecture.
//!
//! The pair is sticky until the host clears it: a model that has been
//! condemned stays condemned until its state is gone. A backend's kernels
//! raise it with the backend's own atomics; the host reads it through
//! [`Fault::read_from`], the one decoding rule.

use crate::DeviceError;

/// How far a layer sits above the site code in the first-layer word: the word
/// a raise stores is `(layer << LAYER_SHIFT) | code`.
pub const LAYER_SHIFT: u32 = 8;

/// The bits of the first-layer word that hold the site code.
pub const CODE_MASK: u32 = (1 << LAYER_SHIFT) - 1;

/// The largest layer a first-layer word holds: the bits above the code.
pub const LAYER_MAX: u32 = u32::MAX >> LAYER_SHIFT;

/// The fault word's value while no fault has been raised: the one word with
/// every bit set, which no raise stores (no site's code is [`CODE_MASK`]).
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

/// The bit of `site` in a layer's mask.
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
    /// Markov head's previous token, a reduced head's listed row — lies past
    /// the table's rows.
    TokenId = 9,
    /// A prefill flash row's live key count was zero or past the cache.
    KeyCount = 10,
    /// A cache append was handed a position at or past the cache's rows.
    CachePos = 11,
    /// A linear-attention conv and prep (`linear::conv`) met a non-finite
    /// input, a conv sum, a query/key norm, β or decay not finite, or a
    /// token position other than `pos[0] + t`.
    LinearConv = 12,
    /// A linear-attention delta step (`linear::delta`) produced a
    /// non-finite output: a non-finite input, or a state that stopped being
    /// finite.
    LinearDelta = 13,
    /// A linear-attention gated norm (`linear::norm_gate`) met a non-finite
    /// input or a mean square that is not finite.
    LinearGate = 14,
    /// A linear-attention delta step (`linear::delta`) read a lane word at
    /// or past the state's lanes.
    DeltaLane = 15,
    /// A cache append (`latent`, `flash::kv_append_pos_buf`, the fused
    /// `kv_norm_rope_append`, and the f16
    /// head appends of `rope_neox`, `rope_norm`, `qk_rms` and
    /// `attention_h128_ring`) met a value that is not finite: a non-finite
    /// input, a norm whose mean square or variance is not finite, or a value
    /// past f16's range.
    CacheValue = 16,
    /// A token-pool selector (`qsa`; GLM's `latent::index_pool` and `kpool`)
    /// met an indexer key that is not finite after its f16 rounding
    /// (`qsa::qsa_key_append`, the raw key), a pooled key or a head's score
    /// that is not finite, a launch whose rows' positions or live counts are
    /// not consecutive (`qsa_key_append`, the pool pass), its top-k a refused
    /// live count, or its selected
    /// flash a list entry at or past the cache. A block selector (MiniMax's
    /// `msa`) raises it for a head's score that is not finite, a selection
    /// that lost the query's own block, and its selected flash for a block
    /// at or past the cache.
    PoolSelect = 18,
    /// A PLE site's gate or conv (`ple`) met a key, stream, gated value or
    /// conv input whose sum of squares, dot or value is not finite, or a
    /// token position other than `pos[0] + t`.
    Ple = 19,
    /// A gated-residual hyper-connection (`hc_gated`) met a stream whose sum
    /// of squares is not finite, or produced a partial dot, a bottleneck
    /// value, a combine weight or a mixed value that is not finite.
    HcMix = 20,
    /// The shared RMSNorm sum of squares (`rms_norm`, and every norm that takes
    /// its scale from `elem::rms_scale_checked` or
    /// `elem::rms_scale_of_mean_checked`: the latent norm of the fused
    /// `kv_norm_rope_append`, `rope_neox`'s head norms, V4.1's hyper-connection,
    /// key and compressor norms) not finite, or a scale that is not finite and
    /// positive; a value the fused `kv_norm_rope_append` wrote to `kv_s` that is
    /// not finite; an f32 product kernel of Qwen3.8's layer (`q38`: the
    /// attention output under its sigmoid gate, the shared expert's gated sum)
    /// or Gemma4a's audio channel normalization, attention score or GLU met
    /// non-finite arithmetic; a gate-up's gate or up dot sum was not finite
    /// before its activation (including an NVFP4 expert's tensor scale).
    F32Product = 21,
    /// A head whose logits no quantizer checked before them (a Q8_0 lm_head
    /// over f32 rows) met a logit that is not finite in its argmax.
    Logit = 22,
    /// A linear-attention delta step over lanes (`linear::delta`'s
    /// `gdn_delta_lanes`) read a lane whose stamp is not the position its
    /// first row stands at: a lane never written, or one a commit did not
    /// leave there.
    DeltaStamp = 23,
    /// A two-level candidate selection (`cand`) met a visible count past its
    /// score rows, a source or moved score that is not finite, a kept list
    /// that is not strictly ascending or does not end at its pin, or a list
    /// entry at or past the candidate rows.
    CandMask = 26,
    /// A q8_0 cache append (`rope_neox`'s quantizing entries) met a
    /// non-finite key or value: the block is stored with a NaN scale, so
    /// every value it dequantizes to is NaN.
    KvQuant = 27,
    /// A Mamba-2 decode step (`mamba2`) produced a `y` that is not finite: a
    /// non-finite input, a state that stopped being finite, or a grouped
    /// norm whose mean square is not finite.
    Mamba2Scan = 28,
}

// A site is one bit of a u32 mask, and no site's code is 0, so a raise never
// stores word 0 with an empty mask: that pair is a head readback's unwritten
// seed, which reads as a state error instead of decoding as a fault. Every
// site's code also reads back as that site, and fits the word's code bits.
const _: () = {
    assert!(
        u32::BITS <= CODE_MASK,
        "the code bits do not hold every bit index of a u32 mask"
    );
    assert!(
        LAYER_NONE <= LAYER_MAX,
        "the word's layer bits do not hold LAYER_NONE"
    );
    assert!(
        FAULT_NONE == (LAYER_MAX << LAYER_SHIFT) | CODE_MASK,
        "FAULT_NONE is a word a raise can store"
    );
    let mut i = 0;
    while i < FaultSite::ALL.len() {
        let code = FaultSite::ALL[i] as u32;
        assert!(code >= 1, "a FaultSite code is 0");
        assert!(code < 32, "a FaultSite code is not one bit of a u32 mask");
        match FaultSite::from_code(code) {
            Some(site) => assert!(
                site as u32 == code,
                "a FaultSite code reads back as another"
            ),
            None => panic!("a FaultSite code does not read back"),
        }
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

    /// The site of `code`, `None` for a code no site has. Device code too: a
    /// kernel handed a site as a number raises through it.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<FaultSite> {
        Some(match code {
            1 => FaultSite::QuantColumn,
            2 => FaultSite::NormQuant,
            3 => FaultSite::Q5Quant,
            4 => FaultSite::HcQuant,
            5 => FaultSite::Router,
            6 => FaultSite::AttnSel,
            7 => FaultSite::ExpertId,
            8 => FaultSite::AttnCount,
            9 => FaultSite::TokenId,
            10 => FaultSite::KeyCount,
            11 => FaultSite::CachePos,
            12 => FaultSite::LinearConv,
            13 => FaultSite::LinearDelta,
            14 => FaultSite::LinearGate,
            15 => FaultSite::DeltaLane,
            16 => FaultSite::CacheValue,
            18 => FaultSite::PoolSelect,
            19 => FaultSite::Ple,
            20 => FaultSite::HcMix,
            21 => FaultSite::F32Product,
            22 => FaultSite::Logit,
            23 => FaultSite::DeltaStamp,
            26 => FaultSite::CandMask,
            27 => FaultSite::KvQuant,
            28 => FaultSite::Mamba2Scan,
            _ => return None,
        })
    }
}

/// An architecture's step order: the order its step meets the fault sites in
/// (a table the engine owns, one per architecture), which a fault's mask
/// prints in, and the architecture's name as the print names it. A site a
/// step meets twice stands where it is first met, and each table names every
/// site exactly once, so no bit of a mask goes unprinted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepOrder {
    /// The architecture's name.
    pub arch: &'static str,
    /// Every site, in the order the step meets them.
    pub sites: &'static [FaultSite],
}

/// A raised fault, decoded from the pair the device holds: the first layer a
/// launch raised in, the smallest site code raised there, and the mask of
/// every site raised there (`1 << code` per site). A code that is no
/// [`FaultSite`] is kept as it came, so a corrupt word still reads as a
/// fault rather than as clean. Equality is on those three; the step order a
/// fault prints its mask in ([`Fault::in_order`]) is presentation.
#[derive(Clone, Copy)]
pub struct Fault {
    /// The raising launch's layer: a model layer, [`LAYER_HEAD`] or
    /// [`LAYER_NONE`].
    pub layer: u32,
    /// The raw site code (`FaultSite as u32` for every known site): the
    /// smallest raised in `layer`.
    pub code: u32,
    /// Every site raised in `layer`, bit `code` per site.
    pub sites: u32,
    /// The step order the mask prints in, once a model has named it.
    order: Option<StepOrder>,
}

// By hand: the step order is presentation, not part of the fault.
impl PartialEq for Fault {
    fn eq(&self, other: &Fault) -> bool {
        (self.layer, self.code, self.sites) == (other.layer, other.code, other.sites)
    }
}

impl Eq for Fault {}

/// An architecture's name, printed bare (as its enum's `Debug` prints it).
struct Bare(&'static str);

impl core::fmt::Debug for Bare {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}

// By hand: the step order prints as the architecture it is named for, the
// one field of it the engine's messages show.
impl core::fmt::Debug for Fault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fault")
            .field("layer", &self.layer)
            .field("code", &self.code)
            .field("sites", &self.sites)
            .field("arch", &self.order.map(|o| Bare(o.arch)))
            .finish()
    }
}

impl Fault {
    /// The fault a first-layer word and that layer's mask hold, `None` for
    /// [`FAULT_NONE`].
    #[must_use]
    pub fn from_words(word: u32, sites: u32) -> Option<Fault> {
        (word != FAULT_NONE).then_some(Fault {
            layer: word >> LAYER_SHIFT,
            code: word & CODE_MASK,
            sites,
            order: None,
        })
    }

    /// The fault the word holds, read through `word(i)` (the word at index
    /// `i` of the allocation): word 0, then, only when it names a fault of a
    /// layer that has a mask, that layer's mask. A word naming a layer past
    /// [`LAYER_NONE`] has no mask and reads as a fault with no sites. The one
    /// decoding rule: every backend's reader and the host model go through it.
    pub fn read_from<E>(mut word: impl FnMut(usize) -> Result<u32, E>) -> Result<Option<Fault>, E> {
        let first = word(0)?;
        if first == FAULT_NONE {
            return Ok(None);
        }
        let sites = if first >> LAYER_SHIFT <= LAYER_NONE {
            word(mask_index(first >> LAYER_SHIFT))?
        } else {
            0
        };
        Ok(Fault::from_words(first, sites))
    }

    /// The fault a host copy of the whole word holds. A `words` shorter than
    /// [`FAULT_WORDS`] is refused by name.
    pub fn decode(words: &[u32]) -> Result<Option<Fault>, DeviceError> {
        check_words(words.len())?;
        Fault::read_from(|i| Ok(words[i]))
    }

    /// The fault of `site` raised alone in `layer`: what a gate that plants
    /// one fault expects.
    #[must_use]
    pub fn at(layer: u32, site: FaultSite) -> Fault {
        Fault::of_sites(layer, &[site])
    }

    /// The fault of every site in `sites` raised in `layer` and nowhere
    /// earlier. An empty list is no fault a device can hold; it reads as code
    /// 0 with an empty mask.
    #[must_use]
    pub fn of_sites(layer: u32, sites: &[FaultSite]) -> Fault {
        Fault {
            layer,
            code: sites.iter().map(|s| *s as u32).min().unwrap_or(0),
            sites: sites.iter().fold(0, |m, s| m | site_mask(*s)),
            order: None,
        }
    }

    /// The first-layer word this fault is stored as.
    ///
    /// # Panics
    /// When the layer is past [`LAYER_MAX`] or the code past [`CODE_MASK`]:
    /// no word holds such a fault, and a wrapped word would name another.
    #[must_use]
    pub fn word(self) -> u32 {
        first_word(self.layer, self.code).unwrap_or_else(|| {
            panic!(
                "layer {} and code {} do not fit the first-layer word",
                self.layer, self.code
            )
        })
    }

    /// This fault, printing its mask in `order`.
    #[must_use]
    pub fn in_order(self, order: StepOrder) -> Fault {
        Fault {
            order: Some(order),
            ..self
        }
    }

    /// The site, when the code is a known one.
    #[must_use]
    pub fn site(self) -> Option<FaultSite> {
        FaultSite::from_code(self.code)
    }

    /// The mask's sites in `order`, then any bit `order` does not name, as
    /// its raw code: nothing the device raised is left out.
    #[must_use]
    pub fn sites_in(self, order: &[FaultSite]) -> Vec<Result<FaultSite, u32>> {
        let mut out: Vec<Result<FaultSite, u32>> = order
            .iter()
            .filter(|s| self.sites & site_mask(**s) != 0)
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

/// Where the fault was raised, its site's name and the mask in step order.
/// The alternate form (`{:#}`) adds the site's [`FaultSite::cause`] after its
/// name: an error's text carries it once, while a gate line that prints a
/// fault beside the one it wants reads the two without it.
impl core::fmt::Display for Fault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let where_ = match self.layer {
            LAYER_HEAD => "the output head".to_string(),
            LAYER_NONE => "an unlabelled launch".to_string(),
            l => format!("layer {l}"),
        };
        write!(f, "{where_}")?;
        match self.site() {
            Some(s) if f.alternate() => write!(f, " site {} ({})", s.name(), s.cause())?,
            Some(s) => write!(f, " site {}", s.name())?,
            None => write!(f, " site code {} (unknown)", self.code)?,
        }
        let (order, named) = match self.order {
            Some(o) => (o.sites, format!("{} step order", o.arch)),
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

/// The fault of several devices' words as read, for a step that ran on more
/// than one: the first layer wins — the smallest first-layer word — and a
/// layer two devices both raised in keeps the smaller code and both masks.
/// `None` when every device is clean.
#[must_use]
pub fn read_cards(cards: &[Option<Fault>]) -> Option<Fault> {
    cards
        .iter()
        .flatten()
        .copied()
        .fold(None, |acc, f| match acc {
            None => Some(f),
            Some(a) if a.layer == f.layer => Some(Fault {
                code: a.code.min(f.code),
                sites: a.sites | f.sites,
                ..a
            }),
            Some(a) if f.word() < a.word() => Some(f),
            Some(a) => Some(a),
        })
}

/// The first-layer word of `layer` and a site `code`, `None` when either does
/// not fit its bits.
fn first_word(layer: u32, code: u32) -> Option<u32> {
    (layer <= LAYER_MAX && code <= CODE_MASK).then_some((layer << LAYER_SHIFT) | code)
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

/// The word's rule, on the host: a raise stores `(layer << LAYER_SHIFT) | site`
/// into word 0 with a minimum and ORs `1 << site` into its own layer's mask. A
/// layer past [`LAYER_NONE`] has no mask word: word 0 names the fault and no
/// mask is written. A `words` shorter than [`FAULT_WORDS`], or a layer past
/// [`LAYER_MAX`] (which no word holds), is refused by name.
///
/// This is the model the gates check a device against: the device does the
/// same two operations atomically, in the kernel.
pub fn raise(words: &mut [u32], layer: u32, site: FaultSite) -> Result<(), DeviceError> {
    check_words(words.len())?;
    let word = first_word(layer, site as u32).ok_or_else(|| DeviceError::Range {
        what: "fault word",
        detail: format!("layer {layer} is past the {LAYER_MAX} the first-layer word holds"),
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Of two devices' faults the first layer wins; one layer raised on both
    /// keeps the smaller code and both masks; clean devices read clean.
    #[test]
    fn read_cards_takes_the_first_layer() {
        let a = Fault::at(9, FaultSite::QuantColumn);
        let b = Fault::at(4, FaultSite::ExpertId);
        assert_eq!(read_cards(&[Some(a), Some(b)]), Some(b));
        assert_eq!(read_cards(&[Some(b), None, Some(a)]), Some(b));
        assert_eq!(read_cards(&[None, None]), None);
        let c = Fault::at(4, FaultSite::QuantColumn);
        assert_eq!(
            read_cards(&[Some(b), Some(c)]),
            Some(Fault::of_sites(
                4,
                &[FaultSite::ExpertId, FaultSite::QuantColumn]
            ))
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
        let sample = f.in_order(StepOrder {
            arch: "Sample",
            sites: &[FaultSite::KeyCount, FaultSite::QuantColumn],
        });
        assert_eq!(sample, f, "the step order is not part of equality");
        let shown = sample.to_string();
        assert!(
            shown.contains("Sample step order") && shown.contains("key_count, quant_column"),
            "{shown}"
        );
        assert_eq!(
            format!("{sample:?}"),
            "Fault { layer: 3, code: 1, sites: 1026, arch: Some(Sample) }"
        );
        assert_eq!(
            format!("{f:?}"),
            "Fault { layer: 3, code: 1, sites: 1026, arch: None }"
        );
        // A mask bit past every site prints as its code: the bit after the highest site's.
        let past = FaultSite::ALL.iter().map(|&s| s as u32).max().unwrap_or(0) + 1;
        let odd = Fault::from_words((3 << 8) | 1, (1 << 1) | (1 << past)).unwrap();
        assert!(odd.to_string().contains(&format!("code {past}")), "{odd}");
    }

    /// A layer the first-layer word has no bits for is refused by name on the
    /// host model and leaves the words as they were; the largest layer it holds
    /// round-trips.
    #[test]
    fn a_layer_past_the_words_bits_is_refused_by_name() {
        let mut words = vec![0u32; FAULT_WORDS];
        words[0] = FAULT_NONE;
        match raise(&mut words, LAYER_MAX + 1, FaultSite::Router) {
            Err(DeviceError::Range { what, .. }) => assert_eq!(what, "fault word"),
            other => panic!("a fault word refusal, not {other:?}"),
        }
        assert_eq!(words[0], FAULT_NONE);
        raise(&mut words, LAYER_MAX, FaultSite::Router).expect("the largest layer fits");
        // A layer past LAYER_NONE has no mask: the word alone names the fault.
        let fault = Fault::decode(&words)
            .expect("a whole word")
            .expect("a raised fault");
        assert_eq!(
            (fault.layer, fault.code, fault.sites),
            (LAYER_MAX, FaultSite::Router as u32, 0)
        );
    }

    /// A fault of a layer no word holds panics by name instead of wrapping to
    /// another layer's word.
    #[test]
    #[should_panic(expected = "do not fit the first-layer word")]
    fn the_word_of_a_layer_past_its_bits_panics_by_name() {
        let _ = Fault::at(LAYER_MAX + 1, FaultSite::Router).word();
    }

    /// The (0, 0) pair no raise can store still decodes — every site's code
    /// is 1 or more, and a raise ORs its bit into the mask — into a code no
    /// site names: why a head readback refuses that pair as unwritten before
    /// it decodes one.
    #[test]
    fn the_zero_pair_decodes_to_a_code_no_site_names() {
        let phantom = Fault::from_words(0, 0).unwrap();
        assert_eq!((phantom.layer, phantom.code, phantom.sites), (0, 0, 0));
        assert_eq!(phantom.site(), None);
    }

    #[test]
    fn the_cause_prints_in_the_alternate_form_and_not_in_the_plain_one() {
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
    }

    /// Every code reads back as its site, and no other code reads as one.
    #[test]
    fn a_code_reads_back_as_its_site_and_only_a_sites_code_does() {
        for code in 0..=u8::MAX.into() {
            let want = FaultSite::ALL.iter().copied().find(|s| *s as u32 == code);
            assert_eq!(FaultSite::from_code(code), want, "code {code}");
        }
        assert_eq!(FaultSite::from_code(u32::MAX), None);
    }

    /// The reader asks for word 0 and, for a fault of a layer with a mask,
    /// that layer's mask: nothing else.
    #[test]
    fn the_reader_asks_for_the_first_word_and_its_layers_mask_only() {
        for (first, want) in [
            (FAULT_NONE, vec![0]),
            ((4 << 8) | 5, vec![0, mask_index(4)]),
            ((LAYER_NONE << 8) | 5, vec![0, mask_index(LAYER_NONE)]),
            (((LAYER_NONE + 1) << 8) | 5, vec![0]),
        ] {
            let mut asked = Vec::new();
            let fault = Fault::read_from(|i| {
                asked.push(i);
                Ok::<u32, ()>(if i == 0 { first } else { 1 << 5 })
            })
            .expect("the reader cannot fail");
            assert_eq!(asked, want, "word {first:#x}");
            assert_eq!(
                fault.map(|f| f.word()),
                (first != FAULT_NONE).then_some(first)
            );
        }
    }
}
