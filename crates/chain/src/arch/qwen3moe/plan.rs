//! A layer's plan: what one layer of a qwen3moe-family chain mixes with and
//! how its experts run, resolved once at load from the file — every weight
//! name, the quantization of each mixed site, the kernels' shape — so the
//! enqueue reads names and never decides a layer's kind. One walk
//! ([`super::walk`]) orders every plan's ops: the mixer is a `match` on
//! [`MixerPlan`] there, and how a token's FFN slots are picked a `match` on
//! [`FfnRoute`] inside a backend's FFN ops.
//!
//! Qwen3-30B-A3B's layers are all [`MixerPlan::Gqa`] at head 128 with no
//! shared expert. Qwen3.6-35B-A3B interleaves [`MixerPlan::Delta`] (gated
//! delta rule) with gated GQA at head 256, each followed by 256 routed
//! experts and a sigmoid-gated shared expert ([`SharedPlan`]): folded in as
//! one more slot of joined stacks where its parts are their stacks' types,
//! else its own dense FFN of its types. Qwen3.5-9B has the same mixers and a dense SwiGLU FFN,
//! which runs as a stack of one expert, every token's one slot on it at
//! weight 1 ([`FfnRoute::Dense`]): the routed FFN's launches without the
//! router. A dense Llama's layers are [`MixerPlan::Gqa`] at head 128 turned
//! by NORM pairs with no head norm ([`GqaKind::Head128`]), each followed by
//! the same dense FFN; its rope may divide by the file's frequency factors
//! ([`GqaPlan::rope_freqs`]). LFM3 interleaves [`MixerPlan::ShortConv`] (a gated
//! short conv over its own conv ring) with attention of head 128 that has an
//! output gate of one value a head ([`GqaKind::HeadGate128`]), a window on
//! some layers and the file's rope frequency factors on the others, each
//! followed by the dense FFN; every half's output joins the stream through the
//! layer's sandwich norm ([`ResidualPlan::Sandwich`]). A layer's kind comes
//! from its [`LayerSpec`] — the file's tensors, cross-checked with its
//! interval key by the reader — never from the layer's number.

use crate::ChainError;
use crate::linear::{self, LinearShape, Ring};
use crate::router::RouterDims;
pub use model::arch::coverage::{DownEntry, GateUpEntry};
use model::arch::coverage::{SiteEntry, qwen35_down, qwen35_gate_up, qwen35_shared, turns_as_neox};
use model::arch::models::shape::{AttnShape, GroupRule, MoeShape, select_gqa};
use model::arch::models::{
    Act, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer, Moe, OutGate,
    Residual, RopeMode, ShortConv,
};

pub use crate::site::SiteTy;

/// One layer's plan.
pub struct LayerPlan {
    pub mixer: MixerPlan,
    pub ffn: FfnPlan,
    /// How each half's output joins the residual stream. The walk has no op for the join: the op
    /// that ends each half ([`ChainOps::attn_out`], `delta_out`, `conv_out` for the mixer,
    /// [`ChainOps::experts`] for the FFN) reads this and joins its output itself, and a backend
    /// that cannot run a sandwich refuses it by name inside that op ([`LayerCtx::plain_only`]).
    ///
    /// [`ChainOps::attn_out`]: super::ops::ChainOps::attn_out
    /// [`ChainOps::experts`]: super::ops::ChainOps::experts
    /// [`LayerCtx::plain_only`]: super::ops::LayerCtx::plain_only
    pub residual: ResidualPlan,
}

/// What a layer mixes its tokens with.
pub enum MixerPlan {
    /// Grouped-query attention over the layer's own K/V planes.
    Gqa(GqaPlan),
    /// The gated delta rule over the layer's recurrent state and conv ring.
    Delta(DeltaPlan),
    /// The gated short conv over the layer's conv ring.
    ShortConv(ShortConvPlan),
}

/// How a layer's two halves, its mixer and its FFN, join the residual stream.
pub enum ResidualPlan {
    /// The half's output is added to the stream: `x += out`.
    Plain,
    /// The half's output goes through its own weighted RMS norm first.
    Sandwich(SandwichPlan),
}

/// A layer's sandwich norms: `x += add[i] · gain[i] ⊙ rms_norm(out)`, the mixer at index 0 and the
/// FFN at index 1, the gain the file's own weight (plain `w · x̂`).
pub struct SandwichPlan {
    /// The norm's epsilon, inside the root.
    pub eps: f32,
    /// The scale each half's normed output is added at.
    pub add: [f32; 2],
    /// The norms' weights, mixer then FFN: `post_attention_norm`, `post_ffw_norm`.
    pub norms: [String; 2],
}

/// The attention geometries a kernel set is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GqaKind {
    /// Qwen3: head 128, the whole head turned by NEOX, no output gate.
    Neox128,
    /// Qwen3.6: head 256, the first 64 values turned by NEOX, `attn_q`
    /// writing `[q | gate]` per head, the flash output multiplied by
    /// `σ(gate)` in the output projection's quantizer.
    Gated256,
    /// A head of 128 values with no output gate and no key selection, the
    /// whole head turned by `pairs`: the q/k norm before the rope when
    /// `head_norm`, none otherwise. [`GqaKind::Neox128`] is `{ Neox, true }`.
    /// A layer's rope may divide by the file's frequency factors
    /// ([`GqaKind::rope_factors`]); it attends every position.
    Head128 { pairs: Pairs, head_norm: bool },
    /// LFM3: a head of 128 values, the whole head turned by NEOX, the q/k norm before the
    /// rope, and an output gate of its own: `attn_gate` projects the layer input to one value
    /// a query head, and the flash output of a head is multiplied by the sigmoid of its value
    /// ([`GqaPlan::gate`]). A layer attends the last `window` positions or every one
    /// ([`GqaPlan::window`]), and its rope divides by the file's frequency factors or does not
    /// ([`GqaPlan::rope_freqs`]).
    HeadGate128,
}

impl GqaKind {
    /// Whether the kind norms each query and key head before its rope: the
    /// layer's q and k norm tensors are then part of its plan.
    #[must_use]
    pub fn head_norm(self) -> bool {
        match self {
            GqaKind::Neox128 | GqaKind::Gated256 | GqaKind::HeadGate128 => true,
            GqaKind::Head128 { head_norm, .. } => head_norm,
        }
    }

    /// Whether the kind gates the flash output by a projection of its own (`attn_gate`, one
    /// value a query head) and so reads the description's window and rope factors per layer.
    #[must_use]
    pub fn head_gate(self) -> bool {
        matches!(self, GqaKind::HeadGate128)
    }

    /// Whether a layer of the kind may have a rope that divides by the file's frequency factors
    /// (the class table's row): the description says which layers do
    /// ([`GqaPlan::rope_freqs`]).
    #[must_use]
    pub fn rope_factors(self) -> bool {
        Self::CLASSES
            .iter()
            .any(|c| c.kind == self && c.rope_factors)
    }
}

/// Which values of a head a rope turns together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pairs {
    /// NEOX: value `i` with value `i + dims/2`.
    Neox,
    /// NORM: the neighbours `2i` and `2i + 1`.
    Norm,
}

impl GqaKind {
    /// Every class with the geometry that selects it: a head's width, the
    /// values of it the rope turns and how it pairs them, the head norm, and
    /// where the output gate comes from; and what a layer of the class keeps
    /// of its own: a window, rope factors.
    const CLASSES: [Class; 4] = [
        Class {
            kind: GqaKind::Neox128,
            head: HEAD_128,
            rope: HEAD_128,
            pairs: Pairs::Neox,
            head_norm: true,
            gate: OutGate::None,
            window: false,
            rope_factors: false,
        },
        Class {
            kind: GqaKind::Head128 {
                pairs: Pairs::Norm,
                head_norm: false,
            },
            head: HEAD_128,
            rope: HEAD_128,
            pairs: Pairs::Norm,
            head_norm: false,
            gate: OutGate::None,
            window: false,
            rope_factors: true,
        },
        Class {
            kind: GqaKind::Gated256,
            head: q35::HEAD_DIM,
            rope: q35::ROPE_DIMS,
            pairs: Pairs::Neox,
            head_norm: true,
            gate: OutGate::Packed,
            window: false,
            rope_factors: false,
        },
        Class {
            kind: GqaKind::HeadGate128,
            head: HEAD_128,
            rope: HEAD_128,
            pairs: Pairs::Neox,
            head_norm: true,
            gate: OutGate::ProjHead,
            window: true,
            rope_factors: true,
        },
    ];
}

/// One attention class and the geometry that selects it ([`GqaKind::CLASSES`]).
struct Class {
    kind: GqaKind,
    /// Values of a head.
    head: u32,
    /// Values of a head the rope turns.
    rope: u32,
    /// How the rope pairs them.
    pairs: Pairs,
    /// Whether each query and key head is normed before its rope.
    head_norm: bool,
    /// Where the output gate comes from.
    gate: OutGate,
    /// Whether a layer of the class may attend a window of the positions, which the description
    /// says per layer.
    window: bool,
    /// Whether a layer of the class may divide its rope's pair frequencies by the file's
    /// factors, which the description says per layer.
    rope_factors: bool,
}

/// How the head-256 flash entries cover a key head's group of query heads:
/// the shape table's row (`models::shape::GQA`) the layer selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flash {
    /// One block of a key head's whole group: eight query heads (the `_256`
    /// entries, and the head-128 entries' only form), or four on the
    /// head-gated head-128 window flash (the `_g4` entries).
    Group,
    /// Blocks of four query heads, `group / 4` a key head (the `_256_p4`
    /// entries): the scalar and the tensor-core segment pass.
    Quads,
    /// Blocks of two query heads, `group / 2` a key head (the `_256_p2`
    /// entries): the scalar segment pass only.
    Pairs,
    /// One block of seven query heads a key head (the head-128 `_g7`
    /// entries): the tensor-core passes only.
    Seven,
}

/// A GQA layer's names, types and geometry.
pub struct GqaPlan {
    pub kind: GqaKind,
    pub flash: Flash,
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_q_norm: String,
    pub attn_k_norm: String,
    pub attn_output: String,
    /// The file's types of `attn_q`, `attn_k`, `attn_v` and `attn_output`:
    /// which launch each takes is a backend's, from the type.
    pub q_ty: SiteTy,
    pub k_ty: SiteTy,
    pub v_ty: SiteTy,
    pub o_ty: SiteTy,
    /// The positions the layer attends: the keys `[max(0, p − w + 1), p]` of position `p`, its
    /// own among them; `None`: every position. Only a kind with its own gate
    /// ([`GqaKind::head_gate`]) has one.
    pub window: Option<u32>,
    /// The output gate's own projection, on a kind that has one ([`GqaKind::head_gate`]).
    pub gate: Option<HeadGate>,
    /// The file's rope frequency factors (`rope_freqs.weight`, F32, one a pair of the rope's
    /// values), on a layer whose rope divides by them; a backend reads the values at load
    /// (`RopeSpec::with_factors`). Only a kind with its own gate has them.
    pub rope_freqs: Option<String>,
}

/// An attention layer's output gate of its own: `attn_gate`, which projects the layer input to
/// one value a query head.
pub struct HeadGate {
    pub name: String,
    /// The file's type of the projection: which launch it takes is a backend's, from the type.
    pub ty: SiteTy,
}

/// A gated short-conv layer's names, types and geometry: `in_proj` writes three chunks of
/// `hidden` values a token, `B | C | x`, and the layer's mixing is `C ⊙ conv(B ⊙ x)` over the
/// conv ring, then `out_proj` back to the residual.
pub struct ShortConvPlan {
    /// The mixer's pre-norm.
    pub attn_norm: String,
    /// The input projection, `3 · hidden` rows (`shortconv.in_proj`).
    pub conv_in: String,
    /// The conv's filters, `[hidden][taps]` F32, the oldest tap first (`shortconv.conv`).
    pub conv: String,
    /// The output projection (`shortconv.out_proj`).
    pub conv_out: String,
    /// Values of the residual stream: the conv's channels.
    pub hidden: usize,
    /// The layer's conv ring; the conv's taps are [`Ring::taps`].
    pub ring: Ring,
    /// The file's types of `shortconv.in_proj` and `shortconv.out_proj`.
    pub in_ty: SiteTy,
    pub out_ty: SiteTy,
}

/// A gated-delta-rule layer's names, types and head counts.
pub struct DeltaPlan {
    pub attn_norm: String,
    /// The q·k·v channels' projection (`attn_qkv`).
    pub qkv: String,
    /// The output gate's projection `z` (`attn_gate`).
    pub gate: String,
    /// The β and α projections, one value per value head.
    pub beta: String,
    pub alpha: String,
    /// The conv taps, `[C][CONV_TAPS]` F32.
    pub conv: String,
    pub dt_bias: String,
    pub ssm_a: String,
    /// The gated norm's per-head gain.
    pub ssm_norm: String,
    /// The output projection back to the residual.
    pub ssm_out: String,
    pub shape: LinearShape,
    /// The file's types of `attn_qkv`, `attn_gate`, β, α and `ssm_out`.
    pub qkv_ty: SiteTy,
    pub gate_ty: SiteTy,
    pub beta_ty: SiteTy,
    pub alpha_ty: SiteTy,
    pub out_ty: SiteTy,
}

/// An FFN's names and types: the three stacks every slot reads (a routed
/// layer's experts, or a dense layer's matrices as one expert) and how a
/// token's slots are picked.
pub struct FfnPlan {
    pub ffn_norm: String,
    pub route: FfnRoute,
    pub gate: String,
    pub up: String,
    pub down: String,
    /// The file's types of the three stacks; a placed plan's host layer
    /// holds the type a host layer's stacks are read as (`read::HOSTED`),
    /// which no launch reads.
    pub gate_ty: SiteTy,
    pub up_ty: SiteTy,
    pub down_ty: SiteTy,
}

/// How a token's FFN slots are picked.
pub enum FfnRoute {
    /// By the router.
    Router {
        /// The router weight: the file's, or the joined one with the shared
        /// expert's gate as its last row.
        gate_inp: String,
        /// The shared expert, on a router whose last row is its gate.
        shared: Option<SharedPlan>,
    },
    /// A dense FFN: one slot a token, on expert 0 of the one-expert stacks,
    /// at weight 1 — a fixed route a backend lays once.
    Dense,
}

/// A layer's sigmoid-gated shared expert: its weight is every token's last
/// slot's, the sigmoid of the router's last row (the router's other rows are
/// the routed experts').
pub enum SharedPlan {
    /// Folded into the routed stacks as their last expert (id the routed
    /// count) in that slot: its gate, up and down are each their stack's
    /// type.
    Joined,
    /// Its own dense FFN at its own types ([`SharedSites`]): a part of
    /// another type than its stack's. The routed launches skip the last
    /// slot, and the combine adds the shared expert's output at the slot's
    /// weight.
    Apart(SharedSites),
}

/// A shared expert kept apart from the routed stacks: its three matrices as
/// the file holds them, each a dense site of its own type ([`SiteTy`]).
pub struct SharedSites {
    pub gate: String,
    pub up: String,
    pub down: String,
    pub gate_ty: SiteTy,
    pub up_ty: SiteTy,
    pub down_ty: SiteTy,
}

impl LayerPlan {
    /// The GQA plan, or a named refusal for a delta or short-conv layer: what a path that
    /// runs attention only (qwen3moe's ubatch) takes.
    pub fn gqa(&self, what: &'static str, l: usize) -> Result<&GqaPlan, ChainError> {
        match &self.mixer {
            MixerPlan::Gqa(g) => Ok(g),
            MixerPlan::Delta(_) => Err(ChainError::shape(
                what,
                format!("layer {l} is a delta-rule layer; this path runs attention layers only"),
            )),
            MixerPlan::ShortConv(_) => Err(ChainError::shape(
                what,
                format!("layer {l} is a short-conv layer; this path runs attention layers only"),
            )),
        }
    }

    /// Slots a token takes in the FFN: the routed ones, plus one for a
    /// folded shared expert; a dense FFN's one.
    pub fn slots(&self, n_used: usize) -> usize {
        match &self.ffn.route {
            FfnRoute::Router { shared, .. } => n_used + usize::from(shared.is_some()),
            FfnRoute::Dense => 1,
        }
    }
}

impl FfnPlan {
    /// The shared expert kept apart, on a layer that keeps one
    /// ([`SharedPlan::Apart`]).
    pub fn apart(&self) -> Option<&SharedSites> {
        match &self.route {
            FfnRoute::Router {
                shared: Some(SharedPlan::Apart(sh)),
                ..
            } => Some(sh),
            FfnRoute::Router { .. } | FfnRoute::Dense => None,
        }
    }
}

/// The gate·up entry the family's FFN dispatch launches gate and up stacks
/// of `gate` and `up` with in one launch ([`qwen35_gate_up`]), or a named
/// refusal: one launch reads one type, and a plan holds only the types the
/// table admits.
pub fn gate_up_entry(
    gate: SiteTy,
    up: SiteTy,
    what: &'static str,
) -> Result<GateUpEntry, ChainError> {
    if gate != up {
        return Err(ChainError::shape(
            what,
            format!("a {gate} gate beside a {up} up; one gate·up launch reads one type"),
        ));
    }
    qwen35_gate_up(gate.ggml()).ok_or_else(|| {
        ChainError::shape(
            what,
            format!("a {gate} gate·up, which the card kernel table names no entry for here"),
        )
    })
}

/// The routed down entry the family's FFN dispatch launches a stack of `ty`
/// with ([`qwen35_down`]), or a named refusal.
pub fn down_entry(ty: SiteTy, what: &'static str) -> Result<DownEntry, ChainError> {
    qwen35_down(ty.ggml()).ok_or_else(|| {
        ChainError::shape(
            what,
            format!("a {ty} down `_sel`, which the card kernel table names no entry for here"),
        )
    })
}

/// The dense site a shared expert's part of `ty` kept apart from its stack
/// runs at ([`qwen35_shared`]), or a named refusal.
pub fn shared_entry(ty: SiteTy, what: &'static str) -> Result<SiteEntry, ChainError> {
    qwen35_shared(ty.ggml()).ok_or_else(|| {
        ChainError::shape(
            what,
            format!(
                "a {ty} shared expert part, which the card kernel table names no dense site for"
            ),
        )
    })
}

/// A layer's mixer kind, from its description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind35 {
    Gqa(GqaKind, Flash),
    Delta(LinearShape),
    /// A gated short conv; its conv ring holds the taps.
    ShortConv(Ring),
}

/// The geometry the card's kernels are built for: query heads of 256 with
/// 64 values turned, in groups the shape table's head-256 rows of eight or
/// of pairs take; delta heads of 128 with a 4-tap conv; experts of 512 with
/// a sigmoid-gated shared expert of 512, or a dense FFN of whole K-quant
/// super-blocks. The routed experts' count and the experts a token keeps are
/// the router instance's the file's shape selects ([`RouterDims::of`]); the
/// head counts are the file's.
pub mod q35 {
    pub const HEAD_DIM: u32 = 256;
    pub const ROPE_DIMS: u32 = 64;
    pub const EXPERT_FF: u32 = 512;
    /// Values of a K-quant super-block: a dense width's unit.
    pub(crate) const SUPER_BLOCK: u32 = 256;
}

const WHAT: &str = "qwen3moe plan";

/// Layer `l`'s mixer kind from its description `s`, by its geometry alone:
/// the attention class its head selects ([`GqaKind`]), the router row and the
/// shared expert its mixture's shape selects, whole K-quant blocks in a
/// dense width. No constant of one backend's instances is read here: a
/// backend holds the layers to the instances it registers (the card's:
/// [`kind35`]). A layer no kernel runs is refused by name with what it holds.
pub fn kind_of(s: &LayerSpec, l: usize) -> Result<Kind35, ChainError> {
    let refuse = |d: String| ChainError::shape(WHAT, format!("layer {l}: {d}"));
    let kind = match &s.mixer {
        Mixer::Gqa(g) => {
            let (kind, flash) = gqa_fits(g).map_err(refuse)?;
            Kind35::Gqa(kind, flash)
        }
        Mixer::DeltaRule(d) => Kind35::Delta(delta_shape(d).map_err(refuse)?),
        Mixer::ShortConv(c) => Kind35::ShortConv(conv_ring(c).map_err(refuse)?),
        other @ (Mixer::Latent(_) | Mixer::Mamba2(_) | Mixer::None) => {
            return Err(refuse(format!(
                "{}, which no kernel here runs",
                other.what()
            )));
        }
    };
    match &s.ffn {
        Ffn::Moe(m) => {
            moe_router(m).map_err(refuse)?;
        }
        Ffn::Dense { ff, act } => dense_fits(*ff, *act).map_err(refuse)?,
    }
    Ok(kind)
}

/// Every layer's kind of a description, in order ([`kind_of`]).
pub fn kinds_of(layers: &[LayerSpec]) -> Result<Vec<Kind35>, ChainError> {
    layers
        .iter()
        .enumerate()
        .map(|(l, s)| kind_of(s, l))
        .collect()
}

/// Layer `l`'s mixer kind for the card: [`kind_of`], and the layer is the
/// gated attention of head 256 with routed experts of [`q35::EXPERT_FF`] and
/// their shared expert, or a dense FFN, the instances the card's kernels are
/// built for. Any other layer is refused by name with what it holds.
pub fn kind35(s: &LayerSpec, l: usize) -> Result<Kind35, ChainError> {
    let kind = kind_of(s, l)?;
    card_fits(s).map_err(|d| ChainError::shape(WHAT, format!("layer {l}: {d}")))?;
    Ok(kind)
}

/// Every layer's kind of a Qwen3.6 description, in order ([`kind35`]).
pub fn kinds35(layers: &[LayerSpec]) -> Result<Vec<Kind35>, ChainError> {
    layers
        .iter()
        .enumerate()
        .map(|(l, s)| kind35(s, l))
        .collect()
}

/// The attention kinds the card's instances run: [`GqaKind::Gated256`], and the head-128
/// attention turned by NORM pairs with no head norm (the `_g7` and group-of-8 head-128 entries).
/// A short-conv layer and [`GqaKind::HeadGate128`] are in no list the card's kernels run.
const CARD_KINDS: [GqaKind; 2] = [
    GqaKind::Gated256,
    GqaKind::Head128 {
        pairs: Pairs::Norm,
        head_norm: false,
    },
];

/// Why the card's instances do not run `s`, which [`kind_of`] reads.
fn card_fits(s: &LayerSpec) -> Result<(), String> {
    if !matches!(s.residual, Residual::Plain) {
        return Err(format!(
            "the residual {:?}; the card's ops join each half's output by the plain add",
            s.residual
        ));
    }
    if let Mixer::ShortConv(c) = &s.mixer {
        return Err(format!(
            "a short-conv mixer of {} taps; the card's kernels take the gated attention of heads \
             of {} or heads of {HEAD_128} turned by NORM pairs with no head norm, and the delta \
             rule",
            c.taps,
            q35::HEAD_DIM
        ));
    }
    if let Mixer::Gqa(g) = &s.mixer {
        let kind = gqa_class(g)?;
        if !CARD_KINDS.contains(&kind) {
            return Err(format!(
                "attention {g:?}; the card's kernels take the gated attention of heads of {} or \
                 heads of {HEAD_128} turned by NORM pairs with no head norm",
                q35::HEAD_DIM
            ));
        }
        // The class table lets a layer keep what its description says (a window, rope factors),
        // and the card's launches cut no window and divide by no factors.
        if g.window.is_some() || g.rope.factors {
            return Err(format!(
                "attention {g:?}; the card's kernels attend every position and turn the rope by \
                 its base alone"
            ));
        }
        // The head-128 group of four is the head-gated window flash's block, which the card's
        // launch lists do not hold for this class: its head-128 flash covers a group of eight
        // or of seven.
        if matches!(kind, GqaKind::Head128 { .. })
            && select_gqa(AttnShape::of(g)).is_ok_and(|row| row.pack == 4)
        {
            return Err(format!(
                "attention {}/{} heads of {HEAD_128}; the card's head-128 flash covers a group of \
                 eight or of seven",
                g.heads, g.kv_heads
            ));
        }
    }
    if let Ffn::Moe(m) = &s.ffn {
        card_router(m)?;
    }
    Ok(())
}

/// Values of the head the head-128 kernels are built for; the rope turns all
/// of them.
pub const HEAD_128: u32 = 128;

/// The attention class `g`'s geometry selects, or why none does: the class
/// table's row ([`GqaKind::CLASSES`]) of its head width, the values the rope
/// turns, how it pairs them, its head norm and its output gate, with no yarn, output norm,
/// key selection, sinks or value scale, and a window or rope factors only where the class
/// keeps them per layer (the row's `window` and `rope_factors`).
pub(crate) fn gqa_class(g: &Gqa) -> Result<GqaKind, String> {
    let plain = g.rope.yarn.is_none()
        && !g.o_norm
        && g.select.is_none()
        && !g.sinks
        && g.value_scale.is_none();
    let pairs = if turns_as_neox(g.rope.mode, g.rope.dims) {
        Some(Pairs::Neox)
    } else if g.rope.mode == RopeMode::NormTail {
        Some(Pairs::Norm)
    } else {
        None
    };
    let class = GqaKind::CLASSES
        .iter()
        .find(|c| {
            g.head_dim == c.head
                && g.rope.dims == c.rope
                && pairs == Some(c.pairs)
                && g.qk_norm == c.head_norm
                && g.out_gate == c.gate
        })
        .filter(|c| (g.window.is_none() || c.window) && (!g.rope.factors || c.rope_factors));
    match class {
        Some(c) if plain => Ok(c.kind),
        _ => {
            let classes: Vec<String> = GqaKind::CLASSES
                .iter()
                .map(|c| {
                    format!(
                        "heads of {} with the first {} values turned by {:?} pairs (q/k norms {}, \
                         output gate {:?})",
                        c.head, c.rope, c.pairs, c.head_norm, c.gate
                    )
                })
                .collect();
            Err(format!(
                "attention {g:?}; the kernels take {} (NEOX by IMROPE sections covering them), \
                 with no yarn, output norm, key selection, sinks or value scale, a window only on \
                 the head with an output gate of its own, and rope factors only there and on the \
                 head-128 NORM attention with no head norm",
                classes.join(" or ")
            ))
        }
    }
}

/// The attention of layer `s`, which a plan reads as attention, with the class its geometry
/// selects ([`gqa_class`]), or why `s` is no attention layer.
pub(crate) fn layer_gqa(s: &LayerSpec) -> Result<(&Gqa, GqaKind), String> {
    match &s.mixer {
        Mixer::Gqa(g) => Ok((g, gqa_class(g)?)),
        other => Err(format!(
            "the plan reads attention; the description's mixer is {}",
            other.what()
        )),
    }
}

/// The class and the flash of `g` when it is an attention the kernels run:
/// the shape table's row for its group ([`select_gqa`]), one of eight, of
/// quads or pairs, the head-128 group of seven, or the head-128 group of four
/// (with an output gate of its own or none): [`Flash::Group`] names the one
/// block of a key head's group the row's pack makes.
fn gqa_fits(g: &Gqa) -> Result<(GqaKind, Flash), String> {
    let kind = gqa_class(g)?;
    let row = select_gqa(AttnShape::of(g)).map_err(|e| e.to_string())?;
    let flash = match (kind, row.pack, row.group) {
        (_, 8, GroupRule::One) => Flash::Group,
        (_, 4, GroupRule::Packs) => Flash::Quads,
        (_, 2, GroupRule::Packs) => Flash::Pairs,
        (GqaKind::Head128 { .. }, 7, GroupRule::One) => Flash::Seven,
        (GqaKind::HeadGate128 | GqaKind::Head128 { .. }, 4, GroupRule::One) => Flash::Group,
        _ => {
            return Err(format!(
                "attention {}/{} heads of {} selects {}, which this body does not launch",
                g.heads, g.kv_heads, g.head_dim, row.at
            ));
        }
    };
    Ok((kind, flash))
}

/// Ok when a dense FFN of `ff` values with `act` is the one the body runs:
/// SwiGLU with no limit through the routed launches at one slot, `ff` whole
/// K-quant super-blocks.
fn dense_fits(ff: u32, act: Act) -> Result<(), String> {
    if ff > 0 && ff.is_multiple_of(q35::SUPER_BLOCK) && matches!(act, Act::SwiGlu { limit: None }) {
        Ok(())
    } else {
        Err(format!(
            "a dense FFN of {ff} values with {act:?}; the body runs SwiGLU without a limit over \
             whole super-blocks of {}",
            q35::SUPER_BLOCK
        ))
    }
}

/// The conv ring of a short conv of `c`, or why none exists: a conv of no tap has no ring.
fn conv_ring(c: &ShortConv) -> Result<Ring, String> {
    if c.taps == 0 {
        return Err("a short conv of no tap".to_string());
    }
    Ok(Ring::new(c.taps as usize))
}

/// The delta kernels' launch shape of `d`, or why they do not run it.
fn delta_shape(d: &DeltaRule) -> Result<LinearShape, String> {
    let map = match d.kind {
        DeltaKind::Gdn {
            khead_map: KHeadMap::Tiled,
            gate: GdnGate::Silu,
        } => linear::KHeadMap::Tiled,
        DeltaKind::Gdn {
            gate: GdnGate::Sigmoid,
            ..
        } => {
            return Err("a sigmoid-gated GDN layer, which this body does not run".into());
        }
        DeltaKind::Kda { .. } => {
            return Err("a Kimi delta rule, which no kernel here runs".into());
        }
        DeltaKind::Rda { .. } => {
            return Err("a Raon delta rule, which no kernel here runs".into());
        }
    };
    let shape = LinearShape {
        n_k: d.k_heads as usize,
        n_v: d.v_heads as usize,
        map,
    };
    let fits = d.d as usize == linear::HEAD
        && d.conv as usize == linear::CONV_TAPS
        && shape.n_k > 0
        && shape.n_v.is_multiple_of(shape.n_k);
    if fits {
        Ok(shape)
    } else {
        Err(format!(
            "delta rule {d:?}; the kernels take heads of {} and a {}-tap conv",
            linear::HEAD,
            linear::CONV_TAPS
        ))
    }
}

/// The router `m` runs on when it is the routed FFN of the card's instances
/// ([`kind35`]): its mixture's shape as [`kind_of`] reads it, with experts of
/// [`q35::EXPERT_FF`] and the shared expert the gated router carries; any
/// other is refused by name as `what`'s.
pub fn moe_fits(m: &Moe, what: &'static str) -> Result<RouterDims, ChainError> {
    card_router(m).map_err(|e| ChainError::shape(what, e))
}

/// [`moe_fits`]'s router, or why the card's instances do not run `m`.
fn card_router(m: &Moe) -> Result<RouterDims, String> {
    use q35::EXPERT_FF;
    let router = moe_router(m)?;
    if router.gated() && m.expert_ff == EXPERT_FF {
        Ok(router)
    } else {
        Err(format!(
            "experts {m:?}; the card's kernels take experts of {EXPERT_FF} and their sigmoid-gated \
             shared expert"
        ))
    }
}

/// The router `m` runs on when it is the routed FFN the router bodies and the
/// stacks run: its instance and pick count from the file's shape
/// ([`RouterDims::of`], which refuses a count or rule no instance serves),
/// SwiGLU experts, softmax renormalized at scale 1, and a shared expert
/// exactly where the router is gated, SwiGLU and of the experts' width.
fn moe_router(m: &Moe) -> Result<RouterDims, String> {
    let router = RouterDims::of(MoeShape::of(m)).map_err(|e| e.to_string())?;
    let swiglu = matches!(m.act, Act::SwiGlu { limit: None });
    let plain = !m.router.hash && m.router.scale == 1.0;
    let shared = match (router.gated(), m.shared) {
        (false, None) => true,
        (true, Some(s)) => s.ff == m.expert_ff && matches!(s.act, Act::SwiGlu { limit: None }),
        (true, None) | (false, Some(_)) => false,
    };
    if swiglu && plain && shared {
        Ok(router)
    } else {
        Err(format!(
            "experts {m:?}; the kernels take SwiGLU experts, softmax renormalized at scale 1, and \
             a shared expert only as a sigmoid-gated SwiGLU one of the experts' width"
        ))
    }
}

/// The descriptions the chain's tests read: Qwen3.6's layers, Qwen3-30B-A3B's and LFM3's.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::q35;
    use model::arch::models::{
        Act, Arch, ChatSpec, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer,
        ModelSpec, Moe, OutGate, Residual, Rope, RopeMode, Router, RouterNorm, Score, Shared,
        ShortConv,
    };

    /// Qwen3.6's 16/2 heads.
    pub fn gqa() -> Mixer {
        gqa_of(16, 2)
    }

    pub fn gqa_of(heads: u32, kv_heads: u32) -> Mixer {
        Mixer::Gqa(Gqa {
            heads,
            kv_heads,
            head_dim: q35::HEAD_DIM,
            value_dim: q35::HEAD_DIM,
            rope: Rope {
                mode: RopeMode::Imrope {
                    sections: [11, 11, 10, 0],
                },
                dims: q35::ROPE_DIMS,
                base: 1e7,
                yarn: None,
                factors: false,
            },
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::Packed,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        })
    }

    pub fn delta() -> Mixer {
        Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Silu,
            },
            k_heads: 16,
            v_heads: 32,
            d: 128,
            conv: 4,
        })
    }

    /// A Qwen3.6 layer: `mixer`, 256 routed experts of 512 and the sigmoid-gated shared expert.
    pub fn layer(mixer: Mixer) -> LayerSpec {
        LayerSpec {
            mixer,
            ffn: Ffn::Moe(Moe {
                experts: 256,
                top_k: 8,
                expert_ff: q35::EXPERT_FF,
                latent: None,
                pre_exp_norm: false,
                post_exp_norm: None,
                act: Act::SwiGlu { limit: None },
                router: Router {
                    score: Score::Softmax,
                    bias: false,
                    norm: RouterNorm::SelectedSum,
                    scale: 1.0,
                    hash: false,
                },
                shared: Some(Shared {
                    ff: q35::EXPERT_FF,
                    act: Act::SwiGlu { limit: None },
                    sigmoid_gate: true,
                }),
            }),
            residual: Residual::Plain,
            extras: Vec::new(),
        }
    }

    /// Qwen3-30B-A3B's attention: 32/4 heads of 128, the whole head turned by NEOX, q/k norms, no
    /// output gate.
    pub fn q30_gqa() -> Gqa {
        Gqa {
            heads: 32,
            kv_heads: 4,
            head_dim: 128,
            value_dim: 128,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: 128,
                base: 1e7,
                yarn: None,
                factors: false,
            },
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        }
    }

    /// A Qwen3-30B-A3B layer: that attention, 128 routed experts of 768 and no shared expert.
    pub fn q30_layer() -> LayerSpec {
        let mut l = layer(Mixer::Gqa(q30_gqa()));
        if let Ffn::Moe(m) = &mut l.ffn {
            m.experts = 128;
            m.expert_ff = 768;
            m.shared = None;
        }
        l
    }

    /// LFM3's attention: 16/2 heads of 128 (a group of eight, a row the shape table has), the whole
    /// head turned by NEOX, q/k norms and a gate of one value a head; `window` the positions it
    /// attends and `factors` whether its rope divides by the file's table.
    pub fn lfm3_gqa(window: Option<u32>, factors: bool) -> Gqa {
        Gqa {
            heads: 16,
            kv_heads: 4,
            head_dim: 128,
            value_dim: 128,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: 128,
                base: if window.is_some() { 1e4 } else { 1e6 },
                yarn: None,
                factors,
            },
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::ProjHead,
            select: None,
            window,
            sinks: false,
            value_scale: None,
        }
    }

    /// A head-128 attention turned by NORM pairs, with no head norm and no output gate, at
    /// `heads`/`kv_heads`; its rope divides by the file's factors when `factors`.
    pub fn head128_gqa(heads: u32, kv_heads: u32, factors: bool) -> Gqa {
        Gqa {
            heads,
            kv_heads,
            head_dim: 128,
            value_dim: 128,
            rope: Rope {
                mode: RopeMode::NormTail,
                dims: 128,
                base: 5e5,
                yarn: None,
                factors,
            },
            qk_norm: false,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        }
    }

    /// A layer of `mixer` over a dense SwiGLU FFN of 512 and the plain residual.
    pub fn dense_layer(mixer: Mixer) -> LayerSpec {
        LayerSpec {
            mixer,
            ffn: Ffn::Dense {
                ff: 512,
                act: Act::SwiGlu { limit: None },
            },
            residual: Residual::Plain,
            extras: Vec::new(),
        }
    }

    /// LFM3's short conv of `taps` taps.
    pub fn lfm3_conv(taps: u32) -> Mixer {
        Mixer::ShortConv(ShortConv { taps })
    }

    /// An LFM3 layer: `mixer`, a dense SwiGLU FFN of 512 and the sandwich residual at 1e-5.
    pub fn lfm3_layer(mixer: Mixer) -> LayerSpec {
        LayerSpec {
            mixer,
            ffn: Ffn::Dense {
                ff: 512,
                act: Act::SwiGlu { limit: None },
            },
            residual: Residual::Sandwich {
                eps: 1e-5,
                add: [1.0, 1.0],
            },
            extras: Vec::new(),
        }
    }

    /// A description of `layers` over `hidden` values a residual row.
    pub fn spec(layers: Vec<LayerSpec>, hidden: u32) -> ModelSpec {
        ModelSpec {
            arch: Arch::Qwen35Moe,
            hidden,
            vocab: 1024,
            ctx_train: 4096,
            rms_eps: 1e-6,
            layers,
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
            chat: ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{
        delta, dense_layer, gqa, gqa_of, head128_gqa, layer, lfm3_conv, lfm3_gqa, lfm3_layer,
        q30_gqa, q30_layer,
    };
    use super::{
        Flash, GqaKind, Kind35, Pairs, gqa_class, kind_of, kind35, kinds_of, kinds35, moe_fits,
    };
    use crate::linear::{KHeadMap as LinearMap, LinearShape, Ring};
    use model::arch::models::{
        Act, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer, OutGate,
        Residual, Rope, RopeMode, Shared,
    };

    /// The plan's kinds follow the description layer by layer, wherever its
    /// attention layers sit: a description whose attention layers are 0 and
    /// 5 of 8 (not every fourth) gets exactly those two, and the rest the
    /// delta rule at the description's heads.
    #[test]
    fn kinds_come_from_the_description_not_the_layer_number() {
        let attn = [0usize, 5];
        let layers: Vec<LayerSpec> = (0..8)
            .map(|l| layer(if attn.contains(&l) { gqa() } else { delta() }))
            .collect();
        let got = kinds35(&layers).unwrap_or_else(|e| panic!("{e}"));
        let shape = LinearShape {
            n_k: 16,
            n_v: 32,
            map: LinearMap::Tiled,
        };
        let want: Vec<Kind35> = (0..8)
            .map(|l| {
                if attn.contains(&l) {
                    Kind35::Gqa(GqaKind::Gated256, Flash::Group)
                } else {
                    Kind35::Delta(shape)
                }
            })
            .collect();
        assert_eq!(got, want);
    }

    /// The routed shape comes from the file: the wide instance's 512/10
    /// and any pick count up to the lane picks run, an expert count no
    /// instance is built for is refused by name with its index.
    #[test]
    fn the_routed_shape_selects_its_instance() {
        let with = |experts: u32, top_k: u32| {
            let mut l = layer(delta());
            if let Ffn::Moe(m) = &mut l.ffn {
                m.experts = experts;
                m.top_k = top_k;
            }
            l
        };
        for (experts, top_k) in [(256u32, 8u32), (512, 10), (256, 1), (512, 32)] {
            let got = kinds35(&[with(experts, top_k)]);
            assert!(got.is_ok(), "{experts}/{top_k}: {got:?}");
        }
        for (experts, top_k) in [(384u32, 8u32), (256, 33), (256, 0)] {
            match kinds35(&[layer(delta()), with(experts, top_k)]) {
                Err(e) => assert!(e.to_string().contains("layer 1:"), "{e}"),
                Ok(k) => panic!("{experts}/{top_k} accepted: {k:?}"),
            }
        }
    }

    /// A layer the kernels were not built for is refused by name with its
    /// index: a head of 128, and an FFN without its shared expert.
    #[test]
    fn a_layer_the_kernels_do_not_take_is_refused_by_name() {
        let mut narrow = layer(gqa());
        if let Mixer::Gqa(g) = &mut narrow.mixer {
            g.head_dim = 128;
        }
        let mut bare = layer(delta());
        if let Ffn::Moe(m) = &mut bare.ffn {
            m.shared = None;
        }
        for (at, bad) in [(1usize, narrow), (2, bare)] {
            let mut layers = vec![layer(delta()), layer(delta()), layer(delta())];
            layers[at] = bad;
            match kinds35(&layers) {
                Err(e) => assert!(e.to_string().contains(&format!("layer {at}:")), "{e}"),
                Ok(k) => panic!("layer {at} accepted: {k:?}"),
            }
        }
    }

    /// Qwen3.5-9B's layers run: 24/4 heads take the pairs flash, 16/48
    /// delta heads the delta shape, and a dense FFN of 17408 the one-slot
    /// route; a group the body launches no flash for (3: neither eight nor a
    /// whole number of pairs) and a dense width of no whole super-block are
    /// refused by name.
    #[test]
    fn a_dense_qwen35_layer_runs_and_its_misfits_are_refused() {
        let dense = |mixer: Mixer, ff: u32| LayerSpec {
            ffn: Ffn::Dense {
                ff,
                act: Act::SwiGlu { limit: None },
            },
            ..layer(mixer)
        };
        let wide = Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Silu,
            },
            k_heads: 16,
            v_heads: 48,
            d: 128,
            conv: 4,
        });
        let got = kinds35(&[dense(wide.clone(), 17408), dense(gqa_of(24, 4), 17408)])
            .unwrap_or_else(|e| panic!("{e}"));
        let shape = LinearShape {
            n_k: 16,
            n_v: 48,
            map: LinearMap::Tiled,
        };
        assert_eq!(
            got,
            [
                Kind35::Delta(shape),
                Kind35::Gqa(GqaKind::Gated256, Flash::Pairs)
            ]
        );
        for (at, bad) in [
            (0usize, dense(gqa_of(24, 8), 17408)),
            (1, dense(wide, 17400)),
        ] {
            let mut layers = vec![dense(gqa_of(24, 4), 17408), dense(gqa_of(24, 4), 17408)];
            layers[at] = bad;
            match kinds35(&layers) {
                Err(e) => assert!(e.to_string().contains(&format!("layer {at}:")), "{e}"),
                Ok(k) => panic!("layer {at} accepted: {k:?}"),
            }
        }
    }

    /// A head-128 attention with no gate: Nanbeige's 56/8 heads (group 7) and a
    /// group of 8 pick the NORM append and their flash by the shape table's
    /// row; a NEOX turn, a head norm, an output norm, a gate, a rope over part
    /// of the head, a group no head-128 flash covers and a head no kernel is
    /// built for are each refused by name with the layer.
    #[test]
    fn a_head_128_norm_attention_picks_its_kind_and_flash() {
        let norm = GqaKind::Head128 {
            pairs: Pairs::Norm,
            head_norm: false,
        };
        let dense = |mixer: Mixer| LayerSpec {
            ffn: Ffn::Dense {
                ff: 22016,
                act: Act::SwiGlu { limit: None },
            },
            ..layer(mixer)
        };
        let head_128 = |heads: u32, kv_heads: u32, edit: &dyn Fn(&mut Gqa)| {
            let mut g = Gqa {
                heads,
                kv_heads,
                head_dim: 128,
                value_dim: 128,
                rope: Rope {
                    mode: RopeMode::NormTail,
                    dims: 128,
                    base: 1.2e8,
                    yarn: None,
                    factors: false,
                },
                qk_norm: false,
                o_norm: false,
                out_gate: OutGate::None,
                select: None,
                window: None,
                sinks: false,
                value_scale: None,
            };
            edit(&mut g);
            dense(Mixer::Gqa(g))
        };
        for (heads, kv_heads, flash) in [(56, 8, Flash::Seven), (64, 8, Flash::Group)] {
            let got = kinds35(&[head_128(heads, kv_heads, &|_: &mut Gqa| {})])
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, [Kind35::Gqa(norm, flash)], "{heads}/{kv_heads}");
        }
        type Misfit<'a> = (&'a str, u32, &'a dyn Fn(&mut Gqa));
        let misfits: [Misfit; 9] = [
            ("a NEOX turn", 8, &|g: &mut Gqa| {
                g.rope.mode = RopeMode::Neox
            }),
            ("a head norm", 8, &|g: &mut Gqa| g.qk_norm = true),
            ("an output norm", 8, &|g: &mut Gqa| g.o_norm = true),
            ("a packed gate", 8, &|g: &mut Gqa| {
                g.out_gate = OutGate::Packed
            }),
            ("a rope over 64 values", 8, &|g: &mut Gqa| g.rope.dims = 64),
            ("a rope with frequency factors", 8, &|g: &mut Gqa| {
                g.rope.factors = true
            }),
            ("no rope", 8, &|g: &mut Gqa| g.rope.mode = RopeMode::None),
            ("a group of 5", 5, &|_: &mut Gqa| {}),
            ("a head of 64", 8, &|g: &mut Gqa| {
                g.head_dim = 64;
                g.value_dim = 64;
            }),
        ];
        for (what, group, edit) in misfits {
            match kinds35(&[dense(delta()), head_128(8 * group, 8, edit)]) {
                Err(e) => assert!(e.to_string().contains("layer 1:"), "{what}: {e}"),
                Ok(k) => panic!("{what} accepted: {k:?}"),
            }
        }
    }

    /// A routed layer's gate·up and down launch at the table's entries of
    /// their types, a Q5_K one the K-quant `_sel` family's; a type with no
    /// entry for the dispatch is refused by name. (Which kernel family a
    /// layer loads for them is the card crate's, tested there.)
    #[test]
    fn a_routed_layer_takes_the_table_entries_of_its_types() {
        use super::{DownEntry, GateUpEntry, SiteTy, down_entry, gate_up_entry};
        use SiteTy::{Q3K, Q4K, Q5K, Q6K, Q8_0};
        assert_eq!(gate_up_entry(Q5K, Q5K, "t").ok(), Some(GateUpEntry::Q5k));
        assert_eq!(down_entry(Q5K, "t").ok(), Some(DownEntry::Q5k));
        for ty in [Q3K, Q6K, Q8_0] {
            let e = gate_up_entry(ty, ty, "t").expect_err("a gate·up with no entry");
            assert!(e.to_string().contains(&format!("{ty} gate·up")), "{e}");
        }
        for ty in [Q3K, Q8_0] {
            let e = down_entry(ty, "t").expect_err("a down with no entry");
            assert!(e.to_string().contains(&format!("{ty} down")), "{e}");
        }
        let e = gate_up_entry(Q4K, Q5K, "t").expect_err("a gate beside another type's up");
        assert!(e.to_string().contains("one gate·up launch"), "{e}");
    }

    /// A shared expert comes only with its router row: the 128-expert router has no gate row, so a
    /// sigmoid-gated shared expert beside it has no router to run on, a shared expert with no
    /// sigmoid gate has none anywhere, and one of another width than the experts is read at the
    /// wrong rows; each is refused by name with the layer.
    #[test]
    fn a_shared_expert_comes_only_with_its_router_row() {
        let shared = |ff: u32, sigmoid_gate: bool| Shared {
            ff,
            act: Act::SwiGlu { limit: None },
            sigmoid_gate,
        };
        let mut gated = q30_layer();
        let mut ungated = q30_layer();
        let mut narrow = layer(delta());
        for (l, s) in [
            (&mut gated, shared(768, true)),
            (&mut ungated, shared(768, false)),
            (&mut narrow, shared(256, true)),
        ] {
            if let Ffn::Moe(m) = &mut l.ffn {
                m.shared = Some(s);
            }
        }
        for (what, bad) in [("gated", gated), ("ungated", ungated), ("narrow", narrow)] {
            match kind_of(&bad, 3) {
                Err(e) => assert!(e.to_string().contains("layer 3:"), "{what}: {e}"),
                Ok(k) => panic!("{what} shared expert accepted: {k:?}"),
            }
        }
    }

    /// The attention class is the head's geometry: Qwen3-30B-A3B's layers are read as the
    /// whole-head NEOX class, and a head-128 attention with an output gate, with a partial rope or
    /// with a window is no class, a head-256 one with no gate neither; each is refused by name with
    /// the layer.
    #[test]
    fn an_attention_no_class_describes_is_refused_by_name() {
        let got = kinds_of(&[q30_layer(), q30_layer()]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, [Kind35::Gqa(GqaKind::Neox128, Flash::Group); 2]);
        assert_eq!(gqa_class(&q30_gqa()), Ok(GqaKind::Neox128));
        let class_of = |m: Mixer| match m {
            Mixer::Gqa(g) => gqa_class(&g),
            other => panic!("{other:?}"),
        };
        assert_eq!(class_of(gqa()), Ok(GqaKind::Gated256));
        let mut gated = q30_gqa();
        gated.out_gate = OutGate::Packed;
        let mut partial = q30_gqa();
        partial.rope.dims = 64;
        let mut windowed = q30_gqa();
        windowed.window = Some(1024);
        let mut bare = match gqa() {
            Mixer::Gqa(g) => g,
            other => panic!("{other:?}"),
        };
        bare.out_gate = OutGate::None;
        for (what, g) in [
            ("gated head 128", gated),
            ("partial rope", partial),
            ("windowed", windowed),
            ("head 256 with no gate", bare),
        ] {
            let mut layers = vec![q30_layer(), q30_layer()];
            layers[1].mixer = Mixer::Gqa(g);
            match kinds_of(&layers) {
                Err(e) => assert!(e.to_string().contains("layer 1:"), "{what}: {e}"),
                Ok(k) => panic!("{what} accepted: {k:?}"),
            }
        }
    }

    /// The chain reads layers by geometry and holds no backend's instance: routed experts of 768
    /// and Qwen3-30B-A3B's attention are read by [`kind_of`], and the card's entry points, whose
    /// kernels are built for experts of 512 and the gated head of 256, refuse each by name.
    #[test]
    fn the_card_refuses_what_the_chain_reads() {
        let mut wide = layer(delta());
        if let Ffn::Moe(m) = &mut wide.ffn {
            m.expert_ff = 768;
            if let Some(s) = &mut m.shared {
                s.ff = 768;
            }
        }
        for (what, l) in [("experts of 768", &wide), ("head 128", &q30_layer())] {
            assert!(kind_of(l, 0).is_ok(), "{what}");
            match kind35(l, 0) {
                Err(e) => assert!(e.to_string().contains("layer 0:"), "{what}: {e}"),
                Ok(k) => panic!("{what}: the card accepted {k:?}"),
            }
        }
        assert!(kinds35(&[q30_layer()]).is_err());
        let Ffn::Moe(m) = &wide.ffn else {
            panic!("a routed layer")
        };
        assert!(moe_fits(m, "t").is_err());
        let Ffn::Moe(m) = &layer(delta()).ffn else {
            panic!("a routed layer")
        };
        assert!(moe_fits(m, "t").is_ok());
    }

    /// LFM3's layers read by their geometry: a short conv is its ring (seven taps keep the reach
    /// of six and a pass of eight, fourteen slots), and attention of head 128 with a gate of one
    /// value a head is one kind on a full layer, whose rope has the file's factors, and on a
    /// sliding one, which has a window.
    #[test]
    fn an_lfm3_layer_reads_as_its_kind() {
        let layers = [
            lfm3_layer(lfm3_conv(7)),
            lfm3_layer(Mixer::Gqa(lfm3_gqa(None, true))),
            lfm3_layer(Mixer::Gqa(lfm3_gqa(Some(1024), false))),
        ];
        let got = kinds_of(&layers).unwrap_or_else(|e| panic!("{e}"));
        let head_gate = Kind35::Gqa(GqaKind::HeadGate128, Flash::Group);
        assert_eq!(got, [Kind35::ShortConv(Ring::new(7)), head_gate, head_gate]);
        let Kind35::ShortConv(ring) = got[0] else {
            panic!("{:?}", got[0])
        };
        assert_eq!((ring.taps(), ring.rows()), (7, 14));
        assert!(GqaKind::HeadGate128.head_gate() && GqaKind::HeadGate128.head_norm());
        assert!(!GqaKind::Neox128.head_gate() && !GqaKind::Gated256.head_gate());
    }

    /// The head with a gate of its own is the one class that keeps a window and rope factors per
    /// layer; what else it does not take is refused by name with the layer: no q/k norm, a gate
    /// of one value a channel, a rope over part of the head or NORM pairs, an output norm, sinks
    /// and a value scale. A short conv of no tap has no ring.
    #[test]
    fn what_the_head_gate_class_does_not_take_is_refused_by_name() {
        type Misfit<'a> = (&'a str, &'a dyn Fn(&mut Gqa));
        let misfits: [Misfit; 7] = [
            ("no q/k norm", &|g: &mut Gqa| g.qk_norm = false),
            ("a gate a channel", &|g: &mut Gqa| {
                g.out_gate = OutGate::Proj
            }),
            ("a rope over 64 values", &|g: &mut Gqa| g.rope.dims = 64),
            ("NORM pairs", &|g: &mut Gqa| {
                g.rope.mode = RopeMode::NormTail
            }),
            ("an output norm", &|g: &mut Gqa| g.o_norm = true),
            ("sinks", &|g: &mut Gqa| g.sinks = true),
            ("a value scale", &|g: &mut Gqa| g.value_scale = Some(2.0)),
        ];
        for (what, edit) in misfits {
            let mut g = lfm3_gqa(Some(1024), true);
            edit(&mut g);
            let layers = [lfm3_layer(lfm3_conv(7)), lfm3_layer(Mixer::Gqa(g))];
            match kinds_of(&layers) {
                Err(e) => assert!(e.to_string().contains("layer 1:"), "{what}: {e}"),
                Ok(k) => panic!("{what} accepted: {k:?}"),
            }
        }
        match kind_of(&lfm3_layer(lfm3_conv(0)), 4) {
            Err(e) => assert!(
                e.to_string().contains("layer 4:") && e.to_string().contains("no tap"),
                "{e}"
            ),
            Ok(k) => panic!("a conv of no tap accepted: {k:?}"),
        }
    }

    /// The card's kernels run neither a short conv, nor the head with a gate of its own, nor a
    /// residual that is not the plain add, so the card's entry points refuse each by name with the
    /// layer while the chain's read them.
    #[test]
    fn the_card_refuses_the_lfm3_layers_the_chain_reads() {
        let gqa_layer = lfm3_layer(Mixer::Gqa(lfm3_gqa(None, true)));
        let mut sandwiched = layer(delta());
        sandwiched.residual = Residual::Sandwich {
            eps: 1e-5,
            add: [1.0, 1.0],
        };
        for (what, l) in [
            ("a short conv", lfm3_layer(lfm3_conv(7))),
            ("the gated head", gqa_layer),
            ("a sandwich residual", sandwiched),
        ] {
            assert!(kind_of(&l, 2).is_ok(), "{what}");
            match kind35(&l, 2) {
                Err(e) => assert!(e.to_string().contains("layer 2:"), "{what}: {e}"),
                Ok(k) => panic!("{what}: the card accepted {k:?}"),
            }
        }
    }
    /// The head-128 NORM attention keeps what its description says of its rope, and a group of
    /// four: the chain reads a layer whose rope divides by the file's factors and the group of
    /// four (32 query heads over 8 key heads) as it reads the group of eight and of seven, one
    /// block of the key head's group, and the card, whose launches cut no window, divide by no
    /// factors and cover the head-128 groups of eight and seven, refuses each by name with the
    /// layer; a layer with neither is the card's as before. A window and a NEOX turn with no head
    /// norm stay no class. Mutants (each red at its line): the class row's `rope_factors` false
    /// (the chain's), `gqa_fits` without the group-of-four arm (the chain's), `card_fits` without
    /// its factors check or without its group-of-four check (the card's).
    #[test]
    fn a_head_128_norm_attention_reads_rope_factors_and_a_group_of_four_the_card_refuses() {
        let norm = GqaKind::Head128 {
            pairs: Pairs::Norm,
            head_norm: false,
        };
        let layer_of = |heads: u32, kv_heads: u32, factors: bool| {
            dense_layer(Mixer::Gqa(head128_gqa(heads, kv_heads, factors)))
        };
        // The group of four is 32 query heads over 8 key heads.
        for (heads, kv_heads, factors, flash) in [
            (32, 8, false, Flash::Group),
            (32, 8, true, Flash::Group),
            (64, 8, true, Flash::Group),
            (56, 8, true, Flash::Seven),
        ] {
            let got = kind_of(&layer_of(heads, kv_heads, factors), 0)
                .unwrap_or_else(|e| panic!("{heads}/{kv_heads} {factors}: {e}"));
            assert_eq!(
                got,
                Kind35::Gqa(norm, flash),
                "{heads}/{kv_heads} {factors}"
            );
        }
        for (what, l, text) in [
            (
                "a group of four",
                layer_of(32, 8, false),
                "eight or of seven",
            ),
            (
                "a group of four with factors",
                layer_of(32, 8, true),
                "attend every position",
            ),
            (
                "factors at a group of eight",
                layer_of(64, 8, true),
                "attend every position",
            ),
            (
                "factors at a group of seven",
                layer_of(56, 8, true),
                "attend every position",
            ),
        ] {
            match kind35(&l, 3) {
                Err(e) => {
                    let e = e.to_string();
                    assert!(e.contains("layer 3:") && e.contains(text), "{what}: {e}");
                }
                Ok(k) => panic!("{what}: the card accepted {k:?}"),
            }
        }
        for (heads, kv_heads, flash) in [(64, 8, Flash::Group), (56, 8, Flash::Seven)] {
            let got =
                kind35(&layer_of(heads, kv_heads, false), 0).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, Kind35::Gqa(norm, flash), "{heads}/{kv_heads}");
        }
        let mut windowed = head128_gqa(64, 8, false);
        windowed.window = Some(1024);
        let mut neox = head128_gqa(64, 8, false);
        neox.rope.mode = RopeMode::Neox;
        for (what, g) in [
            ("a window", windowed),
            ("NEOX pairs with no head norm", neox),
        ] {
            match kind_of(&dense_layer(Mixer::Gqa(g)), 2) {
                Err(e) => assert!(e.to_string().contains("layer 2:"), "{what}: {e}"),
                Ok(k) => panic!("{what} accepted: {k:?}"),
            }
        }
    }

    /// What a kind keeps per layer is its class row's: rope factors on the head-128 NORM attention
    /// and on the head with a gate of its own and on no other.
    #[test]
    fn a_kind_reports_the_rope_its_class_row_says() {
        for class in &GqaKind::CLASSES {
            assert_eq!(
                class.kind.rope_factors(),
                matches!(class.kind, GqaKind::HeadGate128 | GqaKind::Head128 { .. }),
                "{:?}",
                class.kind
            );
            assert_eq!(class.kind.head_norm(), class.head_norm, "{:?}", class.kind);
            assert_eq!(class.window, class.kind.head_gate(), "{:?}", class.kind);
        }
        let neox_bare = GqaKind::Head128 {
            pairs: Pairs::Neox,
            head_norm: false,
        };
        assert!(
            !neox_bare.rope_factors(),
            "a kind no class row holds keeps none"
        );
    }
}
