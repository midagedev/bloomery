//! The groups and the dispatches one qwen35moe token encodes
//! ([`token_count`]), and one verify of two or more rows
//! ([`verify_count`]).
//!
//! The caller passes the layers it has. An empty list is the embed and the
//! head's two groups, not a hidden default trunk.
//!
//! The token encoder is the caller: the e2e gate holds what one token, and
//! one verify, encodes to these counts.

use core::num::NonZeroU32;

use crate::kernels::{Quant, headnorm_m_serves};

/// One layer of the token.
///
/// `qkv`, `v` and `in_proj` are the projection that is P1: one dispatch for
/// each weight type its tensors hold, all in one group (`qkv` on a delta
/// layer, `v` on an attention layer, `in_proj` on a short conv layer). `ffn` is
/// how the layer's FFN picks its slots, `residual` how its two halves join the
/// stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// A gated delta layer. The live-key length does not add a dispatch.
    Delta {
        qkv: Projection,
        /// The out projection's weight type: the verify runs it as one M-row
        /// dispatch where `headnorm_m_serves`, else a dispatch a row. A token
        /// has one row, so `token_count` ignores it.
        out: Quant,
        ffn: FfnKind,
        residual: Residual,
    },
    /// A gated attention layer. Live keys past 64 add the memory dispatch. A
    /// `bias` adds one group of one dispatch after P1.
    Attention {
        v: Projection,
        bias: bool,
        ffn: FfnKind,
        residual: Residual,
    },
    /// A gated short conv layer.
    ShortConv {
        in_proj: Projection,
        ffn: FfnKind,
        residual: Residual,
    },
}

/// How a layer's FFN picks its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// The router picks them: a group of its own ahead of the gate-up.
    Routed,
    /// One slot on expert 0 at weight 1: no router, no group.
    Dense,
}

/// How a layer's two halves, its mixer and its FFN, join the residual stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residual {
    /// The half's out projection (the mixer's out, the FFN's down) adds into
    /// the stream: no group of its own.
    Plain,
    /// The half's out projection adds into the stream, and a hand-on group
    /// after it writes the next reader's normed input: two more groups a
    /// layer.
    PlainHandOn,
    /// The half's projection writes a buffer of its own and a norm-and-add
    /// group joins it to the stream: two more groups a layer.
    Sandwich,
}

/// The groups of a layer's mixer: P1, the core, its out.
const MIXER_GROUPS: u32 = 3;

/// The groups an attention layer's q, k and v biases add: one dispatch after
/// P1, whose output the flash then reads.
const BIAS_GROUPS: u32 = 1;

/// The groups of the embed and of the head: the embed's one, and the head's
/// two, the logits and the pick.
const ENDS_GROUPS: u32 = 3;

/// The groups of a layer's FFN: the gate-up and the down, and the router's
/// ahead of them when it has one.
const FFN_GROUPS: u32 = 2;

/// The groups a sandwich adds: a join after the mixer's out and one after the
/// FFN's down.
const JOIN_GROUPS: u32 = 2;

/// The groups a plain residual that hands on adds: a hand-on after the
/// mixer's out and one after the FFN's down.
const HANDON_GROUPS: u32 = 2;

impl Layer {
    /// How the layer's FFN picks its slots.
    #[must_use]
    pub const fn ffn(&self) -> FfnKind {
        match self {
            Layer::Delta { ffn, .. }
            | Layer::Attention { ffn, .. }
            | Layer::ShortConv { ffn, .. } => *ffn,
        }
    }

    /// How the layer's two halves join the stream.
    #[must_use]
    pub const fn residual(&self) -> Residual {
        match self {
            Layer::Delta { residual, .. }
            | Layer::Attention { residual, .. }
            | Layer::ShortConv { residual, .. } => *residual,
        }
    }

    /// The dispatches of the layer's P1: its group's size.
    #[must_use]
    pub const fn p1(&self) -> Projection {
        match self {
            Layer::Delta { qkv, .. } => *qkv,
            Layer::Attention { v, .. } => *v,
            Layer::ShortConv { in_proj, .. } => *in_proj,
        }
    }

    /// The groups the layer encodes: its mixer's three and an attention
    /// layer's bias group, its FFN's two and a router's ahead of them, and a
    /// sandwich's two joins or a plain residual's two hand-ons.
    #[must_use]
    pub const fn groups(&self) -> u32 {
        let router = match self.ffn() {
            FfnKind::Routed => 1,
            FfnKind::Dense => 0,
        };
        let bias = match self {
            Layer::Attention { bias: true, .. } => BIAS_GROUPS,
            _ => 0,
        };
        let joins = match self.residual() {
            Residual::Plain => 0,
            Residual::PlainHandOn => HANDON_GROUPS,
            Residual::Sandwich => JOIN_GROUPS,
        };
        MIXER_GROUPS + bias + FFN_GROUPS + router + joins
    }
}

/// How many dispatches P1 is, in one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection(NonZeroU32);

impl Projection {
    /// P1 is one dispatch.
    pub const ONE: Projection = Projection(NonZeroU32::MIN);

    /// P1 is two dispatches in one group.
    pub const TWO: Projection = Projection(NonZeroU32::new(2).expect("2 is not zero"));

    /// P1 as `dispatches` dispatches in one group; `None` for none or a
    /// count that does not fit `u32`.
    #[must_use]
    pub fn of(dispatches: usize) -> Option<Projection> {
        u32::try_from(dispatches)
            .ok()
            .and_then(NonZeroU32::new)
            .map(Projection)
    }

    /// The dispatches of P1.
    #[must_use]
    pub const fn dispatches(self) -> u32 {
        self.0.get()
    }
}

/// Whether the live keys are past 64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveKeys {
    /// At most 64: one attention segment.
    Within64,
    /// Past 64: the segment rule splits the keys. Since the flash folds the
    /// merge, the count does not grow past the waterline; the variant keeps
    /// the caller's two scenes apart.
    Past64,
}

/// Whether the router's ticket stays one dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ticket {
    /// The router is one dispatch and one group.
    Folded,
    /// Each layer's router is two dispatches and two groups. The count when a
    /// fence cannot fold those dispatches. Nothing encodes this path; the
    /// tests keep the number.
    Unfolded,
}

/// The groups and the dispatches of one token.
///
/// `groups` is the table's first number, `dispatches` the second.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct TokenCount {
    pub groups: u32,
    pub dispatches: u32,
}

/// The groups and the dispatches one token encodes for `layers`.
///
/// The embed is one group and the head two, the logits and the pick, so the
/// ends are the 3. Each layer adds [`Layer::groups`]: P1, the mixer, its out,
/// the router, the gate-up and the down, so six; a layer of a dense FFN has no
/// router, so five; a sandwich adds a join after the mixer's out and one after
/// the down, and a plain residual that hands on a hand-on after each, so
/// seven; an attention layer with biases adds the bias group after P1. Those already include the flash — the attention's fold
/// runs inside it, so past the segment rule's waterline the count is the
/// same — and the FFN norm, which the router and the gate-up each apply as
/// they load the residual (a dense layer's gate-up alone). A projection of
/// more than one dispatch adds the rest. An unfolded ticket adds one group for
/// each routed layer's router. A count that does not fit `u32` panics by name.
pub fn token_count(layers: &[Layer], _live: LiveKeys, ticket: Ticket) -> TokenCount {
    let mut in_layers = 0u32;
    let mut routers = 0u32;
    let mut n_split = 0u32;
    for layer in layers {
        in_layers = fit("layer groups", in_layers.checked_add(layer.groups()));
        n_split = fit("P1", n_split.checked_add(layer.p1().dispatches() - 1));
        if layer.ffn() == FfnKind::Routed {
            routers = fit("router", routers.checked_add(1));
        }
    }
    let mut groups = fit("group", ENDS_GROUPS.checked_add(in_layers));
    let mut dispatches = fit("dispatch", groups.checked_add(n_split));
    if matches!(ticket, Ticket::Unfolded) {
        groups = fit("group", groups.checked_add(routers));
        dispatches = fit("dispatch", dispatches.checked_add(routers));
    }
    TokenCount { groups, dispatches }
}

fn fit(what: &str, n: Option<u32>) -> u32 {
    match n {
        Some(n) => n,
        None => panic!("token count: the {what} count does not fit u32"),
    }
}

/// The groups and dispatches of a delta layer's core over a verify's rows
/// (`Kernels::delta_rows_ring`): the prep, a barrier, the conv ring's commit
/// and the delta scan, a barrier, the post: three groups, four dispatches.
const VERIFY_DELTA_CORE: (u32, u32) = (3, 4);

/// The groups and dispatches of one launch of an attention layer's flash over
/// a verify's rows (`Kernels::attention_rows_layer`): the append of the rows'
/// keys and values, a barrier, the flash.
const VERIFY_FLASH: (u32, u32) = (2, 2);

/// The groups of the verify's head: the logits, a barrier, the picks.
const VERIFY_HEAD_GROUPS: u32 = 2;

/// The groups and the dispatches one verify of `rows` rows encodes for
/// `layers`, the encoder's own count (`EncodeCount`): a group is the run of
/// dispatches between two barriers, so a verify has one barrier fewer than
/// its groups, the head's picks being its last group.
///
/// The verify's groups are not the token's: its delta core is three groups of
/// four dispatches and its flash two of two for each of `flash_launches`
/// launches (one while its rows share a segment count), the delta layers' out
/// projection one group of one dispatch where `headnorm_m_serves` its type at
/// the verify's rows (an A4 out at two rows) and else a dispatch a row, and
/// the head two groups, the logits and then a pick a row. The embed is one
/// group; P1, the attention's out, the gate-up, the down, an attention
/// layer's biases and each of a plain residual's two hand-ons are one group of
/// one dispatch each, P1 of its dispatches. What the verify does not serve (a
/// short conv layer, a routed FFN, a sandwich) panics by name, as a count that
/// does not fit `u32` does.
pub fn verify_count(layers: &[Layer], rows: u32, flash_launches: u32) -> TokenCount {
    let (mut groups, mut dispatches) = (1u32, 1u32);
    let mut add = |g: u32, d: u32| {
        groups = fit("verify group", groups.checked_add(g));
        dispatches = fit("verify dispatch", dispatches.checked_add(d));
    };
    for layer in layers {
        match *layer {
            Layer::Delta { qkv, out, .. } => {
                add(1, qkv.dispatches());
                add(VERIFY_DELTA_CORE.0, VERIFY_DELTA_CORE.1);
                add(
                    1,
                    if headnorm_m_serves(out, rows as usize) {
                        1
                    } else {
                        rows
                    },
                );
            }
            Layer::Attention { v, bias, .. } => {
                add(1, v.dispatches());
                if bias {
                    add(BIAS_GROUPS, BIAS_GROUPS);
                }
                let launches = |n: u32| fit("verify flash", n.checked_mul(flash_launches));
                add(launches(VERIFY_FLASH.0), launches(VERIFY_FLASH.1));
                add(1, 1);
            }
            Layer::ShortConv { .. } => panic!("verify count: a short conv layer is not served"),
        }
        match layer.ffn() {
            FfnKind::Dense => add(FFN_GROUPS, FFN_GROUPS),
            FfnKind::Routed => panic!("verify count: a routed FFN is not served"),
        }
        match layer.residual() {
            Residual::Plain => {}
            Residual::PlainHandOn => add(HANDON_GROUPS, HANDON_GROUPS),
            Residual::Sandwich => panic!("verify count: a sandwich residual is not served"),
        }
    }
    add(VERIFY_HEAD_GROUPS, rows.saturating_add(1));
    TokenCount { groups, dispatches }
}

/// What one group of a token runs: the kinds a profiled token's groups are
/// tagged with, one kind a group, the groups [`token_count`] counts. P1 is
/// one kind a mixer; the attention is one kind, the flash, which past one
/// segment of keys also folds its partials; the gate-up group and the down
/// group each carry the shared expert's part with the routed picks' — the
/// pair and the fold run them together, so no kind of the shared expert
/// stands alone. A sandwich's join is one kind for both halves, and so is a
/// plain residual's hand-on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupKind {
    /// The token's embedding row.
    Embed,
    /// An attention layer's P1: the norm on load and its projections.
    AttnP1,
    /// An attention layer's q, k and v biases, added to the P1 output.
    AttnBias,
    /// The attention core: the flash over the live keys, which past one
    /// segment also folds its partials.
    AttnFlash,
    /// The attention's output projection back to the residual.
    AttnOut,
    /// A delta layer's P1: the norm on load and its projections.
    DeltaP1,
    /// The gated delta rule's step.
    DeltaCore,
    /// The delta step's head-norm projection back to the residual.
    DeltaOut,
    /// A short conv layer's P1: the norm on load and its input projection.
    ConvP1,
    /// The short conv over the layer's ring.
    ConvCore,
    /// The conv's output projection, into a buffer of its own when the layer
    /// is a sandwich, else into the residual.
    ConvOut,
    /// A sandwich's join: the half's output normed and added to the stream.
    Join,
    /// A plain residual's hand-on: the stream times the next reader's gain,
    /// and that reader's scale.
    HandOn,
    /// The router: the routed logits, the picks and their weights. A dense
    /// FFN has none.
    Router,
    /// The experts' gate and up, the routed picks' and the shared expert's.
    GateUp,
    /// The experts' down, the routed picks' and the shared expert's, folded
    /// into the residual.
    Down,
    /// The head's logits: the norm on load and the output projection over
    /// the vocabulary.
    Head,
    /// The head's pick: the argmax over the logits.
    HeadPick,
}

impl GroupKind {
    /// The kind's name, the `op=` a profile line prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            GroupKind::Embed => "embed",
            GroupKind::AttnP1 => "attn_p1",
            GroupKind::AttnBias => "attn_bias",
            GroupKind::AttnFlash => "attn_flash",
            GroupKind::AttnOut => "attn_out",
            GroupKind::DeltaP1 => "delta_p1",
            GroupKind::DeltaCore => "delta_core",
            GroupKind::DeltaOut => "delta_out",
            GroupKind::ConvP1 => "conv_p1",
            GroupKind::ConvCore => "conv_core",
            GroupKind::ConvOut => "conv_out",
            GroupKind::Join => "join",
            GroupKind::HandOn => "hand_on",
            GroupKind::Router => "router",
            GroupKind::GateUp => "gate_up",
            GroupKind::Down => "down",
            GroupKind::Head => "head",
            GroupKind::HeadPick => "head_pick",
        }
    }
}

/// Every kind, in the token's walk order: the order a profile's lines print.
pub const KINDS: [GroupKind; 18] = [
    GroupKind::Embed,
    GroupKind::AttnP1,
    GroupKind::AttnBias,
    GroupKind::AttnFlash,
    GroupKind::AttnOut,
    GroupKind::DeltaP1,
    GroupKind::DeltaCore,
    GroupKind::DeltaOut,
    GroupKind::ConvP1,
    GroupKind::ConvCore,
    GroupKind::ConvOut,
    GroupKind::Join,
    GroupKind::HandOn,
    GroupKind::Router,
    GroupKind::GateUp,
    GroupKind::Down,
    GroupKind::Head,
    GroupKind::HeadPick,
];

/// The bytes one group of a token moves, by what moves them: the weights
/// the group's launches read, the activations they read and write, and the
/// KV or recurrent state they read and write. The byte rule that fills it is
/// the arena's, beside this table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupBytes {
    /// The weight bytes the group's launches read.
    pub weights: u64,
    /// The activation bytes the group reads and writes.
    pub acts: u64,
    /// The KV rows and recurrent state the group reads and writes.
    pub state: u64,
}

impl GroupBytes {
    /// The group's bytes altogether.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.weights + self.acts + self.state
    }
}

// PIN(2026-10-29): every folded count below is one group and one dispatch more
// than before: the head is two dispatches, the logits then the pick, with a
// barrier between them. The unfolded counts are the old ones: the head's
// ticket handoff is gone, so only a router unfolds.
#[cfg(test)]
mod tests {
    use super::*;

    fn routed_delta(qkv: Projection) -> Layer {
        Layer::Delta {
            qkv,
            out: Quant::A4,
            ffn: FfnKind::Routed,
            residual: Residual::Plain,
        }
    }

    fn routed_attention(v: Projection) -> Layer {
        Layer::Attention {
            v,
            bias: false,
            ffn: FfnKind::Routed,
            residual: Residual::Plain,
        }
    }

    /// `layers` with a dense FFN: no layer has a router.
    fn dense(layers: &[Layer]) -> Vec<Layer> {
        layers
            .iter()
            .map(|l| match *l {
                Layer::Delta {
                    qkv, out, residual, ..
                } => Layer::Delta {
                    qkv,
                    out,
                    ffn: FfnKind::Dense,
                    residual,
                },
                Layer::Attention {
                    v, bias, residual, ..
                } => Layer::Attention {
                    v,
                    bias,
                    ffn: FfnKind::Dense,
                    residual,
                },
                Layer::ShortConv {
                    in_proj, residual, ..
                } => Layer::ShortConv {
                    in_proj,
                    ffn: FfnKind::Dense,
                    residual,
                },
            })
            .collect()
    }

    /// LFM3-1.7B's 24 layers, `ccFcScFcScFcScFcScFcScFc`: a short conv (c), a
    /// full attention layer (F) or a sliding one (S), each over a dense FFN
    /// and a sandwich, each P1 one dispatch.
    fn lfm3_target() -> Vec<Layer> {
        "ccFcScFcScFcScFcScFcScFc"
            .chars()
            .map(|c| match c {
                'c' => Layer::ShortConv {
                    in_proj: Projection::ONE,
                    ffn: FfnKind::Dense,
                    residual: Residual::Sandwich,
                },
                'F' | 'S' => Layer::Attention {
                    v: Projection::ONE,
                    bias: false,
                    ffn: FfnKind::Dense,
                    residual: Residual::Sandwich,
                },
                other => panic!("no layer kind {other}"),
            })
            .collect()
    }

    fn fixture() -> [Layer; 4] {
        [
            routed_delta(Projection::ONE),
            routed_attention(Projection::ONE),
            routed_delta(Projection::TWO),
            routed_attention(Projection::TWO),
        ]
    }

    /// 16 delta layers of one P1 dispatch, 14 of two, 4 attention layers of one,
    /// 6 of two.
    fn target() -> [Layer; 40] {
        let mut layers = [routed_delta(Projection::ONE); 40];
        for layer in &mut layers[16..30] {
            *layer = routed_delta(Projection::TWO);
        }
        for layer in &mut layers[30..34] {
            *layer = routed_attention(Projection::ONE);
        }
        for layer in &mut layers[34..40] {
            *layer = routed_attention(Projection::TWO);
        }
        layers
    }

    #[test]
    fn fixture_folded_within_64() {
        assert_eq!(
            token_count(&fixture(), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 27,
                dispatches: 29
            }
        );
    }

    #[test]
    fn fixture_folded_past_64() {
        // The flash folds the merge, so past the waterline the count is the
        // within-waterline one: 29/31 before the fold, one merge group and
        // dispatch an attention layer.
        assert_eq!(
            token_count(&fixture(), LiveKeys::Past64, Ticket::Folded),
            token_count(&fixture(), LiveKeys::Within64, Ticket::Folded)
        );
        assert_eq!(
            token_count(&fixture(), LiveKeys::Past64, Ticket::Folded),
            TokenCount {
                groups: 27,
                dispatches: 29
            }
        );
    }

    #[test]
    fn fixture_unfolded_within_64() {
        assert_eq!(
            token_count(&fixture(), LiveKeys::Within64, Ticket::Unfolded),
            TokenCount {
                groups: 31,
                dispatches: 33
            }
        );
    }

    #[test]
    fn target_folded_within_64() {
        assert_eq!(
            token_count(&target(), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 243,
                dispatches: 263
            }
        );
    }

    #[test]
    fn target_folded_past_64() {
        // 253/273 before the fold: ten attention layers' merge group and
        // dispatch each.
        assert_eq!(
            token_count(&target(), LiveKeys::Past64, Ticket::Folded),
            TokenCount {
                groups: 243,
                dispatches: 263
            }
        );
    }

    #[test]
    fn target_unfolded_within_64() {
        assert_eq!(
            token_count(&target(), LiveKeys::Within64, Ticket::Unfolded),
            TokenCount {
                groups: 283,
                dispatches: 303
            }
        );
    }

    #[test]
    fn target_unfolded_past_64() {
        assert_eq!(
            token_count(&target(), LiveKeys::Past64, Ticket::Unfolded),
            TokenCount {
                groups: 283,
                dispatches: 303
            }
        );
    }

    /// Qwen3-30B-A3B's 48 attention layers, `attn_v` at Q6_K (a second P1
    /// dispatch) on the layers the Q4_K_M rule (`use_more_bits`) raises: the
    /// first and the last eighth, and every third layer between from the
    /// eighth on.
    fn q30_target() -> Vec<Layer> {
        (0..48)
            .map(|i| {
                routed_attention(if !(6..42).contains(&i) || (i - 6) % 3 == 2 {
                    Projection::TWO
                } else {
                    Projection::ONE
                })
            })
            .collect()
    }

    #[test]
    fn q30_target_folded_within_64() {
        // 3 + 6·48 groups, and the 24 raised layers' second P1 dispatch.
        assert_eq!(
            token_count(&q30_target(), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 291,
                dispatches: 315
            }
        );
    }

    /// A dense layer has no router group: five groups a layer, the folded
    /// ticket's count one group fewer for each, and with no router to unfold
    /// the unfolded count is the folded one.
    #[test]
    fn a_dense_layer_has_no_router_group() {
        assert_eq!(
            token_count(&dense(&fixture()), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 23,
                dispatches: 25
            }
        );
        assert_eq!(
            token_count(&dense(&fixture()), LiveKeys::Within64, Ticket::Unfolded),
            TokenCount {
                groups: 23,
                dispatches: 25
            }
        );
        // 3 + 5 · 64 groups for the 27B's layers.
        let layers = vec![routed_delta(Projection::ONE); 64];
        assert_eq!(
            token_count(&dense(&layers), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 323,
                dispatches: 323
            }
        );
        // One routed layer among dense ones adds its router's group.
        let mut mixed = dense(&fixture());
        mixed[1] = routed_attention(Projection::ONE);
        assert_eq!(
            token_count(&mixed, LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 24,
                dispatches: 26
            }
        );
    }

    /// A sandwich's two joins are two groups and two dispatches a layer, on
    /// the delta, attention and short conv layers alike, and no layer's kind
    /// changes the count of the rest: LFM3's 24 layers are 3 + 24 · 7 = 171
    /// groups and 171 dispatches, 13 of them short conv and 11 attention.
    /// Mutant: `JOIN_GROUPS` is 1, or `Layer::groups` counts the router of a
    /// dense layer.
    #[test]
    fn a_sandwich_adds_two_groups_a_layer() {
        let layers = lfm3_target();
        let convs = layers
            .iter()
            .filter(|l| matches!(l, Layer::ShortConv { .. }))
            .count();
        assert_eq!((layers.len(), convs), (24, 13));
        for ticket in [Ticket::Folded, Ticket::Unfolded] {
            let got = token_count(&layers, LiveKeys::Within64, ticket);
            assert_eq!(
                got,
                TokenCount {
                    groups: 171,
                    dispatches: 171
                },
                "{ticket:?}"
            );
        }
        // The same layers with the plain add: five groups a layer.
        let plain: Vec<Layer> = layers
            .iter()
            .map(|l| match *l {
                Layer::ShortConv { in_proj, ffn, .. } => Layer::ShortConv {
                    in_proj,
                    ffn,
                    residual: Residual::Plain,
                },
                Layer::Attention { v, bias, ffn, .. } => Layer::Attention {
                    v,
                    bias,
                    ffn,
                    residual: Residual::Plain,
                },
                Layer::Delta { qkv, out, ffn, .. } => Layer::Delta {
                    qkv,
                    out,
                    ffn,
                    residual: Residual::Plain,
                },
            })
            .collect();
        assert_eq!(
            token_count(&plain, LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 123,
                dispatches: 123
            }
        );
        // A short conv's P1 of two dispatches adds one dispatch, no group.
        let mut split = layers;
        split[0] = Layer::ShortConv {
            in_proj: Projection::TWO,
            ffn: FfnKind::Dense,
            residual: Residual::Sandwich,
        };
        assert_eq!(
            token_count(&split, LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 171,
                dispatches: 172
            }
        );
    }

    /// Qwen3.8-27B's 64 layers over a dense FFN, the plain residual handing
    /// on: seven groups a layer and the ends' three, 451 groups and 451
    /// dispatches (the plain add's 323 is `a_dense_layer_has_no_router_group`'s).
    /// Mutant: `HANDON_GROUPS` is 1, or `Layer::groups` counts a plain
    /// residual that hands on as one that does not.
    #[test]
    fn a_plain_hand_on_adds_two_groups_a_layer() {
        let layers: Vec<Layer> = dense(&vec![routed_delta(Projection::ONE); 64])
            .into_iter()
            .map(|l| match l {
                Layer::Delta { qkv, out, ffn, .. } => Layer::Delta {
                    qkv,
                    out,
                    ffn,
                    residual: Residual::PlainHandOn,
                },
                other => other,
            })
            .collect();
        assert_eq!(
            token_count(&layers, LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 451,
                dispatches: 451
            }
        );
    }

    /// Qwen2.5-Coder-7B's four-layer cut over a dense FFN and a plain
    /// residual that hands on: seven groups a layer (the mixer's three, the
    /// FFN's two, two hand-ons), an attention layer's bias group one more,
    /// and the ends' three, 35 groups and 35 dispatches against the unbiased
    /// cut's 31. Mutant: `BIAS_GROUPS` is 0, or `Layer::groups` reads the
    /// bias of a layer that has none.
    #[test]
    fn an_attention_layers_biases_add_one_group_and_one_dispatch() {
        let layer = |bias| Layer::Attention {
            v: Projection::ONE,
            bias,
            ffn: FfnKind::Dense,
            residual: Residual::PlainHandOn,
        };
        let count = |bias| token_count(&[layer(bias); 4], LiveKeys::Within64, Ticket::Folded);
        assert_eq!(
            count(false),
            TokenCount {
                groups: 31,
                dispatches: 31
            }
        );
        assert_eq!(
            count(true),
            TokenCount {
                groups: 35,
                dispatches: 35
            }
        );
        assert_eq!(layer(true).groups(), layer(false).groups() + 1);
    }

    /// The verify's layers over a dense FFN and a plain residual that hands
    /// on, `kinds` a delta (`d`) or an attention (`a`) layer of one P1
    /// dispatch each, an attention layer with biases when `bias`.
    fn verify_layers(kinds: &str, bias: bool) -> Vec<Layer> {
        kinds
            .chars()
            .map(|c| match c {
                'd' => Layer::Delta {
                    qkv: Projection::ONE,
                    out: Quant::A4,
                    ffn: FfnKind::Dense,
                    residual: Residual::PlainHandOn,
                },
                'a' => Layer::Attention {
                    v: Projection::ONE,
                    bias,
                    ffn: FfnKind::Dense,
                    residual: Residual::PlainHandOn,
                },
                other => panic!("no layer kind {other}"),
            })
            .collect()
    }

    /// A verify of two rows encodes what the layers' ops do, not the token's
    /// one group an op: a delta layer is P1, a core of three groups and four
    /// dispatches, an out of one group of one dispatch (the M-row head-normed
    /// entry serves its A4 out at two rows), two hand-ons, the pair and the
    /// down, 9 groups and 10 dispatches; an attention layer is P1, a flash of
    /// two groups and two dispatches, the out, two hand-ons, the pair and the
    /// down, 8 and 8; the embed is one of each and the head two groups of
    /// 1 + 2 dispatches. Qwen3.8-27B's four-layer cut (three delta layers and
    /// an attention layer) is 38 groups (1 + 3 · 9 + 8 + 2, the out one group
    /// whatever runs it) and 42 dispatches (1 + 3 · 10 + 8 + 3), Qwen3's four
    /// attention layers 35 and 36, and the 27B's 48 delta and 16 attention
    /// layers 563 (1 + 48 · 9 + 16 · 8 + 2) and 612 (1 + 48 · 10 + 16 · 8 +
    /// 3), where its profile's 498 groups count a core as one and a delta out
    /// as a group a row.
    ///
    /// Mutant: `VERIFY_DELTA_CORE` is (1, 1), the delta out a dispatch a row
    /// at two rows, the head's picks one dispatch, or a plain residual's
    /// hand-ons not counted.
    #[test]
    fn a_verify_encodes_the_ops_groups_not_the_tokens() {
        let count = |kinds: &str| verify_count(&verify_layers(kinds, false), 2, 1);
        assert_eq!(
            count("ddda"),
            TokenCount {
                groups: 38,
                dispatches: 42
            }
        );
        assert_eq!(
            count("aaaa"),
            TokenCount {
                groups: 35,
                dispatches: 36
            }
        );
        assert_eq!(
            count(&"ddda".repeat(16)),
            TokenCount {
                groups: 563,
                dispatches: 612
            }
        );
        assert_eq!(
            count(""),
            TokenCount {
                groups: 3,
                dispatches: 4
            }
        );
    }

    /// A delta layer whose out the M-row head-normed entry does not serve
    /// (a Q4_K out) keeps a dispatch a row at two rows: the same one group of
    /// the out, one dispatch more than the A4 out the entry folds into one
    /// launch.
    ///
    /// Mutant: the served arm is asked of the quant alone, so a Q4_K out at
    /// two rows counts one dispatch.
    #[test]
    fn a_q4k_delta_out_stays_a_dispatch_a_row() {
        let q4k = |kinds: &str| {
            verify_layers(kinds, false)
                .into_iter()
                .map(|l| match l {
                    Layer::Delta {
                        qkv, ffn, residual, ..
                    } => Layer::Delta {
                        qkv,
                        out: Quant::Q4K,
                        ffn,
                        residual,
                    },
                    other => other,
                })
                .collect::<Vec<_>>()
        };
        // One delta layer of 9 groups and the ends' 3: 12 groups either way
        // (the out one group), 15 dispatches at a Q4_K out (the out two:
        // 1 + 11 + 3) against 14 at an A4 out (the out one: 1 + 10 + 3).
        assert_eq!(
            verify_count(&q4k("d"), 2, 1),
            TokenCount {
                groups: 12,
                dispatches: 15
            }
        );
        assert_eq!(
            verify_count(&verify_layers("d", false), 2, 1),
            TokenCount {
                groups: 12,
                dispatches: 14
            }
        );
    }

    /// The rows past the M-row entry's count add a dispatch each to a delta
    /// layer's out (the entry serves two rows only) and to the head's picks,
    /// never a group; the flash's launches add two groups and two dispatches
    /// each to an attention layer; the biases add one of each.
    ///
    /// Mutant: the delta out or the head's picks count a dispatch a launch
    /// and not a row, `flash_launches` is not multiplied, or `BIAS_GROUPS` is
    /// 0.
    #[test]
    fn a_verifys_rows_flash_launches_and_biases_move_its_count() {
        let at = |kinds: &str, bias: bool, rows: u32, launches: u32| {
            verify_count(&verify_layers(kinds, bias), rows, launches)
        };
        let base = at("da", false, 2, 1);
        assert_eq!(
            base,
            TokenCount {
                groups: 20,
                dispatches: 22
            }
        );
        // Two more rows: the delta out leaves the M-row entry's count (a
        // dispatch a row again, one more than the entry's one) and the head's
        // picks a dispatch a row: 3 + 2 = 5.
        assert_eq!(
            at("da", false, 4, 1),
            TokenCount {
                groups: base.groups,
                dispatches: base.dispatches + 5
            }
        );
        // A second launch of the flash.
        assert_eq!(
            at("da", false, 2, 2),
            TokenCount {
                groups: base.groups + 2,
                dispatches: base.dispatches + 2
            }
        );
        // The attention layer's biases.
        assert_eq!(
            at("da", true, 2, 1),
            TokenCount {
                groups: base.groups + 1,
                dispatches: base.dispatches + 1
            }
        );
    }

    /// A routed FFN is refused by name.
    #[test]
    #[should_panic(expected = "a routed FFN is not served")]
    fn a_verify_count_refuses_a_routed_ffn() {
        let _ = verify_count(&[routed_delta(Projection::ONE)], 2, 1);
    }

    /// A short conv layer is refused by name.
    #[test]
    #[should_panic(expected = "a short conv layer is not served")]
    fn a_verify_count_refuses_a_short_conv_layer() {
        let _ = verify_count(&lfm3_target()[..1], 2, 1);
    }

    /// A sandwich residual is refused by name.
    #[test]
    #[should_panic(expected = "a sandwich residual is not served")]
    fn a_verify_count_refuses_a_sandwich() {
        let layer = Layer::Attention {
            v: Projection::ONE,
            bias: false,
            ffn: FfnKind::Dense,
            residual: Residual::Sandwich,
        };
        let _ = verify_count(&[layer], 2, 1);
    }

    #[test]
    fn past_the_waterline_the_count_does_not_grow() {
        for layers in [fixture().as_slice(), target().as_slice()] {
            let within = token_count(layers, LiveKeys::Within64, Ticket::Folded);
            let past = token_count(layers, LiveKeys::Past64, Ticket::Folded);
            assert_eq!(past, within);
        }
    }

    #[test]
    fn no_layers_is_the_embed_and_the_head() {
        assert_eq!(
            token_count(&[], LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 3,
                dispatches: 3
            }
        );
    }

    #[test]
    fn kind_names_are_distinct() {
        for (a, &kind) in KINDS.iter().enumerate() {
            let name = kind.name();
            assert!(name.split_ascii_whitespace().count() == 1 && !name.contains('='));
            for &other in &KINDS[..a] {
                assert_ne!(name, other.name());
            }
        }
    }
}
