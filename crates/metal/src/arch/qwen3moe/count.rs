//! The groups and the dispatches one qwen35moe token encodes.
//!
//! The caller passes the layers it has. An empty list is the embed and the
//! head's two groups, not a hidden default trunk.
//!
//! The token encoder is the caller: the e2e gate holds what one token
//! encodes to this count.

use core::num::NonZeroU32;

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
        ffn: FfnKind,
        residual: Residual,
    },
    /// A gated attention layer. Live keys past 64 add the memory dispatch.
    Attention {
        v: Projection,
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

    /// The groups the layer encodes: its mixer's three, its FFN's two and a
    /// router's ahead of them, and a sandwich's two joins or a plain
    /// residual's two hand-ons.
    #[must_use]
    pub const fn groups(&self) -> u32 {
        let router = match self.ffn() {
            FfnKind::Routed => 1,
            FfnKind::Dense => 0,
        };
        let joins = match self.residual() {
            Residual::Plain => 0,
            Residual::PlainHandOn => HANDON_GROUPS,
            Residual::Sandwich => JOIN_GROUPS,
        };
        MIXER_GROUPS + FFN_GROUPS + router + joins
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
/// seven. Those already include the flash — the attention's fold
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
pub const KINDS: [GroupKind; 17] = [
    GroupKind::Embed,
    GroupKind::AttnP1,
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
            ffn: FfnKind::Routed,
            residual: Residual::Plain,
        }
    }

    fn routed_attention(v: Projection) -> Layer {
        Layer::Attention {
            v,
            ffn: FfnKind::Routed,
            residual: Residual::Plain,
        }
    }

    /// `layers` with a dense FFN: no layer has a router.
    fn dense(layers: &[Layer]) -> Vec<Layer> {
        layers
            .iter()
            .map(|l| match *l {
                Layer::Delta { qkv, residual, .. } => Layer::Delta {
                    qkv,
                    ffn: FfnKind::Dense,
                    residual,
                },
                Layer::Attention { v, residual, .. } => Layer::Attention {
                    v,
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
                Layer::Attention { v, ffn, .. } => Layer::Attention {
                    v,
                    ffn,
                    residual: Residual::Plain,
                },
                Layer::Delta { qkv, ffn, .. } => Layer::Delta {
                    qkv,
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
                Layer::Delta { qkv, ffn, .. } => Layer::Delta {
                    qkv,
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
