//! The groups and the dispatches one qwen35moe token encodes.
//!
//! The caller passes the layers it has. An empty list is the embed and the
//! head, not a hidden default trunk.
//!
//! The token encoder is the caller: the e2e gate holds what one token
//! encodes to this count.

use core::num::NonZeroU32;

/// One layer of the token.
///
/// `qkv` and `v` are the projection that is P1: one dispatch for each weight
/// type its tensors hold, all in one group (`qkv` on a delta layer, `v` on
/// an attention layer). `ffn` is how the layer's FFN picks its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// A gated delta layer. The live-key length does not add a dispatch.
    Delta { qkv: Projection, ffn: FfnKind },
    /// A gated attention layer. Live keys past 64 add the memory dispatch.
    Attention { v: Projection, ffn: FfnKind },
}

/// How a layer's FFN picks its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// The router picks them: a group of its own ahead of the gate-up.
    Routed,
    /// One slot on expert 0 at weight 1: no router, no group.
    Dense,
}

impl Layer {
    /// How the layer's FFN picks its slots.
    #[must_use]
    pub const fn ffn(&self) -> FfnKind {
        match self {
            Layer::Delta { ffn, .. } | Layer::Attention { ffn, .. } => *ffn,
        }
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

/// Whether the router's ticket and the head's stay one dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ticket {
    /// The router and the head are one dispatch and one group each.
    Folded,
    /// Each layer's router, and the head, are two dispatches and two groups.
    /// The count when a fence cannot fold those dispatches. Nothing encodes
    /// this path; the tests keep the number.
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
/// The embed and the head are the 2. Each layer adds six groups: P1, the
/// mixer, its out, the router, the gate-up and the down; a layer of a dense
/// FFN has no router, so five. Those already include the flash — the
/// attention's fold runs inside it, so past the segment rule's waterline the
/// count is the same — and the FFN norm, which the router and the gate-up
/// each apply as they load the residual (a dense layer's gate-up alone). A
/// projection of more than one dispatch adds the rest. An unfolded ticket
/// adds one group for each routed layer's router and one for the head. A
/// count that does not fit `u32` panics by name.
pub fn token_count(layers: &[Layer], _live: LiveKeys, ticket: Ticket) -> TokenCount {
    let mut n = 0u32;
    let mut routers = 0u32;
    let mut n_split = 0u32;
    for layer in layers {
        n = fit("layer", n.checked_add(1));
        let p1 = match layer {
            Layer::Delta { qkv, .. } => qkv,
            Layer::Attention { v, .. } => v,
        };
        n_split = fit("P1", n_split.checked_add(p1.dispatches() - 1));
        if layer.ffn() == FfnKind::Routed {
            routers = fit("router", routers.checked_add(1));
        }
    }
    let per_layer = fit("group", 5u32.checked_mul(n));
    let mut groups = fit(
        "group",
        2u32.checked_add(fit("group", per_layer.checked_add(routers))),
    );
    let mut dispatches = fit("dispatch", groups.checked_add(n_split));
    if matches!(ticket, Ticket::Unfolded) {
        let extra = fit("group", routers.checked_add(1));
        groups = fit("group", groups.checked_add(extra));
        dispatches = fit("dispatch", dispatches.checked_add(extra));
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
/// stands alone.
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
    /// The router: the routed logits, the picks and their weights. A dense
    /// FFN has none.
    Router,
    /// The experts' gate and up, the routed picks' and the shared expert's.
    GateUp,
    /// The experts' down, the routed picks' and the shared expert's, folded
    /// into the residual.
    Down,
    /// The head: the logits and the pick.
    Head,
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
            GroupKind::Router => "router",
            GroupKind::GateUp => "gate_up",
            GroupKind::Down => "down",
            GroupKind::Head => "head",
        }
    }
}

/// Every kind, in the token's walk order: the order a profile's lines print.
pub const KINDS: [GroupKind; 11] = [
    GroupKind::Embed,
    GroupKind::AttnP1,
    GroupKind::AttnFlash,
    GroupKind::AttnOut,
    GroupKind::DeltaP1,
    GroupKind::DeltaCore,
    GroupKind::DeltaOut,
    GroupKind::Router,
    GroupKind::GateUp,
    GroupKind::Down,
    GroupKind::Head,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn routed_delta(qkv: Projection) -> Layer {
        Layer::Delta {
            qkv,
            ffn: FfnKind::Routed,
        }
    }

    fn routed_attention(v: Projection) -> Layer {
        Layer::Attention {
            v,
            ffn: FfnKind::Routed,
        }
    }

    /// `layers` with a dense FFN: no layer has a router.
    fn dense(layers: &[Layer]) -> Vec<Layer> {
        layers
            .iter()
            .map(|l| match *l {
                Layer::Delta { qkv, .. } => Layer::Delta {
                    qkv,
                    ffn: FfnKind::Dense,
                },
                Layer::Attention { v, .. } => Layer::Attention {
                    v,
                    ffn: FfnKind::Dense,
                },
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
                groups: 26,
                dispatches: 28
            }
        );
    }

    #[test]
    fn fixture_folded_past_64() {
        // The flash folds the merge, so past the waterline the count is the
        // within-waterline one: 28/30 before the fold, one merge group and
        // dispatch an attention layer.
        assert_eq!(
            token_count(&fixture(), LiveKeys::Past64, Ticket::Folded),
            token_count(&fixture(), LiveKeys::Within64, Ticket::Folded)
        );
        assert_eq!(
            token_count(&fixture(), LiveKeys::Past64, Ticket::Folded),
            TokenCount {
                groups: 26,
                dispatches: 28
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
                groups: 242,
                dispatches: 262
            }
        );
    }

    #[test]
    fn target_folded_past_64() {
        // 252/272 before the fold: ten attention layers' merge group and
        // dispatch each.
        assert_eq!(
            token_count(&target(), LiveKeys::Past64, Ticket::Folded),
            TokenCount {
                groups: 242,
                dispatches: 262
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
        // 2 + 6·48 groups, and the 24 raised layers' second P1 dispatch.
        assert_eq!(
            token_count(&q30_target(), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 290,
                dispatches: 314
            }
        );
    }

    /// A dense layer has no router group: five groups a layer, the folded
    /// ticket's count one group fewer for each, and the unfolded one's extra
    /// group is the head's alone.
    #[test]
    fn a_dense_layer_has_no_router_group() {
        assert_eq!(
            token_count(&dense(&fixture()), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 22,
                dispatches: 24
            }
        );
        assert_eq!(
            token_count(&dense(&fixture()), LiveKeys::Within64, Ticket::Unfolded),
            TokenCount {
                groups: 23,
                dispatches: 25
            }
        );
        // 2 + 5 · 64 groups for the 27B's layers.
        let layers = vec![routed_delta(Projection::ONE); 64];
        assert_eq!(
            token_count(&dense(&layers), LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 322,
                dispatches: 322
            }
        );
        // One routed layer among dense ones adds its router's group.
        let mut mixed = dense(&fixture());
        mixed[1] = routed_attention(Projection::ONE);
        assert_eq!(
            token_count(&mixed, LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 23,
                dispatches: 25
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
                groups: 2,
                dispatches: 2
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
