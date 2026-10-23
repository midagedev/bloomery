//! The groups and the dispatches one qwen35moe token encodes.
//!
//! The caller passes the layers it has. An empty list is the embed and the
//! head, not a hidden default trunk.
//!
//! The token encoder is the caller: the e2e gate holds what one token
//! encodes to this count.

/// One layer of the token.
///
/// `qkv` and `v` are the projection that is P1. [`Projection::Q6`] is that
/// projection splitting into two dispatches in one group (`qkv` on a delta
/// layer, `v` on an attention layer). It is not a quant of the layer's
/// other tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// A gated delta layer. The live-key length does not add a dispatch.
    Delta { qkv: Projection },
    /// A gated attention layer. Live keys past 64 add the memory dispatch.
    Attention { v: Projection },
}

/// Whether P1 is one dispatch or two in one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// P1 is one dispatch.
    Q4,
    /// P1 is two dispatches in one group.
    Q6,
}

/// Whether the live keys are past 64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveKeys {
    /// At most 64. No layer grows.
    Within64,
    /// Past 64. Each attention layer adds the memory dispatch.
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

use crate::attention_geom as launch_geom;

/// The groups and the dispatches one token encodes for `layers`.
///
/// The embed and the head are the 2. Each layer adds six groups: P1, the
/// mixer, its out, the router, the gate-up and the down. Those six already
/// include the flash, and the FFN norm, which the router and the gate-up each
/// apply as they load the residual. A Q6 projection adds one dispatch. Live keys past one
/// segment add the merge on each attention layer, the second dispatch
/// [`launch_geom::attention_dispatches`] reports. An unfolded ticket adds one
/// group for each layer's router and one for the head. A count that does not
/// fit `u32` panics by name.
pub fn token_count(layers: &[Layer], live: LiveKeys, ticket: Ticket) -> TokenCount {
    let mut n = 0u32;
    let mut n_q6 = 0u32;
    let mut n_attn = 0u32;
    for layer in layers {
        n = fit("layer", n.checked_add(1));
        let q6 = match layer {
            Layer::Delta { qkv } => matches!(qkv, Projection::Q6),
            Layer::Attention { v } => matches!(v, Projection::Q6),
        };
        if q6 {
            n_q6 = fit("Q6", n_q6.checked_add(1));
        }
        if matches!(layer, Layer::Attention { .. }) {
            n_attn = fit("attention", n_attn.checked_add(1));
        }
    }
    let mut groups = fit("group", 2u32.checked_add(fit("group", 6u32.checked_mul(n))));
    let mut dispatches = fit("dispatch", groups.checked_add(n_q6));
    if matches!(live, LiveKeys::Past64) {
        let edge = u32::try_from(launch_geom::SEGMENT_KEYS).expect("the segment holds a u32 count");
        let one = launch_geom::attention_dispatches(launch_geom::segments_for_keys(edge));
        let two = launch_geom::attention_dispatches(launch_geom::segments_for_keys(
            edge.checked_add(1).expect("the next key count fits u32"),
        ));
        if one != 1 {
            panic!(
                "token count: one segment is one dispatch, and the six groups already include it"
            );
        }
        let merge = fit("dispatch", two.checked_sub(one));
        let extra = fit("attention", n_attn.checked_mul(merge));
        groups = fit("group", groups.checked_add(extra));
        dispatches = fit("dispatch", dispatches.checked_add(extra));
    }
    if matches!(ticket, Ticket::Unfolded) {
        let extra = fit("group", n.checked_add(1));
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
/// one kind a mixer; the attention past one segment of keys is two kinds,
/// the flash and the merge; the gate-up group and the down group each carry
/// the shared expert's part with the routed picks' — the pair and the fold
/// run them together, so no kind of the shared expert stands alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupKind {
    /// The token's embedding row.
    Embed,
    /// An attention layer's P1: the norm on load and its projections.
    AttnP1,
    /// The attention core: the flash over the live keys.
    AttnFlash,
    /// The attention merge: the flash's partials folded, past one segment.
    AttnMerge,
    /// The attention's output projection back to the residual.
    AttnOut,
    /// A delta layer's P1: the norm on load and its projections.
    DeltaP1,
    /// The gated delta rule's step.
    DeltaCore,
    /// The delta step's head-norm projection back to the residual.
    DeltaOut,
    /// The router: the routed logits, the picks and their weights.
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
            GroupKind::AttnMerge => "attn_merge",
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
pub const KINDS: [GroupKind; 12] = [
    GroupKind::Embed,
    GroupKind::AttnP1,
    GroupKind::AttnFlash,
    GroupKind::AttnMerge,
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
    use super::launch_geom;
    use super::*;

    fn fixture() -> [Layer; 4] {
        [
            Layer::Delta {
                qkv: Projection::Q4,
            },
            Layer::Attention { v: Projection::Q4 },
            Layer::Delta {
                qkv: Projection::Q6,
            },
            Layer::Attention { v: Projection::Q6 },
        ]
    }

    /// 16 delta Q4, 14 delta Q6, 4 attention Q4, 6 attention Q6.
    fn target() -> [Layer; 40] {
        let mut layers = [Layer::Delta {
            qkv: Projection::Q4,
        }; 40];
        for layer in &mut layers[16..30] {
            *layer = Layer::Delta {
                qkv: Projection::Q6,
            };
        }
        for layer in &mut layers[30..34] {
            *layer = Layer::Attention { v: Projection::Q4 };
        }
        for layer in &mut layers[34..40] {
            *layer = Layer::Attention { v: Projection::Q6 };
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
        assert_eq!(
            token_count(&fixture(), LiveKeys::Past64, Ticket::Folded),
            TokenCount {
                groups: 28,
                dispatches: 30
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
        assert_eq!(
            token_count(&target(), LiveKeys::Past64, Ticket::Folded),
            TokenCount {
                groups: 252,
                dispatches: 272
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
                groups: 293,
                dispatches: 313
            }
        );
    }

    #[test]
    fn attention_count_matches_the_launch() {
        let edge = u32::try_from(launch_geom::SEGMENT_KEYS).expect("the segment holds a u32 count");
        let one = launch_geom::attention_dispatches(launch_geom::segments_for_keys(edge));
        let two = launch_geom::attention_dispatches(launch_geom::segments_for_keys(edge + 1));
        assert_eq!((one, two), (1, 2));
        #[cfg(target_os = "macos")]
        {
            use crate::kernels::{attention_dispatches, segments_for_keys};
            assert_eq!(attention_dispatches(segments_for_keys(edge)), one);
            assert_eq!(attention_dispatches(segments_for_keys(edge + 1)), two);
        }
        let extra = two - one;
        for (layers, n_attn) in [(fixture().as_slice(), 2u32), (target().as_slice(), 10u32)] {
            let within = token_count(layers, LiveKeys::Within64, Ticket::Folded);
            let past = token_count(layers, LiveKeys::Past64, Ticket::Folded);
            assert_eq!(past.groups - within.groups, n_attn * extra);
            assert_eq!(past.dispatches - within.dispatches, n_attn * extra);
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
