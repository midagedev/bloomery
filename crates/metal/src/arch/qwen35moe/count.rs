//! The groups and the dispatches one qwen35moe token encodes.
//!
//! The caller passes the layers it has. An empty list is the embed and the
//! head, not a hidden default trunk.
//!
//! The token encoder is the caller. Until that lands, the tests are the use.

#![allow(
    dead_code,
    reason = "M2.3a's token encoder is the consumer; the tests pin the count until it lands"
)]

/// One layer of the token.
///
/// `qkv` and `v` are the projection that is P1. [`Projection::Q6`] is that
/// projection splitting into two dispatches in one group (`qkv` on a delta
/// layer, `v` on an attention layer). It is not a quant of the layer's
/// other tensors.
#[derive(Clone, Copy)]
pub(crate) enum Layer {
    /// A gated delta layer. The live-key length does not add a dispatch.
    Delta { qkv: Projection },
    /// A gated attention layer. Live keys past 64 add the memory dispatch.
    Attention { v: Projection },
}

/// Whether P1 is one dispatch or two in one group.
#[derive(Clone, Copy)]
pub(crate) enum Projection {
    /// P1 is one dispatch.
    Q4,
    /// P1 is two dispatches in one group.
    Q6,
}

/// Whether the live keys are past 64.
#[derive(Clone, Copy)]
pub(crate) enum LiveKeys {
    /// At most 64. No layer grows.
    Within64,
    /// Past 64. Each attention layer adds the memory dispatch.
    Past64,
}

/// Whether the router's ticket and the head's stay one dispatch.
#[derive(Clone, Copy)]
pub(crate) enum Ticket {
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
pub(crate) struct TokenCount {
    pub(crate) groups: u32,
    pub(crate) dispatches: u32,
}

/// The groups and the dispatches one token encodes for `layers`.
///
/// The embed and the head are the 2. Each layer adds six groups: P1, the
/// mixer, its out, the router, the gate-up and the down. A Q6 projection adds
/// one dispatch. Live keys past 64 add the memory group on each attention
/// layer. An unfolded ticket adds one group for each layer's router and one
/// for the head. A count that does not fit `u32` panics by name.
pub(crate) fn token_count(layers: &[Layer], live: LiveKeys, ticket: Ticket) -> TokenCount {
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
        groups = fit("group", groups.checked_add(n_attn));
        dispatches = fit("dispatch", dispatches.checked_add(n_attn));
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

#[cfg(test)]
mod tests {
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
    fn no_layers_is_the_embed_and_the_head() {
        assert_eq!(
            token_count(&[], LiveKeys::Within64, Ticket::Folded),
            TokenCount {
                groups: 2,
                dispatches: 2
            }
        );
    }
}
