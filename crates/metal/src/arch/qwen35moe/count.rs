//! The groups and the dispatches one qwen35moe token encodes.
//!
//! The caller passes the layers it has. An empty list is the embed and the
//! head, not a hidden default trunk.

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

/// One dispatch and one group: the embed, a folded head, a folded router,
/// and each step that does not split.
const ONE: TokenCount = TokenCount {
    groups: 1,
    dispatches: 1,
};

/// Two dispatches in one group: the Q6 projection beside the Q4 one.
const SPLIT: TokenCount = TokenCount {
    groups: 1,
    dispatches: 2,
};

/// The groups and the dispatches one token encodes for `layers`.
///
/// `live` adds the memory dispatch on attention layers when the keys are
/// past 64. A folded `ticket` keeps the router and the head one dispatch
/// each. A count that does not fit `u32` panics by name.
#[must_use]
pub(crate) fn token_count(layers: &[Layer], live: LiveKeys, ticket: Ticket) -> TokenCount {
    let mut total = add(embed(), head(ticket));
    for layer in layers {
        total = add(total, p1(p1_of(*layer)));
        total = add(total, after_p1(ticket));
        if let Some(memory) = memory(*layer, live) {
            total = add(total, memory);
        }
    }
    total
}

fn p1_of(layer: Layer) -> Projection {
    match layer {
        Layer::Delta { qkv } => qkv,
        Layer::Attention { v } => v,
    }
}

fn p1(projection: Projection) -> TokenCount {
    match projection {
        Projection::Q4 => ONE,
        Projection::Q6 => SPLIT,
    }
}

/// The steps after P1: the mixer (D' on a delta layer, F' on an attention
/// layer), its out (O' or O), the router, the gate-up and the down.
///
/// An unfolded ticket makes the router two dispatches and two groups.
fn after_p1(ticket: Ticket) -> TokenCount {
    let mixer = ONE;
    let out = ONE;
    let router = match ticket {
        Ticket::Folded => ONE,
        Ticket::Unfolded => add(ONE, ONE),
    };
    let gate_up = ONE;
    let down = ONE;
    let mut total = mixer;
    for step in [out, router, gate_up, down] {
        total = add(total, step);
    }
    total
}

/// The memory dispatch. Attention only, and only once the live keys are
/// past 64. A delta layer does not grow.
fn memory(layer: Layer, live: LiveKeys) -> Option<TokenCount> {
    if matches!(layer, Layer::Attention { .. }) && matches!(live, LiveKeys::Past64) {
        Some(ONE)
    } else {
        None
    }
}

fn embed() -> TokenCount {
    ONE
}

fn head(ticket: Ticket) -> TokenCount {
    match ticket {
        Ticket::Folded => ONE,
        Ticket::Unfolded => add(ONE, ONE),
    }
}

fn add(total: TokenCount, next: TokenCount) -> TokenCount {
    TokenCount {
        groups: match total.groups.checked_add(next.groups) {
            Some(n) => n,
            None => panic!("token count: the group count does not fit u32"),
        },
        dispatches: match total.dispatches.checked_add(next.dispatches) {
            Some(n) => n,
            None => panic!("token count: the dispatch count does not fit u32"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PIN(2026-10-22): design-m22c-a section 3.4. Groups, then dispatches.

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
