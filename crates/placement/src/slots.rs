//! What one resident sequence of a load holds, and the rules that count
//! `slots` of them: the bytes a multi-slot plan, a park budget and the
//! context split read ([`SeqTerms`]), the per-layer view a placement plan
//! counts `slots` sequences' stores by ([`SlotsOf`]), and the context each
//! of `slots` slots gets from a total ([`split_ctx`]). A load fills
//! [`SeqTerms`] from its own owners of each term; nothing here knows an
//! architecture.

use std::num::NonZeroU64;

use crate::placement::KvBytes;

/// One sequence's per-layer stores on a load: the bytes each layer holds at
/// a context, and how many layers the load runs, `0..count`. The count is a
/// field, not a [`SeqTerms::bytes`] parameter, because a load and its draft
/// each have their own, and a [`KvBytes`] answers past its last layer by its
/// own rule (0 on one, a fixed window's ring on another): only the load
/// knows where its sum stops.
#[derive(Clone, Copy)]
pub struct Stores<'a> {
    /// The bytes each layer holds at a context.
    pub kv: &'a dyn KvBytes,
    /// The layers the load runs.
    pub count: usize,
}

impl Stores<'_> {
    /// Every layer's bytes at `positions`; `None` past u64.
    fn bytes(&self, positions: u64) -> Option<u64> {
        (0..self.count).try_fold(0u64, |sum, layer| {
            sum.checked_add(self.kv.layer_bytes(layer, positions))
        })
    }
}

/// What one resident sequence of a load holds on its card: the terms a
/// multi-slot plan, a park budget and the context split read.
#[derive(Clone, Copy)]
pub struct SeqTerms<'a> {
    /// The sequence's per-layer stores: its positional rows and each layer's
    /// fixed state.
    pub layers: Stores<'a>,
    /// The draft's per-layer stores of the sequence, on a load that drafts;
    /// `None` on one that does not.
    pub draft: Option<Stores<'a>>,
    /// Card bytes the sequence holds beside its stores, whatever its length:
    /// buffers the load holds once as its own for whichever sequence is live,
    /// apart from a plan's kv class, and every other sequence holds a copy
    /// of.
    pub beside: u64,
}

impl<'a> SeqTerms<'a> {
    /// Card bytes of `slots` sequences of `positions` positions each: every
    /// term — each layer's stores, the draft's, the bytes beside them —
    /// times `slots`; `u64::MAX` past u64 bytes, which no card holds. Exact,
    /// before the card allocator rounds each allocation: a [`KvBytes`] gives
    /// a layer's bytes, not its allocations, and a plan's cache term counts
    /// them unrounded too. Ring shadows are not in it: they are host bytes
    /// (page-locked, [`KvBytes::shadow_bytes`]), which a plan counts on the
    /// host through [`SeqTerms::slots_of`]. These are the card's bytes, not a
    /// park's: a park budget counts a parked sequence in the form the park
    /// holds on the host, which need not keep a fixed state as the card does.
    /// Nor are they a plan's kv class, which holds one `beside` less
    /// ([`SeqTerms::plan_kv`]): this, or `slots` times one sequence's bytes,
    /// set against a plan's kv class is off by exactly one `beside`.
    #[must_use]
    pub fn bytes(&self, positions: u64, slots: u64) -> u64 {
        self.stores(positions)
            .and_then(|stores| stores.checked_add(self.beside))
            .and_then(|one| one.checked_mul(slots))
            .unwrap_or(u64::MAX)
    }

    /// The card bytes a placement plan of `slots` sequences of `positions`
    /// positions holds in its kv classes, the load's and its draft's summed:
    /// every sequence's stores, and the bytes beside them a plan adds
    /// ([`SeqTerms::plan_beside`]). [`SeqTerms::bytes`] less one `beside`; 0
    /// at no slot; `u64::MAX` past u64 bytes.
    #[must_use]
    pub fn plan_kv(&self, positions: u64, slots: u64) -> u64 {
        self.stores(positions)
            .and_then(|stores| stores.checked_mul(slots))
            .zip(self.past_live(slots))
            .and_then(|(stores, beside)| stores.checked_add(beside))
            .unwrap_or(u64::MAX)
    }

    /// The bytes beside the stores a placement plan of `slots` sequences
    /// adds to its kv class: every sequence's but the live one's, which the
    /// load holds as its own; 0 at one slot or none; `u64::MAX` past u64
    /// bytes.
    #[must_use]
    pub fn plan_beside(&self, slots: u64) -> u64 {
        self.past_live(slots).unwrap_or(u64::MAX)
    }

    /// The per-layer view a placement plan counts `slots` sequences' stores
    /// by: [`SeqTerms::layers`] times `slots`.
    #[must_use]
    pub fn slots_of(&self, slots: u64) -> SlotsOf<'a> {
        SlotsOf {
            kv: self.layers.kv,
            slots,
        }
    }

    /// One sequence's stores at `positions`, the draft's with them; `None`
    /// past u64.
    fn stores(&self, positions: u64) -> Option<u64> {
        let draft = self.draft.map_or(Some(0), |d| d.bytes(positions))?;
        self.layers.bytes(positions)?.checked_add(draft)
    }

    /// `beside` of every sequence but the live one; `None` past u64.
    fn past_live(&self, slots: u64) -> Option<u64> {
        self.beside.checked_mul(slots.saturating_sub(1))
    }
}

/// A [`KvBytes`] times `slots` ([`SeqTerms::slots_of`]): each layer's cache
/// and its ring shadows held `slots` times over. A product past u64 bytes is
/// `u64::MAX`, which no card or host holds.
#[derive(Clone, Copy)]
pub struct SlotsOf<'a> {
    kv: &'a dyn KvBytes,
    slots: u64,
}

impl KvBytes for SlotsOf<'_> {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.kv
            .layer_bytes(layer, ctx_max)
            .saturating_mul(self.slots)
    }

    fn shadow_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.kv
            .shadow_bytes(layer, ctx_max)
            .saturating_mul(self.slots)
    }
}

/// Why `total` positions do not split across `slots` slots ([`split_ctx`]),
/// with every term of the split, so a caller names it in its own words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SplitError {
    /// No slot to split the positions across.
    #[error("{total} positions split across no slot: a split serves at least one")]
    NoSlots { total: u64 },
    /// A slot's share below the least a slot loads at.
    #[error(
        "{total} positions across {slots} slots give a slot {split}, below the floor of {floor}"
    )]
    BelowFloor {
        total: u64,
        slots: u64,
        split: u64,
        floor: u64,
    },
}

/// The positions each slot gets when `slots` slots split `total`:
/// ⌊total / slots⌋, so the slots together never hold more than `total`.
/// Refused by name at no slot and below `floor`, the least a slot loads at.
/// What the floor stands for is the caller's, so the refusal names its value,
/// not its reason; it is not zero, because a slot of no position is never a
/// split.
pub fn split_ctx(total: u64, slots: u64, floor: NonZeroU64) -> Result<u64, SplitError> {
    let Some(split) = total.checked_div(slots) else {
        return Err(SplitError::NoSlots { total });
    };
    if split < floor.get() {
        return Err(SplitError::BelowFloor {
            total,
            slots,
            split,
            floor: floor.get(),
        });
    }
    Ok(split)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::{SeqTerms, SplitError, Stores, split_ctx};
    use crate::placement::KvBytes;

    /// Layer 0 a fixed state of 1,000 B; layer 1 7 B a position, with a ring
    /// shadow of 3 B a position; any layer past those 99 B, which a sum over
    /// the load's two layers must not read.
    struct Toy;

    impl KvBytes for Toy {
        fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
            match layer {
                0 => 1000,
                1 => 7 * ctx_max,
                _ => 99,
            }
        }

        fn shadow_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
            if layer == 1 { 3 * ctx_max } else { 0 }
        }
    }

    /// A draft of one layer of 5 B a position.
    struct Draft;

    impl KvBytes for Draft {
        fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
            if layer == 0 { 5 * ctx_max } else { 0 }
        }
    }

    fn terms() -> SeqTerms<'static> {
        SeqTerms {
            layers: Stores { kv: &Toy, count: 2 },
            draft: Some(Stores {
                kv: &Draft,
                count: 1,
            }),
            beside: 11,
        }
    }

    /// (t1) One sequence is its two layers, the draft's layer and the bytes
    /// beside them, and `slots` sequences are every term times `slots`,
    /// the bytes beside included; past u64 the sum is `u64::MAX`.
    #[test]
    fn bytes_count_every_term_per_slot() {
        let t = terms();
        let plain = SeqTerms { draft: None, ..t };
        for c in [1, 4096, 4097] {
            assert_eq!(t.bytes(c, 1), 1000 + 7 * c + 5 * c + 11, "ctx {c}");
            assert_eq!(plain.bytes(c, 1), 1000 + 7 * c + 11, "ctx {c}");
            for n in [2, 3, 8] {
                assert_eq!(t.bytes(c, n), n * t.bytes(c, 1), "ctx {c}, {n} slots");
                assert_eq!(plain.bytes(c, n), n * plain.bytes(c, 1), "ctx {c}");
            }
        }
        assert_eq!(t.bytes(4096, u64::MAX), u64::MAX);
    }

    /// (t1b) A plan's kv classes hold every sequence's stores and the bytes
    /// beside them of all but the live one: one `beside` less than
    /// [`SeqTerms::bytes`] at any count of slots, none at no slot.
    #[test]
    fn plan_kv_is_one_beside_less() {
        let t = terms();
        for c in [1, 4096] {
            for n in [1, 2, 3, 8] {
                assert_eq!(
                    t.plan_kv(c, n) + t.beside,
                    t.bytes(c, n),
                    "ctx {c}, {n} slots"
                );
                assert_eq!(t.plan_beside(n), (n - 1) * 11, "{n} slots");
            }
        }
        assert_eq!((t.plan_kv(4096, 0), t.plan_beside(0)), (0, 0));
        assert_eq!(t.plan_kv(4096, u64::MAX), u64::MAX);
        assert_eq!(t.plan_beside(u64::MAX), u64::MAX);
    }

    /// (t2) The plan's view: every layer's bytes and shadows times `slots`,
    /// whatever the layer.
    #[test]
    fn slots_of_multiplies_every_layer() {
        let t = terms();
        for n in [1, 2, 4] {
            let view = t.slots_of(n);
            for layer in 0..3 {
                for c in [1, 4096] {
                    assert_eq!(view.layer_bytes(layer, c), n * Toy.layer_bytes(layer, c));
                    assert_eq!(view.shadow_bytes(layer, c), n * Toy.shadow_bytes(layer, c));
                }
            }
        }
        assert_eq!(t.slots_of(u64::MAX).layer_bytes(0, 1), u64::MAX);
    }

    /// (t3) The floor of the share, exact or with a remainder; no slot, and a
    /// share below the floor (none at all among them), refused by name with
    /// every term.
    #[test]
    fn split_ctx_floors_and_refuses() {
        let floor = |f: u64| NonZeroU64::new(f).expect("a floor");
        assert_eq!(split_ctx(12, 3, floor(1)), Ok(4));
        assert_eq!(split_ctx(14, 3, floor(1)), Ok(4));
        assert_eq!(split_ctx(12, 3, floor(4)), Ok(4));
        assert_eq!(split_ctx(12, 1, floor(12)), Ok(12));
        assert_eq!(
            split_ctx(12, 0, floor(1)),
            Err(SplitError::NoSlots { total: 12 })
        );
        let below = split_ctx(14, 3, floor(5));
        assert_eq!(
            below,
            Err(SplitError::BelowFloor {
                total: 14,
                slots: 3,
                split: 4,
                floor: 5
            })
        );
        assert_eq!(
            below.map_err(|e| e.to_string()),
            Err("14 positions across 3 slots give a slot 4, below the floor of 5".to_string())
        );
        assert_eq!(
            split_ctx(2, 3, floor(1)),
            Err(SplitError::BelowFloor {
                total: 2,
                slots: 3,
                split: 0,
                floor: 1
            })
        );
    }
}
