//! What one resident sequence of a load holds, and the rules that count
//! `slots` of them: the bytes a multi-slot plan, a park budget and the
//! context split read ([`SeqTerms`]), the per-layer view a placement plan
//! counts `slots` sequences' stores by ([`SlotsOf`]), and the context each
//! of `slots` slots gets from a total ([`split_ctx`]). A load fills
//! [`SeqTerms`] from its own owners of each term; nothing here knows an
//! architecture.

use std::num::NonZeroU64;

use bloomery_levers::KvHost;
use runtime::stores::{Class, PosLayout};

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
            rooms: self.layers.kv.rooms().map(|r| r.times(slots)),
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
#[derive(Clone)]
pub struct SlotsOf<'a> {
    kv: &'a dyn KvBytes,
    slots: u64,
    /// The load's rooms held by every slot ([`PosLayout::times`]).
    rooms: Option<PosLayout>,
}

impl KvBytes for SlotsOf<'_> {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.kv
            .layer_bytes(layer, ctx_max)
            .saturating_mul(self.slots)
    }

    fn rooms(&self) -> Option<&PosLayout> {
        self.rooms.as_ref()
    }

    fn shadow_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.kv
            .shadow_bytes(layer, ctx_max)
            .saturating_mul(self.slots)
    }

    /// One copy for every slot: the slots read one stage's rows.
    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        self.kv.stage_position_bytes(ctx_max)
    }
}

/// The host bytes a host-backed K/V room holds at `positions` positions under the planes' backing
/// `host` (`BLOOMERY_KV_HOST`), `rooms` already times the slots ([`PosLayout::times`]): every
/// `Select` buffer's run rounded up to whole pages of `page`, the page a host-capable room takes.
/// `all` holds them whole; `spill` holds what the card's floor (`floor`, the positions the plan
/// reads the rooms at) does not, the pages past it, and nothing where the plan has no floor (the
/// card holds the cap); `off` holds none. What a plan counts against its host room
/// ([`Plan::kv_host_planes`]) so a session's growth meets no host `NoRoom`; one function for every
/// planner.
///
/// [`Plan::kv_host_planes`]: crate::placement::Plan::kv_host_planes
#[must_use]
pub fn kv_host_room(
    rooms: &PosLayout,
    positions: u64,
    floor: Option<u64>,
    host: KvHost,
    page: u64,
) -> u64 {
    assert!(page > 0, "the page a host-capable room takes is positive");
    let on_host = |class: Class| (class == Class::Host).then_some(page);
    let held = |at: u64| {
        (0..rooms.layers()).fold(0u64, |sum, layer| {
            sum.saturating_add(rooms.paged(layer, at, on_host))
        })
    };
    match host {
        KvHost::Off => 0,
        KvHost::All => held(positions),
        KvHost::Spill => floor.map_or(0, |f| {
            held(positions).saturating_sub(held(f.min(positions)))
        }),
    }
}

/// The card bytes `layer`'s rooms (times the slots) hold beside their exact rows at `positions`
/// positions: every buffer rounded up to whole pages. A card-class buffer rounds at `card_page`;
/// a host-class one at the card's page under `off`, at the page a host-capable room takes
/// (`host_page`) under `spill`, and holds no card bytes under `all`. `None` past u64, or under
/// `spill`/`all` with no `host_page`.
#[must_use]
pub fn layer_padding(
    rooms: &PosLayout,
    layer: usize,
    positions: u64,
    (card_page, host_page): (u64, Option<u64>),
    host: KvHost,
) -> Option<u64> {
    assert!(
        card_page > 0,
        "the device allocation granularity is positive"
    );
    if host != KvHost::Off && host_page.is_none() {
        return None;
    }
    rooms.padding(layer, positions, |class| match class {
        Class::Card => Some(card_page),
        Class::Host => match host {
            KvHost::Off => Some(card_page),
            KvHost::Spill => host_page,
            KvHost::All => None,
        },
    })
}

/// The device's spare pages a card holding a room keeps beside its rooms: the card rooms' shared
/// spare page and, where `spill` or `all` backs a room on the host at another page, that room's
/// own. `None` past u64.
#[must_use]
pub fn room_spare((card_page, host_page): (u64, Option<u64>), host: KvHost) -> Option<u64> {
    match host_page {
        Some(at) if host != KvHost::Off && at != card_page => card_page.checked_add(at),
        _ => Some(card_page),
    }
}

/// The card bytes `rooms` (times the slots) hold beside their exact rows at `positions`
/// positions on one card holding every layer: each layer's rounding ([`layer_padding`]) and the
/// spare pages ([`room_spare`]) once. `None` as they are.
#[must_use]
pub fn room_padding(
    rooms: &PosLayout,
    positions: u64,
    pages: (u64, Option<u64>),
    host: KvHost,
) -> Option<u64> {
    (0..rooms.layers()).try_fold(room_spare(pages, host)?, |sum, layer| {
        sum.checked_add(layer_padding(rooms, layer, positions, pages, host)?)
    })
}

/// The card bytes one card holding `layers` of `rooms` holds beside their exact rows at
/// `positions` positions over `slots` slots: each of its layers' rounding ([`layer_padding`]) on
/// every slot, and the spare pages ([`room_spare`]) once. `beside` is the per-slot rounding of a
/// store outside the layout that this card also holds (a draft's planes), `None` where it holds
/// none. A card with no room layer and no such store pays nothing. `None` past u64, or as
/// [`layer_padding`].
#[must_use]
pub fn card_padding(
    rooms: &PosLayout,
    layers: std::ops::Range<usize>,
    (positions, slots): (u64, u64),
    pages: (u64, Option<u64>),
    host: KvHost,
    beside: Option<u64>,
) -> Option<u64> {
    let mut held = layers.start..layers.end.min(rooms.layers());
    let holds = held.clone().any(|l| !rooms.buffers(l).is_empty());
    if !holds && beside.is_none() {
        return Some(0);
    }
    let per_slot = held.try_fold(beside.unwrap_or(0), |sum, layer| {
        sum.checked_add(layer_padding(rooms, layer, positions, pages, host)?)
    })?;
    per_slot
        .checked_mul(slots)?
        .checked_add(room_spare(pages, host)?)
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

    use std::sync::LazyLock;

    use runtime::stores::{Access, PosBuf, PosLayout};

    use bloomery_levers::KvHost;

    use super::{SeqTerms, SplitError, Stores, kv_host_room, room_padding, split_ctx};
    use crate::placement::KvBytes;

    /// Layer 1's planes of the toy: 4 B a position, selected.
    static TOY_ROOMS: LazyLock<PosLayout> = LazyLock::new(|| {
        PosLayout::new(vec![
            vec![],
            vec![PosBuf {
                what: "planes",
                row: 4,
                per: 1,
                runs: 1,
                access: Access::Select { rows: 1 },
            }],
        ])
    });

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

        fn rooms(&self) -> Option<&PosLayout> {
            Some(&TOY_ROOMS)
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

    /// (t2) The plan's view: every layer's bytes, planes and shadows times
    /// `slots`, whatever the layer.
    #[test]
    fn slots_of_multiplies_every_layer() {
        let t = terms();
        for n in [1, 2, 4] {
            let view = t.slots_of(n);
            for layer in 0..3 {
                for c in [1, 4096] {
                    assert_eq!(view.layer_bytes(layer, c), n * Toy.layer_bytes(layer, c));
                    let rooms = view.rooms().expect("the toy's rooms times the slots");
                    assert_eq!(rooms.host_able(layer, c), n * TOY_ROOMS.host_able(layer, c));
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

    /// Two latent layers of K3.1's shape: a cached row of 1,152 B a position (`Select`, host-able)
    /// and an index key row of 256 B (`Scan`, card-only).
    fn latent() -> PosLayout {
        let layer = || {
            vec![
                PosBuf {
                    what: "cache",
                    row: 1_152,
                    per: 1,
                    runs: 1,
                    access: Access::Select { rows: 2_048 },
                },
                PosBuf {
                    what: "keys",
                    row: 256,
                    per: 1,
                    runs: 1,
                    access: Access::Scan,
                },
            ]
        };
        PosLayout::new(vec![layer(), layer()])
    }

    const MIB: u64 = 1 << 20;

    /// The host bytes of the host-able run [derived by hand, page 2 MiB]: at 262,144 positions
    /// the cached run is 301,989,888 B = 144 whole pages; at a floor of 8,192 it is 9,437,184 B,
    /// 4.5 pages, held as 5 = 10,485,760 B. `spill` holds the difference, 291,504,128 B a layer;
    /// `all` the whole run; `off` and a spill with no floor hold none; the key run never counts.
    /// Mutants: the floor's pages not taken off, the key run counted, `all` counted as `spill`.
    #[test]
    fn the_host_room_counts_the_pages_past_the_floor() {
        let rooms = latent();
        let room = |floor, host| kv_host_room(&rooms, 262_144, floor, host, 2 * MIB);
        assert_eq!(room(Some(8_192), KvHost::Spill), 2 * 291_504_128);
        assert_eq!(room(None, KvHost::All), 2 * 301_989_888);
        assert_eq!(
            room(None, KvHost::Spill),
            0,
            "no floor, the card holds the cap"
        );
        assert_eq!(room(Some(8_192), KvHost::Off), 0);
        assert_eq!(
            room(Some(262_144), KvHost::Spill),
            0,
            "a floor at the cap holds all on the card"
        );
        assert_eq!(
            kv_host_room(
                &rooms.times(3),
                262_144,
                Some(8_192),
                KvHost::Spill,
                2 * MIB
            ),
            3 * 2 * 291_504_128,
            "every slot's rooms"
        );
    }

    /// The page rounding of the rooms at 100,000 positions [derived by hand]: the cached run is
    /// 115,200,000 B, 54.9 pages of 2 MiB (143,360 B past its rows), the key run 25,600,000 B,
    /// 12.2 pages (1,662,976 B past); two layers and the device's spare page, 5,709,824 B under
    /// `off`. Under `spill` at a host page of 4 MiB the cached run rounds there (2,240,512 B past)
    /// and the device holds the host room's spare page beside the card's: 14,098,432 B; under
    /// `all` the cached run holds no card page: 9,617,408 B. A host backing with no page is
    /// `None`. Mutants: the spare page dropped, the host class rounded at the card's page under
    /// `spill`, `all` still padding the cached run.
    #[test]
    fn the_rooms_round_their_pages_beside_the_rows() {
        let rooms = latent();
        let pad = |host, host_page| room_padding(&rooms, 100_000, (2 * MIB, host_page), host);
        assert_eq!(pad(KvHost::Off, None), Some(5_709_824));
        assert_eq!(pad(KvHost::Spill, Some(4 * MIB)), Some(14_098_432));
        assert_eq!(pad(KvHost::All, Some(4 * MIB)), Some(9_617_408));
        assert_eq!(pad(KvHost::Spill, None), None);
        assert_eq!(
            room_padding(&rooms.times(2), 100_000, (2 * MIB, None), KvHost::Off),
            Some(2 * 2 * (143_360 + 1_662_976) + 2 * MIB),
            "the buffers round per slot, the spare page once"
        );
    }
}
