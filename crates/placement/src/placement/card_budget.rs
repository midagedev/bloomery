//! A card byte budget: a card of a plan plans as if it had at most this many
//! usable bytes — `min(usable, budget)` — so one large card can stand in for
//! any smaller one. The expert rule then runs on the same usable − KV −
//! context − scratch − margin arithmetic and keeps fewer experts.
//!
//! The budget is one value every card takes, or one per card by this
//! process's CUDA ordinal ([`CardBudget`], the `BLOOMERY_CARD_BUDGET`
//! lever's value): [`of`] is the budget a card plans with, the one reading
//! every term of the plan takes it from. A list names the devices of this
//! process's census, so a plan whose card has no census device, or is a tier
//! node's, is refused under one ([`check`]), and so is a list naming a device
//! the census does not list ([`check_visible`]).
//!
//! A list may give shares instead of bytes (`CUDA0=3x,CUDA1=1x`,
//! [`CardBudget::Shares`]): the plan turns them into one budget per card from
//! its own terms ([`split`]), so the cards hold the routed experts in that
//! proportion and the record states the bytes. The bytes a card keeps free
//! past its context and scratch are its margin, a lever of its own
//! (`BLOOMERY_CARD_MARGIN`, the same spelling): [`keep_free`] puts it on the
//! card's spec, so the plan's `Card::margin_bytes` is the one owner and no
//! margin hides in a budget.
//!
//! A byte value is decimal bytes with optional `_` separators, or a whole
//! number of MiB or GiB with an `M` or `G` suffix (binary units: `38G` is
//! 40,802,189,312 B).

use super::devices::{DeviceId, DeviceInfo, visible};
use super::workstation::CardSpec;
use super::{Card, Machine, PlacementError};

/// Bytes a value names ([`bloomery_levers::parse_bytes`]): the parse the
/// `BLOOMERY_CARD_BUDGET` lever's kind runs, and the fixture builder's byte
/// flags.
pub use bloomery_levers::parse_bytes as parse;

/// Why [`parse`] refused a value: not a byte count, or one past `u64`.
pub use bloomery_levers::BytesError;

/// The budget a plan's cards plan with: none, one for every card, or one per
/// card by CUDA ordinal.
pub use bloomery_levers::CardBudget;

/// The entry of `budget` for the card on `device`: the one value, the
/// list's entry for the device's ordinal, or none — unset, a card the list
/// does not name, a card with no device. Shares are no bytes: the plan turns
/// them into a list first ([`split`]).
fn entry(budget: &CardBudget, device: Option<DeviceId>) -> Option<u64> {
    match budget {
        CardBudget::Unset | CardBudget::Shares(_) => None,
        CardBudget::Every(b) => Some(*b),
        CardBudget::Per(per) => {
            device.and_then(|d| per.iter().find(|&&(o, _)| o == d.ordinal).map(|&(_, b)| b))
        }
    }
}

/// The budget `card` plans with under `budget`: the one value, the list's
/// entry for the card's device ordinal, or none — unset, or a card the list
/// does not name. A plan refuses a list on a card with no census device
/// first ([`check`]), so a planned card is never left out of a list for
/// want of an ordinal.
#[must_use]
pub fn of(budget: &CardBudget, card: &Card) -> Option<u64> {
    entry(budget, card.device)
}

/// The margin the card on `device` keeps free under `margin` (the
/// `BLOOMERY_CARD_MARGIN` lever's value): the one value, the list's entry for
/// the device's ordinal, or none, in which case the card keeps its own.
#[must_use]
pub fn margin_of(margin: &CardBudget, device: Option<DeviceId>) -> Option<u64> {
    entry(margin, device)
}

/// `spec` keeping the margin `margin` gives its device ([`margin_of`]), or
/// as it is where `margin` names it none. A margin that leaves the card no
/// usable byte is refused by name ([`PlacementError::CardMargin`]).
pub fn keep_free(margin: &CardBudget, spec: CardSpec) -> Result<CardSpec, PlacementError> {
    let Some(bytes) = margin_of(margin, spec.device) else {
        return Ok(spec);
    };
    if bytes >= spec.usable_bytes() {
        return Err(PlacementError::CardMargin {
            card: spec.name.to_owned(),
            margin: bytes,
            usable: spec.usable_bytes(),
        });
    }
    Ok(CardSpec {
        margin_bytes: Some(bytes),
        ..spec
    })
}

/// The budget as a line states it in bytes: `unset`, or the value and its
/// unit.
#[must_use]
pub fn in_bytes(budget: &CardBudget) -> String {
    match budget {
        CardBudget::Unset => "unset".to_owned(),
        b => format!("{b} B"),
    }
}

/// Refuse a per-card `budget` on `machine` when a card of it has no device of
/// this process's census — a census-free card, which a list cannot name — or
/// takes its bytes from a tier node of its own, whose ordinal is that node's
/// process's, not this one's.
pub fn check(budget: &CardBudget, machine: &Machine) -> Result<(), PlacementError> {
    if budget.listed().is_none() {
        return Ok(());
    }
    for (i, card) in machine.all_cards().enumerate() {
        let why = if card.device.is_none() {
            "it was planned from no census, so it has no CUDA ordinal"
        } else if !machine.on_this_node(i) {
            "it is a tier node's, whose ordinals are that node's process's"
        } else {
            continue;
        };
        return Err(PlacementError::CardBudgetCard {
            card: card.name.clone(),
            budget: budget.to_string(),
            why,
        });
    }
    Ok(())
}

/// Refuse a per-card `budget` that names a device `census` does not list,
/// with every visible device.
pub fn check_visible(budget: &CardBudget, census: &[DeviceInfo]) -> Result<(), PlacementError> {
    check_listed("budget", budget, census)
}

/// [`check_visible`] for the per-card `margin` of `BLOOMERY_CARD_MARGIN`.
pub fn check_margin_visible(
    margin: &CardBudget,
    census: &[DeviceInfo],
) -> Result<(), PlacementError> {
    check_listed("margin", margin, census)
}

fn check_listed(
    what: &'static str,
    list: &CardBudget,
    census: &[DeviceInfo],
) -> Result<(), PlacementError> {
    let Some(entries) = list.listed() else {
        return Ok(());
    };
    match entries
        .iter()
        .find(|&&(o, _)| census.iter().all(|d| d.ordinal != o))
    {
        Some(&(ordinal, _)) => Err(PlacementError::CardBudgetUnseen {
            what,
            ordinal,
            budget: list.to_string(),
            visible: visible(census),
        }),
        None => Ok(()),
    }
}

/// One card of a share split ([`split`]): its share in parts and the bytes of
/// experts it can take past its floor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShareCard {
    pub weight: u64,
    pub room: u64,
}

/// The routed-expert bytes each of `cards` is given out of `experts`, in
/// order: in the proportion of the weights, and as many in all as the card
/// whose room is tightest for its share lets be placed (`min(experts,
/// min room × Σweight ÷ weight)` over the cards with a weight), each card's
/// part rounded down. A card of weight 0 is given none and bounds nothing; no
/// weight at all gives every card none. The ratio is kept exactly where it
/// can be, and what the tightest card cannot hold stays for the host tier.
#[must_use]
pub fn split(cards: &[ShareCard], experts: u64) -> Vec<u64> {
    let total: u128 = cards.iter().map(|c| u128::from(c.weight)).sum();
    let tightest = cards
        .iter()
        .filter(|c| c.weight > 0)
        .map(|c| u128::from(c.room).saturating_mul(total) / u128::from(c.weight))
        .min()
        .unwrap_or(0);
    let placed = u128::from(experts).min(tightest);
    cards
        .iter()
        .map(|c| {
            let part = placed * u128::from(c.weight) / total.max(1);
            u64::try_from(part).expect("a part of at most the experts' u64 bytes")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::super::devices::DeviceId;
    use super::super::{Host, Node};
    use super::*;

    /// A card named `name`, on the device of `ordinal` when one is given.
    fn card(name: &str, ordinal: Option<u32>) -> Card {
        Card {
            name: name.to_owned(),
            device: ordinal.map(|o| DeviceId {
                ordinal: o,
                uuid: [u8::try_from(o).expect("small") + 1; 16],
            }),
            usable_bytes: 24 << 30,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: NonZeroU64::new(2 << 20).expect("nonzero"),
            free_bytes: None,
            held_by: None,
            layers: 0..0,
            head: false,
            token_embedding: false,
            reserves: Vec::new(),
            reserve_only: false,
            tier_leg: None,
        }
    }

    fn machine(cards: Vec<Card>, tiers: Vec<Card>, nodes: usize) -> Machine {
        Machine {
            cards,
            tiers,
            nodes: (0..nodes)
                .map(|_| Node {
                    host: Host {
                        usable_bytes: 64 << 30,
                        reserves: Vec::new(),
                    },
                    unified: None,
                })
                .collect(),
        }
    }

    fn per(entries: &[(u32, u64)]) -> CardBudget {
        CardBudget::Per(entries.to_vec())
    }

    /// One value is every card's; a list gives a card its entry by device
    /// ordinal and leaves a card it does not name, or one with no device,
    /// at its own bytes; unset is none.
    #[test]
    fn a_card_plans_with_its_own_entry() {
        let (cuda0, cuda1, free) = (
            card("A6000", Some(0)),
            card("3090", Some(1)),
            card("x", None),
        );
        let list = per(&[(1, 20 << 30), (3, 1 << 30)]);
        assert_eq!(of(&list, &cuda1), Some(20 << 30));
        assert_eq!(of(&list, &cuda0), None);
        assert_eq!(of(&list, &free), None);
        for c in [&cuda0, &cuda1, &free] {
            assert_eq!(of(&CardBudget::Every(9), c), Some(9));
            assert_eq!(of(&CardBudget::Unset, c), None);
        }
    }

    /// A list on a machine with a census-free card or a tier node's card is
    /// refused by name; one value, and a list on census cards of this node,
    /// pass.
    #[test]
    fn a_list_needs_this_process_census_cards() {
        let list = per(&[(0, 1 << 30)]);
        let local = machine(vec![card("A6000", Some(0))], vec![card("3090", Some(1))], 1);
        assert!(check(&list, &local).is_ok());
        let census_free = machine(vec![card("A6000", None)], Vec::new(), 1);
        let e = check(&list, &census_free).expect_err("no ordinal");
        assert!(
            e.to_string().contains("card A6000") && e.to_string().contains("no CUDA ordinal"),
            "{e}"
        );
        assert!(check(&CardBudget::Every(1 << 30), &census_free).is_ok());
        let node = machine(vec![card("A6000", Some(0))], vec![card("3090", Some(0))], 2);
        let e = check(&list, &node).expect_err("a node's card");
        assert!(
            e.to_string().contains("card 3090") && e.to_string().contains("tier node's"),
            "{e}"
        );
    }

    /// A list naming a device the census lacks is refused with the ordinal
    /// and every visible device; one that names only listed devices, and one
    /// value, pass.
    #[test]
    fn a_list_names_visible_devices_only() {
        let census = vec![DeviceInfo {
            ordinal: 0,
            name: "NVIDIA RTX A6000".to_owned(),
            total_bytes: 50_952_536_064,
            free_bytes: 50_000_000_000,
            integrated: false,
            uuid: [1; 16],
            pci_bus: "0000:41:00.0".to_owned(),
            held_by: None,
            compute_capability: Some((8, 6)),
        }];
        assert!(check_visible(&per(&[(0, 1 << 30)]), &census).is_ok());
        assert!(check_visible(&CardBudget::Every(1), &census).is_ok());
        let e = check_visible(&per(&[(0, 1 << 30), (2, 1 << 30)]), &census)
            .expect_err("CUDA2")
            .to_string();
        assert!(
            e.contains("CUDA2") && e.contains("cuda0 NVIDIA RTX A6000"),
            "{e}"
        );
    }

    /// Shares are no bytes: no card plans with one before the plan resolves them, a
    /// list of shares needs census cards as a list of bytes does, and it names a device
    /// the census lists.
    #[test]
    fn shares_cap_no_card_and_name_census_devices() {
        let shares = CardBudget::Shares(vec![(0, 3), (1, 1)]);
        let cuda0 = card("A6000", Some(0));
        assert_eq!(of(&shares, &cuda0), None);
        let local = machine(vec![card("A6000", Some(0))], vec![card("3090", Some(1))], 1);
        assert!(check(&shares, &local).is_ok());
        let census_free = machine(vec![card("A6000", None)], Vec::new(), 1);
        let e = check(&shares, &census_free).expect_err("no ordinal");
        assert!(e.to_string().contains("card A6000"), "{e}");
        let seen = vec![DeviceInfo {
            ordinal: 0,
            name: "NVIDIA RTX A6000".to_owned(),
            total_bytes: 50_952_536_064,
            free_bytes: 50_000_000_000,
            integrated: false,
            uuid: [1; 16],
            pci_bus: "0000:41:00.0".to_owned(),
            held_by: None,
            compute_capability: Some((8, 6)),
        }];
        let e = check_visible(&shares, &seen)
            .expect_err("CUDA1")
            .to_string();
        assert!(
            e.starts_with("the per-card budget CUDA0=3x,CUDA1=1x names CUDA1"),
            "{e}"
        );
        let e = check_margin_visible(&per(&[(2, 1)]), &seen)
            .expect_err("CUDA2")
            .to_string();
        assert!(
            e.starts_with("the per-card margin CUDA2=1 names CUDA2") && e.contains("cuda0"),
            "{e}"
        );
        assert!(check_margin_visible(&CardBudget::Every(1), &seen).is_ok());
    }

    /// A margin goes on the spec of the device it names: the one value on every spec, a list
    /// entry on its ordinal's alone, a spec the list does not name or with no device as it was;
    /// one that leaves no usable byte is refused naming the card.
    #[test]
    fn a_margin_is_kept_by_the_cards_it_names() {
        use super::super::workstation::{A6000, RTX_3090};

        let on = |spec: CardSpec, ordinal: u32| CardSpec {
            device: Some(DeviceId {
                ordinal,
                uuid: [u8::try_from(ordinal).expect("small") + 1; 16],
            }),
            ..spec
        };
        let (a, t) = (on(A6000, 0), on(RTX_3090, 1));
        let margins = |m: &CardBudget| {
            (
                keep_free(m, a).expect("a").margin(),
                keep_free(m, t).expect("t").margin(),
                keep_free(m, RTX_3090).expect("no device").margin(),
            )
        };
        let own = super::super::workstation::MARGIN;
        assert_eq!(margins(&CardBudget::Unset), (own, own, own));
        assert_eq!(margins(&CardBudget::Every(7)), (7, 7, 7));
        assert_eq!(margins(&per(&[(1, 9)])), (own, 9, own));
        assert_eq!(margin_of(&per(&[(1, 9)]), t.device), Some(9));
        assert_eq!(margin_of(&per(&[(1, 9)]), a.device), None);
        assert_eq!(margin_of(&CardBudget::Shares(vec![(1, 9)]), t.device), None);
        let e = keep_free(&CardBudget::Every(RTX_3090.usable_bytes()), t)
            .expect_err("no usable byte")
            .to_string();
        assert!(
            e.starts_with("card 3090: a margin of") && e.ends_with("a plan may use"),
            "{e}"
        );
    }

    /// A share split gives the cards the experts' bytes in the proportion of their weights,
    /// and as many in all as the tightest card's room for its share lets: 3:1 over rooms of
    /// 30 and 20 holds 40 of 100 (30 and 10), 24 of 24 when they fit (18 and 6); 1:1 over 30
    /// and 20 holds 40 (20 each); a weight of 0 gets none and bounds nothing; no weight
    /// gives none; a card of weight and no room bounds the split to nothing.
    #[test]
    fn a_split_keeps_the_ratio_the_tightest_card_allows() {
        let card = |weight, room| ShareCard { weight, room };
        assert_eq!(split(&[card(3, 30), card(1, 20)], 100), [30, 10]);
        assert_eq!(split(&[card(3, 30), card(1, 20)], 24), [18, 6]);
        assert_eq!(split(&[card(1, 30), card(1, 20)], 100), [20, 20]);
        assert_eq!(split(&[card(1, 30), card(0, 0)], 100), [30, 0]);
        assert_eq!(split(&[card(0, 30), card(0, 20)], 100), [0, 0]);
        assert_eq!(split(&[card(1, 30), card(1, 0)], 100), [0, 0]);
        assert_eq!(
            split(&[card(750_000, 30), card(250_000, 20)], 100),
            [30, 10]
        );
        let huge = u64::MAX;
        assert_eq!(
            split(&[card(huge, huge), card(huge, huge)], huge),
            [huge / 2, huge / 2]
        );
    }

    /// A budget reads in bytes in a line: `unset`, or the value and `B`.
    #[test]
    fn a_budget_reads_in_bytes() {
        assert_eq!(in_bytes(&CardBudget::Unset), "unset");
        assert_eq!(in_bytes(&CardBudget::Every(9)), "9 B");
        assert_eq!(in_bytes(&per(&[(0, 1), (1, 2)])), "CUDA0=1,CUDA1=2 B");
    }
}
