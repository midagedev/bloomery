//! Where each expert of each MoE layer runs ([`SlotMap`]): a slot of the
//! stage card, a slot of the tier card ([`TIER`]), or the host ([`HOST`]).

use crate::GpuError;
use model::placement::{Device, Plan, Role};
use std::ops::Range;

/// The slot map's entry for an expert no card holds: the host computes it.
pub const HOST: u32 = u32::MAX;

/// The bit that marks an entry `TIER | s` as slot `s` of the tier card.
/// `TIER | s` equals [`HOST`] only at `s = 2^31 - 1`, which `s < capacity <=
/// n_expert < 2^31` ([`SlotMap::from_rows`]) rules out.
pub const TIER: u32 = 1 << 31;

/// One entry of the map, decoded ([`SlotMap::slot`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// Slot `s` of the stage card's routed stacks.
    Card(u32),
    /// Slot `s` of the tier card's routed stacks.
    Tier(u32),
    /// The host computes the expert.
    Host,
}

impl Slot {
    /// The one decoder of an entry. [`HOST`] has the tier bit set, so it is
    /// tested first.
    fn of(entry: u32) -> Slot {
        if entry == HOST {
            Slot::Host
        } else if entry & TIER != 0 {
            Slot::Tier(entry & !TIER)
        } else {
            Slot::Card(entry)
        }
    }
}

/// Where each expert of each MoE layer runs, host side: per layer of
/// `layers`, a row of `n_expert` entries, each a stage card slot, a tier card
/// slot (`TIER | s`) or [`HOST`]. The one owner of which experts run where:
/// the host serves an id exactly when [`SlotMap::slot`] says [`Slot::Host`],
/// and a card copy is one device's view of the rows
/// ([`SlotMap::stage_view`], [`SlotMap::tier_view`]), never the rows.
///
/// Per row and device the map holds a capacity — the slots of that device's
/// stacks in the layer — and the live entries, which name distinct slots
/// below it. A slot no live entry names is free: no id reaches it, so no
/// kernel reads it, whatever its bytes are. A load's map has capacity = live
/// ([`SlotMap::from_rows`]); the residency machine frees and fills slots
/// between passes through [`SlotMap::evict`] and [`SlotMap::admit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotMap {
    layers: Range<usize>,
    n_expert: usize,
    slots: Vec<u32>,
    /// Per row, the experts on the stage card: its live count.
    on_card: Vec<usize>,
    /// Per row, the experts on the tier card: its live count.
    on_tier: Vec<usize>,
    /// Per row, the stage card's slots.
    cap_card: Vec<usize>,
    /// Per row, the tier card's slots.
    cap_tier: Vec<usize>,
}

impl SlotMap {
    /// Experts `[0, n_l)` of every layer of `layers` in stage card slots
    /// `0..n_l`, the rest on the host.
    pub fn prefix(layers: Range<usize>, n_expert: usize, n_l: usize) -> Result<SlotMap, GpuError> {
        let what = "SlotMap::prefix";
        let n = u32::try_from(n_expert)
            .map_err(|_| GpuError::shape(what, format!("{n_expert} experts pass u32")))?;
        let cut = u32::try_from(n_l)
            .ok()
            .filter(|&c| c <= n)
            .ok_or_else(|| GpuError::shape(what, format!("{n_l} experts on the card of {n}")))?;
        let row: Vec<u32> = (0..n).map(|e| if e < cut { e } else { HOST }).collect();
        let slots = row.repeat(layers.len());
        SlotMap::from_rows(layers, n_expert, slots)
    }

    /// The map from its rows: `layers.len()` rows of `n_expert` entries, with
    /// `n_expert < 2^31`, each device's capacity in a row its live count.
    /// Per row and device, the slots of the row's `k` experts on that device
    /// are then `0..k`, each once; an expert has one entry, so it is on one
    /// device at most. A row that breaks this — a gap, a slot twice, a slot at
    /// or past its device's count, an entry past `n_expert` that is not
    /// [`HOST`] — is refused by name.
    pub fn from_rows(
        layers: Range<usize>,
        n_expert: usize,
        slots: Vec<u32>,
    ) -> Result<SlotMap, GpuError> {
        let what = "SlotMap::from_rows";
        if n_expert == 0
            || n_expert >= TIER as usize
            || layers.len().checked_mul(n_expert) != Some(slots.len())
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} entries for layers {layers:?} of {n_expert} experts",
                    slots.len()
                ),
            ));
        }
        let mut on_card = Vec::with_capacity(layers.len());
        let mut on_tier = Vec::with_capacity(layers.len());
        let (mut card_taken, mut tier_taken) = (vec![false; n_expert], vec![false; n_expert]);
        for (row, l) in slots.chunks_exact(n_expert).zip(layers.clone()) {
            let (mut k, mut k_tier) = (0usize, 0usize);
            for &e in row {
                match Slot::of(e) {
                    Slot::Card(_) => k += 1,
                    Slot::Tier(_) => k_tier += 1,
                    Slot::Host => {}
                }
            }
            card_taken.fill(false);
            tier_taken.fill(false);
            for &e in row {
                let (s, taken, capacity, device) = match Slot::of(e) {
                    Slot::Host => continue,
                    Slot::Card(s) => (s, &mut card_taken, k, "the card"),
                    Slot::Tier(s) => (s, &mut tier_taken, k_tier, "the tier"),
                };
                take_slot(taken, s, capacity, device)
                    .map_err(|why| GpuError::shape(what, format!("layer {l}: {why}")))?;
            }
            on_card.push(k);
            on_tier.push(k_tier);
        }
        Ok(SlotMap {
            layers,
            n_expert,
            slots,
            cap_card: on_card.clone(),
            cap_tier: on_tier.clone(),
            on_card,
            on_tier,
        })
    }

    /// The map a placement plan makes for stage card `card` and, when given,
    /// tier card `tier`: per layer of `layers`, each device's segments of each
    /// routed stack, in plan order, fill that device's slots from 0 with their
    /// experts ([`TIER`] marks the tier's), [`HOST`] for the rest. Refused by
    /// name: a routed segment of a layer of `layers` on any other card, an
    /// expert placed twice, an expert not one of the `n_expert`, routed stacks
    /// of one layer that disagree, and `tier` equal to `card`.
    pub fn of_plan(
        plan: &Plan<'_>,
        card: usize,
        tier: Option<usize>,
        layers: Range<usize>,
        n_expert: usize,
    ) -> Result<SlotMap, GpuError> {
        let refuse = |detail: String| GpuError::Shape {
            what: "SlotMap::of_plan",
            detail,
        };
        if tier == Some(card) {
            return Err(refuse(format!(
                "card {card} is both the stage card and the tier"
            )));
        }
        let mut map = vec![HOST; layers.len() * n_expert];
        let mut first: Vec<Option<&str>> = vec![None; layers.len()];
        let mut stack = vec![HOST; n_expert];
        for row in &plan.rows {
            let t = plan
                .model
                .tensors
                .get(row.tensor)
                .ok_or_else(|| refuse(format!("a plan row names tensor {}", row.tensor)))?;
            let Some(l) = t.layer.filter(|l| layers.contains(l)) else {
                continue;
            };
            if t.role != Role::RoutedExperts {
                continue;
            }
            stack.fill(HOST);
            let (mut n_card, mut n_tier) = (0u32, 0u32);
            for seg in &row.segments {
                let Device::Card(c) = seg.device else {
                    continue;
                };
                let (next, code) = if c == card {
                    (&mut n_card, 0)
                } else if tier == Some(c) {
                    (&mut n_tier, TIER)
                } else {
                    let known = match tier {
                        Some(tier) => format!("the stage card {card} or the tier {tier}"),
                        None => format!("the stage card {card}, and the map has no tier"),
                    };
                    return Err(refuse(format!(
                        "{}: routed experts on card {c}, not {known}",
                        t.name
                    )));
                };
                let experts = seg.experts.clone().ok_or_else(|| {
                    refuse(format!(
                        "{}: a card segment without an expert range",
                        t.name
                    ))
                })?;
                for e in experts {
                    let entry = usize::try_from(e)
                        .ok()
                        .and_then(|e| stack.get_mut(e))
                        .ok_or_else(|| refuse(format!("{}: expert {e} of {n_expert}", t.name)))?;
                    if *entry != HOST {
                        return Err(refuse(format!("{}: expert {e} placed twice", t.name)));
                    }
                    *entry = code | *next;
                    *next += 1;
                }
            }
            let i = l - layers.start;
            let dst = &mut map[i * n_expert..(i + 1) * n_expert];
            let seen = first[i];
            match seen {
                None => {
                    dst.copy_from_slice(&stack);
                    first[i] = Some(&t.name);
                }
                Some(other) if *dst != *stack => {
                    return Err(refuse(format!(
                        "{} puts other experts on the cards than {other}",
                        t.name
                    )));
                }
                Some(_) => {}
            }
        }
        SlotMap::from_rows(layers, n_expert, map)
    }

    /// The layers the map has rows for.
    #[must_use]
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Entries per row: the file's experts per layer.
    #[must_use]
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// Where layer `layer`'s expert `id` runs; `None` for a layer or an id
    /// the map has no entry for. The decoder every host reader goes through.
    #[must_use]
    pub fn slot(&self, layer: usize, id: u32) -> Option<Slot> {
        let at = self.row_offset(layer)?;
        let id = usize::try_from(id).ok().filter(|&id| id < self.n_expert)?;
        self.slots.get(at + id).map(|&e| Slot::of(e))
    }

    /// The stage card's copy, every row in layer order: its slots as they
    /// are, [`HOST`] for every other entry. A map with no tier entry reads
    /// back its rows unchanged.
    #[must_use]
    pub fn stage_view(&self) -> Vec<u32> {
        self.view(|s| match s {
            Slot::Card(s) => s,
            Slot::Tier(_) | Slot::Host => HOST,
        })
    }

    /// The tier card's copy, every row in layer order: `s` for a tier entry
    /// `TIER | s`, [`HOST`] for every other entry.
    #[must_use]
    pub fn tier_view(&self) -> Vec<u32> {
        self.view(|s| match s {
            Slot::Tier(s) => s,
            Slot::Card(_) | Slot::Host => HOST,
        })
    }

    fn view(&self, of: impl Fn(Slot) -> u32) -> Vec<u32> {
        self.slots.iter().map(|&e| of(Slot::of(e))).collect()
    }

    /// Send layer `layer`'s expert `id`, which the host computes, to `slot`, a
    /// free slot of the stage card or the tier: the entry `HOST` becomes the
    /// slot and the device's live count grows by one. Refused by name, the map
    /// unchanged: a layer or an id outside the map, an id a card holds,
    /// [`Slot::Host`], a slot at or past its device's capacity in the row, and
    /// a slot a live entry of the row names.
    pub fn admit(&mut self, layer: usize, id: u32, slot: Slot) -> Result<(), GpuError> {
        const WHAT: &str = "SlotMap::admit";
        let (at, i) = self.entry_at(layer, id, WHAT)?;
        let (s, cap, code) = match slot {
            Slot::Card(s) => (s, self.cap_card[i], s),
            Slot::Tier(s) => (s, self.cap_tier[i], TIER | s),
            Slot::Host => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {layer} expert {id} to the host: that is SlotMap::evict"),
                ));
            }
        };
        if self.slots[at] != HOST {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} expert {id} is on {:?} already",
                    Slot::of(self.slots[at])
                ),
            ));
        }
        if !usize::try_from(s).is_ok_and(|s| s < cap) {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer}: {slot:?} is past the device's {cap} slots in the row"),
            ));
        }
        let row = i * self.n_expert..(i + 1) * self.n_expert;
        if let Some(other) = self.slots[row.clone()].iter().position(|&e| e == code) {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer}: {slot:?} holds expert {other}"),
            ));
        }
        self.slots[at] = code;
        match slot {
            Slot::Card(_) => self.on_card[i] += 1,
            Slot::Tier(_) => self.on_tier[i] += 1,
            Slot::Host => {}
        }
        Ok(())
    }

    /// Send layer `layer`'s expert `id` from its slot to the host and return
    /// the slot, now free: the device's live count drops by one and its
    /// capacity stays. Refused by name, the map unchanged: a layer or an id
    /// outside the map, and an id the host computes already.
    pub fn evict(&mut self, layer: usize, id: u32) -> Result<Slot, GpuError> {
        const WHAT: &str = "SlotMap::evict";
        let (at, i) = self.entry_at(layer, id, WHAT)?;
        let slot = Slot::of(self.slots[at]);
        match slot {
            Slot::Card(_) => self.on_card[i] -= 1,
            Slot::Tier(_) => self.on_tier[i] -= 1,
            Slot::Host => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {layer} expert {id} is on the host already"),
                ));
            }
        }
        self.slots[at] = HOST;
        Ok(slot)
    }

    /// The entry index of layer `layer`'s expert `id` and the row's index,
    /// or the refusal of `what` naming which is outside the map.
    fn entry_at(
        &self,
        layer: usize,
        id: u32,
        what: &'static str,
    ) -> Result<(usize, usize), GpuError> {
        let at = self
            .row_offset(layer)
            .ok_or_else(|| self.outside(layer, what))?;
        let e = usize::try_from(id)
            .ok()
            .filter(|&e| e < self.n_expert)
            .ok_or_else(|| {
                GpuError::shape(what, format!("expert {id} of {} a layer", self.n_expert))
            })?;
        Ok((at + e, at / self.n_expert))
    }

    fn outside(&self, layer: usize, what: &'static str) -> GpuError {
        GpuError::shape(
            what,
            format!(
                "layer {layer} is outside the map's layers {:?}",
                self.layers
            ),
        )
    }

    /// Layer `layer`'s row's first entry in either view, the card copy's word
    /// offset a kernel adds an expert id to; `None` for a layer the map has
    /// no row for. The one owner of that offset: [`SlotMap::row`] reads the
    /// same row.
    #[must_use]
    pub fn row_offset(&self, layer: usize) -> Option<usize> {
        let i = layer.checked_sub(self.layers.start)?;
        (i < self.layers.len()).then_some(i * self.n_expert)
    }

    /// Layer `layer`'s row as the map holds it, all three kinds of entry;
    /// `None` for a layer the map has no row for. Host side only: a reader
    /// decodes it through [`SlotMap::slot`], and a card copy is a view.
    #[must_use]
    pub fn row(&self, layer: usize) -> Option<&[u32]> {
        let at = self.row_offset(layer)?;
        self.slots.get(at..at + self.n_expert)
    }

    /// The experts of layer `layer` the stage card holds; a layer the map
    /// has no row for is refused by name.
    pub fn on_card(&self, layer: usize) -> Result<usize, GpuError> {
        self.count(&self.on_card, layer, "SlotMap::on_card")
    }

    /// The experts of layer `layer` the tier card holds; a layer the map has
    /// no row for is refused by name.
    pub fn on_tier(&self, layer: usize) -> Result<usize, GpuError> {
        self.count(&self.on_tier, layer, "SlotMap::on_tier")
    }

    /// The stage card's slots in layer `layer`: its stacks' rows over the
    /// rows an expert takes, at least [`SlotMap::on_card`]; a layer the map
    /// has no row for is refused by name.
    pub fn capacity(&self, layer: usize) -> Result<usize, GpuError> {
        self.count(&self.cap_card, layer, "SlotMap::capacity")
    }

    /// The tier card's slots in layer `layer`, at least
    /// [`SlotMap::on_tier`]; a layer the map has no row for is refused by
    /// name.
    pub fn tier_capacity(&self, layer: usize) -> Result<usize, GpuError> {
        self.count(&self.cap_tier, layer, "SlotMap::tier_capacity")
    }

    fn count(
        &self,
        per_row: &[usize],
        layer: usize,
        what: &'static str,
    ) -> Result<usize, GpuError> {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| per_row.get(i))
            .copied()
            .ok_or_else(|| self.outside(layer, what))
    }
}

/// Marks slot `s` of a device with `capacity` slots in the row: the row's
/// live slots on it must be distinct and below `capacity`, so a slot at or
/// past it or taken before is refused, with why.
fn take_slot(taken: &mut [bool], s: u32, capacity: usize, device: &str) -> Result<(), String> {
    let Some(seen) = usize::try_from(s)
        .ok()
        .filter(|&i| i < capacity)
        .and_then(|i| taken.get_mut(i))
    else {
        return Err(
            if usize::try_from(s).ok().is_none_or(|i| i >= taken.len()) {
                format!(
                    "slot {s} on {device} is past the row's {} experts: neither a card slot, a tier \
                 slot nor HOST",
                    taken.len()
                )
            } else {
                format!("slot {s} on {device}, and the row has {capacity} slots there")
            },
        );
    };
    if std::mem::replace(seen, true) {
        return Err(format!("slot {s} on {device} holds two experts"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{HOST, Slot, SlotMap, TIER};
    use gguf::GgmlType;
    use model::placement::{
        self, Device, ExpertList, KvBytes, ModelTensor, ModelTensors, Plan, Role, workstation,
    };

    /// The prefix map sends to the host exactly the ids at or past `n_l`, on
    /// every layer it has a row for, and knows no other layer.
    #[test]
    fn prefix_sends_the_ids_past_n_l_to_the_host() {
        let map = SlotMap::prefix(1..4, 64, 22).expect("a prefix of 22 of 64");
        for layer in 1..4 {
            let row = map.row(layer).expect("a row per layer of the range");
            for (id, &slot) in row.iter().enumerate() {
                assert_eq!(slot == HOST, id >= 22, "layer {layer} id {id}: {slot}");
            }
            assert_eq!(map.on_card(layer).ok(), Some(22));
            assert_eq!(map.on_tier(layer).ok(), Some(0));
        }
        assert!(map.row(0).is_none() && map.row(4).is_none());
        assert!(SlotMap::prefix(0..1, 64, 65).is_err());
    }

    /// A layer's row offset is its row's place in the card copy: rows of
    /// different card experts read back at their offsets, and a layer
    /// outside the map has none.
    #[test]
    fn row_offset_is_the_rows_place_in_the_card_copy() {
        let rows = vec![1, HOST, 0, HOST, HOST, HOST, HOST, HOST, 2, 0, HOST, 1];
        let map = SlotMap::from_rows(3..6, 4, rows.clone()).expect("three rows of 4");
        for (i, layer) in (3..6).enumerate() {
            let at = map.row_offset(layer).expect("a row per layer of the range");
            assert_eq!(at, i * 4);
            assert_eq!(map.row(layer), Some(&rows[at..at + 4]));
        }
        assert_eq!(map.on_card(3).ok(), Some(2));
        assert_eq!(map.on_card(4).ok(), Some(0));
        assert_eq!(map.on_card(5).ok(), Some(3));
        assert!(map.row_offset(2).is_none() && map.row_offset(6).is_none());
    }

    /// A row's card slots are `0..k` for its `k` experts on the card, each
    /// once; any other row is refused, and so is a wrong entry count.
    #[test]
    fn from_rows_refuses_a_slot_twice_or_past_the_card_count() {
        assert!(SlotMap::from_rows(0..1, 4, vec![1, HOST, 0, HOST]).is_ok());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, 0, HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, 2, HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..2, 4, vec![0, 1, HOST, HOST]).is_err());
    }

    /// The tier's slots of a row are `0..k_tier` for its `k_tier` tier
    /// experts, each once, beside the card's own `0..k`: a tier slot at or
    /// past `k_tier`, one twice, a gap, and an entry past the experts that is
    /// neither kind nor HOST are refused.
    #[test]
    fn from_rows_holds_each_device_to_its_own_slots() {
        let t = |s: u32| TIER | s;
        let map = SlotMap::from_rows(0..1, 6, vec![1, t(1), HOST, 0, t(0), t(2)])
            .expect("two card and three tier experts");
        assert_eq!(map.on_card(0).ok(), Some(2));
        assert_eq!(map.on_tier(0).ok(), Some(3));
        assert!(SlotMap::from_rows(0..1, 4, vec![0, t(1), HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, t(0), t(0), HOST]).is_err());
        assert!(SlotMap::from_rows(0..1, 4, vec![t(0), t(2), HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, 9, HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, t(9), HOST, HOST]).is_err());
    }

    /// The decoder reads each entry as its kind: HOST is the host's, never a
    /// tier slot, and a tier entry is the tier's, never the host's — so the
    /// ids a host list takes (`Slot::Host`) are exactly the HOST entries.
    #[test]
    fn slot_decodes_the_three_kinds_and_the_host_takes_host_only() {
        let row = vec![0, TIER, HOST, 1, TIER | 1, HOST];
        let map = SlotMap::from_rows(2..3, 6, row.clone()).expect("a row of three kinds");
        let got: Vec<Option<Slot>> = (0..6).map(|id| map.slot(2, id)).collect();
        assert_eq!(
            got,
            [
                Slot::Card(0),
                Slot::Tier(0),
                Slot::Host,
                Slot::Card(1),
                Slot::Tier(1),
                Slot::Host
            ]
            .map(Some)
        );
        let host: Vec<u32> = (0..6)
            .filter(|&id| map.slot(2, id) == Some(Slot::Host))
            .collect();
        let want: Vec<u32> = (0..6).filter(|&id| row[id as usize] == HOST).collect();
        assert_eq!(host, want);
        assert_eq!(map.slot(2, 6), None);
        assert_eq!(map.slot(1, 0), None);
        assert_eq!(map.slot(3, 0), None);
    }

    /// Each device's copy holds its own slots and HOST for the rest; with no
    /// tier entry the stage copy is the rows as they are and the tier copy
    /// all HOST.
    #[test]
    fn views_split_the_rows_by_device() {
        let map = SlotMap::from_rows(0..2, 3, vec![TIER, 0, HOST, 1, 0, TIER])
            .expect("two rows with a tier");
        assert_eq!(map.stage_view(), vec![HOST, 0, HOST, 1, 0, HOST]);
        assert_eq!(map.tier_view(), vec![0, HOST, HOST, HOST, HOST, 0]);
        let plain = SlotMap::prefix(0..3, 5, 2).expect("a prefix of 2 of 5");
        let rows: Vec<u32> = (0..3)
            .flat_map(|l| plain.row(l).expect("a row").to_vec())
            .collect();
        assert_eq!(plain.stage_view(), rows);
        assert_eq!(plain.tier_view(), vec![HOST; 15]);
    }

    /// A layer outside the map is refused by name on either count, never a
    /// quiet 0.
    #[test]
    fn counts_refuse_a_layer_outside_the_map() {
        let map = SlotMap::prefix(1..3, 4, 2).expect("a prefix of 2 of 4");
        for layer in [0, 3] {
            assert!(map.on_card(layer).is_err(), "on_card({layer})");
            assert!(map.on_tier(layer).is_err(), "on_tier({layer})");
            assert!(map.capacity(layer).is_err(), "capacity({layer})");
            assert!(map.tier_capacity(layer).is_err(), "tier_capacity({layer})");
        }
    }

    /// A map from its rows alone has each device's capacity at its live
    /// count, the load's map: capacity = live, slots `0..k` each once.
    #[test]
    fn from_rows_has_capacity_equal_to_live() {
        let t = |s: u32| TIER | s;
        let map = SlotMap::from_rows(
            0..2,
            5,
            vec![1, 0, t(0), HOST, HOST, HOST, HOST, 0, t(1), t(0)],
        )
        .expect("two rows");
        assert_eq!(
            (map.capacity(0).ok(), map.on_card(0).ok()),
            (Some(2), Some(2))
        );
        assert_eq!(
            (map.tier_capacity(0).ok(), map.on_tier(0).ok()),
            (Some(1), Some(1))
        );
        assert_eq!(
            (map.capacity(1).ok(), map.on_card(1).ok()),
            (Some(1), Some(1))
        );
        assert_eq!(
            (map.tier_capacity(1).ok(), map.on_tier(1).ok()),
            (Some(2), Some(2))
        );
    }

    /// An eviction frees its slot and an admission fills a free one: the live
    /// count moves, the capacity stays, the stage card's copy follows; each
    /// refusal leaves the map unchanged.
    #[test]
    fn evict_then_admit_moves_one_entry() {
        let h = HOST;
        let mut map = SlotMap::from_rows(4..5, 6, vec![0, 2, 1, h, TIER, h]).expect("a full row");
        assert_eq!(map.evict(4, 1).ok(), Some(Slot::Card(2)));
        assert_eq!(map.row(4), Some(&[0, h, 1, h, TIER, h][..]));
        let before = map.clone();
        let refused = |r: Result<(), crate::GpuError>, map: &SlotMap, why: &str| {
            assert!(r.is_err(), "{why}");
            assert_eq!(*map, before, "{why}: the map moved");
        };
        refused(map.admit(4, 0, Slot::Card(2)), &map, "an id on the card");
        refused(map.admit(4, 4, Slot::Card(2)), &map, "an id on the tier");
        refused(
            map.admit(4, 1, Slot::Card(1)),
            &map,
            "a slot another id holds",
        );
        refused(
            map.admit(4, 1, Slot::Card(3)),
            &map,
            "a slot at the capacity",
        );
        refused(
            map.admit(4, 1, Slot::Tier(0)),
            &map,
            "a tier slot the tier holds",
        );
        refused(map.admit(4, 1, Slot::Host), &map, "the host");
        refused(map.admit(3, 1, Slot::Card(2)), &map, "a layer outside");
        refused(map.admit(4, 6, Slot::Card(2)), &map, "an id outside");
        assert!(
            map.evict(4, 1).is_err() && map == before,
            "an id on the host"
        );

        map.admit(4, 1, Slot::Card(2)).expect("a free slot");
        assert_eq!(map.row(4), Some(&[0, 2, 1, h, TIER, h][..]));
        assert_eq!(
            (map.on_card(4).ok(), map.capacity(4).ok()),
            (Some(3), Some(3))
        );
        assert_eq!(
            map.stage_view(),
            [0, 2, 1, h, h, h],
            "the tier's entry is HOST in the stage card's copy"
        );
        assert_eq!(map.evict(4, 0).ok(), Some(Slot::Card(0)));
        assert_eq!(map.evict(4, 4).ok(), Some(Slot::Tier(0)));
        assert_eq!(map.row(4), Some(&[h, 2, 1, h, h, h][..]));
        assert_eq!(
            (map.on_card(4).ok(), map.capacity(4).ok()),
            (Some(2), Some(3))
        );
        assert_eq!(
            (map.on_tier(4).ok(), map.tier_capacity(4).ok()),
            (Some(0), Some(1))
        );
        assert_eq!(map.stage_view(), [h, 2, 1, h, h, h]);
        map.admit(4, 3, Slot::Card(0)).expect("the freed slot");
        map.admit(4, 5, Slot::Tier(0)).expect("the freed tier slot");
        assert_eq!(
            SlotMap::from_rows(4..5, 6, map.row(4).expect("a row").to_vec()).ok(),
            Some(map.clone()),
            "the moved row, full again, is a load's map"
        );
    }

    /// No KV cache: the synthetic model's card holds only its tensors.
    struct NoKv;

    impl KvBytes for NoKv {
        fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
            0
        }
    }

    const EXPERTS: u64 = 8;

    fn synthetic(
        name: &str,
        layer: Option<usize>,
        role: Role,
        ty: GgmlType,
        rows: &[u64],
    ) -> ModelTensor {
        let mut dims = vec![256];
        dims.extend_from_slice(rows);
        let blocks = 256 / ty.blck_size().expect("a sized type");
        let file_bytes =
            ty.type_size().expect("a sized type") * blocks * rows.iter().product::<u64>();
        ModelTensor {
            name: name.to_string(),
            shard: 0,
            layer,
            role,
            ty,
            dims,
            file_bytes,
            gathered_rows: (role == Role::TokenEmbedding).then_some(1),
        }
    }

    /// Two layers, each with two routed stacks of 8 experts.
    fn model() -> ModelTensors {
        let mut tensors = vec![synthetic(
            "token_embd.weight",
            None,
            Role::TokenEmbedding,
            GgmlType::Q8_0,
            &[16],
        )];
        for l in 0..2 {
            for part in ["gate", "down"] {
                tensors.push(synthetic(
                    &format!("layer{l}.{part}_experts"),
                    Some(l),
                    Role::RoutedExperts,
                    GgmlType::Q4_K,
                    &[4, EXPERTS],
                ));
            }
        }
        tensors.push(synthetic(
            "output.weight",
            None,
            Role::Head,
            GgmlType::Q8_0,
            &[16],
        ));
        ModelTensors {
            tensors,
            layers: 2,
            experts: EXPERTS,
            experts_used: 2,
        }
    }

    /// Every routed stack of layer `layer` in `plan` gets the segments
    /// `segs`, in that order, each shaped like the row's first.
    fn place(plan: &mut Plan<'_>, layer: usize, segs: &[(Device, &[u32])]) {
        let model = plan.model;
        for row in &mut plan.rows {
            let t = &model.tensors[row.tensor];
            if t.layer != Some(layer) || t.role != Role::RoutedExperts {
                continue;
            }
            let like = row.segments[0].clone();
            row.segments = segs
                .iter()
                .map(|&(device, ids)| placement::Segment {
                    device,
                    experts: Some(ExpertList::new(ids.to_vec(), EXPERTS).expect("a list")),
                    ..like.clone()
                })
                .collect();
        }
    }

    /// With one card and no tier the map is the plan's card segments filled
    /// into slots in plan order, as it always was: the plan's own prefix
    /// equals [`SlotMap::prefix`], and hand-placed lists fill their slots
    /// segment after segment.
    #[test]
    fn of_plan_one_card_no_tier_is_the_card_fill() {
        let (model, machine) = (model(), workstation::plan_a(2));
        let mut plan = placement::plan_with(&model, &machine, 4096, &NoKv, None, None)
            .expect("a synthetic plan");
        let n_l = plan.n_l.clone();
        assert!(
            n_l.iter().any(|&n| n > 0),
            "the plan puts no expert on the card: {n_l:?}"
        );
        let own = SlotMap::of_plan(&plan, 0, None, 0..2, 8).expect("the plan's own map");
        for (l, &n) in n_l.iter().enumerate() {
            let want = SlotMap::prefix(l..l + 1, 8, n as usize).expect("a prefix");
            assert_eq!(own.row(l), want.row(l), "layer {l} of n_l {n}");
        }
        place(
            &mut plan,
            0,
            &[
                (Device::Card(0), &[5, 6]),
                (Device::Host, &[0, 2, 3, 4, 7]),
                (Device::Card(0), &[1]),
            ],
        );
        place(&mut plan, 1, &[(Device::Host, &[0, 1, 2, 3, 4, 5, 6, 7])]);
        let map = SlotMap::of_plan(&plan, 0, None, 0..2, 8).expect("hand-placed lists");
        let h = HOST;
        assert_eq!(map.row(0), Some(&[h, 2, h, h, h, 0, 1, h][..]));
        assert_eq!(map.row(1), Some(&[h; 8][..]));
        assert_eq!(
            map.stage_view(),
            [&[h, 2, h, h, h, 0, 1, h][..], &[h; 8][..]].concat()
        );
        assert_eq!(
            (map.on_card(0).ok(), map.on_tier(0).ok()),
            (Some(3), Some(0))
        );
    }

    /// Without a tier, a routed segment on another card within the map's
    /// layers is refused by name, and one on a layer outside them is not the
    /// map's to read (a two-stage plan's second card).
    #[test]
    fn of_plan_without_a_tier_refuses_another_card() {
        let (model, machine) = (model(), workstation::plan_a(2));
        let mut plan = placement::plan_with(&model, &machine, 4096, &NoKv, None, None)
            .expect("a synthetic plan");
        place(
            &mut plan,
            0,
            &[
                (Device::Card(0), &[0, 1]),
                (Device::Host, &[2, 3, 4, 5, 6, 7]),
            ],
        );
        place(
            &mut plan,
            1,
            &[
                (Device::Card(1), &[0, 1, 2]),
                (Device::Host, &[3, 4, 5, 6, 7]),
            ],
        );
        let err =
            SlotMap::of_plan(&plan, 0, None, 0..2, 8).expect_err("card 1 in the map's layers");
        assert!(err.to_string().contains("card 1"), "{err}");
        let first = SlotMap::of_plan(&plan, 0, None, 0..1, 8).expect("layer 0 alone");
        assert_eq!(first.on_card(0).ok(), Some(2));
        let second = SlotMap::of_plan(&plan, 1, None, 1..2, 8).expect("layer 1 on card 1");
        assert_eq!(second.on_card(1).ok(), Some(3));
    }

    /// With a tier, its segments fill the tier's slots in plan order beside
    /// the stage card's; an expert on both, an expert twice on one card, a
    /// third card, and the tier named as the stage card are refused.
    #[test]
    fn of_plan_fills_the_tier_and_refuses_an_expert_on_both() {
        let (model, machine) = (model(), workstation::plan_a(2));
        let mut plan = placement::plan_with(&model, &machine, 4096, &NoKv, None, None)
            .expect("a synthetic plan");
        place(
            &mut plan,
            0,
            &[
                (Device::Card(0), &[0, 1]),
                (Device::Card(1), &[4, 2, 3]),
                (Device::Host, &[5, 6, 7]),
            ],
        );
        place(
            &mut plan,
            1,
            &[
                (Device::Card(1), &[7]),
                (Device::Host, &[0, 1, 2, 3, 4, 5, 6]),
            ],
        );
        let map = SlotMap::of_plan(&plan, 0, Some(1), 0..2, 8).expect("a stage card and a tier");
        let (h, t) = (HOST, |s: u32| TIER | s);
        assert_eq!(map.row(0), Some(&[0, 1, t(0), t(1), t(2), h, h, h][..]));
        assert_eq!(map.row(1), Some(&[h, h, h, h, h, h, h, t(0)][..]));
        assert_eq!(
            (map.on_card(0).ok(), map.on_tier(0).ok()),
            (Some(2), Some(3))
        );
        assert_eq!(
            (map.on_card(1).ok(), map.on_tier(1).ok()),
            (Some(0), Some(1))
        );
        assert_eq!(
            map.tier_view(),
            [&[h, h, 0, 1, 2, h, h, h][..], &[h, h, h, h, h, h, h, 0][..]].concat()
        );
        assert!(SlotMap::of_plan(&plan, 0, Some(0), 0..2, 8).is_err());
        assert!(SlotMap::of_plan(&plan, 0, Some(2), 0..2, 8).is_err());

        place(
            &mut plan,
            1,
            &[(Device::Card(0), &[0, 1]), (Device::Card(1), &[1, 2])],
        );
        let both = SlotMap::of_plan(&plan, 0, Some(1), 0..2, 8).expect_err("expert 1 on both");
        assert!(both.to_string().contains("expert 1 placed twice"), "{both}");
        place(
            &mut plan,
            1,
            &[(Device::Card(0), &[0]), (Device::Card(0), &[0])],
        );
        assert!(SlotMap::of_plan(&plan, 0, Some(1), 0..2, 8).is_err());
    }
}
