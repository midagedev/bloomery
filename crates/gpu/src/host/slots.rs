//! Which experts of each MoE layer the card holds ([`SlotMap`]); the host
//! computes the rest ([`HOST`]).

use crate::GpuError;
use model::placement::{Device, Plan, Role};
use std::ops::Range;

/// The slot map's entry for an expert the card does not hold: the host
/// computes it.
pub const HOST: u32 = u32::MAX;

/// Which experts of each MoE layer the card holds, host side: per layer of
/// `layers`, a row of `n_expert` entries, each the slot of the layer's routed
/// stack that holds the expert or [`HOST`]. The one owner of which experts
/// run on the host: the tier serves an id exactly when the map sends it
/// there, and a chain that reads the map on the card uploads this one
/// ([`SlotMap::as_slice`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotMap {
    layers: Range<usize>,
    n_expert: usize,
    slots: Vec<u32>,
    /// Per row, the experts on the card.
    on_card: Vec<usize>,
}

impl SlotMap {
    /// Experts `[0, n_l)` of every layer of `layers` in slots `0..n_l`, the
    /// rest on the host.
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

    /// The map from its rows: `layers.len()` rows of `n_expert` entries. The
    /// entries of a row that are not [`HOST`] are the slots `0..k` of the
    /// row's `k` experts on the card, each once.
    pub fn from_rows(
        layers: Range<usize>,
        n_expert: usize,
        slots: Vec<u32>,
    ) -> Result<SlotMap, GpuError> {
        let what = "SlotMap::from_rows";
        if n_expert == 0 || layers.len().checked_mul(n_expert) != Some(slots.len()) {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} entries for layers {layers:?} of {n_expert} experts",
                    slots.len()
                ),
            ));
        }
        let mut on_card = Vec::with_capacity(layers.len());
        let mut taken = vec![false; n_expert];
        for (row, l) in slots.chunks_exact(n_expert).zip(layers.clone()) {
            let k = row.iter().filter(|&&s| s != HOST).count();
            taken.fill(false);
            for &s in row.iter().filter(|&&s| s != HOST) {
                let slot = usize::try_from(s)
                    .ok()
                    .filter(|&s| s < k)
                    .and_then(|s| taken.get_mut(s))
                    .ok_or_else(|| {
                        GpuError::shape(
                            what,
                            format!(
                                "layer {l}: slot {s}, and the row puts {k} experts on the card"
                            ),
                        )
                    })?;
                if std::mem::replace(slot, true) {
                    return Err(GpuError::shape(
                        what,
                        format!("layer {l}: slot {s} holds two experts"),
                    ));
                }
            }
            on_card.push(k);
        }
        Ok(SlotMap {
            layers,
            n_expert,
            slots,
            on_card,
        })
    }

    /// The map a placement plan makes for card `card`: per layer of `layers`,
    /// the card's segments of each routed stack, in plan order, fill its
    /// slots from 0 with their experts, [`HOST`] for the rest. Every routed
    /// stack of a layer must put the same experts in the same slots, and
    /// each expert must be one of the `n_expert`; either is refused by name.
    pub fn of_plan(
        plan: &Plan<'_>,
        card: usize,
        layers: Range<usize>,
        n_expert: usize,
    ) -> Result<SlotMap, GpuError> {
        let refuse = |detail: String| GpuError::Shape {
            what: "SlotMap::of_plan",
            detail,
        };
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
            let mut slot = 0u32;
            for seg in row
                .segments
                .iter()
                .filter(|s| s.device == Device::Card(card))
            {
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
                    *entry = slot;
                    slot += 1;
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
                        "{} puts other experts on the card than {other}",
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

    /// Every row in layer order, as a card-side copy holds it.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.slots
    }

    /// Layer `layer`'s row's first entry in [`SlotMap::as_slice`], the card
    /// copy's word offset a kernel adds an expert id to; `None` for a layer
    /// the map has no row for. The one owner of that offset: [`SlotMap::row`]
    /// reads the same row.
    #[must_use]
    pub fn row_offset(&self, layer: usize) -> Option<usize> {
        let i = layer.checked_sub(self.layers.start)?;
        (i < self.layers.len()).then_some(i * self.n_expert)
    }

    /// Layer `layer`'s row; `None` for a layer the map has no row for.
    #[must_use]
    pub fn row(&self, layer: usize) -> Option<&[u32]> {
        let at = self.row_offset(layer)?;
        self.slots.get(at..at + self.n_expert)
    }

    /// The experts of layer `layer` the card holds; 0 for a layer the map has
    /// no row for.
    #[must_use]
    pub fn on_card(&self, layer: usize) -> usize {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.on_card.get(i))
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{HOST, SlotMap};

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
            assert_eq!(map.on_card(layer), 22);
        }
        assert!(map.row(0).is_none() && map.row(4).is_none());
        assert_eq!(map.on_card(4), 0);
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
        assert_eq!(map.on_card(3), 2);
        assert_eq!(map.on_card(4), 0);
        assert_eq!(map.on_card(5), 3);
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
}
