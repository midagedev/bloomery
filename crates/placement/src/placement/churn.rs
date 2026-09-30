//! Adaptive residency's host share. The residency machine may send any stage
//! card expert past a layer's first `pinned` back to the host, so the host
//! must be able to serve each of them without a fault: that is the churn
//! pool, each layer's card segment's experts from rank `pinned` on (the
//! list's order, which is slot order). A load holds the pool in its host set
//! (`HostSet::of_with`, `bloomery-model`'s `placement::host_lock`) beside the
//! plan's host segments, and the plan's host headroom
//! ([`super::HostTotals::headroom_bytes`], the one budget) must take the
//! pool's bytes.

use super::{Device, ExpertList, PlacementError, Plan, Role, per_expert};

/// The churn pool of one stage card of a plan.
#[derive(Clone, Debug)]
pub struct ChurnPool {
    /// The seed experts of each layer that stay on the card.
    pub pinned: usize,
    /// Per routed stack the card holds: its index in the plan's model
    /// tensors, and its experts from rank `pinned` on.
    pub runs: Vec<(usize, ExpertList)>,
    /// Experts in the pool, summed over layers.
    pub experts: u64,
    /// File bytes of the pool: every stack of every expert in it.
    pub bytes: u64,
}

impl ChurnPool {
    /// The pool of `plan`'s card `card` (an index of
    /// [`super::Machine::all_cards`]) with `pinned` seed experts a layer kept
    /// on the card. A layer whose card holds `pinned` experts or fewer adds
    /// nothing.
    pub fn of(plan: &Plan<'_>, card: usize, pinned: usize) -> Result<ChurnPool, PlacementError> {
        let model = plan.model;
        let mut per_layer = vec![0u64; model.layers];
        let mut pool = ChurnPool {
            pinned,
            runs: Vec::new(),
            experts: 0,
            bytes: 0,
        };
        for row in &plan.rows {
            let t = model.tensors.get(row.tensor).ok_or_else(|| {
                PlacementError::Experts(format!("plan row {} names no tensor", row.tensor))
            })?;
            if t.role != Role::RoutedExperts {
                continue;
            }
            for seg in &row.segments {
                let Some(list) = seg
                    .experts
                    .as_ref()
                    .filter(|_| seg.device == Device::Card(card))
                else {
                    continue;
                };
                let Some(rest) = list.ids().get(pinned..).filter(|r| !r.is_empty()) else {
                    continue;
                };
                let (_, per) = per_expert(t, model.experts)?;
                let n = rest.len() as u64;
                pool.bytes += n * per;
                if let Some(slot) = t.layer.and_then(|l| per_layer.get_mut(l)) {
                    *slot = (*slot).max(n);
                }
                pool.runs
                    .push((row.tensor, ExpertList::new(rest.to_vec(), model.experts)?));
            }
        }
        pool.experts = per_layer.iter().sum();
        Ok(pool)
    }

    /// The plan's host headroom less the pool.
    #[must_use]
    pub fn headroom_after(&self, plan: &Plan<'_>) -> i128 {
        plan.host.headroom_bytes - i128::from(self.bytes)
    }

    /// The pool fits the plan's host headroom; refused by name when it does
    /// not ([`PlacementError::ResidencyOverHost`]). Returns the headroom left.
    pub fn check(&self, plan: &Plan<'_>) -> Result<i128, PlacementError> {
        let left = self.headroom_after(plan);
        if left < 0 {
            return Err(PlacementError::ResidencyOverHost(Box::new(OverHost {
                pinned: self.pinned,
                experts: self.experts,
                bytes: self.bytes,
                headroom: plan.host.headroom_bytes,
            })));
        }
        Ok(left)
    }
}

/// A churn pool the plan's host headroom cannot take
/// ([`PlacementError::ResidencyOverHost`]).
#[derive(Debug, thiserror::Error)]
#[error(
    "residency pinning {pinned} experts a layer: the churn pool's {experts} experts take {bytes} B \
     of host RAM, and the plan leaves the host {headroom} B"
)]
pub struct OverHost {
    pub pinned: usize,
    pub experts: u64,
    pub bytes: u64,
    pub headroom: i128,
}
