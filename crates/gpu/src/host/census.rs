//! The tier census ([`TierCensus`]): where a span's routed picks were
//! served — the stage card, the NVMe tier's arena or the model file's
//! mapping — and the bytes the span moved between the tiers: the arena's
//! fills (file → arena), the residency machine's copies onto the card
//! (promotions) and the experts it sent back to the host (demotions). Beside
//! them, the tier's drops of the mapping's pages, and the pool's dispatches
//! that waited for another caller's job. Each term is read from its owner:
//! the services' columns and host slots ([`super::HybridStats`]), the host
//! computation's reads by source ([`HostReads`]), the arena's counters
//! ([`super::nvtier::NvTierStats`]), the machine's map words
//! ([`super::swap::SwapMachine::moved_bytes`]) and the pool's
//! ([`threads::PoolStats`]). The counts run from the load on; a span is the
//! difference of two reads ([`TierCensus::since`]). The card's picks are the
//! services' picks less the host slots they listed, so the arena's and the
//! file's, counted by the host computation alone, sum with them to the picks
//! only when every host slot was read once by one source ([`TierCensus::adds_up`]).

use super::HybridStats;
use super::nvtier::{NvTier, NvTierStats};

/// The host computation's routed picks by where it read their bytes,
/// counted a pick — a slot of a column's host list — whatever the call
/// shares between its columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostReads {
    /// Picks of ids the NVMe tier serves, read from the arena's slots.
    pub arena: u64,
    /// Picks read through the model file's mapping: the plan's host
    /// segment's, and every pick of a union that reads the mapping alone.
    pub file: u64,
}

impl HostReads {
    /// One call's picks `ids` of layer `layer` by a reader that takes the
    /// ids `tier` serves from the arena and every other through the mapping.
    pub fn arena_or_file(
        &mut self,
        tier: Option<&NvTier>,
        layer: usize,
        ids: impl Iterator<Item = u32>,
    ) {
        for id in ids {
            if tier.is_some_and(|t| t.serves(layer, id)) {
                self.arena += 1;
            } else {
                self.file += 1;
            }
        }
    }

    /// One call's `n` picks by a reader of the mapping alone.
    pub fn file(&mut self, n: usize) {
        self.file += n as u64;
    }
}

/// One span's tier census ([`TierCensus::of`] since the load,
/// [`TierCensus::since`] between two reads).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierCensus {
    /// The tokens the span's host services carried: the columns its
    /// services served over the slot map's layers, a column a token at
    /// every layer — exact on a plan whose every layer of the map the host
    /// serves each walk, as a paged plan's, whose every layer pages ids.
    pub tokens: u64,
    /// The routed picks the services saw: the routed width a column.
    pub picks: u64,
    /// Of those, the ones the slot map sent to a card: the stage card's, a
    /// tier card's, and a prompt's host slots left out for the card route.
    pub card: u64,
    /// Of those, the ones the host computation read from the arena's slots.
    pub arena: u64,
    /// Of those, the ones it read through the mapping.
    pub file: u64,
    /// The bytes the arena's fills read from the file.
    pub fill_bytes: u64,
    /// The bytes of the experts the residency machine put on the stage card,
    /// and of those it sent back to the host, at the map's change.
    pub promote_bytes: u64,
    pub demote_bytes: u64,
    /// The tier's drops of the mapping's pages, and their wall on the
    /// dropper's thread (ns).
    pub drops: u64,
    pub drop_ns: u64,
    /// The pool's dispatches that waited for another caller's job, and their
    /// wait (ns).
    pub dispatch_waits: u64,
    pub dispatch_wait_ns: u64,
}

impl TierCensus {
    /// The census since the load from its owners' counts: the host tier's
    /// `stats` at its routed width `n_used` over its `layers` hybrid layers,
    /// the host computation's `reads`, the arena's `tier` counters, the
    /// machine's `moved` bytes (promoted, demoted) and the `pool`'s.
    #[must_use]
    pub fn of(
        stats: &HybridStats,
        n_used: usize,
        layers: usize,
        reads: HostReads,
        tier: &NvTierStats,
        moved: (u64, u64),
        pool: &threads::PoolStats,
    ) -> TierCensus {
        let step_cols = stats.served - stats.cols_served + stats.cols_cols;
        let cols = step_cols + stats.batch_cols;
        let picks = cols * n_used as u64;
        TierCensus {
            tokens: cols / (layers.max(1) as u64),
            picks,
            card: picks.saturating_sub(stats.host_slots + stats.batch_host_slots),
            arena: reads.arena,
            file: reads.file,
            fill_bytes: tier.fill_bytes,
            promote_bytes: moved.0,
            demote_bytes: moved.1,
            drops: tier.drops,
            drop_ns: tier.drop_ns,
            dispatch_waits: pool.dispatch_waits,
            dispatch_wait_ns: pool.dispatch_wait_ns,
        }
    }

    /// The span from `before` to `self`, field by field.
    #[must_use]
    pub fn since(&self, before: &TierCensus) -> TierCensus {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        TierCensus {
            tokens: d(self.tokens, before.tokens),
            picks: d(self.picks, before.picks),
            card: d(self.card, before.card),
            arena: d(self.arena, before.arena),
            file: d(self.file, before.file),
            fill_bytes: d(self.fill_bytes, before.fill_bytes),
            promote_bytes: d(self.promote_bytes, before.promote_bytes),
            demote_bytes: d(self.demote_bytes, before.demote_bytes),
            drops: d(self.drops, before.drops),
            drop_ns: d(self.drop_ns, before.drop_ns),
            dispatch_waits: d(self.dispatch_waits, before.dispatch_waits),
            dispatch_wait_ns: d(self.dispatch_wait_ns, before.dispatch_wait_ns),
        }
    }

    /// Whether the span's card, arena and file picks sum to its picks: every
    /// host slot the services listed was read once, from one source.
    #[must_use]
    pub fn adds_up(&self) -> bool {
        self.card + self.arena + self.file == self.picks
    }

    /// The census's values in the `tier census` record's order, by name.
    #[must_use]
    pub fn fields(&self) -> [(&'static str, u64); 12] {
        [
            ("tokens", self.tokens),
            ("picks", self.picks),
            ("card", self.card),
            ("arena", self.arena),
            ("file", self.file),
            ("fill_bytes", self.fill_bytes),
            ("promote_bytes", self.promote_bytes),
            ("demote_bytes", self.demote_bytes),
            ("drops", self.drops),
            ("drop_ns", self.drop_ns),
            ("dispatch_waits", self.dispatch_waits),
            ("dispatch_wait_ns", self.dispatch_wait_ns),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{HostReads, TierCensus};

    /// A span's census sums when the host computation read every host slot
    /// the services listed once, and not when a reader left its picks
    /// uncounted; the span is the difference of two reads.
    #[test]
    fn a_census_adds_up_only_with_every_host_slot_read() {
        let before = TierCensus {
            tokens: 1,
            picks: 8,
            card: 6,
            arena: 1,
            file: 1,
            ..TierCensus::default()
        };
        let mut after = TierCensus {
            tokens: 3,
            picks: 24,
            card: 15,
            arena: 7,
            file: 2,
            ..TierCensus::default()
        };
        let span = after.since(&before);
        assert_eq!(
            (span.tokens, span.picks, span.card, span.arena, span.file),
            (2, 16, 9, 6, 1)
        );
        assert!(span.adds_up(), "{span:?}");
        after.arena -= 2;
        assert!(
            !after.since(&before).adds_up(),
            "two picks no reader counted"
        );
        let mut r = HostReads::default();
        r.arena_or_file(None, 0, [3u32, 4].into_iter());
        r.file(2);
        assert_eq!(
            r,
            HostReads { arena: 0, file: 4 },
            "no tier: every pick the mapping's"
        );
    }
}
