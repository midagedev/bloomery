//! The host prompt cache: saved engine states ([`Saved`]) keyed by the ids
//! they cover, least recently used first, within a byte budget
//! ([`crate::Engine::cache_ram`]).
//!
//! A state is ranked by what it would keep of a request — its own
//! [`Saved::keepable`] of the ids it shares with the request, all but the
//! request's last — not by the shared ids alone: an engine may keep less of a
//! prefix than it holds. Taking a state back leaves it in the cache (it
//! becomes the most recent), so two requests that branch from one saved state
//! both find it. A new state removes every older one it keeps all of, and the
//! oldest states leave until the new one fits; a state larger than the whole
//! budget is not kept.

use std::sync::Arc;

use crate::engine::{CacheNote, Saved};

/// How many leading ids `a` and `b` share.
pub(crate) fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

struct Entry {
    ids: Vec<u32>,
    state: Arc<dyn Saved>,
}

/// The cached state a request is best served by ([`PromptCache::best`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pick {
    index: usize,
    /// The ids the state and the request share.
    pub common: usize,
    /// What the state keeps of them, the request's last id left out.
    pub kept: usize,
}

pub(crate) struct PromptCache {
    budget: u64,
    /// Least recently used first.
    entries: Vec<Entry>,
}

impl PromptCache {
    pub(crate) fn new(budget: u64) -> PromptCache {
        PromptCache {
            budget,
            entries: Vec::new(),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.budget > 0
    }

    /// The host bytes the states hold.
    pub(crate) fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.state.n_bytes()).sum()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The state that keeps the most of `ids` (a request of at least one id),
    /// if it keeps more than `floor`; of equals, the most recent.
    pub(crate) fn best(&self, ids: &[u32], floor: usize) -> Option<Pick> {
        let last = ids.len().checked_sub(1)?;
        let mut best: Option<Pick> = None;
        for (index, e) in self.entries.iter().enumerate().rev() {
            let common = common_prefix(&e.ids, ids);
            let kept = e.state.keepable(common.min(last)).min(common.min(last));
            if kept > best.map_or(floor, |b| b.kept) {
                best = Some(Pick {
                    index,
                    common,
                    kept,
                });
            }
        }
        best
    }

    /// `pick`'s ids and state, the entry made the most recent.
    pub(crate) fn take(&mut self, pick: Pick) -> (Vec<u32>, Arc<dyn Saved>) {
        let e = self.entries.remove(pick.index);
        let got = (e.ids.clone(), Arc::clone(&e.state));
        self.entries.push(e);
        got
    }

    /// Drops the entry that holds `state`, if the cache still does: a state
    /// the engine refused to take back.
    pub(crate) fn remove(&mut self, state: &Arc<dyn Saved>) {
        self.entries.retain(|e| !Arc::ptr_eq(&e.state, state));
    }

    /// Whether a state already keeps all of `ids`: they are a prefix of its
    /// ids and it keeps every one of them.
    pub(crate) fn covers(&self, ids: &[u32]) -> bool {
        self.entries.iter().any(|e| {
            e.ids.len() >= ids.len()
                && e.ids[..ids.len()] == *ids
                && e.state.keepable(ids.len()) == ids.len()
        })
    }

    /// Keeps `state` of `ids` (as many ids as it holds positions): the
    /// states it keeps all of leave, then the oldest until it fits. Returns
    /// what happened, in order; a state over the whole budget is refused
    /// (a `Skip`), and the cache is then as it was.
    pub(crate) fn insert(
        &mut self,
        ids: Vec<u32>,
        state: Arc<dyn Saved>,
        ms: f64,
    ) -> Vec<CacheNote> {
        let bytes = state.n_bytes();
        let positions = ids.len();
        if bytes > self.budget {
            return vec![CacheNote::Skip {
                positions,
                why: format!(
                    "its {bytes} bytes pass the cache's budget of {}",
                    self.budget
                ),
            }];
        }
        let mut notes = Vec::new();
        self.entries.retain(|e| {
            let covered = e.ids.len() <= ids.len()
                && ids[..e.ids.len()] == *e.ids
                && state.keepable(e.ids.len()) >= e.state.keepable(e.ids.len());
            if covered {
                notes.push(CacheNote::Evict {
                    positions: e.ids.len(),
                    bytes: e.state.n_bytes(),
                    why: "a newer state keeps all of it",
                });
            }
            !covered
        });
        while !self.entries.is_empty() && self.bytes() + bytes > self.budget {
            let e = self.entries.remove(0);
            notes.push(CacheNote::Evict {
                positions: e.ids.len(),
                bytes: e.state.n_bytes(),
                why: "the budget",
            });
        }
        self.entries.push(Entry { ids, state });
        notes.push(CacheNote::Save {
            positions,
            bytes,
            ms,
            entries: self.entries.len(),
            cache_bytes: self.bytes(),
        });
        notes
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use super::{PromptCache, common_prefix};
    use crate::engine::{CacheNote, Saved};

    /// A state of `n` positions and `bytes` bytes that keeps every position
    /// up to it except those at or past `hole` (when set), where it falls to 0.
    struct Fake {
        n: usize,
        bytes: u64,
        hole: Option<usize>,
    }

    impl Saved for Fake {
        fn n_tokens(&self) -> usize {
            self.n
        }
        fn n_bytes(&self) -> u64 {
            self.bytes
        }
        fn keepable(&self, n: usize) -> usize {
            match self.hole {
                Some(h) if n >= h && n < self.n => 0,
                _ => n.min(self.n),
            }
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn state(n: usize, bytes: u64, hole: Option<usize>) -> Arc<dyn Saved> {
        Arc::new(Fake { n, bytes, hole })
    }

    fn ids(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + seed).collect()
    }

    #[test]
    fn common_prefix_counts_the_shared_head() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
    }

    /// The pick is by what a state keeps, not by what it shares: a state
    /// sharing more ids inside its hole loses to one sharing fewer it keeps;
    /// the request's last id is never counted; `floor` must be beaten.
    #[test]
    fn best_ranks_by_the_kept_prefix() {
        let mut c = PromptCache::new(1 << 20);
        let a = ids(100, 1);
        let mut b = a[..60].to_vec();
        b.extend(ids(40, 1000));
        c.insert(a.clone(), state(100, 10, Some(10)), 0.0);
        c.insert(b.clone(), state(100, 10, None), 0.0);
        let mut req = a[..80].to_vec();
        req.push(5);
        let p = c.best(&req, 0).expect("a pick");
        assert_eq!((p.common, p.kept), (60, 60), "{p:?}");
        assert_eq!(c.best(&req, 60), None, "the floor is not beaten");
        let p = c.best(&a, 0).expect("a pick of the whole of a");
        assert_eq!((p.common, p.kept), (60, 60), "a keeps 0 of its own 99");
    }

    /// A new state removes the older ones it keeps all of, then the oldest
    /// until it fits; one over the budget is refused and changes nothing.
    #[test]
    fn insert_prunes_then_evicts_the_oldest() {
        let mut c = PromptCache::new(100);
        let a = ids(50, 1);
        c.insert(a[..20].to_vec(), state(20, 30, None), 0.0);
        c.insert(ids(30, 9), state(30, 30, None), 0.0);
        let notes = c.insert(a.clone(), state(50, 30, None), 0.0);
        assert!(
            matches!(notes[0], CacheNote::Evict { positions: 20, why, .. }
                if why == "a newer state keeps all of it")
        );
        assert_eq!(c.len(), 2);
        let notes = c.insert(ids(40, 77), state(40, 60, None), 0.0);
        assert!(
            matches!(
                notes[0],
                CacheNote::Evict {
                    positions: 30,
                    why: "the budget",
                    ..
                }
            ),
            "{notes:?}"
        );
        assert_eq!((c.len(), c.bytes()), (2, 90));
        let notes = c.insert(ids(10, 5), state(10, 101, None), 0.0);
        assert!(matches!(notes[..], [CacheNote::Skip { positions: 10, .. }]));
        assert_eq!((c.len(), c.bytes()), (2, 90));
    }

    /// A shorter state inside a newer one's hole stays: the newer one keeps
    /// less of it.
    #[test]
    fn insert_keeps_what_the_newer_state_cannot_keep() {
        let mut c = PromptCache::new(1 << 20);
        let a = ids(100, 3);
        c.insert(a[..40].to_vec(), state(40, 1, None), 0.0);
        c.insert(a.clone(), state(100, 1, Some(10)), 0.0);
        assert_eq!(c.len(), 2);
        assert!(c.covers(&a[..40]));
        assert!(!c.covers(&a[..60]), "the newer state keeps 0 of 60");
    }

    /// Taking a state back leaves it in the cache as the most recent.
    #[test]
    fn take_makes_the_state_the_most_recent() {
        let mut c = PromptCache::new(60);
        c.insert(ids(10, 1), state(10, 20, None), 0.0);
        c.insert(ids(10, 2), state(10, 20, None), 0.0);
        let p = c.best(&ids(10, 1), 0).expect("the first state");
        let (got, _) = c.take(p);
        assert_eq!(got, ids(10, 1));
        c.insert(ids(10, 3), state(10, 30, None), 0.0);
        assert!(
            c.best(&ids(10, 1), 0).is_some(),
            "the taken state was evicted"
        );
        assert!(c.best(&ids(10, 2), 0).is_none(), "the oldest state stayed");
    }
}
