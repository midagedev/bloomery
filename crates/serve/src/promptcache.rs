//! The host prompt cache: saved engine states ([`Saved`]) keyed by what they
//! cover — the ids and the image spans among them ([`Held`]) — least recently
//! used first, within a byte budget ([`crate::Engine::cache_ram`]).
//!
//! A state is ranked by what it would keep of a request — its own
//! [`Saved::keepable`] of the positions it shares with the request, all but
//! the request's last — not by the shared positions alone: an engine may keep
//! less of a prefix than it holds, and a shared prefix never covers an image
//! the request does not carry ([`common_prefix`]). Taking a state back
//! leaves it in the cache (it becomes the most recent), so two requests that
//! branch from one saved state both find it. A new state removes every older
//! one it keeps all of, and the oldest states leave until the new one fits; a
//! state larger than the whole budget is not kept.
//!
//! A state may be held while its copy still runs ([`Pending`]): it counts
//! against the budget and ranks like any other, but it never leaves before
//! its copy has landed ([`PromptCache::settle`]) — not for the budget, not
//! because a newer state keeps all of it — since its staging may not be freed
//! or lent again while the copy writes it. A state that fits only by one
//! leaving is not kept, and the cache is then as it was.

use std::sync::Arc;

use crate::engine::{CacheNote, Pending, Saved};
use crate::media::{Held, common_prefix, keep_whole_spans};

struct Entry {
    held: Held,
    state: Arc<dyn Saved>,
    /// The copy of `state` while it may still run.
    pending: Option<Pending>,
    /// The warm section the request that left `state` closed its residency
    /// stream into ([`crate::Engine::warm_close`]).
    warm: Option<Arc<[u8]>>,
}

impl Entry {
    /// The host bytes the entry holds: the state's and the warm section's.
    fn bytes(&self) -> u64 {
        entry_bytes(&self.state, self.warm.as_deref())
    }
}

fn entry_bytes(state: &Arc<dyn Saved>, warm: Option<&[u8]>) -> u64 {
    state.n_bytes() + warm.map_or(0, |w| w.len() as u64)
}

/// What [`PromptCache::take`] hands back: a state, the ids it holds, and its
/// copy while it may still run — to wait for before the state is read.
pub(crate) struct Taken {
    pub held: Held,
    pub state: Arc<dyn Saved>,
    pub pending: Option<Pending>,
    /// The entry's warm section, if the request that left it closed one.
    pub warm: Option<Arc<[u8]>>,
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

    /// The host bytes the entries hold, their warm sections included.
    pub(crate) fn bytes(&self) -> u64 {
        self.entries.iter().map(Entry::bytes).sum()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The state that keeps the most of `req` (a request of at least one id),
    /// if it keeps more than `floor`; of equals, the most recent.
    pub(crate) fn best(&self, req: &Held, floor: usize) -> Option<Pick> {
        let last = req.ids.len().checked_sub(1)?;
        let mut best: Option<Pick> = None;
        for (index, e) in self.entries.iter().enumerate().rev() {
            let common = common_prefix(&e.held, req);
            let kept = keep_whole_spans(req, common.min(last), |n| e.state.keepable(n));
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

    /// `pick`'s held sequence and state, the entry made the most recent.
    pub(crate) fn take(&mut self, pick: Pick) -> Taken {
        let e = self.entries.remove(pick.index);
        let got = Taken {
            held: e.held.clone(),
            state: Arc::clone(&e.state),
            pending: e.pending.clone(),
            warm: e.warm.clone(),
        };
        self.entries.push(e);
        got
    }

    /// The copies that may still run, oldest entry first.
    pub(crate) fn in_flight(&self) -> Vec<Pending> {
        self.entries
            .iter()
            .filter_map(|e| e.pending.clone())
            .collect()
    }

    /// `pending`'s copy landed: its entry may leave from here on. False when
    /// the cache no longer holds the entry.
    pub(crate) fn settle(&mut self, pending: &Pending) -> bool {
        let mut found = false;
        for e in &mut self.entries {
            if e.pending.as_ref().is_some_and(|p| p.is(pending)) {
                e.pending = None;
                found = true;
            }
        }
        found
    }

    /// Drops the entry that holds `state`, if the cache still does: a state
    /// the engine refused to take back.
    pub(crate) fn remove(&mut self, state: &Arc<dyn Saved>) {
        self.entries.retain(|e| !Arc::ptr_eq(&e.state, state));
    }

    /// Whether a state already keeps all of `req`: its held sequence extends
    /// `req`'s and it keeps every one of its positions.
    pub(crate) fn covers(&self, req: &Held) -> bool {
        self.entries
            .iter()
            .any(|e| e.held.starts_with(req) && e.state.keepable(req.ids.len()) == req.ids.len())
    }

    /// Keeps `state` of `held` (as many ids as it holds positions) with its
    /// warm section `warm`, whose length counts against the budget with the
    /// state's: the entries it keeps all of leave, then the oldest until it
    /// fits. Returns what happened, in order, the `Save` carrying `ms` and
    /// `copied` ([`CacheNote::Save`]); an entry over the whole budget is
    /// refused (a `Skip`), and the cache is then as it was.
    pub(crate) fn insert(
        &mut self,
        held: Held,
        state: Arc<dyn Saved>,
        warm: Option<Arc<[u8]>>,
        ms: f64,
        copied: bool,
    ) -> Vec<CacheNote> {
        let entry = Entry {
            held,
            state,
            pending: None,
            warm,
        };
        self.place(entry, ms, copied, None)
    }

    /// Keeps `state` of `held` as [`PromptCache::insert`] does, a snapshot
    /// taken on the request path because the engine's staging pool had no
    /// room for a copy off it: the `Save` names why (`fallback`,
    /// [`CacheNote::Save`]).
    pub(crate) fn insert_fallback(
        &mut self,
        held: Held,
        state: Arc<dyn Saved>,
        warm: Option<Arc<[u8]>>,
        ms: f64,
        fallback: &'static str,
    ) -> Vec<CacheNote> {
        let entry = Entry {
            held,
            state,
            pending: None,
            warm,
        };
        self.place(entry, ms, true, Some(fallback))
    }

    /// The oldest entry whose copy has landed leaves, for the room its
    /// staging gives back: an entry whose copy still runs never does, since
    /// that copy writes the staging. `None`, the cache as it was, when every
    /// entry's copy still runs or the cache holds none.
    pub(crate) fn evict_oldest_landed(&mut self) -> Option<CacheNote> {
        let i = self.entries.iter().position(|e| e.pending.is_none())?;
        let e = self.entries.remove(i);
        Some(CacheNote::Evict {
            positions: e.held.ids.len(),
            bytes: e.bytes(),
            why: "the engine's staging pool had no room",
        })
    }

    /// Keeps `pending`'s state of `held` as [`PromptCache::insert`] does,
    /// the copy still running: the entry stays until it is
    /// [settled](PromptCache::settle). A state that fits only by an entry
    /// whose copy still runs leaving is refused (a `Skip`), the cache as it
    /// was. `ms` is the call that began the copy.
    pub(crate) fn insert_pending(
        &mut self,
        held: Held,
        pending: Pending,
        warm: Option<Arc<[u8]>>,
        ms: f64,
    ) -> Vec<CacheNote> {
        let state = Arc::clone(pending.state());
        let entry = Entry {
            held,
            state,
            pending: Some(pending),
            warm,
        };
        self.place(entry, ms, true, None)
    }

    fn place(
        &mut self,
        entry: Entry,
        ms: f64,
        copied: bool,
        fallback: Option<&'static str>,
    ) -> Vec<CacheNote> {
        let Entry {
            held,
            state,
            pending,
            warm,
        } = entry;
        let bytes = entry_bytes(&state, warm.as_deref());
        let positions = held.ids.len();
        if bytes > self.budget {
            return vec![CacheNote::Skip {
                positions,
                why: format!(
                    "its {bytes} bytes pass the cache's budget of {}",
                    self.budget
                ),
            }];
        }
        // What leaves, decided before anything does: the entries the new
        // state keeps all of, then the oldest, none of them one whose copy
        // still runs.
        let mut gone: Vec<(usize, &'static str)> = Vec::new();
        let mut held_bytes = self.bytes();
        for (i, e) in self.entries.iter().enumerate() {
            let covered = e.pending.is_none()
                && held.starts_with(&e.held)
                && state.keepable(e.held.ids.len()) >= e.state.keepable(e.held.ids.len());
            if covered {
                gone.push((i, "a newer state keeps all of it"));
                held_bytes -= e.bytes();
            }
        }
        for (i, e) in self.entries.iter().enumerate() {
            if held_bytes + bytes <= self.budget {
                break;
            }
            if e.pending.is_none() && !gone.iter().any(|&(g, _)| g == i) {
                gone.push((i, "the budget"));
                held_bytes -= e.bytes();
            }
        }
        if held_bytes + bytes > self.budget {
            return vec![CacheNote::Skip {
                positions,
                why: format!(
                    "its {bytes} bytes fit the cache's budget of {} only past the states \
                     whose copies still run",
                    self.budget
                ),
            }];
        }
        let mut notes = Vec::new();
        for &(i, why) in &gone {
            let e = &self.entries[i];
            notes.push(CacheNote::Evict {
                positions: e.held.ids.len(),
                bytes: e.bytes(),
                why,
            });
        }
        let mut leaving: Vec<usize> = gone.iter().map(|&(i, _)| i).collect();
        leaving.sort_unstable_by(|a, b| b.cmp(a));
        for i in leaving {
            self.entries.remove(i);
        }
        let off_path = pending.is_some();
        self.entries.push(Entry {
            held,
            state,
            pending,
            warm,
        });
        notes.push(CacheNote::Save {
            positions,
            bytes,
            ms,
            entries: self.entries.len(),
            cache_bytes: self.bytes(),
            copied,
            off_path,
            fallback,
        });
        notes
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use super::PromptCache;
    use crate::engine::{CacheNote, Flight, Landed, Pending, Saved, StateError};
    use crate::media::{Held, ImageKey, MediaSpan};

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

    /// A state of `n` positions that keeps multiples of four.
    struct Fours(usize);

    impl Saved for Fours {
        fn n_tokens(&self) -> usize {
            self.0
        }
        fn n_bytes(&self) -> u64 {
            1
        }
        fn keepable(&self, n: usize) -> usize {
            n.min(self.0) / 4 * 4
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn ids(n: usize, seed: u32) -> Held {
        Held::from((0..n as u32).map(|i| i * 7 + seed).collect::<Vec<_>>())
    }

    /// The first `n` positions of `h`.
    fn head(h: &Held, n: usize) -> Held {
        let mut h = h.clone();
        h.truncate(n);
        h
    }

    /// Ten text ids, an image of four positions and the key `[key; 32]`, then
    /// `tail`.
    fn imaged(key: u8, tail: &[u32]) -> Held {
        let mut ids: Vec<u32> = (0..10).collect();
        ids.extend([7; 4]);
        ids.extend_from_slice(tail);
        Held {
            ids,
            media: vec![MediaSpan {
                at: 10,
                len: 4,
                key: ImageKey([key; 32]),
            }],
        }
    }

    /// An entry's bytes against the budget are its state's and its warm
    /// section's; the section leaves with the entry, nothing else moving.
    #[test]
    fn an_entry_counts_its_warm_section_and_drops_it_on_eviction() {
        let mut c = PromptCache::new(100);
        let section: Arc<[u8]> = Arc::from(vec![7u8; 30]);
        let kept = Arc::clone(&section);
        let notes = c.insert(ids(10, 1), state(10, 50, None), Some(section), 0.0, true);
        assert!(matches!(
            notes.last(),
            Some(CacheNote::Save { bytes: 80, .. })
        ));
        assert_eq!(c.bytes(), 80);
        assert_eq!(Arc::strong_count(&kept), 2);
        // 80 + 40 passes the budget: the entry leaves for the next one.
        let notes = c.insert(ids(10, 2), state(10, 40, None), None, 0.0, true);
        assert!(
            notes
                .iter()
                .any(|n| matches!(n, CacheNote::Evict { bytes: 80, .. }))
        );
        assert_eq!((c.len(), c.bytes()), (1, 40));
        assert_eq!(Arc::strong_count(&kept), 1);
    }

    /// The pick is by what a state keeps, not by what it shares: a state
    /// sharing more ids inside its hole loses to one sharing fewer it keeps;
    /// the request's last id is never counted; `floor` must be beaten.
    #[test]
    fn best_ranks_by_the_kept_prefix() {
        let mut c = PromptCache::new(1 << 20);
        let a = ids(100, 1);
        let mut b = head(&a, 60);
        b.ids.extend(ids(40, 1000).ids);
        c.insert(a.clone(), state(100, 10, Some(10)), None, 0.0, true);
        c.insert(b.clone(), state(100, 10, None), None, 0.0, true);
        let mut req = head(&a, 80);
        req.push(5);
        let p = c.best(&req, 0).expect("a pick");
        assert_eq!((p.common, p.kept), (60, 60), "{p:?}");
        assert_eq!(c.best(&req, 60), None, "the floor is not beaten");
        let p = c.best(&a, 0).expect("a pick of the whole of a");
        assert_eq!((p.common, p.kept), (60, 60), "a keeps 0 of its own 99");
    }

    /// Two states of one prompt with two images of one size: the request's
    /// image picks the state that holds it whole, another image keeps up to
    /// the span's start, and a state whose keep would end inside the span
    /// keeps none of it.
    #[test]
    fn best_needs_the_same_image_whole() {
        let mut c = PromptCache::new(1 << 20);
        let tail: Vec<u32> = (100..116).collect();
        let a = imaged(1, &tail);
        c.insert(a.clone(), state(30, 10, None), None, 0.0, true);
        c.insert(imaged(2, &tail), state(30, 10, None), None, 0.0, true);
        let mut req = a.clone();
        req.push(5);
        let p = c.best(&req, 0).expect("a pick");
        assert_eq!((p.common, p.kept), (30, 30), "{p:?}");
        assert_eq!(c.take(p).held, a, "the state of the request's image");
        let p = c.best(&imaged(3, &tail), 0).expect("a pick");
        assert_eq!(
            (p.common, p.kept),
            (10, 10),
            "another image: the span's start"
        );
        assert!(c.covers(&a));
        assert!(!c.covers(&imaged(3, &tail)), "another image is not covered");
        let mut c = PromptCache::new(1 << 20);
        c.insert(a.clone(), Arc::new(Fours(30)), None, 0.0, true);
        let p = c.best(&imaged(1, &[200, 201]), 0).expect("a pick");
        assert_eq!(
            (p.common, p.kept),
            (14, 8),
            "14 rounds to 12 inside 10..14; 10 rounds to 8"
        );
    }

    /// A new state removes the older ones it keeps all of, then the oldest
    /// until it fits; one over the budget is refused and changes nothing.
    #[test]
    fn insert_prunes_then_evicts_the_oldest() {
        let mut c = PromptCache::new(100);
        let a = ids(50, 1);
        c.insert(head(&a, 20), state(20, 30, None), None, 0.0, true);
        c.insert(ids(30, 9), state(30, 30, None), None, 0.0, true);
        let notes = c.insert(a.clone(), state(50, 30, None), None, 0.0, true);
        assert!(
            matches!(notes[0], CacheNote::Evict { positions: 20, why, .. }
                if why == "a newer state keeps all of it")
        );
        assert_eq!(c.len(), 2);
        let notes = c.insert(ids(40, 77), state(40, 60, None), None, 0.0, true);
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
        let notes = c.insert(ids(10, 5), state(10, 101, None), None, 0.0, true);
        assert!(matches!(notes[..], [CacheNote::Skip { positions: 10, .. }]));
        assert_eq!((c.len(), c.bytes()), (2, 90));
    }

    /// A shorter state inside a newer one's hole stays: the newer one keeps
    /// less of it.
    #[test]
    fn insert_keeps_what_the_newer_state_cannot_keep() {
        let mut c = PromptCache::new(1 << 20);
        let a = ids(100, 3);
        c.insert(head(&a, 40), state(40, 1, None), None, 0.0, true);
        c.insert(a.clone(), state(100, 1, Some(10)), None, 0.0, true);
        assert_eq!(c.len(), 2);
        assert!(c.covers(&head(&a, 40)));
        assert!(!c.covers(&head(&a, 60)), "the newer state keeps 0 of 60");
    }

    /// A copy that runs until `land` is called.
    #[derive(Default)]
    struct Lands(std::sync::atomic::AtomicBool);

    impl Flight for Lands {
        fn poll(&self) -> Result<Option<Landed>, StateError> {
            let done = self.0.load(std::sync::atomic::Ordering::SeqCst);
            Ok(done.then(Landed::default))
        }
        fn wait(&self) -> Result<Landed, StateError> {
            Ok(Landed::default())
        }
    }

    /// A state of `n` positions and `bytes` bytes whose copy runs.
    fn running(n: usize, bytes: u64) -> Pending {
        Pending::new(state(n, bytes, None), Arc::new(Lands::default()))
    }

    /// A state whose copy still runs is never evicted — not for the budget,
    /// not because a newer state keeps all of it — and a state that fits only
    /// by its leaving is refused with the cache as it was; once the copy has
    /// landed it leaves like any other.
    #[test]
    fn a_state_in_flight_never_leaves_before_its_copy_lands() {
        let mut c = PromptCache::new(100);
        let a = ids(50, 1);
        let p = running(50, 60);
        let notes = c.insert_pending(a.clone(), p.clone(), None, 0.0);
        assert!(
            matches!(
                notes[..],
                [CacheNote::Save {
                    off_path: true,
                    entries: 1,
                    ..
                }]
            ),
            "{notes:?}"
        );
        // The budget needs its bytes: 60 + 60 > 100, and the only state is in flight.
        let notes = c.insert(ids(30, 9), state(30, 60, None), None, 0.0, true);
        assert!(
            matches!(&notes[..], [CacheNote::Skip { positions: 30, why }] if why.contains("copies still run")),
            "{notes:?}"
        );
        assert_eq!((c.len(), c.bytes()), (1, 60), "the cache is as it was");
        // A newer state that keeps all of it: it stays while the copy runs.
        let mut longer = a.clone();
        longer.ids.extend(ids(10, 5).ids);
        let notes = c.insert(longer.clone(), state(60, 30, None), None, 0.0, true);
        assert!(
            notes.iter().all(|n| !matches!(n, CacheNote::Evict { .. })),
            "the in-flight state was pruned: {notes:?}"
        );
        assert_eq!(c.len(), 2);
        assert_eq!(c.in_flight().len(), 1);
        // Landed, it leaves for the budget like any other.
        assert!(c.settle(&p));
        assert!(c.in_flight().is_empty());
        let notes = c.insert(ids(30, 9), state(30, 60, None), None, 0.0, true);
        assert!(
            matches!(
                notes[0],
                CacheNote::Evict {
                    positions: 50,
                    why: "the budget",
                    ..
                }
            ),
            "{notes:?}"
        );
    }

    /// The state that leaves for the staging pool's room is the oldest one
    /// whose copy has landed: never one whose copy still runs, even when it is
    /// the oldest; with none landed, nothing leaves.
    #[test]
    fn the_room_made_for_a_copy_is_the_oldest_landed_state_never_one_in_flight() {
        let mut c = PromptCache::new(1000);
        assert!(c.evict_oldest_landed().is_none(), "an empty cache");
        let p = running(10, 20);
        c.insert_pending(ids(10, 1), p.clone(), None, 0.0);
        c.insert(ids(11, 2), state(11, 20, None), None, 0.0, true);
        c.insert(ids(12, 3), state(12, 20, None), None, 0.0, true);
        for positions in [11, 12] {
            assert!(
                matches!(
                    c.evict_oldest_landed(),
                    Some(CacheNote::Evict { positions: n, bytes: 20, .. }) if n == positions
                ),
                "the oldest landed state is next ({positions} positions)"
            );
        }
        assert_eq!(c.len(), 1, "the state in flight stayed");
        assert!(
            c.evict_oldest_landed().is_none(),
            "only a copy in flight is left"
        );
        assert_eq!((c.len(), c.in_flight().len()), (1, 1));
        assert!(c.settle(&p));
        assert!(
            matches!(
                c.evict_oldest_landed(),
                Some(CacheNote::Evict { positions: 10, .. })
            ),
            "landed, it leaves like any other"
        );
    }

    /// A pick of a state in flight hands its copy back, to wait for before
    /// the state is read; a state that landed hands none.
    #[test]
    fn a_pick_hands_back_the_copy_still_running() {
        let mut c = PromptCache::new(100);
        let p = running(10, 20);
        c.insert_pending(ids(10, 1), p.clone(), None, 0.0);
        c.insert(ids(10, 2), state(10, 20, None), None, 0.0, true);
        let pick = c.best(&ids(10, 1), 0).expect("the first state");
        let taken = c.take(pick);
        assert!(taken.pending.as_ref().is_some_and(|q| q.is(&p)));
        let pick = c.best(&ids(10, 2), 0).expect("the second state");
        assert!(c.take(pick).pending.is_none());
    }

    /// Taking a state back leaves it in the cache as the most recent.
    #[test]
    fn take_makes_the_state_the_most_recent() {
        let mut c = PromptCache::new(60);
        c.insert(ids(10, 1), state(10, 20, None), None, 0.0, true);
        c.insert(ids(10, 2), state(10, 20, None), None, 0.0, true);
        let p = c.best(&ids(10, 1), 0).expect("the first state");
        let got = c.take(p).held;
        assert_eq!(got, ids(10, 1));
        c.insert(ids(10, 3), state(10, 30, None), None, 0.0, true);
        assert!(
            c.best(&ids(10, 1), 0).is_some(),
            "the taken state was evicted"
        );
        assert!(c.best(&ids(10, 2), 0).is_none(), "the oldest state stayed");
    }
}
