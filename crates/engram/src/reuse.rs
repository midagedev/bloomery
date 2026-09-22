//! Exact LRU for every capacity at once, by stack distance — and the token
//! streams it is fed.
//!
//! This is the simulator `engram-reuse` reports from, and the second witness
//! the row cache's gates hold [`crate::cache::LruIndex`] to: two algorithms
//! that share nothing but the definition of LRU, agreeing to the integer.
//!
//! For each access the **stack distance** is counted — how many distinct rows
//! were touched since this row was last touched — with a Fenwick tree carrying
//! a 1 at each row's most recent position. An LRU of `C` rows serves the access
//! iff that distance is `< C`, so one distance per access answers every
//! capacity consistently, and a first touch (distance = none) is a compulsory
//! miss at every size.

use std::collections::HashMap;
use std::path::Path;

use crate::EngramError;

/// What one slice of the stream asked for.
#[derive(Clone, Debug)]
pub struct Slice {
    pub requests: u64,
    /// Accesses whose stack distance was under each capacity, in the order
    /// [`Lru::new`] was given the capacities.
    pub hits: Vec<u64>,
    /// Accesses that had been seen before at all.
    pub rehits: u64,
}

impl Slice {
    fn new(caps: usize) -> Slice {
        Slice {
            requests: 0,
            hits: vec![0; caps],
            rehits: 0,
        }
    }
}

/// One distinct row: the position of its most recent access, how often it was
/// asked for, and the slice it belongs to. One map serves both the distance
/// and the frequency tail.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub last: usize,
    pub count: u32,
    pub site: usize,
    pub order: usize,
}

/// Exact LRU for every capacity at once, by stack distance.
pub struct Lru {
    seen: HashMap<u64, Row>,
    /// A 1 at each row's most recent position; the prefix sum between two
    /// positions is the count of distinct rows touched in between.
    fenwick: Vec<u32>,
    pos: usize,
    /// Keyed by (site, n-gram order index), filled as the slices are met.
    slices: HashMap<(usize, usize), Slice>,
    caps: Vec<u64>,
    total: Slice,
}

impl Lru {
    /// A simulator for at most `accesses` accesses, answering every capacity
    /// in `caps` (in rows) at once. The tree is sized here, once.
    pub fn new(accesses: usize, caps: &[u64]) -> Lru {
        Lru {
            seen: HashMap::new(),
            fenwick: vec![0; accesses + 1],
            pos: 0,
            slices: HashMap::new(),
            caps: caps.to_vec(),
            total: Slice::new(caps.len()),
        }
    }

    /// One access of `key`, counted under `(site, order)`.
    ///
    /// More accesses than [`Lru::new`] was sized for is an error: the mark of
    /// the extra access would fall past the end of the tree and every later
    /// distance would be short.
    pub fn access(&mut self, key: u64, site: usize, order: usize) -> Result<(), EngramError> {
        if self.pos + 1 >= self.fenwick.len() {
            return Err(EngramError::Reuse(
                "more accesses than the simulator was sized for",
            ));
        }
        self.pos += 1;
        let pos = self.pos;
        let prev = match self.seen.get_mut(&key) {
            Some(row) => {
                row.count += 1;
                let prev = row.last;
                row.last = pos;
                Some(prev)
            }
            None => {
                self.seen.insert(
                    key,
                    Row {
                        last: pos,
                        count: 1,
                        site,
                        order,
                    },
                );
                None
            }
        };

        // Distinct rows touched strictly between the two accesses: each carries
        // exactly one mark, at its own most recent position. The tree is read
        // before either mark moves.
        let distance = prev.map(|prev| {
            u64::from(
                self.sum(pos - 1)
                    .checked_sub(self.sum(prev))
                    .expect("a row's mark sits at its own latest position, inside the span"),
            )
        });

        let n = self.caps.len();
        let slice = self
            .slices
            .entry((site, order))
            .or_insert_with(|| Slice::new(n));
        slice.requests += 1;
        self.total.requests += 1;
        if let Some(distance) = distance {
            slice.rehits += 1;
            self.total.rehits += 1;
            for (i, &cap) in self.caps.iter().enumerate() {
                if distance < cap {
                    slice.hits[i] += 1;
                    self.total.hits[i] += 1;
                }
            }
        }

        if let Some(prev) = prev {
            self.add(prev, -1);
        }
        self.add(pos, 1);
        Ok(())
    }

    /// The capacities, in rows, in the order every [`Slice::hits`] keeps them.
    pub fn caps(&self) -> &[u64] {
        &self.caps
    }

    /// Every access so far.
    pub fn total(&self) -> &Slice {
        &self.total
    }

    /// The accesses counted under `(site, order)`, if there were any.
    pub fn slice(&self, site: usize, order: usize) -> Option<&Slice> {
        self.slices.get(&(site, order))
    }

    /// Every distinct row seen so far, in no particular order.
    pub fn rows(&self) -> impl Iterator<Item = &Row> {
        self.seen.values()
    }

    /// Distinct rows seen so far: an unbounded cache's compulsory misses.
    pub fn distinct(&self) -> usize {
        self.seen.len()
    }

    fn add(&mut self, mut i: usize, delta: i32) {
        while i < self.fenwick.len() {
            self.fenwick[i] = self.fenwick[i].wrapping_add_signed(delta);
            i += i.isolate_lowest_one();
        }
    }

    fn sum(&self, mut i: usize) -> u32 {
        let mut s = 0u32;
        while i > 0 {
            s = s.wrapping_add(self.fenwick[i]);
            i -= i.isolate_lowest_one();
        }
        s
    }
}

/// A token stream as `tools/ref/engram-corpus.sh` writes it: one decimal id
/// per line, blank lines ignored. Text rather than packed `u32` so the file
/// greps, diffs and truncates like everything else in `$BLOOMERY_DATA`.
pub fn read_ids(path: impl AsRef<Path>) -> Result<Vec<u32>, EngramError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| EngramError::IdsFile {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(line.parse::<u32>().map_err(|source| EngramError::TokenId {
            path: path.to_path_buf(),
            line: n + 1,
            text: line.to_string(),
            source,
        })?);
    }
    Ok(out)
}
