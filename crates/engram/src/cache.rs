//! A DRAM hot-row cache in front of the helper thread: exact LRU, one budget
//! shared by every site and every bucket.
//!
//! The rows a real stream asks for repeat — the heads of one n-gram order
//! re-hit the rows a repeated n-gram produced — so a budget of a few GiB can
//! serve most of a token from memory. What the cache changes is who does the
//! work: a **hit** is the step thread's own copy out of the cache, with no
//! syscall and no fault; only a **miss** goes to the helper of
//! [`crate::prefetch::Prefetcher`], and a token with none skips it entirely.
//!
//! Three shapes the caller has to know about:
//!
//! * **The order is the contract.** Per token, sites in order, and within a
//!   site the buckets in [`crate::Hash::rows_into`] order — the order
//!   `engram-reuse` feeds its simulator, so the cache's hits are the hits
//!   [`crate::reuse::Lru`] reports at the same capacity, to the integer.
//! * **A miss owns its slot at once.** [`RowCache::lookup`] claims the slot,
//!   evicting the least recently used row when the cache is full, and marks it
//!   pending; [`RowCache::complete`] lands the helper's copy. Every miss of one
//!   token is completed before the next token's lookup, and a lookup that would
//!   read a pending slot is an error, never a stale row.
//! * **Nothing allocates after construction.** The payload slab, the index and
//!   the LRU order are sized and written once; a token only overwrites them.
//!   [`RowCache::allocated_bytes`] is the witness a gate compares before and
//!   after a run.
//!
//! [`LruIndex`] is the same cache without the payload: the keys, the slots they
//! own and the order they were used in. [`RowCache`] decides its hits with one,
//! and alone it replays a stream at a capacity whose payload a gate could not
//! afford to allocate.

use crate::EngramError;

/// One site's row id as a single key. Sites are separate tables, so the same
/// row number in two of them is two different rows.
pub fn key_of(site: usize, id: u32) -> u64 {
    ((site as u64) << 32) | u64::from(id)
}

/// Rows a byte budget holds: `budget / row_bytes`, rounded down — the same
/// arithmetic `engram-reuse` prints beside each capacity. Zero for a zero row
/// size, which every constructor here refuses.
pub fn capacity_rows(budget_bytes: u64, row_bytes: u64) -> u64 {
    budget_bytes.checked_div(row_bytes).unwrap_or(0)
}

/// "No slot". Slots are `u32` and a capacity that would reach this value is
/// refused, so no live slot is ever `NIL`.
const NIL: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    /// `NIL` when the entry holds no key, so every `u64` is a legal key.
    slot: u32,
}

/// Key to slot: open addressing, linear probing, at most half full.
///
/// Sized once and never resized. Removal moves the rest of the probe run back
/// over the hole instead of leaving a tombstone, so a full cache that evicts on
/// every miss keeps the same table, the same load and the same probe lengths
/// for as long as it runs. A std `HashMap` under the same churn accumulates
/// tombstones until it rehashes or grows — an allocation, or a stall of the
/// whole table, on the step thread.
struct Table {
    entries: Vec<Entry>,
    /// `entries.len() - 1`; the length is a power of two.
    mask: usize,
    /// `64 - log2(entries.len())`: a key's home is the top bits of its product.
    shift: u32,
}

impl Table {
    fn new(keys: usize) -> Table {
        let len = keys.saturating_mul(2).next_power_of_two().max(2);
        Table {
            // Written, not just allocated: `NIL` is not zero, so every page is
            // touched here and not by the first insert that lands in it.
            entries: vec![Entry { key: 0, slot: NIL }; len],
            mask: len - 1,
            shift: 64 - len.trailing_zeros(),
        }
    }

    /// Fibonacci hashing: the multiply carries every key bit into the top bits.
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// The slot `key` owns, if it is in the table. The table is never more
    /// than half full, so every probe run ends at a free entry.
    fn get(&self, key: u64) -> Option<u32> {
        let mut i = self.home(key);
        loop {
            let e = self.entries[i];
            if e.slot == NIL {
                return None;
            }
            if e.key == key {
                return Some(e.slot);
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Give `key`, which is not in the table, the first free entry of its run.
    fn insert(&mut self, key: u64, slot: u32) {
        let mut i = self.home(key);
        while self.entries[i].slot != NIL {
            i = (i + 1) & self.mask;
        }
        self.entries[i] = Entry { key, slot };
    }

    /// Take `key` out and close the hole behind it.
    ///
    /// An entry later in the run may move back into the hole only if its home
    /// is not inside `(hole, i]` — its probe then still meets the hole before
    /// `i` — and the hole moves to where it was. The run ends at a free entry,
    /// and that is where the last hole is freed.
    fn remove(&mut self, key: u64) {
        let mut hole = self.home(key);
        loop {
            let e = self.entries[hole];
            if e.slot == NIL {
                return;
            }
            if e.key == key {
                break;
            }
            hole = (hole + 1) & self.mask;
        }
        let mut i = hole;
        loop {
            i = (i + 1) & self.mask;
            let e = self.entries[i];
            if e.slot == NIL {
                break;
            }
            let from_home = i.wrapping_sub(self.home(e.key)) & self.mask;
            if from_home >= (i.wrapping_sub(hole) & self.mask) {
                self.entries[hole] = e;
                hole = i;
            }
        }
        self.entries[hole].slot = NIL;
    }
}

/// What one [`LruIndex::access`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// The key was cached in this slot, which is now the most recently used.
    Hit(usize),
    /// The key was not cached. It now owns this slot as the most recently
    /// used: a fresh slot while the index fills, the least recently used
    /// key's once it is full.
    Miss(usize),
}

/// Exact LRU over keys only: which slot each cached key owns, and the order
/// the keys were last used in, as a doubly linked list threaded through the
/// slots.
pub struct LruIndex {
    table: Table,
    /// The key each claimed slot holds, so an eviction can find it in `table`.
    keys: Vec<u64>,
    /// Toward the most recently used; `NIL` at the head.
    prev: Vec<u32>,
    /// Toward the least recently used; `NIL` at the tail.
    next: Vec<u32>,
    /// Most recently used slot.
    head: u32,
    /// Least recently used slot: the next one a full index evicts.
    tail: u32,
    /// Slots claimed so far. They are claimed in order, so `0..used` are live.
    used: u32,
    cap: u32,
}

impl LruIndex {
    /// An empty index of `capacity` slots. Every buffer is allocated and
    /// written here; nothing after this allocates.
    pub fn new(capacity: usize) -> Result<LruIndex, EngramError> {
        if capacity == 0 {
            return Err(EngramError::Cache("an LRU of no rows holds nothing"));
        }
        let cap = u32::try_from(capacity)
            .map_err(|_| EngramError::Cache("a capacity must fit a u32 slot number"))?;
        Ok(LruIndex {
            table: Table::new(capacity),
            // Neither sentinel is zero, so these are written here too.
            keys: vec![u64::MAX; capacity],
            prev: vec![NIL; capacity],
            next: vec![NIL; capacity],
            head: NIL,
            tail: NIL,
            used: 0,
            cap,
        })
    }

    /// Slots in the index: the most keys it holds at once.
    pub fn capacity(&self) -> usize {
        self.keys.len()
    }

    /// Keys held now: every distinct key seen until the index fills, then its
    /// capacity.
    pub fn len(&self) -> usize {
        self.used as usize
    }

    /// No key held yet.
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Look `key` up and make it the most recently used, claiming a slot for
    /// it on a miss.
    pub fn access(&mut self, key: u64) -> Access {
        if let Some(slot) = self.table.get(key) {
            if slot != self.head {
                self.unlink(slot);
                self.push_front(slot);
            }
            return Access::Hit(slot as usize);
        }
        let slot = if self.used < self.cap {
            self.used += 1;
            self.used - 1
        } else {
            let lru = self.tail;
            self.unlink(lru);
            self.table.remove(self.keys[lru as usize]);
            lru
        };
        self.keys[slot as usize] = key;
        self.table.insert(key, slot);
        self.push_front(slot);
        Access::Miss(slot as usize)
    }

    /// Bytes this index holds, from its buffers' capacities.
    pub fn allocated_bytes(&self) -> usize {
        bytes_of(&self.table.entries)
            + bytes_of(&self.keys)
            + bytes_of(&self.prev)
            + bytes_of(&self.next)
    }

    fn unlink(&mut self, slot: u32) {
        let (p, n) = (self.prev[slot as usize], self.next[slot as usize]);
        if p == NIL {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NIL {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
    }

    fn push_front(&mut self, slot: u32) {
        self.prev[slot as usize] = NIL;
        self.next[slot as usize] = self.head;
        if self.head == NIL {
            self.tail = slot;
        } else {
            self.prev[self.head as usize] = slot;
        }
        self.head = slot;
    }
}

fn bytes_of<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}

/// A slot's state, one byte each. None of them is zero, so the vector holding
/// them is written when the cache is built.
const FREE: u8 = 1;
const READY: u8 = 2;
const PENDING: u8 = 3;

/// One miss of the token in flight: the slot it claimed, and the row of the
/// token's output that the helper's copy of it lands in.
#[derive(Clone, Copy)]
struct Miss {
    slot: usize,
    out_row: usize,
}

/// What one token's [`RowCache::lookup`] found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lookup {
    pub hits: usize,
    pub misses: usize,
}

/// The cache: an [`LruIndex`] deciding hits, and a slab holding each slot's
/// row.
///
/// A token's output is every site's rows in site order, each site's in bucket
/// order, `row_bytes` apiece — the same layout the helper fills when it is
/// handed a whole token.
pub struct RowCache {
    index: LruIndex,
    row_bytes: usize,
    /// Rows each site asks per token.
    rows_per_site: Vec<usize>,
    token_bytes: usize,
    /// `capacity × row_bytes`, slot-major.
    slab: Box<[u8]>,
    /// Per slot: `FREE`, `READY`, or `PENDING` between a miss and its fill.
    state: Vec<u8>,
    /// This token's misses, in the cache's order.
    misses: Vec<Miss>,
    /// The same misses' row ids, per site: what the helper is handed.
    miss_ids: Vec<Vec<u32>>,
}

impl RowCache {
    /// A cache of `capacity` rows of `row_bytes` each, for tokens that ask
    /// `rows_per_site[s]` rows of site `s`.
    ///
    /// The capacity must hold at least one whole token: a smaller cache would
    /// evict a row the same token claimed a moment earlier and is still
    /// waiting on.
    pub fn new(
        capacity: usize,
        row_bytes: usize,
        rows_per_site: &[usize],
    ) -> Result<RowCache, EngramError> {
        if row_bytes == 0 {
            return Err(EngramError::Cache("a row must have at least one byte"));
        }
        let per_token: usize = rows_per_site.iter().sum();
        if per_token == 0 {
            return Err(EngramError::Cache("a token must ask for at least one row"));
        }
        if capacity < per_token {
            return Err(EngramError::Cache(
                "the capacity must hold at least one token's rows",
            ));
        }
        let slab_len = capacity
            .checked_mul(row_bytes)
            .ok_or(EngramError::Cache("capacity x row size overflows"))?;
        Ok(RowCache {
            index: LruIndex::new(capacity)?,
            row_bytes,
            rows_per_site: rows_per_site.to_vec(),
            token_bytes: per_token * row_bytes,
            // Written, not just allocated: a hit's copy out of the slab and a
            // fill into it run on the step thread, which must not take the
            // allocator's first-touch faults.
            slab: vec![0xA5u8; slab_len].into_boxed_slice(),
            state: vec![FREE; capacity],
            misses: Vec::with_capacity(per_token),
            miss_ids: rows_per_site
                .iter()
                .map(|&n| Vec::with_capacity(n))
                .collect(),
        })
    }

    /// Rows the cache holds at most.
    pub fn capacity(&self) -> usize {
        self.index.capacity()
    }

    /// Bytes of one token's output.
    pub fn token_bytes(&self) -> usize {
        self.token_bytes
    }

    /// Look up one token's rows: `ids[s]` are site `s`'s row ids in bucket
    /// order, and `out` is one token's output.
    ///
    /// A hit is copied into `out` now. A miss claims its slot, and its id is
    /// listed at [`RowCache::miss_ids`] for the helper; [`RowCache::complete`]
    /// lands it. Until then this cache refuses the next lookup.
    ///
    /// A token that names one row twice is a caller bug. If the first name
    /// missed, the second would read its unfilled slot — a stale read — so the
    /// lookup is refused there, and the misses claimed before it stay pending.
    /// If the first name hit, both are served from the same ready slot.
    pub fn lookup(&mut self, ids: &[Vec<u32>], out: &mut [u8]) -> Result<Lookup, EngramError> {
        if !self.misses.is_empty() {
            return Err(EngramError::Cache(
                "lookup before the previous token's misses were completed",
            ));
        }
        if ids.len() != self.rows_per_site.len() {
            return Err(EngramError::Cache("token has a different site count"));
        }
        if ids
            .iter()
            .zip(&self.rows_per_site)
            .any(|(ids, &n)| ids.len() != n)
        {
            return Err(EngramError::Cache(
                "a site's ids are not the rows per site the cache was built for",
            ));
        }
        if out.len() != self.token_bytes {
            return Err(EngramError::Cache("output is not one token's rows"));
        }

        let rb = self.row_bytes;
        let mut hits = 0;
        let mut out_row = 0;
        for (site, site_ids) in ids.iter().enumerate() {
            for &id in site_ids {
                match self.index.access(key_of(site, id)) {
                    Access::Hit(slot) => {
                        if self.state[slot] != READY {
                            return Err(EngramError::Cache("a lookup found its row still pending"));
                        }
                        out[out_row * rb..][..rb].copy_from_slice(&self.slab[slot * rb..][..rb]);
                        hits += 1;
                    }
                    Access::Miss(slot) => {
                        self.state[slot] = PENDING;
                        self.misses.push(Miss { slot, out_row });
                        self.miss_ids[site].push(id);
                    }
                }
                out_row += 1;
            }
        }
        Ok(Lookup {
            hits,
            misses: self.misses.len(),
        })
    }

    /// The row ids the last lookup missed, per site in that lookup's order:
    /// what [`crate::prefetch::Prefetcher::submit`] takes. Every site is empty
    /// when it hit everything.
    pub fn miss_ids(&self) -> &[Vec<u32>] {
        &self.miss_ids
    }

    /// Land the helper's copy of the last lookup's misses.
    ///
    /// `filled` is every missed row in [`RowCache::miss_ids`] order, site by
    /// site, `row_bytes` each — what [`crate::prefetch::Prefetcher::filled`]
    /// returns for that submit. Each row goes into the slot it claimed and
    /// into `out`, and the slot stops being pending.
    pub fn complete(&mut self, filled: &[u8], out: &mut [u8]) -> Result<(), EngramError> {
        let rb = self.row_bytes;
        if filled.len() != self.misses.len() * rb {
            return Err(EngramError::Cache(
                "the filled rows are not the pending misses",
            ));
        }
        if out.len() != self.token_bytes {
            return Err(EngramError::Cache("output is not one token's rows"));
        }
        for (m, row) in self.misses.iter().zip(filled.chunks_exact(rb)) {
            self.slab[m.slot * rb..][..rb].copy_from_slice(row);
            out[m.out_row * rb..][..rb].copy_from_slice(row);
            self.state[m.slot] = READY;
        }
        self.misses.clear();
        for ids in &mut self.miss_ids {
            ids.clear();
        }
        Ok(())
    }

    /// Bytes this cache holds, from its buffers' capacities: the index, the
    /// slab and the per-token lists. A gate compares it before and after a run.
    pub fn allocated_bytes(&self) -> usize {
        self.index.allocated_bytes()
            + self.slab.len()
            + bytes_of(&self.rows_per_site)
            + bytes_of(&self.state)
            + bytes_of(&self.misses)
            + bytes_of(&self.miss_ids)
            + self.miss_ids.iter().map(bytes_of).sum::<usize>()
    }
}
