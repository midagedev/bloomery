//! The KV cache: the f16 latent rows every block has already computed, kept so that a
//! decode step is one query against a prefix instead of a whole re-prefill.
//!
//! Shape follows decision 2 in `lib.rs`: **a row is keyed by `(sequence, position)`,
//! two dimensions.** The slot table here is that key, and it is separate from the row
//! storage because a speculative branch writes k candidate positions and throws most of
//! them away — a cache whose only index is "append order" cannot express that, and is
//! the shape this crate refuses to grow into later.
//!
//! What a row is: one `kvr` column, `[k_rope(64) ; kv_compressed(512)]`, rounded to f16
//! exactly as `attn.rs` rounds it before attending. The cache stores what the reference
//! stores — ik's `kv_cache-N` is `f16 {576, n_kv}` in the oracle manifest — so caching
//! changes nothing numerically. `tests/kv.rs` asserts that as **bit equality**, not a
//! tolerance, and that is the whole safety argument for this file.
use crate::Slot;

/// One sequence's cached latent rows, all blocks.
///
/// The slot table is shared: every block caches the same positions in the same order,
/// because they are all fed the same tokens. Keeping one table rather than `n_block`
/// copies is what makes a desync impossible to represent rather than merely unlikely —
/// there is no second table to disagree with.
pub struct KvCache {
    width: usize,
    slots: Vec<Slot>,
    /// `rows[block][i]` is the f16 `kvr` row for `slots[i]`.
    rows: Vec<Vec<Vec<u16>>>,
}

impl KvCache {
    /// An empty cache for `n_block` blocks of `width`-wide rows (576 in this model:
    /// `rope_dims + latent`, read from the file by `MlaParams`, never a literal here).
    pub fn new(n_block: usize, width: usize) -> Self {
        Self {
            width,
            slots: Vec::new(),
            rows: vec![Vec::new(); n_block],
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn n_block(&self) -> usize {
        self.rows.len()
    }

    /// Every cached key's slot, in row order — what attention masks against.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// The rows block `b` has cached so far.
    pub fn keys(&self, b: usize) -> &[Vec<u16>] {
        &self.rows[b]
    }

    /// Reserve `new` positions and return the row range every block must now fill.
    ///
    /// Called once per forward pass, before the block loop. Appending the slots inside
    /// block 0 instead would make "block 0 ran" a precondition of the table being
    /// right, and a pass that failed halfway would leave the table ahead of the rows.
    pub fn begin(&mut self, new: &[Slot]) -> std::ops::Range<usize> {
        let base = self.slots.len();
        self.slots.extend_from_slice(new);
        base..self.slots.len()
    }

    /// Append block `b`'s rows for the range [`begin`] handed out.
    ///
    /// The asserts are the structural close, and they are `assert!`, not `debug_assert!`
    /// — every gate in this repo runs `--release`, where a `debug_assert` is a comment.
    /// A block that falls behind the slot table would otherwise attend against a row
    /// belonging to a different position and still produce plausible numbers.
    pub fn push(&mut self, b: usize, range: &std::ops::Range<usize>, rows: Vec<Vec<u16>>) {
        assert_eq!(
            self.rows[b].len(),
            range.start,
            "block {b} is at row {} but this pass starts at {}: a block was skipped or run twice",
            self.rows[b].len(),
            range.start
        );
        assert_eq!(
            rows.len(),
            range.len(),
            "block {b} produced {} rows for {} positions",
            rows.len(),
            range.len()
        );
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(
                r.len(),
                self.width,
                "block {b} row {i} is {} wide, cache is {}",
                r.len(),
                self.width
            );
        }
        self.rows[b].extend(rows);
        assert_eq!(
            self.rows[b].len(),
            self.slots.len(),
            "block {b} has {} rows against {} slots",
            self.rows[b].len(),
            self.slots.len()
        );
    }

    /// The next position for `seq`, i.e. how many of its tokens are already cached.
    ///
    /// Derived from the table rather than counted separately: one source of truth for
    /// "where are we", so a caller that appends out of order gets a wrong answer here
    /// instead of a silently wrong mask later.
    pub fn next_pos(&self, seq: u32) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.seq == seq)
            .map(|s| s.pos + 1)
            .max()
            .unwrap_or(0)
    }
}
