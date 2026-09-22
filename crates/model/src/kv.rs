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
//! stores — ik's `kv_cache-N` is `f16 {576, n_kv}` in the oracle manifest — in the same
//! shape: ONE contiguous row-major buffer per block, fixed stride, so caching changes
//! nothing numerically and the attention walk sees the stride the reference sees.
//! `tests/kv.rs` asserts that as **bit equality**, not a
//! tolerance, and that is the whole safety argument for this file.
use crate::Slot;

/// Cached f16 key rows of one block: `len` rows of `width` u16, row-major, fixed stride.
///
/// A borrowed view (`Copy`) over the block's one flat buffer — attention takes it by
/// value and reads `row(i)`, so a cached row is a stride computation, never a second
/// pointer chase. Bounds and width are checked once at construction.
#[derive(Clone, Copy)]
pub struct KvRows<'a> {
    data: &'a [u16],
    width: usize,
}

impl<'a> KvRows<'a> {
    /// View `data` as rows of `width` u16. `width` must be nonzero and divide the
    /// length, or the row split below would be a lie.
    #[must_use]
    pub fn new(data: &'a [u16], width: usize) -> Self {
        assert!(width > 0, "KvRows: width must be nonzero");
        assert!(
            data.len().is_multiple_of(width),
            "KvRows: {} u16 is not a whole number of {}-wide rows",
            data.len(),
            width
        );
        Self { data, width }
    }

    /// Rows held — the block's cached count, what slots are masked against.
    pub fn len(&self) -> usize {
        self.data.len() / self.width
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The fixed row stride, `rope_dims + latent` in this model.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Row `i`'s f16 bits: `data[i*width..(i+1)*width]`, bounds-checked like a slice.
    pub fn row(&self, i: usize) -> &'a [u16] {
        &self.data[i * self.width..(i + 1) * self.width]
    }

    /// The whole flat buffer, row-major.
    pub fn as_slice(&self) -> &'a [u16] {
        self.data
    }
}

/// One sequence's cached latent rows, all blocks.
///
/// The slot table is shared: every block caches the same positions in the same order,
/// because they are all fed the same tokens. Keeping one table rather than `n_block`
/// copies is what makes a desync impossible to represent rather than merely unlikely —
/// there is no second table to disagree with.
pub struct KvCache {
    width: usize,
    slots: Vec<Slot>,
    /// Block `b`'s f16 `kvr` rows as ONE flat row-major buffer of `len * width` u16:
    /// row `i` is `rows[b][i*width..(i+1)*width]`, the row for `slots[i]` — the
    /// reference's own `kv_cache-N` shape (`f16 {width, n_kv}`), so the flash kernels
    /// walk a fixed stride instead of one heap row each.
    rows: Vec<Vec<u16>>,
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

    /// The rows block `b` has cached so far, as a view of its one flat buffer.
    pub fn keys(&self, b: usize) -> KvRows<'_> {
        KvRows::new(&self.rows[b], self.width)
    }

    /// Reserve `new` positions and return the row range every block must now fill.
    ///
    /// Called once per forward pass, before the block loop. Appending the slots inside
    /// block 0 instead would make "block 0 ran" a precondition of the table being
    /// right, and a pass that failed halfway would leave the table ahead of the rows.
    ///
    /// Every block's buffer is grown HERE, once per pass, geometrically: a steady
    /// decode step then appends into existing capacity instead of reallocating, so
    /// the alloc gate's ratchet never pays for the cache again.
    pub fn begin(&mut self, new: &[Slot]) -> std::ops::Range<usize> {
        let base = self.slots.len();
        self.slots.extend_from_slice(new);
        let add = new.len() * self.width;
        for buf in &mut self.rows {
            buf.reserve(add);
        }
        base..self.slots.len()
    }

    /// Append block `b`'s rows for the range [`begin`] handed out: one flat
    /// row-major buffer of `range.len() * width` u16, row `i` first.
    ///
    /// The asserts are the structural close, and they are `assert!`, not `debug_assert!`
    /// — every gate in this repo runs `--release`, where a `debug_assert` is a comment.
    /// A block that falls behind the slot table would otherwise attend against a row
    /// belonging to a different position and still produce plausible numbers.
    pub fn push(&mut self, b: usize, range: &std::ops::Range<usize>, rows: &[u16]) {
        assert_eq!(
            self.rows[b].len(),
            range.start * self.width,
            "block {b} is at row {} but this pass starts at {}: a block was skipped or run twice",
            self.rows[b].len() / self.width,
            range.start
        );
        // The flat length checks row count and width at once: the caller hands one
        // buffer, so "a row of the wrong width" is not a shape this input has.
        assert_eq!(
            rows.len(),
            range.len() * self.width,
            "block {b} produced {} rows of width {} for {} positions",
            rows.len() / self.width,
            self.width,
            range.len()
        );
        self.rows[b].extend_from_slice(rows);
        assert_eq!(
            self.rows[b].len(),
            self.slots.len() * self.width,
            "block {b} has {} rows against {} slots",
            self.rows[b].len() / self.width,
            self.slots.len()
        );
    }

    /// The next position for `seq`, i.e. how many of its tokens are already cached.
    ///
    /// Derived from the table rather than counted separately: one source of truth for
    /// "where are we", so a caller that appends out of order gets a wrong answer here
    /// instead of a silently wrong mask later.
    #[must_use]
    pub fn next_pos(&self, seq: u32) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.seq == seq)
            .map(|s| s.pos + 1)
            .max()
            .unwrap_or(0)
    }
}
