//! The V4.1 candidate mask's rule on the host, for the gates that check the
//! engine's wiring of it on real select paths: `gate_deepseek41_chain_attn`'s
//! G2s on synthetic index keys, `gate_deepseek41_long --candidates` on the
//! live model. Transcribed here, not taken from the kernel crate: a rule the
//! gate shared with the kernels would move with them (`gate_cand` pins the
//! kernels themselves).
//!
//! One token's check ([`check`]) reads what the card left: the source layer's
//! scores and the kept blocks it wrote, the consumer's scores before its
//! compaction (the piece's armed tap), after it, and the consumer's list. It
//! holds them to the reference's rule exactly:
//!
//! - the kept blocks are `select_candidate_blocks` over the card's own
//!   source scores — a block's key its largest visible score, the newest
//!   block pinned, the best others by (key descending, block ascending),
//!   `−0` and `+0` equal — in ascending order;
//! - the compaction moved each candidate row's score to its slot;
//! - the consumer's list is the exact top-k of its scores over the kept
//!   blocks' rows, by (key descending, row ascending), ascending; where the
//!   token does not select, the exact top-k over every visible row.
//!
//! Besides the verdicts it reports what the data lets a check see: whether
//! the pin would have fallen out without it, whether equal keys straddle
//! the last kept place, and how many list entries the mask changed — the
//! gates require each somewhere, so data that cannot see a broken pin, a
//! broken tie order or a consumer that ignores the mask fails by name.

/// The order a score ranks in: an unsigned key that orders as the value
/// does, `−0` and `+0` equal.
pub fn key_of(v: f32) -> u32 {
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// The `k` best of `rows` by the key of `scores[r]` descending, row
/// ascending, in ascending order.
pub fn top_rows(scores: &[f32], rows: &[usize], k: usize) -> Vec<u32> {
    let mut order = rows.to_vec();
    order.sort_by(|&a, &b| key_of(scores[b]).cmp(&key_of(scores[a])).then(a.cmp(&b)));
    let mut top: Vec<u32> = order[..k.min(order.len())]
        .iter()
        .map(|&r| u32::try_from(r).expect("a row index of a u32 count"))
        .collect();
    top.sort_unstable();
    top
}

/// The rule's kept blocks of a token of `n` visible rows whose source scores
/// are `scores[..n]`, `None` where it does not select (`⌈n / block⌉ <=
/// blocks`); with the pin's and the tie's visibility (module doc).
pub struct Kept {
    pub kept: Vec<u32>,
    /// Without the pin, `blocks` other blocks would rank at or above the
    /// newest: a list that drops the pin differs from the rule here.
    pub pin_below: bool,
    /// Equal keys sit on both sides of the last kept place among the others:
    /// a list that breaks ties to the higher block differs here.
    pub tie: bool,
}

pub fn kept_rule(scores: &[f32], n: usize, blocks: usize, block: usize) -> Option<Kept> {
    let nb = n.div_ceil(block);
    if nb <= blocks {
        return None;
    }
    let keys: Vec<u32> = (0..nb)
        .map(|b| {
            scores[b * block..((b + 1) * block).min(n)]
                .iter()
                .map(|&v| key_of(v))
                .max()
                .expect("a block of at least one row")
        })
        .collect();
    let pin = nb - 1;
    let mut others: Vec<usize> = (0..pin).collect();
    others.sort_by(|&a, &b| keys[b].cmp(&keys[a]).then(a.cmp(&b)));
    let take = blocks - 1;
    let mut kept: Vec<u32> = others[..take]
        .iter()
        .chain(std::iter::once(&pin))
        .map(|&b| u32::try_from(b).expect("a block index of a u32 count"))
        .collect();
    kept.sort_unstable();
    let pin_below = others.iter().filter(|&&b| keys[b] >= keys[pin]).count() >= blocks;
    let tie = take > 0
        && others
            .get(take)
            .is_some_and(|&b| keys[b] == keys[others[take - 1]]);
    Some(Kept {
        kept,
        pin_below,
        tie,
    })
}

/// One token's verdict ([`check`]).
pub struct Row {
    pub n: usize,
    pub selects: bool,
    pub kept_ok: bool,
    pub compact_ok: bool,
    pub list_ok: bool,
    pub pin_below: bool,
    pub tie: bool,
    /// Entries of the rule's list not in the unmasked top-k.
    pub changed: usize,
}

impl Row {
    pub fn pass(&self) -> bool {
        self.kept_ok && self.compact_ok && self.list_ok
    }

    pub fn line(&self) -> String {
        format!(
            "n={} selects={} kept {} compaction {} list {} | pin below the boundary {} tie at \
             the last kept place {} entries the mask changed {}",
            self.n,
            self.selects,
            ok(self.kept_ok),
            ok(self.compact_ok),
            ok(self.list_ok),
            self.pin_below,
            self.tie,
            self.changed
        )
    }
}

fn ok(b: bool) -> &'static str {
    if b { "ok" } else { "WRONG" }
}

/// What the card left for one token of a pass whose source and consumer
/// selected over the same `n` visible rows.
pub struct Card<'a> {
    pub n: usize,
    pub top_k: usize,
    pub blocks: usize,
    pub block: usize,
    /// The source layer's scores, `n` of them.
    pub source: &'a [f32],
    /// The source's kept blocks, `blocks` entries.
    pub kept: &'a [u32],
    /// The consumer's scores before its compaction, and after it.
    pub unmasked: &'a [f32],
    pub compacted: &'a [f32],
    /// The consumer's list, at least `min(top_k, n)` entries.
    pub list: &'a [u32],
}

/// One token's check against the rule (module doc).
pub fn check(c: &Card<'_>) -> Row {
    let all: Vec<usize> = (0..c.n).collect();
    let plain = top_rows(c.unmasked, &all, c.top_k);
    let (selects, kept_ok, compact_ok, want, pin_below, tie) =
        match kept_rule(c.source, c.n, c.blocks, c.block) {
            None => (
                false,
                true,
                c.compacted[..c.n]
                    .iter()
                    .zip(&c.unmasked[..c.n])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                plain.clone(),
                false,
                false,
            ),
            Some(rule) => {
                let kept = c.kept[..c.blocks].to_vec();
                let kept_ok = kept == rule.kept;
                // The candidate rows of the card's own kept blocks, in slot
                // order: compaction and list are held to the list the card
                // made, the list itself to the rule above.
                let rows: Vec<usize> = kept
                    .iter()
                    .flat_map(|&b| {
                        let b = b as usize;
                        (b * c.block..((b + 1) * c.block).min(c.n)).collect::<Vec<_>>()
                    })
                    .collect();
                let in_range = rows.iter().all(|&r| r < c.n);
                let compact_ok = in_range
                    && rows
                        .iter()
                        .enumerate()
                        .all(|(s, &r)| c.compacted[s].to_bits() == c.unmasked[r].to_bits());
                let want = if in_range {
                    top_rows(c.unmasked, &rows, c.top_k)
                } else {
                    Vec::new()
                };
                (true, kept_ok, compact_ok, want, rule.pin_below, rule.tie)
            }
        };
    let k = want.len();
    let list_ok = k > 0 && c.list.len() >= k && c.list[..k] == want[..];
    let changed = want.iter().filter(|r| !plain.contains(r)).count();
    Row {
        n: c.n,
        selects,
        kept_ok,
        compact_ok,
        list_ok,
        pin_below,
        tie,
        changed,
    }
}
