//! The n-gram lookup draft `generate_ds41` serves under `BLOOMERY_DRAFT=lookup`:
//! `lookup-recent` of `tools/ref/draft-accept.py` (`score_recent`), streamed.
//!
//! For n = 3, 2, 1 in that order, the most recent earlier occurrence of the
//! context's last n tokens proposes the token that followed it; no match at
//! any n is no proposal. The n-grams ending at the last token enter the index
//! only when its follower is pushed, so a proposal never sees itself.

use std::collections::HashMap;

/// The streamed lookup: the context so far and, per n, where each n-gram's
/// most recent follower sits in it.
#[derive(Default)]
pub struct Lookup {
    idx3: HashMap<[u32; 3], usize>,
    idx2: HashMap<[u32; 2], usize>,
    idx1: HashMap<u32, usize>,
    ctx: Vec<u32>,
}

impl Lookup {
    /// An empty context.
    pub fn new() -> Lookup {
        Lookup::default()
    }

    /// Append `tok` to the context. The n-grams that end just before it are
    /// indexed with `tok` as their most recent follower first — the update
    /// `score_recent` makes after scoring position `i`.
    pub fn push(&mut self, tok: u32) {
        let i = self.ctx.len();
        if i >= 3 {
            self.idx3
                .insert([self.ctx[i - 3], self.ctx[i - 2], self.ctx[i - 1]], i);
        }
        if i >= 2 {
            self.idx2.insert([self.ctx[i - 2], self.ctx[i - 1]], i);
        }
        if i >= 1 {
            self.idx1.insert(self.ctx[i - 1], i);
        }
        self.ctx.push(tok);
    }

    /// The token proposed to follow the context, from the longest n in 3, 2, 1
    /// whose last-n n-gram occurred earlier; `None` when none did.
    #[must_use]
    pub fn propose(&self) -> Option<u32> {
        let c = &self.ctx;
        let i = c.len();
        let j = (i >= 3)
            .then(|| self.idx3.get(&[c[i - 3], c[i - 2], c[i - 1]]))
            .flatten()
            .or_else(|| {
                (i >= 2)
                    .then(|| self.idx2.get(&[c[i - 2], c[i - 1]]))
                    .flatten()
            })
            .or_else(|| (i >= 1).then(|| self.idx1.get(&c[i - 1])).flatten())?;
        Some(c[*j])
    }
}

#[cfg(test)]
mod tests {
    use super::Lookup;

    /// The synthetic stream: a u32 LCG over a small vocabulary with a
    /// five-token phrase spliced in, so n = 3, 2 and 1 all match. The same
    /// generator in Python wrote the file `draft-accept.py` scored.
    fn stream(n: usize) -> Vec<u32> {
        let mut s: u32 = 1;
        let mut out = Vec::with_capacity(n + 5);
        while out.len() < n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = s >> 16;
            if x.is_multiple_of(7) {
                out.extend_from_slice(&[40, 41, 42, 43, 44]);
            } else {
                out.push(x % 11);
            }
        }
        out.truncate(n);
        out
    }

    /// `draft-accept.py`'s lookup-recent over the same 4000 tokens as one
    /// document: 3999 positions, 3983 proposed, 1644 accepted.
    #[test]
    fn matches_draft_accept_py() {
        let t = stream(4000);
        let mut look = Lookup::new();
        look.push(t[0]);
        let (mut positions, mut proposed, mut accepted) = (0, 0, 0);
        for &tok in &t[1..] {
            positions += 1;
            if let Some(d) = look.propose() {
                proposed += 1;
                if d == tok {
                    accepted += 1;
                }
            }
            look.push(tok);
        }
        assert_eq!((positions, proposed, accepted), (3999, 3983, 1644));
    }

    /// The first pattern of `draft-accept.py --self-test`: seven distinct ids
    /// repeated; position 7 has no proposal, every later one is right.
    #[test]
    fn repeated_pattern() {
        let pat = [11, 12, 13, 14, 15, 16, 17];
        let mut look = Lookup::new();
        let mut hits = Vec::new();
        for (i, &tok) in pat.iter().cycle().take(70).enumerate() {
            if i > 0 {
                hits.push(look.propose().map(|d| d == tok));
            }
            look.push(tok);
        }
        assert!(hits[..7].iter().all(Option::is_none));
        assert!(hits[7..].iter().all(|h| *h == Some(true)));
    }
}
