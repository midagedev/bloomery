//! The n-gram lookup draft: `lookup-recent` of `tools/ref/draft-accept.py`
//! (`score_recent`), streamed.
//!
//! For n = 3, 2, 1 in that order, the most recent earlier occurrence of the
//! context's last n tokens proposes the token that followed it; no match at
//! any n is no proposal. The n-grams ending at the last token enter the index
//! only when its follower is pushed, so a proposal never sees itself. As a
//! [`Draft`] it is fed the prompt, then only the tokens the passes keep: a
//! rejected proposal never enters the context, so it has nothing to take back.
//! Each generation starts it over ([`Draft::begin`]).

use std::collections::HashMap;

use crate::{Draft, TapNeed, Verify};

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

    /// The context so far: the prompt, then every kept token.
    #[must_use]
    pub fn context(&self) -> &[u32] {
        &self.ctx
    }

    /// An empty context again, the tables' memory kept.
    pub fn reset(&mut self) {
        self.idx3.clear();
        self.idx2.clear();
        self.idx1.clear();
        self.ctx.clear();
    }

    /// The context `ctx`, the tables rebuilt over it.
    pub fn rebuild(&mut self, ctx: &[u32]) {
        self.reset();
        for &tok in ctx {
            self.push(tok);
        }
    }

    /// The context's first `n` tokens, the tables rebuilt over them; nothing
    /// when it holds `n` or fewer.
    pub fn truncate(&mut self, n: usize) {
        if n >= self.ctx.len() {
            return;
        }
        let ctx = std::mem::take(&mut self.ctx);
        self.rebuild(&ctx[..n]);
    }

    /// The context made what it holds of a target's positions before `pos`,
    /// then `last`, the token at `pos`: what a target that stepped, passed or
    /// cut back since the context last followed it stands at. A context that
    /// stands there with another token (a sampled id where the argmax was
    /// pushed) takes `last` in its place: the token a context ends with is
    /// in no indexed n-gram yet, so the tables hold. One that holds fewer
    /// than `pos` tokens (the target ran positions it was not fed) keeps
    /// them all.
    pub fn stand(&mut self, pos: usize, last: u32) {
        if self.ctx.len() == pos + 1 {
            self.ctx[pos] = last;
            return;
        }
        self.truncate(pos);
        self.push(last);
    }
}

/// One id a proposal, from the token history alone.
impl<T: Verify> Draft<T> for Lookup {
    const WIDTH: usize = 1;
    const TAPS: TapNeed = TapNeed::None;

    /// The context starts over: this prompt, then `first`.
    fn begin(&mut self, _t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.rebuild(prompt);
        self.push(first);
        Ok(())
    }

    fn propose(&mut self, _t: &mut T, _last: u32, out: &mut [u32]) -> Result<usize, T::Error> {
        Ok(match Lookup::propose(self) {
            Some(d) => {
                out[0] = d;
                1
            }
            None => 0,
        })
    }

    fn accept(
        &mut self,
        _t: &mut T,
        _rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), T::Error> {
        for &tok in &out[..accepted] {
            self.push(tok);
        }
        Ok(())
    }

    fn stepped(&mut self, _t: &mut T, _last: u32, next: u32) -> Result<(), T::Error> {
        self.push(next);
        Ok(())
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

    /// A context that follows a target by [`Lookup::stand`] and its own
    /// pushes holds what a context rebuilt from the target's token history
    /// and the token it stands before holds, and proposes what that one
    /// proposes: through a prompt call, steps, kept passes and cuts back.
    #[test]
    fn stand_follows_history() {
        let t = stream(600);
        let rebuilt = |history: &[u32], next: u32| {
            let mut l = Lookup::new();
            l.rebuild(history);
            l.push(next);
            l
        };
        let mut look = Lookup::new();
        // The prompt call replaces an earlier request's context.
        look.rebuild(&t[300..500]);
        <Lookup as crate::Draft<crate::mock::Mock>>::begin(
            &mut look,
            &crate::mock::Mock::new(),
            &t[..200],
            t[200],
        )
        .unwrap();
        assert_eq!(look.context(), &t[..201]);
        assert_eq!(look.propose(), rebuilt(&t[..200], t[200]).propose());
        let mut pos = 200;
        // Steps and passes keeping 1 or 2 rows, a cut back every 25th.
        for i in 1..150 {
            let keep = 1 + pos % 2;
            look.stand(pos, t[pos]);
            t[pos + 1..=pos + keep].iter().for_each(|&id| look.push(id));
            pos += keep;
            if i % 25 == 0 {
                pos -= 7;
                look.stand(pos, t[pos]);
            }
            let r = rebuilt(&t[..pos], t[pos]);
            assert_eq!(look.context(), r.context());
            assert_eq!(look.propose(), r.propose());
        }
        // A sampled id where the argmax was pushed takes its place in the
        // context, the tables kept.
        let at = look.context().as_ptr();
        look.stand(pos, t[pos] + 1);
        let r = rebuilt(&t[..pos], t[pos] + 1);
        assert_eq!(look.context(), r.context());
        assert_eq!(look.propose(), r.propose());
        assert_eq!(look.context().as_ptr(), at, "the context was rebuilt");
        // A context behind the target keeps what it holds, then the token.
        let mut behind = rebuilt(&t[..100], t[100]);
        behind.stand(300, t[300]);
        assert_eq!(behind.context().len(), 102);
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
