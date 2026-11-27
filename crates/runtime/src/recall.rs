//! The recall draft: the continuation of the longest earlier match of the
//! context's tail, up to [`Draft::WIDTH`] ids, from the prompt and the ids
//! the passes kept. It reads no weights, so what a proposal costs is the
//! verify rows it asks for and nothing else. `score_longest` of
//! `tools/ref/draft-accept.py` is its oracle, streamed.
//!
//! The rule. For each tracked n-gram length from the longest down, the most
//! recent earlier occurrence of the context's last n ids; the first length
//! that has one decides. Its match is widened backward over the ids that
//! still agree, to [`CAP`], and that length is the proposal's confidence; the
//! proposal is the ids that followed the occurrence. Where that run reaches
//! the end of the context it goes on repeating itself, the context extended
//! with the proposal, so a match at distance `p` proposes a period of `p`
//! and a proposal always fills the room it is given. The shortest tracked
//! length is the shortest match that proposes anything. The n-grams ending at
//! the last id enter the index only when its follower is pushed, so a
//! proposal never sees itself. As a [`Draft`] it is fed the prompt, then
//! only the ids the passes keep: a rejected proposal never enters the
//! context, so it has nothing to take back. Each generation starts it over
//! ([`Draft::begin`]).
//!
//! The index is one hash table a tracked length, n-gram to the index of its
//! most recent follower, each updated once an id: a push costs the lengths'
//! sum in hashed ids and a proposal the same in lookups plus the widening, a
//! constant the context's length does not move. A suffix automaton gives the
//! longest match at the same cost but not the most recent occurrence of it
//! without a walk up its link tree. A candidate is checked against the
//! context, so a hash collision can lose a proposal and never make one.
//!
//! [`Draft::propose_p`] reports each id's confidence as `(l + i) / (l + i +
//! DOUBT)`, `l` the match length and `i` the id's index in the proposal: a
//! match is `l` confirmations against [`DOUBT`] doubts, and each id the
//! proposal has run without a miss is one more. The width chooser's
//! calibration scales that to the text it is on, so it rates a window by its
//! own match length and a whole text by what the windows keep.

use std::collections::HashMap;
use std::fmt;

use crate::width::Rule;
use crate::{Draft, TapNeed, Verify};

/// The longest match a proposal reports: a match is widened backward over
/// the ids that still agree, and no further.
const CAP: usize = 32;

/// The doubts a match's confirmations are weighed against.
const DOUBT: f32 = 2.0;

/// The tracked lengths no valid list holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecallError {
    /// No n-gram length to track.
    NoGram,
    /// Lengths that are not strictly ascending from at least 1.
    Grams(Vec<usize>),
}

impl fmt::Display for RecallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecallError::NoGram => f.write_str("recall draft: no n-gram length to track"),
            RecallError::Grams(g) => write!(
                f,
                "recall draft: n-gram lengths are strictly ascending from at least 1, not {g:?}"
            ),
        }
    }
}

impl std::error::Error for RecallError {}

/// An earlier occurrence of the context's tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matched {
    /// The ids of the tail the occurrence agrees on: its tracked n-gram
    /// widened backward, at most [`CAP`].
    pub len: usize,
    /// The index in the context of the id that followed the occurrence.
    pub at: usize,
}

/// The recall draft of at most `W` ids a proposal.
#[derive(Debug)]
pub struct Recall<const W: usize> {
    /// The tracked n-gram lengths, strictly ascending.
    grams: Vec<usize>,
    /// Per tracked length, the hash of an n-gram to the index of the id that
    /// followed its most recent occurrence.
    index: Vec<HashMap<u64, usize>>,
    ctx: Vec<u32>,
}

/// One splitmix64 round: a bijection of the word.
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The hash an n-gram is indexed by.
fn mix(gram: &[u32]) -> u64 {
    gram.iter()
        .fold(gram.len() as u64, |h, &id| splitmix(h ^ u64::from(id)))
}

impl<const W: usize> Recall<W> {
    /// The width chooser's rule for the recall draft: [`Rule::DEFAULT`], the
    /// starting gain counted as one test ([`Rule::prior`]). The confidences
    /// it reports rate a window by its own match; the gain scales them to
    /// the text, and the gate closes once the verify of the widest window
    /// that pays costs more than the ids it keeps.
    pub const RULE: Rule = Rule {
        prior: Some(1.0),
        ..Rule::DEFAULT
    };

    /// The shortest match the draft a seat serves ([`Recall::served`])
    /// proposes from.
    pub const SERVED_N_MIN: usize = 3;

    /// A draft that proposes from matches of at least `n_min` ids: it tracks
    /// `n_min`, twice and four times as many.
    pub fn new(n_min: usize) -> Result<Recall<W>, RecallError> {
        Recall::with_grams(&[n_min, n_min.saturating_mul(2), n_min.saturating_mul(4)])
    }

    /// The draft a seat serves: matches of at least
    /// [`Recall::SERVED_N_MIN`] ids, which it tracks at that length, twice
    /// and four times it.
    #[must_use]
    pub fn served() -> Recall<W> {
        let n = Self::SERVED_N_MIN;
        Recall::tracking(&[n, 2 * n, 4 * n])
    }

    /// A draft over lengths already known to be strictly ascending from at
    /// least 1.
    fn tracking(grams: &[usize]) -> Recall<W> {
        Recall {
            grams: grams.to_vec(),
            index: grams.iter().map(|_| HashMap::new()).collect(),
            ctx: Vec::new(),
        }
    }

    /// A draft tracking the n-gram lengths `grams`, strictly ascending from
    /// at least 1: the first is the shortest match that proposes.
    pub fn with_grams(grams: &[usize]) -> Result<Recall<W>, RecallError> {
        let Some(&first) = grams.first() else {
            return Err(RecallError::NoGram);
        };
        if first == 0 || grams.windows(2).any(|w| w[0] >= w[1]) {
            return Err(RecallError::Grams(grams.to_vec()));
        }
        Ok(Recall::tracking(grams))
    }

    /// Append `id` to the context. The n-grams that end just before it are
    /// indexed with `id` as their most recent follower.
    pub fn push(&mut self, id: u32) {
        let i = self.ctx.len();
        for (&n, table) in self.grams.iter().zip(&mut self.index) {
            if i >= n {
                table.insert(mix(&self.ctx[i - n..]), i);
            }
        }
        self.ctx.push(id);
    }

    /// The earlier occurrence the proposal follows: of the longest tracked
    /// length that has one, the most recent; `None` when no length does.
    #[must_use]
    pub fn longest(&self) -> Option<Matched> {
        let c = &self.ctx;
        let i = c.len();
        for (&n, table) in self.grams.iter().zip(&self.index).rev() {
            if i < n {
                continue;
            }
            let tail = &c[i - n..];
            let Some(&at) = table.get(&mix(tail)) else {
                continue;
            };
            if c[at - n..at] != *tail {
                continue;
            }
            let mut len = n;
            while len < CAP && len < at && c[at - len - 1] == c[i - len - 1] {
                len += 1;
            }
            return Some(Matched { len, at });
        }
        None
    }

    /// The ids that follow the occurrence `m`, written to all of `out`: the
    /// context's own ids, then — past its end — the proposal's, the match's
    /// period on.
    pub fn continuation(&self, m: Matched, out: &mut [u32]) {
        let i = self.ctx.len();
        for q in 0..out.len() {
            let s = m.at + q;
            out[q] = if s < i { self.ctx[s] } else { out[s - i] };
        }
    }

    /// The proposal for the context: [`Recall::longest`] and its
    /// [`Recall::continuation`] over all of `out`; `None` when there is no
    /// match and `out` is untouched.
    pub fn propose(&self, out: &mut [u32]) -> Option<Matched> {
        let m = self.longest()?;
        self.continuation(m, out);
        Some(m)
    }

    /// The context so far: the prompt, then every kept id.
    #[must_use]
    pub fn context(&self) -> &[u32] {
        &self.ctx
    }

    /// An empty context again, the tables' memory kept.
    pub fn reset(&mut self) {
        for table in &mut self.index {
            table.clear();
        }
        self.ctx.clear();
    }

    /// The context `ctx`, the tables rebuilt over it.
    pub fn rebuild(&mut self, ctx: &[u32]) {
        self.reset();
        self.ctx.reserve(ctx.len());
        for &id in ctx {
            self.push(id);
        }
    }

    /// The context's first `n` ids, the tables rebuilt over them; nothing
    /// when it holds `n` or fewer.
    pub fn truncate(&mut self, n: usize) {
        if n >= self.ctx.len() {
            return;
        }
        let ctx = std::mem::take(&mut self.ctx);
        self.rebuild(&ctx[..n]);
    }

    /// The context made what it holds of a target's positions before `pos`,
    /// then `last`, the id at `pos`: what a target that stepped, passed or
    /// cut back since the context last followed it stands at. A context that
    /// stands there with another id (a sampled id where the argmax was
    /// pushed) takes `last` in its place: the id a context ends with is in
    /// no indexed n-gram yet, so the tables hold. One that holds fewer than
    /// `pos` ids (the target ran positions it was not fed) keeps them all.
    pub fn stand(&mut self, pos: usize, last: u32) {
        if self.ctx.len() == pos + 1 {
            self.ctx[pos] = last;
            return;
        }
        self.truncate(pos);
        self.push(last);
    }
}

/// Ids a proposal, from the token history alone.
impl<T: Verify, const W: usize> Draft<T> for Recall<W> {
    const WIDTH: usize = W;
    const TAPS: TapNeed = TapNeed::None;

    /// The context starts over: this prompt, then `first`.
    fn begin(&mut self, _t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.rebuild(prompt);
        self.push(first);
        Ok(())
    }

    fn propose(&mut self, _t: &mut T, _last: u32, out: &mut [u32]) -> Result<usize, T::Error> {
        let n = out.len().min(W);
        Ok(match Recall::propose(self, &mut out[..n]) {
            Some(_) => n,
            None => 0,
        })
    }

    fn propose_p(
        &mut self,
        _t: &mut T,
        _last: u32,
        out: &mut [u32],
        p: &mut [f32],
    ) -> Result<usize, T::Error> {
        let n = out.len().min(W);
        let Some(m) = Recall::propose(self, &mut out[..n]) else {
            return Ok(0);
        };
        for (i, p) in p[..n].iter_mut().enumerate() {
            let l = (m.len + i) as f32;
            *p = l / (l + DOUBT);
        }
        Ok(n)
    }

    fn accept(
        &mut self,
        _t: &mut T,
        _rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), T::Error> {
        for &id in &out[..accepted] {
            self.push(id);
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
    use std::rc::Rc;

    use super::{CAP, Matched, Recall, RecallError};
    use crate::mock::{Clockwork, Mock, MockError};
    use crate::width::{Chosen, Mode};
    use crate::{Advance, Committed, Draft, Out, Speculative, Target, Verify, Want};

    /// The synthetic stream: a u32 LCG over a small vocabulary with a
    /// five-id phrase spliced in. The same generator in Python wrote the
    /// numbers `draft-accept.py` scored.
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

    /// `n` ids of an LCG over `vocab` ids, the state `seed`.
    fn ids(n: usize, vocab: u64, seed: u64) -> Vec<u32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                u32::try_from((s >> 33) % vocab).unwrap()
            })
            .collect()
    }

    /// The rule by scanning, no index: of the tracked lengths from the
    /// longest, the latest earlier position whose n preceding ids are the
    /// context's last n; its match widened backward to [`CAP`]; the ids that
    /// followed it, the proposal itself where they run past the context's end.
    fn reference(ctx: &[u32], grams: &[usize], w: usize) -> Option<(Matched, Vec<u32>)> {
        let i = ctx.len();
        for &n in grams.iter().rev() {
            if i < n {
                continue;
            }
            let Some(at) = (n..i).rev().find(|&at| ctx[at - n..at] == ctx[i - n..]) else {
                continue;
            };
            let len = (n..CAP)
                .find(|&len| len >= at || ctx[at - len - 1] != ctx[i - len - 1])
                .unwrap_or(n.max(CAP));
            let mut d: Vec<u32> = Vec::with_capacity(w);
            for k in 0..w {
                let s = at + k;
                let id = if s < i { ctx[s] } else { d[s - i] };
                d.push(id);
            }
            return Some((Matched { len, at }, d));
        }
        None
    }

    /// The proposals of a streamed [`Recall`] over `t` equal the reference's
    /// at every position.
    fn agree<const W: usize>(name: &str, t: &[u32], grams: &[usize]) {
        let mut rec = Recall::<W>::with_grams(grams).unwrap();
        rec.push(t[0]);
        let mut out = [0u32; W];
        for i in 1..t.len() {
            let want = reference(&t[..i], grams, W);
            let got = rec.propose(&mut out);
            let at = format!("{name}, grams {grams:?}, W {W}, position {i}");
            match (want, got) {
                (None, None) => {}
                (Some((m, d)), Some(g)) => {
                    assert_eq!(m, g, "the match at {at}");
                    assert_eq!(d, out, "the ids at {at}");
                }
                (w, g) => panic!("{at}: reference {w:?}, streamed {g:?}"),
            }
            rec.push(t[i]);
        }
    }

    /// Streams the index rule is checked on: random ones over a few ids (every
    /// n-gram recurs, matches overlap), runs and periods (a match that reaches
    /// the context's end), a block repeated with its last id changed each time
    /// (matches that widen to the cap and break), the spliced phrase.
    fn streams() -> Vec<(&'static str, Vec<u32>)> {
        let block: Vec<u32> = (200..270).collect();
        let mut repeated = Vec::new();
        for k in 0..4 {
            repeated.extend_from_slice(&block);
            *repeated.last_mut().unwrap() += 1000 * (k + 1);
        }
        vec![
            ("spliced phrase", stream(700)),
            ("three ids", ids(700, 3, 7)),
            ("two ids", ids(500, 2, 11)),
            ("eleven ids", ids(700, 11, 5)),
            ("a run of one id", vec![7; 120]),
            ("period two", [1, 2].repeat(60)),
            ("period seven", (11..18).collect::<Vec<u32>>().repeat(20)),
            ("a block repeated, its last id changed", repeated),
        ]
    }

    #[test]
    fn proposals_equal_the_scanning_reference() {
        let ladders: [&[usize]; 6] = [
            &[3, 6, 12],
            &[1],
            &[1, 2, 3],
            &[2, 4, 8],
            &[4, 8, 16],
            &[2, 40],
        ];
        for (name, t) in streams() {
            for grams in ladders {
                agree::<1>(name, &t, grams);
                agree::<3>(name, &t, grams);
                agree::<8>(name, &t, grams);
            }
        }
    }

    /// The tracked lengths `new` takes, the ones the served draft takes and the
    /// ones `with_grams` refuses by name; a proposal holds at most the room it
    /// is handed and at most `W`
    /// ids, and only a match of the shortest tracked length or more makes one.
    #[test]
    fn a_proposal_is_bounded_by_its_room_and_its_shortest_match() {
        assert_eq!(Recall::<2>::served().grams, [3, 6, 12]);
        assert_eq!(
            Recall::<4>::new(0).unwrap_err(),
            RecallError::Grams(vec![0, 0, 0])
        );
        assert_eq!(
            Recall::<4>::with_grams(&[]).unwrap_err(),
            RecallError::NoGram
        );
        for bad in [&[2, 2][..], &[3, 2], &[0, 2]] {
            assert_eq!(
                Recall::<4>::with_grams(bad).unwrap_err(),
                RecallError::Grams(bad.to_vec())
            );
        }
        // Only the 2-gram (5, 6) recurs: the tail (1, 5, 6) has no earlier copy.
        let ctx = [5, 6, 9, 1, 5, 6];
        let short = |n_min| {
            let mut r = Recall::<4>::new(n_min).unwrap();
            r.rebuild(&ctx);
            r
        };
        let mut out = [u32::MAX; 4];
        assert_eq!(short(3).propose(&mut out), None);
        assert_eq!(out, [u32::MAX; 4], "no proposal leaves the room alone");
        assert_eq!(short(2).propose(&mut out), Some(Matched { len: 2, at: 2 }));
        assert_eq!(out, [9, 1, 5, 6]);
        // The draft's room: at most W of it, 0 for no match.
        let t = &mut Mock::new();
        for (room, want) in [(10, 4), (4, 4), (2, 2), (1, 1), (0, 0)] {
            let mut out = [u32::MAX; 10];
            let n = Draft::<Mock>::propose(&mut short(2), t, 6, &mut out[..room]).unwrap();
            assert_eq!(n, want, "a room of {room}");
            assert!(out[want..].iter().all(|&id| id == u32::MAX), "past {want}");
        }
        let mut out = [u32::MAX; 4];
        assert_eq!(
            Draft::<Mock>::propose(&mut short(3), t, 6, &mut out).unwrap(),
            0
        );
        // A match widens backward to 32 ids and no further (the oracle's cap).
        let block: Vec<u32> = (200..270).collect();
        let mut r = Recall::<2>::new(3).unwrap();
        r.rebuild(&[block.clone(), block[..60].to_vec()].concat());
        assert_eq!(r.longest(), Some(Matched { len: 32, at: 60 }));
        // The width chooser's confidences: a match of l ids, its i-th id l + i
        // confirmations against the doubts.
        let mut p = [0.0f32; 4];
        let n = Draft::<Mock>::propose_p(&mut short(2), t, 6, &mut out, &mut p).unwrap();
        assert_eq!(n, 4);
        assert!(p.windows(2).all(|w| w[0] < w[1]) && p.iter().all(|&p| 0.0 < p && p < 1.0));
        assert_eq!(p[0], 2.0 / 4.0);
    }

    /// A generation starts the context over: nothing of the one before it is
    /// proposed from.
    #[test]
    fn begin_starts_the_context_over() {
        let mut r = Recall::<3>::new(2).unwrap();
        r.rebuild(&[1, 2, 3, 1, 2]);
        Draft::<Mock>::begin(&mut r, &Mock::new(), &[7, 8, 9, 7], 8).unwrap();
        assert_eq!(r.context(), [7, 8, 9, 7, 8]);
        let mut out = [0u32; 3];
        assert_eq!(r.propose(&mut out), Some(Matched { len: 2, at: 2 }));
        assert_eq!(out, [9, 7, 8]);
        // (1, 2) was followed by 3 in the generation before: it is no match now.
        r.push(1);
        r.push(2);
        assert_eq!(r.propose(&mut out), None);
    }

    /// Passes of a draft on the mock keep what the target takes: the context
    /// holds the prompt, the first id and each kept id once, so a proposal the
    /// target rejected never entered it.
    #[test]
    fn a_rejected_proposal_never_enters_the_context() {
        let prompt = [1, 2, 3, 1, 2];
        let mut t = Mock::new().with_ctx(1_000_000);
        let mut spec = Speculative::<Recall<3>, 4>::new(Recall::new(2).unwrap());
        let first = spec.prompt(&mut t, &prompt).unwrap();
        spec.begin(&t, &prompt, first).unwrap();
        let mut kept: Vec<u32> = prompt.to_vec();
        kept.push(first);
        let mut last = first;
        let (mut rejected, mut whole) = (0, 0);
        for _ in 0..300 {
            let mut out = Vec::new();
            let c = spec.pass(&mut t, last, &mut out).unwrap();
            assert_eq!(out.len(), c.kept);
            if c.proposed {
                rejected += usize::from(c.kept < c.rows);
                whole += usize::from(c.kept == c.rows);
            }
            kept.extend_from_slice(&out);
            last = *out.last().unwrap();
            assert_eq!(spec.draft().context(), kept);
        }
        assert!(
            rejected > 10 && whole > 10,
            "{rejected} rejected, {whole} whole"
        );
    }

    /// A context that follows a target by [`Recall::stand`] and its own
    /// pushes holds what one rebuilt from the target's history holds, and
    /// proposes what that one proposes: through a begin, kept passes, cuts
    /// back and a sampled id where the argmax was pushed.
    #[test]
    fn stand_follows_history() {
        let t = stream(600);
        let rebuilt = |history: &[u32], next: u32| {
            let mut r = Recall::<5>::new(2).unwrap();
            r.rebuild(history);
            r.push(next);
            r
        };
        let propose = |r: &Recall<5>| {
            let mut out = [0u32; 5];
            (r.propose(&mut out), out)
        };
        let mut rec = Recall::<5>::new(2).unwrap();
        rec.rebuild(&t[300..500]);
        Draft::<Mock>::begin(&mut rec, &Mock::new(), &t[..200], t[200]).unwrap();
        assert_eq!(rec.context(), &t[..201]);
        assert_eq!(propose(&rec), propose(&rebuilt(&t[..200], t[200])));
        let mut pos = 200;
        for i in 1..150 {
            let keep = 1 + pos % 2;
            rec.stand(pos, t[pos]);
            t[pos + 1..=pos + keep].iter().for_each(|&id| rec.push(id));
            pos += keep;
            if i % 25 == 0 {
                pos -= 7;
                rec.stand(pos, t[pos]);
            }
            let r = rebuilt(&t[..pos], t[pos]);
            assert_eq!(rec.context(), r.context());
            assert_eq!(propose(&rec), propose(&r));
        }
        let at = rec.context().as_ptr();
        rec.stand(pos, t[pos] + 1);
        let r = rebuilt(&t[..pos], t[pos] + 1);
        assert_eq!(rec.context(), r.context());
        assert_eq!(propose(&rec), propose(&r));
        assert_eq!(rec.context().as_ptr(), at, "the context was rebuilt");
        let mut behind = rebuilt(&t[..100], t[100]);
        behind.stand(300, t[300]);
        assert_eq!(behind.context().len(), 102);
    }

    /// `draft-accept.py`'s `score_longest` over the same 4000 ids as one
    /// document, one setting a case: the proposals at every position and the
    /// pass walk over them. A proposal at position i is checked against the
    /// stream's next ids up to the one before its last.
    #[test]
    fn matches_draft_accept_py() {
        fn score<const W: usize>(
            n_min: usize,
            t: &[u32],
        ) -> ((usize, usize, usize, Vec<usize>), [usize; 4]) {
            let mut rec = Recall::<W>::new(n_min).unwrap();
            rec.push(t[0]);
            // Per position: None for no proposal, else the leading ids right.
            let mut right: Vec<Option<usize>> = Vec::new();
            for i in 1..t.len() {
                let mut out = [0u32; W];
                right.push(rec.propose(&mut out).map(|_| {
                    let m = W.min(t.len() - 1 - i);
                    (0..m).take_while(|&r| out[r] == t[i + r]).count()
                }));
                rec.push(t[i]);
            }
            let mut hist = vec![0; W + 1];
            let (mut proposed, mut accepted) = (0, 0);
            for r in right.iter().flatten() {
                proposed += 1;
                accepted += r;
                hist[*r] += 1;
            }
            let (mut i, mut passes, mut drafted, mut tokens, mut hits) = (0, 0, 0, 0, 0);
            while i < right.len() {
                passes += 1;
                let r = right[i].unwrap_or(0);
                drafted += usize::from(right[i].is_some());
                hits += r;
                tokens += 1 + r;
                i += 1 + r;
            }
            (
                (right.len(), proposed, accepted, hist),
                [passes, drafted, tokens, hits],
            )
        }
        let t = stream(4000);
        assert_eq!(
            score::<4>(3, &t),
            (
                (3999, 2798, 3394, vec![1377, 415, 351, 343, 312]),
                [2624, 1423, 3999, 1375]
            )
        );
        assert_eq!(
            score::<8>(2, &t),
            (
                (
                    3999,
                    3850,
                    4116,
                    vec![2232, 477, 359, 355, 338, 64, 6, 4, 15]
                ),
                [2386, 2237, 3999, 1613]
            )
        );
        assert_eq!(
            score::<2>(4, &t),
            (
                (3999, 1942, 1873, vec![815, 381, 746]),
                [3125, 1068, 3999, 874]
            )
        );
    }

    /// A target whose greedy text is a script: the id at position `p` is
    /// `text[p]`. Its steps and verifies move the clock by the costs it is
    /// given, µs: index 0 a step, index `r − 1` a verify of `r` rows.
    struct Script {
        text: Vec<u32>,
        pos: u32,
        pending: Option<(u32, usize)>,
        clock: Rc<Clockwork>,
        cost: Vec<u64>,
    }

    impl Target for Script {
        type Error = MockError;

        fn pos(&self) -> u32 {
            self.pos
        }

        fn ctx(&self) -> u32 {
            u32::try_from(self.text.len()).unwrap()
        }

        fn prompt(&mut self, ids: &[u32], _want: Want) -> Result<Out<'_>, MockError> {
            let at = self.pos as usize;
            assert_eq!(ids, &self.text[at..at + ids.len()], "the script's own ids");
            self.pos += u32::try_from(ids.len()).unwrap();
            Ok(Out::Argmax(self.text[self.pos as usize]))
        }

        fn step(&mut self, id: u32, _want: Want) -> Result<Out<'_>, MockError> {
            assert_eq!(id, self.text[self.pos as usize], "the script's own id");
            self.clock.advance(self.cost[0]);
            self.pos += 1;
            Ok(Out::Argmax(self.text[self.pos as usize]))
        }

        fn keepable(&self, n: u32) -> u32 {
            n.min(self.pos)
        }

        fn cut(&mut self, n: u32) -> Result<(), MockError> {
            self.pos = n;
            Ok(())
        }

        fn reset(&mut self) -> Result<(), MockError> {
            (self.pos, self.pending) = (0, None);
            Ok(())
        }
    }

    impl Verify for Script {
        const MAX_ROWS: usize = 8;

        fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], MockError> {
            let at = self.pos as usize;
            assert_eq!(rows[0], self.text[at], "the script's own id");
            self.clock.advance(self.cost[M - 1]);
            self.pending = Some((self.pos, M));
            self.pos += u32::try_from(M).unwrap();
            Ok(std::array::from_fn(|r| self.text[at + r + 1]))
        }

        fn commit(&mut self, accepted: usize) -> Result<(), MockError> {
            let (pos, m) = self.pending.take().ok_or(MockError::Mock("no verify"))?;
            assert!((1..=m).contains(&accepted));
            self.pos = pos + u32::try_from(accepted).unwrap();
            Ok(())
        }
    }

    /// A step, and a verify of 2 to 8 rows at 1.209, 1.4145, 1.620, 1.815,
    /// 2.010, 2.206 and 2.401 steps: a dense 27B's read-once verify, its
    /// measured cells at 2, 4 and 8 rows and the rows between on a line
    /// [derived].
    const STEP_US: u64 = 27_565;
    const VERIFY_STEPS: [f64; 7] = [1.2089, 1.4145, 1.6201, 1.8154, 2.0104, 2.2055, 2.4007];

    /// What a run over a script took.
    struct Run {
        /// Tokens made over the wall of as many plain steps.
        gain: f64,
        /// The windows over the passes, and the mean rows a window kept, of
        /// the passes after the first `skip` tokens.
        windows: f64,
        kept: f64,
        /// The gate at the end: 0 closed.
        width: usize,
    }

    /// The recall draft of 7 ids behind the width chooser, by its own rule on
    /// the dense verify's costs, generating over `text` after its first
    /// `prompt` ids until `made` tokens are made; `skip` tokens in, the
    /// chooser's count starts over.
    fn run(text: &[u32], prompt: usize, made: usize, skip: usize) -> Run {
        let clock = Clockwork::new(STEP_US, 0);
        let mut cost = vec![STEP_US];
        cost.extend(
            VERIFY_STEPS
                .iter()
                .map(|c| (c * STEP_US as f64).round() as u64),
        );
        let mut t = Script {
            text: text.to_vec(),
            pos: 0,
            pending: None,
            clock: Rc::clone(&clock),
            cost,
        };
        let draft = <Recall<7> as Chosen<Script>>::choosing_by(
            Recall::new(3).unwrap(),
            Mode::Cost,
            Recall::<7>::RULE,
            Rc::clone(&clock),
        )
        .unwrap();
        let mut spec = Speculative::<_, 8>::new(draft);
        let first = spec.prompt(&mut t, &text[..prompt]).unwrap();
        spec.begin(&t, &text[..prompt], first).unwrap();
        let (mut last, mut out, mut tokens) = (first, Vec::new(), 0);
        let mut skipped = false;
        while tokens < made {
            if !skipped && tokens >= skip {
                spec.draft_mut().take_tally();
                skipped = true;
            }
            out.clear();
            let c: Committed = spec.pass(&mut t, last, &mut out).unwrap();
            tokens += c.kept;
            last = *out.last().unwrap();
        }
        let tally = spec.draft_mut().take_tally();
        Run {
            gain: tokens as f64 * STEP_US as f64 / clock.now.get() as f64,
            windows: tally.windows as f64 / tally.passes() as f64,
            kept: tally.e(),
            width: spec.draft().width(),
        }
    }

    /// Text of fresh ids with a copy of an earlier stretch after each: `fresh`
    /// new ids from a vocabulary no 3-gram of repeats in, then `copy` ids
    /// copied from a place before.
    fn copying(len: usize, fresh: usize, copy: usize) -> Vec<u32> {
        let new = ids(len, 1 << 20, 3);
        let place = ids(len, u64::MAX >> 34, 9);
        let mut t: Vec<u32> = Vec::with_capacity(len + 64);
        while t.len() < len {
            t.extend_from_slice(&new[t.len()..t.len() + fresh]);
            if t.len() > 4 * copy {
                let from = place[t.len()] as usize % (t.len() - copy);
                (0..copy).for_each(|k| t.push(t[from + k]));
            }
        }
        t.truncate(len);
        t.extend(ids(64, 1 << 20, 5));
        t
    }

    /// The width follows the text. Where the text is long copies of its own
    /// past the gate stays open and the windows are wide: per 52 ids, 15 are
    /// plain steps (12 fresh, 3 to make a match) and 37 go in windows of up to
    /// 8 rows, so a perfect width rule makes 52 ids in 15 + 5 · 2.40 = 27
    /// steps, 1.93 ids a step; the rule's calibration and the window that
    /// starts a copy at 3 ids of match (rated by its length) cost a fifth of
    /// it. Where matches are plenty and never carry on (six ids at random: a
    /// draft's id is the next one one time in six) the gate closes: the best
    /// width keeps 1 + 1/6 ids for 1.209 steps, under the step.
    #[test]
    fn the_width_follows_the_text() {
        let copies = run(&copying(6000, 12, 40), 1000, 5000, 1500);
        assert!(copies.gain > 1.5, "copies: {:.3} ids a step", copies.gain);
        assert!(copies.width >= 1, "copies: the gate closed");
        assert!(
            copies.kept > 3.0,
            "copies: {:.2} rows a window",
            copies.kept
        );
        let random = run(
            &[ids(6300, 6, 13), ids(64, 6, 17)].concat(),
            300,
            6000,
            3000,
        );
        assert!(random.gain > 0.97, "random: {:.3} ids a step", random.gain);
        assert_eq!(random.width, 0, "random: the gate stayed open");
        assert!(
            random.windows < 0.03,
            "random: {:.3} of the passes drafted",
            random.windows
        );
    }
}
