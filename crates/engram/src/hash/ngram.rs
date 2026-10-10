//! The row rule both n-gram hashes share, and the window of the one that
//! hashes raw token ids with an EOS reset (a PLE site). Plain integer
//! arithmetic over slices: no header, no file, no allocation per token.

use std::ops::Range;

/// The `out.len()` row ids of one site for the window `ctx`.
///
/// `ctx[0]` is the current token's context value and `ctx[s]` the one `s`
/// positions back; `mult` has one multiplier per window slot, `prime` and
/// `offset` one entry per bucket, `n_heads` buckets per n-gram order. Order
/// `n` (2 to `ctx.len()`) folds the first `n` slots, `rolling = ⊕_{j<n}
/// ctx[j]·mult[j]` in wrapping `u64`, and its bucket `b = (n − 2)·n_heads +
/// h` is row `rolling % prime[b] + offset[b]`.
///
/// # Panics
///
/// When the slices are not `mult.len() == ctx.len()` and `prime.len() ==
/// offset.len() == out.len() == (ctx.len() − 1)·n_heads`, or a row does not
/// fit a `u32`: the constants' readers check both before a caller gets here.
pub fn rows(
    mult: &[u64],
    prime: &[u64],
    offset: &[u64],
    n_heads: usize,
    ctx: &[u64],
    out: &mut [u32],
) {
    let n_gram = ctx.len();
    let n_cols = (n_gram - 1) * n_heads;
    assert!(
        mult.len() == n_gram
            && prime.len() == n_cols
            && offset.len() == n_cols
            && out.len() == n_cols,
        "ngram::rows: {} multipliers, {} primes, {} offsets and {} out slots for a \
         {n_gram}-slot window of {n_heads} heads an order",
        mult.len(),
        prime.len(),
        offset.len(),
        out.len()
    );
    let mut rolling = ctx[0].wrapping_mul(mult[0]);
    for s in 1..n_gram {
        rolling ^= ctx[s].wrapping_mul(mult[s]);
        let base = (s - 1) * n_heads;
        for h in 0..n_heads {
            let b = base + h;
            let id = rolling % prime[b] + offset[b];
            out[b] = u32::try_from(id)
                .expect("every bucket top was checked to fit u32 when the constants were read");
        }
    }
}

/// A raw-id window's refusals. Each names the input it refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NgramError {
    #[error("token {token} is past the {vocab}-token vocabulary: not a token of this model")]
    TokenPastVocab { token: u32, vocab: u32 },
    #[error(
        "token {token} is the image placeholder; the n-gram hash runs on text only, and an image \
         position has no token id to hash"
    )]
    ImageToken { token: u32 },
    #[error(
        "the n-gram history ends before position {want}, and the tokens start at position {got}: \
         the window would read another position's predecessors (reset the history for a new sequence)"
    )]
    PositionGap { want: u64, got: u64 },
    #[error("{tokens} tokens' row ids want a {want}-id buffer, got {got}")]
    RowBufferSize {
        tokens: usize,
        want: usize,
        got: usize,
    },
    #[error("this hash maps its ids through a token map; it has no raw-id window")]
    NotRawIds,
    #[error(
        "image span {index} ({span:?}) is malformed for a call of {tokens} tokens: {why}; spans \
         are non-empty, ascending, non-overlapping row ranges inside the call"
    )]
    BadSpan {
        index: usize,
        span: Range<usize>,
        tokens: usize,
        why: &'static str,
    },
    #[error(
        "the call has image spans, and this hash carries no image token id (`ple.image_token_id`) \
         to hash an image position by"
    )]
    NoImageToken,
}

/// The raw-id window: a site that hashes token ids as they are, where an
/// end-of-sequence token cuts the n-grams.
///
/// `ctx[0]` is the token; `ctx[s]` is the token `s` positions back, except
/// that once a predecessor is `eos` — or lies before the sequence start,
/// which reads as `eos` — it and every older slot are `eos`. The token's own
/// `eos` does not cut its own window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EosWindow {
    /// The id that resets the window (`ple.eos_token_id`), not necessarily the
    /// tokenizer's end of text.
    pub eos: u32,
    /// The image placeholder id (`ple.image_token_id`): hashed like any id at
    /// an image position, refused by name anywhere else.
    pub image: Option<u32>,
    /// Tokens of the model's vocabulary; an id at or past it is refused.
    pub n_vocab: u32,
}

impl EosWindow {
    /// Refuse an id that is no text token of the model.
    pub fn check(&self, token: u32) -> Result<(), NgramError> {
        self.check_at(token, false)
    }

    /// [`EosWindow::check`] for a position that may be an image position
    /// (`in_image_span`): there the image placeholder is the id the position
    /// hashes by, and passes.
    pub fn check_at(&self, token: u32, in_image_span: bool) -> Result<(), NgramError> {
        if token >= self.n_vocab {
            return Err(NgramError::TokenPastVocab {
                token,
                vocab: self.n_vocab,
            });
        }
        if self.image == Some(token) && !in_image_span {
            return Err(NgramError::ImageToken { token });
        }
        Ok(())
    }

    /// The window of `token` after the predecessors `prev` (oldest first, the
    /// last one right before `token`, `eos` where the sequence had none) into
    /// `ctx`, `prev.len() + 1` slots.
    ///
    /// # Panics
    ///
    /// When `ctx.len() != prev.len() + 1`.
    pub fn window(&self, prev: &[u32], token: u32, ctx: &mut [u64]) {
        assert_eq!(
            ctx.len(),
            prev.len() + 1,
            "EosWindow::window: a {}-slot window after {} predecessors",
            ctx.len(),
            prev.len()
        );
        ctx[0] = u64::from(token);
        let mut cut = false;
        for s in 1..ctx.len() {
            let t = prev[prev.len() - s];
            cut |= t == self.eos;
            ctx[s] = u64::from(if cut { self.eos } else { t });
        }
    }
}

/// One sequence's raw-id history: its last `n_gram − 1` tokens, oldest
/// first, and the position the next token takes. A new sequence starts from
/// [`History::new`] (every slot `eos`, position 0). It is small and `Clone`:
/// a caller that may roll a pass back keeps the copy from before the pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct History {
    prev: Vec<u32>,
    next: u64,
}

impl History {
    /// The history before a sequence's first token.
    #[must_use]
    pub fn new(n_gram: usize, eos: u32) -> History {
        History {
            prev: vec![eos; n_gram.saturating_sub(1)],
            next: 0,
        }
    }

    /// The last `n_gram − 1` tokens, oldest first.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.prev
    }

    /// The position the next token takes.
    #[must_use]
    pub fn next_pos(&self) -> u64 {
        self.next
    }

    /// Append `token`, the one at [`History::next_pos`].
    pub fn push(&mut self, token: u32) {
        if !self.prev.is_empty() {
            self.prev.rotate_left(1);
            let last = self.prev.len() - 1;
            self.prev[last] = token;
        }
        self.next += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{EosWindow, History, NgramError, rows};

    const EOS: u32 = 248_044;

    fn win() -> EosWindow {
        EosWindow {
            eos: EOS,
            image: Some(248_056),
            n_vocab: 248_320,
        }
    }

    /// An EOS predecessor cuts it and every older slot; the token's own EOS
    /// cuts nothing; the tokenizer's end of text (248,046) is an ordinary id.
    #[test]
    fn eos_cuts_older_slots_only() {
        let w = win();
        let mut ctx = [0u64; 3];
        w.window(&[7, EOS], 9, &mut ctx);
        assert_eq!(ctx, [9, u64::from(EOS), u64::from(EOS)], "EOS right before");
        w.window(&[EOS, 7], 9, &mut ctx);
        assert_eq!(ctx, [9, 7, u64::from(EOS)], "EOS two back");
        w.window(&[5, 7], EOS, &mut ctx);
        assert_eq!(ctx, [u64::from(EOS), 7, 5], "the token's own EOS");
        w.window(&[5, 248_046], 9, &mut ctx);
        assert_eq!(ctx, [9, 248_046, 5], "248,046 is not the PLE EOS");
    }

    /// Ids past the vocabulary and the image placeholder are refused by
    /// name; the last id of the vocabulary and the EOS pass.
    #[test]
    fn refusals_name_the_id() {
        let w = win();
        assert_eq!(w.check(248_319), Ok(()));
        assert_eq!(w.check(EOS), Ok(()));
        assert_eq!(
            w.check(248_320),
            Err(NgramError::TokenPastVocab {
                token: 248_320,
                vocab: 248_320
            })
        );
        assert_eq!(
            w.check(248_056),
            Err(NgramError::ImageToken { token: 248_056 })
        );
    }

    /// The image placeholder passes inside an image span only; the vocabulary
    /// bound holds there too.
    #[test]
    fn image_id_passes_inside_a_span_only() {
        let w = win();
        assert_eq!(w.check_at(248_056, true), Ok(()));
        assert_eq!(
            w.check_at(248_056, false),
            Err(NgramError::ImageToken { token: 248_056 })
        );
        assert_eq!(
            w.check_at(248_320, true),
            Err(NgramError::TokenPastVocab {
                token: 248_320,
                vocab: 248_320
            })
        );
    }

    /// A fresh history is all EOS at position 0; pushes shift it oldest
    /// first and count positions.
    #[test]
    fn history_shifts_oldest_first() {
        let mut h = History::new(3, EOS);
        assert_eq!((h.tokens(), h.next_pos()), (&[EOS, EOS][..], 0));
        h.push(1);
        h.push(2);
        h.push(3);
        assert_eq!((h.tokens(), h.next_pos()), (&[2, 3][..], 3));
    }

    /// Order n folds the first n slots in wrapping `u64` and reduces by its
    /// own bucket's prime before its offset: with one head an order,
    /// `mult[0] = u64::MAX` (so `2·mult[0]` wraps), order 2's row is
    /// `(2·m0 ^ 4·m1) % 1,000,003 + 10` and order 3's that xor `6·m2`, reduced
    /// by 1,000,033, plus 2,000,000 — the values below, computed in Python.
    #[test]
    fn rows_fold_each_order_over_its_slots() {
        let mut out = [0u32; 2];
        rows(
            &[u64::MAX, 3, 5],
            &[1_000_003, 1_000_033],
            &[10, 2_000_000],
            1,
            &[2, 4, 6],
            &mut out,
        );
        assert_eq!(out, [350_683, 2_960_610]);
    }
}
