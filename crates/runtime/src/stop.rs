//! When a generation ends: the one rule [`crate::generate`] reads before each
//! pass.

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The last token is one of the end-of-generation ids.
    Eog,
    /// The tokens reached the asked count.
    Length,
    /// The target's caches have no room for the next pass's rows.
    Ctx,
}

impl StopReason {
    /// The word a record prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            StopReason::Eog => "eog",
            StopReason::Length => "length",
            StopReason::Ctx => "ctx",
        }
    }
}

/// The refusal of a generation of no tokens: token 0 comes out of the prompt
/// call, so there is no count below one to stop at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoTokens;

impl std::fmt::Display for NoTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a generation of 0 tokens: token 0 comes out of the prompt call")
    }
}

impl std::error::Error for NoTokens {}

/// The stop rule: the asked count, the end-of-generation ids, the context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stop {
    max_tokens: usize,
    eog: Vec<u32>,
    ctx: u32,
}

impl Stop {
    /// Stop after `max_tokens` tokens, generated token 0 counted, or when the
    /// next pass's rows do not fit caches of `ctx` positions; no
    /// end-of-generation id until [`Stop::with_eog`].
    pub fn new(max_tokens: usize, ctx: u32) -> Result<Stop, NoTokens> {
        if max_tokens == 0 {
            return Err(NoTokens);
        }
        Ok(Stop {
            max_tokens,
            eog: Vec::new(),
            ctx,
        })
    }

    /// Also stop at any of `ids`, the vocabulary's end-of-generation set.
    #[must_use]
    pub fn with_eog(mut self, ids: &[u32]) -> Stop {
        self.eog = ids.to_vec();
        self
    }

    /// The asked count.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Why the generation ends after `emitted` tokens whose last is `last`,
    /// the target standing at `pos` and the next pass running up to `rows`
    /// positions from it — or `None` to run that pass. An end-of-generation
    /// id ends it first, then the count, then the context.
    #[must_use]
    pub fn check(&self, emitted: usize, last: u32, pos: u32, rows: usize) -> Option<StopReason> {
        let room = self.ctx.saturating_sub(pos);
        if self.eog.contains(&last) {
            Some(StopReason::Eog)
        } else if emitted >= self.max_tokens {
            Some(StopReason::Length)
        } else if u32::try_from(rows).map_or(true, |r| r > room) {
            Some(StopReason::Ctx)
        } else {
            None
        }
    }

    /// The index of the first end-of-generation id in `tokens`, a pass's kept
    /// tokens: where the generation's tokens end.
    #[must_use]
    pub fn first_eog(&self, tokens: &[u32]) -> Option<usize> {
        tokens.iter().position(|t| self.eog.contains(t))
    }
}

#[cfg(test)]
mod tests {
    use super::{NoTokens, Stop, StopReason};

    /// The count stops at exactly `max_tokens`, not one past: the plain loop
    /// runs `max_tokens − 1` steps after token 0.
    #[test]
    fn length_at_the_count() {
        let s = Stop::new(4, 100).unwrap();
        assert_eq!(s.check(3, 7, 10, 1), None);
        assert_eq!(s.check(4, 7, 10, 1), Some(StopReason::Length));
        assert_eq!(s.check(5, 7, 10, 1), Some(StopReason::Length));
    }

    /// An end-of-generation id wins over the count and the context: the
    /// token that ends the turn is reported as the end of the turn.
    #[test]
    fn eog_first() {
        let s = Stop::new(4, 10).unwrap().with_eog(&[2, 9]);
        assert_eq!(s.check(1, 9, 3, 1), Some(StopReason::Eog));
        assert_eq!(s.check(4, 2, 10, 1), Some(StopReason::Eog));
        assert_eq!(s.check(4, 3, 10, 1), Some(StopReason::Length));
        assert_eq!(s.check(2, 3, 10, 1), Some(StopReason::Ctx));
        assert_eq!(s.check(2, 3, 9, 1), None);
        assert_eq!(s.first_eog(&[5, 9, 2]), Some(1));
        assert_eq!(s.first_eog(&[5, 3]), None);
    }

    /// The context stops the loop when the next pass's rows do not all fit:
    /// a pass of `rows` at `pos` needs positions up to `pos + rows − 1`.
    #[test]
    fn ctx_counts_the_pass_rows() {
        let s = Stop::new(100, 10).unwrap();
        assert_eq!(s.check(2, 3, 8, 2), None);
        assert_eq!(s.check(2, 3, 9, 2), Some(StopReason::Ctx));
        assert_eq!(s.check(2, 3, 9, 1), None);
        assert_eq!(s.check(2, 3, 7, 4), Some(StopReason::Ctx));
        assert_eq!(s.check(2, 3, 12, 1), Some(StopReason::Ctx));
    }

    /// No count below one exists.
    #[test]
    fn zero_tokens_refused() {
        assert_eq!(Stop::new(0, 10), Err(NoTokens));
    }
}
