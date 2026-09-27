//! When a generation ends: the one rule [`crate::generate`] reads before each
//! pass.

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The last token is one of the end-of-generation ids.
    Eog,
    /// The tokens reached the asked count.
    Length,
    /// The target stands at the end of its caches.
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
    /// target stands at `ctx`; no end-of-generation id until
    /// [`Stop::with_eog`].
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
    /// the target standing at `pos` — or `None` to run another pass. An
    /// end-of-generation id ends it first, then the count, then the context.
    #[must_use]
    pub fn check(&self, emitted: usize, last: u32, pos: u32) -> Option<StopReason> {
        if self.eog.contains(&last) {
            Some(StopReason::Eog)
        } else if emitted >= self.max_tokens {
            Some(StopReason::Length)
        } else if pos >= self.ctx {
            Some(StopReason::Ctx)
        } else {
            None
        }
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
        assert_eq!(s.check(3, 7, 10), None);
        assert_eq!(s.check(4, 7, 10), Some(StopReason::Length));
        assert_eq!(s.check(5, 7, 10), Some(StopReason::Length));
    }

    /// An end-of-generation id wins over the count and the context: the
    /// token that ends the turn is reported as the end of the turn.
    #[test]
    fn eog_first() {
        let s = Stop::new(4, 10).unwrap().with_eog(&[2, 9]);
        assert_eq!(s.check(1, 9, 3), Some(StopReason::Eog));
        assert_eq!(s.check(4, 2, 10), Some(StopReason::Eog));
        assert_eq!(s.check(4, 3, 10), Some(StopReason::Length));
        assert_eq!(s.check(2, 3, 10), Some(StopReason::Ctx));
        assert_eq!(s.check(2, 3, 9), None);
    }

    /// No count below one exists.
    #[test]
    fn zero_tokens_refused() {
        assert_eq!(Stop::new(0, 10), Err(NoTokens));
    }
}
