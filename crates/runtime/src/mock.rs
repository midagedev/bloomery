//! A target on the host for the tests: a deterministic "model" whose greedy
//! next token is a function of the history, and a log of every call.

use crate::{Committed, Out, PassSink, Target, Verify, Want};

/// One call the mock ran, at the position it started from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Call {
    Prompt(u32, usize),
    Step(u32),
    Verify(u32, usize),
    Commit(usize),
}

#[derive(Debug)]
pub(crate) struct MockError(&'static str);

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for MockError {}

pub(crate) struct Mock {
    history: Vec<u32>,
    calls: Vec<Call>,
    fail_prompt: bool,
    fail_at: Option<u32>,
    rows: Option<(usize, usize)>,
    ctx: u32,
}

impl Mock {
    pub(crate) fn new() -> Mock {
        Mock {
            history: Vec::new(),
            calls: Vec::new(),
            fail_prompt: false,
            fail_at: None,
            rows: None,
            ctx: 1000,
        }
    }

    /// Caches of `ctx` positions: a step at `ctx` and a verify whose rows
    /// pass it are refused, as the engine's model refuses them.
    pub(crate) fn with_ctx(mut self, ctx: u32) -> Mock {
        self.ctx = ctx;
        self
    }

    /// Every prompt call fails.
    pub(crate) fn fail_prompt(mut self) -> Mock {
        self.fail_prompt = true;
        self
    }

    /// The step at position `pos` fails.
    pub(crate) fn fail_at(mut self, pos: u32) -> Mock {
        self.fail_at = Some(pos);
        self
    }

    pub(crate) fn calls(&self) -> &[Call] {
        &self.calls
    }

    /// The greedy token after the history: an order-two chain over five ids
    /// whose rule flips every seven positions, so a lookup both hits and
    /// misses.
    fn argmax(&self) -> u32 {
        let n = self.history.len();
        let b = if n >= 2 { self.history[n - 2] } else { 0 };
        rule(n, self.history[n - 1], b)
    }

    /// The token a step of `id` would return, the history left as it is.
    pub(crate) fn next_after(&self, id: u32) -> u32 {
        let b = self.history.last().copied().unwrap_or(0);
        rule(self.history.len() + 1, id, b)
    }

    /// Refused unless positions `at()` to `at() + rows − 1` fit the caches.
    fn fits(&self, rows: usize) -> Result<(), MockError> {
        if self.history.len() + rows > self.ctx as usize {
            return Err(MockError("rows past the caches' ctx"));
        }
        Ok(())
    }

    fn at(&self) -> u32 {
        u32::try_from(self.history.len()).unwrap()
    }

    fn idle(&self) -> Result<(), MockError> {
        match self.rows {
            Some(_) => Err(MockError("a verify's rows wait for commit")),
            None => Ok(()),
        }
    }
}

/// [`Mock`]'s next token after a history of `n` ids ending `b`, `a`.
fn rule(n: usize, a: u32, b: u32) -> u32 {
    (a * 3 + b + u32::from((n / 7) % 2 == 1)) % 5
}

impl Target for Mock {
    type Error = MockError;

    fn pos(&self) -> u32 {
        self.at()
    }

    fn ctx(&self) -> u32 {
        self.ctx
    }

    fn prompt(&mut self, ids: &[u32], _want: Want) -> Result<Out<'_>, MockError> {
        self.calls.push(Call::Prompt(self.at(), ids.len()));
        self.idle()?;
        if self.fail_prompt {
            return Err(MockError("the prompt call failed"));
        }
        self.fits(ids.len())?;
        self.history.extend_from_slice(ids);
        Ok(Out::Argmax(self.argmax()))
    }

    fn step(&mut self, id: u32, _want: Want) -> Result<Out<'_>, MockError> {
        self.calls.push(Call::Step(self.at()));
        self.idle()?;
        if self.fail_at == Some(self.at()) {
            return Err(MockError("the step failed"));
        }
        self.fits(1)?;
        self.history.push(id);
        Ok(Out::Argmax(self.argmax()))
    }

    fn keepable(&self, n: u32) -> u32 {
        n.min(self.at())
    }

    fn cut(&mut self, n: u32) -> Result<(), MockError> {
        self.history.truncate(n as usize);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), MockError> {
        self.history.clear();
        self.rows = None;
        Ok(())
    }
}

impl Verify for Mock {
    const MAX_ROWS: usize = 4;

    fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], MockError> {
        self.calls.push(Call::Verify(self.at(), M));
        self.idle()?;
        self.fits(M)?;
        let first = self.history.len();
        let mut out = [0u32; M];
        for (o, &id) in out.iter_mut().zip(&rows) {
            self.history.push(id);
            *o = self.argmax();
        }
        self.rows = Some((first, M));
        Ok(out)
    }

    fn commit(&mut self, accepted: usize) -> Result<(), MockError> {
        self.calls.push(Call::Commit(accepted));
        let (first, m) = self
            .rows
            .take()
            .ok_or(MockError("a commit with no verify"))?;
        if !(1..=m).contains(&accepted) {
            return Err(MockError("a commit outside the verify's rows"));
        }
        self.history.truncate(first + accepted);
        Ok(())
    }
}

/// A sink that hears nothing.
pub(crate) struct Quiet;

impl PassSink<Mock> for Quiet {
    type Error = MockError;

    fn begin(&mut self, _t: &Mock) -> Result<(), MockError> {
        Ok(())
    }

    fn pass(
        &mut self,
        _t: &Mock,
        _c: &Committed,
        _tokens: &[u32],
        _wall: std::time::Duration,
    ) -> Result<(), MockError> {
        Ok(())
    }
}
