//! A target on the host for the tests: a deterministic "model" whose greedy
//! next token is a function of the history, and a log of every call.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use crate::width::{Clock, WidthError};
use crate::{Committed, Out, PassSink, RowLogits, Target, Verify, Want};

/// The ids the mock's rows span.
pub(crate) const VOCAB: usize = 5;

/// One call the mock ran, at the position it started from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Call {
    Prompt(u32, usize),
    Step(u32),
    Verify(u32, usize),
    Commit(usize),
}

#[derive(Debug)]
pub(crate) enum MockError {
    Mock(&'static str),
    Width(WidthError),
}

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MockError::Mock(why) => f.write_str(why),
            MockError::Width(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for MockError {}

impl From<WidthError> for MockError {
    fn from(e: WidthError) -> MockError {
        MockError::Width(e)
    }
}

/// A clock the mock's calls move: each step and each verify advances it by
/// its cost, in microseconds; a test changes the costs as it runs, the
/// verify's by the rows it ran when a cost a row count is set.
#[derive(Debug, Default)]
pub(crate) struct Clockwork {
    pub(crate) now: Cell<u64>,
    pub(crate) step: Cell<u64>,
    pub(crate) verify: Cell<u64>,
    /// A verify of `r` rows' cost, `r − 1` values; unset, every verify the
    /// one `verify` cost.
    pub(crate) by_rows: RefCell<Option<Vec<u64>>>,
}

impl Clockwork {
    pub(crate) fn new(step: u64, verify: u64) -> Rc<Clockwork> {
        Rc::new(Clockwork {
            now: Cell::new(0),
            step: Cell::new(step),
            verify: Cell::new(verify),
            by_rows: RefCell::new(None),
        })
    }

    /// A verify of `r` rows costs `us`, one value a row count from 2 on;
    /// a row count unset keeps the one `verify` cost.
    pub(crate) fn verify_rows(&self, us: &[u64]) {
        *self.by_rows.borrow_mut() = Some(us.to_vec());
    }

    fn cost_of(&self, rows: usize) -> u64 {
        self.by_rows
            .borrow()
            .as_ref()
            .and_then(|c| c.get(rows - 2).copied())
            .unwrap_or(self.verify.get())
    }

    pub(crate) fn advance(&self, us: u64) {
        self.now.set(self.now.get() + us);
    }
}

impl Clock for Rc<Clockwork> {
    fn now(&mut self) -> Duration {
        Duration::from_micros(self.now.get())
    }
}

pub(crate) struct Mock {
    history: Vec<u32>,
    calls: Vec<Call>,
    fail_prompt: bool,
    fail_at: Option<u32>,
    rows: Option<(usize, usize)>,
    ctx: u32,
    clock: Option<Rc<Clockwork>>,
    /// The last row read back ([`RowLogits::row_logits`]).
    row: [f32; VOCAB],
    /// Rows read back: [`RowLogits::row_logits`] calls and calls that asked
    /// [`Want::Logits`].
    logit_reads: usize,
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
            clock: None,
            row: [0.0; VOCAB],
            logit_reads: 0,
        }
    }

    /// Rows read back so far.
    pub(crate) fn logit_reads(&self) -> usize {
        self.logit_reads
    }

    /// Each step and verify advances `clock` by its cost.
    pub(crate) fn with_clock(mut self, clock: Rc<Clockwork>) -> Mock {
        self.clock = Some(clock);
        self
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

    /// The `n` tokens greedy steps from `id` would return, the history left as
    /// it is: [`Mock::next_after`] first.
    pub(crate) fn greedy_after(&self, id: u32, n: usize) -> Vec<u32> {
        let (mut len, mut a, mut b) = (
            self.history.len() + 1,
            id,
            self.history.last().copied().unwrap_or(0),
        );
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let next = rule(len, a, b);
            out.push(next);
            (len, a, b) = (len + 1, next, a);
        }
        out
    }

    /// Refused unless positions `at()` to `at() + rows − 1` fit the caches.
    fn fits(&self, rows: usize) -> Result<(), MockError> {
        if self.history.len() + rows > self.ctx as usize {
            return Err(MockError::Mock("rows past the caches' ctx"));
        }
        Ok(())
    }

    fn at(&self) -> u32 {
        u32::try_from(self.history.len()).unwrap()
    }

    fn idle(&self) -> Result<(), MockError> {
        match self.rows {
            Some(_) => Err(MockError::Mock("a verify's rows wait for commit")),
            None => Ok(()),
        }
    }
}

/// [`Mock`]'s next token after a history of `n` ids ending `b`, `a`.
fn rule(n: usize, a: u32, b: u32) -> u32 {
    (a * 3 + b + u32::from((n / 7) % 2 == 1)) % 5
}

/// The logits after `hist` (at least one id): the rule's argmax 2 above a
/// noise in [0, 1) that hashes the whole of `hist`, so a row moves with every
/// id before it and its argmax is the rule's.
fn logits_after(hist: &[u32], out: &mut [f32; VOCAB]) {
    let n = hist.len();
    let b = if n >= 2 { hist[n - 2] } else { 0 };
    let top = rule(n, hist[n - 1], b);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in hist {
        h = (h ^ u64::from(id)).wrapping_mul(0x0100_0000_01b3);
    }
    for (v, o) in (0u32..).zip(out.iter_mut()) {
        h = (h ^ u64::from(v)).wrapping_mul(0x0100_0000_01b3);
        let noise = (h >> 40) as f32 / (1u64 << 24) as f32;
        *o = noise + if v == top { 2.0 } else { 0.0 };
    }
}

impl Target for Mock {
    type Error = MockError;

    fn pos(&self) -> u32 {
        self.at()
    }

    fn ctx(&self) -> u32 {
        self.ctx
    }

    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, MockError> {
        self.calls.push(Call::Prompt(self.at(), ids.len()));
        self.logit_reads += usize::from(want == Want::Logits);
        self.idle()?;
        if self.fail_prompt {
            return Err(MockError::Mock("the prompt call failed"));
        }
        self.fits(ids.len())?;
        self.history.extend_from_slice(ids);
        Ok(Out::Argmax(self.argmax()))
    }

    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, MockError> {
        self.calls.push(Call::Step(self.at()));
        self.logit_reads += usize::from(want == Want::Logits);
        self.idle()?;
        if self.fail_at == Some(self.at()) {
            return Err(MockError::Mock("the step failed"));
        }
        self.fits(1)?;
        if let Some(c) = &self.clock {
            c.advance(c.step.get());
        }
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
        if let Some(c) = &self.clock {
            c.advance(c.cost_of(M));
        }
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
            .ok_or(MockError::Mock("a commit with no verify"))?;
        if !(1..=m).contains(&accepted) {
            return Err(MockError::Mock("a commit outside the verify's rows"));
        }
        self.history.truncate(first + accepted);
        Ok(())
    }
}

impl RowLogits for Mock {
    fn row_logits(&mut self, r: usize) -> Result<&[f32], MockError> {
        let end = match self.rows {
            Some((first, m)) if r < m => first + r + 1,
            None if r == 0 && !self.history.is_empty() => self.history.len(),
            _ => return Err(MockError::Mock("a row the last call did not run")),
        };
        self.logit_reads += 1;
        logits_after(&self.history[..end], &mut self.row);
        Ok(&self.row)
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
