//! The session's greedy generation: the one loop over the session as the
//! generation loop's target ([`runtime::generate`] over [`Plain`]), stopped
//! by the token count or an end-of-generation id, each picked id handed to a
//! caller's sink as it is picked.
//!
//! This owns no rule the loop's own owners hold: the stop rule and its
//! reasons are [`runtime::Stop`]'s, the prompt's schedule and the chained
//! step are the target's (`target.rs`), and the end-of-generation set is the
//! vocabulary's ([`tokenizer::Tokenizer::eog`]), passed in by the caller. It
//! adds what a caller reading text needs around the loop: the walls of the
//! prompt and the decode, and the sink as a stream.

use std::fmt;
use std::time::{Duration, Instant};

use runtime::{Advance, Committed, NoTokens, PassSink, Plain, Stop, StopReason, Target};

use super::session::{Session, SessionError};

/// A greedy run's refusal or failure.
#[derive(Debug)]
pub enum GenError {
    /// The session refused or failed; its text names what.
    Session(SessionError),
    /// A token budget of no tokens ([`NoTokens`], whose rule this carries
    /// unchanged).
    NoTokens(NoTokens),
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::Session(e) => write!(f, "{e}"),
            GenError::NoTokens(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GenError {}

impl From<SessionError> for GenError {
    fn from(e: SessionError) -> Self {
        GenError::Session(e)
    }
}

impl From<NoTokens> for GenError {
    fn from(e: NoTokens) -> Self {
        GenError::NoTokens(e)
    }
}

/// What a greedy run tells its caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generated {
    /// Every generated token, generated token 0 first, up to the first
    /// end-of-generation id among them.
    pub ids: Vec<u32>,
    /// Why the run ended ([`StopReason`], the stop rule's own word).
    pub stop: StopReason,
    /// The wall of the prompt call: every prompt id through the target's
    /// prompt schedule, read back after the last.
    pub prompt_wall: Duration,
    /// The wall of the loop proper: every pass after the prompt's argmax.
    pub decode_wall: Duration,
    /// The completed passes counted by the runtime after the prompt.
    pub passes: usize,
    /// The wall of each pass of the loop, in order: a first pass that stands
    /// out from the rest is a cost the prompt left behind.
    pub pass_walls: Vec<Duration>,
}

impl Generated {
    /// The tokens the loop decoded: [`Generated::ids`]'s length.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }
}

/// Greedy generation over the session: `prompt`'s ids through the target,
/// then the loop feeding each pick back until the count or an
/// end-of-generation id ends it, every generated id passed to `on_id` as it
/// is picked — generated token 0 with it, which the prompt call picks.
///
/// The sink may allocate; the steps it hears of do not (the target's own
/// contract, held by the `t_target` clause).
///
/// # Errors
/// [`GenError::NoTokens`] for a budget of no tokens; [`GenError::Session`]
/// for every refusal and failure of the session, which a caller must not
/// step on again without a reset (a poisoned session refuses by name).
pub fn greedy<F>(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError>
where
    F: FnMut(u32),
{
    let mut on_id = on_id;
    let stop = Stop::new(budget, Target::ctx(s))?;
    let mut plain = Plain;
    let t0 = Instant::now();
    let first = plain.prompt(s, prompt)?;
    let prompt_wall = t0.elapsed();
    on_id(first);
    let mut walls = Vec::new();
    let mut sink = Sink {
        on_id: &mut on_id,
        walls: &mut walls,
    };
    let t1 = Instant::now();
    let outcome = runtime::generate(s, &mut plain, prompt, first, &stop.with_eog(eog), &mut sink)?;
    Ok(Generated {
        ids: outcome.tokens,
        stop: outcome.stop,
        prompt_wall,
        decode_wall: t1.elapsed(),
        passes: outcome.passes,
        pass_walls: walls,
    })
}

/// The caller's sink as the loop's [`PassSink`]: every token a pass keeps,
/// in order.
struct Sink<'a, F: FnMut(u32)> {
    on_id: &'a mut F,
    walls: &'a mut Vec<Duration>,
}

impl<F: FnMut(u32)> PassSink<Session> for Sink<'_, F> {
    type Error = SessionError;

    fn begin(&mut self, _t: &Session) -> Result<(), SessionError> {
        Ok(())
    }

    fn pass(
        &mut self,
        _t: &Session,
        _c: &Committed,
        tokens: &[u32],
        wall: Duration,
    ) -> Result<(), SessionError> {
        self.walls.push(wall);
        // A pass keeps whole rows; the loop has already cut `tokens` at the
        // first end-of-generation id, so the sink hears no id past the end.
        for &id in tokens {
            (self.on_id)(id);
        }
        Ok(())
    }
}
