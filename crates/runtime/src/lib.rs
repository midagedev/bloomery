//! The generation loop and the traits it drives — host only, no card.
//!
//! A [`Target`] runs a model: a prompt call from where it stands, a one-token
//! step, and — where the model can — a pass of several rows ([`Verify`]) that
//! verifies a draft's proposal. How a generation moves on is an [`Advance`]:
//! [`Plain`] steps once a pass, [`Speculative`] runs a [`Draft`]'s proposal
//! through a verify and keeps the rows the target agrees with, and [`Gated`]
//! lets a draft's verify passes run only while they pay. [`generate`] is the
//! one loop over both, until the [`Stop`] rule ends it.
//!
//! Everything here is greedy: a pass keeps the target's argmax, so a draft
//! changes which passes run and never a token. Sampling — the sampler chain,
//! and verification by sampling each verified row — needs the rows' logits.
//!
//! Beneath a target's call: [`sched`], the order in which a schedule runs a
//! layer program's parts over its units ([`sched::walk`]), [`state`], the
//! stores a layer keeps and how later positions read them, [`stores`], their
//! sizes, [`seqstate`], the
//! checkpoints of the recurrent ones and which cuts they serve, and
//! [`layer`], the sub-layer programs a layer's description names.
//! [`words`] is smaller and sits below all of them: the word count of a
//! flat weight stream, the one count the loader's staging and the packings
//! that pin it must agree on.

pub mod combine;
pub mod gate;
pub mod hc_gated;
pub mod layer;
mod lookup;
pub mod qsa;
pub mod sched;
pub mod seqstate;
mod speculative;
pub mod state;
mod stop;
pub mod stores;
pub mod swaprule;
pub mod words;

pub use gate::Gated;
pub use lookup::Lookup;
pub use speculative::{
    Draft, NotBuilt, Program, Speculative, TapNeed, Tapped, Width, Widths, Window, accepted_rows,
    program,
};
pub use stop::{NoTokens, Stop, StopReason};

use std::time::{Duration, Instant};

/// What a call of the target reads back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// The argmax alone: greedy never reads the row back.
    Argmax,
    /// The logits row of the last position as well.
    Logits,
}

/// What a call of the target read back, as its [`Want`] asked.
#[derive(Debug)]
pub enum Out<'a> {
    Argmax(u32),
    /// The argmax and the logits row it is the argmax of.
    Logits {
        argmax: u32,
        row: &'a [f32],
    },
}

impl Out<'_> {
    /// The target's greedy next token.
    #[must_use]
    pub fn argmax(&self) -> u32 {
        match *self {
            Out::Argmax(t) | Out::Logits { argmax: t, .. } => t,
        }
    }
}

/// A model standing at a position: the next id it runs lands in cache row
/// [`Target::pos`].
///
/// A call that fails returns its error and nothing here calls again on its
/// own: a target that took a failed call's positions back stands where its
/// error says, and one whose device raised a fault refuses every call until
/// [`Target::reset`].
pub trait Target {
    /// What a failed call says.
    type Error: std::error::Error + 'static;

    /// The cache row the next id lands in: the target's own position, held
    /// nowhere else.
    fn pos(&self) -> u32;

    /// The positions the target's caches hold; [`Target::pos`] never passes it.
    fn ctx(&self) -> u32;

    /// Run `ids` from [`Target::pos`] on by the target's prompt schedule and
    /// read back after the last; the target stands `ids.len()` positions on.
    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, Self::Error>;

    /// Run `id` at [`Target::pos`], one position, and read back after it.
    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, Self::Error>;

    /// The longest prefix of at most `n` positions [`Target::cut`] keeps.
    fn keepable(&self, n: u32) -> u32;

    /// [`Target::keepable`] with its reason. A target whose rule states none
    /// says [`seqstate::Why::Rule`].
    fn kept(&self, n: u32) -> seqstate::Kept {
        seqstate::Kept::rule(n, self.pos(), self.keepable(n))
    }

    /// Take back the positions from `n` on; refused by name unless
    /// [`Target::keepable`] grants `n` whole.
    fn cut(&mut self, n: u32) -> Result<(), Self::Error>;

    /// Empty caches at position 0.
    fn reset(&mut self) -> Result<(), Self::Error>;
}

/// A target that runs several consecutive positions as one pass: what a
/// draft's proposal is verified by.
pub trait Verify: Target {
    /// The most rows one pass runs, 2 or more.
    const MAX_ROWS: usize;

    /// Run `rows[r]` at position `pos + r`, for the target's position `pos`,
    /// as one pass of `M` rows and return each row's argmax; the target stands
    /// `M` positions on until [`Verify::commit`], which must come next. `M` is
    /// 2 to [`Verify::MAX_ROWS`], or the call does not compile.
    fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], Self::Error>;

    /// Keep the first `accepted` rows of the last verify, 1 to its `M`; the
    /// positions of the rest are taken back.
    fn commit(&mut self, accepted: usize) -> Result<(), Self::Error>;
}

/// What one pass of an [`Advance`] kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Committed {
    /// The target's position before the pass: kept token `r` is the argmax
    /// after position `pos + r`.
    pub pos: u32,
    /// The rows the pass kept on the target, 1 to `rows`.
    pub kept: usize,
    /// The rows the pass ran: 1 for a step, the verify's `M` for a proposal.
    pub rows: usize,
    /// Whether a proposal was verified; a pass with none is one step.
    pub proposed: bool,
}

/// How a generation moves on from the prompt.
pub trait Advance<T: Target> {
    /// The most rows one pass runs: the loop stops at the context when that
    /// many positions do not fit the caches. A one-step advance keeps 1; one
    /// that runs more and does not say so ends its last pass in the target's
    /// refusal at the caches' end.
    const ROWS: usize = 1;

    /// Feed the prompt `ids` through `t` from where it stands, with whatever
    /// this advance takes of it; the argmax after the last id.
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error>;

    /// Generation begins after `prompt`, whose argmax `first` is generated
    /// token 0.
    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error>;

    /// One pass from `last`, the token at the target's position: the tokens it
    /// keeps are appended to `out`, in position order.
    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error>;
}

/// One step a pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plain;

impl<T: Target> Advance<T> for Plain {
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        Ok(t.prompt(ids, Want::Argmax)?.argmax())
    }

    fn begin(&mut self, _t: &T, _prompt: &[u32], _first: u32) -> Result<(), T::Error> {
        Ok(())
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        let pos = t.pos();
        out.push(t.step(last, Want::Argmax)?.argmax());
        Ok(Committed {
            pos,
            kept: 1,
            rows: 1,
            proposed: false,
        })
    }
}

/// What [`generate`] tells its caller around the passes.
pub trait PassSink<T: Target> {
    /// What the sink fails with; a target's error converts into it.
    type Error: From<T::Error>;

    /// Before the first pass, after [`Advance::begin`].
    fn begin(&mut self, t: &T) -> Result<(), Self::Error>;

    /// After each pass: what it kept, its `tokens` in position order, and the
    /// wall time around the advance. The tokens end at an end-of-generation
    /// id: fewer than `c.kept` when one came before the pass's last row.
    fn pass(
        &mut self,
        t: &T,
        c: &Committed,
        tokens: &[u32],
        wall: Duration,
    ) -> Result<(), Self::Error>;
}

/// What a generation ended with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenOutcome {
    /// Every token the passes kept, generated token 0 first, up to the first
    /// end-of-generation id. A pass keeps whole rows, so the last may carry
    /// the count past the stop's length, and the target may stand past an
    /// end-of-generation id its pass kept before its last row.
    pub tokens: Vec<u32>,
    pub stop: StopReason,
    pub passes: usize,
}

/// The one generation loop, after the prompt: `first` is the prompt call's
/// argmax, generated token 0. Before each pass the [`Stop`] rule reads the
/// tokens so far, the last of them, the target's position and the advance's
/// [`Advance::ROWS`]; after it, the pass's tokens end at the first
/// end-of-generation id among them, which the next check stops at. Each pass
/// is timed around the advance alone, and `sink` hears of it after the wall
/// is taken. An error of the target, the advance or the sink ends the loop at
/// once.
pub fn generate<T, A, S>(
    t: &mut T,
    a: &mut A,
    prompt: &[u32],
    first: u32,
    stop: &Stop,
    sink: &mut S,
) -> Result<GenOutcome, S::Error>
where
    T: Target,
    A: Advance<T>,
    S: PassSink<T>,
{
    a.begin(t, prompt, first)?;
    sink.begin(t)?;
    let mut tokens = Vec::with_capacity(stop.max_tokens());
    tokens.push(first);
    let mut passes = 0;
    let reason = loop {
        let last = tokens[tokens.len() - 1];
        if let Some(r) = stop.check(tokens.len(), last, t.pos(), A::ROWS) {
            break r;
        }
        let from = tokens.len();
        let t0 = Instant::now();
        let c = a.pass(t, last, &mut tokens)?;
        let wall = t0.elapsed();
        passes += 1;
        if let Some(k) = stop.first_eog(&tokens[from..]) {
            tokens.truncate(from + k + 1);
        }
        sink.pass(t, &c, &tokens[from..], wall)?;
    };
    Ok(GenOutcome {
        tokens,
        stop: reason,
        passes,
    })
}

#[cfg(test)]
pub(crate) mod mock;
