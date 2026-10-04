//! A seat's MTP draft: the session's prompt, steps, passes and reset through
//! the shared window ([`app::mtp::MtpDraft`]) when the seat drives one, the
//! session's own calls when it does not. A seat holds one [`DraftedSeat`]
//! beside its session and forwards the [`Seat`](bloomery_gpu_gates::bind::Seat)
//! calls that move positions to it.
//!
//! The draft rejoins a sequence only where its last call left it. A seat
//! whose saved state keeps the draft's side (its store and the arenas its
//! waiting rows sit in) parks the draft with the state ([`DraftedSeat::park`])
//! and puts it back with it ([`DraftedSeat::unpark`]), so a returning
//! sequence drafts as if no other had run. A seat whose state does not keep
//! that side turns the draft off after a cut or a put-back state
//! ([`DraftedSeat::turn_off`]): every call is the session's own until the next
//! reset, and each prompt call (or the first step when the call is empty)
//! prints an `mtp prompt` record that names why.

use app::Session;
use app::mtp::{MtpBody, MtpDraft, Parked};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::record::{self, Record};
use runtime::{Advance, Draft, Plain, Speculative, Target as _, Want, Widths, Window};
use serve::Drafted;

/// The draft's side of a saved sequence ([`DraftedSeat::park`]): what the
/// draft held of it, and why the seat had turned it off, if it had.
#[derive(Clone, Debug)]
pub(crate) struct ParkedDraft<A> {
    draft: Parked<A>,
    off: Option<&'static str>,
}

/// The windows a seat drives its session with, `M` = the body's
/// [`MtpBody::VERIFY_ROWS`], or none: every pass a plain step.
pub(crate) struct DraftedSeat<B: MtpBody, const M: usize>
where
    Window<M>: Widths,
{
    spec: Option<Speculative<MtpDraft<B>, M>>,
    /// Why the draft is off until the next reset, and whether a record has
    /// said so since it was turned off; `None` while it drafts.
    off: Option<(&'static str, bool)>,
}

impl<B: MtpBody, const M: usize> DraftedSeat<B, M>
where
    Window<M>: Widths,
{
    /// The seat over `spec`'s windows, or with no draft.
    pub(crate) fn new(spec: Option<Speculative<MtpDraft<B>, M>>) -> Self {
        DraftedSeat { spec, off: None }
    }

    /// The draft off until the next reset, for `why`: the session's position
    /// moved where the draft holds no rows to rejoin at. Nothing without a
    /// draft.
    pub(crate) fn turn_off(&mut self, why: &'static str) {
        if self.spec.is_some() {
            self.off = Some((why, false));
        }
    }

    /// The `mtp prompt` record of a call the draft sits out, at the session's
    /// position.
    fn print_off(&mut self, s: &Session<B>) {
        if let Some((why, told)) = &mut self.off {
            *told = true;
            Record::new(&record::MTP_PROMPT)
                .u("start", s.pos())
                .u("caught_up", 0)
                .w("skipped", *why)
                .eprint();
        }
    }

    /// Whether a draft runs.
    pub(crate) fn drafts(&self) -> bool {
        self.spec.is_some()
    }

    /// Whether the draft is off until the next reset ([`DraftedSeat::turn_off`]).
    pub(crate) fn is_off(&self) -> bool {
        self.off.is_some()
    }

    /// The prompt: under the draft the draft's own prompt call, its store
    /// walked over the prompt's units, then its join record; without it the
    /// session's prompt call.
    pub(crate) fn prefill(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, GateError> {
        if self.off.is_some() {
            self.print_off(s);
            return Ok(s.prompt(ids, Want::Argmax)?.argmax());
        }
        match &mut self.spec {
            Some(spec) => {
                let next = Advance::prompt(spec, s, ids)?;
                print_join(spec);
                Ok(next)
            }
            None => Ok(s.prompt(ids, Want::Argmax)?.argmax()),
        }
    }

    /// One step; under the draft the rows it left waiting walked first
    /// ([`MtpDraft::before_step`]: a request that continues the held
    /// sequence joins it here when its prompt call is empty), then the step
    /// told to the draft ([`Draft::stepped`]).
    pub(crate) fn step(&mut self, s: &mut Session<B>, last: u32) -> Result<u32, GateError> {
        self.step_reading(s, last, None)
    }

    /// [`DraftedSeat::step`] with the target's logits of the step into `row`
    /// (`n_vocab` f32), read after the target's step and before the step is
    /// told to the draft: whatever the draft's walks write, the row is the
    /// target's.
    pub(crate) fn step_with_row(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        row: &mut [f32],
    ) -> Result<u32, GateError> {
        self.step_reading(s, last, Some(row))
    }

    /// The step's order, the one owner of it: the draft's waiting rows, the
    /// target's step, the target's row when asked, then the draft told.
    fn step_reading(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        row: Option<&mut [f32]>,
    ) -> Result<u32, GateError> {
        if matches!(self.off, Some((_, false))) {
            self.print_off(s);
        }
        let drafting = self.off.is_none();
        if let Some(spec) = self.spec.as_mut().filter(|_| drafting) {
            spec.draft_mut().before_step(s, last)?;
            print_join(spec);
        }
        let next = s.step(last, Want::Argmax)?.argmax();
        if let Some(row) = row {
            s.model()
                .logits_into(row)
                .map_err(|e| format!("logits of the step: {e}"))?;
        }
        if let Some(spec) = self.spec.as_mut().filter(|_| drafting) {
            Draft::stepped(spec.draft_mut(), s, last, next)?;
        }
        Ok(next)
    }

    /// One pass from `last`: under the draft one window, its kept tokens and
    /// counts; without it one step ([`DraftedSeat::step`]).
    pub(crate) fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        match self.spec.as_mut().filter(|_| self.off.is_none()) {
            Some(spec) => {
                let c = Advance::pass(spec, s, last, out)?;
                Ok(Drafted {
                    proposed: if c.proposed { c.rows - 1 } else { 0 },
                    accepted: c.kept - 1,
                })
            }
            None => {
                out.push(self.step(s, last)?);
                Ok(Drafted::default())
            }
        }
    }

    /// The most positions one pass runs: a window's rows while the draft
    /// runs, one step's without it or while it is off.
    pub(crate) fn pass_rows(&self) -> usize {
        match (&self.spec, self.off) {
            (Some(_), None) => <Speculative<MtpDraft<B>, M> as Advance<Session<B>>>::ROWS,
            _ => <Plain as Advance<Session<B>>>::ROWS,
        }
    }

    /// The session's reset, then the draft started over and on again.
    pub(crate) fn reset(&mut self, s: &mut Session<B>) -> Result<(), GateError> {
        s.reset()?;
        self.off = None;
        if let Some(spec) = &mut self.spec {
            spec.draft_mut().restart();
        }
        Ok(())
    }
}

impl<B: MtpBody, const M: usize> DraftedSeat<B, M>
where
    Window<M>: Widths,
{
    /// The draft's side of the session's sequence, for a state the seat
    /// saves ([`MtpDraft::park`]; the turn-off why with it); `None` without
    /// a draft. The state keeps the draft's store and the arenas the
    /// waiting rows sit in.
    pub(crate) fn park(&self) -> Option<ParkedDraft<B::Arena>> {
        self.spec.as_ref().map(|spec| ParkedDraft {
            draft: spec.draft().park(),
            off: self.off.map(|(why, _)| why),
        })
    }

    /// Refused by name unless a state whose draft's side is `p` fits this
    /// seat: a side where a draft runs, none where none does.
    pub(crate) fn takes(&self, p: Option<&ParkedDraft<B::Arena>>) -> Result<(), GateError> {
        if p.is_some() == self.spec.is_some() {
            return Ok(());
        }
        Err(format!(
            "a state saved with the MTP draft {} put back with it {}",
            if p.is_some() { "on" } else { "off" },
            if self.spec.is_some() { "on" } else { "off" }
        )
        .into())
    }

    /// `p`, which [`DraftedSeat::park`] took with a state, back in place once
    /// that state is put back after the session's reset: the draft joins the
    /// sequence's next call where it left it, or stays off for the parked
    /// why. Refused by name as [`DraftedSeat::takes`] refuses.
    pub(crate) fn unpark(&mut self, p: Option<&ParkedDraft<B::Arena>>) -> Result<(), GateError> {
        self.takes(p)?;
        if let (Some(spec), Some(p)) = (&mut self.spec, p) {
            spec.draft_mut().unpark(&p.draft);
            self.off = p.off.map(|why| (why, false));
        }
        Ok(())
    }
}

/// The draft's join to the held sequence, when a call made one: how many
/// rows it caught up, or why it skips.
fn print_join<B: MtpBody, const M: usize>(spec: &mut Speculative<MtpDraft<B>, M>) {
    if let Some(j) = spec.draft_mut().take_joined() {
        Record::new(&record::MTP_PROMPT)
            .u("start", j.start)
            .u("caught_up", j.caught_up)
            .w("skipped", j.skipped.unwrap_or("none"))
            .eprint();
    }
}
