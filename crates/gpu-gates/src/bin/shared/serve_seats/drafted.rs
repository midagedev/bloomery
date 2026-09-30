//! A seat's MTP draft: the session's prompt, steps, passes and reset through
//! the shared window ([`app::mtp::MtpDraft`]) when the seat drives one, the
//! session's own calls when it does not. A seat holds one [`DraftedSeat`]
//! beside its session and forwards the [`Seat`](bloomery_gpu_gates::bind::Seat)
//! calls that move positions to it.

use app::Session;
use app::mtp::{MtpBody, MtpDraft};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::record::{self, Record};
use runtime::{Advance, Draft, Plain, Speculative, Target as _, Want, Widths, Window};
use serve::Drafted;

/// The windows a seat drives its session with, `M` = the body's
/// [`MtpBody::VERIFY_ROWS`], or none: every pass a plain step.
pub(crate) struct DraftedSeat<B: MtpBody, const M: usize>
where
    Window<M>: Widths,
{
    spec: Option<Speculative<MtpDraft<B>, M>>,
}

impl<B: MtpBody, const M: usize> DraftedSeat<B, M>
where
    Window<M>: Widths,
{
    /// The seat over `spec`'s windows, or with no draft.
    pub(crate) fn new(spec: Option<Speculative<MtpDraft<B>, M>>) -> Self {
        DraftedSeat { spec }
    }

    /// Whether a draft runs.
    pub(crate) fn drafts(&self) -> bool {
        self.spec.is_some()
    }

    /// The prompt: under the draft the draft's own prompt call, its store
    /// walked over the prompt's units, then its join record; without it the
    /// session's prompt call.
    pub(crate) fn prefill(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, GateError> {
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
        if let Some(spec) = &mut self.spec {
            spec.draft_mut().before_step(s, last)?;
            print_join(spec);
        }
        let next = s.step(last, Want::Argmax)?.argmax();
        if let Some(spec) = &mut self.spec {
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
        match &mut self.spec {
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

    /// The most positions one pass runs: a window's rows under the draft,
    /// one step's without it.
    pub(crate) fn pass_rows(&self) -> usize {
        match self.spec {
            Some(_) => <Speculative<MtpDraft<B>, M> as Advance<Session<B>>>::ROWS,
            None => <Plain as Advance<Session<B>>>::ROWS,
        }
    }

    /// The session's reset, then the draft started over.
    pub(crate) fn reset(&mut self, s: &mut Session<B>) -> Result<(), GateError> {
        s.reset()?;
        if let Some(spec) = &mut self.spec {
            spec.draft_mut().restart();
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
