//! The in-file MTP draft of a hybrid seat: one resident sequence's window ([`SlotDrafts`]) behind
//! the object a seat holds ([`SeatDraft`]), so the common hybrid seat serves a body's next-token layer
//! without a branch per family. A family names it ([`super::hybrid::Family::mtp`]); the seat opens it
//! under `BLOOMERY_DRAFT=mtp` and routes its prompt calls, steps and passes through it.

use app::Session;
use bloomery_gpu::model::{ChainBody, Slots};
use bloomery_gpu_gates::GateError;
use runtime::Target as _;
use runtime::prompt::CallEnd;
use serve::{Drafted, Sampler};

use app::mtp::MtpBody;

use super::drafted::SlotDrafts;

/// The draft of the seat's one sequence, driven through the session `s` the seat holds.
pub trait SeatDraft<B: ChainBody> {
    /// Whether the draft is running (not turned off).
    fn drafts(&self) -> bool;

    /// The ids a window proposes.
    fn width(&self) -> u64;

    /// The prompt call of `ids`, the draft's store walked over its units, the target's head run
    /// when `end` reads it.
    fn prompt_call(
        &mut self,
        s: &mut Session<B>,
        ids: &[u32],
        end: CallEnd,
    ) -> Result<u32, GateError>;

    /// One plain step, told to the draft.
    fn step(&mut self, s: &mut Session<B>, last: u32) -> Result<u32, GateError>;

    /// [`SeatDraft::step`] with the target's logits row of the step into `row`.
    fn step_row(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        row: &mut [f32],
    ) -> Result<u32, GateError>;

    /// One window's pass: the draft's proposal verified behind `last`, the kept ids into `out`.
    fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError>;

    /// The most positions one pass runs.
    fn pass_rows(&self, s: &Session<B>) -> usize;

    /// [`SeatDraft::pass`] of a request that samples: each kept id its draw from its verified row.
    #[allow(
        clippy::too_many_arguments,
        reason = "the sampled pass's operands, as `Seat::pass_sampled` names them"
    )]
    fn pass_sampled(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError>;

    /// The session's reset, the draft started over with it.
    fn reset(&mut self, s: &mut Session<B>) -> Result<(), GateError>;

    /// The session's cut to `pos`, which [`SeatDraft::keep`] granted: the draft's store stays
    /// through the entry at `pos`, and the next prompt call, from `pos` over the id that entry was
    /// written for, joins the draft to it. A cut to 0 is the session's reset.
    fn cut(&mut self, s: &mut Session<B>, pos: u32) -> Result<(), GateError>;

    /// The longest prefix of at most `n` positions a cut keeps (`body` the target's rule), and the
    /// rule that kept less. A cut that moves the position keeps one position less than the target's
    /// rule grants: the draft's entry at the first fed position reads the target's hidden row
    /// before it, which a cut takes back, so the prompt call feeds one more id and its pass writes
    /// that row; the entry at the cut's own position stays in the draft's store
    /// ([`SeatDraft::cut`]). Nothing is cut when every held position is kept, or from an empty
    /// model.
    fn keep(
        &self,
        s: &Session<B>,
        n: usize,
        body: &dyn Fn(usize) -> (usize, Option<String>),
    ) -> (usize, Option<String>);

    /// The draft's record printed.
    fn close(&mut self);
}

impl<B, const M: usize> SeatDraft<B> for SlotDrafts<B, M>
where
    B: MtpBody + Slots,
    B::Seq: 'static,
    runtime::Window<M>: runtime::Widths,
{
    fn drafts(&self) -> bool {
        SlotDrafts::drafts(self)
    }

    fn width(&self) -> u64 {
        <B as MtpBody>::WIDTH as u64
    }

    fn prompt_call(
        &mut self,
        s: &mut Session<B>,
        ids: &[u32],
        end: CallEnd,
    ) -> Result<u32, GateError> {
        let sel = s.selected();
        SlotDrafts::prompt_call(self, s, sel, ids, end)
    }

    fn step(&mut self, s: &mut Session<B>, last: u32) -> Result<u32, GateError> {
        let sel = s.selected();
        SlotDrafts::step(self, s, sel, last)
    }

    fn step_row(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        row: &mut [f32],
    ) -> Result<u32, GateError> {
        let sel = s.selected();
        self.step_with_row(s, sel, last, row)
    }

    fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let sel = s.selected();
        SlotDrafts::pass(self, s, sel, last, out)
    }

    fn pass_rows(&self, s: &Session<B>) -> usize {
        SlotDrafts::pass_rows(self, s.selected())
    }

    fn pass_sampled(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let sel = s.selected();
        SlotDrafts::pass_sampled(self, s, sel, last, history, sampler, best, out)
    }

    fn reset(&mut self, s: &mut Session<B>) -> Result<(), GateError> {
        let sel = s.selected();
        SlotDrafts::reset(self, s, sel)
    }

    fn cut(&mut self, s: &mut Session<B>, pos: u32) -> Result<(), GateError> {
        let sel = s.selected();
        if pos == 0 {
            return SlotDrafts::reset(self, s, sel);
        }
        let held = s.pos();
        s.cut(pos)?;
        if pos < held {
            self.rejoin_through(sel, pos)?;
        }
        Ok(())
    }

    fn keep(
        &self,
        s: &Session<B>,
        n: usize,
        body: &dyn Fn(usize) -> (usize, Option<String>),
    ) -> (usize, Option<String>) {
        if n == 0 || n >= s.pos() as usize || !SlotDrafts::mtp(self) {
            return body(n);
        }
        let (k, why) = body(n - 1);
        let why = format!(
            "the MTP draft's entry at the first fed position reads the target's hidden row before \
             it, which a cut takes back, so the prompt call feeds one more id: cut to at most {}{}",
            n - 1,
            why.map(|w| format!(", then {w}")).unwrap_or_default()
        );
        (k, Some(why))
    }

    fn close(&mut self) {
        SlotDrafts::close(self);
    }
}
