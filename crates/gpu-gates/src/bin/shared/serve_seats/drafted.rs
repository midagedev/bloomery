//! The seats' MTP drafts, one a resident slot ([`SlotDrafts`]): each slot's
//! prompt, steps, passes and reset through the shared window
//! ([`app::mtp::MtpDraft`]) when the seat drives one, the session's own calls
//! when it does not. A seat holds one [`SlotDrafts`] beside its session and
//! forwards the [`Seat`](bloomery_gpu_gates::bind::Seat) calls that move
//! positions to it, each naming the slot the session has selected: a call
//! acts on that slot's entry alone. A round of several slots' drafted passes
//! takes every busy slot's window out of the table at once
//! (`serve_seats::rounds::pass_rows_one_pass`).
//!
//! The table never selects and nothing in it moves on a select: the seat
//! selects through the session, the draft's device side (its store and the
//! arenas its waiting rows sit in) travelling with the slot's sequence, and
//! the slot's entry stays where it is, so the slot's next call drafts as if
//! no other slot had run. The seat opens each slot's draft (its prompt path
//! and step mode are the seat's), and the table drives it.
//!
//! The draft rejoins a sequence only where its last call left it. A seat
//! whose saved state keeps the draft's side (its store and the arenas its
//! waiting rows sit in) parks the slot's draft with the state
//! ([`SlotDrafts::park`]) and puts it back with it ([`SlotDrafts::unpark`]),
//! so a returning sequence drafts as if no other had run. A seat whose state
//! does not keep that side turns the slot's draft off after a cut or a
//! put-back state ([`SlotDrafts::turn_off`]): every call on that slot is the
//! session's own until the slot's next reset, and each prompt call (or the
//! first step when the call is empty) prints an `mtp prompt` record that
//! names why.

use app::mtp::{MtpBody, MtpDraft, Parked};
use app::{RowsLog, Session, SessionError};
use bloomery_gpu::GpuModel;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::record::{self, Record};
use runtime::width::{Choosing, Chosen, Mode as WidthMode};
use runtime::{
    Advance, Argmax, Draft, Pick, Plain, Sample, Speculative, Target as _, Want, Widths, Window,
};
use serve::{Drafted, Sampler};

/// A slot's draft side of a saved sequence ([`SlotDrafts::park`]): what the
/// draft held of it, and why the seat had turned it off, if it had.
#[derive(Clone, Debug)]
pub(crate) struct ParkedDraft<A> {
    draft: Parked<A>,
    off: Option<&'static str>,
}

/// One slot's drafted window over its own [`MtpDraft`] behind the shared
/// width chooser (`runtime::width`), which `fixed` passes through untouched:
/// every family's seat drives the one type.
type Spec<B, const M: usize> = Speculative<Choosing<MtpDraft<B>>, M>;

/// The drafts a seat drives its session's slots with, one a resident slot,
/// `M` = the body's [`MtpBody::VERIFY_ROWS`], or none: every pass a plain
/// step.
pub(crate) struct SlotDrafts<B: MtpBody, const M: usize>
where
    Window<M>: Widths,
{
    /// Each slot's window over its own draft; empty without a draft. An
    /// entry is `None` only while a round holds its window out
    /// ([`SlotDrafts::specs_mut`]).
    specs: Vec<Option<Spec<B, M>>>,
    /// Why each slot's draft is off until the slot's next reset, and whether
    /// a record has said so since it was turned off; `None` while it drafts.
    off: Vec<Option<(&'static str, bool)>>,
}

impl<B: MtpBody, const M: usize> SlotDrafts<B, M>
where
    Window<M>: Widths,
{
    /// No draft: every call the session's own, every pass one plain step.
    pub(crate) fn none() -> Self {
        SlotDrafts {
            specs: Vec::new(),
            off: Vec::new(),
        }
    }

    /// One draft a resident slot, `slots` of them, each opened by the seat's
    /// `open` (the draft's prompt path and step mode are the seat's) and set
    /// behind the width chooser of `mode` (`runtime::width`): slot
    /// 0's driven through [`Session::with_draft`], which captures every
    /// width's verify pass on the model (`log` told of each), each further
    /// slot's opened beside it, the model's captures shared by every slot's
    /// draft. Every slot starts as a fresh draft: nothing waits, nothing
    /// skipped. Load-time, once the session serves its slots; refused by name
    /// for no slot and a rule the chooser cannot run by.
    pub(crate) fn open(
        s: &mut Session<B>,
        slots: usize,
        mode: WidthMode,
        log: &mut impl RowsLog,
        mut open: impl FnMut(&GpuModel<B>) -> Result<MtpDraft<B>, SessionError>,
    ) -> Result<Self, GateError> {
        if slots == 0 {
            return Err("the MTP drafts of no slot".into());
        }
        let choose = |d: MtpDraft<B>| -> Result<Choosing<MtpDraft<B>>, SessionError> {
            Ok(d.choosing(mode)?)
        };
        let first = choose(open(s.model())?)?;
        let mut specs = Vec::with_capacity(slots);
        specs.push(Some(s.with_draft(first, log)?));
        for _ in 1..slots {
            specs.push(Some(Speculative::new(choose(open(s.model())?)?)));
        }
        Ok(SlotDrafts {
            specs,
            off: vec![None; slots],
        })
    }

    /// Whether a draft runs.
    pub(crate) fn drafts(&self) -> bool {
        !self.specs.is_empty()
    }

    /// Whether `slot` drafts through a window of the table: false with no
    /// draft. Refused by name for a slot past the table, or one whose window
    /// a round holds out.
    fn has(&self, slot: usize) -> Result<bool, GateError> {
        if self.specs.is_empty() {
            return Ok(false);
        }
        match self.specs.get(slot) {
            Some(Some(_)) => Ok(true),
            Some(None) => Err(format!(
                "slot {slot}: its MTP draft is out with a round of several slots"
            )
            .into()),
            None => Err(format!(
                "slot {slot}: the MTP drafts hold {} slots",
                self.specs.len()
            )
            .into()),
        }
    }

    /// `slot`'s window; `None` with no draft, refused as [`SlotDrafts::has`].
    fn spec_mut(&mut self, slot: usize) -> Result<Option<&mut Spec<B, M>>, GateError> {
        Ok(if self.has(slot)? {
            self.specs[slot].as_mut()
        } else {
            None
        })
    }

    /// Whether `slot`'s draft is off until the slot's next reset
    /// ([`SlotDrafts::turn_off`]); false for a slot the table holds no draft
    /// of.
    pub(crate) fn is_off(&self, slot: usize) -> bool {
        self.off.get(slot).is_some_and(Option::is_some)
    }

    /// `slot`'s draft off until the slot's next reset, for `why`: the
    /// session's position moved where the draft holds no rows to rejoin at.
    /// The chooser's passes over the request before are closed out first
    /// (an `mtp width` record under the cost mode): a call the draft sits
    /// out prints none. Nothing without a draft; refused as
    /// [`SlotDrafts::has`].
    pub(crate) fn turn_off(&mut self, slot: usize, why: &'static str) -> Result<(), GateError> {
        if self.has(slot)? {
            if let Some(spec) = self.specs[slot].as_mut() {
                print_widths(spec);
            }
            self.off[slot] = Some((why, false));
        }
        Ok(())
    }

    /// The `mtp prompt` record of a call `slot`'s draft sits out, at the
    /// session's position.
    fn print_off(&mut self, s: &Session<B>, slot: usize) {
        if let Some(Some((why, told))) = self.off.get_mut(slot) {
            *told = true;
            Record::new(&record::MTP_PROMPT)
                .u("start", s.pos())
                .u("caught_up", 0)
                .w("skipped", *why)
                .eprint();
        }
    }

    /// The prompt on `slot`, the session's selected one: the chooser's
    /// passes over the request that just left the slot closed out first —
    /// an `mtp width` record under the cost mode, none under `fixed` —
    /// then under the draft the draft's own prompt call, its store walked
    /// over the prompt's units, then its join record; without it, or while
    /// it is off, the session's prompt call.
    pub(crate) fn prefill(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        ids: &[u32],
    ) -> Result<u32, GateError> {
        if self.is_off(slot) {
            self.print_off(s, slot);
            return Ok(s.prompt(ids, Want::Argmax)?.argmax());
        }
        match self.spec_mut(slot)? {
            Some(spec) => {
                print_widths(spec);
                let next = Advance::prompt(spec, s, ids)?;
                print_join(spec);
                Ok(next)
            }
            None => Ok(s.prompt(ids, Want::Argmax)?.argmax()),
        }
    }

    /// One step on `slot`, the session's selected one; under the draft the
    /// rows it left waiting walked first ([`MtpDraft::before_step`]: a
    /// request that continues the held sequence joins it here when its
    /// prompt call is empty), then the step told to the draft
    /// ([`Draft::stepped`]).
    pub(crate) fn step(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
    ) -> Result<u32, GateError> {
        self.step_reading(s, slot, last, None)
    }

    /// [`SlotDrafts::step`] with the target's logits of the step into `row`
    /// (`n_vocab` f32), read after the target's step and before the step is
    /// told to the draft: whatever the draft's walks write, the row is the
    /// target's.
    pub(crate) fn step_with_row(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        row: &mut [f32],
    ) -> Result<u32, GateError> {
        self.step_reading(s, slot, last, Some(row))
    }

    /// The step's order, the one owner of it: the draft's waiting rows, the
    /// target's step, the target's row when asked, then the draft told.
    fn step_reading(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        row: Option<&mut [f32]>,
    ) -> Result<u32, GateError> {
        if matches!(self.off.get(slot), Some(Some((_, false)))) {
            self.print_off(s, slot);
        }
        let drafting = !self.is_off(slot);
        if let Some(spec) = self.spec_mut(slot)?.filter(|_| drafting) {
            print_widths(spec);
            spec.draft_mut().draft_mut().before_step(s, last)?;
            print_join(spec);
        }
        let next = s.step(last, Want::Argmax)?.argmax();
        if let Some(row) = row {
            s.model()
                .logits_into(row)
                .map_err(|e| format!("logits of the step: {e}"))?;
        }
        if let Some(spec) = self.spec_mut(slot)?.filter(|_| drafting) {
            Draft::stepped(spec.draft_mut(), s, last, next)?;
        }
        Ok(next)
    }

    /// One greedy pass on `slot`, the session's selected one, from `last`
    /// ([`SlotDrafts::pass_taking`] by each row's argmax).
    pub(crate) fn pass(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        self.pass_taking(s, slot, last, &mut Argmax, out)
    }

    /// One sampled pass on `slot`, the session's selected one, from `last`
    /// (`serve::Engine::advance_sampled`): [`SlotDrafts::pass_taking`] with
    /// each row's id drawn by `sampler` from that row's logits, given
    /// `history` and the ids this pass took before it. `history` comes back
    /// as it was given.
    pub(crate) fn pass_sampled(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut Sampler,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let from = history.len();
        let mut pick = Sample(|row: &[f32]| {
            let id = sampler(row, history);
            history.push(id);
            id
        });
        let d = self.pass_taking(s, slot, last, &mut pick, out);
        history.truncate(from);
        d
    }

    /// One pass on `slot`, the session's selected one, from `last`, each
    /// row's id taken by `pick`: under the draft one window
    /// ([`Speculative::pass_picking`]), its taken ids and counts; without
    /// it, or while it is off, one step ([`SlotDrafts::step`]: no draft walk
    /// follows it) and its row's id.
    fn pass_taking<P: Pick<Session<B>>>(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        pick: &mut P,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let drafting = !self.is_off(slot);
        match self.spec_mut(slot)?.filter(|_| drafting) {
            Some(spec) => {
                let c = spec.pass_picking(s, last, pick, out)?;
                Ok(Drafted {
                    proposed: if c.proposed { c.rows - 1 } else { 0 },
                    accepted: c.kept - 1,
                })
            }
            None => {
                let argmax = self.step(s, slot, last)?;
                out.push(pick.pick(s, 0, argmax)?);
                Ok(Drafted::default())
            }
        }
    }

    /// The most positions one pass on `slot` runs: a window's rows while its
    /// draft runs, one step's without it or while it is off.
    pub(crate) fn pass_rows(&self, slot: usize) -> usize {
        match (self.specs.get(slot), self.is_off(slot)) {
            (Some(Some(_)), false) => <Spec<B, M> as Advance<Session<B>>>::ROWS,
            _ => <Plain as Advance<Session<B>>>::ROWS,
        }
    }

    /// The session's reset, then `slot`'s draft — the session's selected
    /// slot's, the sequence the reset emptied — started over and on again;
    /// every other slot's draft stands as it is. Refused as
    /// [`SlotDrafts::has`] refuses, before the reset.
    pub(crate) fn reset(&mut self, s: &mut Session<B>, slot: usize) -> Result<(), GateError> {
        let drafting = self.has(slot)?;
        s.reset()?;
        if drafting {
            self.off[slot] = None;
            if let Some(spec) = self.specs[slot].as_mut() {
                print_widths(spec);
                spec.draft_mut().draft_mut().restart();
            }
        }
        Ok(())
    }

    /// `slot`'s draft side of the session's sequence, for a state the seat
    /// saves of that slot ([`MtpDraft::park`]; the turn-off why with it);
    /// `None` without a draft. The state keeps the draft's store and the
    /// arenas the waiting rows sit in. Refused as [`SlotDrafts::has`].
    pub(crate) fn park(&self, slot: usize) -> Result<Option<ParkedDraft<B::Arena>>, GateError> {
        if !self.has(slot)? {
            return Ok(None);
        }
        Ok(self.specs[slot].as_ref().map(|spec| ParkedDraft {
            draft: spec.draft().draft().park(),
            off: self.off[slot].map(|(why, _)| why),
        }))
    }

    /// Refused by name unless a state whose draft's side is `p` fits this
    /// seat: a side where a draft runs, none where none does.
    pub(crate) fn takes(&self, p: Option<&ParkedDraft<B::Arena>>) -> Result<(), GateError> {
        if p.is_some() == self.drafts() {
            return Ok(());
        }
        Err(format!(
            "a state saved with the MTP draft {} put back with it {}",
            if p.is_some() { "on" } else { "off" },
            if self.drafts() { "on" } else { "off" }
        )
        .into())
    }

    /// `p`, which [`SlotDrafts::park`] took with a state, back in `slot`'s
    /// entry once that state is put back on the slot after the session's
    /// reset: the draft joins the slot's next call where it left it, or
    /// stays off for the parked why. Refused by name as
    /// [`SlotDrafts::takes`] and [`SlotDrafts::has`] refuse.
    pub(crate) fn unpark(
        &mut self,
        slot: usize,
        p: Option<&ParkedDraft<B::Arena>>,
    ) -> Result<(), GateError> {
        self.takes(p)?;
        if let (Some(spec), Some(p)) = (self.spec_mut(slot)?, p) {
            spec.draft_mut().draft_mut().unpark(&p.draft);
            self.off[slot] = p.off.map(|why| (why, false));
        }
        Ok(())
    }

    /// Every slot's window, by slot, for a round that drives several slots'
    /// drafts at once (`serve_seats::rounds::pass_rows_one_pass`, which
    /// takes each busy slot's window out and puts it back); empty without a
    /// draft.
    pub(crate) fn specs_mut(&mut self) -> &mut [Option<Spec<B, M>>] {
        &mut self.specs
    }
}

/// The draft's join to the held sequence, when a call made one: how many
/// rows it caught up, or why it skips.
fn print_join<B: MtpBody, const M: usize>(spec: &mut Spec<B, M>) {
    if let Some(j) = spec.draft_mut().draft_mut().take_joined() {
        Record::new(&record::MTP_PROMPT)
            .u("start", j.start)
            .u("caught_up", j.caught_up)
            .w("skipped", j.skipped.unwrap_or("none"))
            .eprint();
    }
}

/// The chooser's passes over the request a slot just finished and its state
/// at the request's end, as its `mtp width` record: nothing under `fixed`,
/// whose passes are the draft's own, and nothing for a request that ran no
/// pass. The ds41 seat drains its own drafts the same way.
pub(crate) fn print_widths<D, const M: usize>(spec: &mut Speculative<Choosing<D>, M>) {
    let choosing = spec.draft_mut();
    if choosing.mode() != WidthMode::Cost {
        return;
    }
    let t = choosing.take_tally();
    if t.passes() == 0 {
        return;
    }
    let r = Record::new(&record::MTP_WIDTH)
        .u("windows", t.windows)
        .csv("kept", &t.kept)
        .csv("widths", &t.widths)
        .f("e", t.e());
    record::width_gate(r, &choosing.gate_state(), t.closed).eprint();
}
