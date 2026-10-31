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
//! The draft rejoins a sequence where its last call left it, or where the
//! body kept its side. A seat whose saved state keeps the draft's side (its
//! store and the arenas its waiting rows sit in) parks the slot's draft with
//! the state ([`SlotDrafts::park`]) and puts it back with it
//! ([`SlotDrafts::unpark`]), so a returning sequence drafts as if no other
//! had run. After a cut, or a state put back as its prefix, a seat whose
//! body kept the draft's store below the kept position and the target's
//! hidden row before it rejoins the slot's draft there
//! ([`SlotDrafts::rejoin`]); otherwise it turns the slot's draft off
//! ([`SlotDrafts::turn_off`]): every call on that slot is the session's own
//! until the slot's next reset, and each prompt call (or the first step when
//! the call is empty) prints an `mtp prompt` record that names why.
//!
//! The lookup draft (`runtime::Lookup`, `BLOOMERY_DRAFT=lookup`) has one
//! owner too, [`Lookups`]: it needs no weights and no draft side in the
//! body, only a body whose session verifies rows, so every seat whose body
//! has a multi-row pass serves it through this one type, the MTP table
//! ([`SlotDrafts::lookup`]) and the served draft (`served::ServedDraft`)
//! alike.

use app::mtp::{MtpBody, MtpDraft, Parked, SlotWalks};
use app::{Keep, Prompt, RowsLog, Session, SessionError};
use bloomery_gpu::GpuModel;
use bloomery_gpu::best;
use bloomery_gpu::checkpoint::bytes;
use bloomery_gpu::model::{Rollback, Rows, Slots};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::record::{self, Record};
use runtime::width::{Choosing, Chosen, Clock, Mode as WidthMode, Wall};
use runtime::{
    Advance, Argmax, Committed, Draft, Lookup, Pick, Plain, RowLogits, Sample, SampleBest,
    Speculative, Target, Verify, Want, WidthError, Widths, Window,
};
use serve::{DraftProps, Drafted, EngineProps, Sampler};

/// The rows a lookup pass verifies: the token at the target's position and
/// the lookup's one proposed id (its `WIDTH` + 1, which
/// [`Session::with_draft`] holds it to).
pub(crate) const LOOKUP_ROWS: usize = 2;

/// The selected slot's token history, for a target that keeps one: the ids
/// of the positions before the target's, in order.
pub(crate) type History<T> = fn(&T) -> Option<&[u32]>;

/// How a lookup pass takes each verified row's id.
pub(crate) enum Taking<'a> {
    /// The row's argmax.
    Argmax,
    /// Drawn from the row's logits, read whole.
    Sample(&'a mut dyn FnMut(&[f32]) -> u32),
    /// Drawn from the row read as its best logits, for a draw that reads only
    /// those ([`reads_best`]).
    SampleBest(&'a mut dyn FnMut(&[f32]) -> u32),
}

/// Whether a draw that reads at most `best` of a row's best logits
/// (`serve::Engine::advance_sampled`'s `best`) takes each row as the card's
/// best read of it ([`bloomery_gpu::best`]), which holds [`best::BEST`].
pub(crate) fn reads_best(best: Option<usize>) -> bool {
    best.is_some_and(|k| k <= best::BEST)
}

/// A slot's lookup window behind its width chooser, as [`Lookups`] drives
/// it: a trait object, so the owner is generic over its target alone.
trait LookupWindow<T> {
    fn lookup(&self) -> &Lookup;
    fn lookup_mut(&mut self) -> &mut Lookup;
    /// The chooser's record of the request before printed
    /// ([`print_widths`]), then its pass in flight discarded.
    fn close(&mut self);
    /// One pass from `last`, its wall landed at its end
    /// ([`Choosing::finish_pass`]): another slot's call before this slot's
    /// next pass is no part of it.
    fn pass(
        &mut self,
        t: &mut T,
        last: u32,
        take: Taking<'_>,
        out: &mut Vec<u32>,
    ) -> Result<Committed, GateError>;
    #[cfg(test)]
    fn gate_state(&self) -> runtime::width::GateState;
    /// The passes the chooser holds no record of yet, taken.
    #[cfg(test)]
    fn unprinted(&mut self) -> u64;
}

impl<T, C> LookupWindow<T> for Speculative<Choosing<Lookup, C>, LOOKUP_ROWS>
where
    T: Verify + RowLogits,
    T::Error: From<WidthError> + std::error::Error + 'static,
    C: Clock,
{
    fn lookup(&self) -> &Lookup {
        self.draft().draft()
    }

    fn lookup_mut(&mut self) -> &mut Lookup {
        self.draft_mut().draft_mut()
    }

    fn close(&mut self) {
        print_widths(self);
        self.draft_mut().prompt_by(|_| ());
    }

    fn pass(
        &mut self,
        t: &mut T,
        last: u32,
        take: Taking<'_>,
        out: &mut Vec<u32>,
    ) -> Result<Committed, GateError> {
        Ok(match take {
            Taking::Argmax => pass_in_turn(self, t, last, &mut Argmax, out),
            Taking::Sample(f) => {
                pass_in_turn(self, t, last, &mut Sample(|row: &[f32]| f(row)), out)
            }
            Taking::SampleBest(f) => {
                pass_in_turn(self, t, last, &mut SampleBest(|row: &[f32]| f(row)), out)
            }
        }?)
    }

    #[cfg(test)]
    fn gate_state(&self) -> runtime::width::GateState {
        self.draft().gate_state()
    }

    #[cfg(test)]
    fn unprinted(&mut self) -> u64 {
        self.draft_mut().take_tally().passes()
    }
}

/// A single slot's pass, completed before the caller runs another slot.
fn pass_in_turn<T, D, C, P, const M: usize>(
    spec: &mut Speculative<Choosing<D, C>, M>,
    t: &mut T,
    last: u32,
    pick: &mut P,
    out: &mut Vec<u32>,
) -> Result<Committed, T::Error>
where
    T: Verify,
    T::Error: From<WidthError>,
    D: Draft<T>,
    C: Clock,
    P: Pick<T>,
    Window<M>: Widths,
{
    let committed = spec.pass_picking(t, last, pick, out)?;
    spec.draft_mut().finish_pass(t.pos())?;
    Ok(committed)
}

/// A call on a slot's sequence, by what it closes of the slot's chooser
/// ([`Lookups::at`]).
#[derive(Clone, Copy)]
enum Call {
    /// The window's own pass: nothing closed. The slot's pass before landed
    /// at its end ([`LookupWindow::pass`]), and every other slot's calls go
    /// to their own chooser.
    Pass,
    /// Any other call that moves or replaces the slot's sequence (a prompt
    /// call, a plain step, a reset, a state put back): the chooser's record
    /// of the request before is printed and the pass in flight discarded,
    /// so no wall that holds the call is read as a pass. Every request
    /// opens with one, its prompt's last id stepped.
    Moved,
}

/// The lookup draft a seat serves, its one owner: each resident slot's
/// window over its own context behind its own width chooser, as each slot
/// has its own MTP draft ([`SlotDrafts::open`]), verified by the target's
/// pass of [`LOOKUP_ROWS`] rows. A chooser's acceptance, costs, pass in
/// flight and request record are its slot's alone: slots that take turns
/// neither drop nor mix each other's. Every call reaches a slot's context
/// through [`Lookups::at`], which closes the slot's chooser for the call's
/// kind ([`Call`]) first.
/// A context follows its slot by the positions the calls stand at
/// ([`Lookup::stand`]): a cut back needs nothing told, the next call keeps
/// the context below its position. A slot whose sequence is put back from a
/// saved state is forgotten ([`Lookups::forget`]) or given the context saved
/// with it ([`Lookups::unpark`]). On a target that keeps its token history
/// ([`History`]), a context behind the target's position is rebuilt from
/// it, so the positions a round of plain steps ran without the draft come
/// back too.
pub(crate) struct Lookups<T> {
    /// Each slot's window, by slot.
    windows: Vec<Box<dyn LookupWindow<T>>>,
    history: Option<History<T>>,
}

impl<B: Prompt + Keep + Rows + Rollback> Lookups<Session<B>> {
    /// The lookup of `slots` resident slots on `s`, each behind a width
    /// chooser of `mode`, the verify pass captured now (`log` told of it)
    /// with slot 0's window and shared by every slot's. Load-time; refused by
    /// name for no slot and a rule the chooser cannot run by.
    pub(crate) fn open(
        s: &mut Session<B>,
        slots: usize,
        mode: WidthMode,
        log: &mut impl RowsLog,
        history: Option<History<Session<B>>>,
    ) -> Result<Self, GateError> {
        record::require(record::groups::drafted::WIDTH, "drafted::Lookups::open");
        if slots == 0 {
            return Err("the lookup draft of no slot".into());
        }
        let choosing = || {
            <Lookup as Chosen<Session<B>>>::choosing_by(
                Lookup::new(),
                mode,
                Lookup::RULE,
                Wall::new(),
            )
        };
        let mut windows: Vec<Box<dyn LookupWindow<Session<B>>>> = Vec::with_capacity(slots);
        windows.push(Box::new(s.with_draft::<_, LOOKUP_ROWS>(choosing()?, log)?));
        for _ in 1..slots {
            windows.push(Box::new(Speculative::<_, LOOKUP_ROWS>::new(choosing()?)));
        }
        Ok(Lookups { windows, history })
    }
}

impl<T: Target> Lookups<T> {
    /// `slot`'s context, the slot's chooser closed for `call` first: the one
    /// way to a slot's context. Refused by name past the slots.
    fn at(&mut self, slot: usize, call: Call) -> Result<&mut Lookup, GateError> {
        let slots = self.windows.len();
        let w = self
            .windows
            .get_mut(slot)
            .ok_or_else(|| format!("slot {slot}: the lookup draft holds {slots} slots"))?;
        if let Call::Moved = call {
            w.close();
        }
        Ok(w.lookup_mut())
    }

    /// [`Lookups::at`], the context rebuilt from the target's history of the
    /// positions before `pos` when it holds fewer and the target keeps one.
    fn before(
        &mut self,
        t: &T,
        slot: usize,
        pos: usize,
        call: Call,
    ) -> Result<&mut Lookup, GateError> {
        let history = self.history.and_then(|h| h(t)).filter(|h| h.len() >= pos);
        let l = self.at(slot, call)?;
        if l.context().len() < pos
            && let Some(h) = history
        {
            l.rebuild(&h[..pos]);
        }
        Ok(l)
    }

    /// The prompt call of `ids` on `slot`, the target's selected one: the
    /// chooser closed for it, then `call` (the seat's own prompt call), then
    /// the call's ids and its argmax into the context, the context first
    /// kept below the call's position: a continued prompt's first id takes
    /// the place of the argmax pushed after the call before. Refused by name
    /// for no id.
    pub(crate) fn prompt_call(
        &mut self,
        t: &mut T,
        slot: usize,
        ids: &[u32],
        call: impl FnOnce(&mut T) -> Result<u32, GateError>,
    ) -> Result<u32, GateError> {
        let Some((&first, rest)) = ids.split_first() else {
            return Err(format!("slot {slot}: the lookup's prompt call of no id").into());
        };
        let pos = t.pos() as usize;
        let l = self.before(t, slot, pos, Call::Moved)?;
        let next = call(t)?;
        l.stand(pos, first);
        for &id in rest.iter() {
            l.push(id);
        }
        l.push(next);
        Ok(next)
    }

    /// The seat's plain step of `last` on `slot` took `next`.
    pub(crate) fn stepped(
        &mut self,
        t: &T,
        slot: usize,
        last: u32,
        next: u32,
    ) -> Result<(), GateError> {
        let pos = (t.pos() as usize).saturating_sub(1);
        let l = self.before(t, slot, pos, Call::Moved)?;
        l.stand(pos, last);
        l.push(next);
        Ok(())
    }

    /// One pass on `slot`, the target's selected one, from `last`: the
    /// context stood at the target's position, then the slot's window's
    /// pass ([`LookupWindow::pass`]), each row's id taken by `take`; the kept
    /// ids into `out`.
    pub(crate) fn pass(
        &mut self,
        t: &mut T,
        slot: usize,
        last: u32,
        take: Taking<'_>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let pos = t.pos() as usize;
        self.before(t, slot, pos, Call::Pass)?.stand(pos, last);
        Ok(drafted(self.windows[slot].pass(t, last, take, out)?))
    }

    /// `slot`'s sequence is not the one its context followed (a reset, a
    /// state put back): the context starts over, from the target's history
    /// where it keeps one.
    pub(crate) fn forget(&mut self, slot: usize) -> Result<(), GateError> {
        self.at(slot, Call::Moved)?.reset();
        Ok(())
    }

    /// `slot`'s context, for a state the seat saves of that slot.
    pub(crate) fn park(&self, slot: usize) -> Result<Vec<u32>, GateError> {
        match self.windows.get(slot) {
            Some(w) => Ok(w.lookup().context().to_vec()),
            None => Err(format!("slot {slot}: the lookup draft holds no such slot").into()),
        }
    }

    /// `ctx`, which [`Lookups::park`] took with a state, as `slot`'s context
    /// once that state is put back on the slot.
    pub(crate) fn unpark(&mut self, slot: usize, ctx: &[u32]) -> Result<(), GateError> {
        self.at(slot, Call::Moved)?.rebuild(ctx);
        Ok(())
    }

    /// The server's stop: every slot's chooser closed, the record of the
    /// slot's last request printed, since no call follows it.
    pub(crate) fn close(&mut self) {
        self.windows.iter_mut().for_each(|w| w.close());
    }
}

/// A pass's draft counts: the proposal's ids, the ones kept.
pub(crate) fn drafted(c: Committed) -> Drafted {
    if c.proposed {
        Drafted {
            proposed: c.rows - 1,
            accepted: c.kept - 1,
        }
    } else {
        Drafted::default()
    }
}

/// `engine.draft` of the lookup draft ([`Lookups`]): no file, no device
/// bytes.
pub(crate) fn lookup_props(mut p: EngineProps) -> EngineProps {
    p.draft = Some(DraftProps {
        model: "lookup".to_owned(),
        n_max: Some(LOOKUP_ROWS as u64 - 1),
        kind: Some("lookup".to_owned()),
        ..DraftProps::default()
    });
    p
}

/// A slot's draft side of a saved sequence ([`SlotDrafts::park`]): what the
/// draft held of it, and why the seat had turned it off, if it had.
#[derive(Clone, Debug)]
pub(crate) struct ParkedDraft<A> {
    draft: Side<A>,
    off: Option<&'static str>,
}

/// What a slot's draft held of a sequence: the MTP draft's side, or the
/// lookup's context.
#[derive(Clone, Debug)]
enum Side<A> {
    Mtp(Parked<A>),
    Lookup(Vec<u32>),
}

impl<A: Copy> ParkedDraft<A> {
    /// A saved state's draft side in a slot file ([`bytes`]): `side` none,
    /// or the MTP draft's ([`Parked::put`], `code` its arenas) or the
    /// lookup's context (an array of ids) after its kind (u8: 0 none, 1 the
    /// MTP draft's, 2 the lookup's), then why the seat turned it off, by its
    /// place in `whys`, the reasons the seat turns a draft off for
    /// ([`bytes::Put::named`]).
    pub(crate) fn put(
        side: Option<&ParkedDraft<A>>,
        w: &mut bytes::Put<'_>,
        code: impl Fn(A) -> u8,
        whys: &[&'static str],
    ) -> Result<(), bytes::Fail> {
        let Some(p) = side else {
            return w.one(0u8);
        };
        match &p.draft {
            Side::Mtp(d) => {
                w.one(1u8)?;
                d.put(w, code)?;
            }
            Side::Lookup(ctx) => {
                w.one(2u8)?;
                w.array(ctx)?;
            }
        }
        w.named(p.off, whys, "the draft's off reason")
    }

    /// What [`ParkedDraft::put`] wrote, `arena` each code's kind and row
    /// capacity, and `whys` the reasons it coded.
    pub(crate) fn take(
        r: &mut bytes::Take<'_>,
        arena: impl Fn(u8) -> Option<(A, usize)>,
        whys: &[&'static str],
    ) -> Result<Option<ParkedDraft<A>>, bytes::Fail> {
        let draft = match r.one::<u8>("the draft's side")? {
            0 => return Ok(None),
            1 => Side::Mtp(Parked::take(r, arena)?),
            2 => Side::Lookup(r.array("the lookup's context")?),
            c => return Err(bytes::refused(format!("a draft's side of code {c}"))),
        };
        let off = r.named(whys, "the draft's off reason")?;
        Ok(Some(ParkedDraft { draft, off }))
    }

    /// Refused by name unless the side can be put back on a state of `pos`
    /// positions of a model of `vocab` ids: every id it holds below `vocab`,
    /// and the MTP draft's waiting rows as [`Parked::check`] checks them,
    /// `rows` checking each walk's target rows against the state.
    pub(crate) fn check(
        &self,
        vocab: usize,
        pos: u32,
        rows: impl Fn(A, (usize, usize), u32) -> Result<(), bytes::Fail>,
    ) -> Result<(), bytes::Fail> {
        let ids = |ts: &[u32]| match ts.iter().find(|&&t| t as usize >= vocab) {
            Some(t) => Err(bytes::refused(format!(
                "a saved draft id {t} past the vocabulary of {vocab}"
            ))),
            None => Ok(()),
        };
        match &self.draft {
            Side::Mtp(d) => d.check(pos, ids, rows),
            Side::Lookup(ctx) => ids(ctx),
        }
    }
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
    /// ([`SlotDrafts::round_mut`]).
    specs: Vec<Option<Spec<B, M>>>,
    batch: runtime::width::Batch,
    widths: Vec<usize>,
    /// Why each slot's draft is off until the slot's next reset, and whether
    /// a record has said so since it was turned off; `None` while it drafts.
    off: Vec<Option<(&'static str, bool)>>,
    /// The lookup draft of every slot, in place of the MTP draft
    /// ([`SlotDrafts::lookup`]); `None` otherwise.
    lookups: Option<Lookups<Session<B>>>,
}

impl<B: MtpBody, const M: usize> SlotDrafts<B, M>
where
    Window<M>: Widths,
{
    /// No draft: every call the session's own, every pass one plain step.
    pub(crate) fn none() -> Self {
        SlotDrafts {
            specs: Vec::new(),
            batch: runtime::width::Batch::default(),
            widths: Vec::new(),
            off: Vec::new(),
            lookups: None,
        }
    }

    /// The lookup draft on each of `slots` resident slots ([`Lookups::open`]:
    /// its verify pass captured now, `log` told of it, behind the width
    /// chooser of `mode`) in place of the MTP draft: no slot's draft is ever
    /// off, and a cut or a state put back needs no rejoin. Load-time.
    pub(crate) fn lookup(
        s: &mut Session<B>,
        slots: usize,
        mode: WidthMode,
        log: &mut impl RowsLog,
    ) -> Result<Self, GateError> {
        Ok(SlotDrafts {
            specs: Vec::new(),
            batch: runtime::width::Batch::default(),
            widths: Vec::new(),
            off: Vec::new(),
            lookups: Some(Lookups::open(s, slots, mode, log, None)?),
        })
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
        record::require(record::groups::drafted::PROMPT, "drafted::SlotDrafts::open");
        record::require(record::groups::drafted::WIDTH, "drafted::SlotDrafts::open");
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
            batch: runtime::width::Batch::default(),
            widths: Vec::new(),
            off: vec![None; slots],
            lookups: None,
        })
    }

    /// Whether a draft runs, the MTP draft or the lookup.
    pub(crate) fn drafts(&self) -> bool {
        !self.specs.is_empty() || self.lookups.is_some()
    }

    /// Whether the MTP draft runs: a cut or a state put back rejoins it or
    /// turns it off ([`SlotDrafts::rejoin`], [`SlotDrafts::turn_off`]).
    pub(crate) fn mtp(&self) -> bool {
        !self.specs.is_empty()
    }

    /// Whether the lookup draft runs ([`SlotDrafts::lookup`]).
    pub(crate) fn looks_up(&self) -> bool {
        self.lookups.is_some()
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
    /// out prints none. The draft's own state starts over, so an off draft
    /// holds no rows waiting for a walk ([`MtpDraft::restart`]); its reset or
    /// rejoin sets them anew. Nothing without a draft; refused as
    /// [`SlotDrafts::has`].
    pub(crate) fn turn_off(&mut self, slot: usize, why: &'static str) -> Result<(), GateError> {
        if self.has(slot)? {
            if let Some(spec) = self.specs[slot].as_mut() {
                print_widths(spec);
                spec.draft_mut().draft_mut().restart();
            }
            self.off[slot] = Some((why, false));
        }
        Ok(())
    }

    /// `slot`'s draft rejoined at `pos`, where the body kept its side after a
    /// cut or a state put back as its first `pos` positions
    /// ([`MtpDraft::rejoin`]), and on again: the slot's next call walks the
    /// row before `pos` and drafts from there. Nothing without a draft;
    /// refused as [`SlotDrafts::has`].
    pub(crate) fn rejoin(&mut self, slot: usize, pos: u32) -> Result<(), GateError> {
        if let Some(spec) = self.spec_mut(slot)? {
            spec.draft_mut().draft_mut().rejoin(pos);
            self.off[slot] = None;
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
        self.prefill_by(
            s,
            slot,
            ids,
            |s| Ok(s.prompt(ids, Want::Argmax)?.argmax()),
            |spec, s| Ok(Advance::prompt(spec, s, ids)?),
        )
    }

    /// [`SlotDrafts::prefill`] of a prompt's last call whose last id the
    /// call runs as its last row (`serve::Engine::prefill_next` through a
    /// seat whose body runs it so), under the draft the window anchored at
    /// the prompt's end ([`Advance::begin`]) as the step after a call less
    /// that id anchors it ([`Draft::stepped`]).
    pub(crate) fn prompt_close(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        ids: &[u32],
    ) -> Result<u32, GateError> {
        self.prefill_by(
            s,
            slot,
            ids,
            |s| Ok(s.prompt(ids, Want::Argmax)?.argmax()),
            |spec, s| {
                let next = Advance::prompt(spec, s, ids)?;
                Advance::begin(spec, s, ids, next)?;
                Ok(next)
            },
        )
    }

    /// The same prompt and draft-store walk without the target's head.
    pub(crate) fn prefill_only(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        ids: &[u32],
        path: B::Path,
    ) -> Result<(), GateError> {
        self.prefill_by(
            s,
            slot,
            ids,
            |s| {
                s.prefill(ids, path)?;
                Ok(0)
            },
            |spec, s| {
                spec.draft_mut().prompt_by(|d| d.prefill(s, ids))?;
                Ok(0)
            },
        )
        .map(|_| ())
    }

    /// [`SlotDrafts::prefill`] of a prompt that holds an image: `spans` are the
    /// body's spans of the image's rows, spliced in place of their positions'
    /// embeddings. Under the draft the draft's own media prompt call
    /// ([`MtpDraft::prompt_media`]: its store walked over the call's units, a
    /// span's positions included), then its join record; without it, or while
    /// it is off, the session's ([`Session::prompt_media`]).
    #[cfg(feature = "vision")]
    pub(crate) fn prefill_media(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        ids: &[u32],
        spans: &[<B as bloomery_gpu::media::MediaBody>::Span<'_>],
    ) -> Result<u32, GateError>
    where
        B: bloomery_gpu::media::MediaBody,
    {
        self.prefill_by(
            s,
            slot,
            ids,
            |s| Ok(s.prompt_media(ids, spans, Want::Argmax)?.argmax()),
            |spec, s| {
                Ok(spec
                    .draft_mut()
                    .prompt_by(|d| d.prompt_media(s, ids, spans))?)
            },
        )
    }

    /// The prompt call's order, the one owner of it: under the lookup
    /// `plain` with its ids fed to the slot's context
    /// ([`Lookups::prompt_call`]); the off slot's record and the session's
    /// own call, else under the draft the chooser's passes closed out,
    /// `drafted` (the window's prompt call) and the join record, else
    /// `plain`.
    fn prefill_by(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        ids: &[u32],
        plain: impl FnOnce(&mut Session<B>) -> Result<u32, GateError>,
        drafted: impl FnOnce(&mut Spec<B, M>, &mut Session<B>) -> Result<u32, GateError>,
    ) -> Result<u32, GateError> {
        if let Some(l) = &mut self.lookups {
            return l.prompt_call(s, slot, ids, plain);
        }
        if self.is_off(slot) {
            self.print_off(s, slot);
            return plain(s);
        }
        match self.spec_mut(slot)? {
            Some(spec) => {
                print_widths(spec);
                let next = drafted(spec, s)?;
                print_join(spec);
                Ok(next)
            }
            None => plain(s),
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

    /// The step's order, the one owner of it: the draft's waiting rows
    /// ([`SlotDrafts::before_row`]), the target's step, the target's row when
    /// asked, then the draft told.
    fn step_reading(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        row: Option<&mut [f32]>,
    ) -> Result<u32, GateError> {
        self.before_row(s, slot, last)?;
        let drafting = !self.is_off(slot);
        let next = s.step(last, Want::Argmax)?.argmax();
        if let Some(row) = row {
            s.model()
                .logits_into(row)
                .map_err(|e| format!("logits of the step: {e}"))?;
        }
        if let Some(l) = &mut self.lookups {
            l.stepped(s, slot, last, next)?;
        }
        if let Some(spec) = self.spec_mut(slot)?.filter(|_| drafting) {
            Draft::stepped(spec.draft_mut(), s, last, next)?;
        }
        Ok(next)
    }

    /// What a plain step of `last` on `slot`, the session's selected one,
    /// does to the slot's draft before the target runs it (a row of a pass of
    /// several slots gathers instead, [`SlotDrafts::gather_row`]): an off
    /// draft's record, once; under the draft the chooser's passes closed
    /// out, the rows the draft left waiting walked with `last` their last
    /// ([`MtpDraft::before_step`]: the target's call overwrites the arena
    /// they read) and the join record. Refused as [`SlotDrafts::has`].
    pub(crate) fn before_row(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
    ) -> Result<(), GateError> {
        self.before_row_by(s, slot, |d, s| d.before_step(s, last))
    }

    /// [`SlotDrafts::before_row`]'s records around `walk`, the call that
    /// catches the slot's draft up: an off draft's record, once; under the
    /// draft, the chooser's passes closed out, `walk`, then the join record.
    fn before_row_by(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        walk: impl FnOnce(&mut MtpDraft<B>, &mut Session<B>) -> Result<(), SessionError>,
    ) -> Result<(), GateError> {
        if matches!(self.off.get(slot), Some(Some((_, false)))) {
            self.print_off(s, slot);
        }
        let drafting = !self.is_off(slot);
        if let Some(spec) = self.spec_mut(slot)?.filter(|_| drafting) {
            print_widths(spec);
            walk(spec.draft_mut().draft_mut(), s)?;
            print_join(spec);
        }
        Ok(())
    }

    /// `slot`'s plain row of `last`, run from the position `p0` in a pass of
    /// several slots ([`super::rounds::step_rows_drafted`]) after
    /// [`SlotDrafts::before_row`], took `next`: under the draft the row told
    /// to it as a step is ([`MtpDraft::passed_plain`]: its hidden row the
    /// pass's), the chooser's pass in flight and batch acceptance dropped as
    /// for every call of the caller's own (`Choosing::prompt_by`); nothing
    /// while the draft is off. Refused as [`SlotDrafts::has`].
    pub(crate) fn after_row(
        &mut self,
        slot: usize,
        p0: u32,
        last: u32,
        next: u32,
    ) -> Result<(), GateError> {
        let drafting = !self.is_off(slot);
        if let Some(spec) = self.spec_mut(slot)?.filter(|_| drafting) {
            spec.draft_mut()
                .prompt_by(|d| d.passed_plain(p0, last, next))?;
        }
        Ok(())
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
        if let Some(l) = &mut self.lookups {
            return l.pass(s, slot, last, Taking::Argmax, out);
        }
        self.pass_taking(s, slot, last, &mut Argmax, out)
    }

    /// One sampled pass on `slot`, the session's selected one, from `last`
    /// (`serve::Engine::advance_sampled`): [`SlotDrafts::pass_taking`] with
    /// each row's id drawn by `sampler` from that row's logits, given
    /// `history` and the ids this pass took before it; each row read as its
    /// best logits when the sampler reads at most `best` of them and the
    /// card's best read holds that many ([`reads_best`]). `history` comes
    /// back as it was given.
    #[allow(
        clippy::too_many_arguments,
        reason = "the sampled pass's operands, as `Seat::pass_sampled` names them"
    )]
    pub(crate) fn pass_sampled(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let from = history.len();
        let mut draw = |row: &[f32]| {
            let id = sampler(row, history);
            history.push(id);
            id
        };
        let d = match (&mut self.lookups, reads_best(best)) {
            (Some(l), true) => l.pass(s, slot, last, Taking::SampleBest(&mut draw), out),
            (Some(l), false) => l.pass(s, slot, last, Taking::Sample(&mut draw), out),
            (None, true) => self.pass_taking(s, slot, last, &mut SampleBest(&mut draw), out),
            (None, false) => self.pass_taking(s, slot, last, &mut Sample(&mut draw), out),
        };
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
            Some(spec) => Ok(drafted(pass_in_turn(spec, s, last, pick, out)?)),
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
        if self.lookups.is_some() {
            return LOOKUP_ROWS;
        }
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
        if let Some(l) = &mut self.lookups {
            l.forget(slot)?;
        }
        if drafting {
            self.off[slot] = None;
            if let Some(spec) = self.specs[slot].as_mut() {
                print_widths(spec);
                spec.draft_mut().draft_mut().restart();
            }
        }
        Ok(())
    }

    /// The server's stop: the chooser's record of every slot's last request
    /// printed (the MTP draft's or the lookup's), since no call follows it
    /// to close it out.
    pub(crate) fn close(&mut self) {
        for spec in self.specs.iter_mut().flatten() {
            print_widths(spec);
        }
        if let Some(l) = &mut self.lookups {
            l.close();
        }
    }

    /// `slot`'s draft side of the session's sequence, for a state the seat
    /// saves of that slot ([`MtpDraft::park`]; the turn-off why with it);
    /// `None` without a draft. The state keeps the draft's store and the
    /// arenas the waiting rows sit in. Refused as [`SlotDrafts::has`].
    pub(crate) fn park(&self, slot: usize) -> Result<Option<ParkedDraft<B::Arena>>, GateError> {
        if let Some(l) = &self.lookups {
            return Ok(Some(ParkedDraft {
                draft: Side::Lookup(l.park(slot)?),
                off: None,
            }));
        }
        if !self.has(slot)? {
            return Ok(None);
        }
        Ok(self.specs[slot].as_ref().map(|spec| ParkedDraft {
            draft: Side::Mtp(spec.draft().draft().park()),
            off: self.off[slot].map(|(why, _)| why),
        }))
    }

    /// Refused by name unless a state whose draft's side is `p` fits this
    /// seat: a side of the draft that runs where one does, none where none
    /// does.
    pub(crate) fn takes(&self, p: Option<&ParkedDraft<B::Arena>>) -> Result<(), GateError> {
        if p.is_some() != self.drafts() {
            return Err(format!(
                "a state saved with the MTP draft {} put back with it {}",
                if p.is_some() { "on" } else { "off" },
                if self.drafts() { "on" } else { "off" }
            )
            .into());
        }
        if p.is_some_and(|p| matches!(p.draft, Side::Lookup(_)) != self.lookups.is_some()) {
            return Err("a state saved under one draft put back under the other".into());
        }
        Ok(())
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
        let Some(p) = p else {
            return Ok(());
        };
        match (&p.draft, &mut self.lookups) {
            (Side::Lookup(ctx), Some(l)) => l.unpark(slot, ctx),
            (Side::Mtp(d), None) => {
                if let Some(spec) = self.spec_mut(slot)? {
                    spec.draft_mut().draft_mut().unpark(d);
                    self.off[slot] = p.off.map(|why| (why, false));
                }
                Ok(())
            }
            _ => unreachable!("takes refused a side of the other draft"),
        }
    }

    /// Every slot's window, by slot, for a round that drives several slots'
    /// drafts at once (`serve_seats::rounds::pass_rows_one_pass`, which
    /// takes each busy slot's window out and puts it back); empty without a
    /// draft.
    pub(crate) fn round_mut(
        &mut self,
    ) -> (
        &mut [Option<Spec<B, M>>],
        &mut runtime::width::Batch,
        &mut Vec<usize>,
    ) {
        (&mut self.specs, &mut self.batch, &mut self.widths)
    }
}

impl<B: MtpBody + Slots, const M: usize> SlotDrafts<B, M>
where
    B::Seq: 'static,
    Window<M>: Widths,
{
    /// [`SlotDrafts::before_row`] on `slot`, the session's selected one, the
    /// draft caught up through [`MtpDraft::gather`]: shared prompt rows
    /// walked now, step and verify rows gathered into `walks` for one store
    /// walk of the round's gathered rows ([`SlotWalks::run`]) before its
    /// pass. Refused as [`SlotDrafts::has`].
    pub(crate) fn gather_row(
        &mut self,
        s: &mut Session<B>,
        slot: usize,
        last: u32,
        walks: &mut SlotWalks<B>,
    ) -> Result<(), GateError> {
        self.before_row_by(s, slot, |d, s| d.gather(s, last, walks))
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

/// A seat that prints the chooser's record ([`print_widths`]) names its group at its
/// start, so a list without it fails before the first drafted request.
pub(crate) fn serves_widths() {
    record::require(record::groups::drafted::WIDTH, "drafted::serves_widths");
}

/// The chooser's passes over the request a slot just finished and its state
/// at the request's end, as its `mtp width` record, printed at the slot's
/// next call that closes it out or at the server's stop: nothing under
/// `fixed`, whose passes are the draft's own, and nothing for a request that
/// ran no pass.
pub(crate) fn print_widths<D, C: Clock, const M: usize>(spec: &mut Speculative<Choosing<D, C>, M>) {
    record::require(record::groups::drafted::WIDTH, "drafted::print_widths");
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

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::Out;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Duration;

    /// A draw takes the card's best read only when it reads no more of a
    /// row's best than the read holds.
    #[test]
    fn a_draw_reads_best_up_to_the_cards_list() {
        let b = best::BEST;
        let cases = [
            (Some(1), true),
            (Some(b), true),
            (Some(b + 1), false),
            (None, false),
        ];
        assert_eq!(cases.map(|(k, _)| reads_best(k)), cases.map(|(_, r)| r));
    }

    /// Each side keeps its kind and bytes, and is taken only under that draft.
    #[test]
    fn parked_sides_keep_their_kind_and_refuse_a_different_draft() {
        use bloomery_gpu::arch::qwen3moe::{Body38, TargetRows};

        let arena = |c| (c == 0).then_some((TargetRows::Step, 1));
        let mut input = &[0u8; 3][..];
        let mtp = ParkedDraft {
            draft: Side::Mtp(Parked::take(&mut bytes::Take::new(&mut input), arena).unwrap()),
            off: None,
        };
        let lookup = ParkedDraft {
            draft: Side::Lookup(vec![2, 5, 7]),
            off: None,
        };
        for (side, tag) in [(None, 0u8), (Some(&mtp), 1), (Some(&lookup), 2)] {
            let mut out = Vec::new();
            let mut w = bytes::Put::new(&mut out);
            ParkedDraft::put(side, &mut w, |_| 0, &[]).unwrap();
            w.flush().unwrap();
            assert_eq!(out[0], tag, "saved files keep each draft kind's tag");
            let mut input = out.as_slice();
            let mut r = bytes::Take::new(&mut input);
            let back = ParkedDraft::take(&mut r, arena, &[]).unwrap();
            r.end().unwrap();
            assert_eq!(
                format!("{back:?}"),
                format!("{side:?}"),
                "the side keeps its kind and context"
            );
        }
        let none = SlotDrafts::<Body38, 2>::none();
        let mut mtp_table = SlotDrafts::<Body38, 2>::none();
        mtp_table.specs.push(None);
        let mut lookup_table = SlotDrafts::<Body38, 2>::none();
        lookup_table.lookups = Some(Lookups {
            windows: Vec::new(),
            history: None,
        });
        none.takes(None).expect("no draft takes no side");
        mtp_table.takes(Some(&mtp)).expect("MTP takes its own side");
        lookup_table
            .takes(Some(&lookup))
            .expect("lookup takes its own side");
        for (table, side) in [(&mtp_table, &lookup), (&lookup_table, &mtp)] {
            let e = table.takes(Some(side)).expect_err("a different draft");
            assert!(
                e.to_string()
                    .contains("under one draft put back under the other"),
                "{e}"
            );
        }
        for (table, side) in [
            (&none, Some(&lookup)),
            (&mtp_table, None),
            (&lookup_table, None),
        ] {
            table.takes(side).expect_err("a different draft presence");
        }
    }

    /// Every id a saved side holds is checked against the vocabulary, the
    /// MTP draft's refresh as the lookup's context.
    #[test]
    fn parked_sides_refuse_ids_past_the_vocabulary() {
        use bloomery_gpu::arch::qwen3moe::TargetRows;

        let mut b = Vec::new();
        let mut w = bytes::Put::new(&mut b);
        w.one(1u8).unwrap();
        w.array(&[4u32, 9]).unwrap();
        w.one(5u32).unwrap();
        w.one(1u8).unwrap();
        w.count(0).unwrap();
        w.vals(&[0u8; 2]).unwrap(); // No last unit or skip.
        w.flush().unwrap();
        let arena = |c| (c == 1).then_some((TargetRows::Pass, 8));
        let mtp = ParkedDraft {
            draft: Side::Mtp(
                Parked::take(&mut bytes::Take::new(&mut b.as_slice()), arena).unwrap(),
            ),
            off: None,
        };
        let lookup = ParkedDraft {
            draft: Side::Lookup(vec![2, 9]),
            off: None,
        };
        let rows = |_, _, _| Ok(());
        for side in [&mtp, &lookup] {
            side.check(10, 6, rows).expect("ids below the vocabulary");
            let why = side.check(9, 6, rows).unwrap_err().to_string();
            assert!(
                why.contains("a saved draft id 9 past the vocabulary of 9"),
                "{why}"
            );
        }
    }

    /// The walls the target's calls take: a verify of the lookup's two rows
    /// costs 1.35 of the plain step.
    const STEP: Duration = Duration::from_millis(10);
    const VERIFY: Duration = Duration::from_micros(13_500);
    const PROMPT: Duration = Duration::from_millis(40);

    /// A target that walks a fixed id stream, the id at position `i` its
    /// `ids[i % len]` and each call's argmax the id at the position after
    /// the call's last row, each call taking its wall of the clock every
    /// slot's target shares.
    struct Position {
        pos: u32,
        verified: u32,
        verify_wall: Duration,
        ids: Rc<[u32]>,
        clock: Rc<Cell<Duration>>,
    }

    impl Position {
        fn new(ids: &Rc<[u32]>, clock: &Rc<Cell<Duration>>) -> Position {
            Position {
                pos: 0,
                verified: 0,
                verify_wall: VERIFY,
                ids: Rc::clone(ids),
                clock: Rc::clone(clock),
            }
        }

        fn id(&self, at: u32) -> u32 {
            self.ids[at as usize % self.ids.len()]
        }

        fn took(&self, wall: Duration) {
            self.clock.set(self.clock.get() + wall);
        }

        /// The lookup's prompt call on `slot` of the stream's id at the
        /// target's position.
        fn begin(&mut self, l: &mut Lookups<Position>, slot: usize) {
            let id = self.id(self.pos);
            l.prompt_call(self, slot, &[id], |t| {
                Ok(t.prompt(&[id], Want::Argmax)?.argmax())
            })
            .unwrap();
        }

        /// The lookup's pass on `slot` from the stream's id at the target's
        /// position.
        fn pass(&mut self, l: &mut Lookups<Position>, slot: usize) {
            let last = self.id(self.pos);
            l.pass(self, slot, last, Taking::Argmax, &mut Vec::new())
                .unwrap();
        }
    }

    impl Target for Position {
        type Error = WidthError;
        fn pos(&self) -> u32 {
            self.pos
        }
        fn ctx(&self) -> u32 {
            1 << 16
        }
        fn prompt(&mut self, ids: &[u32], _: Want) -> Result<Out<'_>, WidthError> {
            self.pos += u32::try_from(ids.len()).unwrap();
            self.took(PROMPT);
            Ok(Out::Argmax(self.id(self.pos)))
        }
        fn step(&mut self, _: u32, _: Want) -> Result<Out<'_>, WidthError> {
            self.pos += 1;
            self.took(STEP);
            Ok(Out::Argmax(self.id(self.pos)))
        }
        fn keepable(&self, n: u32) -> u32 {
            n.min(self.pos)
        }
        fn cut(&mut self, n: u32) -> Result<(), WidthError> {
            self.pos = n;
            Ok(())
        }
        fn reset(&mut self) -> Result<(), WidthError> {
            self.pos = 0;
            Ok(())
        }
    }

    impl Verify for Position {
        const MAX_ROWS: usize = LOOKUP_ROWS;
        fn verify<const M: usize>(&mut self, _: [u32; M]) -> Result<[u32; M], WidthError> {
            let at = self.pos;
            self.verified = u32::try_from(M).unwrap();
            self.pos += self.verified;
            self.took(self.verify_wall);
            Ok(std::array::from_fn(|r| {
                self.id(at + 1 + u32::try_from(r).unwrap())
            }))
        }
        fn commit(&mut self, accepted: usize) -> Result<(), WidthError> {
            self.pos -= self.verified - u32::try_from(accepted).unwrap();
            Ok(())
        }
    }

    impl RowLogits for Position {
        fn row_logits(&mut self, _: usize) -> Result<&[f32], WidthError> {
            unreachable!()
        }
    }

    /// The clock the targets' calls move.
    struct Shared(Rc<Cell<Duration>>);
    impl Clock for Shared {
        fn now(&mut self) -> Duration {
            self.0.get()
        }
    }

    /// The lookup of `slots` slots under the cost mode, by `rule`, on
    /// `clock`.
    fn lookups(slots: usize, rule: runtime::Rule, clock: &Rc<Cell<Duration>>) -> Lookups<Position> {
        let window = |_| -> Box<dyn LookupWindow<Position>> {
            Box::new(Speculative::<_, LOOKUP_ROWS>::new(
                <Lookup as Chosen<Position>>::choosing_by(
                    Lookup::new(),
                    WidthMode::Cost,
                    rule,
                    Shared(Rc::clone(clock)),
                )
                .unwrap(),
            ))
        };
        Lookups {
            windows: (0..slots).map(window).collect(),
            history: None,
        }
    }

    /// What `slot`'s chooser's gate stands on.
    fn gate(l: &Lookups<Position>, slot: usize) -> runtime::width::GateState {
        l.windows[slot].gate_state()
    }

    /// The medians of a plain step's and a verify's own walls.
    fn walls() -> Vec<Option<f64>> {
        [STEP, VERIFY].map(|w| Some(w.as_secs_f64() * 1e3)).to_vec()
    }

    /// A table of no draft runs every slot's pass as one plain step, none of
    /// its slots off: the condition the Qwen3.8 seat reads a step's PLE rows
    /// ahead on.
    #[test]
    fn a_table_of_no_draft_passes_one_row_on_every_slot() {
        let none = SlotDrafts::<bloomery_gpu::arch::qwen3moe::Body38, 2>::none();
        for slot in 0..3 {
            assert_eq!(none.pass_rows(slot), 1, "slot {slot}");
            assert!(!none.is_off(slot), "slot {slot}");
        }
    }

    #[test]
    fn lookup_prompt_discards_the_prior_last_token() {
        let clock = Rc::new(Cell::new(Duration::ZERO));
        let ids: Rc<[u32]> = Rc::new([1_u32, 2, 3, 4]);
        let mut t = Position::new(&ids, &clock);
        let mut l = lookups(1, Lookup::RULE, &clock);
        t.begin(&mut l, 0);
        t.begin(&mut l, 0);
        assert_eq!(l.windows[0].lookup().context(), &ids[..3]);
    }

    #[test]
    fn a_lookup_prompt_call_of_no_id_is_refused() {
        let clock = Rc::new(Cell::new(Duration::ZERO));
        let ids: Rc<[u32]> = Rc::new([1_u32]);
        let mut t = Position::new(&ids, &clock);
        let mut l = lookups(1, Lookup::RULE, &clock);
        let e = l.prompt_call(&mut t, 0, &[], |_| Ok(1)).unwrap_err();
        assert_eq!(e.to_string(), "slot 0: the lookup's prompt call of no id");
    }

    #[test]
    fn turns_land_each_slots_cost_before_the_other_slots_expensive_pass() {
        let clock = Rc::new(Cell::new(Duration::ZERO));
        let ids: Rc<[u32]> = Rc::new([7_u32]);
        let mut targets = [Position::new(&ids, &clock), Position::new(&ids, &clock)];
        targets[1].verify_wall = Duration::from_millis(400);
        let mut specs = std::array::from_fn::<_, 2, _>(|slot| {
            let choosing = <Lookup as Chosen<Position>>::choosing_by(
                Lookup::new(),
                WidthMode::Cost,
                runtime::Rule {
                    min_timings: 1,
                    ..Lookup::RULE
                },
                Shared(Rc::clone(&clock)),
            )
            .unwrap();
            let mut spec = Speculative::<_, LOOKUP_ROWS>::new(choosing);
            spec.prompt(&mut targets[slot], &[7]).unwrap();
            spec
        });
        for _ in 0..12 {
            for slot in 0..2 {
                let t = &mut targets[slot];
                pass_in_turn(&mut specs[slot], t, 7, &mut Argmax, &mut Vec::new()).unwrap();
            }
        }
        for slot in 0..2 {
            assert_eq!(
                specs[slot].draft().gate_state().costs,
                [STEP, targets[slot].verify_wall].map(|wall| Some(wall.as_secs_f64() * 1e3)),
                "slot {slot}: only its own pass contributes to its cost"
            );
        }
    }

    /// A pass's wall is read as its width's cost, whatever a call between
    /// it and its slot's next pass does ([`LookupWindow::pass`]), and no
    /// such call's wall is: nothing, a plain step back to where the pass
    /// ended after a cut (a request whose prompt the cache holds but its
    /// last id), a prompt call back to it, a reset or a state put back of
    /// the slot, and another slot's prompt call and pass. Each case's first
    /// pass verifies the warm-up's width 1, its second steps the warm-up's
    /// plain step.
    #[test]
    fn a_pass_reads_its_own_wall_alone() {
        type Between = fn(&mut Lookups<Position>, &mut Position, &mut Position);
        let cases: [(&str, Between); 6] = [
            ("nothing", |_, _, _| {}),
            ("a plain step", |l, t, _| {
                let end = t.pos();
                t.cut(end - 1).unwrap();
                let last = t.id(end - 1);
                let next = t.step(last, Want::Argmax).unwrap().argmax();
                l.stepped(t, 0, last, next).unwrap();
            }),
            ("a prompt call", |l, t, _| {
                let end = t.pos();
                t.cut(end - 1).unwrap();
                t.begin(l, 0);
            }),
            ("a reset", |l, _, _| l.forget(0).unwrap()),
            ("a state put back", |l, _, _| {
                let ctx = l.park(0).unwrap();
                l.unpark(0, &ctx).unwrap();
            }),
            ("another slot's pass", |l, _, other| {
                other.begin(l, 1);
                other.pass(l, 1);
            }),
        ];
        let ids: Rc<[u32]> = Rc::new([7_u32]);
        let mut wrong = Vec::new();
        for (what, between) in cases {
            let clock = Rc::new(Cell::new(Duration::ZERO));
            let rule = runtime::Rule {
                min_timings: 1,
                ..Lookup::RULE
            };
            let mut l = lookups(2, rule, &clock);
            let mut t = Position::new(&ids, &clock);
            let mut other = Position::new(&ids, &clock);
            t.begin(&mut l, 0);
            t.pass(&mut l, 0);
            let end = t.pos();
            between(&mut l, &mut t, &mut other);
            assert_eq!(
                t.pos(),
                end,
                "{what}: the target stands where the pass ended"
            );
            t.pass(&mut l, 0);
            let costs = gate(&l, 0).costs;
            if costs != walls() {
                wrong.push((what, costs));
            }
        }
        assert!(
            wrong.is_empty(),
            "the costs read are not the passes' own walls {:?}: {wrong:?}",
            walls()
        );
    }

    /// The server's stop prints every slot's record of its last request
    /// ([`Lookups::close`]): no call follows that request to close its
    /// chooser out, and without the stop its passes go unprinted.
    #[test]
    fn the_stop_prints_every_slots_last_request() {
        let ids: Rc<[u32]> = Rc::new([7_u32]);
        let unprinted = |stop: bool| {
            let clock = Rc::new(Cell::new(Duration::ZERO));
            let mut l = lookups(2, Lookup::RULE, &clock);
            for slot in 0..2 {
                let mut t = Position::new(&ids, &clock);
                t.begin(&mut l, slot);
                t.pass(&mut l, slot);
            }
            if stop {
                l.close();
            }
            l.windows
                .iter_mut()
                .map(|w| w.unprinted())
                .collect::<Vec<_>>()
        };
        assert_eq!(unprinted(false), [1, 1], "each slot ran one pass");
        assert_eq!(unprinted(true), [0, 0], "the stop printed every slot's");
    }

    /// Two slots that take turns, each a lookup whose proposals the target
    /// keeps about one in eight while a verify costs 1.35 of the plain step:
    /// each slot's chooser reads every pass at its own wall, so its warm-up
    /// ends, and the gate closes on what the slot's proposals kept.
    #[test]
    fn slots_that_take_turns_each_close_a_lookup_that_does_not_pay() {
        let mut x = 0x2545_f491_u32;
        let ids: Rc<[u32]> = (0..4093)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x % 8
            })
            .collect();
        let clock = Rc::new(Cell::new(Duration::ZERO));
        let mut l = lookups(2, Lookup::RULE, &clock);
        let mut t = [Position::new(&ids, &clock), Position::new(&ids, &clock)];
        for (slot, t) in t.iter_mut().enumerate() {
            t.begin(&mut l, slot);
        }
        for i in 0..400 {
            t[i % 2].pass(&mut l, i % 2);
        }
        for slot in 0..2 {
            let state = gate(&l, slot);
            assert_eq!(state.costs, walls(), "slot {slot}: the warm-up's costs");
            assert_eq!(
                state.width, 0,
                "slot {slot}: the gate stays open, {state:?}"
            );
        }
    }
}
