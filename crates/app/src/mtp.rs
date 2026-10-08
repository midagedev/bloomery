//! The shared MTP window: [`MtpDraft<B>`], the runtime rule's `Draft` over
//! a model's MTP layer, and the [`MtpBody`] adapter it drives that layer
//! through, over the body a [`Session<B>`] wraps, with the
//! model-independent types its calls take. Qwen3.8's adapter is
//! `arch::qwen3moe`'s, over `Body38`.
//!
//! An MTP layer's row at position `q` reads the token at `q` and the
//! target's final hidden row at `q − 1`, and predicts the token at `q + 1`.
//! The window drives it under the runtime's rule ([`runtime::Speculative`]
//! over a [`runtime::Draft`] on the session): a proposal of up to
//! [`MtpBody::WIDTH`] ids, one verify of up to [`MtpBody::VERIFY_ROWS`] rows,
//! and the commit of the rows the target agrees with. Between those:
//!
//! - the prompt call ([`MtpDraft`'s `prompt`](Draft::prompt)) walks the
//!   draft over the prompt as its units complete — a unit's arena holds its
//!   rows' hidden rows until the next unit overwrites them — through the
//!   store's append alone ([`WalkMode::Store`]), filling the draft's store
//!   from position 0 (the row at 0 reads a zero hidden row: the target holds
//!   nothing before it) and leaving the last unit's last row for
//!   [`MtpDraft`'s `begin`](Draft::begin), whose walk is the next window's
//!   anchor;
//! - each window's [`MtpDraft`'s `propose`](Draft::propose) runs one chain
//!   ([`MtpBody::chain`], one readback): the refresh — the rows the target
//!   kept, at the positions after the verify's first, each with the hidden
//!   row the verify's row before it wrote, the last row the target's own
//!   next token (the anchor, whose prediction is the first proposal) — then
//!   own walks, each reading the walk before it on the card;
//! - [`MtpDraft`'s `accept`](Draft::accept) records the refresh before the
//!   commit takes the rejected rows back, the verify's arena
//!   ([`MtpBody::VERIFY_ARENA`]) still holding every row's hidden row.
//!
//! A prompt call that continues a held sequence (a request that extends the
//! one before it) first walks the rows the last call left waiting — the
//! refresh a window or a step recorded, or a prompt call's anchor — their
//! last row the call's first id, while their arena still holds their hidden
//! rows; a first step that feeds the next id does the same before the step
//! overwrites the step's arena ([`MtpDraft::before_step`]), and so does the
//! width chooser before each pass it holds back ([`Draft::before_plain`]).
//! When no such rows
//! wait, or the store does not hold the positions below them, the draft
//! skips: it proposes nothing, every pass a plain step, until a prompt call
//! from position 0 or a restart; the call's [`Join`] names why.
//!
//! Every decision of those calls — which rows wait, which walks a call runs
//! and in which mode, the join checks, the refusals, what a park keeps — is
//! the window's policy (`policy::Policy`), over positions, the context, token
//! ids and arena kinds alone; [`MtpDraft`] runs its walks on the card, and the
//! policy's unit tests run the same calls against a model of the body's
//! records with no card.
//!
//! A server that switches the target between sequences parks the draft's
//! waiting rows with the sequence's state ([`MtpDraft::park`]) and puts them
//! back with it ([`MtpDraft::unpark`]): the returning sequence's next call
//! joins it as the call after no switch would.
//!
//! The proposal changes which passes run, never a token: every kept token is
//! the target's own argmax. The hidden rows never leave the card: a walk
//! names them by the arena a target call left them in and the row they
//! start at ([`Hidden::Target`]). [`runtime::Tapped`] on the session is the
//! host's readback of the same rows, for the gates; the window does not read
//! it.
//!
//! A partial accept is the body's own: the window calls
//! [`runtime::Verify::commit`] with the rows kept and nothing else, and the
//! session's commit takes the rest back through the body's
//! [`Rollback::rollback_on`], which settles whatever recurrent state the body
//! holds for the rows it drops. No rewind lives in the window, and no call
//! here moves the target's position: the draft's walks write only the
//! draft's store and arena, and the target's position moves by the rule's
//! verify and commit alone.

use std::fmt;

use bloomery_gpu::model::{Rollback, Rows, SlotRows, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use runtime::{Draft, TapNeed, Target, accepted_rows};

use crate::{Keep, Prompt, Session, SessionError};

mod policy;

use policy::{Exec, Fail, Policy, Refresh, Refused, Shape, Source, Walk};

/// The window's name in its refusals.
const WHAT: &str = "MTP window";

/// A width chooser's refusal is the session's: the window's own refusal,
/// by name.
impl From<runtime::WidthError> for SessionError {
    fn from(e: runtime::WidthError) -> SessionError {
        SessionError::Refused(e.to_string())
    }
}

/// A refusal of the window's policy is the session's: the window's own
/// refusal, by name.
impl From<Refused> for SessionError {
    fn from(e: Refused) -> SessionError {
        SessionError::Refused(e.0)
    }
}

/// A call that ran walks ends as the walk's error, or as the window's
/// refusal.
impl From<Fail<GpuError>> for SessionError {
    fn from(e: Fail<GpuError>) -> SessionError {
        match e {
            Fail::Refused(r) => r.into(),
            Fail::Exec(e) => e.into(),
        }
    }
}

/// How a walk runs ([`MtpBody::walk`], [`MtpBody::chain`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkMode {
    /// Enqueued launch by launch.
    Eager,
    /// Replayed from its capture, captured on first use.
    Graph,
    /// Enqueued launch by launch through the store's append and no further:
    /// the rows' keys and values, which are all a later row reads of them (a
    /// row's input is its token and the target's hidden row, never the
    /// draft's output). No flash, feed-forward block or head, no readback
    /// and no own row: the prompt's warmup. The store's bits are a
    /// [`WalkMode::Eager`] walk's. A body with no such walk refuses it by
    /// name; a chain never takes it.
    Store,
}

impl From<StepMode> for WalkMode {
    /// A step's mode as a walk's: the window's walks with a head run as the
    /// target's steps do.
    fn from(m: StepMode) -> WalkMode {
        match m {
            StepMode::Eager => WalkMode::Eager,
            StepMode::Graph => WalkMode::Graph,
        }
    }
}

/// Where a walk's hidden rows come from; `A` is the body's
/// [`MtpBody::Arena`].
#[derive(Clone, Copy, Debug)]
pub enum Hidden<'a, A> {
    /// Rows written from the host, [`MtpBody::hidden_width`] values a row:
    /// the row at position 0 reads a zero row, the target holding nothing
    /// before it.
    Host(&'a [f32]),
    /// The target's final hidden rows in the arena `walk`, from its row
    /// `first` on, as the target's last call of that arena left them.
    Target { walk: A, first: usize },
}

/// A walk's rows: `tokens` at positions `pos0 ..`, each beside its hidden
/// row (the target's at the position before it).
#[derive(Clone, Copy, Debug)]
pub struct Feed<'a, A> {
    pub tokens: &'a [u32],
    pub pos0: u32,
    pub hidden: Hidden<'a, A>,
}

/// A prompt call's tap ([`MtpBody::prompt_with`]): called after each unit
/// the call runs with the arena its rows' final hidden rows sit in, the
/// unit's first position and its rows.
pub type UnitSink<'a, B> =
    dyn FnMut(&mut GpuModel<B>, <B as MtpBody>::Arena, u32, usize) -> Result<(), GpuError> + 'a;

/// A body whose load carries an MTP draft layer beside the target, drafted
/// in the session's verify ([`Rows`], [`Rollback`]). [`MtpBody::walk`] and
/// [`MtpBody::chain`] are refused by name on a load without the layer and on
/// a poisoned model; [`MtpBody::head`], [`MtpBody::held`] and
/// [`MtpBody::hidden_width`] answer the load's defaults.
///
/// A partial accept is the body's own: the window calls
/// [`runtime::Verify::commit`] with the rows kept and nothing else, and the
/// body's [`Rollback::rollback_on`], which that commit calls on every verify
/// (a whole accept too), settles its recurrent state for the rows dropped.
/// The window rewinds nothing.
pub trait MtpBody: Prompt + Keep + Rows + Rollback {
    /// The most ids a proposal holds (the window's `Draft::WIDTH`): a verify
    /// runs them behind the token at the target's position.
    const WIDTH: usize;

    /// The most rows a verify runs, the window's `M`: a proposal and the
    /// token before it. [`MtpBody::FITS`] holds it to [`MtpBody::WIDTH`] + 1.
    const VERIFY_ROWS: usize = Self::WIDTH + 1;

    /// The most rows one walk takes: a prompt unit's rows are walked in runs
    /// of at most this many, and a refresh (at most [`MtpBody::VERIFY_ROWS`]
    /// rows) is one walk.
    const WALK_ROWS: usize;

    /// Whether the chain's one readback holds each proposed id's probability
    /// beside it, so [`MtpBody::chain`] takes `p` at no further device work;
    /// a body whose readback holds the ids alone refuses `p` by name, and its
    /// window reports certainty ([`Draft::propose_p`]'s default).
    const PROBS: bool = false;

    /// The most rows one store walk ([`WalkMode::Store`], the prompt's
    /// warmup) takes: a body whose store walk runs wider than its whole walks
    /// names its width; the default is [`MtpBody::WALK_ROWS`].
    const STORE_ROWS: usize = Self::WALK_ROWS;

    /// Holds when the widths fit together: at least one id a proposal, a
    /// verify the body's passes take ([`Rows::MAX_ROWS`]), a refresh one
    /// walk. The window reads it, so a body that breaks it fails to compile
    /// where it is drafted.
    const FITS: () = assert!(
        Self::WIDTH >= 1
            && Self::VERIFY_ROWS == Self::WIDTH + 1
            && Self::VERIFY_ROWS <= <Self as Rows>::MAX_ROWS
            && Self::VERIFY_ROWS <= Self::WALK_ROWS,
        "an MTP body's WIDTH + 1 rows must fit one verify and one walk"
    );

    /// A target arena whose final hidden rows a walk reads: where a step, a
    /// verify and each prompt unit leave their rows.
    type Arena: Copy + Eq + fmt::Debug;

    /// The arena a plain step leaves its one row in. The next step overwrites
    /// it, so rows waiting there are walked before that step or are lost, and
    /// the window refuses by name a step after rows that wait there. A body
    /// whose verify writes this same arena ([`MtpBody::VERIFY_ARENA`] equal
    /// to it) makes that refusal fire on a plain step after an accept.
    const STEP_ARENA: Self::Arena;

    /// The arena a verify leaves its rows in, row `r` the verify's row `r`.
    /// The commit leaves them in place: the next window's refresh reads the
    /// kept rows there.
    const VERIFY_ARENA: Self::Arena;

    /// Which projection the draft's head runs: the full vocabulary, or a row
    /// list the load opened.
    type Head: Copy + fmt::Debug;

    /// How the target runs a prompt: the body's own paths (steps, passes,
    /// ubatches).
    type Path: Copy + fmt::Debug;

    /// The head the load opened for the draft. Load-time only.
    fn head(m: &GpuModel<Self>) -> Result<Self::Head, GpuError>;

    /// Values of one hidden row a walk reads: the length of each row of a
    /// [`Hidden::Host`] feed.
    fn hidden_width(m: &GpuModel<Self>) -> Result<usize, GpuError>;

    /// Positions of the current sequence the draft's store holds: a walk may
    /// start at or below it, never past it. The model's reset empties it.
    fn held(m: &GpuModel<Self>) -> Result<usize, GpuError>;

    /// The prompt call: `ids` from the model's position by `path`, and the
    /// argmax after the last. `sink` is called after each unit the call runs,
    /// whatever the path's unit is (a step, a pass, a ubatch), with the arena
    /// its rows' final hidden rows sit in, the unit's first position and its
    /// rows, before the next unit overwrites them: the window walks the draft
    /// over the unit there, filling the draft's store. The units cover the
    /// prompt in order, each from the position the one before ended at, the
    /// last ending at the prompt's end. A sink's error ends the call with
    /// it. A prompt the body refuses (past the context, an id it does not
    /// take) is refused before any unit runs. `None` is the body's plain
    /// prompt call.
    fn prompt_with(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        path: Self::Path,
        sink: Option<&mut UnitSink<'_, Self>>,
    ) -> Result<u32, GpuError>;

    /// One walk of the draft with no readback: `feed`'s rows, at most
    /// [`MtpBody::WALK_ROWS`] from a position at most [`MtpBody::held`],
    /// through the draft layer into `head`, enqueued launch by launch or
    /// replayed from a capture, writing the draft's store at their positions.
    /// It writes only the draft's store and arena, never the target's. A
    /// fault the walk raises stays on the card's fault word, which the next
    /// readback names.
    fn walk(
        m: &mut GpuModel<Self>,
        feed: Feed<'_, Self::Arena>,
        head: Self::Head,
        mode: WalkMode,
    ) -> Result<(), GpuError>;

    /// One window's chain, one readback, never in [`WalkMode::Store`]:
    /// `refresh` walked (the rows the target kept with their hidden rows, its
    /// last row the target's own next token), then `own` walks, each reading the walk before it on the card;
    /// the proposal, `1 + own` ids with the refresh's last row's prediction
    /// first, written to the front of `out`, and how many. With `p`, each
    /// id's probability among the head's rows, as the same readback holds
    /// it, written to the front of `p`; a body whose readback holds none
    /// refuses `p` by name. `out` or `p` shorter than the proposal is
    /// refused by name. A fault any walk raised is the chain's error and
    /// poisons the model as the target's would.
    fn chain(
        m: &mut GpuModel<Self>,
        refresh: Feed<'_, Self::Arena>,
        own: usize,
        head: Self::Head,
        mode: WalkMode,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError>;
}

/// One drafted window as the draft saw it ([`MtpDraft::keep_windows`]): the
/// target's position before its verify, the proposal's ids and each one's
/// probability among the head's rows, how many of those ids the pass
/// verified (the width chooser's cut; the proposal's whole length when
/// nothing cut it) and how many of them the target kept.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowDraft {
    pub pos: u32,
    pub ids: Vec<u32>,
    pub p: Vec<f32>,
    /// The ids the pass verified, at most the proposal's length.
    pub width: usize,
    pub accepted: usize,
}

/// What [`MtpDraft::keep_windows`] keeps: the last proposal until its
/// accept, and every window since the last take.
struct Windows {
    pending: Option<(Vec<u32>, Vec<f32>)>,
    kept: Vec<WindowDraft>,
}

/// What a draft holds of the target's sequence between two calls, kept
/// apart while another sequence runs ([`MtpDraft::park`]) and put back with
/// it ([`MtpDraft::unpark`]): the rows waiting for its next walk, or why it
/// proposes nothing.
#[derive(Clone, Debug)]
pub struct Parked<A> {
    next: Option<Refresh<A>>,
    last_unit: Option<(A, usize)>,
    skip: Option<&'static str>,
}

/// How a prompt call that continues a held sequence joined the draft to it
/// ([`MtpDraft::take_joined`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Join {
    /// The call's first position.
    pub start: u32,
    /// The rows the draft walked before the call to hold every position
    /// below it.
    pub caught_up: usize,
    /// Why the draft proposes nothing for the call's generation; `None`
    /// when it drafts.
    pub skipped: Option<&'static str>,
}

/// The MTP draft over a body's loaded draft layer (the module doc): the host
/// policy of the windows ([`Policy`]: which rows wait, which walks run, the
/// refresh the next chain opens with) and its executor on the card — which
/// head the load opened, how the prompt is fed, the walks themselves. A
/// caller drives it as `Speculative<MtpDraft<B>, M>` with `M` =
/// [`MtpBody::VERIFY_ROWS`], which [`Session::with_draft`] holds to the
/// draft's width.
pub struct MtpDraft<B: MtpBody> {
    /// The prompt path.
    path: B::Path,
    head: B::Head,
    /// What the window knows between two calls, and every decision a call
    /// makes of it.
    policy: Policy<B::Arena>,
    /// A zero hidden row: the row at position 0's input.
    zeros: Vec<f32>,
    /// The windows kept under [`MtpDraft::keep_windows`]; `None`, nothing
    /// is kept.
    windows: Option<Windows>,
    /// The last chain's probabilities, when it read them: a body whose
    /// readback holds them ([`MtpBody::PROBS`]) behind the width chooser,
    /// or the kept windows.
    p: Vec<f32>,
}

/// The card's side of a call's walks ([`Exec`]): the model, the head the load
/// opened and the zero row, for the length of one call.
struct Card<'a, B: MtpBody> {
    m: &'a mut GpuModel<B>,
    head: B::Head,
    zeros: &'a [f32],
}

impl<B: MtpBody> Exec<B::Arena> for Card<'_, B> {
    type Err = GpuError;

    fn walk(&mut self, w: &Walk<'_, B::Arena>) -> Result<(), GpuError> {
        let hidden = match w.hidden {
            Source::Zero => Hidden::Host(self.zeros),
            Source::Target { walk, first } => Hidden::Target { walk, first },
        };
        let feed = Feed {
            tokens: w.tokens,
            pos0: w.pos0,
            hidden,
        };
        B::walk(self.m, feed, self.head, w.mode)
    }

    fn held(&mut self) -> Result<usize, GpuError> {
        B::held(self.m)
    }
}

impl<B: MtpBody> MtpDraft<B> {
    /// The draft over the model `m` loaded beside its target: the head the
    /// load opened (a row list or the full vocabulary), the prompt fed by
    /// `path`, the walks with a head and the chains run in `mode`.
    /// Load-time only.
    pub fn open(m: &GpuModel<B>, path: B::Path, mode: StepMode) -> Result<Self, SessionError> {
        let () = B::FITS;
        let head = B::head(m)?;
        let zeros = vec![0.0; B::hidden_width(m)?];
        Ok(MtpDraft {
            path,
            head,
            policy: Policy::new(Shape {
                step: B::STEP_ARENA,
                verify: B::VERIFY_ARENA,
                mode: WalkMode::from(mode),
                width: B::WIDTH,
                store_rows: B::STORE_ROWS,
            }),
            zeros,
            windows: None,
            p: vec![0.0; B::WIDTH],
        })
    }

    /// Keep every drafted window from here on ([`WindowDraft`]): the chain
    /// hands back each id's probability, which its one readback already
    /// holds. Load-time only.
    pub fn keep_windows(&mut self) {
        self.windows = Some(Windows {
            pending: None,
            kept: Vec::new(),
        });
    }

    /// The windows kept since the last take, in order; empty when
    /// [`MtpDraft::keep_windows`] was never called.
    pub fn take_windows(&mut self) -> Vec<WindowDraft> {
        self.windows
            .as_mut()
            .map(|w| std::mem::take(&mut w.kept))
            .unwrap_or_default()
    }

    /// The join of the last call that continued a held sequence, once: a
    /// second take is `None` until another call joins.
    pub fn take_joined(&mut self) -> Option<Join> {
        self.policy.take_joined()
    }

    /// The head the load opened.
    #[must_use]
    pub fn head(&self) -> B::Head {
        self.head
    }

    /// The draft's own state started over: no refresh waits, no skip, the
    /// model's reset having emptied the draft's store. The session's reset
    /// calls it.
    pub fn restart(&mut self) {
        self.policy.restart();
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// Whether the draft proposes nothing until the next prompt call from
    /// position 0 or a restart ([`Join::skipped`] names why).
    #[must_use]
    pub fn skipping(&self) -> bool {
        self.policy.skipping()
    }

    /// The next refresh recorded for a verify of `rows` that ran from
    /// position `p0`, keeping its first `accepted` rows, `out` the id the
    /// pass took at each ([`Draft::accept`]), before the commit takes the
    /// rest back: the kept rows at the positions after the verify's first,
    /// each with the hidden row the verify's row before it wrote
    /// ([`MtpBody::VERIFY_ARENA`]), the last row the id the pass took after
    /// the last kept row — the argmax on a greedy pass, the draw on a
    /// sampled one. What [`Draft::accept`] records, `p0` given by the caller
    /// that ran the verify.
    pub fn record(
        &mut self,
        p0: u32,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        self.policy.record(p0, rows, out, accepted)?;
        if let Some(w) = &mut self.windows {
            let (ids, p) = w.pending.take().ok_or_else(|| {
                SessionError::Refused(format!(
                    "{WHAT}: an accept of a window whose proposal the draft did not keep"
                ))
            })?;
            w.kept.push(WindowDraft {
                pos: p0,
                ids,
                p,
                width: rows.len() - 1,
                accepted: accepted - 1,
            });
        }
        Ok(())
    }

    /// What the draft holds of the target's sequence between two calls, for
    /// the caller's sequence state ([`MtpDraft::unpark`] puts it back): the
    /// rows waiting for its next walk when they sit in the step's or the
    /// verify's arena ([`MtpBody::STEP_ARENA`], [`MtpBody::VERIFY_ARENA`]),
    /// or why it proposes nothing — its own skip, or rows a prompt call's
    /// arena holds, which a sequence state does not keep. The caller's
    /// state keeps the draft's store and those two arenas' rows with the
    /// positions they hold, so the walk after the put-back reads the rows it
    /// would have read; a body whose walk does not refuse an arena that holds
    /// other positions must not park.
    #[must_use]
    pub fn park(&self) -> Parked<B::Arena> {
        self.policy.park()
    }

    /// `p`, which [`MtpDraft::park`] took, back in place once the target's
    /// sequence stands where it stood then (the caller's state put back,
    /// the draft's store and the two arenas with it): the next call walks
    /// the waiting rows as it would have, or skips for the parked why, its
    /// [`Join`] naming it.
    pub fn unpark(&mut self, p: &Parked<B::Arena>) {
        self.policy.unpark(p);
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// Before a plain step of `last` at the target's position: a refresh an
    /// earlier step or window left waiting reads an arena the step may
    /// overwrite (its own), so it is walked now, its last row `last` — a
    /// first step that continues the held sequence joins it here. A draft
    /// that cannot walk it skips, the [`Join`] naming why
    /// ([`MtpDraft::take_joined`]). A prompt call's anchor is left for
    /// [`Draft::stepped`]. A server seat calls it before each of its steps,
    /// and the width chooser makes the same walk through
    /// [`Draft::before_plain`] before each pass it holds back.
    ///
    /// # Errors
    ///
    /// The walk's.
    pub fn before_step(&mut self, t: &mut Session<B>, last: u32) -> Result<(), SessionError> {
        let pos = t.pos();
        let mut card = Card {
            m: t.model_mut(),
            head: self.head,
            zeros: &self.zeros,
        };
        Ok(self.policy.before_step(&mut card, pos, last)?)
    }
}

/// The proposal depth a caller gives each slot's window of a pass of two
/// slots ([`SlotWindow::depth`]) unless it chooses another: two rows a slot,
/// a pass of four rows, the shape the cost model prices first at two slots.
/// The depth is the caller's data, given a window at a time.
pub const SLOT_DEPTH: usize = 1;

/// One busy slot's drafted window in a pass of several slots
/// ([`pass_slots`]): the slot, its own draft (the window's state for that
/// slot's sequence: each slot holds one), the token at its position, the
/// most ids its proposal holds, and where its kept ids are appended.
pub struct SlotWindow<'a, B: MtpBody> {
    pub slot: usize,
    pub draft: &'a mut MtpDraft<B>,
    pub last: u32,
    pub depth: usize,
    pub out: &'a mut Vec<u32>,
}

/// One drafted round of several slots in one pass of the target, each
/// window's in turn as [`runtime::Speculative`]'s pass runs it for one
/// sequence, the verifies of every window run together:
/// - each window's slot selected and its draft's proposal made, at most its
///   `depth` ids ([`Draft::propose`] into that much room);
/// - every window's rows — the token at its slot's position, then the
///   proposal — run as one pass ([`Session::verify_slots`]), each row bit
///   for bit its step;
/// - each window's kept rows by the runtime's rule (row 0, then while the
///   argmax is the next proposed id), recorded by its draft from the
///   position its rows ran from ([`MtpDraft::record`]) before the commit of
///   every window's kept rows together ([`Session::commit_slots`]);
/// - each window's kept ids appended to its `out`.
///
/// Each slot's proposals, kept rows and ids are bit for bit its own window
/// run alone: its verify rows are its steps, and its draft walks its own
/// store over the hidden rows its own rows left. The session stands with
/// slot 0 selected. A draft that proposes nothing is one that skips (an
/// [`MtpDraft`] proposes at least one id otherwise): its row is a plain step
/// the draft is told nothing of, as its [`Draft::stepped`] would do nothing.
/// Refused by name: a depth of 0, a draft that proposes nothing while it
/// drafts, and one that moves its slot's position.
pub fn pass_slots<B>(
    t: &mut Session<B>,
    windows: &mut [SlotWindow<'_, B>],
) -> Result<Vec<runtime::Committed>, SessionError>
where
    B: MtpBody + SlotRows,
    B::Seq: 'static,
{
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(windows.len());
    let mut firsts = Vec::with_capacity(windows.len());
    for w in windows.iter_mut() {
        if w.depth == 0 {
            return Err(SessionError::Refused(format!(
                "{WHAT}: slot {}'s window of depth 0 (a plain row is a pass of no draft)",
                w.slot
            )));
        }
        t.select_slot(w.slot)?;
        let pos = t.pos();
        let mut ids = vec![w.last; 1 + w.depth.min(B::WIDTH)];
        let n = Draft::propose(w.draft, t, w.last, &mut ids[1..])?;
        if t.pos() != pos {
            return Err(SessionError::Refused(format!(
                "{WHAT}: slot {}'s proposal moved the target from position {pos} to {}",
                w.slot,
                t.pos()
            )));
        }
        if n == 0 && !w.draft.skipping() {
            return Err(SessionError::Refused(format!(
                "{WHAT}: slot {}'s draft proposed nothing while it drafts",
                w.slot
            )));
        }
        ids.truncate(1 + n);
        rows.push(ids);
        firsts.push(pos);
    }
    let pass: Vec<(usize, &[u32])> = windows
        .iter()
        .zip(&rows)
        .map(|(w, r)| (w.slot, &r[..]))
        .collect();
    let argmax = t.verify_slots(&pass)?.ids;
    let mut kept = Vec::with_capacity(windows.len());
    let mut at = 0;
    for ((w, r), &p0) in windows.iter_mut().zip(&rows).zip(&firsts) {
        let got = argmax.get(at..at + r.len()).ok_or_else(|| {
            SessionError::Refused(format!(
                "{WHAT}: a pass of {} rows read back {} ids",
                at + r.len(),
                argmax.len()
            ))
        })?;
        let k = accepted_rows(r, got);
        if r.len() > 1 {
            w.draft.record(p0, r, got, k)?;
        }
        kept.push(k);
        at += r.len();
    }
    t.commit_slots(&kept)?;
    let mut done = Vec::with_capacity(windows.len());
    let mut at = 0;
    for (((w, r), &k), &p0) in windows.iter_mut().zip(&rows).zip(&kept).zip(&firsts) {
        w.out.extend_from_slice(&argmax[at..at + k]);
        at += r.len();
        done.push(runtime::Committed {
            pos: p0,
            kept: k,
            rows: r.len(),
            proposed: r.len() > 1,
        });
    }
    Ok(done)
}

impl<B: MtpBody> MtpDraft<B> {
    /// One chain, one readback (the module doc): the recorded refresh's
    /// walk — its last row `last`, the token at the target's position, whose
    /// prediction is the first proposal — then own walks while the context
    /// holds them and the proposal fits `out`: at most `out.len()` ids, so a
    /// caller caps a window's depth by the room it hands. No room is refused
    /// by name. The refresh's last row is `last` whatever the call that
    /// recorded it took there: a step tells the draft its argmax, and a
    /// sampled request feeds its draw. With `read_p`, each id's probability
    /// as the same readback holds it, into the draft's own `p`.
    fn chain(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        out: &mut [u32],
        read_p: bool,
    ) -> Result<usize, SessionError> {
        let Some(c) = self.policy.chain(t.pos(), t.ctx(), last, out.len())? else {
            return Ok(0);
        };
        let n = B::chain(
            t.model_mut(),
            c.refresh.feed(),
            c.own,
            self.head,
            c.mode,
            out,
            read_p.then_some(&mut self.p[..]),
        )?;
        if let Some(w) = &mut self.windows {
            w.pending = Some((out[..n].to_vec(), self.p[..n].to_vec()));
        }
        Ok(n)
    }
}

impl<B: MtpBody> Draft<Session<B>> for MtpDraft<B> {
    /// The most ids a proposal holds: the body's [`MtpBody::WIDTH`].
    const WIDTH: usize = B::WIDTH;
    const TAPS: TapNeed = TapNeed::Final;

    /// The prompt through the target's own call, the draft walked over its
    /// units as they complete (the module doc): position 0 first — the
    /// target holds no hidden row before it — or, for a call that continues
    /// the held sequence, the rows the last call left waiting, the call's
    /// first id their last ([`Join`]); then every unit's rows from its second
    /// on, each with the hidden row its own arena holds for the position
    /// before it, the last unit's last row left for `begin`. A draft that
    /// cannot join skips: the target's call alone.
    fn prompt(&mut self, t: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError> {
        let (path, head) = (self.path, self.head);
        let start = t.pos();
        let _busy = t.model().prompt_busy();
        let mut card = Card {
            m: t.model_mut(),
            head,
            zeros: &self.zeros,
        };
        let Some(mut call) = self.policy.prompt_start(&mut card, start, ids)? else {
            return Ok(B::prompt_with(t.model_mut(), ids, path, None)?);
        };
        let (policy, zeros) = (&mut self.policy, &self.zeros[..]);
        let mut sink = |m: &mut GpuModel<B>,
                        walk: B::Arena,
                        first: u32,
                        rows: usize|
         -> Result<(), GpuError> {
            policy.unit(&mut call, &mut Card { m, head, zeros }, walk, first, rows)
        };
        Ok(B::prompt_with(t.model_mut(), ids, path, Some(&mut sink))?)
    }

    /// The next window's anchor: the target's own token `first` at the
    /// prompt's end, with the hidden row the prompt's last position wrote —
    /// unless a step since the prompt call already recorded one (the
    /// server's cut: the prompt less its last id, then the last id as a
    /// step, [`Draft::stepped`] recording the anchor the step's window
    /// proposes from).
    fn begin(&mut self, t: &Session<B>, _prompt: &[u32], first: u32) -> Result<(), SessionError> {
        Ok(self.policy.begin(t.pos(), first)?)
    }

    /// One chain, one readback ([`MtpDraft::chain`]); each id's
    /// probability read back only for the kept windows
    /// ([`MtpDraft::keep_windows`]).
    fn propose(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        out: &mut [u32],
    ) -> Result<usize, SessionError> {
        self.chain(t, last, out, self.windows.is_some())
    }

    /// The same chain with each proposed id's probability among the head's
    /// rows written to `p` beside it, from the readback that already holds
    /// it ([`MtpBody::PROBS`]); a body whose readback holds none reports
    /// certainty, every 1.
    fn propose_p(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        out: &mut [u32],
        p: &mut [f32],
    ) -> Result<usize, SessionError> {
        let n = self.chain(t, last, out, B::PROBS || self.windows.is_some())?;
        let places = p.len();
        let p = p.get_mut(..n).ok_or_else(|| {
            SessionError::Refused(format!(
                "{WHAT}: a proposal of {n} ids into {places} probabilities"
            ))
        })?;
        if B::PROBS {
            p.copy_from_slice(&self.p[..n]);
        } else {
            p.fill(1.0);
        }
        Ok(n)
    }

    /// The last proposal was never verified: its kept window, if any, is
    /// dropped, so the next accept records the window that ran.
    fn unproposed(&mut self) {
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// The next refresh recorded before the commit takes the rejected rows
    /// back ([`MtpDraft::record`]), the verify's first position the one its
    /// rows stood the target past.
    fn accept(
        &mut self,
        t: &mut Session<B>,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        self.record(t.pos() - rows.len() as u32, rows, out, accepted)
    }

    /// A plain step on `last`: the rows that waited for the next chain —
    /// the refresh an accept recorded, after a prompt call with no `begin`
    /// the anchor (`last` at the prompt's end), or on a first step at
    /// position 0 its row beside a zero hidden row — walked now, since
    /// the step's row follows them and the store must hold every position
    /// below the next chain's; then `next`, the id taken after the step, at
    /// the position after it, with the hidden row the step wrote
    /// ([`MtpBody::STEP_ARENA`]), as the next refresh, its rows left for
    /// the call that walks them — the width chooser's hook
    /// ([`Draft::before_plain`]) or a seat's step before the next plain
    /// step, the next chain for its head, a continuing prompt's join — so
    /// the store never holds a position past the target's, which a
    /// sequence state needs. Refused by name when the waiting rows read
    /// the step's own arena, which the step has overwritten.
    fn stepped(&mut self, t: &mut Session<B>, last: u32, next: u32) -> Result<(), SessionError> {
        let pos = t.pos();
        let mut card = Card {
            m: t.model_mut(),
            head: self.head,
            zeros: &self.zeros,
        };
        Ok(self.policy.stepped(&mut card, pos, last, next)?)
    }

    /// The waiting rows walked before the chooser's held pass steps the
    /// target ([`MtpDraft::before_step`]: the step would overwrite the
    /// arena they read); nothing waits, and this does nothing, on every
    /// pass that runs a proposal.
    fn before_plain(&mut self, t: &mut Session<B>, last: u32) -> Result<(), SessionError> {
        self.before_step(t, last)
    }
}
