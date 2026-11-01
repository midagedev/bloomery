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
//!   anchor. A prompt that holds an image ([`MtpDraft::prompt_media`]) walks
//!   the same way over the target's media call, a span's positions walked
//!   like any other but for the embedding: the row the target consumed
//!   there, the span's own, in place of the image token's ([`Feed::media`]);
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
//! Those walks run through the store's append alone ([`WalkMode::Store`]):
//! no head reads them, the next chain walking its own refresh.
//! When no such rows
//! wait, or the store does not hold the positions below them, the draft
//! skips: it proposes nothing, every pass a plain step, until a prompt call
//! from position 0, a restart or a rejoin ([`MtpDraft::rejoin`]); the call's
//! [`Join`] names why.
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
//! the target's own argmax, or the request's draw from that target row.
//! The hidden rows never leave the card: a walk
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

use bloomery_gpu::checkpoint::bytes;
use bloomery_gpu::media::{CallRows, MediaBody};
use bloomery_gpu::model::{Rollback, Rows, SlotRows, Slots, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use runtime::{Draft, TapNeed, Target, accepted_rows};

use crate::{Keep, Prompt, Session, SessionError};

/// The types a draft walk is given, owned by the card crate (`bloomery_gpu`):
/// how a walk runs, where its hidden rows come from and the rows it walks.
pub use bloomery_gpu::draft::{DraftHead, Feed, Hidden, WalkMode};

mod policy;

use policy::{Exec, Fail, Policy, Refresh, Refused, Shape, Walk};

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
    /// walk and one store walk. The window reads it, so a body that breaks it
    /// fails to compile where it is drafted.
    const FITS: () = assert!(
        Self::WIDTH >= 1
            && Self::VERIFY_ROWS == Self::WIDTH + 1
            && Self::VERIFY_ROWS <= <Self as Rows>::MAX_ROWS
            && Self::VERIFY_ROWS <= Self::WALK_ROWS
            && Self::VERIFY_ROWS <= Self::STORE_ROWS,
        "an MTP body's WIDTH + 1 rows must fit one verify, one walk and one store walk"
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

    /// How the target runs a prompt: the body's own paths (steps, passes,
    /// ubatches).
    type Path: Copy + fmt::Debug;

    /// The head the load opened for the draft. Load-time only.
    fn head(m: &GpuModel<Self>) -> Result<DraftHead, GpuError>;

    /// Values of one hidden row a walk reads: the length of each row of a
    /// [`Hidden::Host`] feed.
    fn hidden_width(m: &GpuModel<Self>) -> Result<usize, GpuError>;

    /// Values of one embedding row: the width of a media span's rows, which a
    /// media prompt call's walks read in place of a position's token embedding
    /// ([`Feed::media`]). The default is [`MtpBody::hidden_width`], a body
    /// whose hidden row is one embedding row wide; a body whose walks read
    /// several streams a hidden row names its embedding's width.
    fn embed_width(m: &GpuModel<Self>) -> Result<usize, GpuError> {
        Self::hidden_width(m)
    }

    /// Positions of the current sequence the draft's store holds: a walk may
    /// start at or below it, never past it. The model's reset empties it.
    fn held(m: &GpuModel<Self>) -> Result<usize, GpuError>;

    /// Copy a shared prompt arena's final hidden row `first`, at position
    /// `pos - 1`, to the sequence's step arena row 0, including its record.
    fn retain_prompt_row(
        m: &mut GpuModel<Self>,
        walk: Self::Arena,
        first: usize,
        pos: u32,
    ) -> Result<(), GpuError>;

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

    /// A prompt call whose token is unused, with the same hidden-row sink.
    fn prefill_with(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        path: Self::Path,
        sink: Option<&mut UnitSink<'_, Self>>,
    ) -> Result<(), GpuError> {
        Self::prompt_with(m, ids, path, sink).map(|_| ())
    }

    /// Prepare a call's sequence state before the draft joins it, keep that state through
    /// `run`, and finish it on success or failure. Bodies with no per-call sequence state
    /// run the call directly; media rows are still supplied by `prompt_media_with`.
    fn media_frame(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        spans: &[<Self as MediaBody>::Span<'_>],
        run: impl FnOnce(&mut GpuModel<Self>) -> Result<u32, GpuError>,
    ) -> Result<u32, GpuError>
    where
        Self: MediaBody,
    {
        let _ = (ids, spans);
        run(m)
    }

    /// [`MtpBody::prompt_with`] of `ids` with `spans` spliced in ([`MediaBody::prefill_media`]):
    /// a span's rows stand in for its positions' token embeddings in the target. The caller
    /// runs this inside [`MtpBody::media_frame`], after the draft joined the prepared sequence
    /// state. `sink` is called after each unit as it is on a text prompt, with the target's hidden rows of the
    /// unit, a span position's among them. The call is the body's own prompt call, one residency
    /// pass it opens itself: a caller inside a pass of its own
    /// ([`Session::prompt_media`](crate::Session::prompt_media)) never calls this. No span is
    /// [`MtpBody::prompt_with`]. A body whose draft has no such call refuses it by name, the
    /// body's, before anything runs.
    fn prompt_media_with(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        spans: &[<Self as MediaBody>::Span<'_>],
        sink: Option<&mut UnitSink<'_, Self>>,
    ) -> Result<u32, GpuError>
    where
        Self: MediaBody,
    {
        let _ = (m, ids, spans, sink);
        Err(GpuError::Shape {
            what: std::any::type_name::<Self>(),
            detail: "this body's draft has no media prompt call".to_string(),
        })
    }

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
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<(), GpuError>;

    /// One store walk ([`WalkMode::Store`]) of several resident sequences'
    /// rows: each `(slot, feed)` the rows of that slot's draft, as
    /// [`MtpBody::walk`] walks them with the slot selected, every slot named
    /// once and at most [`MtpBody::STORE_ROWS`] rows in all. Each slot's
    /// draft store comes out as its own walk leaves it, and the selected
    /// slot stays selected. The default walks the slots in turn; a body
    /// whose store walk runs several sequences' rows at once overrides it.
    fn store_slots(
        m: &mut GpuModel<Self>,
        walks: &[(usize, Feed<'_, Self::Arena>)],
        head: DraftHead,
    ) -> Result<(), GpuError>
    where
        Self: Slots,
        Self::Seq: 'static,
    {
        let home = m.selected();
        let walked = walks.iter().try_for_each(|&(slot, feed)| {
            m.select_slot(slot)?;
            Self::walk(m, feed, head, WalkMode::Store)
        });
        let back = m.select_slot(home);
        walked.and(back)
    }

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
        head: DraftHead,
        mode: WalkMode,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError>;

    /// Several resident sequences' window chains in one call, every slot
    /// named once: each [`SlotChain`] the chain [`MtpBody::chain`] runs with
    /// its slot selected, its proposal written to the front of its `out` and,
    /// when it holds `p`, each id's probability to the front of `p`, bit for
    /// bit what that chain alone writes. Returns each chain's proposal
    /// length, in order. The default runs the chains in turn and leaves the
    /// selected slot selected; a body whose walks run several sequences'
    /// rows at once overrides it and may leave another slot selected.
    fn chain_slots(
        m: &mut GpuModel<Self>,
        chains: &mut [SlotChain<'_, Self::Arena>],
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Vec<usize>, GpuError>
    where
        Self: Slots,
        Self::Seq: 'static,
    {
        let home = m.selected();
        let mut lens = Vec::with_capacity(chains.len());
        let ran = chains.iter_mut().try_for_each(|c| {
            m.select_slot(c.slot)?;
            let n = Self::chain(m, c.refresh, c.own, head, mode, c.out, c.p.as_deref_mut())?;
            lens.push(n);
            Ok(())
        });
        let back = m.select_slot(home);
        ran.and(back).map(|()| lens)
    }
}

/// One slot's window chain of a round of several slots' chains
/// ([`MtpBody::chain_slots`]): the slot, the refresh its draft recorded (its
/// last row the token at the slot's position), the own walks after it, the
/// room of its proposal and, when its draft reads them, of each id's
/// probability.
pub struct SlotChain<'a, A> {
    pub slot: usize,
    pub refresh: Feed<'a, A>,
    pub own: usize,
    pub out: &'a mut [u32],
    pub p: Option<&'a mut [f32]>,
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

impl<A: Copy> Parked<A> {
    /// Its byte form ([`bytes`]), `code` naming each arena it holds rows of:
    /// the refresh waiting (u8: 0 none, 1 then its tokens as an array, its
    /// first position, u32, its arena's code, u8, and its first row, a
    /// count); the last prompt unit (u8: 0 none, 1 then its arena's code and
    /// its rows, a count); the skip, by its place in the window's reasons
    /// ([`bytes::Put::named`]).
    pub fn put(&self, w: &mut bytes::Put<'_>, code: impl Fn(A) -> u8) -> Result<(), bytes::Fail> {
        match &self.next {
            None => w.one(0u8)?,
            Some(r) => {
                w.one(1u8)?;
                w.array(&r.tokens)?;
                w.one(r.pos0)?;
                w.one(code(r.walk))?;
                w.count(r.first)?;
            }
        }
        match self.last_unit {
            None => w.one(0u8)?,
            Some((walk, rows)) => {
                w.one(1u8)?;
                w.one(code(walk))?;
                w.count(rows)?;
            }
        }
        w.named(self.skip, &policy::SKIPS, "a parked draft's skip")
    }

    /// What [`Parked::put`] wrote, `arena` the kind and row capacity of each
    /// code; refused by name for an unknown code or reason, an empty refresh,
    /// rows outside their arena, or a last unit of no rows.
    pub fn take(
        r: &mut bytes::Take<'_>,
        arena: impl Fn(u8) -> Option<(A, usize)>,
    ) -> Result<Parked<A>, bytes::Fail> {
        let arena_of = |r: &mut bytes::Take<'_>| {
            let c = r.one::<u8>("a parked draft's arena")?;
            arena(c).ok_or_else(|| bytes::refused(format!("a parked draft's arena of code {c}")))
        };
        let next = match r.one::<u8>("a parked draft's refresh")? {
            0 => None,
            1 => {
                let tokens: Vec<u32> = r.array("a parked draft's refresh")?;
                let pos0 = r.one("a parked draft's refresh")?;
                let (walk, capacity) = arena_of(r)?;
                let first = r.count("a parked draft's refresh")?;
                if tokens.is_empty()
                    || first
                        .checked_add(tokens.len())
                        .is_none_or(|end| end > capacity)
                {
                    return Err(bytes::refused("a parked draft's refresh outside its arena"));
                }
                Some(Refresh {
                    tokens,
                    pos0,
                    walk,
                    first,
                })
            }
            c => {
                return Err(bytes::refused(format!(
                    "a parked draft's refresh of code {c}"
                )));
            }
        };
        let last_unit = match r.one::<u8>("a parked draft's last unit")? {
            0 => None,
            1 => {
                let (walk, capacity) = arena_of(r)?;
                let rows = r.count("a parked draft's last unit")?;
                if rows == 0 {
                    return Err(bytes::refused("a parked draft's last unit of no rows"));
                }
                if rows > capacity {
                    return Err(bytes::refused("a parked draft's last unit past its arena"));
                }
                Some((walk, rows))
            }
            c => {
                return Err(bytes::refused(format!(
                    "a parked draft's last unit of code {c}"
                )));
            }
        };
        Ok(Parked {
            next,
            last_unit,
            skip: r.named(&policy::SKIPS, "a parked draft's skip")?,
        })
    }

    /// Refused by name unless the rows waiting can be walked on a state of
    /// `pos` positions, before the state is put back: the refresh's ids by
    /// `ids`, its last row the one at `pos`, and each walk's target rows (its
    /// arena, its first row and rows, its first position) by `rows`; the last
    /// unit read as the join's walk at `pos` reads it.
    pub fn check(
        &self,
        pos: u32,
        ids: impl Fn(&[u32]) -> Result<(), bytes::Fail>,
        rows: impl Fn(A, (usize, usize), u32) -> Result<(), bytes::Fail>,
    ) -> Result<(), bytes::Fail> {
        if let Some(r) = &self.next {
            ids(&r.tokens)?;
            if u64::from(r.pos0) + r.tokens.len() as u64 != u64::from(pos) + 1 {
                return Err(bytes::refused(format!(
                    "a parked draft's refresh of {} rows from position {} on a state of {pos}",
                    r.tokens.len(),
                    r.pos0
                )));
            }
            rows(r.walk, (r.first, r.tokens.len()), r.pos0)?;
        }
        if let Some((walk, n)) = self.last_unit {
            let last = n
                .checked_sub(1)
                .ok_or_else(|| bytes::refused("a parked draft's last unit of no rows"))?;
            rows(walk, (last, 1), pos)?;
        }
        Ok(())
    }
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
    head: DraftHead,
    /// What the window knows between two calls, and every decision a call
    /// makes of it.
    policy: Policy<B::Arena>,
    /// The windows kept under [`MtpDraft::keep_windows`]; `None`, nothing
    /// is kept.
    windows: Option<Windows>,
    /// The last chain's probabilities, when it read them: a body whose
    /// readback holds them ([`MtpBody::PROBS`]) behind the width chooser,
    /// or the kept windows.
    p: Vec<f32>,
}

/// The card's side of a call's walks ([`Exec`]): the model, the head the load
/// opened and the call's span rows (`None` on a text call), for the length of
/// one call. Every walk of the call reads `media`: the join's and each unit's.
struct Card<'a, B: MtpBody> {
    m: &'a mut GpuModel<B>,
    head: DraftHead,
    media: Option<&'a CallRows<'a>>,
}

impl<B: MtpBody> Exec<B::Arena> for Card<'_, B> {
    type Err = GpuError;

    fn walk(&mut self, w: &Walk<'_, B::Arena>) -> Result<(), GpuError> {
        let feed = Feed {
            tokens: w.tokens,
            pos0: w.pos0,
            hidden: w.hidden,
            media: self.media,
        };
        B::walk(self.m, feed, self.head, w.mode)
    }

    fn held(&mut self) -> Result<usize, GpuError> {
        B::held(self.m)
    }
}

/// The waiting step and verify rows of several slots' drafts, gathered
/// before a round's target call ([`MtpDraft::gather`]) and walked together
/// ([`SlotWalks::run`]): one store walk of every slot's rows
/// ([`MtpBody::store_slots`]) in place of one walk a slot. Rows gathered and
/// never run are lost to their drafts, whose next call then skips.
#[must_use = "the gathered rows wait for SlotWalks::run"]
pub struct SlotWalks<B: MtpBody> {
    head: Option<DraftHead>,
    walks: Vec<(usize, Refresh<B::Arena>)>,
}

impl<B: MtpBody> Default for SlotWalks<B> {
    fn default() -> Self {
        SlotWalks {
            head: None,
            walks: Vec::new(),
        }
    }
}

impl<B: MtpBody + Slots> SlotWalks<B>
where
    B::Seq: 'static,
{
    /// Every gathered slot's rows walked, in store walks of at most
    /// [`MtpBody::STORE_ROWS`] rows, before the target call that may
    /// overwrite the arenas they read; nothing stays gathered. The selected
    /// slot stays selected.
    ///
    /// # Errors
    ///
    /// The walk's.
    pub fn run(&mut self, t: &mut Session<B>) -> Result<(), SessionError> {
        let Some(head) = self.head.take() else {
            return Ok(());
        };
        let walks = std::mem::take(&mut self.walks);
        let feeds: Vec<(usize, Feed<'_, B::Arena>)> =
            walks.iter().map(|(slot, r)| (*slot, r.feed())).collect();
        let mut start = 0;
        let mut rows = 0;
        for (i, (_, f)) in feeds.iter().enumerate() {
            if rows + f.tokens.len() > B::STORE_ROWS {
                B::store_slots(t.model_mut(), &feeds[start..i], head)?;
                (start, rows) = (i, 0);
            }
            rows += f.tokens.len();
        }
        Ok(B::store_slots(t.model_mut(), &feeds[start..], head)?)
    }
}

/// The card's side of a gathered call ([`MtpDraft::gather`]): the positions
/// the selected slot's draft store holds read from the model. Walks of its
/// step and verify rows wait for [`SlotWalks::run`]; shared arenas are read
/// before the selected slot changes.
struct Gather<'a, B: MtpBody> {
    m: &'a mut GpuModel<B>,
    slot: usize,
    head: DraftHead,
    walks: &'a mut SlotWalks<B>,
}

impl<B: MtpBody> Exec<B::Arena> for Gather<'_, B> {
    type Err = SessionError;

    fn walk(&mut self, w: &Walk<'_, B::Arena>) -> Result<(), SessionError> {
        let (WalkMode::Store, Hidden::Target { walk, first }) = (w.mode, w.hidden) else {
            return Err(SessionError::Refused(format!(
                "{WHAT}: a gathered walk is a store walk of the target's rows"
            )));
        };
        if walk != B::STEP_ARENA && walk != B::VERIFY_ARENA {
            return Ok(Card {
                m: &mut *self.m,
                head: self.head,
                media: None,
            }
            .walk(w)?);
        }
        if *self.walks.head.get_or_insert(self.head) != self.head {
            return Err(SessionError::Refused(format!(
                "{WHAT}: the gathered walks name two heads"
            )));
        }
        self.walks.walks.push((
            self.slot,
            Refresh {
                tokens: w.tokens.to_vec(),
                pos0: w.pos0,
                walk,
                first,
            },
        ));
        Ok(())
    }

    fn held(&mut self) -> Result<usize, SessionError> {
        Ok(B::held(self.m)?)
    }
}

impl<B: MtpBody> Session<B> {
    /// Feed a prompt without producing a token, retaining the session's idle check.
    pub fn prefill(&mut self, ids: &[u32], path: B::Path) -> Result<(), SessionError> {
        self.idle("prefill")?;
        let _busy = self.model().prompt_busy();
        Ok(B::prefill_with(self.model_mut(), ids, path, None)?)
    }
}

impl<B: MtpBody> MtpDraft<B> {
    /// Feed prompt positions without requesting the target's head.
    pub fn prefill(&mut self, t: &mut Session<B>, ids: &[u32]) -> Result<(), SessionError> {
        let path = self.path;
        Ok(self.prompt_joined(t.model_mut(), ids, None, |m, sink| {
            B::prefill_with(m, ids, path, sink)
        })?)
    }
    /// The draft over the model `m` loaded beside its target: the head the
    /// load opened (a row list or the full vocabulary), the prompt fed by
    /// `path`, the chains run in `mode`. Load-time only.
    pub fn open(m: &GpuModel<B>, path: B::Path, mode: StepMode) -> Result<Self, SessionError> {
        let () = B::FITS;
        let head = B::head(m)?;
        let hidden_width = B::hidden_width(m)?;
        Ok(MtpDraft {
            path,
            head,
            policy: Policy::new(Shape {
                step: B::STEP_ARENA,
                verify: B::VERIFY_ARENA,
                mode: WalkMode::from(mode),
                width: B::WIDTH,
                store_rows: B::STORE_ROWS,
                hidden_width,
            }),
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
    pub fn head(&self) -> DraftHead {
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

    /// The target cut back to `pos`, or a state put back as its first `pos`
    /// positions, where the body rejoined the draft to the sequence: its
    /// store holds every position below `pos` and the step's arena
    /// ([`MtpBody::STEP_ARENA`]) the target's hidden row at `pos - 1`. The
    /// next call walks that row with the id it feeds at `pos` and drafts on,
    /// as after a plain step at `pos - 1`; nothing is skipped, and the next
    /// [`Join`] is the first since.
    pub fn rejoin(&mut self, pos: u32) {
        self.policy.rejoin(pos);
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
            media: None,
        };
        Ok(self.policy.before_step(&mut card, pos, last)?)
    }

    /// A plain row of `last` that a pass of several slots ran at the target's
    /// position `p0`, `next` the id taken after it, its rows waiting walked
    /// before the pass ([`MtpDraft::before_step`]): `next` at `p0 + 1`, with
    /// the hidden row the pass's row wrote ([`MtpBody::VERIFY_ARENA`]), as
    /// the next refresh, the refresh [`Draft::stepped`] records after a step
    /// of its own arena. A window of depth zero in [`pass_slots`] and a
    /// seat's round of plain steps run as one pass both end here.
    pub fn passed_plain(&mut self, p0: u32, last: u32, next: u32) -> Result<(), SessionError> {
        Ok(self.policy.record(p0, &[last], &[next], 1)?)
    }
}

impl<B: MtpBody + Slots> MtpDraft<B>
where
    B::Seq: 'static,
{
    /// Preserve the prompt's waiting row before another resident sequence
    /// can overwrite its shared arena. The target is never run again.
    pub fn retain_prompt(&mut self, t: &mut Session<B>) -> Result<(), SessionError> {
        let pos = t.pos();
        Ok(self.policy.retain_prompt(pos, |walk, first, pos| {
            B::retain_prompt_row(t.model_mut(), walk, first, pos)
        })?)
    }

    /// [`MtpDraft::before_step`] on the session's selected slot, its walk
    /// gathered into `walks` when it reads the slot's own step or verify
    /// arena. Shared prompt rows are walked immediately, before a slot
    /// change invalidates them. [`SlotWalks::run`] walks the gathered rows
    /// together before the target's next call. Every decision (the rows
    /// walked, a skip and its join) is made here as `before_step` makes it.
    ///
    /// # Errors
    ///
    /// The store's positions unread, or a walk `before_step` would not make.
    pub fn gather(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        walks: &mut SlotWalks<B>,
    ) -> Result<(), SessionError> {
        let (slot, pos) = (t.selected(), t.pos());
        let mut g = Gather {
            m: t.model_mut(),
            slot,
            head: self.head,
            walks,
        };
        self.policy.before_step(&mut g, pos, last)
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
/// most ids its proposal holds, where its kept ids are appended, and for a
/// request that draws its draw: the id it takes from a row's logits, one
/// call a taken id (`runtime::Sample`'s closure). `None` takes each row's
/// argmax.
pub struct SlotWindow<'a, B: MtpBody> {
    pub slot: usize,
    pub draft: &'a mut MtpDraft<B>,
    pub last: u32,
    pub depth: usize,
    pub out: &'a mut Vec<u32>,
    pub draw: Option<Draw<'a>>,
}

/// A drawing window's draw ([`SlotWindow::draw`]): the id it takes from a
/// row's logits, and whether it reads only the row's best
/// ([`Session::slots_best`]) in place of the whole row
/// ([`Session::slots_row`]).
pub struct Draw<'a> {
    pub take: &'a mut dyn FnMut(&[f32]) -> u32,
    pub best: bool,
}

/// One drafted round of several slots in passes of the target, each
/// window's in turn as [`runtime::Speculative`]'s pass runs it for one
/// sequence, the verifies of the windows run together:
/// - each window's slot selected and its draft's proposal made, at most its
///   `depth` ids (the chain [`Draft::propose`] runs into that much room),
///   every window's before any pass: a pass may write the arena whose rows a
///   draft walks. The chains whose refresh reads the step's or the verify's
///   arena, which a select keeps, run together once every window's chain is
///   decided ([`MtpBody::chain_slots`]: one walk a depth over every slot's
///   rows, on a body that runs them so); any other chain runs at once, before
///   the next select;
/// - the windows' rows — the token at its slot's position, then the
///   proposal — cut between windows into passes of at most
///   [`SlotRows::MAX_ROWS`] rows and [`SlotRows::MAX_SLOTS`] sequences,
///   the pass that holds slot 0 last
///   (`round_passes`), each pass run as one
///   ([`Session::verify_slots`]), each row bit for bit its step;
/// - each window's ids taken row by row in position order — the argmax, or
///   a drawing window's draw from that row's logits ([`Session::slots_row`],
///   or [`Session::slots_best`] for a draw that reads only the row's best)
///   — up to the first that is not the proposal's next (runtime's
///   `Speculative::pass_picking`), those rows kept, recorded by its draft
///   from the position its rows ran from ([`MtpDraft::record`]) with the
///   ids taken, and appended to its `out`, before the commit of the pass's
///   kept rows together ([`Session::commit_slots`]).
///
/// Each slot's proposals, kept rows and ids are bit for bit its own window
/// run alone: its verify rows are its steps, its draws the ones its pass
/// alone draws from them, and its draft walks its own store over the
/// hidden rows its own rows left. The session stands with
/// slot 0 selected. A draft that proposes nothing is one that skips (an
/// [`MtpDraft`] proposes at least one id otherwise): its row is a plain step
/// the draft is told nothing of, as its [`Draft::stepped`] would do nothing.
/// Depth zero runs a plain row and keeps the draft ready for a later pass:
/// each such draft's shared prompt rows are walked immediately, and its
/// step and verify rows are gathered ([`MtpDraft::gather`]) and walked
/// together before the first pass ([`SlotWalks::run`]).
/// Returns each window's commit, in the windows' order, and the passes run.
/// Refused by name: a draft that proposes nothing at a positive depth while
/// it drafts, one that moves its slot's position, and as
/// `round_passes` refuses the windows' rows.
pub fn pass_slots<B>(
    t: &mut Session<B>,
    windows: &mut [SlotWindow<'_, B>],
) -> Result<(Vec<runtime::Committed>, usize), SessionError>
where
    B: MtpBody + SlotRows,
    B::Seq: 'static,
{
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(windows.len());
    let mut lens = Vec::with_capacity(windows.len());
    let mut firsts = Vec::with_capacity(windows.len());
    let mut plain = SlotWalks::default();
    let mut batch = Vec::new();
    for (i, w) in windows.iter_mut().enumerate() {
        t.select_slot(w.slot)?;
        let pos = t.pos();
        let mut ids = vec![w.last; 1 + w.depth.min(B::WIDTH)];
        let n = if w.depth == 0 {
            w.draft.gather(t, w.last, &mut plain)?;
            0
        } else {
            let read_p = w.draft.windows.is_some();
            match w.draft.policy.chain(pos, t.ctx(), w.last, ids.len() - 1)? {
                None => 0,
                Some(c) if c.refresh.walk == B::STEP_ARENA || c.refresh.walk == B::VERIFY_ARENA => {
                    batch.push((i, c));
                    0
                }
                Some(c) => w.draft.run_chain(t, &c, &mut ids[1..], read_p)?,
            }
        };
        if t.pos() != pos {
            return Err(SessionError::Refused(format!(
                "{WHAT}: slot {}'s proposal moved the target from position {pos} to {}",
                w.slot,
                t.pos()
            )));
        }
        rows.push(ids);
        lens.push(n);
        firsts.push(pos);
    }
    for ((i, _), n) in batch
        .iter()
        .zip(slot_chains(t, windows, &mut rows, &batch)?)
    {
        lens[*i] = n;
    }
    for ((w, ids), &n) in windows.iter().zip(&mut rows).zip(&lens) {
        if n == 0 && w.depth > 0 && !w.draft.skipping() {
            return Err(SessionError::Refused(format!(
                "{WHAT}: slot {}'s draft proposed nothing while it drafts",
                w.slot
            )));
        }
        ids.truncate(1 + n);
    }
    plain.run(t)?;
    let cuts = crate::round_passes(
        windows.iter().zip(&rows).map(|(w, r)| (w.slot, r.len())),
        (<B as SlotRows>::MAX_ROWS, <B as SlotRows>::MAX_SLOTS),
    )?;
    let mut kept = vec![0; windows.len()];
    let mut taken = Vec::new();
    for cut in &cuts {
        let (ws, rs) = (&mut windows[cut.clone()], &rows[cut.clone()]);
        let pass: Vec<(usize, &[u32])> = ws.iter().zip(rs).map(|(w, r)| (w.slot, &r[..])).collect();
        let argmax = t.verify_slots(&pass)?.ids;
        let (mut fetched, mut best_fetched) = (false, false);
        let mut at = 0;
        for (((w, r), &p0), k) in ws
            .iter_mut()
            .zip(rs)
            .zip(&firsts[cut.clone()])
            .zip(&mut kept[cut.clone()])
        {
            let got = argmax.get(at..at + r.len()).ok_or_else(|| {
                SessionError::Refused(format!(
                    "{WHAT}: a pass of {} rows read back {} ids",
                    at + r.len(),
                    argmax.len()
                ))
            })?;
            let ids = match w.draw.as_mut() {
                None => {
                    *k = accepted_rows(r, got);
                    got
                }
                Some(draw) => {
                    taken.clear();
                    for i in 0..r.len() {
                        let id = if draw.best {
                            let row = t.slots_best(at + i, best_fetched)?;
                            best_fetched = true;
                            (draw.take)(row)
                        } else {
                            let row = t.slots_row(at + i, fetched)?;
                            fetched = true;
                            (draw.take)(row)
                        };
                        taken.push(id);
                        if r.get(i + 1) != Some(&id) {
                            break;
                        }
                    }
                    *k = taken.len();
                    &taken[..]
                }
            };
            if w.depth == 0 {
                w.draft.passed_plain(p0, w.last, ids[0])?;
            } else if r.len() > 1 {
                w.draft.record(p0, r, ids, *k)?;
            }
            w.out.extend_from_slice(&ids[..*k]);
            at += r.len();
        }
        t.commit_slots(&kept[cut.clone()])?;
    }
    let done = rows
        .iter()
        .zip(&kept)
        .zip(&firsts)
        .map(|((r, &k), &p0)| runtime::Committed {
            pos: p0,
            kept: k,
            rows: r.len(),
            proposed: r.len() > 1,
        })
        .collect();
    Ok((done, cuts.len()))
}

/// The chains of a pass's windows that [`pass_slots`] held for one call,
/// `batch` each window's index and the chain its policy decided, run
/// together ([`MtpBody::chain_slots`]), each proposal into its window's
/// `rows` after the token at its slot's position, kept by its draft as its
/// own chain keeps it. A lone chain, or chains of two heads or two walk
/// modes, run in turn on their own slots. Returns each chain's proposal
/// length, in `batch`'s order.
fn slot_chains<B>(
    t: &mut Session<B>,
    windows: &mut [SlotWindow<'_, B>],
    rows: &mut [Vec<u32>],
    batch: &[(usize, policy::Chain<B::Arena>)],
) -> Result<Vec<usize>, SessionError>
where
    B: MtpBody + Slots,
    B::Seq: 'static,
{
    let Some(&(first, ref c0)) = batch.first() else {
        return Ok(Vec::new());
    };
    let (head, mode) = (windows[first].draft.head, c0.mode);
    // One chain, or chains of drafts opened with different heads or walk
    // modes, run in turn: each slot selected, its arenas kept by the select.
    if batch.len() == 1
        || batch
            .iter()
            .any(|(i, c)| windows[*i].draft.head != head || c.mode != mode)
    {
        let mut lens = Vec::with_capacity(batch.len());
        for (i, c) in batch {
            let w = &mut windows[*i];
            t.select_slot(w.slot)?;
            let read_p = w.draft.windows.is_some();
            lens.push(w.draft.run_chain(t, c, &mut rows[*i][1..], read_p)?);
        }
        return Ok(lens);
    }
    let lens = {
        let mut next = batch.iter().peekable();
        let mut chains = Vec::with_capacity(batch.len());
        for (i, (w, ids)) in windows.iter_mut().zip(rows.iter_mut()).enumerate() {
            let Some((_, c)) = next.next_if(|(j, _)| *j == i) else {
                continue;
            };
            let read_p = w.draft.windows.is_some();
            chains.push(SlotChain {
                slot: w.slot,
                refresh: c.refresh.feed(),
                own: c.own,
                out: &mut ids[1..],
                p: read_p.then_some(&mut w.draft.p[..]),
            });
        }
        B::chain_slots(t.model_mut(), &mut chains, head, mode)?
    };
    for ((i, _), &n) in batch.iter().zip(&lens) {
        windows[*i].draft.proposed(&rows[*i][1..1 + n]);
    }
    Ok(lens)
}

impl<B: MtpBody> MtpDraft<B> {
    /// One chain, one readback (the module doc): the recorded refresh's
    /// walk — its last row `last`, the token at the target's position, whose
    /// prediction is the first proposal — then own walks while the verify of
    /// the proposal fits the context and the proposal fits `out`: at most
    /// `out.len()` ids, so a caller caps a window's depth by the room it
    /// hands. No room is refused by name. The refresh's last row is `last`
    /// whatever the call that recorded it took there: a step tells the draft
    /// its argmax, and a sampled request feeds its draw. With `read_p`, each
    /// id's probability as the same readback holds it, into the draft's own
    /// `p`.
    fn chain(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        out: &mut [u32],
        read_p: bool,
    ) -> Result<usize, SessionError> {
        match self.policy.chain(t.pos(), t.ctx(), last, out.len())? {
            None => Ok(0),
            Some(c) => self.run_chain(t, &c, out, read_p),
        }
    }

    /// The chain `c` the policy decided, run on the session's selected
    /// slot ([`MtpBody::chain`]) into `out`, with each id's probability into
    /// the draft's own `p` when `read_p`, and kept ([`MtpDraft::proposed`]).
    fn run_chain(
        &mut self,
        t: &mut Session<B>,
        c: &policy::Chain<B::Arena>,
        out: &mut [u32],
        read_p: bool,
    ) -> Result<usize, SessionError> {
        let n = B::chain(
            t.model_mut(),
            c.refresh.feed(),
            c.own,
            self.head,
            c.mode,
            out,
            read_p.then_some(&mut self.p[..]),
        )?;
        self.proposed(&out[..n]);
        Ok(n)
    }

    /// The proposal `ids` a chain made, its probabilities in the draft's
    /// own `p`, kept until its accept under [`MtpDraft::keep_windows`].
    fn proposed(&mut self, ids: &[u32]) {
        if let Some(w) = &mut self.windows {
            w.pending = Some((ids.to_vec(), self.p[..ids.len()].to_vec()));
        }
    }
}

impl<B: MtpBody> MtpDraft<B> {
    /// The prompt call's one core ([`Draft::prompt`], [`MtpDraft::prompt_media`]): `call` is the
    /// target's prompt call of `ids` on the model, handed the sink that walks the draft over
    /// each unit as it completes, or `None` when the draft cannot join and the call is the
    /// target's alone. `media` is the call's span rows ([`CallRows`], `None` on a text call):
    /// the join's walk and every unit's walk read it.
    fn prompt_joined<R>(
        &mut self,
        m: &mut GpuModel<B>,
        ids: &[u32],
        media: Option<&CallRows<'_>>,
        call: impl FnOnce(&mut GpuModel<B>, Option<&mut UnitSink<'_, B>>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        let head = self.head;
        let start = m.pos();
        let _busy = m.prompt_busy();
        let mut card = Card { m, head, media };
        let Some(mut join) = self.policy.prompt_start(&mut card, start, ids)? else {
            return call(m, None);
        };
        let policy = &mut self.policy;
        let mut sink = |m: &mut GpuModel<B>,
                        walk: B::Arena,
                        first: u32,
                        rows: usize|
         -> Result<(), GpuError> {
            let mut card = Card { m, head, media };
            policy.unit(&mut join, &mut card, walk, first, rows)
        };
        call(m, Some(&mut sink))
    }
}

impl<B: MtpBody + MediaBody> MtpDraft<B> {
    /// [`Draft::prompt`] of `ids` with `spans` spliced in (the module doc's prompt call, the
    /// target's media call in place of its text call): the draft is walked over the call's
    /// units as they complete, a span's positions included. At a span's position the draft
    /// reads the span's row beside the target's hidden row there, the row the target consumed
    /// in place of the image token's embedding, widened exactly: the reference's feed (its MTP
    /// draft adds the target's vision row at an image position). The call is the body's own,
    /// one residency pass it opens ([`MtpBody::prompt_media_with`]), inside the prepared
    /// sequence state ([`MtpBody::media_frame`]) from the join through the last unit, so it
    /// never runs inside [`Session::prompt_media`]'s. Refused before anything runs as [`Session::prompt_media`]
    /// refuses ([`MediaBody::media_refusal`]), and for a span that is not whole rows of the
    /// embedding's width ([`MtpBody::embed_width`]) or that the others do not follow in order
    /// ([`CallRows::new`]).
    pub fn prompt_media(
        &mut self,
        t: &mut Session<B>,
        ids: &[u32],
        spans: &[<B as MediaBody>::Span<'_>],
    ) -> Result<u32, SessionError> {
        if let Some(why) = B::media_refusal(t.model(), spans.len()) {
            return Err(SessionError::Refused(why));
        }
        let rows = CallRows::new(
            t.pos(),
            B::embed_width(t.model())?,
            spans.iter().map(|s| {
                let s = B::media_span(s);
                (&s.at, s.rows)
            }),
        )
        .map_err(|r| SessionError::Refused(format!("{WHAT}: {r}")))?;
        Ok(B::media_frame(t.model_mut(), ids, spans, |m| {
            self.prompt_joined(m, ids, Some(&rows), |m, sink| {
                B::prompt_media_with(m, ids, spans, sink)
            })
        })?)
    }
}

impl<B: MtpBody> Draft<Session<B>> for MtpDraft<B> {
    /// The most ids a proposal holds: the body's [`MtpBody::WIDTH`].
    const WIDTH: usize = B::WIDTH;
    const TAPS: TapNeed = TapNeed::Final;
    /// Each id past the first is one of the chain's own walks, which stop at
    /// the room the proposal is handed.
    const CHAIN: bool = true;

    /// The prompt through the target's own call, the draft walked over its
    /// units as they complete (the module doc): position 0 first — the
    /// target holds no hidden row before it — or, for a call that continues
    /// the held sequence, the rows the last call left waiting, the call's
    /// first id their last ([`Join`]); then every unit's rows from its second
    /// on, each with the hidden row its own arena holds for the position
    /// before it, the last unit's last row left for `begin`. A draft that
    /// cannot join skips: the target's call alone.
    fn prompt(&mut self, t: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError> {
        let path = self.path;
        Ok(self.prompt_joined(t.model_mut(), ids, None, |m, sink| {
            B::prompt_with(m, ids, path, sink)
        })?)
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
            media: None,
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
