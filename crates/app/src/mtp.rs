//! The shared MTP window on the card: [`MtpDraft`], the runtime's
//! `runtime::mtp::MtpDraft` over a [`Session<B>`], and the [`MtpBody`] adapter
//! the session's [`Host`] impl drives a body's draft layer through, over the
//! body a [`Session<B>`] wraps, with the model-independent types its calls
//! take. Qwen3.8's adapter is `arch::qwen3moe`'s, over `Body38`.
//!
//! The window — the proposal, the verify, the refresh, the prompt call's
//! walks, the skips and the joins — is `runtime::mtp`'s (its module doc); the
//! runtime's rule drives it as [`runtime::Speculative`] over a
//! [`runtime::Draft`] on the session: a proposal of up to [`MtpBody::WIDTH`]
//! ids, one verify of up to [`MtpBody::VERIFY_ROWS`] rows, and the commit of
//! the rows the target agrees with. What only the card has lives here:
//!
//! - the adapter ([`MtpBody`]) and the session's [`Host`] impl over it, the
//!   executor of a call's walks ([`Card`]) among them;
//! - a prompt that holds an image ([`prompt_media`]) walks the draft the same
//!   way over the target's media call, a span's positions walked like any
//!   other but for the embedding: the row the target consumed there, the
//!   span's own, in place of the image token's ([`Feed::media`]);
//! - a pass of several slots' windows ([`pass_slots`], [`SlotWalks`]), which
//!   walks several resident sequences' rows at once ([`SlotDraft`]);
//! - a parked draft's byte form ([`ParkedBytes`]).
//!
//! A partial accept is the body's own: the window calls
//! [`runtime::Verify::commit`] with the rows kept and nothing else, and the
//! session's commit takes the rest back through the body's
//! [`Rollback::rollback_on`], which settles whatever recurrent state the body
//! holds for the rows it drops.

use std::fmt;

use bloomery_gpu::checkpoint::bytes;
use bloomery_gpu::media::{CallRows, MediaBody};
use bloomery_gpu::model::{Rollback, Rows, SlotRows, Slots, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use runtime::Target;
use runtime::prompt::CallEnd;

use crate::{Draw, Keep, Prompt, RoundWindow, Session, SessionError};

/// The types a draft walk is given: which head it runs and the rows it walks
/// are the card crate's (`bloomery_gpu`); how it runs, where its hidden rows
/// come from and how a call joined the draft to a held sequence are the
/// runtime's (`runtime::mtp`), which holds no card.
pub use bloomery_gpu::draft::{DraftHead, Feed};
use bloomery_gpu::draft::{Proposal, SlotDraft};
use runtime::mtp::{Chain, Exec, Host, HostSink, Refresh, Refused, SKIPS, Unit, WHAT, Walk};
pub use runtime::mtp::{Hidden, Join, Parked, WalkMode, WindowDraft};

/// The MTP draft over a session: `runtime::mtp::MtpDraft` with the session as
/// its [`Host`] ([`open_draft`] opens it). A caller drives it as
/// `Speculative<MtpDraft<B>, M>` with `M` = [`MtpBody::VERIFY_ROWS`], which
/// [`Session::with_draft`] holds to the draft's width.
pub type MtpDraft<B> = runtime::mtp::MtpDraft<Session<B>>;

/// The draft over the model `m` loaded beside its target: the head the load
/// opened (a row list or the full vocabulary), the prompt fed by `path`, the
/// chains run in `mode`. Load-time only.
pub fn open_draft<B: MtpBody>(
    m: &GpuModel<B>,
    path: B::Path,
    mode: StepMode,
) -> Result<MtpDraft<B>, SessionError> {
    let head = B::head(m)?;
    let hidden_width = B::hidden_width(m)?;
    Ok(runtime::mtp::MtpDraft::new(
        path,
        head,
        WalkMode::from(mode),
        hidden_width,
    ))
}

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

/// A refresh's rows as a walk's feed: the card crate's [`Feed`] names the
/// call's span rows, which the runtime's values do not know.
trait RefreshFeed<A> {
    /// The refresh's feed.
    fn feed(&self) -> Feed<'_, A>;
}

impl<A: Copy> RefreshFeed<A> for Refresh<A> {
    fn feed(&self) -> Feed<'_, A> {
        Feed {
            tokens: &self.tokens,
            pos0: self.pos0,
            hidden: Hidden::Target {
                walk: self.walk,
                first: self.first,
            },
            media: None,
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
/// [`MtpBody::hidden_width`] answer the load's defaults. A pass of several
/// slots' windows ([`pass_slots`], [`SlotWalks`]) runs on a body whose draft
/// also walks several resident sequences' rows at once ([`SlotDraft`]).
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

    /// The prompt call: `ids` from the model's position by `path`, ending as
    /// `end` asks ([`CallEnd`]): the argmax after the last when it runs the
    /// head ([`CallEnd::head`]), `None` for [`CallEnd::Unread`]. A body asked
    /// for [`CallEnd::Close`] runs the last id as the call's last row with the
    /// checkpoints of the call less that id, or refuses it by name before
    /// anything runs; a seat asks it only of a body that folds the id.
    /// `sink` is called after each unit the call runs,
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
        end: CallEnd,
        sink: Option<&mut UnitSink<'_, Self>>,
    ) -> Result<Option<u32>, GpuError>;

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
}

/// A chain's readback `d` written to the front of `out` and, when asked, its
/// probabilities to the front of `p`; how many. Room short of the proposal,
/// and probabilities asked of a readback that holds none, are refused by
/// name.
pub fn write_proposal(
    d: &Proposal,
    out: &mut [u32],
    p: Option<&mut [f32]>,
) -> Result<usize, GpuError> {
    const WHAT: &str = "mtp proposal";
    let n = d.tokens.len();
    let places = out.len();
    out.get_mut(..n)
        .ok_or_else(|| {
            GpuError::shape(WHAT, format!("a proposal of {n} ids into {places} places"))
        })?
        .copy_from_slice(&d.tokens);
    if let Some(p) = p {
        if d.p.len() != n {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a proposal's probabilities: its readback holds {} for {n} ids",
                    d.p.len()
                ),
            ));
        }
        let places = p.len();
        p.get_mut(..n)
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("a proposal's {n} probabilities into {places} places"),
                )
            })?
            .copy_from_slice(&d.p);
    }
    Ok(n)
}

/// What a draft holds of the target's sequence between two calls, kept
/// apart while another sequence runs ([`MtpDraft::park`]) and put back with
/// it ([`MtpDraft::unpark`]): the rows waiting for its next walk, or why it
/// proposes nothing ([`Parked`], the runtime's values). The byte form and the
/// check against a state are this crate's, over the checkpoint's bytes
/// ([`bytes`]).
pub trait ParkedBytes: Sized {
    /// The arena kind the parked rows name.
    type Arena: Copy;

    /// Its byte form ([`bytes`]), `code` naming each arena it holds rows of:
    /// the refresh waiting (u8: 0 none, 1 then its tokens as an array, its
    /// first position, u32, its arena's code, u8, and its first row, a
    /// count); the last prompt unit (u8: 0 none, 1 then its arena's code and
    /// its rows, a count); the skip, by its place in the window's reasons
    /// ([`bytes::Put::named`]).
    fn put(
        &self,
        w: &mut bytes::Put<'_>,
        code: impl Fn(Self::Arena) -> u8,
    ) -> Result<(), bytes::Fail>;

    /// What [`ParkedBytes::put`] wrote, `arena` the kind and row capacity of each
    /// code; refused by name for an unknown code or reason, an empty refresh,
    /// rows outside their arena, or a last unit of no rows.
    fn take(
        r: &mut bytes::Take<'_>,
        arena: impl Fn(u8) -> Option<(Self::Arena, usize)>,
    ) -> Result<Self, bytes::Fail>;

    /// Refused by name unless the rows waiting can be walked on a state of
    /// `pos` positions, before the state is put back: the refresh's ids by
    /// `ids`, its last row the one at `pos`, and each walk's target rows (its
    /// arena, its first row and rows, its first position) by `rows`; the last
    /// unit read as the join's walk at `pos` reads it.
    fn check(
        &self,
        pos: u32,
        ids: impl Fn(&[u32]) -> Result<(), bytes::Fail>,
        rows: impl Fn(Self::Arena, (usize, usize), u32) -> Result<(), bytes::Fail>,
    ) -> Result<(), bytes::Fail>;
}

impl<A: Copy> ParkedBytes for Parked<A> {
    type Arena = A;

    fn put(&self, w: &mut bytes::Put<'_>, code: impl Fn(A) -> u8) -> Result<(), bytes::Fail> {
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
        w.named(self.skip, &SKIPS, "a parked draft's skip")
    }

    fn take(
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
            skip: r.named(&SKIPS, "a parked draft's skip")?,
        })
    }

    fn check(
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

/// The session as the window's host: each call is one call into the body's
/// own API ([`MtpBody`]), the walks of a text call with no span rows.
impl<B: MtpBody> Host for Session<B> {
    const WIDTH: usize = B::WIDTH;
    const WALK_ROWS: usize = B::WALK_ROWS;
    const STORE_ROWS: usize = B::STORE_ROWS;
    const PROBS: bool = B::PROBS;
    /// The body's own check, which also holds [`MtpBody::VERIFY_ROWS`] to the
    /// proposal's width.
    const FITS: () = B::FITS;

    type Arena = B::Arena;
    const STEP_ARENA: B::Arena = B::STEP_ARENA;
    const VERIFY_ARENA: B::Arena = B::VERIFY_ARENA;
    type Path = B::Path;
    type Head = DraftHead;
    type Cause = GpuError;

    fn held(&self) -> Result<usize, GpuError> {
        B::held(self.model())
    }

    fn walk(&mut self, w: &Walk<'_, B::Arena>, head: DraftHead) -> Result<(), GpuError> {
        let feed = Feed {
            tokens: w.tokens,
            pos0: w.pos0,
            hidden: w.hidden,
            media: None,
        };
        B::walk(self.model_mut(), feed, head, w.mode)
    }

    fn chain(
        &mut self,
        c: &Chain<B::Arena>,
        head: DraftHead,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError> {
        B::chain(
            self.model_mut(),
            c.refresh.feed(),
            c.own,
            head,
            c.mode,
            out,
            p,
        )
    }

    fn prompt_with(
        &mut self,
        ids: &[u32],
        path: B::Path,
        end: CallEnd,
        head: DraftHead,
        sink: Option<&mut HostSink<'_, Self>>,
    ) -> Result<Option<u32>, GpuError> {
        let m = self.model_mut();
        match sink {
            None => B::prompt_with(m, ids, path, end, None),
            Some(sink) => {
                let mut unit = unit_sink::<B>(sink, head, None);
                B::prompt_with(m, ids, path, end, Some(&mut unit))
            }
        }
    }

    fn retain_prompt_row(
        &mut self,
        walk: B::Arena,
        first: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        B::retain_prompt_row(self.model_mut(), walk, first, pos)
    }

    /// The engine watchdog's mark of a prompt call, held from the draft's join
    /// through the call's last unit: the join's walk waits on the card for
    /// the copy its inputs ride on.
    fn prompt_scope<R>(&mut self, call: impl FnOnce(&mut Self) -> R) -> R {
        let _busy = self.model().prompt_busy();
        call(self)
    }
}

/// A prompt call's sink ([`Unit`]) as the body's unit sink
/// ([`MtpBody::prompt_with`]): each unit's walks run on a [`Card`] over the
/// unit's model, `media` the call's span rows (`None` on a text call).
fn unit_sink<'s, B: MtpBody>(
    sink: &'s mut Unit<'_, B::Arena, GpuError>,
    head: DraftHead,
    media: Option<&'s CallRows<'s>>,
) -> impl FnMut(&mut GpuModel<B>, B::Arena, u32, usize) -> Result<(), GpuError> + 's {
    move |m, walk, first, rows| sink(&mut Card { m, head, media }, walk, first, rows)
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
/// before a round's target call ([`SlotWalks::gather`]) and walked together
/// ([`SlotWalks::run`]): one store walk of every slot's rows
/// ([`GpuModel::draft_store_slots`]) in place of one walk a slot. Rows
/// gathered and never run are lost to their drafts, whose next call then
/// skips.
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

impl<B: MtpBody + SlotDraft<<B as MtpBody>::Arena>> SlotWalks<B>
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
                t.model_mut().draft_store_slots(&feeds[start..i], head)?;
                (start, rows) = (i, 0);
            }
            rows += f.tokens.len();
        }
        Ok(t.model_mut().draft_store_slots(&feeds[start..], head)?)
    }
}

/// The card's side of a gathered call ([`SlotWalks::gather`]): the positions
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

impl<B: MtpBody + Slots> SlotWalks<B>
where
    B::Seq: 'static,
{
    /// [`MtpDraft::before_step`] of `draft` on the session's selected slot, its
    /// walk gathered here when it reads the slot's own step or verify arena.
    /// Shared prompt rows are walked immediately, before a slot change
    /// invalidates them. [`SlotWalks::run`] walks the gathered rows together
    /// before the target's next call. Every decision (the rows walked, a skip
    /// and its join) is made here as `before_step` makes it.
    ///
    /// # Errors
    ///
    /// The store's positions unread, or a walk `before_step` would not make.
    pub fn gather(
        &mut self,
        draft: &mut MtpDraft<B>,
        t: &mut Session<B>,
        last: u32,
    ) -> Result<(), SessionError> {
        let (slot, pos) = (t.selected(), t.pos());
        let mut g = Gather {
            m: t.model_mut(),
            slot,
            head: draft.head(),
            walks: self,
        };
        draft.policy_mut().before_step(&mut g, pos, last)
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

/// One drafted round of several slots in passes of the target, each
/// window's in turn as [`runtime::Speculative`]'s pass runs it for one
/// sequence, the verifies of the windows run together:
/// - each window's slot selected and its draft's proposal made, at most its
///   `depth` ids (the chain [`Draft::propose`] runs into that much room),
///   every window's before any pass: a pass may write the arena whose rows a
///   draft walks. The chains whose refresh reads the step's or the verify's
///   arena, which a select keeps, run together once every window's chain is
///   decided ([`GpuModel::draft_chain_slots`]: one walk a depth over every
///   slot's rows); any other chain runs at once, before the next select;
/// - the windows' rows — the token at its slot's position, then the
///   proposal — verified together by the session's one owner of a round of
///   any draft's windows ([`Session::verify_rounds`]: passes cut between
///   windows at the body's [`SlotRows`] bounds, each row's id taken, the
///   kept rows committed), each window's kept rows recorded by its draft
///   from the position its rows ran from ([`MtpDraft::record`]) with the
///   ids taken before the commit.
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
/// step and verify rows are gathered ([`SlotWalks::gather`]) and walked
/// together before the first pass ([`SlotWalks::run`]).
/// Returns each window's commit, in the windows' order, and the passes run.
/// Refused by name: a draft that proposes nothing at a positive depth while
/// it drafts, one that moves its slot's position, and as
/// [`Session::verify_rounds`] refuses the windows' rows.
pub fn pass_slots<B>(
    t: &mut Session<B>,
    windows: &mut [SlotWindow<'_, B>],
) -> Result<(Vec<runtime::Committed>, usize), SessionError>
where
    B: MtpBody + SlotRows + SlotDraft<<B as MtpBody>::Arena>,
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
            plain.gather(w.draft, t, w.last)?;
            0
        } else {
            let read_p = w.draft.keeps_windows();
            match w
                .draft
                .policy_mut()
                .chain(pos, t.ctx(), w.last, ids.len() - 1)?
            {
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
    let mut tells = Vec::with_capacity(windows.len());
    let mut round = Vec::with_capacity(windows.len());
    for ((w, ids), &pos) in windows.iter_mut().zip(&rows).zip(&firsts) {
        let SlotWindow {
            slot,
            draft,
            last,
            depth,
            out,
            draw,
        } = w;
        tells.push((&mut **draft, *last, *depth));
        round.push(RoundWindow {
            slot: *slot,
            pos,
            rows: ids,
            out,
            draw: draw.as_mut().map(|d| Draw {
                take: &mut *d.take,
                best: d.best,
            }),
        });
    }
    t.verify_rounds(&mut round, |i, ids, kept| {
        let (draft, last, depth) = &mut tells[i];
        let (p0, r) = (firsts[i], &rows[i]);
        if *depth == 0 {
            draft.passed_plain(p0, *last, ids[0])
        } else if r.len() > 1 {
            draft.record(p0, r, ids, kept)
        } else {
            Ok(())
        }
    })
}

/// The chains of a pass's windows that [`pass_slots`] held for one call,
/// `batch` each window's index and the chain its policy decided, run
/// together ([`GpuModel::draft_chain_slots`]), each proposal into its
/// window's `rows` after the token at its slot's position ([`write_proposal`])
/// and, when its draft keeps windows, each id's probability into the draft's
/// own `p`, kept by its draft as its own chain keeps it. A lone chain, or
/// chains of two heads or two walk modes, run in turn on their own slots.
/// Probabilities asked of a draft whose readback holds none
/// ([`MtpBody::PROBS`]) are refused by name before the round runs. Returns
/// each chain's proposal length, in `batch`'s order.
fn slot_chains<B>(
    t: &mut Session<B>,
    windows: &mut [SlotWindow<'_, B>],
    rows: &mut [Vec<u32>],
    batch: &[(usize, Chain<B::Arena>)],
) -> Result<Vec<usize>, SessionError>
where
    B: MtpBody + SlotDraft<<B as MtpBody>::Arena>,
    B::Seq: 'static,
{
    let Some(&(first, ref c0)) = batch.first() else {
        return Ok(Vec::new());
    };
    let (head, mode) = (windows[first].draft.head(), c0.mode);
    // One chain, or chains of drafts opened with different heads or walk
    // modes, run in turn: each slot selected, its arenas kept by the select.
    if batch.len() == 1
        || batch
            .iter()
            .any(|(i, c)| windows[*i].draft.head() != head || c.mode != mode)
    {
        let mut lens = Vec::with_capacity(batch.len());
        for (i, c) in batch {
            let w = &mut windows[*i];
            t.select_slot(w.slot)?;
            let read_p = w.draft.keeps_windows();
            lens.push(w.draft.run_chain(t, c, &mut rows[*i][1..], read_p)?);
        }
        return Ok(lens);
    }
    if !B::PROBS && batch.iter().any(|(i, _)| windows[*i].draft.keeps_windows()) {
        return Err(SessionError::Refused(format!(
            "{WHAT}: a chain's probabilities: this draft's readback holds its ids alone"
        )));
    }
    let feeds: Vec<_> = batch
        .iter()
        .map(|(i, c)| (windows[*i].slot, c.refresh.feed(), c.own))
        .collect();
    let drafts = t.model_mut().draft_chain_slots(&feeds, head, mode)?;
    let lens = batch
        .iter()
        .zip(&drafts)
        .map(|((i, _), d)| {
            let w = &mut windows[*i];
            write_proposal(d, &mut rows[*i][1..], w.draft.kept_probs())
        })
        .collect::<Result<Vec<_>, _>>()?;
    for ((i, _), &n) in batch.iter().zip(&lens) {
        windows[*i].draft.proposed(&rows[*i][1..1 + n]);
    }
    Ok(lens)
}

/// [`runtime::Draft::prompt`] of `ids` with `spans` spliced in (the window's prompt call, the
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
pub fn prompt_media<B: MtpBody + MediaBody>(
    draft: &mut MtpDraft<B>,
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
    let head = draft.head();
    Ok(B::media_frame(t.model_mut(), ids, spans, |m| {
        let start = m.pos();
        let _busy = m.prompt_busy();
        let mut card = Card {
            m,
            head,
            media: Some(&rows),
        };
        draft.prompt_joined(&mut card, start, ids, |card, sink| match sink {
            None => B::prompt_media_with(card.m, ids, spans, None),
            Some(sink) => {
                let mut unit = unit_sink::<B>(sink, head, Some(&rows));
                B::prompt_media_with(card.m, ids, spans, Some(&mut unit))
            }
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::{Parked, ParkedBytes, Refresh, SKIPS, write_proposal};
    use bloomery_gpu::GpuError;
    use bloomery_gpu::checkpoint::bytes;
    use bloomery_gpu::draft::Proposal;

    /// The arena kinds of a body whose prompt runs through ubatches or
    /// passes, its verify through the pass arena and its plain step through
    /// the step's: the shape Qwen3.8's window has.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ar {
        Step = 0,
        Pass = 1,
        Ubatch = 2,
    }

    /// A proposal fills the front of its rooms; room short of it, and
    /// probabilities asked of a readback that holds none, are refused by name.
    #[test]
    fn a_proposal_fills_its_rooms_or_is_refused_by_name() {
        let d = Proposal {
            tokens: vec![4, 5],
            p: vec![0.5, 0.25],
        };
        let (mut out, mut p) = ([0; 3], [0.0; 3]);
        assert_eq!(write_proposal(&d, &mut out, Some(&mut p)).expect("room"), 2);
        assert_eq!((out, p), ([4, 5, 0], [0.5, 0.25, 0.0]));
        let refused = |r: Result<usize, GpuError>| {
            matches!(
                r,
                Err(GpuError::Shape {
                    what: "mtp proposal",
                    ..
                })
            )
        };
        assert!(refused(write_proposal(&d, &mut [0; 1], None)));
        assert!(refused(write_proposal(&d, &mut out, Some(&mut [0.0; 1]))));
        let ids_alone = Proposal {
            tokens: vec![1],
            p: vec![],
        };
        assert!(refused(write_proposal(
            &ids_alone,
            &mut out,
            Some(&mut [0.0])
        )));
        assert_eq!(
            write_proposal(&ids_alone, &mut out, None).expect("ids alone"),
            1
        );
    }

    /// A parked draft goes to bytes and back whole, each arena by its code
    /// and each reason the window skips for by its place: a refresh waiting
    /// in the verify's arena, a last unit in the step's with a skip, and
    /// nothing; unknown codes, empty refreshes and rows outside their
    /// arena are refused by name, while rows ending at capacity pass.
    #[test]
    fn a_parked_draft_goes_to_bytes_and_back() {
        let arena = |c| {
            [Ar::Step, Ar::Pass, Ar::Ubatch]
                .get(usize::from(c))
                .copied()
                .map(|a| (a, 8))
        };
        let put = |p: &Parked<Ar>| {
            let mut out = Vec::new();
            let mut w = bytes::Put::new(&mut out);
            p.put(&mut w, |a| a as u8).unwrap();
            w.flush().unwrap();
            out
        };
        let take = |mut b: &[u8]| Parked::take(&mut bytes::Take::new(&mut b), arena);
        let parks = [
            Parked {
                next: Some(Refresh {
                    tokens: vec![4, 5, 0],
                    pos0: 9,
                    walk: Ar::Pass,
                    first: 1,
                }),
                last_unit: None,
                skip: None,
            },
            Parked {
                next: None,
                last_unit: Some((Ar::Step, 1)),
                skip: Some(SKIPS[3]),
            },
            Parked {
                next: None,
                last_unit: None,
                skip: None,
            },
        ];
        for p in &parks {
            let back = take(&put(p)).unwrap();
            assert_eq!(format!("{back:?}"), format!("{p:?}"));
        }
        let mut p = parks[0].clone();
        p.next.as_mut().unwrap().first = 5;
        take(&put(&p)).expect("a refresh ending at capacity");
        for first in [6, usize::MAX] {
            p.next.as_mut().unwrap().first = first;
            let why = take(&put(&p)).unwrap_err().to_string();
            assert!(why.contains("refresh outside its arena"), "{why}");
        }
        p.next.as_mut().unwrap().tokens.clear();
        p.next.as_mut().unwrap().first = 0;
        let why = take(&put(&p)).unwrap_err().to_string();
        assert!(why.contains("refresh outside its arena"), "{why}");
        let mut p = parks[1].clone();
        p.last_unit = Some((Ar::Step, 8));
        take(&put(&p)).expect("a last unit filling its arena");
        p.last_unit = Some((Ar::Step, 9));
        let why = take(&put(&p)).unwrap_err().to_string();
        assert!(why.contains("last unit past its arena"), "{why}");
        let mut b = put(&parks[1]);
        assert_eq!(
            b.last(),
            Some(&4),
            "the code saved files hold for PROMPT_ARENA"
        );
        *b.last_mut().unwrap() = SKIPS.len() as u8 + 1;
        let why = take(&b).unwrap_err().to_string();
        assert!(why.contains("a parked draft's skip of code 5"), "{why}");
        b = put(&parks[0]);
        b[1 + 8 + 3 * 4 + 4] = 3;
        let why = take(&b).unwrap_err().to_string();
        assert!(why.contains("a parked draft's arena of code 3"), "{why}");
        b = put(&parks[1]);
        b[3..11].fill(0);
        let why = take(&b).unwrap_err().to_string();
        assert!(
            why.contains("a parked draft's last unit of no rows"),
            "{why}"
        );
    }

    /// A parked draft is checked against the state it goes back on before
    /// the put back: the refresh's ids and rows as its walk reads them, its
    /// last row the state's next position, and a last unit's last row as the
    /// join's walk at that position reads it; either check's refusal is the
    /// check's.
    #[test]
    fn a_parked_draft_is_checked_against_its_state() {
        let seen = std::cell::RefCell::new(Vec::new());
        let ids = |ts: &[u32]| -> Result<(), bytes::Fail> {
            seen.borrow_mut().push(format!("ids {ts:?}"));
            Ok(())
        };
        let rows = |walk: Ar, (first, m): (usize, usize), pos0: u32| -> Result<(), bytes::Fail> {
            seen.borrow_mut()
                .push(format!("{walk:?} {first} {m} {pos0}"));
            Ok(())
        };
        let mut p = Parked {
            next: Some(Refresh {
                tokens: vec![4, 5, 0],
                pos0: 9,
                walk: Ar::Pass,
                first: 1,
            }),
            last_unit: None,
            skip: None,
        };
        p.check(11, ids, rows).unwrap();
        assert_eq!(*seen.borrow(), ["ids [4, 5, 0]", "Pass 1 3 9"]);
        for pos in [10, 12] {
            let why = p.check(pos, ids, rows).unwrap_err().to_string();
            assert!(why.contains("refresh of 3 rows from position 9"), "{why}");
        }
        let why = p
            .check(11, |_| Err(bytes::refused("an id")), rows)
            .unwrap_err();
        assert!(why.to_string().contains("an id"), "{why}");
        let why = p
            .check(11, ids, |_, _, _| Err(bytes::refused("a row")))
            .unwrap_err();
        assert!(why.to_string().contains("a row"), "{why}");
        p.next = None;
        p.last_unit = Some((Ar::Pass, 4));
        seen.borrow_mut().clear();
        p.check(7, ids, rows).unwrap();
        assert_eq!(*seen.borrow(), ["Pass 3 1 7"]);
    }
}
