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
//! overwrites the step's arena ([`MtpDraft::before_step`]). When no such rows
//! wait, or the store does not hold the positions below them, the draft
//! skips: it proposes nothing, every pass a plain step, until a prompt call
//! from position 0 or a restart; the call's [`Join`] names why.
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

use bloomery_gpu::model::{Rollback, Rows, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use runtime::{Draft, TapNeed, Target};

use crate::{Keep, Prompt, Session, SessionError};

/// The window's name in its refusals.
const WHAT: &str = "MTP window";

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
    /// first, written to the front of `out`, and how many. `out` shorter than
    /// the proposal is refused by name. A fault any walk raised is the
    /// chain's error and poisons the model as the target's would.
    fn chain(
        m: &mut GpuModel<Self>,
        refresh: Feed<'_, Self::Arena>,
        own: usize,
        head: Self::Head,
        mode: WalkMode,
        out: &mut [u32],
    ) -> Result<usize, GpuError>;
}

/// The refresh a window's chain opens with: its rows' tokens, their first
/// position, and where their hidden rows sit — the arena `walk`'s rows
/// `first` on, one row a walk row.
struct Refresh<A> {
    tokens: Vec<u32>,
    pos0: u32,
    walk: A,
    first: usize,
}

impl<A: Copy> Refresh<A> {
    /// The refresh's feed.
    fn feed(&self) -> Feed<'_, A> {
        Feed {
            tokens: &self.tokens,
            pos0: self.pos0,
            hidden: Hidden::Target {
                walk: self.walk,
                first: self.first,
            },
        }
    }
}

/// A join the draft cannot make: no rows of the held sequence wait for it.
const NOTHING_WAITS: &str = "no rows of the held sequence wait for the draft: it did not walk \
                             the sequence the target holds";
/// A join the draft cannot make: the waiting rows end elsewhere.
const ENDS_ELSEWHERE: &str = "the rows waiting for the draft do not end at the call's first \
                              position";
/// A join the draft cannot make: the store is behind the waiting rows.
const STORE_BEHIND: &str = "the draft's store does not hold the positions below the rows \
                            waiting for it";

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
/// policy of the windows — which head the load opened, how the prompt is
/// fed, the refresh the next chain opens with — the walks themselves on the
/// card. A caller drives it as `Speculative<MtpDraft<B>, M>` with `M` =
/// [`MtpBody::VERIFY_ROWS`], which [`Session::with_draft`] holds to the
/// draft's width.
pub struct MtpDraft<B: MtpBody> {
    /// The prompt path.
    path: B::Path,
    /// How the walks with a head and the chains run.
    mode: WalkMode,
    head: B::Head,
    /// The next window's refresh; `None` between a prompt call and its
    /// `begin`.
    next: Option<Refresh<B::Arena>>,
    /// The last prompt unit's arena and rows, for [`Draft::begin`]'s
    /// anchor.
    last_unit: Option<(B::Arena, usize)>,
    /// A zero hidden row: the row at position 0's input.
    zeros: Vec<f32>,
    /// The last join to a held sequence a call continued, until taken.
    joined: Option<Join>,
    /// Why the draft proposes nothing until the next prompt call from
    /// position 0 or restart; `None` while it drafts.
    skip: Option<&'static str>,
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
            mode: WalkMode::from(mode),
            head,
            next: None,
            last_unit: None,
            zeros,
            joined: None,
            skip: None,
        })
    }

    /// The join of the last call that continued a held sequence, once: a
    /// second take is `None` until another call joins.
    pub fn take_joined(&mut self) -> Option<Join> {
        self.joined.take()
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
        self.next = None;
        self.last_unit = None;
        self.joined = None;
        self.skip = None;
    }

    /// Before a plain step of `last` at the target's position: a refresh an
    /// earlier step or window left waiting reads an arena the step may
    /// overwrite (its own), so it is walked now, its last row `last` — a
    /// first step that continues the held sequence joins it here. A draft
    /// that cannot walk it skips, the [`Join`] naming why
    /// ([`MtpDraft::take_joined`]). A prompt call's anchor is left for
    /// [`Draft::stepped`]. A server seat calls it before each of its steps.
    ///
    /// # Errors
    ///
    /// The walk's.
    pub fn before_step(&mut self, t: &mut Session<B>, last: u32) -> Result<(), SessionError> {
        if self.skip.is_some() || self.next.is_none() {
            return Ok(());
        }
        let start = t.pos();
        if let Err(why) = self.catch_up(t, last)? {
            self.skip_from(start, why);
        }
        Ok(())
    }

    /// The rows an earlier call left waiting — the refresh a window or a
    /// step recorded, or a prompt call's anchor — walked with `token` as
    /// their last row, the row at the target's position, so the store holds
    /// every position through it: the rows walked, or why the draft cannot
    /// walk them. Nothing has run on the target since they were recorded
    /// (the caller's contract), so their arena still holds their hidden rows.
    fn catch_up(
        &mut self,
        t: &mut Session<B>,
        token: u32,
    ) -> Result<Result<usize, &'static str>, SessionError> {
        let here = t.pos();
        let mut r = match (self.next.take(), self.last_unit.take()) {
            (Some(r), _) => r,
            (None, Some((walk, rows))) => Refresh {
                tokens: vec![token],
                pos0: here,
                walk,
                first: rows - 1,
            },
            (None, None) => return Ok(Err(NOTHING_WAITS)),
        };
        if r.pos0 as usize + r.tokens.len() != here as usize + 1 {
            return Ok(Err(ENDS_ELSEWHERE));
        }
        let held = B::held(t.model())?;
        if r.pos0 as usize > held {
            return Ok(Err(STORE_BEHIND));
        }
        if let Some(l) = r.tokens.last_mut() {
            *l = token;
        }
        B::walk(t.model_mut(), r.feed(), self.head, self.mode)?;
        Ok(Ok(r.tokens.len()))
    }

    /// The draft proposes nothing from the call at `start` on, for `why`.
    fn skip_from(&mut self, start: u32, why: &'static str) {
        self.next = None;
        self.last_unit = None;
        self.skip = Some(why);
        self.joined = Some(Join {
            start,
            caught_up: 0,
            skipped: Some(why),
        });
    }

    /// One run of the warmup: `tokens` at `pos0`, their hidden rows the
    /// arena `walk`'s from `first` on, walked through the store's append
    /// alone ([`WalkMode::Store`]: a warm row's keys and values are all a
    /// later row reads of it) in runs of [`MtpBody::STORE_ROWS`] with no
    /// readback — a fault stays on the fault word, which the next readback
    /// (the first chain's) names.
    fn warm(
        &self,
        m: &mut GpuModel<B>,
        walk: B::Arena,
        pos0: u32,
        first: usize,
        tokens: &[u32],
    ) -> Result<(), GpuError> {
        for (i, run) in tokens.chunks(B::STORE_ROWS).enumerate() {
            B::walk(
                m,
                Feed {
                    tokens: run,
                    pos0: pos0
                        + u32::try_from(i * B::STORE_ROWS)
                            .expect("a prompt's positions lie below its context"),
                    hidden: Hidden::Target {
                        walk,
                        first: first + i * B::STORE_ROWS,
                    },
                },
                self.head,
                WalkMode::Store,
            )?;
        }
        Ok(())
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
        let Some(&id) = ids.first() else {
            return Ok(B::prompt_with(t.model_mut(), ids, path, None)?);
        };
        if start == 0 {
            self.next = None;
            self.last_unit = None;
            self.joined = None;
            self.skip = None;
        } else if let Some(why) = self.skip {
            self.skip_from(start, why);
        } else {
            match self.catch_up(t, id)? {
                Ok(rows) => {
                    self.joined = Some(Join {
                        start,
                        caught_up: rows,
                        skipped: None,
                    });
                }
                Err(why) => self.skip_from(start, why),
            }
        }
        if self.skip.is_some() {
            return Ok(B::prompt_with(t.model_mut(), ids, path, None)?);
        }
        let end_of_prompt = start + ids.len() as u32;
        let mut first_unit = start == 0;
        let mut sink = |m: &mut GpuModel<B>,
                        walk: B::Arena,
                        first: u32,
                        rows: usize|
         -> Result<(), GpuError> {
            self.last_unit = Some((walk, rows));
            // The row at `first + i` reads the target's hidden row at
            // `first + i − 1`: the unit's own arena holds it from its second
            // row on; the unit's first row was walked by the unit before (or
            // is position 0's zero-hidden row below, or the join's last row),
            // save the last unit's last row — the next window's anchor,
            // `begin`'s.
            let last = first + rows as u32 == end_of_prompt;
            if first_unit {
                first_unit = false;
                B::walk(
                    m,
                    Feed {
                        tokens: &ids[..1],
                        pos0: first,
                        hidden: Hidden::Host(&self.zeros),
                    },
                    head,
                    WalkMode::Store,
                )?;
            }
            let end = first + rows as u32 + 1 - u32::from(last);
            if end > first + 1 {
                let at = usize::try_from(first + 1 - start)
                    .expect("a prompt's positions lie below its context");
                let to = usize::try_from(end - start).expect("a prompt fits usize");
                // Position first + 1 + i reads the unit's row i, the hidden
                // row of the position before it.
                self.warm(m, walk, first + 1, 0, &ids[at..to])?;
            }
            Ok(())
        };
        let next = B::prompt_with(t.model_mut(), ids, path, Some(&mut sink))?;
        Ok(next)
    }

    /// The next window's anchor: the target's own token `first` at the
    /// prompt's end, with the hidden row the prompt's last position wrote.
    fn begin(&mut self, t: &Session<B>, _prompt: &[u32], first: u32) -> Result<(), SessionError> {
        if self.skip.is_some() {
            return Ok(());
        }
        let (walk, rows) = self.last_unit.ok_or_else(|| {
            SessionError::Refused(format!("{WHAT}: a prompt the draft never saw"))
        })?;
        self.next = Some(Refresh {
            tokens: vec![first],
            pos0: t.pos(),
            walk,
            first: rows - 1,
        });
        Ok(())
    }

    /// One chain, one readback (the module doc): the recorded refresh's
    /// walk — its last row the target's own next token, whose prediction is
    /// the first proposal — then own walks while the context holds them.
    fn propose(
        &mut self,
        t: &mut Session<B>,
        _last: u32,
        out: &mut [u32],
    ) -> Result<usize, SessionError> {
        if self.skip.is_some() {
            return Ok(0);
        }
        let Some(r) = self.next.take() else {
            return Err(SessionError::Refused(format!(
                "{WHAT}: a proposal before the draft's refresh (its prompt call, or the accept \
                 before it)"
            )));
        };
        // The refresh's last row is the token at the target's position, which
        // the target has not run yet: the refresh ends one past it.
        let end = r.pos0 as usize + r.tokens.len();
        let here = t.pos() as usize;
        if end != here + 1 {
            return Err(SessionError::Refused(format!(
                "{WHAT}: a refresh of {} rows ending at {end}, where the target stands at {here} \
                 (its next token's row ends at {})",
                r.tokens.len(),
                here + 1
            )));
        }
        let own = (Self::WIDTH - 1).min(t.ctx() as usize - end);
        Ok(B::chain(
            t.model_mut(),
            r.feed(),
            own,
            self.head,
            self.mode,
            out,
        )?)
    }

    /// The next refresh recorded before the commit takes the rejected rows
    /// back: the kept rows at the positions after the verify's first, each
    /// with the hidden row the verify's row before it wrote
    /// ([`MtpBody::VERIFY_ARENA`]), the last row the target's own next token.
    fn accept(
        &mut self,
        t: &mut Session<B>,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        let p0 = t.pos() - rows.len() as u32;
        let last = out.get(accepted - 1).copied().ok_or_else(|| {
            SessionError::Refused(format!(
                "{WHAT}: a verify that kept {accepted} rows read back no argmax"
            ))
        })?;
        self.next = Some(Refresh {
            tokens: rows[1..accepted].iter().copied().chain([last]).collect(),
            pos0: p0 + 1,
            walk: B::VERIFY_ARENA,
            first: 0,
        });
        Ok(())
    }

    /// A plain step on `last`: the rows that waited for the next chain —
    /// the refresh an accept recorded, after a prompt call with no `begin`
    /// the anchor (`last` at the prompt's end), or on a first step at
    /// position 0 its row beside a zero hidden row — walked now, since
    /// the step's row follows them and the store must hold every position
    /// below the next chain's; then the token `next` at the position the
    /// step's argmax names, with the hidden row the step wrote
    /// ([`MtpBody::STEP_ARENA`]), as the next refresh. Refused by name when
    /// the waiting rows read the step's own arena, which the step has
    /// overwritten.
    fn stepped(&mut self, t: &mut Session<B>, last: u32, next: u32) -> Result<(), SessionError> {
        if self.skip.is_some() {
            return Ok(());
        }
        let waiting = match (self.next.take(), self.last_unit.take()) {
            (Some(r), _) => Some(r),
            (None, Some((walk, rows))) => Some(Refresh {
                tokens: vec![last],
                pos0: t.pos() - 1,
                walk,
                first: rows - 1,
            }),
            (None, None) => {
                // A one-id prompt feeds no prompt call: the step ran position
                // 0, whose row reads a zero hidden row, as the prompt call's
                // first row does.
                if t.pos() == 1 {
                    B::walk(
                        t.model_mut(),
                        Feed {
                            tokens: &[last],
                            pos0: 0,
                            hidden: Hidden::Host(&self.zeros),
                        },
                        self.head,
                        self.mode,
                    )?;
                }
                None
            }
        };
        if let Some(r) = waiting {
            if r.walk == B::STEP_ARENA {
                return Err(SessionError::Refused(format!(
                    "{WHAT}: a step after rows whose hidden rows the step's own arena held (a \
                     prompt fed by steps): the step overwrote them"
                )));
            }
            B::walk(t.model_mut(), r.feed(), self.head, self.mode)?;
        }
        self.next = Some(Refresh {
            tokens: vec![next],
            pos0: t.pos(),
            walk: B::STEP_ARENA,
            first: 0,
        });
        Ok(())
    }
}
