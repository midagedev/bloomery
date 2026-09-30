//! What a model's MTP layer gives the window that drafts with it: the
//! [`MtpBody`] adapter over the body a [`Session<B>`] wraps, and the
//! model-independent types its calls take.
//!
//! An MTP layer's row at position `q` reads the token at `q` and the
//! target's final hidden row at `q − 1`, and predicts the token at `q + 1`.
//! The window drives it under the runtime's rule ([`runtime::Speculative`]
//! over a [`runtime::Draft`] on the session): a proposal of up to
//! [`MtpBody::WIDTH`] ids, one verify of up to [`MtpBody::VERIFY_ROWS`] rows,
//! and the commit of the rows the target agrees with. The calls here are the
//! ones the window makes on the model between those: the prompt call with a
//! sink over its units, the draft's walks and its chain, and what the load
//! opened (the head, the positions the draft's store holds, a hidden row's
//! width). Qwen3.8's is `arch::qwen3moe`'s, over `Body38`.
//!
//! The hidden rows never leave the card: a walk names them by the arena a
//! target call left them in and the row they start at ([`Hidden::Target`]).
//! [`runtime::Tapped`] on the session is the host's readback of the same
//! rows, for the gates; the window does not read it.
//!
//! A partial accept is the body's own: the window calls
//! [`runtime::Verify::commit`] with the rows kept and nothing else, and the
//! session's commit takes the rest back through the body's
//! [`Rollback::rollback_on`], which settles whatever recurrent state the body
//! holds for the rows it drops. No rewind lives in the window, and no call
//! here moves the target's position.

use std::fmt;

use bloomery_gpu::model::{Rollback, Rows, StepMode};
use bloomery_gpu::{GpuError, GpuModel};

use crate::{Keep, Prompt};

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
    /// last ending at the prompt's end. A sink's error ends the call with it. A prompt the body refuses (past
    /// the context, an id it does not take) is refused before any unit runs.
    /// `None` is the body's plain prompt call.
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
        mode: StepMode,
    ) -> Result<(), GpuError>;

    /// One window's chain, one readback: `refresh` walked (the rows the
    /// target kept with their hidden rows, its last row the target's own next
    /// token), then `own` walks, each reading the walk before it on the card;
    /// the proposal, `1 + own` ids with the refresh's last row's prediction
    /// first, written to the front of `out`, and how many. `out` shorter than
    /// the proposal is refused by name. A fault any walk raised is the
    /// chain's error and poisons the model as the target's would.
    fn chain(
        m: &mut GpuModel<Self>,
        refresh: Feed<'_, Self::Arena>,
        own: usize,
        head: Self::Head,
        mode: StepMode,
        out: &mut [u32],
    ) -> Result<usize, GpuError>;
}
