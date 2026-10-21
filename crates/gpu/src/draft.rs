//! The types a draft walk is given: how it runs ([`WalkMode`]), which
//! projection its head runs ([`DraftHead`]), where its hidden rows come from
//! ([`Hidden`]), the rows it walks ([`Feed`], [`DraftFeed`]) and the names of
//! its row bounds ([`WalkBounds`]). Each draft program — Qwen3.8's MTP layer
//! and GLM's NextN layer — walks these.

use crate::GpuError;
use crate::media::CallRows;
use crate::model::StepMode;

/// How a walk runs: the program's own walk, or its chain of walks.
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
    /// A step's mode as a walk's: the draft's walks with a head run as the
    /// target's steps do.
    fn from(m: StepMode) -> WalkMode {
        match m {
            StepMode::Eager => WalkMode::Eager,
            StepMode::Graph => WalkMode::Graph,
        }
    }
}

/// Which projection a draft's head runs. Whether the load opened a row list
/// is the load's fact; a program that opens none holds [`DraftHead::Full`]
/// alone and refuses the other by name ([`DraftHead::check_opened`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftHead {
    /// The target's `output`: every token of the vocabulary.
    Full,
    /// The load's row list: the target's `output` rows of its ids.
    Rows,
}

impl DraftHead {
    /// Refused by name: the row-list head on a load that opened the full head
    /// alone (`opened_rows` false).
    ///
    /// # Errors
    /// [`GpuError::State`] naming `what` for [`DraftHead::Rows`] when the load
    /// opened no row list.
    pub fn check_opened(self, what: &'static str, opened_rows: bool) -> Result<(), GpuError> {
        if self == DraftHead::Rows && !opened_rows {
            return Err(GpuError::state(
                what,
                "a row-list head: this draft was opened with the full head",
            ));
        }
        Ok(())
    }
}

/// Where a walk's hidden rows come from; `A` is the arena kind the caller's
/// body names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hidden<'a, A> {
    /// Rows written from the host, as many values a row as the caller's body
    /// says: the row at position 0 reads a zero row, the target holding
    /// nothing before it.
    Host(&'a [f32]),
    /// The target's final hidden rows in the arena `walk`, from its row
    /// `first` on, as the target's last call of that arena left them.
    Target { walk: A, first: usize },
}

/// A walk's rows: `tokens` at positions `pos0 ..`, each beside its hidden
/// row (the target's at the position before it). `media` is the prompt
/// call's span rows by absolute position: a row the lookup holds replaces
/// the token's embedding row at its position, as the target's media call
/// took it. Only a prompt call that holds an image carries it; a body whose
/// walk cannot take it refuses it by name.
#[derive(Clone, Copy, Debug)]
pub struct Feed<'a, A> {
    pub tokens: &'a [u32],
    pub pos0: u32,
    pub hidden: Hidden<'a, A>,
    pub media: Option<&'a CallRows<'a>>,
}

/// What a draft program is fed one walk: the rows of a [`Feed`], or the
/// draft's own last row.
#[derive(Clone, Copy, Debug)]
pub enum DraftFeed<'a, A> {
    /// `tokens` at positions `pos0 ..`, each beside its hidden row.
    Rows(Feed<'a, A>),
    /// One row at `pos0`: the last walk's last row's token and streams. A
    /// program whose proposal holds one id has no such walk and refuses it by
    /// name ([`DraftFeed::target_rows`]).
    Own { pos0: u32 },
}

impl<'a, A> DraftFeed<'a, A> {
    /// The position of the feed's first row.
    #[must_use]
    pub fn pos0(&self) -> u32 {
        match self {
            DraftFeed::Rows(f) => f.pos0,
            DraftFeed::Own { pos0 } => *pos0,
        }
    }

    /// How many rows the feed walks: its tokens, or the one own row.
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            DraftFeed::Rows(f) => f.tokens.len(),
            DraftFeed::Own { .. } => 1,
        }
    }

    /// The feed's rows, for a program that walks the target's rows alone: the
    /// own row is refused by name ([`refuse_own`]).
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming `what` for [`DraftFeed::Own`].
    pub fn target_rows(self, what: &'static str) -> Result<Feed<'a, A>, GpuError> {
        match self {
            DraftFeed::Rows(f) => Ok(f),
            DraftFeed::Own { pos0 } => Err(refuse_own(
                what,
                &format!("a walk of the draft's own row at position {pos0}"),
            )),
        }
    }
}

/// The refusal by name of the draft's own row ([`DraftFeed::Own`]) by a
/// program whose walks read the target's hidden rows alone, `detail` naming
/// the call.
#[must_use]
pub fn refuse_own(what: &'static str, detail: &str) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "{detail}: this program's walk reads the target's hidden rows and its proposal holds \
             one id, so it has no own row (DraftFeed::Own)"
        ),
    )
}

/// The names of a draft program's row bounds: the most rows one walk takes
/// in each mode. A program implements it with its own values; the names, and
/// the refusal of a walk past them ([`refuse_rows`]), are this module's.
pub trait WalkBounds {
    /// The most rows one eager walk takes.
    const WALK_ROWS: usize;
    /// The most rows a captured walk takes; 0 for a program that captures
    /// none.
    const GRAPH_ROWS: usize;
    /// The most rows one store walk ([`WalkMode::Store`]) takes.
    const STORE_ROWS: usize;
}

/// The most rows one walk of `P` takes in `mode`.
#[must_use]
pub const fn most_rows<P: WalkBounds>(mode: WalkMode) -> usize {
    match mode {
        WalkMode::Eager => P::WALK_ROWS,
        WalkMode::Graph => P::GRAPH_ROWS,
        WalkMode::Store => P::STORE_ROWS,
    }
}

/// Refused by name: a walk of `m` rows outside `1..=` [`most_rows`] of `P`'s
/// `mode`.
///
/// # Errors
/// [`GpuError::Shape`] naming `what`.
pub fn refuse_rows<P: WalkBounds>(
    what: &'static str,
    mode: WalkMode,
    m: usize,
) -> Result<(), GpuError> {
    let most = most_rows::<P>(mode);
    if !(1..=most).contains(&m) {
        return Err(GpuError::shape(
            what,
            format!("a walk of {m} rows; the {mode:?} walk takes 1..={most}"),
        ));
    }
    Ok(())
}
