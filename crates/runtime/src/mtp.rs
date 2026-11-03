//! The MTP window's card-free part: the plain values a draft walk is given
//! ([`WalkMode`], [`Hidden`]), what a draft holds between two calls
//! ([`Parked`], [`Join`]) and the window's host policy over them ([`Policy`]),
//! which decides every call from positions, the context, token ids and arena
//! kinds alone. The card that walks the draft layer is the executor
//! ([`Exec`]) a call runs its walks on: `bloomery_app`'s `MtpDraft` today.

mod policy;

pub use policy::{Call, Chain, Exec, Fail, Policy, Refresh, Refused, SKIPS, Shape, Walk};

/// The window's name in its refusals.
pub const WHAT: &str = "MTP window";

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

/// What a draft holds of the target's sequence between two calls, kept
/// apart while another sequence runs (`MtpDraft::park` in `bloomery_app`) and
/// put back with it (`MtpDraft::unpark`): the rows waiting for its next walk,
/// or why it proposes nothing.
#[derive(Clone, Debug)]
pub struct Parked<A> {
    pub next: Option<Refresh<A>>,
    pub last_unit: Option<(A, usize)>,
    pub skip: Option<&'static str>,
}

/// How a prompt call that continues a held sequence joined the draft to it
/// (`MtpDraft::take_joined` in `bloomery_app`).
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
