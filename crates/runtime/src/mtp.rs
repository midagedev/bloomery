//! The MTP window: [`MtpDraft`], the runtime rule's [`crate::Draft`] over a
//! [`Host`]'s MTP layer, and the card-free values and policy under it. A host
//! is a target ([`crate::Verify`]) that also runs the draft layer beside it:
//! the GPU session in `bloomery_app`, the Metal session, a model in a test.
//!
//! An MTP layer's row at position `q` reads the token at `q` and the
//! target's final hidden row at `q − 1`, and predicts the token at `q + 1`.
//! The window drives it under the runtime's rule ([`crate::Speculative`]
//! over a [`crate::Draft`] on the host): a proposal of up to [`Host::WIDTH`]
//! ids, one verify of up to [`Host::WIDTH`] + 1 rows, and the commit of the
//! rows the target agrees with. Between those:
//!
//! - the prompt call ([`MtpDraft`'s `prompt`](crate::Draft::prompt)) walks
//!   the draft over the prompt as its units complete — a unit's arena holds
//!   its rows' hidden rows until the next unit overwrites them — through the
//!   store's append alone ([`WalkMode::Store`]), filling the draft's store
//!   from position 0 (the row at 0 reads a zero hidden row: the target holds
//!   nothing before it) and leaving the last unit's last row for
//!   [`MtpDraft`'s `begin`](crate::Draft::begin), whose walk is the next
//!   window's anchor. A host whose target call carries rows of its own at
//!   some positions (an image's) walks the same way over that call, through
//!   [`MtpDraft::prompt_joined`] on an executor of its own;
//! - each window's [`MtpDraft`'s `propose`](crate::Draft::propose) runs one
//!   chain ([`Host::chain`], one readback): the refresh — the rows the target
//!   kept, at the positions after the verify's first, each with the hidden
//!   row the verify's row before it wrote, the last row the target's own
//!   next token (the anchor, whose prediction is the first proposal) — then
//!   own walks, each reading the walk before it;
//! - [`MtpDraft`'s `accept`](crate::Draft::accept) records the refresh before
//!   the commit takes the rejected rows back, the verify's arena
//!   ([`Host::VERIFY_ARENA`]) still holding every row's hidden row.
//!
//! A prompt call that continues a held sequence (a request that extends the
//! one before it) first walks the rows the last call left waiting — the
//! refresh a window or a step recorded, or a prompt call's anchor — their
//! last row the call's first id, while their arena still holds their hidden
//! rows; a first step that feeds the next id does the same before the step
//! overwrites the step's arena ([`MtpDraft::before_step`]), and so does the
//! width chooser before each pass it holds back
//! ([`crate::Draft::before_plain`]). Those walks run through the store's
//! append alone ([`WalkMode::Store`]): no head reads them, the next chain
//! walking its own refresh. When no such rows wait, or the store does not
//! hold the positions below them, the draft skips: it proposes nothing, every
//! pass a plain step, until a prompt call from position 0, a restart or a
//! rejoin ([`MtpDraft::rejoin`]); the call's [`Join`] names why.
//!
//! Every decision of those calls — which rows wait, which walks a call runs
//! and in which mode, the join checks, the refusals, what a park keeps — is
//! the window's policy ([`Policy`]), over positions, the context, token ids
//! and arena kinds alone; [`MtpDraft`] runs its walks on the host, and the
//! policy's unit tests run the same calls against a model of the body's
//! records with no card.
//!
//! A server that switches the target between sequences parks the draft's
//! waiting rows with the sequence's state ([`MtpDraft::park`]) and puts them
//! back with it ([`MtpDraft::unpark`]): the returning sequence's next call
//! joins it as the call after no switch would.
//!
//! The proposal changes which passes run, never a token: every kept token is
//! the target's own argmax, or the request's draw from that target row. The
//! hidden rows never leave the host: a walk names them by the arena a target
//! call left them in and the row they start at ([`Hidden::Target`]).
//!
//! A partial accept is the host's own: the window calls
//! [`crate::Verify::commit`] with the rows kept and nothing else, and the
//! host's commit settles whatever recurrent state it holds for the rows it
//! drops. No rewind lives in the window, and no call here moves the target's
//! position: the draft's walks write only the draft's store and arena, and
//! the target's position moves by the rule's verify and commit alone.

mod draft;
mod host;
mod points;
mod policy;

pub use draft::{MtpDraft, WindowDraft};
pub use host::{Host, HostSink, Unit};
pub use points::PointRows;
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
/// apart while another sequence runs ([`MtpDraft::park`]) and put back with it
/// ([`MtpDraft::unpark`]): the rows waiting for its next walk, or why it
/// proposes nothing.
#[derive(Clone, Debug)]
pub struct Parked<A> {
    pub next: Option<Refresh<A>>,
    pub last_unit: Option<(A, usize)>,
    pub skip: Option<&'static str>,
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
