//! The target an MTP window drafts on: [`Host`], the calls [`super::MtpDraft`]
//! makes of it, and [`On`], the host's walks as the executor ([`Exec`]) the
//! window's policy runs them on.

use std::fmt;

use super::{Chain, Exec, Walk};
use crate::Verify;
use crate::prompt::CallEnd;

/// A prompt call's tap: called after each unit the call runs, with the
/// executor of that unit's walks (`E` its error), the arena the unit's rows'
/// final hidden rows sit in, the unit's first position and its rows.
pub type Unit<'a, A, E> = dyn FnMut(&mut dyn Exec<A, Err = E>, A, u32, usize) -> Result<(), E> + 'a;

/// [`Unit`] of a [`Host`]'s prompt call ([`Host::prompt_with`]).
pub type HostSink<'a, H> = Unit<'a, <H as Host>::Arena, <H as Host>::Cause>;

/// A target the MTP window drafts on: a [`Verify`] target that also runs the
/// draft layer beside it. A draft layer's row at position `q` reads the token
/// at `q` and the target's final hidden row at `q − 1`; the hidden rows never
/// leave the host, a walk names them by the arena a target call left them in
/// and the row they start at ([`super::Hidden`]).
pub trait Host: Verify {
    /// The most ids a proposal holds (`Draft::WIDTH`): a verify runs them
    /// behind the token at the target's position.
    const WIDTH: usize;

    /// The most rows one walk takes: a prompt unit's rows are walked in runs
    /// of at most this many, and a refresh (at most `WIDTH + 1` rows) is one
    /// walk.
    const WALK_ROWS: usize;

    /// The most rows one store walk ([`super::WalkMode::Store`], the prompt's
    /// warmup) takes: a host whose store walk runs wider than its whole walks
    /// names its width; the default is [`Host::WALK_ROWS`]. A host whose load
    /// chooses its width names the widest it can, and the draft is opened
    /// with the chosen one ([`super::MtpDraft::new`]).
    const STORE_ROWS: usize = Self::WALK_ROWS;

    /// Whether the chain's one readback holds each proposed id's probability
    /// beside it, so [`Host::chain`] takes `p` at no further device work; a
    /// host whose readback holds the ids alone refuses `p` by name, and its
    /// window reports certainty (`Draft::propose_p`'s default).
    const PROBS: bool = false;

    /// Holds when the widths fit together: at least one id a proposal, a
    /// verify the host's passes take ([`Verify::MAX_ROWS`]), a refresh one
    /// walk and one store walk. The window reads it, so a host that breaks it
    /// fails to compile where it is drafted.
    const FITS: () = assert!(
        Self::WIDTH >= 1
            && Self::WIDTH < <Self as Verify>::MAX_ROWS
            && Self::WIDTH < Self::WALK_ROWS
            && Self::WIDTH < Self::STORE_ROWS,
        "a host's WIDTH + 1 rows must fit one verify, one walk and one store walk"
    );

    /// A target arena whose final hidden rows a walk reads: where a step, a
    /// verify and each prompt unit leave their rows.
    type Arena: Copy + Eq + fmt::Debug;

    /// The arena a plain step leaves its one row in. The next step overwrites
    /// it, so rows waiting there are walked before that step or are lost, and
    /// the window refuses by name a step after rows that wait there.
    const STEP_ARENA: Self::Arena;

    /// The arena a verify leaves its rows in, row `r` the verify's row `r`.
    /// The commit leaves them in place: the next window's refresh reads the
    /// kept rows there.
    const VERIFY_ARENA: Self::Arena;

    /// How the target runs a prompt: the host's own paths (steps, passes,
    /// ubatches).
    type Path: Copy + fmt::Debug;

    /// Which head the draft's chains run, as the load opened it.
    type Head: Copy + fmt::Debug;

    /// What a draft call fails with before the target wraps it as
    /// [`Target::Error`](crate::Target::Error).
    type Cause: Into<Self::Error>;

    /// Positions of the current sequence the draft's store holds: a walk may
    /// start at or below it, never past it. The target's reset empties it.
    ///
    /// # Errors
    ///
    /// The host's.
    fn held(&self) -> Result<usize, Self::Cause>;

    /// One walk of the draft with no readback: `w`'s rows, at most
    /// [`Host::WALK_ROWS`] from a position at most [`Host::held`], into `head`.
    /// It writes only the draft's store and arena, never the target's. A fault
    /// the walk raises stays on the host's fault word, which the next
    /// readback names.
    ///
    /// # Errors
    ///
    /// The host's.
    fn walk(&mut self, w: &Walk<'_, Self::Arena>, head: Self::Head) -> Result<(), Self::Cause>;

    /// One window's chain, one readback, never in [`super::WalkMode::Store`]:
    /// the refresh walked (the rows the target kept with their hidden rows,
    /// its last row the target's own next token), then `c.own` walks, each
    /// reading the walk before it; the proposal, `1 + c.own` ids with the
    /// refresh's last row's prediction first, written to the front of `out`,
    /// and how many. With `p`, each id's probability among the head's rows,
    /// as the same readback holds it, written to the front of `p`; a host
    /// whose readback holds none refuses `p` by name. `out` or `p` shorter
    /// than the proposal is refused by name.
    ///
    /// # Errors
    ///
    /// The host's. A fault any walk raised is the chain's error.
    fn chain(
        &mut self,
        c: &Chain<Self::Arena>,
        head: Self::Head,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, Self::Cause>;

    /// The target's prompt call of `ids` from its position by `path`, ending
    /// as `end` asks: the argmax after the last id when the call runs the
    /// head, `None` for [`CallEnd::Unread`]. A host asked for
    /// [`CallEnd::Close`] runs the last id as the call's last row with the
    /// checkpoints of the call less that id, or refuses it by name before
    /// anything runs.
    ///
    /// `sink` is called after each unit the call runs, whatever the path's
    /// unit is (a step, a pass, a ubatch), with the executor of that unit's
    /// walks, the arena its rows' final hidden rows sit in, the unit's first
    /// position and its rows, before the next unit overwrites them: the
    /// window walks the draft over the unit there, filling the draft's store.
    /// The units cover the prompt in order, each from the position the one
    /// before ended at, the last ending at the prompt's end. A sink's error
    /// ends the call with it. A prompt the host refuses (past the context, an
    /// id it does not take) is refused before any unit runs. `None` is the
    /// target's plain prompt call.
    ///
    /// # Errors
    ///
    /// The host's, or the sink's.
    fn prompt_with(
        &mut self,
        ids: &[u32],
        path: Self::Path,
        end: CallEnd,
        head: Self::Head,
        sink: Option<&mut HostSink<'_, Self>>,
    ) -> Result<Option<u32>, Self::Cause>;

    /// [`Host::prompt_with`] that leaves each of `marks` keepable
    /// ([`Target::prompt_marked`](crate::Target::prompt_marked)'s contract):
    /// positions ascending, each past the target's position and at most the
    /// call's end; a mark out of order or outside the call is the caller's
    /// breach and panics by name. The default runs the call a part, each part
    /// before the last ending [`CallEnd::Unread`] and the sink hearing every
    /// part's units in order: a host whose calls' ends are its points keeps a
    /// mark by ending a part there. A host that keeps a point inside a call
    /// answers here with one call.
    ///
    /// # Errors
    ///
    /// The host's, or the sink's.
    fn prompt_marked_with(
        &mut self,
        ids: &[u32],
        marks: &[u32],
        path: Self::Path,
        end: CallEnd,
        head: Self::Head,
        mut sink: Option<&mut HostSink<'_, Self>>,
    ) -> Result<Option<u32>, Self::Cause> {
        let from = u64::from(self.pos());
        let to = from + ids.len() as u64;
        let mut at = 0;
        let mut prev = from;
        for &u in marks {
            let u = u64::from(u);
            assert!(
                prev < u && u <= to,
                "a prompt of positions {from}..{to} marked at {marks:?}: marks go ascending, \
                 past the call's start and at most its end"
            );
            prev = u;
            let i = usize::try_from(u - from).expect("a mark inside the call is an index of it");
            if i < ids.len() {
                self.prompt_with(
                    &ids[at..i],
                    path,
                    CallEnd::Unread,
                    head,
                    sink.as_deref_mut(),
                )?;
                at = i;
            }
        }
        self.prompt_with(&ids[at..], path, end, head, sink)
    }

    /// Copy a shared prompt arena's final hidden row `first`, at position
    /// `pos - 1`, to the step arena's row 0, including its record.
    ///
    /// # Errors
    ///
    /// The host's.
    fn retain_prompt_row(
        &mut self,
        walk: Self::Arena,
        first: usize,
        pos: u32,
    ) -> Result<(), Self::Cause>;

    /// Run one prompt call of the draft inside the host's own scope for it:
    /// `call` holds the join's walks and the target's prompt call together,
    /// handed the host back. A host that marks a prompt call as one (a card
    /// engine's watchdog, which a join's walk waiting on the card must not
    /// escape) holds its mark here for the whole of `call`; the default runs
    /// `call` bare.
    fn prompt_scope<R>(&mut self, call: impl FnOnce(&mut Self) -> R) -> R {
        call(self)
    }
}

/// A host's walks as the executor the window's policy runs them on, for the
/// head the draft was opened with.
pub(super) struct On<'a, H: Host> {
    pub(super) t: &'a mut H,
    pub(super) head: H::Head,
}

impl<H: Host> Exec<H::Arena> for On<'_, H> {
    type Err = H::Cause;

    fn walk(&mut self, w: &Walk<'_, H::Arena>) -> Result<(), H::Cause> {
        self.t.walk(w, self.head)
    }

    fn held(&mut self) -> Result<usize, H::Cause> {
        self.t.held()
    }
}
