//! The session as the generation loop's target: [`runtime::Target`] over
//! [`Session`](super::session::Session), the seam a loop, a CLI or a serve
//! seat drives.
//!
//! The impl sits on `Session` itself, here and not in `session.rs`: stepping
//! keeps its one owner (`session.rs`, every position moving through
//! [`Session::step`](super::session::Session::step)), and no wrapper type
//! stands between a caller and the session it drives.
//!
//! The prompt call's schedule — the ids cut at the checkpoint marks, each run
//! between two marks a pass or the ids' steps, a caller's marks taken as
//! points inside a pass — is [`crate::session::prompt`]'s, generic over the
//! [`Family`] seam this module implements for the session; this file owns the
//! seam itself and the impls.
//!
//! What the target answers:
//!
//! - `pos` and `ctx`, the session's; a cache of more rows than a `u32`
//!   position names is refused at open, not truncated.
//! - `step`, one position. The mode comes from the read-back asked:
//!   [`Want::Argmax`] chains — a greedy loop feeds each pick as the next id,
//!   which is the id a token ahead embeds, so the host's wait overlaps the
//!   card — while [`Want::Logits`] steps host-fed: a token ahead would
//!   overwrite the head's row while the caller holds it, so none is
//!   committed. The row is the head's shared pages themselves, zero copy,
//!   and the argmax is the row's own by the head's tie rule (the smaller id
//!   at an equal value).
//! - `prompt`, `prompt_marked` and `verify`/`commit` as the prompt and verify
//!   machinery run them (the doc links above).
//!
//! What it refuses:
//!
//! - a prompt of no ids, one that would pass the cache's last row, and
//!   marks out of order or outside the call, before any step runs;
//! - a cut where no checkpoint stands between 0 and the position (the
//!   ledger's refusal, by name); a cut to the position does nothing and a
//!   cut to 0 resets;
//! - every mutating call of a session a fault poisoned, until `reset`;
//! - a step, a prompt, a pass, a second verify, a cut and a checkpoint while a
//!   verify waits for its commit, a commit of none or of more than its rows,
//!   and a verify of a file whose layers the pass does not serve
//!   ([`Session::verify_serves`]).

use runtime::prompt::PrefillMode;
use runtime::seqstate::Kept;
use runtime::{Out, Target, Verify, Want};

use super::lanes::VERIFY_ROWS;
use super::session::{
    HiddenSrc, MtpArena, PassOut, Passed, Session, SessionError, StateOut, StepMode, Stepped,
};
use crate::session::prompt::{Family, prompt_call};

const WHAT: &str = "qwen35moe target";

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// The step arena's one row: where the last step of a run left its position's
/// final hidden row.
const STEP_ROW: HiddenSrc = HiddenSrc {
    arena: MtpArena::Step,
    row: 0,
};

/// The qwen35moe session as the prompt machinery's family: every call the
/// schedule makes on it, at the session's own owners.
impl Family for Session {
    type Error = SessionError;
    type Arena = MtpArena;
    type Row = HiddenSrc;

    const STEP_ARENA: MtpArena = MtpArena::Step;
    const ROWS_ARENA: MtpArena = MtpArena::Rows;

    fn step_row() -> HiddenSrc {
        STEP_ROW
    }

    fn rows_row(row: usize) -> HiddenSrc {
        HiddenSrc {
            arena: MtpArena::Rows,
            row,
        }
    }

    fn pos(&self) -> u32 {
        Session::pos(self)
    }

    fn ctx(&self) -> usize {
        Session::ctx(self)
    }

    fn prefill(&self) -> PrefillMode {
        Session::prefill(self)
    }

    fn pass_rows(&self) -> usize {
        Session::pass_rows(self)
    }

    fn copies_in_pass(&self) -> bool {
        Session::copies_in_pass(self)
    }

    fn checkpoint_marks(&self, from: u32, to: u32) -> Vec<u32> {
        Session::checkpoint_marks(self, from, to)
    }

    fn checkpoint(&mut self) -> Result<(), SessionError> {
        Session::checkpoint(self)
    }

    fn checkpoint_hidden(&mut self, hidden: HiddenSrc) -> Result<(), SessionError> {
        Session::checkpoint_hidden(self, hidden)
    }

    fn checkpoint_copy_hidden(&mut self, at: u32, hidden: HiddenSrc) -> Result<(), SessionError> {
        Session::checkpoint_copy_hidden(self, at, hidden)
    }

    fn step(&mut self, id: u32, mode: StepMode) -> Result<Stepped, SessionError> {
        Session::step(self, id, mode)
    }

    fn pass(
        &mut self,
        ids: &[u32],
        out: PassOut,
        snaps: StateOut<'_>,
    ) -> Result<Passed, SessionError> {
        Session::pass(self, ids, out, snaps)
    }

    fn refuse(detail: String) -> SessionError {
        refuse(detail)
    }
}

impl Target for Session {
    type Error = SessionError;

    fn pos(&self) -> u32 {
        Session::pos(self)
    }

    fn ctx(&self) -> u32 {
        // Open refused a cache of more rows than a u32 position names.
        u32::try_from(Session::ctx(self)).expect("open refused a cache a u32 position cannot name")
    }

    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, SessionError> {
        Target::prompt_marked(self, ids, &[], want)
    }

    fn prompt_marked(
        &mut self,
        ids: &[u32],
        marks: &[u32],
        want: Want,
    ) -> Result<Out<'_>, SessionError> {
        let pick = prompt_call(self, ids, marks, None)?;
        read(self, pick, want)
    }

    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, SessionError> {
        // The mode by the read-back asked: the pick a greedy loop feeds back
        // is the id a chained token embeds ahead, unless the caller chose
        // host-fed picks ([`Session::set_argmax_mode`]); a Logits call's row
        // must be this token's, and a token ahead would overwrite it.
        let mode = match want {
            Want::Argmax => Session::argmax_mode(self),
            Want::Logits => StepMode::HostFed,
        };
        let pick = Session::step(self, id, mode)?.id;
        read(self, pick, want)
    }

    fn keepable(&self, n: u32) -> u32 {
        Target::kept(self, n).at
    }

    fn kept(&self, n: u32) -> Kept {
        Session::kept(self, n)
    }

    fn cut(&mut self, n: u32) -> Result<(), SessionError> {
        Session::cut(self, n)
    }

    fn reset(&mut self) -> Result<(), SessionError> {
        Session::reset(self)
    }
}

impl Verify for Session {
    const MAX_ROWS: usize = VERIFY_ROWS;

    fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], SessionError> {
        const {
            assert!(
                M >= 2 && M <= VERIFY_ROWS,
                "a verify of the Metal session takes 2 to VERIFY_ROWS rows"
            );
        }
        let mut picks = [0u32; M];
        Session::verify_rows(self, &rows, &mut picks)?;
        Ok(picks)
    }

    fn commit(&mut self, accepted: usize) -> Result<(), SessionError> {
        Session::commit_rows(self, accepted)
    }
}

/// `pick` as `want` asks: the pick alone, or beside the head's row of the
/// token just stepped, whose pick `pick` is. [`Want::Logits`] runs after a
/// host-fed step, which settled its token and committed nothing ahead; the
/// row is the head's shared pages, zero copy
/// ([`Session::logits_row`]'s proof).
fn read<'a>(s: &'a Session, pick: u32, want: Want) -> Result<Out<'a>, SessionError> {
    match want {
        Want::Argmax => Ok(Out::Argmax(pick)),
        Want::Logits => Ok(Out::Logits {
            argmax: pick,
            row: s.logits_row()?,
        }),
    }
}
