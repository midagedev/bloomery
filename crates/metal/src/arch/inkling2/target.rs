//! The runtime's [`Target`] over an Inkling 2 session: a prompt runs through
//! the common prompt call ([`prompt_call`]) as passes of rows, a batch of a
//! few ids as steps, a step reads back the pick or the logits, and a cut keeps
//! the whole sequence or nothing.
//!
//! The family keeps no checkpoint and no hidden row: its conv rings hold only
//! the last inputs, so the call has no mark to take, and its arenas and rows
//! name nothing.

use runtime::prompt::PrefillMode;
use runtime::{Out, Target, Want};

use super::session::{Session, SessionError, refuse};
use crate::session::clock::{StepMode, Stepped};
use crate::session::pass::{PassOut, Passed, StateOut};
use crate::session::prompt::{Family, prompt_call};

/// The read-back `want` asks for after a step that picked `pick`.
fn read(s: &mut Session, pick: u32, want: Want) -> Result<Out<'_>, SessionError> {
    match want {
        Want::Argmax => Ok(Out::Argmax(pick)),
        Want::Logits => Ok(Out::Logits {
            argmax: pick,
            row: s.logits_row()?,
        }),
    }
}

/// The Inkling 2 session as the prompt machinery's family.
impl Family for Session {
    type Error = SessionError;
    type Arena = ();
    type Row = ();

    const STEP_ARENA: () = ();
    const ROWS_ARENA: () = ();

    fn step_row() {}

    fn rows_row(_row: usize) {}

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
        false
    }

    fn checkpoint_marks(&self, _from: u32, _to: u32) -> Vec<u32> {
        Vec::new()
    }

    fn checkpoint(&mut self) -> Result<(), SessionError> {
        Ok(())
    }

    fn checkpoint_hidden(&mut self, _hidden: ()) -> Result<(), SessionError> {
        Ok(())
    }

    fn checkpoint_copy_hidden(&mut self, at: u32, _hidden: ()) -> Result<(), SessionError> {
        Err(refuse(format!(
            "a checkpoint at {at} inside a pass: the family keeps none"
        )))
    }

    /// Every step is host-fed: the family runs no token ahead, so an ahead
    /// mode's next id is fed by the call after. The count is what the
    /// token's command buffers encoded, summed.
    fn step(&mut self, id: u32, _mode: StepMode) -> Result<Stepped, SessionError> {
        Session::step_counted(self, id)
    }

    fn pass(
        &mut self,
        ids: &[u32],
        out: PassOut,
        snaps: StateOut<'_>,
    ) -> Result<Passed, SessionError> {
        if !snaps.after_rows.is_empty() {
            return Err(refuse(format!(
                "copies after rows {:?} of a pass: the family keeps none",
                snaps.after_rows
            )));
        }
        Session::pass(self, ids, out)
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
        // The check refused a cache of more rows than a u32 position names.
        u32::try_from(Session::ctx(self)).expect("the check refused a cache a u32 cannot name")
    }

    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, SessionError> {
        let pick = prompt_call(self, ids, &[], None)?;
        read(self, pick, want)
    }

    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, SessionError> {
        let pick = Session::step(self, id)?;
        read(self, pick, want)
    }

    fn keepable(&self, n: u32) -> u32 {
        Session::keepable(self, n)
    }

    fn cut(&mut self, n: u32) -> Result<(), SessionError> {
        Session::cut(self, n)
    }

    fn reset(&mut self) -> Result<(), SessionError> {
        Session::reset(self)
    }
}
