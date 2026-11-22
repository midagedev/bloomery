//! The runtime's [`Target`] over an Inkling 2 session: a prompt runs as steps
//! (the family has no batched pass yet), a step reads back the pick or the
//! logits, and a cut keeps the whole sequence or nothing.

use runtime::{Out, Target, Want};

use super::session::{Session, SessionError, refuse};

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
        let Some((&last, before)) = ids.split_last() else {
            return Err(refuse("a prompt of no ids"));
        };
        for &id in before {
            Session::step(self, id)?;
        }
        Target::step(self, last, want)
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
