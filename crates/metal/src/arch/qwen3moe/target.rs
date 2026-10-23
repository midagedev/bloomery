//! The session as the generation loop's target: [`runtime::Target`] over
//! [`Session`](super::session::Session), the seam a loop, a CLI or a serve
//! seat drives.
//!
//! The impl sits on `Session` itself, here and not in `session.rs`: stepping
//! keeps its one owner (`session.rs`, every position moving through
//! [`Session::step`](super::session::Session::step)), and no wrapper type
//! stands between a caller and the session it drives.
//!
//! What it answers:
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
//! - `prompt`, the ids one host-fed step each — the session has no batched
//!   prompt schedule — and a checkpoint at each position
//!   [`runtime::seqstate::marks`] names: the call's start when the session
//!   already holds one, every
//!   [`CHECKPOINT_EVERY`](super::session::CHECKPOINT_EVERY) positions inside,
//!   and the call's end. The row is read back after the last as asked.
//!
//! What it refuses:
//!
//! - a prompt of no ids, and one that would pass the cache's last row,
//!   before any step runs;
//! - a cut where no checkpoint stands between 0 and the position (the
//!   ledger's refusal, by name); a cut to the position does nothing and a
//!   cut to 0 resets;
//! - every mutating call of a session a fault poisoned, until `reset`.

use runtime::seqstate::Kept;
use runtime::{Out, Target, Want};

use super::session::{Session, SessionError, StepMode};

const WHAT: &str = "qwen35moe target";

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
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
        if ids.is_empty() {
            return Err(refuse(format!(
                "a prompt of no ids at position {}",
                Session::pos(self)
            )));
        }
        let end = u64::from(Session::pos(self)) + ids.len() as u64;
        if end > u64::from(Target::ctx(self)) {
            return Err(refuse(format!(
                "a prompt of {} ids at position {}: the cache holds {} rows",
                ids.len(),
                Session::pos(self),
                Session::ctx(self)
            )));
        }
        let from = Session::pos(self);
        let to = u32::try_from(end).map_err(|_| {
            refuse(format!(
                "a prompt of {} ids at position {from} passes the u32 positions",
                ids.len()
            ))
        })?;
        // The id at an inner mark is stepped by the next span, after the take.
        let marks = self.checkpoint_marks(from, to);
        let mut at = from;
        let mut last_pick = None;
        for m in marks {
            let start = (at - from) as usize;
            let end_i = (m - from) as usize;
            for &id in &ids[start..end_i] {
                last_pick = Some(Session::step(self, id, StepMode::HostFed)?.id);
            }
            Session::checkpoint(self)?;
            at = m;
        }
        let pick = last_pick.ok_or_else(|| {
            refuse(format!(
                "a prompt of {} ids at position {from} stepped no id",
                ids.len()
            ))
        })?;
        read(self, pick, want)
    }

    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, SessionError> {
        // The mode by the read-back asked: the pick a greedy loop feeds back
        // is the id a chained token embeds ahead; a Logits call's row must
        // be this token's, and a token ahead would overwrite it.
        let mode = match want {
            Want::Argmax => StepMode::Chain,
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
