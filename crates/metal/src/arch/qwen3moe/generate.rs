//! The session's greedy generation: the one loop over the session as the
//! generation loop's target ([`runtime::generate`] over [`Plain`]), stopped
//! by the token count or an end-of-generation id, each picked id handed to a
//! caller's sink as it is picked — the loop itself is
//! [`crate::session::generate`]'s, generic over the target; this module adds
//! the drafts that run on this session.
//!
//! [`greedy_lookup`] is the loop over the lookup draft
//! ([`runtime::Speculative`] of [`runtime::Lookup`] behind the width chooser
//! [`runtime::width::Choosing`], two rows a verify), and [`greedy_mtp`] over
//! the MTP draft the session opened ([`Session::mtp_draft`]): the ids they
//! pick are [`greedy`]'s, and the passes they ran, the rows each kept, the wall
//! of each and the chooser's account of them are in the result for a caller
//! that prints them.

use runtime::width::{Chosen, Mode, Wall};
use runtime::{Choosing, Draft, Lookup, Speculative};

use super::session::{Session, SessionError, StepMode};

pub use crate::session::generate::{
    Counted, GenError, Generated, PassTally, WidthAccount, greedy, greedy_with,
};

/// The session's refusal is the run's, as the generic loop's error over this
/// session's own.
impl From<SessionError> for GenError<SessionError> {
    fn from(e: SessionError) -> Self {
        GenError::Session(e)
    }
}

/// [`greedy`] over the lookup draft behind the width chooser
/// ([`runtime::width::Choosing`] by [`Lookup::RULE`], `width` its mode): each
/// pass proposes the one id the context's n-grams name, and the chooser
/// verifies it, with the token at the position, in one pass of two rows or
/// runs one step instead, by the pass walls it reads off the host clock — a
/// pass whose verify costs more than the proposal's kept rows repay is not
/// run. The rows a verify keeps are the ones the session's own argmax agrees
/// with (`runtime::Speculative`); a pass with no proposal is one step. The ids
/// are [`greedy`]'s, bit for bit, whatever the mode, because a verify row is
/// the bits of the step at its position and the chooser changes which rows a
/// pass runs, never a token. [`Generated::width`] holds the chooser's account.
///
/// # Errors
/// As [`greedy`], the session's refusal by name of a verify over a file
/// whose layers the pass does not serve ([`Session::verify_serves`]), and the
/// chooser's: [`GenError::Width`], [`GenError::Counts`].
pub fn greedy_lookup<F>(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    width: Mode,
    on_id: F,
) -> Result<Generated, GenError<SessionError>>
where
    F: FnMut(u32),
{
    let draft =
        <Lookup as Chosen<Session>>::choosing_by(Lookup::new(), width, Lookup::RULE, Wall::new())?;
    greedy_drafted(s, draft, prompt, budget, eog, on_id)
}

/// [`greedy_lookup`] over the MTP draft the session opened ([`Session::open_mtp`])
/// behind the same chooser, by the chooser's default rule: each pass proposes
/// the one id the draft layer predicts from the target's last hidden row, and
/// the chooser verifies it in one pass of two rows or runs one step. The draft
/// is `runtime::mtp::MtpDraft` with the session as its host, the window policy
/// and chooser the CUDA drafts run; the ids are [`greedy`]'s, bit for bit,
/// whatever the draft proposes. A session whose layer reads the target's hidden
/// row as `pre` or `post` sets it first ([`Session::set_mtp_reading`]).
///
/// # Errors
/// As [`greedy_lookup`], and the session's refusal by name when no sidecar is
/// open.
pub fn greedy_mtp<F>(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    width: Mode,
    on_id: F,
) -> Result<Generated, GenError<SessionError>>
where
    F: FnMut(u32),
{
    let draft = s.mtp_draft()?.choosing(width)?;
    greedy_drafted(s, draft, prompt, budget, eog, on_id)
}

/// The loop of a draft behind the chooser, two rows a verify: the ids are
/// [`greedy`]'s, the chooser's account is in the result.
fn greedy_drafted<D, F>(
    s: &mut Session,
    draft: Choosing<D>,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError<SessionError>>
where
    D: Draft<Session>,
    F: FnMut(u32),
{
    s.verify_serves()?;
    // A pass with no proposal is one step, and the next pass may be a verify,
    // which drops the token a chained step ran ahead: its device time is a
    // whole step, where a host-fed step costs the wake and the encode the
    // chain hides. The chooser reads those host-fed walls as the plain step's
    // cost. A draft that reads the step's residual needs no token ahead of it.
    let chained = s.argmax_mode();
    s.set_argmax_mode(StepMode::HostFed)?;
    let mut spec: Speculative<Choosing<D>, 2> = Speculative::new(draft);
    let out = greedy_with(s, &mut spec, prompt, budget, eog, on_id);
    s.set_argmax_mode(chained)?;
    let mut out = out?;
    out.width = Some(WidthAccount::read(spec.draft_mut(), &out.commits)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::Committed;

    fn pass(kept: usize, rows: usize, proposed: bool) -> Committed {
        Committed {
            pos: 0,
            kept,
            rows,
            proposed,
        }
    }

    fn lookup_chooser(mode: Mode) -> Choosing<Lookup> {
        <Lookup as Chosen<Session>>::choosing_by(Lookup::new(), mode, Lookup::RULE, Wall::new())
            .expect("the lookup's own rule is one the chooser runs by")
    }

    /// The account of a cost chooser is of the loop's own passes: a chooser
    /// that counted other passes than the loop ran is refused by name, one
    /// that counted a run's none over a run of none is the empty account, and
    /// a fixed chooser, which counts nothing, has no counts to compare.
    ///
    /// Mutant: the passes compared against nothing, or the fixed mode's zero
    /// count read as a cost chooser's.
    #[test]
    fn the_width_account_is_the_loops_own_passes_or_refused() {
        let verify = pass(2, 2, true);
        let refused =
            WidthAccount::read::<_, _, SessionError>(&mut lookup_chooser(Mode::Cost), &[verify]);
        assert!(
            matches!(refused, Err(GenError::Counts(_))),
            "a chooser that counted no pass beside a loop that ran one: {refused:?}"
        );
        let empty = WidthAccount::read::<_, _, SessionError>(&mut lookup_chooser(Mode::Cost), &[])
            .expect("no pass counted and none run agree");
        let counted = empty.counted.expect("a cost chooser counts");
        assert_eq!((empty.mode, counted.tally.passes()), (Mode::Cost, 0));
        let fixed =
            WidthAccount::read::<_, _, SessionError>(&mut lookup_chooser(Mode::Fixed), &[verify])
                .expect("a fixed chooser counts nothing to compare");
        assert_eq!((fixed.mode, fixed.counted), (Mode::Fixed, None));
    }
}
