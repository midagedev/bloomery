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
//! - `prompt`, the ids cut at the checkpoint marks ([`runtime::seqstate::marks`]:
//!   the call's start when the session already holds one, every
//!   [`CHECKPOINT_EVERY`](super::session::CHECKPOINT_EVERY) positions inside,
//!   and the call's end) and each run between two marks cut by
//!   [`runtime::prompt::call_batches`] at the session's pass rows. A batch of
//!   more than a step's window is one [`Session::pass`]; a shorter one steps
//!   its ids one ahead each where the next id is known and inner —
//!   [`StepMode::Ahead`], no head on a row whose pick nobody reads — the
//!   span's last row and the row before the prompt's last committing nothing
//!   ahead, the mark's take and the last row's own head being the one pick
//!   read back after the last as asked. Each mark's checkpoint waits out
//!   whatever token runs ahead of it. A session in [`PrefillMode::Steps`]
//!   steps every batch, the same-binary arm and the oracle's.
//! - `prompt_marked`, `prompt` with a point at each of the caller's marks
//!   (a server's message starts) and no cut a mark: a mark inside a pass is
//!   a copy the pass leaves after the row before it
//!   ([`StateOut::after_rows`]), taken as a point once the pass settles
//!   ([`Session::checkpoint_copy`]), so the point holds the bits the pass
//!   held there; a mark inside a run of steps ends that run, its point the
//!   step's own state (a step runs one position whatever ends the run); a
//!   mark at a batch's end is that batch's point. A pass takes at most
//!   [`SNAPS`] copies: a further mark ends the pass there. A mark out of
//!   order or outside the call is refused by name before any step runs.
//!
//! - `verify` and `commit`, [`runtime::Verify`]: `M` ids of 2 to
//!   [`VERIFY_ROWS`] as one pass whose row `r` is the step at the position
//!   after `r` rows (`session/verify.rs`), and the commit that keeps the first
//!   rows of it by moving the position: no launch, no copy. The session waits
//!   for the commit; the token a chain step ran ahead is dropped when the
//!   verify starts.
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

use runtime::prompt::{PrefillMode, call_batches};
use runtime::seqstate::Kept;
use runtime::{Out, Target, Verify, Want};

use super::lanes::VERIFY_ROWS;
use super::session::{HiddenSrc, MtpArena, PassOut, Session, SessionError, StateOut, StepMode};
use crate::kernels::{PASS_ROWS, SNAPS};

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

/// What a prompt call tells after each unit it ran: the arena the unit's final
/// hidden rows sit in, the unit's first position and its rows.
pub(super) type Unit<'a> =
    dyn FnMut(&mut Session, MtpArena, u32, usize) -> Result<(), SessionError> + 'a;

/// The prompt call of [`Target::prompt_marked`] with no mark of the caller's
/// beyond `marks`, the pick after its last id returned; `unit`, when given, is
/// called after each unit the call ran, before the next unit overwrites the
/// unit's rows. A call with a unit steps every id that is not its last with no
/// token ahead (the step arena's residual is that token's), where a call with
/// none steps them one ahead each; the programs of a call with none are the
/// call's alone.
pub(super) fn prompt_call(
    s: &mut Session,
    ids: &[u32],
    marks: &[u32],
    mut unit: Option<&mut Unit<'_>>,
) -> Result<u32, SessionError> {
    if ids.is_empty() {
        return Err(refuse(format!(
            "a prompt of no ids at position {}",
            Session::pos(s)
        )));
    }
    let end = u64::from(Session::pos(s)) + ids.len() as u64;
    if end > u64::from(Target::ctx(s)) {
        return Err(refuse(format!(
            "a prompt of {} ids at position {}: the cache holds {} rows",
            ids.len(),
            Session::pos(s),
            Session::ctx(s)
        )));
    }
    let from = Session::pos(s);
    let to = u32::try_from(end).map_err(|_| {
        refuse(format!(
            "a prompt of {} ids at position {from} passes the u32 positions",
            ids.len()
        ))
    })?;
    let inside = marks.first().is_none_or(|&u| u > from)
        && marks.last().is_none_or(|&u| u <= to)
        && marks.windows(2).all(|w| w[0] < w[1]);
    if !inside {
        return Err(refuse(format!(
            "a prompt of positions {from}..{to} marked at {marks:?}: marks go ascending, \
             past the call's start and at most its end"
        )));
    }
    // The id at an inner point is stepped by the next span, after the take.
    // With no recurrent store the call takes no checkpoint: its end is
    // its one point, and a mark has nothing to copy.
    let mut points = s.checkpoint_marks(from, to);
    let marks = if points.is_empty() {
        points.push(to);
        &[][..]
    } else {
        marks
    };
    let call = Call {
        ids,
        from,
        last: ids.len() - 1,
        steps: Session::prefill(s) == PrefillMode::Steps,
    };
    let batch_rows = Session::pass_rows(s);
    let mut at = from;
    let mut last_ran: Option<Ran> = None;
    for p in points {
        // The runs between two points, each cut into batches: a pass or
        // its ids' steps, with the marks inside it. A point at the start of
        // the call (the position the session already stands at) has no run
        // before it and keeps the row it was taken with, if any.
        let mut ended: Option<HiddenSrc> = None;
        for batch in call_batches(at, p, &[p], batch_rows) {
            let inner: Vec<u32> = marks
                .iter()
                .copied()
                .filter(|&u| batch.start < u && u < batch.end)
                .collect();
            let ran = call.run(s, batch.start, batch.end, &inner, unit.as_deref_mut())?;
            ended = Some(ran.hidden);
            last_ran = Some(ran);
            if batch.end != p && marks.contains(&batch.end) {
                point(s, ended)?;
            }
        }
        point(s, ended)?;
        at = p;
    }
    let pick = last_ran.map(|ran| ran.pick).ok_or_else(|| {
        refuse(format!(
            "a prompt of {} ids at position {from} stepped no id",
            ids.len()
        ))
    })?;
    Ok(pick)
}

/// The checkpoint at the session's position, beside the target's final hidden
/// row of the position before it where the run that ended there left one.
fn point(s: &mut Session, hidden: Option<HiddenSrc>) -> Result<(), SessionError> {
    match hidden {
        Some(h) => Session::checkpoint_hidden(s, h),
        None => Session::checkpoint(s),
    }
}

/// What a run of a prompt call left: the pick of its last row, and where the
/// final hidden row of its last position sits.
#[derive(Clone, Copy)]
struct Ran {
    pick: u32,
    hidden: HiddenSrc,
}

/// The step arena's one row: where the last step of a run left its position's
/// final hidden row.
const STEP_ROW: HiddenSrc = HiddenSrc {
    arena: MtpArena::Step,
    row: 0,
};

/// A prompt call's ids, the position of its first, the index of its last
/// and whether the session steps every id.
struct Call<'a> {
    ids: &'a [u32],
    from: u32,
    last: usize,
    steps: bool,
}

impl Call<'_> {
    /// Positions `a..e` of the call with a point at each of `marks`
    /// (ascending, inside the span): a pass taking a copy at each, or the
    /// steps of each run between two marks; past [`SNAPS`] marks the span is
    /// cut at the next one, whose point the first part's end is. Returns the
    /// last pick and where the last position's hidden row sits.
    fn run(
        &self,
        s: &mut Session,
        a: u32,
        e: u32,
        marks: &[u32],
        mut unit: Option<&mut Unit<'_>>,
    ) -> Result<Ran, SessionError> {
        let i = |u: u32| (u - self.from) as usize;
        if self.steps || i(e) - i(a) <= PASS_ROWS {
            let mut from = a;
            for &u in marks {
                step_batch(s, self.ids, i(from), i(u), self.last, unit.as_deref_mut())?;
                Session::checkpoint_hidden(s, STEP_ROW)?;
                from = u;
            }
            let pick = step_batch(s, self.ids, i(from), i(e), self.last, unit)?;
            return Ok(Ran {
                pick,
                hidden: STEP_ROW,
            });
        }
        if !marks.is_empty() && !s.copies_in_pass() {
            // No copy inside a pass: a pass to each mark, then its point.
            let mut from = a;
            for &u in marks {
                let ran = self.run(s, from, u, &[], unit.as_deref_mut())?;
                point(s, Some(ran.hidden))?;
                from = u;
            }
            return self.run(s, from, e, &[], unit);
        }
        if let Some(&cut) = marks.get(SNAPS) {
            let ran = self.run(s, a, cut, &marks[..SNAPS], unit.as_deref_mut())?;
            point(s, Some(ran.hidden))?;
            return self.run(s, cut, e, &marks[SNAPS + 1..], unit);
        }
        let after: Vec<u32> = marks.iter().map(|&u| u - a - 1).collect();
        let out = if i(e) == self.ids.len() {
            PassOut::Last
        } else {
            PassOut::None
        };
        let passed = Session::pass(
            s,
            &self.ids[i(a)..i(e)],
            out,
            StateOut { after_rows: &after },
        )?;
        if let Some(unit) = unit {
            unit(s, MtpArena::Rows, a, (e - a) as usize)?;
        }
        // The pass's rows hold each position's final hidden row: row `r` is
        // position `a + r`, so the point at `u` carries row `u - a - 1`.
        for &u in marks {
            s.checkpoint_copy_hidden(
                u,
                HiddenSrc {
                    arena: MtpArena::Rows,
                    row: (u - a - 1) as usize,
                },
            )?;
        }
        Ok(Ran {
            pick: passed.id,
            hidden: HiddenSrc {
                arena: MtpArena::Rows,
                row: (e - a - 1) as usize,
            },
        })
    }
}

/// The ids `start..end_i` of a prompt stepped one ahead each where the next
/// id is known and inner, none of them but the prompt's last carrying a
/// head: every id but the batch's last steps the next id ahead, both tokens
/// without the head, since no inner pick is read. The batch's last row
/// commits nothing ahead — the mark's take follows — and so does the row before
/// the prompt's last (index `last`), whose next runs the head on its own commit,
/// the one pick the caller reads. Returns the last step's pick.
///
/// With a `unit`, no id runs ahead: each is stepped with no head and no token
/// ahead but the prompt's last (the one that reads its pick), and `unit` hears
/// of each after its step settled, the step's row in the arena.
fn step_batch(
    s: &mut Session,
    ids: &[u32],
    start: usize,
    end_i: usize,
    last: usize,
    mut unit: Option<&mut Unit<'_>>,
) -> Result<u32, SessionError> {
    let mut pick = None;
    for i in start..end_i {
        let mode = if i == last {
            StepMode::HostFed
        } else if unit.is_none() && i + 1 < end_i && i + 1 != last {
            StepMode::Ahead {
                next: Some(ids[i + 1]),
            }
        } else {
            StepMode::Ahead { next: None }
        };
        let at = Session::pos(s);
        pick = Some(Session::step(s, ids[i], mode)?.id);
        if let Some(unit) = unit.as_deref_mut() {
            unit(s, MtpArena::Step, at, 1)?;
        }
    }
    pick.ok_or_else(|| refuse("a batch of no ids".to_string()))
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
