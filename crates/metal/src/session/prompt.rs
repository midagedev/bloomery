//! The prompt machinery: how a prompt call's ids run over a family's session
//! — cut at the checkpoint marks, each run between two marks a pass or the
//! ids' steps — generic over [`Family`], the seam derived from what this
//! machinery calls on a session. The marks and batches are the runtime's own
//! (`runtime::seqstate::marks`, `runtime::prompt::call_batches`); this module
//! owns the schedule around them.
//!
//! A call's ids are cut at its checkpoint marks: the call's start when the
//! session already holds one, every inner spacing inside, and the call's
//! end. Each run between two marks is cut by `call_batches` at the session's
//! pass rows; a batch of more than a step's window is one pass; a shorter
//! one steps its ids one ahead each where the next id is known and inner —
//! no head on a row whose pick nobody reads — the span's last row and the
//! row before the prompt's last committing nothing ahead, the mark's take
//! and the last row's own head being the one pick read back after the last
//! as asked. Each mark's checkpoint waits out whatever token runs ahead of
//! it. A session in steps mode steps every batch, the same-binary arm and
//! the oracle's.
//!
//! A caller's mark inside a pass is a copy the pass leaves after the row
//! before it, taken as a point once the pass settles; a mark inside a run of
//! steps ends that run, its point the step's own state; a mark at a batch's
//! end is that batch's point. A pass takes at most the kernel's snap count:
//! a further mark ends the pass there. A mark out of order or outside the
//! call is refused by name before any step runs.

use runtime::prompt::{PrefillMode, call_batches};

use crate::kernels::{PASS_ROWS, SNAPS};
use crate::session::clock::{StepMode, Stepped};
use crate::session::pass::{PassOut, Passed, StateOut};

/// What a family's session gives the prompt machinery, derived from what
/// this module calls on it: the positions, marks and passes of a prompt
/// call, the checkpoints it takes, and the words of its refusals. A family's
/// session implements this beside its own step protocol; nothing here reads
/// the family's arena or ops.
pub trait Family {
    /// The family's refusals and failures, as its session words them.
    type Error;

    /// The arenas a run's final hidden rows sit in, as a unit hears them.
    type Arena: Copy;

    /// Where the final hidden row of one position sits, as a checkpoint
    /// keeps it.
    type Row: Copy;

    /// The step arena: a unit's rows there are the last step's.
    const STEP_ARENA: Self::Arena;

    /// The rows arena: a unit's rows there are a pass's.
    const ROWS_ARENA: Self::Arena;

    /// The step arena's one row: where a step leaves its position's final
    /// hidden row.
    fn step_row() -> Self::Row;

    /// Row `row` of the rows arena: where a pass leaves the final hidden row
    /// of the position that row is.
    fn rows_row(row: usize) -> Self::Row;

    /// The next step's position.
    fn pos(&self) -> u32;

    /// The rows the cache holds.
    fn ctx(&self) -> usize;

    /// How a prompt call feeds its ids: in passes, or a decode step each.
    fn prefill(&self) -> PrefillMode;

    /// The most rows a pass of this session holds.
    fn pass_rows(&self) -> usize;

    /// Whether a pass can copy the recurrent stores after one of its rows:
    /// a caller marks a file it cannot between passes.
    fn copies_in_pass(&self) -> bool;

    /// The positions a prompt call from `from` to `to` takes a checkpoint
    /// at.
    fn checkpoint_marks(&self, from: u32, to: u32) -> Vec<u32>;

    /// A checkpoint at the position the session stands at, the stream idle.
    fn checkpoint(&mut self) -> Result<(), Self::Error>;

    /// [`Family::checkpoint`] that also keeps the hidden row `hidden` of the
    /// position before it.
    fn checkpoint_hidden(&mut self, hidden: Self::Row) -> Result<(), Self::Error>;

    /// A checkpoint at `at`, a position inside the last pass, that also
    /// keeps the hidden row `hidden`.
    fn checkpoint_copy_hidden(&mut self, at: u32, hidden: Self::Row) -> Result<(), Self::Error>;

    /// One token at the next position, `id` its input.
    fn step(&mut self, id: u32, mode: StepMode) -> Result<Stepped, Self::Error>;

    /// `ids` as one pass at the next positions.
    fn pass(
        &mut self,
        ids: &[u32],
        out: PassOut,
        snaps: StateOut<'_>,
    ) -> Result<Passed, Self::Error>;

    /// The family's refusal of `detail`, by name.
    fn refuse(detail: String) -> Self::Error;
}

/// What a prompt call tells after each unit it ran: the arena the unit's
/// final hidden rows sit in, the unit's first position and its rows.
pub type Unit<'a, F> =
    dyn FnMut(&mut F, <F as Family>::Arena, u32, usize) -> Result<(), <F as Family>::Error> + 'a;

/// The prompt call of a target's `prompt_marked` with no mark of the caller's
/// beyond `marks`, the pick after its last id returned; `unit`, when given, is
/// called after each unit the call ran, before the next unit overwrites the
/// unit's rows. A call with a unit steps every id that is not its last with no
/// token ahead (the step arena's residual is that token's), where a call with
/// none steps them one ahead each; the programs of a call with none are the
/// call's alone.
pub fn prompt_call<F>(
    s: &mut F,
    ids: &[u32],
    marks: &[u32],
    mut unit: Option<&mut Unit<'_, F>>,
) -> Result<u32, F::Error>
where
    F: Family,
{
    if ids.is_empty() {
        return Err(F::refuse(format!(
            "a prompt of no ids at position {}",
            s.pos()
        )));
    }
    let end = u64::from(s.pos()) + ids.len() as u64;
    // The open refuses a cache of more rows than a u32 position names.
    let ctx = u32::try_from(s.ctx()).expect("open refused a cache a u32 position cannot name");
    if end > u64::from(ctx) {
        return Err(F::refuse(format!(
            "a prompt of {} ids at position {}: the cache holds {} rows",
            ids.len(),
            s.pos(),
            s.ctx()
        )));
    }
    let from = s.pos();
    let to = u32::try_from(end).map_err(|_| {
        F::refuse(format!(
            "a prompt of {} ids at position {from} passes the u32 positions",
            ids.len()
        ))
    })?;
    let inside = marks.first().is_none_or(|&u| u > from)
        && marks.last().is_none_or(|&u| u <= to)
        && marks.windows(2).all(|w| w[0] < w[1]);
    if !inside {
        return Err(F::refuse(format!(
            "a prompt of positions {from}..{to} marked at {marks:?}: marks go ascending, past \
             the call's start and at most its end"
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
        steps: s.prefill() == PrefillMode::Steps,
    };
    let batch_rows = s.pass_rows();
    let mut at = from;
    let mut last_ran: Option<Ran<F>> = None;
    for p in points {
        // The runs between two points, each cut into batches: a pass or
        // its ids' steps, with the marks inside it. A point at the start of
        // the call (the position the session already stands at) has no run
        // before it and keeps the row it was taken with, if any.
        let mut ended: Option<F::Row> = None;
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
        F::refuse(format!(
            "a prompt of {} ids at position {from} stepped no id",
            ids.len()
        ))
    })?;
    Ok(pick)
}

/// The checkpoint at the session's position, beside the target's final hidden
/// row of the position before it where the run that ended there left one.
fn point<F>(s: &mut F, hidden: Option<F::Row>) -> Result<(), F::Error>
where
    F: Family,
{
    match hidden {
        Some(h) => s.checkpoint_hidden(h),
        None => s.checkpoint(),
    }
}

/// What a run of a prompt call left: the pick of its last row, and where the
/// final hidden row of its last position sits.
struct Ran<F: Family> {
    pick: u32,
    hidden: F::Row,
}

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
    /// steps of each run between two marks; past the kernel's snap count of
    /// marks the span is cut at the next one, whose point the first part's
    /// end is. Returns the last pick and where the last position's hidden
    /// row sits.
    fn run<F>(
        &self,
        s: &mut F,
        a: u32,
        e: u32,
        marks: &[u32],
        mut unit: Option<&mut Unit<'_, F>>,
    ) -> Result<Ran<F>, F::Error>
    where
        F: Family,
    {
        let i = |u: u32| (u - self.from) as usize;
        if self.steps || i(e) - i(a) <= PASS_ROWS {
            let mut from = a;
            for &u in marks {
                step_batch(s, self.ids, i(from), i(u), self.last, unit.as_deref_mut())?;
                s.checkpoint_hidden(F::step_row())?;
                from = u;
            }
            let pick = step_batch(s, self.ids, i(from), i(e), self.last, unit)?;
            return Ok(Ran {
                pick,
                hidden: F::step_row(),
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
        let passed = s.pass(&self.ids[i(a)..i(e)], out, StateOut { after_rows: &after })?;
        if let Some(unit) = unit {
            unit(s, F::ROWS_ARENA, a, (e - a) as usize)?;
        }
        // The pass's rows hold each position's final hidden row: row `r` is
        // position `a + r`, so the point at `u` carries row `u - a - 1`.
        for &u in marks {
            s.checkpoint_copy_hidden(u, F::rows_row((u - a - 1) as usize))?;
        }
        Ok(Ran {
            pick: passed.id,
            hidden: F::rows_row((e - a - 1) as usize),
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
fn step_batch<F>(
    s: &mut F,
    ids: &[u32],
    start: usize,
    end_i: usize,
    last: usize,
    mut unit: Option<&mut Unit<'_, F>>,
) -> Result<u32, F::Error>
where
    F: Family,
{
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
        let at = s.pos();
        pick = Some(s.step(ids[i], mode)?.id);
        if let Some(unit) = unit.as_deref_mut() {
            unit(s, F::STEP_ARENA, at, 1)?;
        }
    }
    pick.ok_or_else(|| F::refuse("a batch of no ids".to_string()))
}
