//! The serve seats' one owner of a round run as passes: a seat whose body
//! has [`SlotRows`] and decided at its open to run one pass
//! (`bind::Pass::of_body`, by what the body can run) overrides
//! [`Seat::step_slots`] through [`step_round`] with [`step_rows_one_pass`]
//! ([`step_rows_drafted`] on a load that drafts with the MTP draft,
//! [`step_rows_lookup`] with the lookup) and its drafted twin
//! [`Seat::pass_slots`] through [`pass_round`] with [`pass_rows_one_pass`]
//! (the MTP draft) or [`pass_rows_lookup`] (the lookup), both one round of
//! any draft's windows ([`pass_windows`] over [`RoundDrafts`]); any other
//! seat keeps the defaults, [`bind::step_rows_in_turn`] and
//! [`bind::pass_rows_in_turn`].
//!
//! [`bind::step_rows_in_turn`]: bloomery_gpu_gates::bind::step_rows_in_turn
//! [`bind::pass_rows_in_turn`]: bloomery_gpu_gates::bind::pass_rows_in_turn

use app::mtp::{MtpBody, SlotWalks};
use app::{Draw, Keep, Prompt};
use bloomery_gpu::model::{Rollback, Rows, SlotRows, slots_a_pass};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{
    RowDraw, Seat, SlotPassRow, SlotStep, pass_rows_in_turn, step_rows_in_turn,
};
use bloomery_gpu_gates::record;
use runtime::width::{Mode as WidthMode, WidthError};
use runtime::{Committed, Target, Widths, Window};
use serve::Drafted;

use super::drafted::{LOOKUP_ROWS, Lookups, SlotDrafts, reads_best};

/// One round of several slots' steps, a seat's [`Seat::step_slots`]
/// override, its one owner: the fallback loop ([`step_rows_in_turn`], which
/// prints its own record) unless the seat's open decided `one_pass`; else
/// `run`, the seat's pass of the rows (the passes it ran back), then
/// `after`, what the seat prints of the call (its `residency pass` records),
/// then — while the seat counts its rounds ([`Seat::step_stats`]) — the
/// `slots round` record. A refusal on either path is the server's to die
/// on: the open decides once, so a load that cannot run one pass never
/// tries it at run time.
pub fn step_round<S: Seat>(
    seat: &mut S,
    one_pass: bool,
    rows: &mut [SlotStep],
    run: impl FnOnce(&mut S, &mut [SlotStep]) -> Result<usize, GateError>,
    after: impl FnOnce(&mut S) -> Result<(), GateError>,
) -> Result<(), String> {
    if !one_pass {
        return step_rows_in_turn(seat, rows);
    }
    let passes = run(seat, rows).map_err(|e| e.to_string())?;
    after(seat).map_err(|e| e.to_string())?;
    if seat.step_stats() {
        record::require(record::groups::rounds::ROUND, "rounds::step_round");
        record::slots_round("step", rows.len(), passes, seat.slots()).eprint();
    }
    Ok(())
}

/// One round of several slots' drafted passes, a seat's [`Seat::pass_slots`]
/// override, its one owner: the fallback loop ([`pass_rows_in_turn`], which
/// prints its own record) unless the seat's open decided `one_pass`; else
/// `run`, the seat's pass of the rows — what it first does to the rows a
/// seat's own rule asks (GLM turns a skipping slot's draft off before the
/// pass, Qwen3.8 prints each slot's own pre-call) stays seat code inside
/// the closure — then `after`, what the seat prints of the call (its
/// `residency pass` records), then — while the seat counts its rounds
/// ([`Seat::step_stats`]) — the `slots round` record, cmd `pass`. A
/// refusal on either path is the server's to die on: the open decides
/// once, so a load that cannot run one pass never tries it at run time.
pub fn pass_round<S: Seat>(
    seat: &mut S,
    one_pass: bool,
    rows: &mut [SlotPassRow],
    run: impl FnOnce(&mut S, &mut [SlotPassRow]) -> Result<usize, GateError>,
    after: impl FnOnce(&mut S) -> Result<(), GateError>,
) -> Result<(), String> {
    if !one_pass {
        return pass_rows_in_turn(seat, rows);
    }
    let passes = run(seat, rows).map_err(|e| e.to_string())?;
    after(seat).map_err(|e| e.to_string())?;
    if seat.step_stats() {
        record::require(record::groups::rounds::ROUND, "rounds::pass_round");
        record::slots_round("pass", rows.len(), passes, seat.slots()).eprint();
    }
    Ok(())
}

/// The order a pass of several slots lays its rows in: the row indices `at`
/// sorted by their slots (`slot` of an index). A body may require it:
/// body38's `slot_rows` holds slot 0's rows from row 0, so a pass that runs
/// slot 0 lays its rows first. The rows come in the order their requests
/// arrived; each index is the row its answers go back to, never the row at
/// its pass position.
fn slot_order(mut at: Vec<usize>, slot: impl Fn(usize) -> usize) -> Vec<usize> {
    at.sort_by_key(|&i| slot(i));
    at
}

/// One round of several slots' steps as passes of `GpuModel::step_slots`
/// (the seat override of every family whose body has `SlotRows`):
/// [`app::Session::step_slots_rounds`] over the rows — one a slot, the
/// server's rounds — laid in slot order ([`slot_order`]), each answer into
/// its own row's `next` and each lent logits row filled from the pass's own
/// per-row heads, the NaN guard the fallback loop keeps. Returns the passes
/// it ran. Refused by name before any row is written: answers or logits
/// rows other than one a row, no logits rows while a row lent one, and a
/// lent row of another length than the pass's row.
pub fn step_rows_one_pass<B>(
    s: &mut app::Session<B>,
    rows: &mut [SlotStep],
) -> Result<usize, GateError>
where
    B: SlotRows,
    B::Seq: 'static,
{
    let order = slot_order((0..rows.len()).collect(), |i| rows[i].slot);
    let last: Vec<u32> = order.iter().map(|&i| rows[i].last).collect();
    let round: Vec<(usize, &[u32])> = order
        .iter()
        .zip(last.chunks(1))
        .map(|(&i, one)| (rows[i].slot, one))
        .collect();
    let lent: Vec<bool> = order.iter().map(|&i| rows[i].logits.is_some()).collect();
    let out = s.step_slots_rounds(&round, &lent)?;
    if out.ids.len() != order.len() {
        return Err(format!(
            "a round of {} rows answered {} ids",
            order.len(),
            out.ids.len()
        )
        .into());
    }
    match &out.logits {
        Some(all) if all.rows().len() != order.len() => {
            return Err(format!(
                "a round of {} rows read back {} logits rows",
                order.len(),
                all.rows().len()
            )
            .into());
        }
        Some(all) => {
            for (&i, row) in order.iter().zip(all.rows()) {
                let want = rows[i].logits.as_ref().map(Vec::len);
                if want != row.map(<[f32]>::len) {
                    return Err(format!(
                        "slot {}: a logits request of {want:?} values lent for the pass's row of {:?}",
                        rows[i].slot,
                        row.map(<[f32]>::len)
                    )
                    .into());
                }
            }
        }
        None if lent.iter().any(|&want| want) => {
            return Err("a round whose rows lent logits rows read back none".into());
        }
        None => {}
    }
    for (&i, &next) in order.iter().zip(&out.ids) {
        rows[i].next = next;
    }
    if let Some(all) = out.logits {
        for (&i, row) in order.iter().zip(all.rows()) {
            if let (Some(dst), Some(row)) = (rows[i].logits.as_deref_mut(), row) {
                dst.copy_from_slice(row);
            }
        }
    }
    Ok(out.passes)
}

/// One round of several slots' plain steps on a load that drafts with the
/// MTP draft as passes of `GpuModel::step_slots` ([`step_rows_one_pass`]),
/// each slot's draft told of its row as a step alone tells it: before the
/// pass, on each row's slot, [`SlotDrafts::gather_row`] (shared prompt rows
/// walked now, step and verify rows gathered), then the gathered rows in one
/// store walk ([`SlotWalks::run`], the pass overwriting the arenas they
/// read); after it [`SlotDrafts::after_row`] with the position the row ran
/// from and its answer (the next refresh the row's, its hidden row in the
/// pass's arena, which no later pass of the round overwrites: a round past
/// [`SlotRows::MAX_ROWS`] rows or [`SlotRows::MAX_SLOTS`] sequences runs the
/// pass that holds slot 0 last, [`app::Session::step_slots_rounds`]).
/// A slot whose draft is off rides the pass as a plain row its draft is
/// told nothing of, as its step alone. A row's answer and lent logits row
/// are the pass's, bit for bit its step alone, so a sampling or
/// id-banning request's ids are its alone ids. Returns the passes it ran;
/// refused as [`step_rows_one_pass`] refuses, and as the table refuses a
/// slot past it.
pub(crate) fn step_rows_drafted<B, const M: usize>(
    s: &mut app::Session<B>,
    rows: &mut [SlotStep],
    drafts: &mut SlotDrafts<B, M>,
) -> Result<usize, GateError>
where
    B: MtpBody + SlotRows,
    B::Seq: 'static,
    Window<M>: Widths,
{
    let mut p0 = Vec::with_capacity(rows.len());
    let mut walks = SlotWalks::default();
    for r in rows.iter() {
        s.select_slot(r.slot)?;
        p0.push(s.pos());
        drafts.gather_row(s, r.slot, r.last, &mut walks)?;
    }
    walks.run(s)?;
    let passes = step_rows_one_pass(s, rows)?;
    for (r, &p) in rows.iter().zip(&p0) {
        drafts.after_row(r.slot, p, r.last, r.next)?;
    }
    Ok(passes)
}

/// One round of several slots' plain steps on a load that drafts with the
/// lookup as passes of `GpuModel::step_slots` ([`step_rows_one_pass`]), each
/// slot's lookup context told of its row as a step alone tells it
/// ([`Lookups::stepped_at`], from the position the row ran at): a lookup
/// load's plain rounds run as one pass as a plain load's do, its contexts
/// following each row at its own slot's position. Returns the passes it
/// ran; refused as [`step_rows_one_pass`] refuses.
pub(crate) fn step_rows_lookup<B>(
    s: &mut app::Session<B>,
    rows: &mut [SlotStep],
    lookups: &mut Lookups<app::Session<B>>,
) -> Result<usize, GateError>
where
    B: SlotRows + Prompt + Keep + Rows + Rollback,
    B::Seq: 'static,
{
    let mut p0 = Vec::with_capacity(rows.len());
    for r in rows.iter() {
        s.select_slot(r.slot)?;
        p0.push(s.pos());
    }
    let passes = step_rows_one_pass(s, rows)?;
    for (r, &p) in rows.iter().zip(&p0) {
        lookups.stepped_at(s, r.slot, p as usize, r.last, r.next)?;
    }
    Ok(passes)
}

/// One busy slot's window of a round verified together
/// ([`RoundDrafts::run`]): its slot, the token at its position, the most ids
/// its proposal holds (0: a plain row, its draft told of it as of a step),
/// where its kept ids are appended, and for a request that draws, its draw.
pub(crate) struct RoundRow<'a> {
    pub slot: usize,
    pub last: u32,
    pub width: usize,
    pub out: &'a mut Vec<u32>,
    pub draw: Option<Draw<'a>>,
}

/// A seat's table of drafts, one a resident slot, that runs a round of
/// several slots' windows verified together: the MTP table
/// ([`SlotDrafts`]) and the lookup's ([`Lookups`]). Each slot's width
/// chooser stands aside for the round, the round's batch chooser
/// (`runtime::width::Batch`) cutting the windows' widths from each slot's
/// acceptance profile and the round's own wall; [`pass_windows`] drives it.
pub(crate) trait RoundDrafts<B: SlotRows> {
    /// The most ids `slot`'s window proposes at `depth`: 0 for a draft that
    /// proposes nothing until its next reset. Refused by name for a slot the
    /// table holds no draft of.
    fn width(&self, slot: usize, depth: usize) -> Result<usize, GateError>;

    /// The width chooser's mode of `slot`'s draft (every slot's is the
    /// seat's one).
    fn mode(&self, slot: usize) -> WidthMode;

    /// `widths`, one a slot of `slots`, cut by the round's batch chooser
    /// (`Batch::choose` over each slot's acceptance profile, within
    /// `allowed` and the body's pass `limits`).
    fn choose(
        &mut self,
        slots: &[usize],
        widths: &mut [usize],
        allowed: std::ops::RangeInclusive<usize>,
        limits: (usize, usize),
    ) -> Result<(), WidthError>;

    /// The round: each row's window proposed on its own slot, every row's
    /// before any pass, then verified together
    /// ([`app::Session::verify_rounds`]), each slot's draft told of its
    /// kept rows. Returns each row's commit, in order, and the passes run.
    fn run(
        &mut self,
        s: &mut app::Session<B>,
        round: &mut [RoundRow<'_>],
    ) -> Result<(Vec<Committed>, usize), GateError>;

    /// `slot`'s window ran `rows` and kept `kept` in the round: its width
    /// chooser counts it (`Choosing::round_ran`).
    fn ran(&mut self, slot: usize, rows: usize, kept: usize);

    /// The round's wall in milliseconds, each window's rows in order, for
    /// the batch chooser (`Batch::observe`).
    fn observe(&mut self, rows: &[usize], ms: f64) -> Result<(), WidthError>;
}

/// One round of several slots' drafted passes as passes of
/// `GpuModel::verify_slots` (the drafted seat override of every family
/// whose body verifies several slots' windows together,
/// `bind::verifies_slots`), under the MTP draft: each slot whose draft the
/// seat turned off ([`SlotDrafts::is_off`]) runs before the passes, alone —
/// its slot selected and the table's own pass on it, the plain step the
/// fallback loop runs for it, its draft told nothing and never asked to
/// propose, since an off draft holds no refresh at the slot's position (a
/// round with off slots runs one more pass for each) — and every other row
/// is a window of [`pass_windows`] at `depth`, the depth the seat's
/// one-slot pass proposes, which [`app::mtp::pass_slots`] proposes and
/// verifies. A slot whose draft skips (its window the one plain row) rides
/// the same pass as a drafted one, [`app::mtp::pass_slots`]' own case.
/// Returns the passes it ran.
pub(crate) fn pass_rows_one_pass<B, const M: usize>(
    s: &mut app::Session<B>,
    rows: &mut [SlotPassRow],
    drafts: &mut SlotDrafts<B, M>,
    depth: usize,
    allowed: std::ops::RangeInclusive<usize>,
) -> Result<usize, GateError>
where
    B: MtpBody + SlotRows,
    B::Seq: 'static,
    Window<M>: Widths,
{
    let mut passes = 0;
    // The off slots first, each alone; the rest are the windows below.
    let mut on = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter_mut().enumerate() {
        if !drafts.is_off(r.slot) {
            on.push(i);
            continue;
        }
        s.select_slot(r.slot)?;
        r.out.clear();
        r.drafted = match r.draw.as_mut() {
            None => drafts.pass(s, r.slot, r.last, &mut r.out)?,
            Some(d) => drafts.pass_sampled(
                s,
                r.slot,
                r.last,
                &mut d.history,
                &mut d.sampler,
                d.best,
                &mut r.out,
            )?,
        };
        passes += 1;
    }
    if on.is_empty() {
        return Ok(passes);
    }
    Ok(passes + pass_windows(s, rows, on, drafts, depth.min(B::WIDTH), allowed)?)
}

/// One round of several slots' lookup passes as passes of
/// `GpuModel::verify_slots`: every row a window of [`pass_windows`] over
/// its slot's lookup context, proposing at most the lookup's one id
/// ([`LOOKUP_ROWS`]), or none where the round's chooser holds it back (a
/// width of 0, which the cost mode picks when the proposal does not pay).
/// The lookup is never off, so no row runs alone. Returns the passes it
/// ran.
pub(crate) fn pass_rows_lookup<B>(
    s: &mut app::Session<B>,
    rows: &mut [SlotPassRow],
    lookups: &mut Lookups<app::Session<B>>,
) -> Result<usize, GateError>
where
    B: SlotRows + Prompt + Keep + Rows + Rollback,
    B::Seq: 'static,
{
    let depth = LOOKUP_ROWS - 1;
    pass_windows(
        s,
        rows,
        (0..rows.len()).collect(),
        lookups,
        depth,
        0..=depth,
    )
}

/// The windows `on` (indices of `rows`) of a round of several slots'
/// drafted passes, every draft's window proposed and the windows verified
/// together (the seat's one owner of a round of any draft's windows over
/// [`RoundDrafts`]): each row its slot, its `last`, its slot's draft out of
/// `drafts` (a slot with none is refused by name) at `depth`, its kept ids
/// into `out` and what it drafted into `drafted`, exactly as the fallback
/// loop's `pass` fills them. The windows run the draft's batch width
/// selected by `runtime::width::Batch`, using each slot's acceptance profile
/// and the complete round's wall time. A zero width skips that pass's
/// proposal without turning the slot's draft off.
///
/// Rows are laid in slot order ([`slot_order`]), each row's kept ids and
/// counts back to that row; each slot's rows are bit for bit its own
/// wherever they sit in the pass. A round whose windows' rows
/// pass the body's [`SlotRows::MAX_ROWS`] or [`SlotRows::MAX_SLOTS`] limit
/// runs as more than one pass, cut between windows by their proposals'
/// rows, every window's proposal made first and the pass that holds slot 0
/// last ([`app::Session::verify_rounds`]) — a window is never split: its
/// rows are one verify of its sequence. No row the server sends is refused
/// here that the fallback loop would run: the rows are distinct slots below
/// the seat's [`Seat::slots`] (`bind`'s own named refusal), so the fallback
/// is the seat's open decision alone (`--parallel 1`, a body that does not
/// verify several slots' windows) and a refusal mid-round is the server's
/// error, never a fallback. Returns the passes it ran.
///
/// [`Seat::slots`]: bloomery_gpu_gates::bind::Seat::slots
fn pass_windows<B, D>(
    s: &mut app::Session<B>,
    rows: &mut [SlotPassRow],
    on: Vec<usize>,
    drafts: &mut D,
    depth: usize,
    allowed: std::ops::RangeInclusive<usize>,
) -> Result<usize, GateError>
where
    B: SlotRows,
    B::Seq: 'static,
    D: RoundDrafts<B>,
{
    // `on` keeps each window's row index, the key its answer goes back by.
    let on = slot_order(on, |i| rows[i].slot);
    if slots_a_pass::<B>(1 + depth) == 0 {
        return Err(format!(
            "a drafted window of {} rows past this body's pass of several slots' {} \
             (SlotRows::MAX_ROWS)",
            1 + depth,
            <B as SlotRows>::MAX_ROWS
        )
        .into());
    }
    let slots: Vec<usize> = on.iter().map(|&i| rows[i].slot).collect();
    let lasts: Vec<u32> = on.iter().map(|&i| rows[i].last).collect();
    let mut widths = slots
        .iter()
        .map(|&slot| drafts.width(slot, depth))
        .collect::<Result<Vec<_>, _>>()?;
    let mode = drafts.mode(slots[0]);
    if on.len() > 1 {
        let limits = (<B as SlotRows>::MAX_ROWS, <B as SlotRows>::MAX_SLOTS);
        drafts.choose(&slots, &mut widths, allowed, limits)?;
    }
    let started = std::time::Instant::now();
    let mut outs: Vec<Vec<u32>> = on
        .iter()
        .map(|&i| std::mem::take(&mut rows[i].out))
        .collect();
    let mut draws: Vec<Option<RowDraw>> = on.iter().map(|&i| rows[i].draw.take()).collect();
    let given: Vec<usize> = draws
        .iter()
        .map(|d| d.as_ref().map_or(0, |d| d.history.len()))
        .collect();
    let run = {
        // A sampled row's draws see its history, then the ids its pass took
        // before (`SlotDrafts::pass_sampled`'s draw).
        let mut drawing: Vec<_> = draws
            .iter_mut()
            .map(|d| {
                d.as_mut().map(|d| {
                    let best = reads_best(d.best);
                    let draw = move |row: &[f32]| {
                        let id = (d.sampler)(row, &d.history);
                        d.history.push(id);
                        id
                    };
                    (draw, best)
                })
            })
            .collect();
        let mut round: Vec<RoundRow<'_>> = Vec::with_capacity(on.len());
        for ((((&slot, &last), out), &width), draw) in slots
            .iter()
            .zip(&lasts)
            .zip(&mut outs)
            .zip(&widths)
            .zip(&mut drawing)
        {
            out.clear();
            round.push(RoundRow {
                slot,
                last,
                width,
                out,
                draw: draw.as_mut().map(|(f, best)| Draw {
                    take: f,
                    best: *best,
                }),
            });
        }
        drafts.run(s, &mut round)
    };
    for (((&i, out), mut draw), given) in on.iter().zip(outs).zip(draws).zip(given) {
        if let Some(d) = draw.as_mut() {
            d.history.truncate(given);
        }
        rows[i].out = out;
        rows[i].draw = draw;
    }
    let (done, ran) = run?;
    for ((&i, &slot), c) in on.iter().zip(&slots).zip(&done) {
        drafts.ran(slot, c.rows, c.kept);
        rows[i].drafted = Drafted {
            proposed: if c.proposed { c.rows - 1 } else { 0 },
            accepted: c.kept - 1,
        };
    }
    if on.len() > 1 && mode == WidthMode::Cost {
        let ran_rows: Vec<usize> = done.iter().map(|c| c.rows).collect();
        drafts.observe(&ran_rows, started.elapsed().as_secs_f64() * 1000.0)?;
    }
    Ok(ran)
}
