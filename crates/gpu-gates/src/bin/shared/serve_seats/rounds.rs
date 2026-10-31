//! The serve seats' one owner of a round run as passes: a seat whose body
//! has [`SlotRows`] and decided at its open to run one pass overrides
//! [`Seat::step_slots`] through [`step_round`] with [`step_rows_one_pass`]
//! ([`step_rows_drafted`] on a load that drafts with the MTP draft) and its
//! drafted twin [`Seat::pass_slots`] through [`pass_round`] with
//! [`pass_rows_one_pass`]; any other seat keeps the defaults,
//! [`bind::step_rows_in_turn`] and [`bind::pass_rows_in_turn`].
//!
//! [`bind::step_rows_in_turn`]: bloomery_gpu_gates::bind::step_rows_in_turn
//! [`bind::pass_rows_in_turn`]: bloomery_gpu_gates::bind::pass_rows_in_turn

use app::mtp::{Draw, MtpBody, SlotWalks, SlotWindow};
use bloomery_gpu::model::{SlotRows, slots_a_pass};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{
    RowDraw, Seat, SlotPassRow, SlotStep, pass_rows_in_turn, step_rows_in_turn,
};
use bloomery_gpu_gates::record;
use runtime::{Target, Widths, Window};
use serve::Drafted;

use super::drafted::{SlotDrafts, reads_best};

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

/// One round of several slots' drafted passes as passes of
/// `GpuModel::verify_slots` (the drafted seat override of every family
/// whose body has `SlotRows`): [`app::mtp::pass_slots`] over the rows —
/// one a slot, the server's rounds — each row its slot, its `last`, its
/// slot's own window out of `drafts` (the seat's table, one draft a
/// resident slot; a slot with none is refused by name) and `depth`, the
/// depth the seat's one-slot pass proposes, its kept ids into `out` and
/// what it drafted into `drafted`, exactly as the fallback loop's `pass`
/// fills them. Returns the passes it ran. The windows run the draft's
/// batch width selected by `runtime::width::Batch`, using each slot's
/// acceptance profile and the complete round's wall time. A zero width
/// skips that pass's proposal without turning the slot's draft off.
///
/// Rows are laid in slot order ([`slot_order`]), each row's kept ids and
/// counts back to that row; each slot's rows are bit for bit its own
/// wherever they sit in the pass. A round whose windows' rows
/// pass the body's [`SlotRows::MAX_ROWS`] or [`SlotRows::MAX_SLOTS`] limit
/// runs as more than one pass, cut
/// between windows by their proposals' rows, every window's proposal made
/// first and the pass that holds slot 0 last ([`pass_slots`]) — a window
/// is never split: its rows are one verify of its sequence — every pass
/// bit for bit its windows run alone. A slot
/// whose draft skips (its window the one plain row) rides the same pass as
/// a drafted one, [`pass_slots`]'s own case. A slot whose draft the seat
/// turned off ([`SlotDrafts::is_off`]) runs before the passes, alone: its
/// slot selected and the table's own pass on it, the plain step the
/// fallback loop runs for it, its draft told nothing and never asked to
/// propose, since an off draft holds no refresh at the slot's position.
/// A round with off slots runs one more pass for each. No row
/// the server sends is refused here that the fallback loop would run: the
/// rows are distinct slots below the seat's [`Seat::slots`] (`bind`'s own
/// named refusal), so the fallback is the seat's open decision alone
/// (`--parallel 1`, a body without `SlotRows`) and a refusal mid-round is
/// the server's error, never a fallback.
///
/// [`pass_slots`]: app::mtp::pass_slots
/// [`Seat::slots`]: bloomery_gpu_gates::bind::Seat::slots
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
    // `on` keeps each window's row index, the key its answer goes back by.
    let on = slot_order(on, |i| rows[i].slot);
    let w = 1 + depth.min(B::WIDTH);
    if slots_a_pass::<B>(w) == 0 {
        return Err(format!(
            "a drafted window of {w} rows past this body's pass of several slots' {} \
             (SlotRows::MAX_ROWS)",
            <B as SlotRows>::MAX_ROWS
        )
        .into());
    }
    let (specs, batch, widths) = drafts.round_mut();
    if let Some(&i) = on
        .iter()
        .find(|&&i| specs.get(rows[i].slot).is_none_or(Option::is_none))
    {
        return Err(format!("slot {}: the seat holds no draft of it", rows[i].slot).into());
    }
    widths.clear();
    widths.extend(on.iter().map(|&i| {
        let draft = specs[rows[i].slot]
            .as_ref()
            .expect("a checked draft")
            .draft();
        if draft.draft().skipping() {
            0
        } else {
            depth.min(B::WIDTH)
        }
    }));
    let mode = specs[rows[on[0]].slot]
        .as_ref()
        .expect("a checked draft")
        .draft()
        .mode();
    if on.len() > 1 {
        let profiles = on.iter().map(|&i| {
            specs[rows[i].slot]
                .as_ref()
                .expect("a checked draft")
                .draft()
                .batch_profile()
        });
        let limits = (<B as SlotRows>::MAX_ROWS, <B as SlotRows>::MAX_SLOTS);
        batch.choose(mode, profiles, widths, allowed, limits)?;
    }
    let started = std::time::Instant::now();
    let mut taken: Vec<_> = on.iter().map(|&i| specs[rows[i].slot].take()).collect();
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
        let mut windows: Vec<SlotWindow<'_, B>> = Vec::with_capacity(on.len());
        for ((((&i, spec), out), &width), draw) in on
            .iter()
            .zip(&mut taken)
            .zip(&mut outs)
            .zip(&*widths)
            .zip(&mut drawing)
        {
            out.clear();
            windows.push(SlotWindow {
                slot: rows[i].slot,
                last: rows[i].last,
                depth: width,
                // The check above took every row's draft.
                draft: spec
                    .as_mut()
                    .expect("a draft the check above took")
                    .draft_mut()
                    .draft_mut(),
                out,
                draw: draw.as_mut().map(|(f, best)| Draw {
                    take: f,
                    best: *best,
                }),
            });
        }
        app::mtp::pass_slots(s, &mut windows)
    };
    for ((((&i, spec), out), mut draw), given) in
        on.iter().zip(taken).zip(outs).zip(draws).zip(given)
    {
        if let Some(d) = draw.as_mut() {
            d.history.truncate(given);
        }
        specs[rows[i].slot] = spec;
        rows[i].out = out;
        rows[i].draw = draw;
    }
    let (done, ran) = run?;
    for (&i, c) in on.iter().zip(&done) {
        if let Some(spec) = specs[rows[i].slot].as_mut() {
            spec.draft_mut().round_ran(c.rows, c.kept);
        }
        rows[i].drafted = Drafted {
            proposed: if c.proposed { c.rows - 1 } else { 0 },
            accepted: c.kept - 1,
        };
    }
    passes += ran;
    if on.len() > 1 && mode == runtime::width::Mode::Cost {
        batch.observe(
            done.iter().map(|c| c.rows),
            started.elapsed().as_secs_f64() * 1000.0,
        )?;
    }
    Ok(passes)
}
