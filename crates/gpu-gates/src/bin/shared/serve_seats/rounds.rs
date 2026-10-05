//! The serve seats' one owner of a round run as passes: a seat whose body
//! has [`SlotRows`] and decided at its open to run one pass overrides
//! [`Seat::step_slots`] through [`step_round`] with [`step_rows_one_pass`]
//! and its drafted twin [`Seat::pass_slots`] with [`pass_rows_one_pass`];
//! any other seat keeps the defaults, [`bind::step_rows_in_turn`] and
//! [`bind::pass_rows_in_turn`].
//!
//! [`bind::step_rows_in_turn`]: bloomery_gpu_gates::bind::step_rows_in_turn
//! [`bind::pass_rows_in_turn`]: bloomery_gpu_gates::bind::pass_rows_in_turn

use app::mtp::{MtpBody, SlotWindow};
use bloomery_gpu::model::SlotRows;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{Seat, SlotPassRow, SlotStep, step_rows_in_turn};
use bloomery_gpu_gates::record;
use runtime::{Widths, Window};
use serve::Drafted;

use super::drafted::SlotDrafts;

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
        record::slots_round("step", rows.len(), passes, seat.slots()).eprint();
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
    let lent = rows.iter().any(|r| r.logits.is_some());
    let out = s.step_slots_rounds(&round, lent)?;
    if out.ids.len() != order.len() {
        return Err(format!(
            "a round of {} rows answered {} ids",
            order.len(),
            out.ids.len()
        )
        .into());
    }
    match &out.logits {
        Some(all) if all.len() != order.len() => {
            return Err(format!(
                "a round of {} rows read back {} logits rows",
                order.len(),
                all.len()
            )
            .into());
        }
        Some(all) => {
            for (&i, row) in order.iter().zip(all) {
                let want = rows[i].logits.as_ref().map_or(row.len(), Vec::len);
                if want != row.len() {
                    return Err(format!(
                        "slot {}: a logits row of {want} values lent for the pass's row of {}",
                        rows[i].slot,
                        row.len()
                    )
                    .into());
                }
            }
        }
        None if lent => {
            return Err("a round whose rows lent logits rows read back none".into());
        }
        None => {}
    }
    for (&i, &next) in order.iter().zip(&out.ids) {
        rows[i].next = next;
    }
    if let Some(all) = out.logits {
        for (&i, row) in order.iter().zip(all) {
            if let Some(dst) = rows[i].logits.as_deref_mut() {
                dst.copy_from_slice(&row);
            }
        }
    }
    Ok(out.passes)
}

/// One round of several slots' drafted passes as passes of
/// `GpuModel::verify_slots` (the drafted seat override of every family
/// whose body has `SlotRows`): [`app::mtp::pass_slots`] over the rows —
/// one a slot, the server's rounds — each row its slot, its `last`, its
/// slot's own window out of `drafts` (the seat's table, one draft a
/// resident slot; a slot with none is refused by name) and `depth`, the
/// depth the seat's one-slot pass proposes, its kept ids into `out` and
/// what it drafted into `drafted`, exactly as the fallback loop's `pass`
/// fills them. Returns the passes it ran.
///
/// Rows are laid in slot order ([`slot_order`]), each row's kept ids and
/// counts back to that row; each slot's rows are bit for bit its own
/// wherever they sit in the pass. A round whose windows' rows
/// pass the body's [`SlotRows::MAX_ROWS`] runs as more than one pass, cut
/// between windows — a window is never split: its rows are one verify of
/// its sequence — every pass bit for bit its windows run alone. A slot
/// whose draft skips (its window the one plain row) rides the same pass as
/// a drafted one, [`pass_slots`]'s own case. A slot whose draft the seat
/// turned off ([`SlotDrafts::is_off`]) runs before the passes, alone: its
/// slot selected and the table's own pass on it, the plain step the
/// fallback loop runs for it, its draft told nothing and never asked to
/// propose — [`pass_slots`] proposes on every window and refuses a window
/// of depth 0 by name, and an off draft holds no refresh at the slot's
/// position — so a round with off slots runs one more pass for each. No row
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
        r.drafted = drafts.pass(s, r.slot, r.last, &mut r.out)?;
        passes += 1;
    }
    if on.is_empty() {
        return Ok(passes);
    }
    // `on` keeps each window's row index, the key its answer goes back by.
    let on = slot_order(on, |i| rows[i].slot);
    // A pass's windows: as many as their most rows — the token before a
    // full-depth proposal — fit the body's pass; a window past the bound is
    // refused by name, a depth the body's pass of several slots cannot hold.
    let w = 1 + depth.min(B::WIDTH);
    let per = <B as SlotRows>::MAX_ROWS / w;
    if per == 0 {
        return Err(format!(
            "a drafted window of {w} rows past this body's pass of several slots' {} \
             (SlotRows::MAX_ROWS)",
            <B as SlotRows>::MAX_ROWS
        )
        .into());
    }
    for pass in on.chunks(per) {
        let specs = drafts.specs_mut();
        if let Some(&i) = pass
            .iter()
            .find(|&&i| specs.get(rows[i].slot).is_none_or(Option::is_none))
        {
            return Err(format!("slot {}: the seat holds no draft of it", rows[i].slot).into());
        }
        let mut taken: Vec<_> = pass.iter().map(|&i| specs[rows[i].slot].take()).collect();
        let mut outs: Vec<Vec<u32>> = pass
            .iter()
            .map(|&i| std::mem::take(&mut rows[i].out))
            .collect();
        let run = {
            let mut windows: Vec<SlotWindow<'_, B>> = Vec::with_capacity(pass.len());
            for ((&i, spec), out) in pass.iter().zip(&mut taken).zip(&mut outs) {
                out.clear();
                windows.push(SlotWindow {
                    slot: rows[i].slot,
                    last: rows[i].last,
                    depth,
                    // The check above took every row's draft.
                    draft: spec
                        .as_mut()
                        .expect("a draft the check above took")
                        .draft_mut(),
                    out,
                });
            }
            app::mtp::pass_slots(s, &mut windows)
        };
        for ((&i, spec), out) in pass.iter().zip(taken).zip(outs) {
            specs[rows[i].slot] = spec;
            rows[i].out = out;
        }
        for (&i, c) in pass.iter().zip(run?) {
            rows[i].drafted = Drafted {
                proposed: if c.proposed { c.rows - 1 } else { 0 },
                accepted: c.kept - 1,
            };
        }
        passes += 1;
    }
    Ok(passes)
}
