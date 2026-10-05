//! The serve seats' one owner of a round run as passes: a seat whose body
//! has [`SlotRows`] and decided at its open to run one pass overrides
//! [`Seat::step_slots`] with [`step_rows_one_pass`] and its drafted twin
//! [`Seat::pass_slots`] with [`pass_rows_one_pass`]; any other seat keeps
//! the defaults, [`bind::step_rows_in_turn`] and [`bind::pass_rows_in_turn`].
//!
//! [`Seat::step_slots`]: bloomery_gpu_gates::bind::Seat::step_slots
//! [`Seat::pass_slots`]: bloomery_gpu_gates::bind::Seat::pass_slots
//! [`bind::step_rows_in_turn`]: bloomery_gpu_gates::bind::step_rows_in_turn
//! [`bind::pass_rows_in_turn`]: bloomery_gpu_gates::bind::pass_rows_in_turn

use app::mtp::{MtpBody, SlotWindow};
use bloomery_gpu::model::SlotRows;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{SlotPassRow, SlotStep};
use runtime::{Widths, Window};
use serve::Drafted;

use super::drafted::SlotDrafts;

/// One round of several slots' steps as passes of `GpuModel::step_slots`
/// (the seat override of every family whose body has `SlotRows`): [`Session::step_slots_rounds`]
/// over the rows — one a slot, the server's rounds — each row's answer into
/// `next`, each lent logits row filled from the pass's own per-row heads,
/// the NaN guard the fallback loop keeps. Returns the passes it ran.
pub fn step_rows_one_pass<B>(
    s: &mut app::Session<B>,
    rows: &mut [SlotStep],
) -> Result<usize, GateError>
where
    B: SlotRows,
    B::Seq: 'static,
{
    let last: Vec<u32> = rows.iter().map(|r| r.last).collect();
    let round: Vec<(usize, &[u32])> = rows
        .iter()
        .zip(last.chunks(1))
        .map(|(r, one)| (r.slot, one))
        .collect();
    let out = s.step_slots_rounds(&round, rows.iter().any(|r| r.logits.is_some()))?;
    for (r, next) in rows.iter_mut().zip(out.ids) {
        r.next = next;
    }
    if let Some(all) = out.logits {
        for (r, row) in rows.iter_mut().zip(all) {
            if let Some(dst) = r.logits.as_deref_mut() {
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
/// Rows are laid in slot order; a body may require it (body38's `slot_rows`
/// holds slot 0's rows from row 0). The rows come in the order their
/// requests arrived, and each row's kept ids and counts go back to that
/// row, never to the row at its pass position; each slot's rows are bit for
/// bit its own wherever they sit in the pass. A round whose windows' rows
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
    // Rows are laid in slot order; a body may require it (body38's
    // `slot_rows`). `on` keeps each window's row index, the key its answer
    // goes back by.
    on.sort_by_key(|&i| rows[i].slot);
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
