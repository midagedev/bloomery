//! The serve seats' one owner of a round run as passes: a seat whose body
//! has [`SlotRows`] and decided at its open to run one pass overrides
//! [`Seat::step_slots`] with [`step_rows_one_pass`]; any other seat keeps
//! the default, [`bind::step_rows_in_turn`].
//!
//! [`Seat::step_slots`]: bloomery_gpu_gates::bind::Seat::step_slots
//! [`bind::step_rows_in_turn`]: bloomery_gpu_gates::bind::step_rows_in_turn

use bloomery_gpu::model::SlotRows;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::SlotStep;

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
