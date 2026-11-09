//! The elastic expert tenant's arithmetic, pure: how many withdrawn slots a
//! layer may fill from the card's free bytes, and whether a fill may run at
//! this boundary. The residency machine (`bloomery_gpu::host::swap`) gives
//! slots back when a later allocation does not fit and fills them again when
//! the card has room; this module decides the counts, nothing here touches a
//! device.
//!
//! A quantum is one slot on every layer the machine holds stacks of: the
//! smallest give-back, and the unit a fill is counted in. A fill never takes
//! the last [`FILL_KEEP_QUANTA`] quanta above the margin, so a request's page
//! growth within them never evicts what was just filled, and a fill waits
//! [`FILL_HOLD`] boundaries after a give-back before it runs, so a card that
//! oscillates between the two settles instead of thrashing.

/// Quanta of free bytes above the margin a fill leaves untouched.
pub const FILL_KEEP_QUANTA: u64 = 2;

/// Residency boundaries after a give-back before a fill may run.
pub const FILL_HOLD: u64 = 16;

/// One slot on each of `layers` layers: `layers × slot_bytes`, saturating.
#[must_use]
pub fn quantum(layers: u64, slot_bytes: u64) -> u64 {
    layers.saturating_mul(slot_bytes)
}

/// The slots a layer may fill now: the largest `k ≤ withdrawn` whose
/// `k` quanta fit in `free − margin − FILL_KEEP_QUANTA × quantum`; 0 when
/// nothing fits, when the margin is not met, or when a quantum is 0 (no layer
/// or no stack bytes: the machine fills nothing and says so).
#[must_use]
pub fn fill_slots(free: u64, margin: u64, layers: u64, slot_bytes: u64, withdrawn: usize) -> usize {
    let q = quantum(layers, slot_bytes);
    if q == 0 {
        return 0;
    }
    let keep = margin.saturating_add(FILL_KEEP_QUANTA.saturating_mul(q));
    let room = free.saturating_sub(keep);
    usize::try_from(room / q).map_or(withdrawn, |k| k.min(withdrawn))
}

/// Whether a fill may run at boundary `boundary`: no give-back yet, or the
/// last one ([`Some`]`(at)`) at least [`FILL_HOLD`] boundaries ago.
#[must_use]
pub fn fill_allowed(boundary: u64, last_give_back: Option<u64>) -> bool {
    last_give_back.is_none_or(|at| boundary.saturating_sub(at) >= FILL_HOLD)
}

/// The most card bytes a fill could ever take back from a plan's card: the
/// terms the plan assumed rather than summed from the file (its context,
/// scratch and reserve bytes) and the headroom it left past the margin. The
/// dense, expert, rounding, K/V and stage-row bytes are exact sums and never
/// come back.
#[must_use]
pub fn reclaimable(totals: &super::CardTotals, margin: u64) -> u64 {
    let past_margin = totals
        .headroom_bytes
        .saturating_sub(i128::from(margin))
        .max(0);
    totals
        .context_bytes
        .saturating_add(totals.scratch_bytes)
        .saturating_add(totals.reserve_bytes)
        .saturating_add(u64::try_from(past_margin).unwrap_or(u64::MAX))
}

/// The slots a layer's card stacks reserve on an elastic load: the plan's
/// `live` experts and the slots `reclaimable` bytes fill at one `quantum`
/// each, at most the model's `n_expert`; `live` when the quantum is 0.
#[must_use]
pub fn capacity(live: u64, reclaimable: u64, quantum: u64, n_expert: u64) -> u64 {
    let extra = reclaimable.checked_div(quantum).unwrap_or(0);
    live.saturating_add(extra).min(n_expert)
}

/// The rows of an expert stack reserved for `slots` experts, each expert
/// holding `rows_per_expert` rows (`dims[1]` of the stack tensor): a slot is
/// that many rows, and an upload counts rows. `None` when the product passes
/// `usize`.
#[must_use]
pub fn stack_rows(slots: usize, rows_per_expert: usize) -> Option<usize> {
    slots.checked_mul(rows_per_expert)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capacity is counted in slots and an upload in rows: 117 slots of a
    /// 2560-row expert are 299 520 rows, which hold the 240 640 rows of 94
    /// live experts; read as rows, 117 is below them.
    ///
    /// FAIL-first: a `stack_rows` that returns `slots` goes red on the first
    /// assertion, and the second shows the unit slip the load refused by.
    #[test]
    fn a_capacity_in_slots_is_rows_per_expert_times_slots() {
        let (per, live, slots) = (2560, 94, 117);
        assert_eq!(stack_rows(slots, per), Some(299_520));
        assert!(stack_rows(slots, per).unwrap() >= live * per);
        assert!(slots < live * per);
        assert_eq!(stack_rows(usize::MAX, 2), None);
    }

    /// The fill keeps the margin and two quanta: with 10 quanta of free bytes
    /// above a margin, 8 slots fill, never 9.
    ///
    /// FAIL-first: a `fill_slots` that forgets `FILL_KEEP_QUANTA` returns 10 and the
    /// first assertion goes red; one that forgets the margin returns 9 on the second.
    #[test]
    fn a_fill_keeps_the_margin_and_two_quanta() {
        let (layers, slot) = (4, 100);
        let q = quantum(layers, slot);
        assert_eq!(q, 400);
        assert_eq!(fill_slots(1_000 + 10 * q, 1_000, layers, slot, 64), 8);
        assert_eq!(fill_slots(10 * q, 1_000, layers, slot, 64), 5);
        assert_eq!(fill_slots(1_000 + 3 * q, 1_000, layers, slot, 64), 1);
        assert_eq!(fill_slots(1_000 + 2 * q, 1_000, layers, slot, 64), 0);
    }

    /// A fill never passes the slots withdrawn, and a zero quantum fills nothing.
    #[test]
    fn a_fill_is_bounded_by_the_withdrawn_slots_and_a_zero_quantum() {
        assert_eq!(fill_slots(u64::MAX, 0, 4, 100, 3), 3);
        assert_eq!(fill_slots(u64::MAX, 0, 0, 100, 3), 0);
        assert_eq!(fill_slots(u64::MAX, 0, 4, 0, 3), 0);
    }

    /// The reclaimable bytes are the assumed terms and the headroom past the
    /// margin, never a negative headroom; the capacity adds their quanta to the
    /// live count and stops at the model's experts.
    ///
    /// FAIL-first: a `reclaimable` that adds the whole headroom returns 1_300 on
    /// the first case; a `capacity` that forgets the cap returns 12 on the last.
    #[test]
    fn the_capacity_adds_the_reclaimable_quanta_to_the_live_count() {
        let totals = |headroom: i128| super::super::CardTotals {
            dense_bytes: 0,
            expert_bytes: 0,
            rounding_bytes: 0,
            experts: 0,
            kv_bytes: 0,
            stage_rows_bytes: 0,
            scratch_bytes: 100,
            context_bytes: 200,
            reserve_bytes: 300,
            headroom_bytes: headroom,
        };
        assert_eq!(reclaimable(&totals(700), 400), 900);
        assert_eq!(reclaimable(&totals(300), 400), 600);
        assert_eq!(reclaimable(&totals(-50), 400), 600);
        assert_eq!(capacity(4, 900, 400, 16), 6);
        assert_eq!(capacity(4, 900, 0, 16), 4);
        assert_eq!(capacity(10, 900, 400, 11), 11);
    }

    /// A fill waits `FILL_HOLD` boundaries after a give-back.
    ///
    /// FAIL-first: a `fill_allowed` that reads `>` for `>=` goes red on the boundary case.
    #[test]
    fn a_fill_waits_after_a_give_back() {
        assert!(fill_allowed(0, None));
        assert!(!fill_allowed(10, Some(10)));
        assert!(!fill_allowed(10 + FILL_HOLD - 1, Some(10)));
        assert!(fill_allowed(10 + FILL_HOLD, Some(10)));
    }
}
