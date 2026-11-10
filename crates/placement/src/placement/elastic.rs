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
//! [`FILL_HOLD`] quiet boundaries (a sequence's end) after a give-back before it runs, so a card that
//! oscillates between the two settles instead of thrashing.

/// Quanta of free bytes above the margin a fill leaves untouched.
pub const FILL_KEEP_QUANTA: u64 = 2;

/// Quiet boundaries (a sequence's end) after a give-back before a fill may run.
pub const FILL_HOLD: u64 = 16;

/// Quiet boundaries between two looks at the card for a fill: a settle at
/// boundary `b` is followed by the next look at `b + FILL_HOLD` at the
/// earliest, so a request-by-request server reads the card once per hold and
/// not at every request's end.
pub const FILL_LOOK: u64 = FILL_HOLD;

/// The tenant's moves since the load, and the boundary clock its hold counts
/// in: a *quiet boundary* is a sequence's end (the seat ticks it,
/// [`ElasticStats::tick`]), where the K/V rooms have given their pages past
/// the floor back and no allocation is in flight. Every give-back, whoever
/// ran it (a settle, a retried allocation, a K/V fit), stamps the clock
/// through [`ElasticStats::gave_back`], so the fill's hold has one owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticStats {
    pub give_backs: u64,
    pub give_back_bytes: u64,
    pub fills: u64,
    pub fill_bytes: u64,
    /// Quiet boundaries since the load.
    pub boundary: u64,
    /// The boundary the last give-back ran at, for the fill's hold.
    pub last_give_back: Option<u64>,
    /// The boundary the card was last read for a settle at.
    pub last_settle: Option<u64>,
}

impl ElasticStats {
    /// A quiet boundary passed; the new boundary.
    pub fn tick(&mut self) -> u64 {
        self.boundary = self.boundary.saturating_add(1);
        self.boundary
    }

    /// A give-back of `bytes` ran now.
    pub fn gave_back(&mut self, bytes: u64) {
        self.give_backs += 1;
        self.give_back_bytes = self.give_back_bytes.saturating_add(bytes);
        self.last_give_back = Some(self.boundary);
    }

    /// A fill that backed `bytes` ran now.
    pub fn filled(&mut self, bytes: u64) {
        self.fills += 1;
        self.fill_bytes = self.fill_bytes.saturating_add(bytes);
    }

    /// The card was read for a settle now.
    pub fn settled(&mut self) {
        self.last_settle = Some(self.boundary);
    }

    /// Whether a quiet boundary should read the card for a fill: a layer has
    /// a `withdrawn` slot to fill, the hold after the last give-back is over
    /// ([`fill_allowed`]) and the card was not read within [`FILL_LOOK`]
    /// boundaries. False costs nothing: no stream sync, no card read.
    #[must_use]
    pub fn fill_due(&self, withdrawn: usize) -> bool {
        withdrawn > 0
            && fill_allowed(self.boundary, self.last_give_back)
            && self
                .last_settle
                .is_none_or(|at| self.boundary.saturating_sub(at) >= FILL_LOOK)
    }
}

/// The place bound the card kernels read for a layer whose stacks hold
/// `slots` slots and whose map has `live` experts on the card: the stacks'
/// slots, not the live count. The live count moves with a give-back, a fill
/// and every promotion into a spare slot, while a captured step holds the
/// bound as a launch argument and a route table is sized by it; the slots
/// are fixed at the load, every place the map can name is below them, and a
/// slot no expert holds is named by no place. `None` when the stacks hold
/// fewer slots than the live experts (a load the leg refuses by name).
#[must_use]
pub fn reach(live: usize, slots: usize) -> Option<usize> {
    (slots >= live).then_some(slots)
}

/// The experts a layer's seed list ranks past its live experts, so the rule's card set bound can
/// grow with a fill (`SwapRule::give` refuses a bound past the experts the seed ranks): every
/// expert of the layer's `n_expert` that is neither one of the `live` experts the seed names nor
/// on a tier card (`tier`), in id order. A fill's slots take them after the seed's own, so the
/// first request sees a filled card, and the residency's heat reorders them from there. Empty
/// when nothing is reserved past the live experts, so a load that reserves nothing keeps its
/// seed list as it is.
#[must_use]
pub fn seed_tail(n_expert: usize, live: &[u32], tier: &[u32], reserved: usize) -> Vec<u32> {
    if reserved == 0 {
        return Vec::new();
    }
    let mut named = vec![false; n_expert];
    for &id in live.iter().chain(tier) {
        if let Some(n) = named.get_mut(id as usize) {
            *n = true;
        }
    }
    (0..n_expert as u32)
        .filter(|&id| !named[id as usize])
        .collect()
}

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
/// dense, expert, rounding, K/V, stage-row and prompt-pass bytes are exact
/// sums and never come back.
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
            prompt_bytes: 0,
            ring_bytes: 0,
            ring_half: 0,
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

    /// The place bound is the stacks' slots whatever the live count does: after
    /// a fill and the promotions into the filled slots (live 94 → 117 of 117
    /// slots) every place a map can name is below it, and after a give-back
    /// (live 94 → 70) it does not move, so a captured step's argument and a
    /// route table sized by it stay right. Stacks with fewer slots than live
    /// experts have no bound.
    ///
    /// FAIL-first: a `reach` that returns the live count reds the first loop
    /// at live 70 (it reads 70, not 117).
    #[test]
    fn the_place_bound_is_the_slots_and_covers_every_filled_place() {
        let slots = 117;
        for live in [70, 94, 100, 117] {
            assert_eq!(reach(live, slots), Some(slots), "live {live}");
        }
        let live = 94;
        for filled in 0..=(slots - live) {
            let highest = live + filled - 1;
            assert!(highest < reach(live, slots).unwrap(), "{filled} filled");
        }
        assert_eq!(reach(118, 117), None);
        assert_eq!(reach(0, 0), Some(0));
    }

    /// The seed list past the live experts names the experts a fill's slots take: none when
    /// nothing is reserved, else every expert neither live nor on a tier card, in id order.
    ///
    /// FAIL-first: a `seed_tail` that forgets the tier experts lists 3 and the rule refuses
    /// it (an away expert in the seed); one that returns nothing leaves the rule unable to
    /// fill past its spares, which the runtime's seed test shows.
    #[test]
    fn the_seed_tail_is_every_expert_neither_live_nor_on_a_tier_card() {
        assert_eq!(seed_tail(8, &[5, 0, 2], &[3], 4), [1, 4, 6, 7]);
        assert!(seed_tail(8, &[5, 0, 2], &[3], 0).is_empty());
        assert_eq!(seed_tail(3, &[0, 1, 2], &[], 2), Vec::<u32>::new());
    }

    /// A give-back starts the fill's hold whoever ran it, on the quiet-boundary
    /// clock: the fill is due [`FILL_HOLD`] boundaries after it and not before.
    /// This is the sequence the serving gate holds: a give-back under K/V
    /// growth, the K/V shrunk at the sequence's end, quiet boundaries, then
    /// the fill.
    ///
    /// FAIL-first: a `gave_back` that leaves `last_give_back` unset (the
    /// K/V fit's give-back stamped nothing) makes the first assertion in the
    /// loop red at boundary 3.
    #[test]
    fn a_give_back_holds_the_fill_for_the_hold_on_the_quiet_clock() {
        let mut s = ElasticStats::default();
        for _ in 0..3 {
            s.tick();
        }
        s.settled();
        s.gave_back(4_096);
        assert_eq!((s.give_backs, s.give_back_bytes), (1, 4_096));
        assert_eq!(s.last_give_back, Some(3));
        for _ in 0..FILL_HOLD - 1 {
            s.tick();
            assert!(!s.fill_due(5), "boundary {} is inside the hold", s.boundary);
        }
        s.tick();
        assert_eq!(s.boundary, 3 + FILL_HOLD);
        assert!(s.fill_due(5));
    }

    /// A quiet boundary reads the card only when a fill could run: nothing
    /// withdrawn, a look within [`FILL_LOOK`] boundaries before, or the hold
    /// still on is no read. A fill counts its bytes and a later give-back
    /// restarts the hold.
    ///
    /// FAIL-first: a `fill_due` that ignores `withdrawn` reds the first
    /// assertion; one that ignores `last_settle` reds the second.
    #[test]
    fn a_quiet_boundary_reads_the_card_only_when_a_fill_could_run() {
        let mut s = ElasticStats::default();
        assert!(!s.fill_due(0), "nothing withdrawn: nothing to fill");
        assert!(s.fill_due(1));
        s.settled();
        s.tick();
        assert!(!s.fill_due(1), "the card was read a boundary ago");
        for _ in 0..FILL_LOOK {
            s.tick();
        }
        assert!(s.fill_due(1));
        s.filled(1_000);
        s.filled(24);
        assert_eq!((s.fills, s.fill_bytes), (2, 1_024));
        s.gave_back(1);
        assert!(!s.fill_due(1), "a give-back restarts the hold");
    }
}
