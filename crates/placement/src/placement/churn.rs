//! Adaptive residency's host share. The residency machine may send any stage
//! card expert past a layer's first `pinned` back to the host: that is the
//! churn pool, each layer's card segment's experts from rank `pinned` on (the
//! list's order, which is slot order). Where the pool is hosted, a load reads
//! it in and locks it in its host set (`HostSet::of_known`, `bloomery-model`'s
//! `placement::host_lock`) beside the plan's host segments, the plan's host
//! headroom ([`super::HostTotals::headroom_bytes`], the one budget) takes its
//! bytes, and the host serves a victim from resident pages. Where an unset
//! word kept its P because the pool fit neither the headroom nor what
//! `MemAvailable` leaves, the load's residency spec says unhosted
//! (`bloomery_levers::ChurnHosting`) and the set only knows the pool: the
//! host serves a victim from the file's mapping, read back at a page fault's
//! cost and counted in the machine's `rereads` and `unresident` sends —
//! measured on a 64 GB host at no decode cost against the P a hosted pool
//! would have raised. A stack the plan pages to the NVMe tier adds nothing:
//! its victims are the tier's arena's ([`super::expert_nvme_tier`]).

use super::{Device, ExpertList, PlacementError, Plan, Role, per_expert};

/// The churn pool of one stage card of a plan.
#[derive(Clone, Debug)]
pub struct ChurnPool {
    /// The seed experts of each layer that stay on the card.
    pub pinned: usize,
    /// Per routed stack the card holds: its index in the plan's model
    /// tensors, and its experts from rank `pinned` on.
    pub runs: Vec<(usize, ExpertList)>,
    /// Experts in the pool, summed over layers.
    pub experts: u64,
    /// File bytes of the pool: every stack of every expert in it.
    pub bytes: u64,
}

impl ChurnPool {
    /// The pool of `plan`'s card `card` (an index of
    /// [`super::Machine::all_cards`]) with `pinned` seed experts a layer kept
    /// on the card. A layer whose card holds `pinned` experts or fewer adds
    /// nothing, and neither does a stack with a segment on the NVMe tier.
    pub fn of(plan: &Plan<'_>, card: usize, pinned: usize) -> Result<ChurnPool, PlacementError> {
        let model = plan.model;
        let mut per_layer = vec![0u64; model.layers];
        let mut pool = ChurnPool {
            pinned,
            runs: Vec::new(),
            experts: 0,
            bytes: 0,
        };
        for row in &plan.rows {
            let t = model.tensors.get(row.tensor).ok_or_else(|| {
                PlacementError::Experts(format!("plan row {} names no tensor", row.tensor))
            })?;
            // A paged stack's victims are the NVMe tier's arena's, not the
            // host's pages.
            if t.role != Role::RoutedExperts
                || row.segments.iter().any(|s| s.device == Device::Nvme)
            {
                continue;
            }
            for seg in &row.segments {
                let Some(list) = seg
                    .experts
                    .as_ref()
                    .filter(|_| seg.device == Device::Card(card))
                else {
                    continue;
                };
                let Some(rest) = list.ids().get(pinned..).filter(|r| !r.is_empty()) else {
                    continue;
                };
                let (_, per) = per_expert(t, model.experts)?;
                let n = rest.len() as u64;
                pool.bytes += n * per;
                if let Some(slot) = t.layer.and_then(|l| per_layer.get_mut(l)) {
                    *slot = (*slot).max(n);
                }
                pool.runs
                    .push((row.tensor, ExpertList::new(rest.to_vec(), model.experts)?));
            }
        }
        pool.experts = per_layer.iter().sum();
        Ok(pool)
    }

    /// The host room a churn pool must fit on `plan`: its host headroom less the `beside` bytes
    /// the load holds outside the plan. The one definition the unset residency word's pick
    /// (`PlanRoom::headroom`) and the load's check ([`ChurnPool::check_beside`]) both read.
    #[must_use]
    pub fn room(plan: &Plan<'_>, beside: u64) -> i128 {
        plan.host.headroom_bytes - i128::from(beside)
    }

    /// The plan's host headroom less the pool.
    #[must_use]
    pub fn headroom_after(&self, plan: &Plan<'_>) -> i128 {
        plan.host.headroom_bytes - i128::from(self.bytes)
    }

    /// The pool fits the plan's host headroom; refused by name when it does
    /// not ([`PlacementError::ResidencyOverHost`]). Returns the headroom left.
    pub fn check(&self, plan: &Plan<'_>) -> Result<i128, PlacementError> {
        self.check_beside(plan, 0)
    }

    /// [`ChurnPool::check`] for a load whose host set also holds `beside`
    /// bytes the plan does not count (a draft layer's experts the load hosts
    /// beside the plan's own): the pool fits the plan's host headroom less
    /// them, refused by name when it does not. Returns the headroom left.
    pub fn check_beside(&self, plan: &Plan<'_>, beside: u64) -> Result<i128, PlacementError> {
        self.check_taking(plan, self.experts, self.bytes, beside)
    }

    /// [`ChurnPool::check_beside`] for a load whose host set only knows the
    /// pool, reading none of it in: `beside` alone must fit the plan's host
    /// headroom, refused by name as the pool's check is, the pool taking 0
    /// bytes. Returns the headroom left.
    pub fn check_known_beside(&self, plan: &Plan<'_>, beside: u64) -> Result<i128, PlacementError> {
        self.check_taking(plan, 0, 0, beside)
    }

    /// The plan's host headroom less `bytes` of `experts` the host set takes
    /// of the pool and `beside`; refused by name below 0.
    fn check_taking(
        &self,
        plan: &Plan<'_>,
        experts: u64,
        bytes: u64,
        beside: u64,
    ) -> Result<i128, PlacementError> {
        let left = ChurnPool::room(plan, beside) - i128::from(bytes);
        if left < 0 {
            return Err(PlacementError::ResidencyOverHost(Box::new(OverHost {
                pinned: self.pinned,
                experts,
                bytes,
                headroom: plan.host.headroom_bytes,
                beside,
                room: plan.host.room.as_ref().map(|(read, _)| *read),
            })));
        }
        Ok(left)
    }
}

/// A churn pool the plan's host headroom cannot take
/// ([`PlacementError::ResidencyOverHost`]).
#[derive(Debug, thiserror::Error)]
#[error(
    "residency pinning {pinned} experts a layer: the churn pool's {experts} experts take {bytes} B \
     of host RAM, and the plan leaves the host {headroom} B{}{}",
    beside_text(*beside),
    room_text(*room)
)]
pub struct OverHost {
    pub pinned: usize,
    pub experts: u64,
    pub bytes: u64,
    pub headroom: i128,
    /// Host bytes the load holds beside the plan's own
    /// ([`ChurnPool::check_beside`]), out of `headroom`.
    pub beside: u64,
    /// The host room the plan was made at, as read or given
    /// ([`HostTotals::room`](super::HostTotals::room)): the `room` of the
    /// `plan` record. A headroom past the figure follows it, so a refusal
    /// that differs from the pick's names the reading that moved.
    pub room: Option<u64>,
}

/// [`OverHost`]'s words for the bytes held beside the plan: none when 0.
fn beside_text(beside: u64) -> String {
    match beside {
        0 => String::new(),
        b => format!(", {b} B of it held for the layers the load hosts beside the plan"),
    }
}

/// [`OverHost`]'s words for the room the plan was made at: none when it was made at none.
fn room_text(room: Option<u64>) -> String {
    match room {
        None => String::new(),
        Some(r) => format!("; the plan was made at a host room of {r} B"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::workstation::{HOST_DRIFT, HostNeed, HostRead, load_plan_room};
    use crate::placement::{Host, HostTotals, Machine, ModelTensors, Node};
    use bloomery_levers::KvHost;

    /// A plan whose host leaves `headroom` bytes, and nothing else: the
    /// check reads the headroom alone.
    fn with_headroom<'a>(
        model: &'a ModelTensors,
        machine: &'a Machine,
        headroom: i128,
    ) -> Plan<'a> {
        Plan {
            model,
            machine,
            ctx_max: 1,
            rows: Vec::new(),
            cards: Vec::new(),
            host: HostTotals {
                expert_bytes: 0,
                experts: 0,
                nvme_expert_bytes: 0,
                nvme_arena_bytes: 0,
                arena_columns: 1,
                table_bytes: 0,
                shadow_bytes: 0,
                kv_host_bytes: 0,
                stage_kv_bytes: 0,
                reserve_bytes: 0,
                row_reserve_bytes: 0,
                pool_card_bytes: 0,
                headroom_bytes: headroom,
                room: None,
                grow: crate::placement::HostGrow::default(),
            },
            nvme_bytes: 0,
            n_l: Vec::new(),
            tier_n_l: Vec::new(),
            card_budget: crate::placement::card_budget::CardBudget::Unset,
            budget_from_pool: false,
            expert_rank: None,
            kv_floor: None,
            kv_host: KvHost::default(),
            card_experts_cap: None,
        }
    }

    /// An unsplit plan whose host terms are 300 B, made at `room` on `machine`: its headroom is
    /// what [`Plan::plan_at`] leaves it, the machine's figure's less the terms' where a read room
    /// is no more than the figure, the room's less the terms' where it is stated or past it.
    fn plan_made_at<'a>(
        model: &'a ModelTensors,
        machine: &'a Machine,
        room: (u64, HostRead),
    ) -> Plan<'a> {
        plan_with_experts(model, machine, 300, room)
    }

    /// [`plan_made_at`] for a plan whose host holds `expert_bytes` of routed experts.
    fn plan_with_experts<'a>(
        model: &'a ModelTensors,
        machine: &'a Machine,
        expert_bytes: u64,
        room: (u64, HostRead),
    ) -> Plan<'a> {
        let mut plan = with_headroom(model, machine, 0);
        plan.host.expert_bytes = expert_bytes;
        plan.host.headroom_bytes = i128::from(HostNeed::machine_room(machine.host()))
            - i128::from(HostNeed::of(&plan, 0).bytes());
        plan.plan_at(&room);
        plan
    }

    /// The pool and the bytes held beside the plan fit the headroom exactly;
    /// one byte more of either is refused by name, the refusal naming the
    /// bytes held beside; with none beside, the check is the pool's alone.
    #[test]
    fn the_pool_fits_the_headroom_less_the_bytes_beside() {
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 1,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let machine = Machine {
            cards: Vec::new(),
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: 0,
                    reserves: Vec::new(),
                },
                unified: None,
            }],
        };
        let plan = with_headroom(&model, &machine, 150);
        let pool = ChurnPool {
            pinned: 0,
            runs: Vec::new(),
            experts: 10,
            bytes: 100,
        };
        assert_eq!(pool.check(&plan).expect("the pool alone fits"), 50);
        assert_eq!(pool.check_beside(&plan, 50).expect("exactly fits"), 0);
        let short = pool.check_beside(&plan, 51).expect_err("one byte short");
        let text = short.to_string();
        for part in [
            "100 B of host RAM",
            "leaves the host 150 B",
            "51 B of it held",
        ] {
            assert!(text.contains(part), "{part:?} in {text}");
        }
        let alone = with_headroom(&model, &machine, 99);
        let text = pool
            .check(&alone)
            .expect_err("the pool past it")
            .to_string();
        assert!(!text.contains("held"), "{text}");
    }

    /// A room the caller states binds an unsplit plan's host headroom, the
    /// room less the plan's host need: the need fits a stated room and the
    /// pool beside it does not, so the pool is refused by name with the
    /// room's headroom. A room that was read under the machine's figure
    /// leaves the figure's; a read or stated room past it is the headroom's
    /// own.
    #[test]
    fn a_stated_room_binds_an_unsplit_plans_headroom() {
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 1,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        // The machine's figure is 1000 B and the plan's host terms 300 B:
        // 700 B of headroom with no room named.
        let machine = Machine {
            cards: Vec::new(),
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: 1000,
                    reserves: Vec::new(),
                },
                unified: None,
            }],
        };
        let pool = ChurnPool {
            pinned: 0,
            runs: Vec::new(),
            experts: 10,
            bytes: 100,
        };
        let made_at = |room: (u64, HostRead)| plan_made_at(&model, &machine, room);
        // A stated room of 350 B leaves 50 B past the need: 100 B of pool
        // does not fit, and the refusal names the 50 B.
        let stated = made_at((350, HostRead::Given));
        let text = pool
            .check(&stated)
            .expect_err("the pool past the stated room's headroom")
            .to_string();
        assert!(text.contains("leaves the host 50 B"), "{text}");
        assert_eq!(made_at((400, HostRead::Given)).host.headroom_bytes, 100);
        assert_eq!(
            pool.check(&made_at((400, HostRead::Given))).expect("fits"),
            0
        );
        assert!(pool.check(&made_at((399, HostRead::Given))).is_err());
        // A read room under the machine's figure leaves it.
        let read = made_at((350, HostRead::MemAvailable));
        assert_eq!(read.host.headroom_bytes, 700);
        assert_eq!(pool.check(&read).expect("the machine's figure"), 600);
        // PIN(2026-11-17): replaces "a stated room past the machine's figure
        // leaves it" (`(u64::MAX, Given)` read 700). The figure is the nominal
        // workstation's, not a limit of the machine ([`Plan::plan_at`]): a
        // stated or read room past it is the plan's room. Derivation: 2,000 B
        // given less the 300 B of host terms is 1,700 B of headroom, and the
        // 100 B pool leaves 1,600 B; a read 1,500 B + the drift is 1,500 B of
        // room, 1,200 B of headroom, 1,100 B left.
        let past = made_at((2000, HostRead::Given));
        assert_eq!(past.host.headroom_bytes, 1700);
        assert_eq!(pool.check(&past).expect("the stated room"), 1600);
        let read_past = made_at((1500 + HOST_DRIFT, HostRead::MemAvailable));
        assert_eq!(read_past.host.headroom_bytes, 1200);
        assert_eq!(pool.check(&read_past).expect("the read room"), 1100);
    }

    /// The room that leaves a pool one byte short of the headroom is the plan's host terms and
    /// the pool less one byte, wherever the plan stands: made at a read room under the machine's
    /// figure, at a read one past it, or at a stated one past it, the plan made at that room as
    /// given refuses the pool by name with 99 B, and one byte more room takes it with none to
    /// spare. A residency gate's `refuse` clause derived this room from the machine's figure
    /// (`figure - headroom + pool - 1`), which is right only where the figure bound the room.
    ///
    /// FAIL-first: that derivation on the plan made at a stated 2,000 B reads 1,000 - 1,700 + 99
    /// = -601, and on the one at a read 1,500 B 1,000 - 1,200 + 99 = -101, where the room is 399.
    // PIN(2026-11-18): the residency gates' `refuse` clauses (glm5next, qwen38, V4.1) lowered the
    // machine's usable bytes for the short host, which no longer binds a read room past the
    // figure (`Plan::plan_at`); the short host is now a stated room.
    #[test]
    fn the_room_one_byte_short_of_a_pool_is_the_terms_and_the_pool_less_one() {
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 1,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let machine = Machine {
            cards: Vec::new(),
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: 1000,
                    reserves: Vec::new(),
                },
                unified: None,
            }],
        };
        let pool = ChurnPool {
            pinned: 0,
            runs: Vec::new(),
            experts: 10,
            bytes: 100,
        };
        let figure = i128::from(HostNeed::machine_room(machine.host()));
        for (room, by_figure_holds) in [
            ((350, HostRead::MemAvailable), true),
            ((1500 + HOST_DRIFT, HostRead::MemAvailable), false),
            ((2000, HostRead::Given), false),
        ] {
            let base = plan_made_at(&model, &machine, room);
            let short = base
                .room_leaving(i128::from(pool.bytes) - 1)
                .expect("a room of the terms and 99 B");
            assert_eq!(short, 399, "the 300 B of terms and the pool less one");
            let at = plan_made_at(&model, &machine, (short, HostRead::Given));
            assert_eq!(at.host.headroom_bytes, 99);
            let text = pool
                .check(&at)
                .expect_err("one byte short of the pool")
                .to_string();
            assert!(text.contains("leaves the host 99 B"), "{text}");
            let fits = plan_made_at(&model, &machine, (short + 1, HostRead::Given));
            assert_eq!(pool.check(&fits).expect("the pool fits"), 0);
            let by_figure = figure - base.host.headroom_bytes + i128::from(pool.bytes) - 1;
            assert_eq!(
                by_figure == i128::from(short),
                by_figure_holds,
                "{by_figure}"
            );
        }
        let base = plan_made_at(&model, &machine, (2000, HostRead::Given));
        assert_eq!(base.room_leaving(-300), Some(0));
        assert_eq!(base.room_leaving(-301), None, "below 0 is no room");
    }

    /// V4.1's host on the 251.5 GiB workstation: the model's 40 layers, the machine at the
    /// nominal host's figure and the plan's host terms, 243,724,957,696 B past the OS reserve
    /// the need leaves out.
    fn v41_host() -> (ModelTensors, Machine, u64) {
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 40,
            experts: 256,
            experts_used: 6,
            bf16_words: |_| false,
        };
        let machine = Machine {
            cards: Vec::new(),
            tiers: Vec::new(),
            nodes: vec![Node {
                host: crate::placement::workstation::host(),
                unified: None,
            }],
        };
        assert_eq!(
            HostNeed::machine_room(machine.host()),
            crate::placement::workstation::PLAN_ROOM
        );
        (
            model,
            machine,
            243_724_957_696 + crate::placement::workstation::OS_OTHER,
        )
    }

    /// The read room of the plan the pick ruled by: `MemAvailable` 264,214,507,520 B, a plan
    /// room of 263,677,636,608 B, past the machine's figure of 263,376,371,712 B, so the
    /// headroom follows the reading: 19,952,678,912 B.
    const PICK_READ: u64 = 264_214_507_520;

    /// What a load's reading fell by after the pick's plan.
    const FALL: u64 = 83_034_112;

    /// A plan made at a lower reading of the host's room refuses the pool the first took, and
    /// the refusal names the room it was made at. The 19,914,301,440 B pool of `mid-p8-s1` fits
    /// the pick's headroom with 38,377,472 B to spare; a plan made 83,034,112 B lower leaves
    /// 19,869,644,800 B.
    ///
    /// FAIL-first: the refusal named no room, so it read as another headroom than the pick's
    /// with nothing to say which reading had moved.
    #[test]
    fn a_plan_made_at_a_lower_reading_refuses_the_pool_the_first_took() {
        let (model, machine, experts) = v41_host();
        let pool = ChurnPool {
            pinned: 8,
            runs: Vec::new(),
            experts: 1182,
            bytes: 19_914_301_440,
        };
        let pick = plan_with_experts(
            &model,
            &machine,
            experts,
            (PICK_READ, HostRead::MemAvailable),
        );
        assert_eq!(pick.host.headroom_bytes, 19_952_678_912);
        assert_eq!(pool.check(&pick).expect("the pick's plan"), 38_377_472);

        let load = plan_with_experts(
            &model,
            &machine,
            experts,
            (PICK_READ - FALL, HostRead::MemAvailable),
        );
        assert_eq!(load.host.headroom_bytes, 19_869_644_800);
        let text = pool
            .check(&load)
            .expect_err("the lower reading")
            .to_string();
        for part in [
            "residency pinning 8 experts a layer",
            "1182 experts take 19914301440 B of host RAM",
            "the plan leaves the host 19869644800 B",
            "the plan was made at a host room of 264131473408 B",
        ] {
            assert!(text.contains(part), "{part:?} in {text}");
        }

        // A plan made at no room names none.
        let bare = with_headroom(&model, &machine, 19_869_644_800);
        let text = pool.check(&bare).expect_err("no room").to_string();
        assert!(!text.contains("made at"), "{text}");
    }

    /// The pick and the load hold a churn pool to one room: the unset word's pick, over the
    /// plan made at the read room, keeps the first pinned count whose pool fits
    /// ([`ChurnPool::room`] and what the host leaves), and the load's check passes it on the plan
    /// the load makes at the room it states ([`load_plan_room`]) however far the reading has
    /// fallen since. A load that reads a room of its own is refused.
    ///
    /// FAIL-first: a load that takes the reading of its own over the one stated plans 83,034,112
    /// B lower, leaves 19,869,644,800 B and refuses the 19,914,300,540 B pool the pick kept, as
    /// the V4.1 serve start did.
    #[test]
    fn the_pick_and_the_load_hold_the_pool_to_one_room() {
        let (model, machine, experts) = v41_host();
        // The pool at a pinned count: the 1,502 card experts of 40 layers less the pinned ones.
        let pool_at = |pinned: usize| (1502 - 40 * pinned as u64) * 16_847_970;
        let pick_room = (PICK_READ, HostRead::MemAvailable);
        let pick = plan_with_experts(&model, &machine, experts, pick_room.clone());
        let to = match bloomery_levers::room_for::<std::convert::Infallible>(
            false,
            0,
            1,
            (0..40).map(|l| if l < 22 { 38 } else { 37 }),
            |pinned| Ok(pool_at(pinned)),
            ChurnPool::room(&pick, 0),
            HostNeed::of(&pick, 0).room_left(&pick_room),
        ) {
            Ok(bloomery_levers::Room::MovedForPool { to, .. }) => to,
            other => panic!("the climb to the first pool that fits: {other:?}"),
        };
        assert_eq!(to, 8);
        let pool = ChurnPool {
            pinned: to,
            runs: Vec::new(),
            experts: 1502 - 40 * to as u64,
            bytes: pool_at(to),
        };
        assert_eq!(pool.check(&pick).expect("the pick's own plan"), 38_378_372);

        let fell = (PICK_READ - FALL, HostRead::MemAvailable);
        let own = load_plan_room(None, fell.clone());
        let load = plan_with_experts(&model, &machine, experts, own);
        assert!(pool.check(&load).is_err(), "a room of the load's own");
        let stated = load_plan_room(Some(&pick_room), fell);
        let load = plan_with_experts(&model, &machine, experts, stated);
        assert_eq!(
            pool.check(&load).expect("the stated room"),
            38_378_372,
            "the pick's headroom"
        );
    }
}
