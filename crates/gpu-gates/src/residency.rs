//! What a family's unset-residency rule reads of the plan its load runs by,
//! assembled once ([`PlanRoom`]), and the pair rule's call on it ([`at_plan`]:
//! V4.1's serving word, the one every family that takes
//! [`bloomery_levers::residency_at_plan`] resolves against its plan). The
//! families' rules differ by clause and record (`residency38`, the GLM seat,
//! the V4.1 and MiMo seats print their own whys); the inputs they read of the
//! plan are these five: the card experts a layer, the experts the plan leaves
//! on the host, the churn pool's bytes at a pinned count, the host headroom and
//! what the host's room leaves past the plan's host need (the reading less the
//! drift margin, [`PlanRoom::mem_left`]).

use crate::GateError;
use crate::record::ClimbRead;
use bloomery_levers::{ResidencyPick, UnhostedPool, residency_at_plan};
use model::placement::churn::ChurnPool;
use model::placement::workstation::{HostNeed, HostRead, host_available_read};
use model::placement::{PlacementError, Plan};

/// `plan`'s room for the churn pool of card `card`'s experts, with `beside`
/// bytes the load hosts outside the plan (a draft layer's experts; 0 for a
/// plain load): every term an unset residency rule reads of a plan.
pub struct PlanRoom<'p, 'm> {
    plan: &'p Plan<'m>,
    card: usize,
    beside: u64,
}

impl<'p, 'm> PlanRoom<'p, 'm> {
    /// The room of `plan` for the churn pool of card `card`, `beside` bytes
    /// hosted outside the plan.
    #[must_use]
    pub fn new(plan: &'p Plan<'m>, card: usize, beside: u64) -> PlanRoom<'p, 'm> {
        PlanRoom { plan, card, beside }
    }

    /// The plan's card experts a layer (a layer holding none included).
    pub fn card_experts(&self) -> impl Iterator<Item = u64> + '_ {
        self.plan.n_l.iter().copied()
    }

    /// The routed experts the plan leaves on the host (the `plan` record's
    /// `host_experts`); the experts the load hosts beside the plan are not
    /// counted.
    #[must_use]
    pub fn host_experts(&self) -> u64 {
        self.plan.host.experts
    }

    /// The plan's host headroom less what the load hosts beside the plan.
    #[must_use]
    pub fn headroom(&self) -> i128 {
        self.plan.host.headroom_bytes - i128::from(self.beside)
    }

    /// What the host's room leaves past the plan's host need
    /// ([`HostNeed::bytes`], the load's checkpoints and `beside` inside it),
    /// from a fresh reading ([`host_available_read`]): the term the load's own
    /// check (`HostNeed::check`, the churn pool as its extra bytes) uses, so a
    /// word this keeps passes the load even when the reading falls by
    /// [`HOST_DRIFT`](model::placement::workstation::HOST_DRIFT) or less before
    /// the check ([`PlanRoom::mem_left_at`]).
    pub fn mem_left(&self) -> Result<i128, GateError> {
        Ok(self.climb_read()?.mem_left)
    }

    /// A fresh reading of the host ([`host_available_read`]) and the room
    /// it leaves ([`PlanRoom::mem_left_at`]): what an unset climb budgets
    /// from, kept whole for the `residency unset` record to print.
    pub fn climb_read(&self) -> Result<ClimbRead, GateError> {
        let reading = host_available_read()?;
        let mem_left = self.mem_left_at(&reading);
        Ok(ClimbRead { reading, mem_left })
    }

    /// [`PlanRoom::mem_left`] at a reading a caller took: the room the plan is
    /// made at ([`HostNeed::room_left`], `plan_room`'s one rule) less the
    /// need, so every unset climb keeps the margin off a read room and takes
    /// a given room as given.
    #[must_use]
    pub fn mem_left_at(&self, reading: &(u64, HostRead)) -> i128 {
        HostNeed::of(self.plan, self.beside).room_left(reading)
    }

    /// The churn pool's bytes at `pinned` seed experts a layer: card
    /// `card`'s experts past them ([`ChurnPool::of`]); it shrinks as the
    /// count grows.
    pub fn pool_bytes(&self, pinned: usize) -> Result<u64, PlacementError> {
        ChurnPool::of(self.plan, self.card, pinned).map(|pool| pool.bytes)
    }
}

/// `BLOOMERY_RESIDENCY` against `plan` on the pair rule
/// ([`residency_at_plan`]): `pick` as it is unless it is the serving word
/// the plan has not yet seen, which the plan's card slots, host experts, host
/// headroom and the host's room then keep, move or turn `off` with why, on every body (an
/// unset word never asks for its pool unhosted; `carry` is the loading body's
/// `ChainBody::UNHOSTED_POOL`, a value, so this module reaches no body). The
/// churn pool's bytes are card `card`'s.
pub fn at_plan(
    carry: UnhostedPool,
    plan: &Plan<'_>,
    card: usize,
    pick: ResidencyPick,
) -> Result<ResidencyPick, GateError> {
    let room = PlanRoom::new(plan, card, 0);
    Ok(residency_at_plan(
        carry,
        pick,
        room.card_experts(),
        room.host_experts(),
        |pinned| room.pool_bytes(pinned),
        room.headroom(),
        room.mem_left()?,
    )
    .map_err(|e| format!("BLOOMERY_RESIDENCY unset: the churn pool: {e}"))?)
}
