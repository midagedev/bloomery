//! The context a load serves, from the user's slots and context: the ask
//! comes first, and the plan fills the card around it.
//!
//! A set `--ctx` is the total over the slots, as llama-server reads `-c`;
//! each slot takes its floor share. The plan is made at that share and the
//! load either holds it or stops by name, with the largest share the card
//! would hold ([`AskError::DoesNotFit`]). It is never cut.
//!
//! With no `--ctx`, a slot is planned at the model's limit. Only this default
//! is fitted: when that plan is short, the slot takes the largest context
//! under it, a multiple of the step and no less than the floor, whose plan
//! holds. A plan is short when its card cannot hold the load's resident bytes
//! plus that context's sequence state, when its unified pool no card share
//! fits, or when its host cannot hold what the card then leaves it where the
//! body has no tier past the host ([`Probe::Short`]); any other refusal is
//! the seat's error as it stands. A seat whose unset residency runs on its
//! plan also leaves the default the experts that rule needs: a plan whose
//! card holds fewer than the rule's least in a layer is short too
//! ([`card_experts_floor`]), so the card's room is not all spent on context.
//! A set context is the user's and is not read that way. Callers validate a
//! set context against the model's limit; here it only sets the default.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroU64;

use super::workstation::{HostNeed, HostShort};
use super::{PlacementError, Plan, Role, Violation, per_expert};

/// The slots a load serves when `--parallel` is unset.
pub const DEFAULT_SLOTS: usize = 1;

/// The most slots `--parallel auto` serves ([`most_slots`]).
pub const AUTO_MOST: usize = 8;

/// `--parallel auto`'s count: the most slots in `1..=most` whose plan at one
/// request's context holds, each count planned at most once by `plan_of`, by
/// bisection (a plan of more slots holds more bytes, so the counts that hold
/// are those below a point). `None` when one slot is short, which the seat's
/// one-slot ask refuses by name.
///
/// # Errors
/// The first error `plan_of` returns: a refusal that is not a short memory.
pub fn most_slots<P, E>(
    most: usize,
    mut plan_of: impl FnMut(usize) -> Result<Probe<P>, E>,
) -> Result<Option<usize>, E> {
    if most == 0 || matches!(plan_of(1)?, Probe::Short(_)) {
        return Ok(None);
    }
    super::ctx::largest(1, most, |n| {
        Ok((matches!(plan_of(n)?, Probe::Holds(_)), None))
    })
    .map(Some)
}

/// The user's context ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ask {
    /// A set `--ctx`: the total over the slots.
    Set {
        /// The positions asked over every slot.
        total: u64,
    },
    /// `--ctx` unset: the model's limit, fitted to the machine.
    Default,
}

/// The most positions a slot serves, and the word that names its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limit {
    /// The positions.
    pub positions: u64,
    /// The name of the model bound, such as `trained`.
    pub source: &'static str,
}

/// The contexts the default and a refused set ask search: every one a
/// multiple of `step`, none below `floor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Steps {
    /// The least slot context searched.
    pub floor: u64,
    /// The rounding of every context searched.
    pub step: u64,
}

/// A seat's plan at one slot context, as the seat reads its outcome.
pub enum Probe<P> {
    /// The plan holds.
    Holds(P),
    /// A smaller context is what the plan asks for ([`Short`]). Only this
    /// outcome lets the default search below.
    Short(Short),
}

/// Which memory a context leaves short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The card: the load's resident bytes plus the context's sequence
    /// state pass its bytes, or leave its experts under the floor
    /// ([`card_experts_floor`]).
    Card,
    /// The host: the experts the card leaves it pass its room, on a body
    /// that serves no tier past the host.
    Host,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Card => "card",
            Self::Host => "host",
        })
    }
}

/// A plan short of one memory at a context: the bytes it would hold there
/// and the bytes it has for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Short {
    /// The memory.
    pub on: Side,
    /// The bytes it would hold.
    pub need: u64,
    /// Its bytes for them.
    pub room: u64,
}

impl<P> From<Short> for Probe<P> {
    fn from(s: Short) -> Self {
        Probe::Short(s)
    }
}

/// A plan's violations read as a short card: some, and every one a card
/// over its usable bytes ([`Violation::CardOver`]) or the shared pool a
/// unified card allocates from ([`Violation::PoolOver`], the cards, the host
/// and what sits beside them against the pool); the one past by most. Any
/// other violation is a plan that is wrong, not short.
pub fn short_of(violations: &[Violation]) -> Option<Short> {
    let mut worst: Option<Short> = None;
    for v in violations {
        let (need, room) = match v {
            Violation::CardOver { total, limit, .. } => (*total, *limit),
            Violation::PoolOver {
                cards,
                host,
                beside,
                pool,
            } => (cards + host + beside, *pool),
            _ => return None,
        };
        let s = Short {
            on: Side::Card,
            need,
            room,
        };
        if worst.is_none_or(|w| s.need.saturating_sub(s.room) > w.need.saturating_sub(w.room)) {
            worst = Some(s);
        }
    }
    worst
}

impl PlacementError {
    /// This refusal as a short card: a card whose floor (its dense bytes,
    /// KV, context, scratch, reserves and margin) passes its budget or what
    /// its device had free, or a unified pool no card share of which leaves
    /// the host its NVMe tier's floor ([`PlacementError::PoolBudget`]: the
    /// card and the host read one memory, and a context's sequence state is
    /// part of it), short on the host by the first plan's floor and room; or
    /// a plan whose card leaves a layer under the experts a default context
    /// keeps for the residency rule ([`PlacementError::CardExpertsFloor`]).
    /// Every other refusal, the host's room and the NVMe tier's floor at one
    /// share among them, is `None`: a smaller context is not what it asks
    /// for.
    pub fn card_short(&self) -> Option<Short> {
        match self {
            Self::PoolBudget { room, floor, .. } => Some(Short {
                on: Side::Host,
                need: *floor,
                room: *room,
            }),
            Self::CardBudgetFloor { budget, floor, .. } => Some(Short {
                on: Side::Card,
                need: *floor,
                room: *budget,
            }),
            Self::CardFreeFloor(f) => Some(Short {
                on: Side::Card,
                need: f.floor,
                room: f.free,
            }),
            Self::CardExpertsFloor { need, room, .. } => Some(Short {
                on: Side::Card,
                need: *need,
                room: *room,
            }),
            _ => None,
        }
    }
}

impl HostShort {
    /// This refusal as a short host: on a unified pool (the need counts the cards' share), the
    /// host's room is the pool a context's sequence state takes from, so a smaller context is
    /// what it asks for. Where each card has memory of its own, `None`.
    pub fn pool_short(&self) -> Option<Short> {
        (self.need.cards > 0).then(|| Short {
            on: Side::Host,
            need: self.need.bytes(),
            room: self.available,
        })
    }
}

/// A plan that pages routed experts to the NVMe expert tier, refused by a load that serves no
/// tier past the host (a part source's): the host would hold the paged experts too. A longer
/// context or more slots leave the host more experts, so a smaller one is what it asks for
/// ([`Unpaged::short`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unpaged {
    /// The routed-expert bytes the plan pages (`HostTotals::nvme_expert_bytes`).
    pub paged: u64,
    /// The plan's host need with them on the host and no NVMe tier arena. Its reserves are the
    /// paged plan's, the prompt run-ahead's page-cache window among them.
    pub need: u64,
    /// The host room the plan was made at.
    pub room: u64,
}

impl Unpaged {
    /// `plan`'s refusal by a load that serves no NVMe expert tier; `None` when it pages none.
    #[must_use]
    pub fn of(plan: &Plan<'_>) -> Option<Unpaged> {
        let paged = plan.host.nvme_expert_bytes;
        if paged == 0 {
            return None;
        }
        let held = HostNeed::of(plan, 0);
        let need = HostNeed {
            experts: held.experts.saturating_add(paged),
            arena: 0,
            ..held
        }
        .bytes();
        let room = plan.host.room.as_ref().map_or_else(
            || HostNeed::machine_room(plan.machine.host()),
            |(_, (room, _))| *room,
        );
        Some(Unpaged { paged, need, room })
    }

    /// This refusal as a short host.
    #[must_use]
    pub fn short(&self) -> Short {
        Short {
            on: Side::Host,
            need: self.need,
            room: self.room,
        }
    }
}

impl fmt::Display for Unpaged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the plan pages {} B of routed experts to the NVMe expert tier, which a load from a \
             part source does not serve: the host would hold {} B with them, in a room of {} B; \
             ask a shorter context or fewer slots, or free host memory",
            self.paged, self.need, self.room
        )
    }
}

impl std::error::Error for Unpaged {}

/// A family's plan refusal, read as a short memory ([`PlanRefusal::short`]):
/// each family's error says which of its variants carry a placement error
/// and a plan's violations, and the card's reading is this one; a family
/// whose body serves no tier past the host adds its host refusal.
pub trait PlanRefusal {
    /// The placement error this refusal carries, if any.
    fn placement(&self) -> Option<&PlacementError>;
    /// The plan's violations this refusal carries, if any.
    fn violations(&self) -> Option<&[Violation]>;
    /// This refusal as a short card, else `None`.
    fn card_short(&self) -> Option<Short> {
        self.placement()
            .and_then(PlacementError::card_short)
            .or_else(|| self.violations().and_then(short_of))
    }
    /// This refusal as a short memory, else `None`: the card's reading
    /// unless the family's body adds the host's.
    fn short(&self) -> Option<Short> {
        self.card_short()
    }
}

/// A seat's plan outcome as a probe: a refusal that reads as a short memory
/// ([`PlanRefusal::short`]) fits the default below; any other stays the
/// seat's error.
///
/// # Errors
/// The plan's refusal when it is not short.
pub fn probe_of<P, E: PlanRefusal>(planned: Result<P, E>) -> Result<Probe<P>, E> {
    match planned {
        Ok(plan) => Ok(Probe::Holds(plan)),
        Err(e) => e.short().map(Probe::Short).ok_or(e),
    }
}

/// A default context's floor on the card's experts: stage card `card` of
/// `plan` holds, in every layer its stage runs that has routed experts, at
/// least `least` of them (the layer's whole stack where that is fewer) — the
/// count the unset residency rule needs of the plan's fewest layer
/// (`bloomery_levers::residency38_least_fewest`), under which the word
/// resolves `off`. A plan under it is short on the card by the experts it
/// lacks ([`PlacementError::CardExpertsFloor`], read as a short card by
/// [`PlacementError::card_short`]), so a default that fits under it takes the
/// largest context that leaves them. A stage with no routed experts has none
/// to ask for.
///
/// # Errors
/// [`PlacementError::CardExpertsFloor`] for a plan under the floor;
/// [`PlacementError::Experts`] for a `card` the plan's machine has no stage
/// for, and the refusal of a routed stack that is not one of the model's
/// experts ([`per_expert`]).
pub fn card_experts_floor(plan: &Plan<'_>, card: usize, least: u64) -> Result<(), PlacementError> {
    let model = plan.model;
    let stage = plan.machine.cards.get(card).ok_or_else(|| {
        PlacementError::Experts(format!(
            "card {card} is no stage card of a plan of {} cards",
            plan.machine.cards.len()
        ))
    })?;
    let (mut need, mut room, mut layers) = (0u64, 0u64, 0usize);
    for l in stage.layers.clone() {
        // One expert of the layer: every routed stack's bytes of it.
        let mut stacks = model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts && t.layer == Some(l))
            .peekable();
        if stacks.peek().is_none() {
            continue;
        }
        let mut per = 0u64;
        for t in stacks {
            per = per.saturating_add(per_expert(t, model.experts_of(t))?.1);
        }
        let want = least.min(model.layer_experts(l));
        let held = plan.n_l.get(l).copied().unwrap_or(0).min(want);
        need = need.saturating_add(want.saturating_mul(per));
        room = room.saturating_add(held.saturating_mul(per));
        layers += usize::from(held < want);
    }
    if layers == 0 {
        return Ok(());
    }
    Err(PlacementError::CardExpertsFloor {
        card: stage.name.clone(),
        least,
        layers,
        need,
        room,
    })
}

/// What set the context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// The user's `--ctx`.
    Set,
    /// The default: the model's limit held.
    Target,
    /// The default fitted under `from`: the largest context that holds,
    /// the context one step above it `short`.
    Fitted {
        /// The target the default began at.
        from: u64,
        /// The plan one step above the context chosen.
        short: Short,
    },
}

/// The context chosen and the plan made at it.
pub struct Chosen<P> {
    /// The positions a slot serves.
    pub slot_ctx: u64,
    /// The slots.
    pub slots: usize,
    /// The positions served over every slot: `slot_ctx` × `slots`.
    pub total: u64,
    /// What set it.
    pub rule: Rule,
    /// The model's limit.
    pub limit: Limit,
    /// The seat's plan at `slot_ctx`.
    pub plan: P,
}

impl Rule {
    /// The rule's word on the `ctx` line.
    #[must_use]
    pub fn word(&self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::Target => "target",
            Self::Fitted { .. } => "fitted",
        }
    }
}

impl<P> Chosen<P> {
    /// The one stderr line that says why the load serves this context.
    pub fn line(&self) -> String {
        let word = self.rule.word();
        let line = format!(
            "ctx rule={word} ctx={} slots={} total={} limit={} ({})",
            self.slot_ctx, self.slots, self.total, self.limit.positions, self.limit.source
        );
        match self.rule {
            Rule::Fitted {
                from,
                short: Short { on, need, room },
            } => format!("{line} from={from} {on}-short need={need} room={room}"),
            Rule::Set | Rule::Target => line,
        }
    }
}

/// Why no context was chosen.
#[derive(Debug, PartialEq, Eq)]
pub enum AskError<E> {
    /// `--parallel 0`.
    NoSlot,
    /// A set total smaller than the slot count: a slot gets no position.
    NoPosition {
        /// The asked total.
        total: u64,
        /// The slots.
        slots: usize,
    },
    /// A set context the machine cannot hold.
    DoesNotFit {
        /// The asked total.
        total: u64,
        /// Its share a slot.
        slot_ctx: u64,
        /// The slots.
        slots: usize,
        /// The plan at `slot_ctx`.
        short: Short,
        /// The largest share a slot holds, `None` when not even `floor`.
        largest: Option<u64>,
        /// The least share searched.
        floor: u64,
    },
    /// The default searched down to its floor and the machine still cannot
    /// hold it.
    FloorShort {
        /// The least context searched.
        floor: u64,
        /// The slots.
        slots: usize,
        /// The plan at `floor`.
        short: Short,
    },
    /// The seat's plan refused for a reason other than a short memory;
    /// never retried.
    Plan(E),
}

impl<E: fmt::Display> fmt::Display for AskError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSlot => f.write_str("--parallel 0: the server serves no slot"),
            Self::NoPosition { total, slots } => write!(
                f,
                "--ctx {total} over {slots} slots gives a slot no position: ask at least {slots}"
            ),
            Self::DoesNotFit {
                total,
                slot_ctx,
                slots,
                short: Short { on, need, room },
                largest,
                floor,
            } => {
                write!(
                    f,
                    "--ctx {total} over {slots} slot(s) asks {slot_ctx} positions a slot, and \
                     the {on} holds {room} of the {need} bytes that takes: "
                )?;
                match largest {
                    Some(n) => write!(
                        f,
                        "a slot holds at most {n} here; ask {} or less, or fewer slots",
                        n.saturating_mul(u64::try_from(*slots).unwrap_or(u64::MAX))
                    ),
                    None => write!(
                        f,
                        "not even {floor} positions a slot fit; ask fewer slots or a smaller \
                         model file"
                    ),
                }
            }
            Self::FloorShort {
                floor,
                slots,
                short: Short { on, need, room },
            } => write!(
                f,
                "with --ctx unset, {slots} slot(s) at {floor} positions, the least the default \
                 tries, take {need} bytes on a {on} that holds {room}: ask fewer slots or a \
                 smaller model file"
            ),
            Self::Plan(e) => write!(f, "{e}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for AskError<E> {}

/// The slot context for `ask` over `slots`, each candidate planned once by
/// `plan_at`, and the plan made at the one chosen.
///
/// # Errors
/// [`AskError::NoSlot`] and [`AskError::NoPosition`] before any plan;
/// [`AskError::DoesNotFit`] for a set context the card cannot hold;
/// [`AskError::FloorShort`] for a default the card cannot hold even at
/// `steps.floor`; [`AskError::Plan`] for any refusal of the seat's own.
///
/// # Panics
/// On a `steps` with a zero floor or step: a seat's constants, not input.
pub fn choose<P, E>(
    ask: Ask,
    slots: usize,
    limit: Limit,
    steps: Steps,
    mut plan_at: impl FnMut(u64) -> Result<Probe<P>, E>,
) -> Result<Chosen<P>, AskError<E>> {
    assert!(
        steps.floor > 0 && steps.step > 0,
        "ctx_ask::choose: a zero floor or step ({steps:?})"
    );
    if slots == 0 {
        return Err(AskError::NoSlot);
    }
    let n = u64::try_from(slots).unwrap_or(u64::MAX);
    let mut probe = |c: u64| plan_at(c).map_err(AskError::Plan);
    let chosen = |slot_ctx: u64, rule: Rule, plan: P| Chosen {
        slot_ctx,
        slots,
        total: slot_ctx.saturating_mul(n),
        rule,
        limit,
        plan,
    };
    match ask {
        Ask::Set { total } => {
            let slot_ctx = crate::slots::split_ctx(total, n, NonZeroU64::MIN)
                .map_err(|_| AskError::NoPosition { total, slots })?;
            let short = match probe(slot_ctx)? {
                Probe::Holds(plan) => return Ok(chosen(slot_ctx, Rule::Set, plan)),
                Probe::Short(s) => s,
            };
            let largest = match largest_held(steps, (slot_ctx, short), &mut probe)? {
                Below::Held { slot_ctx, .. } => Some(slot_ctx),
                Below::Nothing { .. } => None,
            };
            Err(AskError::DoesNotFit {
                total,
                slot_ctx,
                slots,
                short,
                largest,
                floor: steps.floor,
            })
        }
        Ask::Default => {
            let cap = limit.positions;
            let target = round_down(cap, steps.step).max(steps.floor.min(cap));
            let short = match probe(target)? {
                Probe::Holds(plan) => return Ok(chosen(target, Rule::Target, plan)),
                Probe::Short(s) => s,
            };
            match largest_held(steps, (target, short), &mut probe)? {
                Below::Held {
                    slot_ctx,
                    plan,
                    above,
                } => Ok(chosen(
                    slot_ctx,
                    Rule::Fitted {
                        from: target,
                        short: above,
                    },
                    plan,
                )),
                Below::Nothing { least, short } => Err(AskError::FloorShort {
                    floor: least,
                    slots,
                    short,
                }),
            }
        }
    }
}

/// `c` rounded down to a multiple of `step`.
fn round_down(c: u64, step: u64) -> u64 {
    c - c % step
}

/// What the search under a short context found ([`largest_held`]).
enum Below<P> {
    /// The largest context that holds, its plan, and the plan of the least
    /// context above it that was short.
    Held {
        slot_ctx: u64,
        plan: P,
        above: Short,
    },
    /// No context holds: the least one planned and its short plan.
    Nothing { least: u64, short: Short },
}

/// The steps a search under a short context asks by its shortfalls' line
/// ([`largest_held`]): the step under the first short plan, then on the
/// line the step past the fit and the fit.
const STEERED: usize = 3;

/// The largest multiple of `steps.step` in `[steps.floor, ctx)` whose plan
/// holds, `ctx` being a context whose plan is `short`, by bisection (a plan
/// holds below a point and not above it) steered by the shortfalls: the
/// first plan under `ctx` that is short has the step under it asked next,
/// and once two plans under `ctx` are short on one memory, the context
/// asked next is the step past the largest their line holds ([`fit_of`]),
/// then that one, at most [`STEERED`] asks in all. `ctx`'s own shortfall is not on the line: on
/// a unified pool its plan can be refused by the pool's share before the
/// host's need the plans under it are short by. The floor is planned only
/// when every context above it is short. Each context is planned at most
/// once.
fn largest_held<P, E>(
    steps: Steps,
    (ctx, short): (u64, Short),
    probe: &mut impl FnMut(u64) -> Result<Probe<P>, AskError<E>>,
) -> Result<Below<P>, AskError<E>> {
    let lo_k = steps.floor.div_ceil(steps.step);
    let hi_k = (ctx - 1) / steps.step;
    if lo_k > hi_k {
        return Ok(Below::Nothing { least: ctx, short });
    }
    let mut seen = BTreeSet::new();
    let mut held: Option<(u64, P)> = None;
    let mut ask = |c: u64| -> Result<Option<Short>, AskError<E>> {
        assert!(seen.insert(c), "ctx_ask: {c} planned twice");
        Ok(match probe(c)? {
            Probe::Holds(plan) => {
                if held.as_ref().is_none_or(|(h, _)| c > *h) {
                    held = Some((c, plan));
                }
                None
            }
            Probe::Short(s) => Some(s),
        })
    };
    // The short plans by context, the least first and `ctx`'s last.
    let mut shorts = vec![(ctx, short)];
    let lo = usize::try_from(lo_k).expect("supported hosts have 64-bit usize");
    let hi = usize::try_from(hi_k).expect("supported hosts have 64-bit usize");
    // The step named last, and the named steps asked.
    let (mut named, mut steered) = (None, 0);
    // The search takes its floor as held; it is planned here only when the
    // search ends on it.
    let k = super::ctx::largest(lo, hi, |k| {
        let c = k as u64 * steps.step;
        let short = ask(c)?;
        if let Some(s) = short {
            shorts.insert(shorts.partition_point(|&(h, _)| h < c), (c, s));
        }
        steered += usize::from(named == Some(k));
        let under = &shorts[..shorts.len() - 1];
        named = match fit_of(under, steps.step) {
            _ if steered >= STEERED => None,
            Some(fit) if under.iter().any(|&(h, _)| h == (fit + 1) * steps.step) => Some(fit),
            Some(fit) => Some(fit + 1),
            // One short plan under `ctx`: the step under it is the line's other point.
            None if under.len() == 1 => Some(under[0].0 / steps.step - 1),
            None => None,
        }
        .and_then(|n| usize::try_from(n).ok());
        Ok((short.is_none(), named))
    })?;
    if k == lo
        && let Some(s) = ask(lo_k * steps.step)?
    {
        shorts.insert(0, (lo_k * steps.step, s));
    }
    let (least, short) = shorts[0];
    Ok(match held {
        Some((slot_ctx, plan)) => Below::Held {
            slot_ctx,
            plan,
            above: short,
        },
        None => Below::Nothing { least, short },
    })
}

/// The steps in the largest context the line through the two least short
/// plans on the least one's memory holds, each plan's shortfall (`need`
/// past `room`) a point of it; `None` without such a pair or on a line
/// that does not fall with the context.
fn fit_of(shorts: &[(u64, Short)], step: u64) -> Option<u64> {
    let gap = |s: &Short| u128::from(s.need.saturating_sub(s.room));
    let (c1, s1) = shorts.first()?;
    let (c2, s2) = shorts[1..].iter().find(|(_, s)| s.on == s1.on)?;
    let (d1, d2) = (gap(s1), gap(s2));
    if d2 <= d1 {
        return None;
    }
    let back = (d1 * u128::from(c2 - c1)).div_ceil(d2 - d1);
    let fit = u128::from(*c1).checked_sub(back)?;
    u64::try_from(fit / u128::from(step)).ok()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::super::workstation::{HostNeed, HostRead};
    use super::*;

    const STEPS: Steps = Steps {
        floor: 4096,
        step: 1024,
    };

    /// A card that holds `resident + c × per_pos` bytes up to `room`, and a
    /// seat error at `fails_at`; every context it is asked for is recorded.
    struct Card {
        resident: u64,
        per_pos: u64,
        room: u64,
        fails_at: Option<u64>,
        calls: RefCell<Vec<u64>>,
    }

    impl Card {
        fn holding(up_to: u64) -> Card {
            Card {
                resident: 1000,
                per_pos: 10,
                room: 1000 + up_to * 10,
                fails_at: None,
                calls: RefCell::new(Vec::new()),
            }
        }

        fn plan(&self, c: u64) -> Result<Probe<u64>, String> {
            self.calls.borrow_mut().push(c);
            if self.fails_at == Some(c) {
                return Err(format!("the seat's own refusal at {c}"));
            }
            let need = self.resident + c * self.per_pos;
            Ok(if need <= self.room {
                Probe::Holds(c)
            } else {
                Probe::Short(Short {
                    on: Side::Card,
                    need,
                    room: self.room,
                })
            })
        }

        fn calls(&self) -> Vec<u64> {
            self.calls.borrow().clone()
        }
    }

    const TRAINED: Limit = Limit {
        positions: 262_144,
        source: "trained",
    };

    /// [`run`] of the default on one slot, each short plan's [`Short`] as
    /// `short` reads it at its context; the context chosen.
    fn run_with(card: &Card, short: impl Fn(u64, Short) -> Short) -> u64 {
        choose(Ask::Default, 1, TRAINED, STEPS, |c| {
            Ok::<_, String>(match card.plan(c)? {
                Probe::Short(s) => Probe::Short(short(c, s)),
                held @ Probe::Holds(_) => held,
            })
        })
        .expect("holds")
        .slot_ctx
    }

    fn run(
        card: &Card,
        ask: Ask,
        slots: usize,
        limit: Limit,
    ) -> Result<Chosen<u64>, AskError<String>> {
        choose(ask, slots, limit, STEPS, |c| card.plan(c))
    }

    /// Unset, a slot is planned once at the target, the model's limit
    /// down to a step.
    #[test]
    fn the_default_is_the_target_when_it_holds() {
        let card = Card::holding(u64::MAX / 20);
        let got = run(&card, Ask::Default, 1, TRAINED).expect("holds");
        assert_eq!(
            (got.slot_ctx, got.rule, got.plan),
            (262_144, Rule::Target, 262_144)
        );
        assert_eq!(card.calls(), [262_144]);
        for (positions, target) in [
            (32_768, 32_768),
            (1_048_576, 1_048_576),
            (40_960 + 7, 40_960),
        ] {
            let limit = Limit {
                positions,
                source: "trained",
            };
            let card = Card::holding(u64::MAX / 20);
            let got = run(&card, Ask::Default, 1, limit).expect("holds");
            assert_eq!((got.slot_ctx, got.rule), (target, Rule::Target));
            assert_eq!(card.calls(), [target]);
        }
    }

    /// A short card fits the default to the largest step that holds, each
    /// context planned once and at most a bisection's count; the rule names
    /// the target and the plan one step above the context chosen.
    ///
    /// PIN(2026-10-27): replaces `a_short_card_halves_only_the_default`; an
    /// unset context is fitted, not halved.
    #[test]
    fn a_short_card_fits_only_the_default() {
        for up_to in [70_000, 4096, 5119, 261_120, 262_143] {
            let card = Card::holding(up_to);
            let got = run(&card, Ask::Default, 1, TRAINED).expect("holds");
            let want = up_to / 1024 * 1024;
            assert_eq!((got.slot_ctx, got.plan), (want, want), "up to {up_to}");
            assert_eq!(
                got.rule,
                Rule::Fitted {
                    from: 262_144,
                    short: Short {
                        on: Side::Card,
                        need: 1000 + (want + 1024) * 10,
                        room: card.room,
                    },
                },
                "up to {up_to}"
            );
            let calls = card.calls();
            assert_eq!(calls[0], 262_144);
            let distinct: BTreeSet<u64> = calls.iter().copied().collect();
            assert_eq!(distinct.len(), calls.len(), "planned twice: {calls:?}");
            // The target, then a bisection over the 252 steps under it; the
            // first plan under the target that is short has the step under
            // it asked, and the line through the two names the step past the
            // fit and the fit itself (mutant: the short plans kept in the
            // order asked, so the line runs through the wrong pair).
            assert!(calls.len() <= 1 + 9, "{} plans: {calls:?}", calls.len());
            if up_to < 133_120 {
                assert!(calls.len() <= 5, "{calls:?}");
            }
        }
        // The target's shortfall stays off the line: a pool's share refused
        // there by a shortfall off the card's line takes the same plans.
        // Mutant: the target on the line, whose fit then is near 30,000.
        let card = Card::holding(70_000);
        let got = run_with(&card, |c, s| match c {
            262_144 => Short {
                need: 1000 + 133_120 * 10 + 789_760,
                ..s
            },
            _ => s,
        });
        assert_eq!(got, 69_632);
        assert_eq!(card.calls().len(), 5, "{:?}", card.calls());
    }

    /// Whatever the shortfalls' shape, the steered search ends on the
    /// largest step that holds, plans each context once and takes at most
    /// a bisection's plans and the steered ones. Mutant: the steered asks
    /// uncapped, so a quartic shortfall's line walks down a step at a time.
    #[test]
    fn a_steered_search_ends_on_the_largest_step_whatever_the_shortfalls() {
        type Need = fn(u64) -> u64;
        let needs: [(&str, Need); 4] = [
            ("line", |c| 1000 + c * 10),
            ("quartic", |c| 1000 + (c / 1024).pow(4)),
            ("concave", |c| 1000 + 300 * c.isqrt()),
            ("stepped", |c| {
                1000 + c * 10 + if c > 90_000 { 400_000 } else { 0 }
            }),
        ];
        for (word, need) in needs {
            // A fit at every third step, and one between steps.
            let rooms = (4..256).step_by(3).map(|k| need(k * 1024));
            for room in rooms.chain([need(50_000) + 7]) {
                let calls = RefCell::new(Vec::new());
                let got = choose(Ask::Default, 1, TRAINED, STEPS, |c| {
                    calls.borrow_mut().push(c);
                    let n = need(c);
                    Ok::<_, String>(if n <= room {
                        Probe::Holds(c)
                    } else {
                        Probe::Short(Short {
                            on: Side::Host,
                            need: n,
                            room,
                        })
                    })
                })
                .expect("the floor holds");
                let want = (4..256u64)
                    .map(|k| k * 1024)
                    .filter(|&c| need(c) <= room)
                    .max()
                    .expect("the floor holds");
                let calls = calls.into_inner();
                assert_eq!(got.slot_ctx, want, "{word} room {room}: {calls:?}");
                let distinct: BTreeSet<u64> = calls.iter().copied().collect();
                assert_eq!(
                    distinct.len(),
                    calls.len(),
                    "{word}: planned twice: {calls:?}"
                );
                assert!(
                    calls.len() <= 1 + 9 + STEERED,
                    "{word}: {} plans: {calls:?}",
                    calls.len()
                );
            }
        }
    }

    /// The default stops at its floor, naming it and the bytes.
    #[test]
    fn a_default_short_at_the_floor_is_refused() {
        let card = Card::holding(1000);
        let err = run(&card, Ask::Default, 2, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("short");
        assert_eq!(*card.calls().last().expect("a call"), 4096);
        assert_eq!(
            err,
            AskError::FloorShort {
                floor: 4096,
                slots: 2,
                short: Short {
                    on: Side::Card,
                    need: 1000 + 4096 * 10,
                    room: card.room,
                },
            }
        );
        // A target that is no power of two searches down to the floor too.
        let limit = Limit {
            positions: 24_576,
            source: "trained",
        };
        let card = Card::holding(4096);
        let got = run(&card, Ask::Default, 1, limit).expect("holds at the floor");
        assert_eq!(*card.calls().last().expect("a call"), 4096);
        assert_eq!(got.slot_ctx, 4096);
        let card = Card::holding(4095);
        let err = run(&card, Ask::Default, 1, limit)
            .map(|c| c.slot_ctx)
            .expect_err("short at the floor");
        assert_eq!(*card.calls().last().expect("a call"), 4096);
        assert!(matches!(err, AskError::FloorShort { floor: 4096, .. }));
        // A limit under the floor is its own one context.
        let limit = Limit {
            positions: 2048,
            source: "trained",
        };
        let err = run(&Card::holding(1000), Ask::Default, 1, limit)
            .map(|c| c.slot_ctx)
            .expect_err("short under the floor");
        assert!(matches!(err, AskError::FloorShort { floor: 2048, .. }));
    }

    /// A seat's own refusal ends the default at once.
    #[test]
    fn a_seat_error_is_never_retried() {
        let mut card = Card::holding(70_000);
        // The search's first context under the target: the midpoint of the
        // steps 4 to 255.
        card.fails_at = Some(130 * 1024);
        let err = run(&card, Ask::Default, 1, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("seat error");
        assert_eq!(
            err,
            AskError::Plan("the seat's own refusal at 133120".to_string())
        );
        assert_eq!(card.calls(), [262_144, 133_120]);
    }

    /// A set total is split over the slots and planned once.
    #[test]
    fn a_set_total_is_split_and_planned_once() {
        let card = Card::holding(u64::MAX / 20);
        let got = run(&card, Ask::Set { total: 524_288 }, 2, TRAINED).expect("holds");
        assert_eq!(
            (got.slot_ctx, got.total, got.rule),
            (262_144, 524_288, Rule::Set)
        );
        assert_eq!(card.calls(), [262_144]);
        let card = Card::holding(u64::MAX / 20);
        let got = run(&card, Ask::Set { total: 10_001 }, 2, TRAINED).expect("holds");
        assert_eq!((got.slot_ctx, got.total), (5000, 10_000));
    }

    /// A set context the card cannot hold is refused with the largest share
    /// that holds, each candidate planned once; none when the floor is short.
    #[test]
    fn a_set_context_that_does_not_fit_names_the_largest_that_does() {
        let card = Card::holding(100_000);
        let err = run(&card, Ask::Set { total: 262_144 }, 1, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("short");
        let AskError::DoesNotFit {
            largest, slot_ctx, ..
        } = err
        else {
            panic!("{err:?}, not DoesNotFit");
        };
        assert_eq!((slot_ctx, largest), (262_144, Some(99_328)));
        let calls = card.calls();
        let distinct: BTreeSet<u64> = calls.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            calls.len(),
            "a context planned twice: {calls:?}"
        );
        let card = Card::holding(1000);
        let err = run(&card, Ask::Set { total: 8192 }, 1, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("short");
        assert!(
            matches!(err, AskError::DoesNotFit { largest: None, .. }),
            "{err:?}"
        );
    }

    /// No slot and no position are refused before any plan.
    #[test]
    fn no_slot_and_no_position_plan_nothing() {
        let card = Card::holding(1 << 20);
        let err = run(&card, Ask::Default, 0, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("no slot");
        assert_eq!(err, AskError::NoSlot);
        let err = run(&card, Ask::Set { total: 1 }, 2, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("none");
        assert_eq!(err, AskError::NoPosition { total: 1, slots: 2 });
        assert!(card.calls().is_empty());
    }

    /// The line names the rule, the context, the slots, the total and the
    /// limit with its source; a fitted default adds its target and the
    /// bytes one step above.
    #[test]
    fn the_line_says_why() {
        let chosen = |rule| Chosen {
            slot_ctx: 65_536,
            slots: 1,
            total: 65_536,
            rule,
            limit: TRAINED,
            plan: (),
        };
        assert_eq!(
            chosen(Rule::Set).line(),
            "ctx rule=set ctx=65536 slots=1 total=65536 limit=262144 (trained)"
        );
        assert_eq!(
            chosen(Rule::Target).line(),
            "ctx rule=target ctx=65536 slots=1 total=65536 limit=262144 (trained)"
        );
        assert_eq!(
            chosen(Rule::Fitted {
                from: 262_144,
                short: Short {
                    on: Side::Card,
                    need: 9,
                    room: 8,
                },
            })
            .line(),
            "ctx rule=fitted ctx=65536 slots=1 total=65536 limit=262144 (trained) from=262144 \
             card-short need=9 room=8"
        );
        assert_eq!(
            chosen(Rule::Fitted {
                from: 262_144,
                short: Short {
                    on: Side::Host,
                    need: 9,
                    room: 8,
                },
            })
            .line(),
            "ctx rule=fitted ctx=65536 slots=1 total=65536 limit=262144 (trained) from=262144 \
             host-short need=9 room=8"
        );
    }

    /// Violations read as a short card only when every one is a card or its
    /// shared pool over its bytes, the one past by most; a placement error
    /// only when it is a card's floor.
    #[test]
    fn only_a_card_over_its_bytes_reads_as_short() {
        let over = |total, limit| Violation::CardOver {
            card: "c".to_string(),
            total,
            limit,
        };
        assert_eq!(
            short_of(&[over(110, 100), over(150, 120)]),
            Some(Short {
                on: Side::Card,
                need: 150,
                room: 120
            })
        );
        let other = Violation::WrongCard {
            tensor: "t".to_string(),
            card: "c".to_string(),
        };
        assert_eq!(short_of(&[over(110, 100), other]), None);
        // A unified pool's shortage alone is short too: the cards, the host and
        // what sits beside them against the pool.
        let pool = Violation::PoolOver {
            cards: 70,
            host: 50,
            beside: 10,
            pool: 100,
        };
        assert_eq!(
            short_of(&[over(110, 100), pool]),
            Some(Short {
                on: Side::Card,
                need: 130,
                room: 100
            })
        );
        assert_eq!(short_of(&[]), None);
        let floor = PlacementError::CardBudgetFloor {
            card: "c".to_string(),
            budget: 90,
            floor: 120,
            dense: 1,
            kv: 1,
            context: 1,
            scratch: 1,
            reserves: 1,
            margin: 1,
            beside: 1,
        };
        assert_eq!(
            floor.card_short(),
            Some(Short {
                on: Side::Card,
                need: 120,
                room: 90
            })
        );
        assert_eq!(PlacementError::Experts("e".to_string()).card_short(), None);
        let pool = PlacementError::PoolBudget {
            rounds: 3,
            budget: 80,
            last: Box::new(PlacementError::Experts("e".to_string())),
            room: 40,
            floor: 70,
        };
        assert_eq!(
            pool.card_short(),
            Some(Short {
                on: Side::Host,
                need: 70,
                room: 40
            })
        );
        // A host need past the reading is short where the cards share the pool, and stops
        // where each card has memory of its own.
        let need = HostNeed {
            experts: 0,
            tables: 0,
            shadows: 0,
            reserves: 20,
            extra: 0,
            read_ahead: 15,
            arena: 0,
            cards: 90,
            os: 0,
            card_budget: false,
        };
        let host = |need| HostShort {
            need,
            available: 120,
            read: Box::new(HostRead::MemAvailable),
            floor: None,
        };
        assert_eq!(
            host(need).pool_short(),
            Some(Short {
                on: Side::Host,
                need: 125,
                room: 120
            })
        );
        assert_eq!(host(HostNeed { cards: 0, ..need }).pool_short(), None);
    }

    /// `--parallel auto` counts the most slots whose plan holds a slot's
    /// context, up to its most, each count planned once; one short slot is
    /// `None`, and a seat's own refusal is the error. Mutants: a linear
    /// search past the short count (a count planned twice or above the
    /// fit), the most ignored (nine slots on a card that holds more).
    #[test]
    fn auto_counts_the_most_slots_that_hold() {
        const SLOT: u64 = 4096;
        for fits in 0..=10u64 {
            let card = Card::holding(fits * SLOT);
            let got = most_slots(AUTO_MOST, |n| card.plan(n as u64 * SLOT)).expect("no refusal");
            let want = (fits > 0).then(|| fits.min(AUTO_MOST as u64) as usize);
            assert_eq!(got, want, "a card that holds {fits} slots");
            let mut calls = card.calls();
            let asked = calls.len();
            calls.sort_unstable();
            calls.dedup();
            assert_eq!(calls.len(), asked, "a count planned twice: {calls:?}");
            assert!(asked <= 4, "{asked} plans for {fits} slots");
            assert!(calls.iter().all(|&c| c <= AUTO_MOST as u64 * SLOT));
        }
        let refusing = Card {
            fails_at: Some(5 * SLOT),
            ..Card::holding(8 * SLOT)
        };
        assert_eq!(
            most_slots(AUTO_MOST, |n| refusing.plan(n as u64 * SLOT)),
            Err(format!("the seat's own refusal at {}", 5 * SLOT))
        );
        assert_eq!(most_slots(0, |n| refusing.plan(n as u64 * SLOT)), Ok(None));
    }
}
