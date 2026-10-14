//! The context a load serves, from the user's slots and context: the ask
//! comes first, and the plan fills the card around it.
//!
//! A set `--ctx` is the total over the slots, as llama-server reads `-c`;
//! each slot takes its floor share. The plan is made at that share and the
//! load either holds it or stops by name, with the largest share the card
//! would hold ([`AskError::DoesNotFit`]). It is never cut.
//!
//! With no `--ctx`, a slot is planned at [`DEFAULT_TARGET`], or the model's
//! limit where that is less. Only this default steps down, by halves, and
//! only on a plan whose card cannot hold the load's resident bytes plus that
//! context's sequence state, or whose host cannot hold what the card then
//! leaves it where the body has no tier past the host ([`Probe::Short`]); any
//! other refusal is the seat's error as it stands. Callers validate a set context against the
//! model's limit; here it only caps the default target.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroU64;

use super::{PlacementError, Violation};

/// The context a slot is planned at when `--ctx` is unset.
pub const DEFAULT_TARGET: u64 = 262_144;

/// The slots a load serves when `--parallel` is unset.
pub const DEFAULT_SLOTS: usize = 1;

/// The user's context ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ask {
    /// A set `--ctx`: the total over the slots.
    Set {
        /// The positions asked over every slot.
        total: u64,
    },
    /// `--ctx` unset: the default target.
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

/// The rungs the default steps down on, and the set ask's search: every rung
/// is a multiple of `step`, none below `floor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Steps {
    /// The least slot context the default tries.
    pub floor: u64,
    /// The rounding of every rung.
    pub step: u64,
}

/// A seat's plan at one slot context, as the seat reads its outcome.
pub enum Probe<P> {
    /// The plan holds.
    Holds(P),
    /// A smaller context is what the plan asks for ([`Short`]). Only this
    /// outcome lets the default step down.
    Short(Short),
}

/// Which memory a context leaves short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The card: the load's resident bytes plus the context's sequence
    /// state pass its bytes.
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
    /// its device had free. Every other refusal, the host's room and the
    /// NVMe tier's floor among them, is `None`: a smaller context is not
    /// what it asks for.
    pub fn card_short(&self) -> Option<Short> {
        match self {
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
            _ => None,
        }
    }
}

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
/// ([`PlanRefusal::short`]) steps the default down; any other stays the
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

/// What set the context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// The user's `--ctx`.
    Set,
    /// The default target.
    Target,
    /// The default stepped down from `from`: the rung above the one that
    /// held was `short`.
    Halved {
        /// The target the default began at.
        from: u64,
        /// The last short plan.
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

impl<P> Chosen<P> {
    /// The one stderr line that says why the load serves this context.
    pub fn line(&self) -> String {
        let word = match self.rule {
            Rule::Set => "set",
            Rule::Target => "target",
            Rule::Halved { .. } => "halved",
        };
        let line = format!(
            "ctx rule={word} ctx={} slots={} total={} limit={} ({})",
            self.slot_ctx, self.slots, self.total, self.limit.positions, self.limit.source
        );
        match self.rule {
            Rule::Halved {
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
    /// The default stepped down to its floor and the machine still cannot
    /// hold it.
    FloorShort {
        /// The least rung.
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
            let largest = largest_held(steps, slot_ctx, &mut probe)?;
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
            let cap = DEFAULT_TARGET.min(limit.positions);
            let target = round_down(cap, steps.step).max(steps.floor.min(cap));
            let mut rung = target;
            let mut short = None;
            loop {
                match probe(rung)? {
                    Probe::Holds(plan) => {
                        let rule = match short {
                            None => Rule::Target,
                            Some(short) => Rule::Halved {
                                from: target,
                                short,
                            },
                        };
                        return Ok(chosen(rung, rule, plan));
                    }
                    Probe::Short(s) => {
                        short = Some(s);
                        // The floor is a rung of its own: a target that is
                        // no power of two halves past it otherwise.
                        let next = round_down(rung / 2, steps.step).max(steps.floor.min(target));
                        if next == rung {
                            return Err(AskError::FloorShort {
                                floor: rung,
                                slots,
                                short: s,
                            });
                        }
                        rung = next;
                    }
                }
            }
        }
    }
}

/// `c` rounded down to a multiple of `step`.
fn round_down(c: u64, step: u64) -> u64 {
    c - c % step
}

/// The largest multiple of `steps.step` in `[steps.floor, short)` whose plan
/// holds, by bisection (a plan holds below a point and not above it); `None`
/// when the floor's does not. Each context is planned at most once.
fn largest_held<P, E>(
    steps: Steps,
    short: u64,
    probe: &mut impl FnMut(u64) -> Result<Probe<P>, AskError<E>>,
) -> Result<Option<u64>, AskError<E>> {
    let mut seen = BTreeSet::new();
    let mut holds = |c: u64| -> Result<bool, AskError<E>> {
        assert!(seen.insert(c), "ctx_ask: {c} planned twice");
        Ok(matches!(probe(c)?, Probe::Holds(_)))
    };
    let lo_k = steps.floor.div_ceil(steps.step);
    let hi_k = (short - 1) / steps.step;
    if lo_k > hi_k || !holds(lo_k * steps.step)? {
        return Ok(None);
    }
    let lo = usize::try_from(lo_k).expect("supported hosts have 64-bit usize");
    let hi = usize::try_from(hi_k).expect("supported hosts have 64-bit usize");
    let k = super::ctx::largest(lo, hi, |k| holds(k as u64 * steps.step))?;
    Ok(Some(k as u64 * steps.step))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

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

    fn run(
        card: &Card,
        ask: Ask,
        slots: usize,
        limit: Limit,
    ) -> Result<Chosen<u64>, AskError<String>> {
        choose(ask, slots, limit, STEPS, |c| card.plan(c))
    }

    /// Unset, a slot is planned once at the target, the least of
    /// [`DEFAULT_TARGET`] and the limit.
    #[test]
    fn the_default_is_the_target_when_it_holds() {
        let card = Card::holding(u64::MAX / 20);
        let got = run(&card, Ask::Default, 1, TRAINED).expect("holds");
        assert_eq!(
            (got.slot_ctx, got.rule, got.plan),
            (262_144, Rule::Target, 262_144)
        );
        assert_eq!(card.calls(), [262_144]);
        let small = Limit {
            positions: 32_768,
            source: "trained",
        };
        let card = Card::holding(u64::MAX / 20);
        let got = run(&card, Ask::Default, 1, small).expect("holds");
        assert_eq!((got.slot_ctx, got.rule), (32_768, Rule::Target));
        assert_eq!(card.calls(), [32_768]);
    }

    /// A short card halves the default, rung by rung, until a plan holds;
    /// the rule names the target and the last short plan's bytes.
    #[test]
    fn a_short_card_halves_only_the_default() {
        let card = Card::holding(70_000);
        let got = run(&card, Ask::Default, 1, TRAINED).expect("holds at 65,536");
        assert_eq!(card.calls(), [262_144, 131_072, 65_536]);
        assert_eq!(got.slot_ctx, 65_536);
        assert_eq!(
            got.rule,
            Rule::Halved {
                from: 262_144,
                short: Short {
                    on: Side::Card,
                    need: 1000 + 131_072 * 10,
                    room: card.room,
                },
            }
        );
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
        // A target that is no power of two steps onto the floor before it
        // refuses: 24,576 halves to 12,288 and 6,144, whose half is under it.
        let limit = Limit {
            positions: 24_576,
            source: "trained",
        };
        let card = Card::holding(4096);
        let got = run(&card, Ask::Default, 1, limit).expect("holds at the floor");
        assert_eq!(card.calls(), [24_576, 12_288, 6_144, 4096]);
        assert_eq!(got.slot_ctx, 4096);
        let card = Card::holding(4095);
        let err = run(&card, Ask::Default, 1, limit)
            .map(|c| c.slot_ctx)
            .expect_err("short at the floor");
        assert_eq!(card.calls(), [24_576, 12_288, 6_144, 4096]);
        assert!(matches!(err, AskError::FloorShort { floor: 4096, .. }));
    }

    /// A seat's own refusal ends the default at once.
    #[test]
    fn a_seat_error_is_never_retried() {
        let mut card = Card::holding(70_000);
        card.fails_at = Some(131_072);
        let err = run(&card, Ask::Default, 1, TRAINED)
            .map(|c| c.slot_ctx)
            .expect_err("seat error");
        assert_eq!(
            err,
            AskError::Plan("the seat's own refusal at 131072".to_string())
        );
        assert_eq!(card.calls(), [262_144, 131_072]);
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
    /// limit with its source; a halved default adds its target and bytes.
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
            chosen(Rule::Halved {
                from: 262_144,
                short: Short {
                    on: Side::Card,
                    need: 9,
                    room: 8,
                },
            })
            .line(),
            "ctx rule=halved ctx=65536 slots=1 total=65536 limit=262144 (trained) from=262144 \
             card-short need=9 room=8"
        );
        assert_eq!(
            chosen(Rule::Halved {
                from: 262_144,
                short: Short {
                    on: Side::Host,
                    need: 9,
                    room: 8,
                },
            })
            .line(),
            "ctx rule=halved ctx=65536 slots=1 total=65536 limit=262144 (trained) from=262144 \
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
        // A unified pool's shortage alone halves too: the cards, the host and
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
    }
}
