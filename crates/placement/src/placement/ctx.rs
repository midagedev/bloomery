//! The serving seats' shared `--ctx` rules: the bisection the seats'
//! searches run ([`largest`]), the context a slot takes when the flag names
//! none ([`default_slot_ctx_by_card`], over [`fit_of`]'s largest context the
//! plan holds), the prompt-cache guard that bounds it by the budget the plan
//! leaves at each context ([`within_cache`]), the search over a context grid
//! the mimo2 default runs ([`searched`]) and the slot count a `--ctx` flag
//! buys ([`slots_of`]) — more positions on the card push card experts to the
//! host, and decode crawls, so a context the flag did not name never trades
//! away more than the margin of stage-card expert bytes, and never misses the
//! arena on more than [`NVME_SLACK_DIV`]'s fraction of the base's missed
//! expert bytes past them ([`ExpertBytes::miss`]): none at all on a plan whose
//! base misses none. The plan a seat
//! probes at each context is its own (the models' placements differ); these
//! are the terms the seats already share, and every model-specific input —
//! the plan's bytes, the base a slot, the step, the margin, the cap — enters
//! as an argument.
//!
//! A context the flag did not name is decided by the card; the host room may
//! veto it, never choose it. On a plan that pages nothing at the base, a
//! deeper context's missed bytes are its whole host need past the room, so a
//! rule that asked only "does the deeper plan still fit the room read" would
//! ride the room's slack one step at a time (a checkpoint a slot every few
//! thousand positions, the KV positions' card experts moved to the host), and
//! two readings of one machine a few hundred megabytes apart would give two
//! defaults. A seat therefore plans each context twice: `card`, at an
//! unbounded host room, and `probe`, at the room read. [`default_slot_ctx_by_card`]
//! takes the context `c*` the margin rule gives over `card` (which pages
//! nothing, so the card clause alone decides), and keeps it only when the
//! `probe` plan at `c*` also pages no more than the base's: the answer is
//! `c*` or the base, both fixed by the card and the file, and the room enters
//! through one comparison, at the host bytes `c*` needs past the base. That
//! holds where the card has memory of its own ([`Veto::Binary`]). Where it
//! allocates from the host's memory ([`Veto::Ladder`], the placement's
//! `Machine.unified`) no answer of the card is independent of the room: the
//! pool is the card's room, a plan the pool cannot hold is refused for the
//! card too, and `c*` would ride the reading wherever the cap does not fit.
//! There the answer is the first rung of the halving ladder from the cap
//! (rounded down to the step) down to the base whose `probe` plan stands and
//! is within the base's, so every answer is one of a few contexts fixed by the
//! cap and the base, and the reading moves it only where it crosses one rung's
//! need. A base that pages already (the room is under its need) or a card that
//! holds under the base runs the one-probe rule ([`default_slot_ctx`]) over
//! `probe` alone: there the room cancels out of a deeper context's extra
//! missed bytes, which are its extra host need, so the answer does not ride
//! it.
//!
//! The prompt cache's bound ([`within_cache`]) is the same class: its budget
//! is half of what the host's reading leaves past the plan's need, so a bound
//! that took the largest context whose sequence state holds the budget would
//! ride the reading at every step. It takes the halving ladder too: the
//! answer is the first rung, from the default down to the base, whose state
//! holds the budget there, and it changes only where the budget crosses one
//! rung's state.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt;

use super::Machine;

/// Why a slot count was refused.
#[derive(Debug, thiserror::Error)]
pub enum CtxError {
    /// A seat's own parse refuses `--parallel 0` before the rule; the rule
    /// refuses it too, so no call can serve a count of no slot.
    #[error("--parallel 0: the server serves no slot")]
    NoSlot,
    /// The prompt cache's budget holds no sequence state at all: at one
    /// position the state takes `state` bytes and the budget is `ram`.
    #[error(
        "the prompt cache's {ram} bytes hold no sequence state ({state} bytes at one position, \
         two recurrent copies among its terms); --cache-ram 0 turns it off"
    )]
    NoCacheState { state: u64, ram: u64 },
}

/// The resident sequences a seat serves and what set the count, for its
/// `parallel` line's `from` word: `parallel` (`--parallel`) as given
/// (`flag`); unset beside a set `--ctx`, one slot at the whole context —
/// a `--ctx-size` is what one request gets, as llama-server reads it
/// (`ctx`); unset with no context named, the seat's own default
/// (`seat_default`) over its automatic context (`default`).
///
/// # Errors
/// [`CtxError::NoSlot`] when the count comes out as zero.
pub fn slots_of(
    parallel: Option<usize>,
    ctx_set: bool,
    seat_default: usize,
) -> Result<(usize, &'static str), CtxError> {
    let (slots, from) = match parallel {
        Some(n) => (n, "flag"),
        None if ctx_set => (1, "ctx"),
        None => (seat_default, "default"),
    };
    if slots == 0 {
        return Err(CtxError::NoSlot);
    }
    Ok((slots, from))
}

/// The largest `c` in `lo..=hi` for which `ok` holds, `ok(lo)` given: `ok`
/// holds below a point and not above it.
///
/// # Errors
/// The first error `ok` returns.
pub fn largest<E>(
    mut lo: usize,
    mut hi: usize,
    ok: impl Fn(usize) -> Result<bool, E>,
) -> Result<usize, E> {
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if ok(mid)? {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Ok(lo)
}

/// A plan's routed-expert bytes by tier at one context, what the default
/// rule counts; the host's share is the rest of the experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertBytes {
    /// Experts resident on the stage card.
    pub card: u64,
    /// Experts the plan leaves on the NVMe tier.
    pub nvme: u64,
    /// The RAM arena the plan leaves that tier ([`ExpertBytes::miss`]'s
    /// other term): the plan's `nvme_arena_bytes`, 0 on a plan that pages
    /// nothing or holds no arena.
    pub arena: u64,
}

/// The divisor of the missed expert bytes a deeper default context may add
/// to the base's: at most `miss(base) / NVME_SLACK_DIV`.
///
/// When NVMe reads bound the decode step, reading `Δ` more bytes from the
/// tier slows it by at most `Δ / miss(base)`: routing spreads uniformly over
/// the paged experts, the arena holds its share of them, and every missed
/// byte adds its share of the step's reads and no more. A deeper context
/// moves the missed bytes by two terms that add: the experts the plan moves
/// to the tier (`Δnvme`), and the arena it gives up (`-Δarena`), because the
/// KV positions and the checkpoints a slot reserves on the host (one more
/// point for each ubatch of the prompt) come out of the room the arena is
/// carved from. A hundredth is a decode cost under the A/B ruler (same-binary
/// SD 0.6 %, a 95 % interval of ±1.0 % at four rounds), so the default buys
/// positions at a price no A/B can see, and a plan that misses nothing at the
/// base misses nothing at a deeper context (`0 / NVME_SLACK_DIV`).
pub const NVME_SLACK_DIV: u64 = 100;

impl ExpertBytes {
    /// The expert bytes a step reads from the NVMe tier past the arena: the
    /// tier's bytes less the arena that holds a share of them.
    #[must_use]
    pub fn miss(self) -> u64 {
        self.nvme.saturating_sub(self.arena)
    }

    /// The missed expert bytes a plan deeper than this one (the base's) may
    /// add: [`NVME_SLACK_DIV`]'s fraction of [`ExpertBytes::miss`], 0 when
    /// the base misses none.
    #[must_use]
    pub fn nvme_bound(self) -> u64 {
        self.miss() / NVME_SLACK_DIV
    }

    /// Whether this plan, a deeper context's, is within `margin` of `base`'s:
    /// at most `margin` fewer card expert bytes and at most
    /// [`ExpertBytes::nvme_bound`] more missed expert bytes. The one owner of
    /// "within" for the default rule ([`default_slot_ctx`],
    /// [`default_slot_ctx_by_card`]).
    fn within(self, base: Self, margin: u64) -> bool {
        base.card.saturating_sub(self.card) <= margin
            && self.miss() <= base.miss().saturating_add(base.nvme_bound())
    }
}

/// The plan is refused at `at` positions a slot, `source` the probe's own
/// error: at 1 position no context fits the card, at the base the default's
/// plan is not the one the fit promised.
#[derive(Debug, PartialEq, Eq)]
pub struct Refused<E> {
    pub at: usize,
    pub source: E,
}

impl<E: fmt::Display> fmt::Display for Refused<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.at == 1 {
            write!(
                f,
                "no context fits the card: the plan at 1 position a slot is refused ({})",
                self.source
            )
        } else {
            write!(
                f,
                "the plan at {} positions a slot is refused ({})",
                self.at, self.source
            )
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for Refused<E> {}

/// A seat's probe, each context planned once: the fit search, the base's
/// plan and the margin scan share the plan solves they ask of one context.
struct Probes<'a, E> {
    plan: &'a dyn Fn(usize) -> Result<ExpertBytes, E>,
    seen: RefCell<BTreeMap<usize, ExpertBytes>>,
}

impl<'a, E> Probes<'a, E> {
    fn new(plan: &'a dyn Fn(usize) -> Result<ExpertBytes, E>) -> Self {
        Self {
            plan,
            seen: RefCell::new(BTreeMap::new()),
        }
    }

    /// The plan's bytes at `ctx`, solved the first time it is asked.
    fn at(&self, ctx: usize) -> Result<ExpertBytes, E> {
        if let Some(&bytes) = self.seen.borrow().get(&ctx) {
            return Ok(bytes);
        }
        let bytes = (self.plan)(ctx)?;
        self.seen.borrow_mut().insert(ctx, bytes);
        Ok(bytes)
    }

    /// The plan's bytes at `ctx`, its refusal named by where it came.
    fn named(&self, ctx: usize) -> Result<ExpertBytes, Refused<E>> {
        self.at(ctx).map_err(|source| Refused { at: ctx, source })
    }
}

/// [`fit_of`] over the shared probes.
fn fit_probed<E>(cap: usize, probes: &Probes<'_, E>) -> Result<(usize, ExpertBytes), Refused<E>> {
    assert!(cap > 0, "a context cap of 0");
    let at_cap = probes.named(cap);
    if cap == 1 || at_cap.is_ok() {
        return at_cap.map(|bytes| (cap, bytes));
    }
    // The plan stands at one position or nowhere: its error is the refusal.
    probes.named(1)?;
    let Ok(fit) = largest(1, cap - 1, |c| Ok::<_, Infallible>(probes.at(c).is_ok()));
    Ok((fit, probes.named(fit)?))
}

/// The largest context up to `cap` whose plan stands, and that plan's bytes:
/// `cap` when its plan does, else the bisection's — a plan that is refused
/// stays refused as the context deepens. One probe when the cap fits, and
/// `⌈log2 cap⌉` more past the probe at 1 when it does not.
///
/// # Errors
/// [`Refused`] at 1 when no context fits the card.
///
/// # Panics
/// On a `cap` of 0.
pub fn fit_of<E>(
    cap: usize,
    probe: &dyn Fn(usize) -> Result<ExpertBytes, E>,
) -> Result<(usize, ExpertBytes), Refused<E>> {
    fit_probed(cap, &Probes::new(probe))
}

/// The multiple of `step` up to `fit` the margin allows, and its plan's
/// bytes: `fit` when its plan is within `margin` of `base`'s (the plan at
/// `base_at`), `base_at` when no step past it is, else the largest multiple
/// of `step` above `base_at` whose plan is. A probe's error is a context
/// that is not within. `base_at` is the context `base` was planned at; it
/// is at or under `fit`, and the base's own plan is within by definition,
/// so it is never probed.
///
/// Bisection, after one probe at `fit`: [`ExpertBytes::within`] is a prefix
/// of the contexts when the probe is monotone (see [`default_slot_ctx`]).
fn scan<E>(
    fit: usize,
    base_at: usize,
    base: ExpertBytes,
    step: usize,
    margin: u64,
    probes: &Probes<'_, E>,
) -> (usize, ExpertBytes) {
    if fit <= base_at {
        return (base_at, base);
    }
    // The last context probed within, which a bisection moves only upward.
    let best = Cell::new((base_at, base));
    let within = |ctx: usize| {
        let Ok(plan) = probes.at(ctx) else {
            return false;
        };
        let holds = plan.within(base, margin);
        if holds {
            best.set((ctx, plan));
        }
        holds
    };
    if within(fit) {
        return best.get();
    }
    // Step `i` is the `i`-th multiple of `step` past the base's; step 0 is
    // the base itself, which `largest` never probes. The steps under `fit`
    // are numbered up to `(fit - 1) / step - base_at / step` (the fit's own
    // probe is above).
    let steps = (fit - 1) / step - base_at / step;
    let at = |i: usize| (base_at / step + i) * step;
    let Ok(_) = largest(0, steps, |i| Ok::<_, Infallible>(within(at(i))));
    best.get()
}

/// What decided a default context ([`SlotCtx::rule`]): the card holds fewer
/// than a base context.
pub const RULE_CARD: &str = "card";
/// No multiple of the step past the base is within the margin.
pub const RULE_BASE: &str = "base";
/// The fit itself is within the margin (under [`Veto::Ladder`], the ladder's
/// top rung, the cap rounded down to the step, is).
pub const RULE_FIT: &str = "fit";
/// The margin binds: the largest multiple of the step within it.
pub const RULE_MARGIN: &str = "margin";
/// The host room vetoed the card's deeper context: its plan at the room read
/// pages more than the base's does, or is refused, so the base stands.
pub const RULE_HOST: &str = "host";
/// The unified pool's reading took a lower rung of the halving ladder than the
/// top one ([`Veto::Ladder`]), the base itself the lowest.
pub const RULE_POOL: &str = "pool";

/// How far a default the host room vetoed falls ([`default_slot_ctx_by_card`]):
/// the placement's fact, never a model's ([`Veto::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Veto {
    /// The card has memory of its own: the card's context stands whole or the
    /// base is the answer ([`RULE_HOST`]).
    Binary,
    /// The card allocates from the host's memory (`Machine.unified`): the
    /// pool is the card's room, so a card answer independent of the room does
    /// not exist (a plan the pool cannot hold is refused for the card too).
    /// The answer is the first rung of the halving ladder ([`rungs`]) from
    /// the cap that the plan at the room read holds ([`RULE_POOL`] below the
    /// cap's), and the card probe is not asked.
    Ladder,
}

impl Veto {
    /// The veto of `machine`: the ladder where its card shares the host's
    /// memory, the one comparison where each card has memory of its own. The
    /// one owner of the mapping: a seat hands its machine over and picks no
    /// word.
    #[must_use]
    pub fn of(machine: &Machine) -> Veto {
        if machine.unified.is_some() {
            Veto::Ladder
        } else {
            Veto::Binary
        }
    }
}

/// The halving ladder from `from` down to `floor` (`floor <= from`): `from`
/// first, then `from >> k` for `k = 1, 2, …` rounded down to a multiple of
/// `step` and raised to `floor` when under it, ending at the first rung equal
/// to `floor`; a rung equal to the last is skipped. `from == floor` is the one
/// rung. The `k`-th rung is under half the last until the floor, so the ladder
/// has `⌈log2(from / floor)⌉ + 1` rungs at most.
///
/// # Panics
/// On a `step` of 0, or a `floor` over `from`.
fn rungs(from: usize, floor: usize, step: usize) -> Vec<usize> {
    assert!(step > 0, "a context step of 0");
    assert!(floor <= from, "a ladder floor {floor} over its top {from}");
    let mut out = vec![from];
    let mut last = from;
    let mut k = 1;
    while last > floor {
        let rung = ((from >> k) / step * step).max(floor);
        if rung < last {
            out.push(rung);
            last = rung;
        }
        k += 1;
    }
    out
}

/// What [`default_slot_ctx`] chose and the terms it chose by: a seat prints
/// them as one line ([`SlotCtx::line`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotCtx {
    /// The context a slot takes. A seat that bounds it further (a prompt
    /// cache's room, a set `--ctx`) sets it, [`SlotCtx::rule`] and
    /// [`SlotCtx::chosen`] at the new context, and leaves the scan's terms
    /// below as they are.
    pub ctx: usize,
    /// What set it: [`RULE_CARD`], [`RULE_BASE`], [`RULE_FIT`],
    /// [`RULE_MARGIN`], [`RULE_HOST`] or [`RULE_POOL`], or the seat's own word
    /// for a bound it applied.
    pub rule: &'static str,
    /// The most a slot may take, as the seat's model gives it (the trained
    /// context, capped where the engine's is).
    pub cap: usize,
    /// The largest context a slot holds up to `cap` ([`fit_of`]) and its
    /// plan's bytes: the card's, at an unbounded host room, where the card
    /// decided ([`default_slot_ctx_by_card`]).
    pub fit: usize,
    pub fit_bytes: ExpertBytes,
    /// The scan's answer: `ctx` as the rule returns it, except under
    /// [`RULE_HOST`], where it is the card's context the host room vetoed.
    /// Under [`Veto::Ladder`] no card context exists beside the answer, and
    /// it is `ctx`.
    pub margin_ctx: usize,
    /// The context the margin counts against, `base` or `fit` when that is
    /// fewer, and its plan's bytes at the room read.
    pub base_ctx: usize,
    pub base: ExpertBytes,
    /// The plan's bytes at `ctx`, at the room read.
    pub chosen: ExpertBytes,
    /// The margin the rule held `chosen` to.
    pub margin: u64,
    /// The missed expert bytes the rule let `chosen` add to the base's
    /// ([`ExpertBytes::nvme_bound`] of `base`).
    pub nvme_bound: u64,
}

impl SlotCtx {
    /// The `ctx` line a seat prints on stderr: every context a slot's,
    /// `total` the sum of the `slots`', `base` the context the margin counts
    /// against, `lost_bytes` the card expert bytes the chosen plan holds
    /// fewer than the base's, `nvme_bound` the missed expert bytes it may add
    /// to the base's: `nvme_expert_bytes` less `arena` against
    /// `base_nvme_expert_bytes` less `base_arena`.
    #[must_use]
    pub fn line(&self, slots: usize) -> String {
        format!(
            "ctx rule={} ctx={} slots={slots} total={} cap={} fit={} fit_card_expert_bytes={} \
             margin_ctx={} base={} base_card_expert_bytes={} base_nvme_expert_bytes={} \
             card_expert_bytes={} nvme_expert_bytes={} arena={} base_arena={} nvme_bound={} \
             lost_bytes={} margin_bytes={}",
            self.rule,
            self.ctx,
            slots * self.ctx,
            self.cap,
            self.fit,
            self.fit_bytes.card,
            self.margin_ctx,
            self.base_ctx,
            self.base.card,
            self.base.nvme,
            self.chosen.card,
            self.chosen.nvme,
            self.chosen.arena,
            self.base.arena,
            self.nvme_bound,
            self.base.card.saturating_sub(self.chosen.card),
            self.margin
        )
    }
}

/// The terms every form of the default rule shares: the cap, the base a slot,
/// the step the answer moves in and the card expert bytes it may trade.
#[derive(Clone, Copy)]
struct Rule {
    cap: usize,
    base: usize,
    step: usize,
    margin: u64,
}

impl Rule {
    /// The answer over the one probe `probes`, given its fit and its plan at
    /// the base (`min(base, fit)`).
    fn over<E>(
        self,
        probes: &Probes<'_, E>,
        (fit, fit_bytes): (usize, ExpertBytes),
        at_base: ExpertBytes,
    ) -> SlotCtx {
        let Rule {
            cap,
            base,
            step,
            margin,
        } = self;
        let base_ctx = base.min(fit);
        let (ctx, rule, chosen) = if fit < base {
            (fit, RULE_CARD, at_base)
        } else {
            let (ctx, chosen) = scan(fit, base_ctx, at_base, step, margin, probes);
            let rule = match ctx {
                c if c == base_ctx => RULE_BASE,
                c if c == fit => RULE_FIT,
                _ => RULE_MARGIN,
            };
            (ctx, rule, chosen)
        };
        SlotCtx {
            ctx,
            rule,
            cap,
            fit,
            fit_bytes,
            margin_ctx: ctx,
            base_ctx,
            base: at_base,
            chosen,
            margin,
            nvme_bound: at_base.nvme_bound(),
        }
    }

    /// The answer where the base stands at the read room and pages nothing
    /// and the card holds the base (`card_fit`, over `cards`): the margin
    /// rule's context over `cards` if `probes`' plan there pages no more than
    /// the base's, else the base ([`RULE_HOST`]).
    fn by_card<E>(
        self,
        (probes, cards): (&Probes<'_, E>, &Probes<'_, E>),
        (card_fit, card_fit_bytes): (usize, ExpertBytes),
        at_base: ExpertBytes,
    ) -> Result<SlotCtx, Refused<E>> {
        let Rule {
            cap,
            base,
            step,
            margin,
        } = self;
        let card_base = cards.named(base)?;
        let (star, _) = scan(card_fit, base, card_base, step, margin, cards);
        let (ctx, rule, chosen) = if star == base {
            (base, RULE_BASE, at_base)
        } else {
            match probes.at(star) {
                Ok(plan) if plan.within(at_base, margin) => {
                    let rule = if star == card_fit {
                        RULE_FIT
                    } else {
                        RULE_MARGIN
                    };
                    (star, rule, plan)
                }
                _ => (base, RULE_HOST, at_base),
            }
        };
        Ok(SlotCtx {
            ctx,
            rule,
            cap,
            fit: card_fit,
            fit_bytes: card_fit_bytes,
            margin_ctx: star,
            base_ctx: base,
            base: at_base,
            chosen,
            margin,
            nvme_bound: at_base.nvme_bound(),
        })
    }
}

impl Rule {
    /// The answer where the base stands at the read room and pages nothing on
    /// a machine whose card shares the host's memory ([`Veto::Ladder`]): the
    /// first rung of the halving ladder from the cap (rounded down to the
    /// step, at least the base) down to the base whose `probes` plan stands
    /// and is within the base's. The card is not asked. A rung above the fit
    /// is refused without a probe (a refused plan stays refused as the context
    /// deepens). The word: [`RULE_FIT`] at the top rung ([`RULE_BASE`] when
    /// that is the base), [`RULE_POOL`] at every rung below it, the base
    /// included.
    fn ladder<E>(
        self,
        probes: &Probes<'_, E>,
        (fit, fit_bytes): (usize, ExpertBytes),
        at_base: ExpertBytes,
    ) -> SlotCtx {
        let Rule {
            cap,
            base,
            step,
            margin,
        } = self;
        let mut answer = (0, base, at_base);
        for (i, rung) in rungs((cap / step * step).max(base), base, step)
            .into_iter()
            .enumerate()
        {
            let held = if rung > fit {
                None
            } else {
                probes
                    .at(rung)
                    .ok()
                    .filter(|plan| plan.within(at_base, margin))
            };
            if let Some(plan) = held {
                answer = (i, rung, plan);
                break;
            }
        }
        let (i, ctx, chosen) = answer;
        let rule = match i {
            0 if ctx == base => RULE_BASE,
            0 => RULE_FIT,
            _ => RULE_POOL,
        };
        SlotCtx {
            ctx,
            rule,
            cap,
            fit,
            fit_bytes,
            margin_ctx: ctx,
            base_ctx: base,
            base: at_base,
            chosen,
            margin,
            nvme_bound: at_base.nvme_bound(),
        }
    }
}

/// The context a slot takes when the flag names none, over one probe, every
/// number a slot's: `fit` ([`fit_of`] up to `cap`) when it is under `base`
/// ([`RULE_CARD`]: the card holds fewer than a base context), else the
/// largest multiple of `step` up to `fit` whose plan holds at most `margin`
/// fewer card expert bytes than the plan at `base` and misses the arena on at
/// most [`ExpertBytes::nvme_bound`] more expert bytes — `fit` when its plan
/// does ([`RULE_FIT`]), `base` when no step past it does ([`RULE_BASE`], and a
/// fit at the base), else the step that binds ([`RULE_MARGIN`]). The miss
/// clause stops a deeper context from buying positions with reads from a tier
/// slower than the host's beyond [`NVME_SLACK_DIV`]'s fraction of the base's,
/// whether it pays with experts moved to the tier or with the arena it gives
/// up: none on a base that misses none.
///
/// A seat whose plan reads a host room does not call this form: over a probe
/// at the room read, a base that misses none makes the miss clause "the
/// deeper plan's whole host need fits the room", and the answer rides the
/// room's reading. That seat plans a second probe at an unbounded room and
/// calls [`default_slot_ctx_by_card`], which runs this form where the room
/// cancels out (a base that pages, a card under the base). This form is the
/// whole rule for a plan that reads no room.
///
/// `probe` plans one context. A context past the base whose plan is refused
/// is not within (a probe is never the seat's refusal), and each context is
/// planned once however many terms ask for it.
///
/// The scan is a bisection over the multiples of `step`, after the probes of
/// the fit search, where a linear scan would take one probe a step: it
/// assumes `probe` is monotone in the context — card expert bytes never
/// rise, missed expert bytes never fall (the NVMe bytes never fall and the
/// arena never grows) and a refused plan stays refused as
/// the context deepens, the solver trading experts for positions only one
/// way — so the contexts within are a prefix. A probe is a plan solve: past
/// the fit search's and the base's, the `n` multiples of `step` between the
/// base and the fit cost at most `⌈log2(n + 1)⌉` more.
///
/// # Errors
/// [`Refused`] when no context fits the card (at 1), or the plan at the
/// base is refused (a probe that is not monotone).
///
/// # Panics
/// On a `cap` or a `step` of 0.
pub fn default_slot_ctx<E>(
    cap: usize,
    base: usize,
    step: usize,
    margin: u64,
    probe: &dyn Fn(usize) -> Result<ExpertBytes, E>,
) -> Result<SlotCtx, Refused<E>> {
    assert!(step > 0, "a context step of 0");
    let probes = Probes::new(probe);
    let (fit, fit_bytes) = fit_probed(cap, &probes)?;
    let at_base = probes.named(base.min(fit))?;
    let rule = Rule {
        cap,
        base,
        step,
        margin,
    };
    Ok(rule.over(&probes, (fit, fit_bytes), at_base))
}

/// The context a slot takes when the flag names none, decided by the card:
/// the host room may veto it, never choose it. Two probes plan a context: `card`
/// at an unbounded host room and `probe` at the room read. `veto` is the
/// placement's fact ([`Veto::of`]).
///
/// 1. `probe`'s fit and its plan at the base come first, as in
///    [`default_slot_ctx`]. If `probe` holds fewer than a base context
///    ([`RULE_CARD`]), or its base plan misses any expert byte
///    ([`ExpertBytes::miss`]: the room is under the base's need, so a deeper
///    context's extra missed bytes are its extra host need and the room
///    cancels out of the miss clause), the answer is [`default_slot_ctx`]'s
///    over `probe` alone.
/// 2. Otherwise the base stands at the room read and pages nothing. Under
///    [`Veto::Ladder`] the card is not asked: the machine's card allocates
///    from the host's memory, a plan its pool cannot hold is refused for the
///    card too, so a card context independent of the room does not exist, and
///    the answer is the first rung of the halving ladder from the cap (rounded
///    down to the step) down to the base whose `probe` plan stands and is
///    within the base's, no expert byte missed past it and no more than the
///    margin of card expert bytes lost ([`ExpertBytes::within`]): the top
///    rung is [`RULE_FIT`] ([`RULE_BASE`] when it is the base), a lower one
///    [`RULE_POOL`]. Every answer is a context the cap and the base fix, and
///    the reading moves it only where it crosses one rung's need.
/// 3. Under [`Veto::Binary`] `card`'s fit is taken; if the card holds under
///    the base the answer is again [`default_slot_ctx`]'s over `probe`.
/// 4. Otherwise `c*` is [`default_slot_ctx`]'s scan over `card` (the fit and
///    the margin from `card`; its plan pages nothing, so the card clause
///    alone decides). When `c*` is the base the answer is the base
///    ([`RULE_BASE`]). Else the `probe` plan at `c*` must stand and be within
///    the margin of `probe`'s base plan, with no expert byte missed past it
///    ([`ExpertBytes::within`]); then the answer is `c*` ([`RULE_FIT`] or
///    [`RULE_MARGIN`]), else the base ([`RULE_HOST`], `c*` kept as
///    [`SlotCtx::margin_ctx`]).
///
/// Stable by construction: in step 4 the answer is the base or `c*`, both
/// fixed by the card and the file; the room enters through one comparison, at
/// `need(c*) - need(base)`, so a reading's noise changes the answer only when
/// the reading falls within that noise of that one threshold. A host whose
/// slack lies between 0 and that threshold gets the base, where
/// [`default_slot_ctx`] gave it an intermediate depth that moved with the
/// reading. In step 2 the answer is a rung, and the reading's noise changes it
/// only within that noise of one rung's need, of which there are `⌈log2(cap /
/// base)⌉ + 1`.
///
/// `fit` and `fit_bytes` are `card`'s in step 4 and `probe`'s in step 2. Each
/// probe plans a context at most once, and `card` is asked only in steps 3 and
/// 4. A plan that reads no room gets [`default_slot_ctx`]'s answer from this
/// form too under [`Veto::Binary`] (the same plan as `card` and `probe`), at
/// twice the probes.
///
/// # Errors
/// [`Refused`] as [`default_slot_ctx`]'s over `probe`; in steps 3 and 4 also
/// `card`'s refusal at 1 or at the base while `probe`'s base stands (the
/// probes disagree: a room cannot make a plan stand).
///
/// # Panics
/// On a `cap` or a `step` of 0.
pub fn default_slot_ctx_by_card<E>(
    cap: usize,
    base: usize,
    step: usize,
    margin: u64,
    veto: Veto,
    card: &dyn Fn(usize) -> Result<ExpertBytes, E>,
    probe: &dyn Fn(usize) -> Result<ExpertBytes, E>,
) -> Result<SlotCtx, Refused<E>> {
    assert!(step > 0, "a context step of 0");
    let rule = Rule {
        cap,
        base,
        step,
        margin,
    };
    let probes = Probes::new(probe);
    let (fit, fit_bytes) = fit_probed(cap, &probes)?;
    let at_base = probes.named(base.min(fit))?;
    if fit >= base && at_base.miss() == 0 {
        if veto == Veto::Ladder {
            return Ok(rule.ladder(&probes, (fit, fit_bytes), at_base));
        }
        let cards = Probes::new(card);
        let (card_fit, card_fit_bytes) = fit_probed(cap, &cards)?;
        if card_fit >= base {
            return rule.by_card((&probes, &cards), (card_fit, card_fit_bytes), at_base);
        }
    }
    Ok(rule.over(&probes, (fit, fit_bytes), at_base))
}

/// What the prompt cache's bound left of a default ([`within_cache`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheBound {
    /// The context a slot takes.
    pub ctx: usize,
    /// Whether the cache's budget holds one sequence state at `ctx`. False
    /// only when no rung holds and `ctx` is the ladder's floor: the cache
    /// keeps the sessions it holds, which are shorter than `ctx`, and refuses
    /// the longer ones (a state over the whole budget is refused).
    pub holds: bool,
}

/// The context a default keeps when the prompt cache must hold one sequence
/// state there: the first rung of the halving ladder from `from` down to
/// `base` whose `state` fits the budget `ram_at` gives at that rung ([`rungs`]:
/// `from`, then `from >> k` rounded down to a multiple of `step` and raised to
/// `base`). When no rung holds the answer is the floor, `base` (or `from`
/// when that is under it), with [`CacheBound::holds`] false: the cache keeps
/// the shorter sessions its budget holds and the caller says so. The budget
/// is read at the rung it bounds, not at `from`: a plan's host need moves
/// with the context (a unified pool counts the positions' bytes, a split
/// moves experts), so the budget a smaller context leaves is larger. `from` is
/// tried first and alone: the budget is not monotone where the plan's tables
/// change tier, and a search that skipped it could pass a holding `from` for
/// a smaller context.
///
/// Why a ladder: the largest context holding the budget changes at every
/// multiple of `step` the budget crosses, so a reading of the host's available
/// bytes that moves by `D` moves the answer by `D / 2` over one step's state,
/// wherever the budget sits. With the ladder the answer changes only where the
/// budget crosses the state of one rung, so it is constant between two
/// consecutive rungs' states, and a reading that moves by `D` moves it only
/// within `D` of one of those few budgets (one per rung, each rung half the
/// last). The floor is the base and not `from`: a run with less memory never
/// gets a deeper context (a floor at `from` would jump between the base and
/// `from` across one budget).
///
/// # Errors
/// [`CtxError::NoCacheState`] when not even one position's state holds and no
/// rung does; the first error `ram_at` returns.
///
/// # Panics
/// On a `step` or a `base` of 0.
pub fn within_cache<E: From<CtxError>>(
    from: usize,
    base: usize,
    step: usize,
    state: &dyn Fn(usize) -> u64,
    ram_at: &dyn Fn(usize) -> Result<u64, E>,
) -> Result<CacheBound, E> {
    assert!(base > 0, "a base context of 0");
    let holds = |n: usize| Ok::<_, E>(state(n) <= ram_at(n)?);
    let floor = base.min(from);
    for ctx in rungs(from, floor, step) {
        if holds(ctx)? {
            return Ok(CacheBound { ctx, holds: true });
        }
    }
    if !holds(1)? {
        return Err(CtxError::NoCacheState {
            state: state(1),
            ram: ram_at(1)?,
        }
        .into());
    }
    Ok(CacheBound {
        ctx: floor,
        holds: false,
    })
}

/// The search a seat's whole-card and placed defaults run over the context
/// grid: `Some(trained)` when the trained context itself fits — probed
/// once, after the floor — else the largest `floor + k·gran` below it that
/// `fits` takes, halved between the floor, which fits, and the trained
/// context, which does not: every caller's `fits` is monotone in the
/// context, so the halving finds that largest one. `None` when the file
/// states no trained context (`trained`) or nothing at `floor` fits; a
/// trained context at or under `floor` is its own answer, unprobed.
///
/// # Errors
/// The first error `fits` returns.
pub fn searched<E>(
    trained: Option<usize>,
    floor: usize,
    gran: usize,
    fits: &dyn Fn(usize) -> Result<bool, E>,
) -> Result<Option<usize>, E> {
    let Some(trained) = trained else {
        return Ok(None);
    };
    if trained <= floor {
        return Ok(Some(trained));
    }
    if !fits(floor)? {
        return Ok(None);
    }
    if fits(trained)? {
        return Ok(Some(trained));
    }
    let mut lo = floor;
    let mut hi = trained;
    while hi - lo > gran {
        let mid = lo + (hi - lo) / 2 / gran * gran;
        if mid == lo {
            break;
        }
        if fits(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(Some(lo))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::convert::Infallible;

    use super::*;

    /// A bisection's probe, counted: a search over `n` points takes at most
    /// `⌈log2 n⌉ + 1` of them, and a linear scan, or a loop that never
    /// closes, is a panic here rather than a hang.
    struct Count(Cell<usize>);

    impl Count {
        fn new() -> Self {
            Self(Cell::new(0))
        }

        fn take(&self) {
            self.0.set(self.0.get() + 1);
            assert!(self.0.get() <= 40, "a search that does not close");
        }
    }

    /// The slot count and its word: `--parallel` as given, one slot at the
    /// whole context beside a set `--ctx` (the flag is one request's
    /// context, not a total the seat's slots split), the seat's own default
    /// with neither, and a count of none refused by name.
    #[test]
    fn slots_of_rows() {
        type Row = (Option<usize>, bool, usize, (usize, &'static str));
        let rows: [Row; 6] = [
            (Some(3), false, 2, (3, "flag")),
            (Some(3), true, 2, (3, "flag")),
            (Some(1), true, 2, (1, "flag")),
            (None, true, 2, (1, "ctx")),
            (None, false, 2, (2, "default")),
            (None, false, 1, (1, "default")),
        ];
        for (parallel, ctx_set, seat, want) in rows {
            let got = slots_of(parallel, ctx_set, seat).expect("a count");
            assert_eq!(got, want, "parallel {parallel:?}, ctx set {ctx_set}");
        }
        for (parallel, ctx_set, seat) in [(Some(0), false, 2), (Some(0), true, 2), (None, false, 0)]
        {
            let refused = slots_of(parallel, ctx_set, seat).expect_err("no slot");
            assert_eq!(
                refused.to_string(),
                "--parallel 0: the server serves no slot"
            );
        }
    }

    /// The bisection finds the last context a monotone `ok` holds at, over a
    /// range, its ends and a one-point range, with `ok(lo)` given (never
    /// probed) and no more probes than a halving takes.
    #[test]
    fn largest_rows() {
        for (lo, hi) in [(1usize, 1usize), (1, 2), (1, 3), (1, 4096), (4096, 262_144)] {
            for edge in [lo, lo + 1, (lo + hi) / 2, hi.saturating_sub(1).max(lo), hi] {
                let edge = edge.clamp(lo, hi);
                let probes = Count::new();
                let got = largest(lo, hi, |c| {
                    probes.take();
                    assert_ne!(c, lo, "ok(lo) is given, never probed");
                    Ok::<_, Infallible>(c <= edge)
                })
                .expect("infallible");
                assert_eq!(got, edge, "range {lo}..={hi}, ok through {edge}");
                let width = (hi - lo + 1) as u64;
                assert!(
                    probes.0.get() <= (width.ilog2() + 1) as usize,
                    "range {lo}..={hi}: {} probes",
                    probes.0.get()
                );
            }
        }
        let refused: Result<usize, &str> = largest(1, 8, |_| Err("refused"));
        assert_eq!(refused, Err("refused"), "a probe's error is the search's");
    }

    /// The cache guard's rows: the first rung of the halving ladder from
    /// `from` down to the base whose state holds the budget there, the budget
    /// shrinking with the context as a unified pool's need grows with it —
    /// each rung `from >> k` rounded down to the step, a rung under the base
    /// the base — and a budget that does not move with the context
    /// (`--cache-ram`) the plain bound; each answer within a probe per rung.
    /// PIN(2026-10-12): the rows (199, 1), (199, 16) and (400, 256) answered
    /// 198, 192 and 198, the largest context that holds, and answer rungs now
    /// (99, 96 and 16); the constant budget's rows (262,144 and 31, step 256)
    /// answered 30, the largest context that holds, and answer the base 16.
    #[test]
    fn within_cache_rows() {
        let state = |n: usize| 3 * n as u64 + 10;
        // Holds while 3n + 10 <= 1000 - 2n: through 198.
        let shrinking = |n: usize| Ok::<_, Box<dyn std::error::Error>>(1000 - 2 * n as u64);
        let rows: [(usize, usize, usize); 6] = [
            (150, 16, 150),
            (198, 16, 198),
            (199, 1, 99),
            (199, 16, 96),
            (400, 16, 192),
            (400, 256, 16),
        ];
        for (from, step, want) in rows {
            let probes = Count::new();
            let counted = |n: usize| {
                probes.take();
                shrinking(n)
            };
            let got = within_cache(from, 16, step, &state, &counted).expect("a context");
            assert_eq!(
                got,
                CacheBound {
                    ctx: want,
                    holds: true
                },
                "from {from}, step {step}"
            );
            assert!(state(got.ctx) <= 1000 - 2 * got.ctx as u64);
            assert!(
                probes.0.get() <= rungs(from, 16.min(from), step).len(),
                "from {from}, step {step}: {} probes",
                probes.0.get()
            );
        }
        // A constant budget of 100 B holds through 30 positions. A `from`
        // that holds is its own; over it the base 16 holds (58 B) and is the
        // first rung that does; over the base 64 no rung holds (202 B), and
        // the base is the answer with the budget's say that it holds none.
        let constant = |_: usize| Ok::<_, Box<dyn std::error::Error>>(100u64);
        let kept = |ctx| CacheBound { ctx, holds: true };
        for (from, base, want) in [
            (30, 16, kept(30)),
            (262_144, 16, kept(16)),
            (31, 16, kept(16)),
            (10, 16, kept(10)),
            (
                262_144,
                64,
                CacheBound {
                    ctx: 64,
                    holds: false,
                },
            ),
        ] {
            let got = within_cache(from, base, 256, &state, &constant).expect("a context");
            assert_eq!(got, want, "from {from}, base {base}");
        }
    }

    /// The ladder's rungs: `from`, then `from >> k` rounded down to a multiple
    /// of the step and raised to the floor, ending at the first rung equal to
    /// the floor; the trained 262,144 over 4096 in steps of 256 is the seven
    /// powers of two, and a top under twice the floor is two rungs.
    #[test]
    fn rungs_rows() {
        assert_eq!(
            rungs(262_144, 4096, 256),
            [262_144, 131_072, 65_536, 32_768, 16_384, 8192, 4096]
        );
        assert_eq!(
            rungs(196_608, 4096, 256),
            [196_608, 98_304, 49_152, 24_576, 12_288, 6144, 4096]
        );
        assert_eq!(rungs(5000, 4096, 256), [5000, 4096]);
        assert_eq!(rungs(4096, 4096, 256), [4096]);
        assert_eq!(
            rungs(400, 16, 256),
            [400, 16],
            "a half under the step is the floor"
        );
        assert_eq!(rungs(199, 1, 1), [199, 99, 49, 24, 12, 6, 3, 1]);
    }

    /// `from` is tried first: a budget that holds at `from` and fails in
    /// between (the plan's table changing tier at a context) keeps `from`,
    /// where a search over the contexts under it lands under the gap.
    /// PIN(2026-10-12): (239, 1) answered 100, the largest context that
    /// holds; the ladder from 239 over the base 64 answers 64, the first rung
    /// that holds.
    #[test]
    fn within_cache_tries_from_before_the_ladder() {
        let state = |n: usize| n as u64;
        let gap = |n: usize| {
            Ok::<_, Box<dyn std::error::Error>>(if n <= 100 || n >= 240 { 10_000 } else { 0 })
        };
        let kept = |ctx| CacheBound { ctx, holds: true };
        assert_eq!(
            within_cache(250, 64, 1, &state, &gap).expect("a context"),
            kept(250)
        );
        assert_eq!(
            within_cache(239, 64, 1, &state, &gap).expect("a context"),
            kept(64)
        );
    }

    /// The cache guard's refusals, by name: no context holds one position's
    /// state, and a budget read that fails. A budget that holds one position
    /// but no rung is not a refusal: the base is the answer and the cache
    /// keeps the shorter sessions.
    /// PIN(2026-10-12): the rounding refusal (`CacheRounding`, a bisection's
    /// context rounded to the step where the state did not hold) is gone with
    /// the bisection; its case, a budget holding 1, 5 and 7 positions only,
    /// answers the base 4 with `holds` false.
    #[test]
    fn within_cache_refuses_by_name() {
        let state = |n: usize| 3 * n as u64 + 10;
        let none = |_: usize| Ok::<_, Box<dyn std::error::Error>>(5u64);
        let err = within_cache(1000, 16, 16, &state, &none).expect_err("no state holds");
        assert_eq!(
            err.downcast_ref::<CtxError>().map(ToString::to_string),
            Some(
                "the prompt cache's 5 bytes hold no sequence state (13 bytes at one position, two \
                 recurrent copies among its terms); --cache-ram 0 turns it off"
                    .to_string()
            )
        );
        // Holds at 1, 5 and 7 only: no rung of 9 over the base 4 holds (9 is
        // not a multiple of the step 4; the rungs are 9 and 4), and one
        // position's does.
        let state = |n: usize| n as u64;
        let sparse = |n: usize| {
            Ok::<_, Box<dyn std::error::Error>>(if matches!(n, 1 | 5 | 7) { 100 } else { 0 })
        };
        let got = within_cache(9, 4, 4, &state, &sparse).expect("the base");
        assert_eq!(
            got,
            CacheBound {
                ctx: 4,
                holds: false
            }
        );
        let refused = |n: usize| -> Result<u64, Box<dyn std::error::Error>> {
            if n < 100 {
                Ok(1_000)
            } else {
                Err("refused".into())
            }
        };
        let err = within_cache(200, 4, 1, &state, &refused).expect_err("a plan refused");
        assert_eq!(err.to_string(), "refused");
    }

    /// The default rule's base context, step and margin, and a plan whose
    /// card holds `CARD` there.
    const AT: usize = 4096;
    const STEP: usize = 256;
    const MARGIN: u64 = 1_000;
    const CARD: u64 = 1_000_000;

    /// A plan's bytes at a context: the card holds `card` at `at` and loses
    /// `per` bytes a position past it, to none at the least; the NVMe tier
    /// holds `nvme(c)`.
    fn plan(
        card: u64,
        at: usize,
        per: u64,
        nvme: impl Fn(usize) -> u64,
    ) -> impl Fn(usize) -> Result<ExpertBytes, String> {
        move |c| {
            Ok(ExpertBytes {
                card: card.saturating_sub((c.saturating_sub(at) as u64) * per),
                nvme: nvme(c),
                arena: 0,
            })
        }
    }

    /// `probe` as a card refuses it past `at`, whatever its bytes.
    fn refused_past(
        at: usize,
        probe: impl Fn(usize) -> Result<ExpertBytes, String>,
    ) -> impl Fn(usize) -> Result<ExpertBytes, String> {
        move |c| {
            if c > at {
                Err("refused".to_owned())
            } else {
                probe(c)
            }
        }
    }

    /// `probe` at an unbounded host room: the same plan, with no expert on
    /// the NVMe tier and no arena (a plan pages for want of room, and a given
    /// room of every byte pages nothing); a refusal stays a refusal.
    fn unbounded(
        probe: &impl Fn(usize) -> Result<ExpertBytes, String>,
    ) -> impl Fn(usize) -> Result<ExpertBytes, String> + '_ {
        move |c| {
            probe(c).map(|bytes| ExpertBytes {
                nvme: 0,
                arena: 0,
                ..bytes
            })
        }
    }

    /// The default up to `cap` over `probe` and its unbounded twin
    /// ([`unbounded`]), with the most probes either asked for (each probe is
    /// held to the one-probe rule's bound alone).
    fn run(
        cap: usize,
        step: usize,
        margin: u64,
        probe: impl Fn(usize) -> Result<ExpertBytes, String>,
    ) -> (Result<SlotCtx, Refused<String>>, usize) {
        let (read, card) = (Count::new(), Count::new());
        let free = unbounded(&probe);
        let counted_read = |c: usize| {
            read.take();
            probe(c)
        };
        let counted_card = |c: usize| {
            card.take();
            free(c)
        };
        let got = default_slot_ctx_by_card(
            cap,
            AT,
            step,
            margin,
            Veto::Binary,
            &counted_card,
            &counted_read,
        );
        (got, read.0.get().max(card.0.get()))
    }

    /// The probes a default may take over a context range: the fit search's
    /// at the cap, the base's and a bisection's `⌈log2(n + 1)⌉` over the `n`
    /// multiples of `step` between them, counted here one at a time.
    fn probe_bound(fit: usize, step: usize) -> usize {
        let n = (1..)
            .map(|i| (AT / step + i) * step)
            .take_while(|&c| c < fit)
            .count();
        2 + (n + 1).next_power_of_two().ilog2() as usize
    }

    /// The largest context a plan stands at, `edge`: its card holds
    /// `CARD - c` bytes at `c`, and a plan past it is refused.
    fn stands_to(edge: usize) -> impl Fn(usize) -> Result<ExpertBytes, String> {
        move |c| {
            if c <= edge {
                Ok(ExpertBytes {
                    card: CARD - c as u64,
                    nvme: 0,
                    arena: 0,
                })
            } else {
                Err(format!("past {edge}"))
            }
        }
    }

    /// The fit is the largest context up to the cap whose plan stands, with
    /// that plan's bytes: the cap on one probe when its plan does, else the
    /// edge of what stands after the probe at 1 and a halving; no context
    /// standing is the refusal at 1 with the probe's own error, and a cap of
    /// 1 that is refused costs the one probe.
    #[test]
    fn fit_is_the_largest_context_the_plan_stands_at() {
        for edge in [1usize, 2, 4095, 4096, 9000, 100_000, 262_143] {
            let count = Count::new();
            let probe = |c: usize| {
                count.take();
                stands_to(edge)(c)
            };
            let (fit, bytes) = fit_of(262_144, &probe).expect("a fit");
            assert_eq!(fit, edge, "the plan stands through {edge}");
            assert_eq!(bytes.card, CARD - edge as u64, "the plan's bytes at {edge}");
            assert!(count.0.get() <= 2 + 18, "{} probes", count.0.get());
        }
        let count = Count::new();
        let probe = |c: usize| {
            count.take();
            stands_to(262_144)(c)
        };
        let (fit, _) = fit_of(262_144, &probe).expect("a fit");
        assert_eq!((fit, count.0.get()), (262_144, 1), "the cap's own plan");
        let count = Count::new();
        let none = |c: usize| {
            count.take();
            Err::<ExpertBytes, _>(format!("none at {c}"))
        };
        assert_eq!(
            fit_of(262_144, &none),
            Err(Refused {
                at: 1,
                source: "none at 1".to_owned()
            }),
            "the refusal at one position is the probe's own"
        );
        assert_eq!(count.0.get(), 2, "the cap and one position");
        let count = Count::new();
        let none = |c: usize| {
            count.take();
            Err::<ExpertBytes, _>(format!("none at {c}"))
        };
        assert_eq!(fit_of(1, &none).expect_err("no fit").at, 1);
        assert_eq!(count.0.get(), 1, "a cap of one position, refused once");
    }

    /// A cap of 0 is refused by name, not by the subtraction it would reach.
    #[test]
    #[should_panic(expected = "a context cap of 0")]
    fn fit_of_a_cap_of_zero_is_refused_by_name() {
        let _ = fit_of(0, &stands_to(10));
    }

    /// The refusal reads by where it came: nothing fits the card at 1
    /// position, or the base's plan is not the one the fit promised.
    #[test]
    fn refused_names_where_the_plan_stopped() {
        let at = |at, source: &str| {
            Refused {
                at,
                source: source.to_owned(),
            }
            .to_string()
        };
        assert_eq!(
            at(1, "no room"),
            "no context fits the card: the plan at 1 position a slot is refused (no room)"
        );
        assert_eq!(
            at(4096, "no room"),
            "the plan at 4096 positions a slot is refused (no room)"
        );
    }

    /// A fit under the base is the answer, off the step as it is, by the
    /// `card` rule: the one probe is the cap's, which is the base's and the
    /// chosen plan's too; a fit at the base is the base's, with no step past
    /// it.
    #[test]
    fn default_ctx_under_the_base_is_the_fit_by_the_card_rule() {
        let (got, probes) = run(3000, STEP, MARGIN, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule, got.base_ctx), (3000, "card", 3000));
        assert_eq!(got.base, got.chosen, "the plan at the fit is both");
        assert_eq!(got.fit_bytes, got.chosen, "and the fit's");
        assert_eq!(probes, 1);
        let (got, probes) = run(AT - 1, STEP, MARGIN, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule), (AT - 1, "card"));
        assert_eq!(probes, 1);
        // PIN(2026-10-11): a fit at the base is `base`, not `margin`. The
        // rule's words are one set in one order ([`default_ctx_words`]); the
        // base is the answer whenever no step past it is within, a fit at it
        // included. Re-pin by specification, not a relaxation: the context
        // and the probe count are the ones this row held.
        let (got, probes) = run(AT, STEP, MARGIN, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule, got.base_ctx), (AT, "base", AT));
        assert_eq!(probes, 1, "a fit at the base has nothing to scan");
    }

    /// The card margin binds: the largest multiple of the step within it, a
    /// multiple of the step itself (not the base plus whole steps, a base
    /// off the step has its own), never under the base, over a short range
    /// and a wide one (a linear scan is a panic past the probe cap). No
    /// context here sits at the margin's last byte (the next test's).
    #[test]
    fn default_ctx_margin_binds_at_the_last_step_within_it() {
        // The card loses 1 B a position past the base: 1,000 B of margin
        // holds through 5096.
        let rows = [
            (
                9000,
                256,
                4864,
                "margin",
                "5096 rounded down to a step of 256",
            ),
            (
                9000,
                1024,
                4096,
                "base",
                "no whole step past the base holds",
            ),
            (
                9000,
                100,
                5000,
                "margin",
                "a multiple of 100 itself, off the base's 4096",
            ),
            (262_144, 256, 4864, "margin", "the wide range"),
        ];
        for (cap, step, want, rule, why) in rows {
            let (got, probes) = run(cap, step, MARGIN, plan(CARD, AT, 1, |_| 0));
            let got = got.expect("a plan");
            assert_eq!((got.ctx, got.rule), (want, rule), "{why}");
            assert_eq!(got.margin_ctx, want, "{why}");
            assert_eq!(got.base_ctx, AT, "{why}");
            assert_eq!(got.fit, cap, "{why}");
            let at = |c| plan(CARD, AT, 1, |_| 0)(c).expect("a plan");
            assert_eq!(got.base, at(AT), "{why}");
            assert_eq!(got.fit_bytes, at(cap), "the plan at the fit: {why}");
            assert_eq!(
                got.chosen,
                at(want),
                "the plan at the chosen context: {why}"
            );
            assert!(probes <= probe_bound(cap, step), "{why}: {probes} probes");
        }
    }

    /// Every context within is the fit — off the step, and at the margin's
    /// last byte (at most `margin` fewer is within) — after the cap's probe
    /// and the base's alone; one byte of margin less and the answer is the
    /// last step.
    #[test]
    fn default_ctx_every_context_within_is_the_fit() {
        let lost_at_fit = 9000 - AT as u64;
        let (got, probes) = run(9000, STEP, lost_at_fit, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule), (9000, "fit"));
        assert_eq!(got.chosen.card, CARD - lost_at_fit);
        assert_eq!(probes, 2, "the cap and the base");
        let (got, _) = run(9000, STEP, lost_at_fit - 1, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!(
            (got.ctx, got.rule),
            (8960, "margin"),
            "8999 rounded down to a step"
        );
    }

    /// A plan the card refuses past a context bounds the fit there, and the
    /// scan reads a refused step below it as not within, whatever its bytes
    /// (a probe is never the seat's refusal).
    #[test]
    fn default_ctx_refused_plan_bounds_the_fit_and_is_not_within() {
        // PIN(2026-10-11): the cap, not the fit, is what a seat hands the
        // rule now, so a card that refuses past 4600 gives the fit of 4600
        // (the margin holds through 5096, so the fit is the answer) where the
        // fit was a given 9000 and 4600 rounded down to 4352. Re-pin by
        // specification: the refusal still bounds the default, one step
        // higher where the fit is exact.
        let (got, _) = run(
            9000,
            STEP,
            MARGIN,
            refused_past(4600, plan(CARD, AT, 1, |_| 0)),
        );
        let got = got.expect("a plan");
        assert_eq!((got.fit, got.ctx, got.rule), (4600, 4600, "fit"));
        // A refusal past 8999 leaves the margin's answer under a fit of 8999.
        let (got, _) = run(
            9000,
            STEP,
            MARGIN,
            refused_past(8999, plan(CARD, AT, 1, |_| 0)),
        );
        let got = got.expect("a plan");
        assert_eq!((got.fit, got.ctx, got.rule), (8999, 4864, "margin"));
        // One step refused inside the range (a probe that is not monotone) is
        // not within: the scan's bisection falls below it, 4864 to 4608.
        let hole = |c: usize| {
            if c == 4864 {
                Err("refused".to_owned())
            } else {
                plan(CARD, AT, 1, |_| 0)(c)
            }
        };
        let (got, _) = run(9000, STEP, MARGIN, hole);
        let got = got.expect("a plan");
        assert_eq!((got.fit, got.ctx, got.rule), (9000, 4608, "margin"));
    }

    /// A whole-card load — every expert on the card, nothing on the NVMe tier,
    /// at every context — is within a margin of none: the fit, after two
    /// probes, however wide the range.
    #[test]
    fn default_ctx_whole_card_load_is_the_fit() {
        for cap in [5000, 9000, 262_144] {
            let (got, probes) = run(cap, STEP, 0, |_| {
                Ok(ExpertBytes {
                    card: CARD,
                    nvme: 0,
                    arena: 0,
                })
            });
            let got = got.expect("a plan");
            assert_eq!((got.ctx, got.rule), (cap, "fit"), "cap {cap}");
            assert_eq!(probes, 2, "cap {cap}: the cap and the base");
        }
    }

    /// NVMe expert bytes growing past the base's bound ([`NVME_SLACK_DIV`])
    /// stop the default below the context where they do, though the card's
    /// margin holds there: a base that pages keeps its bound, and bytes that
    /// stay or fall are within. A base that pages none has the card decide
    /// and the room veto: the plan's NVMe bytes there are the room's, so the
    /// card's context is kept whole or not at all.
    #[test]
    fn default_ctx_stops_where_nvme_grows() {
        let grows_past_6000 = |base: u64| move |c: usize| base + c.saturating_sub(6000) as u64 * 10;
        // A base that pages 500 B: the margin never binds, the NVMe bytes
        // grow past the bound of 5 B at 6001 and the answer is the last
        // multiple of 256 under it.
        let (got, probes) = run(
            262_144,
            STEP,
            1 << 40,
            plan(CARD, AT, 1, grows_past_6000(500)),
        );
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule), (5888, "margin"), "base NVMe 500");
        assert_eq!(
            (got.base.nvme, got.chosen.nvme),
            (500, 500),
            "the base's NVMe bytes held"
        );
        assert!(probes <= probe_bound(262_144, STEP), "{probes} probes");
        // A base that pages none. PIN(2026-10-12): the base, by the host's
        // veto, where the answer was 5888 by the margin rule. The card
        // (an unbounded room) pages nothing at any context, so its context is
        // the fit, 262,144, and the plan at the room read pages there: the
        // room the NVMe bytes grew with decided the old answer. Re-pin by the
        // rule of #46 (the card decides, the room only vetoes), not a
        // relaxation: the one-probe form still gives 5888.
        let grows = plan(CARD, AT, 1, grows_past_6000(0));
        let (got, probes) = run(262_144, STEP, 1 << 40, &grows);
        let got = got.expect("a plan");
        assert_eq!(
            (got.ctx, got.rule, got.margin_ctx),
            (AT, "host", 262_144),
            "base NVMe 0"
        );
        assert_eq!(
            (got.base.nvme, got.chosen.nvme),
            (0, 0),
            "the base's NVMe bytes held"
        );
        assert!(probes <= probe_bound(262_144, STEP), "{probes} probes");
        let one = default_slot_ctx(262_144, AT, STEP, 1 << 40, &grows).expect("a plan");
        assert_eq!((one.ctx, one.rule), (5888, "margin"), "the one-probe form");
        // The same NVMe bytes at every context, or fewer, are within.
        let (got, _) = run(9000, STEP, 1 << 40, plan(CARD, AT, 1, |_| 500));
        assert_eq!(got.expect("a plan").ctx, 9000, "equal NVMe bytes");
        let fewer = |c: usize| if c <= AT { 500 } else { 400 };
        let (got, _) = run(9000, STEP, 1 << 40, plan(CARD, AT, 1, fewer));
        assert_eq!(got.expect("a plan").ctx, 9000, "fewer NVMe bytes");
    }

    /// A base that pages grants the deeper contexts a hundredth of its own
    /// NVMe expert bytes more, no further: NVMe bytes growing linearly past
    /// the base's give the largest multiple of the step within
    /// `base × (1 + 1 / NVME_SLACK_DIV)`, the line's `nvme_bound` the
    /// allowance and its chosen bytes under it.
    #[test]
    fn default_ctx_pages_a_hundredth_past_the_base() {
        const BASE: u64 = 10_000_000;
        const PER: u64 = 10;
        let probe = plan(CARD, AT, 1, |c: usize| {
            BASE + c.saturating_sub(AT) as u64 * PER
        });
        let got = default_slot_ctx(20_000, AT, STEP, 1 << 40, &probe).expect("a plan");
        // 100,000 B allowed at 10 B a position: 10,000 positions past 4,096,
        // and 14,080 is the last multiple of 256 within them.
        assert_eq!(BASE / NVME_SLACK_DIV, 100_000);
        assert_eq!((got.ctx, got.rule), (14_080, RULE_MARGIN));
        assert_eq!(got.nvme_bound, 100_000);
        assert!(got.chosen.nvme - got.base.nvme <= got.nvme_bound);
        assert!(
            probe(14_336).expect("a plan").nvme - BASE > got.nvme_bound,
            "the next step pages more than the bound"
        );
        assert!(
            got.line(1).contains("nvme_bound=100000 "),
            "{}",
            got.line(1)
        );
    }

    /// The miss clause counts both terms of the trade: a deeper context whose
    /// NVMe bytes equal the base's but whose arena is smaller by more than a
    /// hundredth of the base's missed bytes is refused (the NVMe-bytes clause
    /// alone passes it), one inside the bound is accepted, the bound itself
    /// is within, and the card clause still binds.
    #[test]
    fn within_counts_the_arena_a_deeper_context_gives_up() {
        const NVME: u64 = 10_000_000;
        const ARENA: u64 = 4_000_000;
        let base = ExpertBytes {
            card: CARD,
            nvme: NVME,
            arena: ARENA,
        };
        let bound = (NVME - ARENA) / NVME_SLACK_DIV;
        assert_eq!((base.miss(), base.nvme_bound()), (6_000_000, 60_000));
        let deeper = |nvme: u64, arena: u64, card: u64| ExpertBytes { card, nvme, arena };
        // The arena alone gives up the bound and one byte: refused, though
        // the NVMe bytes are the base's and a hundredth of them is 100,000 B.
        let refused = deeper(NVME, ARENA - bound - 1, CARD);
        assert!(refused.nvme <= NVME + NVME / NVME_SLACK_DIV);
        assert!(!refused.within(base, 0), "the arena's shrink passes unseen");
        assert!(
            deeper(NVME, ARENA - bound, CARD).within(base, 0),
            "the bound"
        );
        assert!(deeper(NVME, ARENA - bound / 2, CARD).within(base, 0));
        // Both terms add: experts moved to the tier and arena given up.
        assert!(!deeper(NVME + 30_000, ARENA - 30_001, CARD).within(base, 0));
        assert!(deeper(NVME + 30_000, ARENA - 30_000, CARD).within(base, 0));
        // An arena that grows pays for experts moved to the tier.
        assert!(deeper(NVME + 100_000, ARENA + 100_000, CARD).within(base, 0));
        // The card clause is the margin's, whatever the miss.
        assert!(!deeper(NVME, ARENA, CARD - 1).within(base, 0));
        assert!(deeper(NVME, ARENA, CARD - 1).within(base, 1));
        // A base that misses nothing (its arena holds every paged byte, or it
        // pages none) misses nothing deeper.
        let covered = ExpertBytes {
            card: CARD,
            nvme: NVME,
            arena: NVME,
        };
        assert_eq!(covered.nvme_bound(), 0);
        assert!(deeper(NVME, NVME, CARD).within(covered, 0));
        assert!(!deeper(NVME, NVME - 1, CARD).within(covered, 0));
    }

    /// A base that pages nothing pages nothing deeper: its bound is 0 bytes,
    /// not a floor of some bytes, so the first byte the plan pages past the
    /// base ends the search. Over one probe the search stops at the context
    /// before; the card's context (an unbounded room pages nothing) is kept
    /// whole or vetoed to the base.
    #[test]
    fn default_ctx_pages_nothing_past_a_base_that_pages_none() {
        let probe = plan(CARD, AT, 1, |c: usize| c.saturating_sub(8192) as u64 * 100);
        let got = default_slot_ctx(20_000, AT, STEP, 1 << 40, &probe).expect("a plan");
        assert_eq!(got.nvme_bound, 0);
        assert_eq!((got.ctx, got.rule), (8192, RULE_MARGIN));
        assert_eq!(got.chosen.nvme, 0, "no expert paged");
        // PIN(2026-10-12): the base, by the host's veto, where the answer was
        // 8192 by the margin rule. The card's context is the fit, 20,000, and
        // the plan at the room read pages there. Re-pin by the rule of #46
        // (the card decides, the room only vetoes), not a relaxation.
        let (got, _) = run(20_000, STEP, 1 << 40, &probe);
        let got = got.expect("a plan");
        assert_eq!(got.nvme_bound, 0);
        assert_eq!((got.ctx, got.rule, got.margin_ctx), (AT, RULE_HOST, 20_000));
        assert_eq!(got.chosen.nvme, 0, "no expert paged");
    }

    /// The base's own plan refused is the rule's error at the base, though
    /// the cap's stands (a probe that is not monotone), and nothing is probed
    /// past it; no plan standing anywhere is the refusal at 1 with the
    /// card rule's cap as the base.
    #[test]
    fn default_ctx_base_probe_error_is_returned() {
        let card = plan(CARD, AT, 1, |_| 0);
        let base_refused = |c: usize| {
            if c == AT {
                Err("no base".to_owned())
            } else {
                card(c)
            }
        };
        let (got, probes) = run(9000, STEP, MARGIN, base_refused);
        let want = Refused {
            at: AT,
            source: "no base".to_owned(),
        };
        assert_eq!((got, probes), (Err(want), 2), "the cap's and the base's");
        let all_refused = |_: usize| Err::<ExpertBytes, _>("none".to_owned());
        let (got, probes) = run(3000, STEP, MARGIN, all_refused);
        let want = Refused {
            at: 1,
            source: "none".to_owned(),
        };
        assert_eq!(
            (got, probes),
            (Err(want), 2),
            "the cap's and one position's"
        );
    }

    /// A step of 0 is refused by name, not by the divide it would reach.
    #[test]
    #[should_panic(expected = "a context step of 0")]
    fn default_ctx_step_of_zero_is_refused_by_name() {
        let _ = run(9000, 0, MARGIN, plan(CARD, AT, 1, |_| 0));
    }

    /// The rule's words are one set in one order, whatever the seat: `card`
    /// when the card holds fewer than a base context; else `base` when the
    /// answer is the base (no step past it within, or a fit at it), `fit` when
    /// the answer is the fit, `margin` when the margin or the NVMe tier binds
    /// below it, `host` when the card's deeper context is vetoed by the room
    /// the plan was made at. A seat's own bound (a set `--ctx`, a prompt
    /// cache's room) names itself.
    #[test]
    fn default_ctx_words() {
        assert_eq!(
            [RULE_CARD, RULE_BASE, RULE_FIT, RULE_MARGIN, RULE_HOST],
            ["card", "base", "fit", "margin", "host"]
        );
        let word = |cap: usize, step: usize, margin: u64, nvme: fn(usize) -> u64| {
            run(cap, step, margin, plan(CARD, AT, 1, nvme))
                .0
                .expect("a plan")
                .rule
        };
        assert_eq!(word(AT - 1, STEP, MARGIN, |_| 0), "card", "under the base");
        assert_eq!(word(AT, STEP, MARGIN, |_| 0), "base", "a fit at the base");
        assert_eq!(
            word(9000, 1024, MARGIN, |_| 0),
            "base",
            "no whole step past the base within the margin"
        );
        assert_eq!(
            word(9000, STEP, 1 << 40, |c| if c > AT { 510 } else { 500 }),
            "base",
            "no step past a paging base holds the NVMe bytes"
        );
        // PIN(2026-10-12): `host`, where the row read `base` ("no step past
        // the base holds the NVMe bytes"): over a base that pages none the
        // card's context is the fit, and the plan at the room read pages
        // there, which the room vetoes. The paged base's row above keeps the
        // word. Re-pin by the rule of #46, not a relaxation.
        assert_eq!(
            word(9000, STEP, 1 << 40, |c| if c > AT { 10 } else { 0 }),
            "host",
            "the room pages the card's context"
        );
        assert_eq!(
            word(9000, STEP, 1 << 40, |_| 0),
            "fit",
            "every context to the fit within"
        );
        assert_eq!(
            word(9000, STEP, MARGIN, |_| 0),
            "margin",
            "the margin binds below the fit"
        );
        assert_eq!(
            word(262_144, STEP, 1 << 40, |c| if c > 6000 { 510 } else { 500 }),
            "margin",
            "the NVMe tier binds a paging base's default below the fit"
        );
        // PIN(2026-10-12): `host`, where the row read `margin` ("the NVMe tier
        // binds below the fit"), over a base that pages none; as above.
        assert_eq!(
            word(262_144, STEP, 1 << 40, |c| if c > 6000 { 10 } else { 0 }),
            "host",
            "the room pages the card's context below the fit"
        );
    }

    /// A context is planned once by each of the two probes, however many of
    /// the rule's terms ask for it: the fit's plan is the scan's first probe
    /// and the base's the scan's, over a range, a refusing card, a whole-card
    /// load, a card context the room vetoes (the plan at the room read is the
    /// veto's one more) and a base that pages or a card under the base, where
    /// the card is never planned.
    #[test]
    fn default_ctx_plans_each_context_once() {
        // (cap, margin, refused past, pages past, the card probe is asked)
        type Row = (usize, u64, usize, usize, bool);
        let rows: [Row; 7] = [
            (9000, MARGIN, usize::MAX, usize::MAX, true),
            (262_144, MARGIN, usize::MAX, usize::MAX, true),
            (262_144, 1 << 40, usize::MAX, usize::MAX, true),
            (9000, MARGIN, 4600, usize::MAX, true),
            (3000, MARGIN, usize::MAX, usize::MAX, false),
            (262_144, 1 << 40, usize::MAX, 6000, true),
            (262_144, 1 << 40, usize::MAX, 0, false),
        ];
        for (cap, margin, refused_at, pages_past, asks_card) in rows {
            let (seen_read, seen_card) = (RefCell::new(Vec::new()), RefCell::new(Vec::new()));
            let read = refused_past(
                refused_at,
                plan(CARD, AT, 1, |c| if c > pages_past { 10 } else { 0 }),
            );
            let free = unbounded(&read);
            let counted_read = |c: usize| {
                seen_read.borrow_mut().push(c);
                read(c)
            };
            let counted_card = |c: usize| {
                seen_card.borrow_mut().push(c);
                free(c)
            };
            default_slot_ctx_by_card(
                cap,
                AT,
                STEP,
                margin,
                Veto::Binary,
                &counted_card,
                &counted_read,
            )
            .expect("a plan");
            for (which, seen) in [("read", seen_read), ("card", seen_card)] {
                let mut asked = seen.take();
                let n = asked.len();
                asked.sort_unstable();
                asked.dedup();
                assert_eq!(
                    asked.len(),
                    n,
                    "cap {cap}, margin {margin}, pages past {pages_past}: a context twice by \
                     the {which} probe"
                );
                assert!(
                    which == "read" || (n > 0) == asks_card,
                    "cap {cap}, margin {margin}, pages past {pages_past}: the card probe asked \
                     {n} times"
                );
            }
        }
    }

    /// The report's machine as a probe model [derived, not fitted]: Qwen3.8
    /// UD-Q4_K_XL, two slots, one card. The plan needs 59,660,632,064 B of
    /// host at 4096 a slot and, deeper, one 130,162,688 B checkpoint a slot
    /// for every 4096 positions (`⌈ctx / 4096⌉ + 1` points a slot: two at
    /// 4096, four at 12288, one more at 4352) and 24,576 B a position a slot of
    /// KV (12 GQA layers × K and V × 2 heads × 256 × f16, the model's
    /// dimensions) whose card experts move to the host. The card holds
    /// [`Q38_CARD`] of experts at 4096 and loses those 24,576 B a position a
    /// slot. A plan at a room pages what the room cannot hold
    /// (`need - room`, no arena); at an unbounded room, nothing.
    const Q38_NEED: u64 = 59_660_632_064;
    const Q38_CKPT: u64 = 130_162_688;
    const Q38_KV: u64 = 24_576;
    const Q38_SLOTS: u64 = 2;
    const Q38_CARD: u64 = 24 << 30;
    const GIB: u64 = 1 << 30;

    /// The host bytes the model's plan needs at `c >= 4096` positions a slot.
    fn q38_need(c: usize) -> u64 {
        let points = c.div_ceil(4096) as u64 + 1;
        Q38_NEED + Q38_SLOTS * Q38_CKPT * (points - 2) + Q38_SLOTS * Q38_KV * (c as u64 - 4096)
    }

    /// The model's plan at an unbounded room.
    fn q38_card(c: usize) -> Result<ExpertBytes, String> {
        Ok(ExpertBytes {
            card: Q38_CARD - Q38_SLOTS * Q38_KV * (c as u64 - 4096),
            nvme: 0,
            arena: 0,
        })
    }

    /// The model's plan at `room` bytes.
    fn q38_read(room: u64) -> impl Fn(usize) -> Result<ExpertBytes, String> {
        move |c| {
            Ok(ExpertBytes {
                nvme: q38_need(c).saturating_sub(room),
                ..q38_card(c)?
            })
        }
    }

    /// The room that leaves `slack` bytes past the base's need.
    fn q38_room(slack: i64) -> u64 {
        Q38_NEED.checked_add_signed(slack).expect("a room")
    }

    /// The default of the card rule at `slack`, the cap of the trained 262,144.
    fn q38_by_card(slack: i64) -> SlotCtx {
        let read = q38_read(q38_room(slack));
        default_slot_ctx_by_card(262_144, AT, STEP, GIB, Veto::Binary, &q38_card, &read)
            .expect("a plan")
    }

    /// The default of the one-probe rule at `slack`: the room read alone.
    fn q38_one_probe(slack: i64) -> SlotCtx {
        let read = q38_read(q38_room(slack));
        default_slot_ctx(262_144, AT, STEP, GIB, &read).expect("a plan")
    }

    /// The default context does not ride the host's slack [FAIL-first: the
    /// room-read rule answers 4096 and 12288 at the report's two rooms]. The
    /// room's slack past the base's need was 225,464,320 B in one run and
    /// 1,003,790,336 B in the next of one command, 778,326,016 B of reading
    /// between them. The card's context is 25,856 (a margin of 1 GiB over
    /// 2 × 24,576 B a position), which the host holds from 2,631,499,776 B of
    /// slack on: both rooms are under it, so both answer the base, by `host`.
    #[test]
    fn default_ctx_does_not_ride_the_hosts_slack() {
        const NEAR: i64 = 225_464_320;
        const FAR: i64 = 1_003_790_336;
        const STAR: u64 = 2_631_499_776;
        assert_eq!(
            [q38_one_probe(NEAR).ctx, q38_one_probe(FAR).ctx],
            [4096, 12_288],
            "the model holds the report: the room-read rule rides the slack"
        );
        assert_eq!(q38_need(25_856) - q38_need(AT), STAR, "the threshold");
        let (near, far) = (q38_by_card(NEAR), q38_by_card(FAR));
        assert_eq!(
            (near.ctx, near.rule),
            (far.ctx, far.rule),
            "the two readings answer alike"
        );
        assert_eq!(
            (near.ctx, near.rule, near.margin_ctx),
            (4096, RULE_HOST, 25_856)
        );
        assert_eq!((near.fit, near.base_ctx), (262_144, 4096));
        assert_eq!(near.chosen, near.base, "the base's plan is the load's");
        let line = near.line(2);
        assert!(
            line.starts_with("ctx rule=host ctx=4096 slots=2 total=8192 cap=262144 fit=262144 ")
                && line.contains(" margin_ctx=25856 base=4096 "),
            "{line}"
        );
        // The threshold is one comparison: the card's context at the byte it
        // is held, the base a byte under it.
        let held = q38_by_card(STAR as i64);
        assert_eq!((held.ctx, held.rule), (25_856, RULE_MARGIN));
        assert_eq!(held.chosen.nvme, 0, "no expert paged");
        let under = q38_by_card(STAR as i64 - 1);
        assert_eq!((under.ctx, under.rule), (4096, RULE_HOST));
        // Any room past it answers the same.
        assert_eq!(q38_by_card(64 << 30).ctx, 25_856);
    }

    /// The default moves only within one comparison of the host's slack
    /// [FAIL-first]: over a slack from 0 to 6 GiB, the set where the answer
    /// differs from the answer a reading of 778,326,016 B higher gives is the
    /// one band under the threshold, at most that reading and one grid step
    /// long, and every answer is the base or the card's context. The
    /// room-read rule moves over several times that length.
    #[test]
    fn default_ctx_moves_only_across_one_threshold() {
        const DRIFT: i64 = 778_326_016;
        const GRID: i64 = 16 << 20;
        let (mut by_card, mut one_probe) = (0, 0);
        for slack in (0..=6 << 30).step_by(GRID as usize) {
            let (low, high) = (q38_by_card(slack), q38_by_card(slack + DRIFT));
            for got in [low, high] {
                assert!(
                    [AT, 25_856].contains(&got.ctx),
                    "slack {slack}: {} is neither the base nor the card's context",
                    got.ctx
                );
            }
            by_card += i64::from(low.ctx != high.ctx) * GRID;
            one_probe +=
                i64::from(q38_one_probe(slack).ctx != q38_one_probe(slack + DRIFT).ctx) * GRID;
        }
        assert!(
            by_card <= DRIFT + GRID,
            "{by_card} B of slack answer differently across the reading"
        );
        assert!(
            one_probe > 2 * DRIFT,
            "{one_probe} B for the room-read rule: the model no longer shows the defect"
        );
    }

    /// Where the room cancels out the card rule is the one-probe rule: a base
    /// that pages (a room under its need, from 12 GB under) answers as the
    /// room-read rule does, line for line, and so does a card that holds
    /// under the base; and the probes' disagreements are refused by name: a
    /// card plan refused at 1 or at the base while the plan at the room read
    /// stands, and a card context the room cannot plan is the host's veto.
    #[test]
    fn default_ctx_is_the_one_probe_rule_where_the_room_cancels() {
        for slack in (-(12i64 << 30)..0).step_by(64 << 20) {
            assert_eq!(
                q38_by_card(slack),
                q38_one_probe(slack),
                "a base that pages, slack {slack}"
            );
        }
        assert_eq!(q38_by_card(-1), q38_one_probe(-1));
        // The card holds 3000, the room-read plan stands to the cap.
        let read = plan(CARD, AT, 1, |_| 0);
        let card = refused_past(3000, plan(CARD, AT, 1, |_| 0));
        let got = default_slot_ctx_by_card(9000, AT, STEP, MARGIN, Veto::Binary, &card, &read)
            .expect("a plan");
        let one = default_slot_ctx(9000, AT, STEP, MARGIN, &read).expect("a plan");
        assert_eq!(got, one, "a card under the base");
        // The card plan refused at 1 and at the base, the room-read plan standing.
        let nothing = |c: usize| Err::<ExpertBytes, _>(format!("no card at {c}"));
        let got = default_slot_ctx_by_card(9000, AT, STEP, MARGIN, Veto::Binary, &nothing, &read);
        assert_eq!(
            got,
            Err(Refused {
                at: 1,
                source: "no card at 1".to_owned()
            })
        );
        let no_base = |c: usize| {
            if c == AT {
                Err("no card base".to_owned())
            } else {
                plan(CARD, AT, 1, |_| 0)(c)
            }
        };
        let got = default_slot_ctx_by_card(9000, AT, STEP, MARGIN, Veto::Binary, &no_base, &read);
        assert_eq!(
            got,
            Err(Refused {
                at: AT,
                source: "no card base".to_owned()
            })
        );
        // The room-read plan refused at the card's context: vetoed, not an error.
        let tight = refused_past(4600, plan(CARD, AT, 1, |_| 0));
        let got = default_slot_ctx_by_card(9000, AT, STEP, MARGIN, Veto::Binary, &read, &tight)
            .expect("a plan");
        assert_eq!(
            (got.ctx, got.rule, got.margin_ctx, got.fit),
            (AT, RULE_HOST, 4864, 9000)
        );
        assert_eq!(got.chosen, got.base, "the base's plan");
    }

    /// The `ctx` line the seats print: its field names are what the gates
    /// read, in this order. PIN(2026-10-12): `arena=` and `base_arena=` join
    /// the line beside `nvme_bound=`, whose bound is now the missed bytes'
    /// hundredth; every other field and its order are the line's before.
    /// The margin rule's line (an example plan whose
    /// card loses 48 KiB a position past the base, for a margin of 1 GiB) and
    /// the card rule's.
    #[test]
    fn default_ctx_line_names_its_terms() {
        let probe = plan(9_000_000_000, AT, 49_152, |_| 0);
        // The card refuses past 100,000, so the fit is under the cap.
        let bounded = refused_past(100_000, &probe);
        let got = default_slot_ctx(262_144, AT, STEP, 1 << 30, &bounded).expect("a plan");
        assert_eq!(
            got.line(2),
            "ctx rule=margin ctx=25856 slots=2 total=51712 cap=262144 fit=100000 \
             fit_card_expert_bytes=4286126592 margin_ctx=25856 base=4096 \
             base_card_expert_bytes=9000000000 base_nvme_expert_bytes=0 \
             card_expert_bytes=7930452480 nvme_expert_bytes=0 arena=0 base_arena=0 \
             nvme_bound=0 lost_bytes=1069547520 margin_bytes=1073741824"
        );
        let got = default_slot_ctx(3000, AT, STEP, 1 << 30, &probe).expect("a plan");
        assert_eq!(
            got.line(1),
            "ctx rule=card ctx=3000 slots=1 total=3000 cap=3000 fit=3000 \
             fit_card_expert_bytes=9000000000 margin_ctx=3000 base=3000 \
             base_card_expert_bytes=9000000000 base_nvme_expert_bytes=0 \
             card_expert_bytes=9000000000 nvme_expert_bytes=0 arena=0 base_arena=0 \
             nvme_bound=0 lost_bytes=0 margin_bytes=1073741824"
        );
    }

    /// A DGX Spark (GB10) at two slots as a probe model [derived from the
    /// serve line `ctx rule=cache ctx=196608 slots=2 … fit=262144`, whose
    /// `cache` line gave `available=124659433472 need=112142376240
    /// checkpoints=8330412032`]: the plan needs [`GB10_NEED`] of the pool at
    /// 196,608 positions a slot and [`GB10_PER`] more a position (52,584 B a
    /// position a slot: 31,832 B of state, fitted to the cache line, and one
    /// 85,004,204 B checkpoint per 4096 positions, 8,330,412,032 over 2 × 49);
    /// the card's bytes do not move (every routed expert is in the pool).
    /// The plan stands iff its need is within the room.
    const GB10_AT: usize = 196_608;
    const GB10_NEED: u64 = 112_142_376_240;
    const GB10_PER: u64 = 2 * 52_584;
    const GB10_AVAILABLE: u64 = 124_659_433_472;
    /// The rungs of the halving ladder from the trained 262,144 to the base.
    const GB10_RUNGS: [usize; 7] = [262_144, 131_072, 65_536, 32_768, 16_384, 8192, 4096];
    /// The spread of #46's two readings of one machine.
    const DRIFT: u64 = 778_326_016;
    /// The sweeps' grid, 16 MiB.
    const GRID: u64 = 16 << 20;

    /// The pool bytes the GB10 model's plan needs at `c` positions a slot.
    fn gb10_need(c: usize) -> u64 {
        let apart = GB10_PER * c.abs_diff(GB10_AT) as u64;
        if c >= GB10_AT {
            GB10_NEED + apart
        } else {
            GB10_NEED - apart
        }
    }

    /// The room the plan is made at: the available bytes less the host drift.
    const GB10_ROOM: u64 = GB10_AVAILABLE - (512 << 20);

    /// The GB10 model's plan at a `room`: its bytes where it stands, a
    /// refusal where the pool cannot hold it.
    fn gb10_read(room: u64) -> impl Fn(usize) -> Result<ExpertBytes, String> {
        move |c| {
            if gb10_need(c) <= room {
                Ok(ExpertBytes {
                    card: GIB,
                    nvme: 0,
                    arena: 0,
                })
            } else {
                Err(format!(
                    "the pool holds {room} B, the plan needs {}",
                    gb10_need(c)
                ))
            }
        }
    }

    /// The default of the ladder at `room` for the trained 262,144; the card
    /// is not asked on a unified machine.
    fn gb10_ladder(room: u64) -> Result<SlotCtx, Refused<String>> {
        let asked = Cell::new(false);
        let card = |c: usize| {
            asked.set(true);
            Err::<ExpertBytes, _>(format!("the card is asked at {c}"))
        };
        let got = default_slot_ctx_by_card(
            262_144,
            AT,
            STEP,
            GIB,
            Veto::Ladder,
            &card,
            &gb10_read(room),
        );
        assert!(!asked.get(), "room {room}: a unified machine asks no card");
        got
    }

    /// The largest rung whose plan the GB10 model's `room` holds, the base
    /// when none above it does.
    fn gb10_rung(room: u64) -> usize {
        GB10_RUNGS
            .into_iter()
            .find(|&r| gb10_need(r) <= room)
            .unwrap_or(AT)
    }

    /// The rooms the unified sweeps cover: from the base's need on (below it
    /// the plan of the pool is the fit's, [`RULE_CARD`], or refused), in 16 MiB
    /// steps, to the reading and 5 GB past it.
    fn gb10_rooms() -> impl Iterator<Item = u64> {
        let first = gb10_need(AT).div_ceil(GRID) * GRID;
        (0..)
            .map(move |i| first + i * GRID)
            .take_while(|&room| room <= GB10_ROOM + 5_000_000_000)
    }

    /// On a unified pool the read room decides through a halving ladder
    /// [FAIL-first: the room-read rule's fit rides the room between the base's
    /// need and the cap's, 91.9 and 119.0 GB]. The model is the GB10's at two
    /// slots; its rungs' needs are 119.03, 105.25, 98.36, 94.91, 93.19, 92.33
    /// and 91.90 GB [derived]. Over the rooms of [`gb10_rooms`] every answer is
    /// the largest rung the room holds: the cap's is `fit`, a lower one
    /// `pool`, and the card is never asked. Below the base's need the plan of
    /// the pool is the fit's ([`RULE_CARD`]) and below one position's need it
    /// is refused.
    #[test]
    fn default_ctx_on_a_unified_pool_is_a_rung() {
        let needs: Vec<u64> = GB10_RUNGS
            .iter()
            .map(|&r| (gb10_need(r) as f64 / 1e7).round() as u64)
            .collect();
        assert_eq!(needs, [11903, 10525, 9836, 9491, 9319, 9233, 9190]);
        let mut before = 0;
        for room in gb10_rooms() {
            let got = gb10_ladder(room).expect("a plan");
            assert_eq!(got.ctx, gb10_rung(room), "room {room}");
            let word = if got.ctx == 262_144 {
                RULE_FIT
            } else {
                RULE_POOL
            };
            assert_eq!(got.rule, word, "room {room}");
            assert_eq!(got.base_ctx, AT, "room {room}");
            assert!(got.ctx >= before, "room {room}: {} after {before}", got.ctx);
            before = got.ctx;
        }
        let under = gb10_ladder(gb10_need(AT) - 1).expect("a plan");
        assert_eq!((under.ctx, under.rule), (AT - 1, RULE_CARD));
        let none = gb10_ladder(gb10_need(1) - 1);
        assert_eq!(none.map_err(|e| e.at), Err(1), "refused at one position");
    }

    /// The default moves only across a rung's need [FAIL-first: the
    /// room-read rule moves at every step of the room]. The answer at a room
    /// differs from the answer a spread of #46's two readings higher only where
    /// a rung's need lies between them, so the rooms that flip are the union of
    /// one spread under each of seven needs, 4.16 GB together [derived], at
    /// most seven spreads.
    #[test]
    fn default_ctx_on_a_unified_pool_moves_only_across_a_rungs_need() {
        let mut flipped = 0;
        for room in gb10_rooms() {
            let (low, high) = (
                gb10_ladder(room).expect("a plan"),
                gb10_ladder(room + DRIFT).expect("a plan"),
            );
            if low.ctx != high.ctx {
                flipped += GRID;
                assert!(
                    GB10_RUNGS
                        .iter()
                        .any(|&r| room < gb10_need(r) && gb10_need(r) <= room + DRIFT),
                    "room {room}: {} and {} with no rung's need between",
                    low.ctx,
                    high.ctx
                );
            }
        }
        assert!(
            flipped <= GB10_RUNGS.len() as u64 * DRIFT,
            "{flipped} B of room answer differently across the reading"
        );
    }

    /// The reading #46 reports, 5 GB under it and 5 GB over it all answer the
    /// cap, 262,144 a slot: the cap's need is 119.03 GB, the reading's room less
    /// 5 GB is 119.12 GB.
    #[test]
    fn default_ctx_on_a_unified_pool_keeps_the_cap_at_the_reading() {
        for room in [
            GB10_ROOM - 5_000_000_000,
            GB10_ROOM,
            GB10_ROOM + 5_000_000_000,
        ] {
            let got = gb10_ladder(room).expect("a plan");
            assert_eq!((got.ctx, got.rule), (262_144, RULE_FIT), "room {room}");
            assert_eq!(got.chosen, got.base, "the plan's bytes do not move");
        }
    }

    /// A pool that cannot hold the cap: with a room 1 GiB under the cap's
    /// need the room-read rule's answer is the fit, which moves with every
    /// 105,168 B of room, and the ladder answers 131,072, the first rung the
    /// room holds.
    #[test]
    fn default_ctx_on_a_pool_under_the_cap_is_a_rung() {
        let room = gb10_need(262_144) - GIB;
        let read = gb10_read(room);
        let rides = default_slot_ctx(262_144, AT, STEP, GIB, &read).expect("a plan");
        let moved = default_slot_ctx(262_144, AT, STEP, GIB, &gb10_read(room + 512 * GB10_PER))
            .expect("a plan");
        assert!(
            !GB10_RUNGS.contains(&rides.ctx) && moved.ctx > rides.ctx,
            "the room-read rule answers {} and {}: no rung, riding the room",
            rides.ctx,
            moved.ctx
        );
        let got = gb10_ladder(room).expect("a plan");
        assert_eq!((got.ctx, got.rule), (131_072, RULE_POOL));
        assert_eq!(got.fit, rides.fit, "the fit is the room-read plan's");
        assert_eq!(got.chosen, got.base, "the plan's bytes do not move");
    }

    /// A ladder of one rung: a cap that is the base answers the base by
    /// `base`, the word the discrete scan gives a fit at the base, not `fit`.
    #[test]
    fn default_ctx_on_a_unified_pool_at_a_cap_of_the_base_is_the_base() {
        let read = gb10_read(GB10_ROOM);
        let card = |c: usize| Err::<ExpertBytes, _>(format!("the card is asked at {c}"));
        let got = default_slot_ctx_by_card(AT, AT, STEP, GIB, Veto::Ladder, &card, &read)
            .expect("a plan");
        assert_eq!((got.ctx, got.rule, got.fit), (AT, RULE_BASE, AT));
    }

    /// The prompt cache's bound on the GB10 model: a sequence state is 31,832 B
    /// a position [fitted: it reproduces `ctx=196608` at `ram=6258528616`
    /// under the largest-holding bound], the default is the trained 262,144
    /// over the base 4096 in steps of 256, and the budget is the lesser of
    /// 8 GiB and half of the slack S, swept in 16 MiB steps.
    fn gb10_cache(slack: u64) -> Result<CacheBound, CtxError> {
        let ram = (8u64 << 30).min(slack / 2);
        within_cache(262_144, AT, STEP, &|n| 31_832 * n as u64, &|_| Ok(ram))
    }

    /// The budget at which a rung's state holds: twice its state.
    fn gb10_budget(rung: usize) -> u64 {
        2 * 31_832 * rung as u64
    }

    /// The prompt cache's bound is a rung [FAIL-first: the largest context that
    /// holds rides the budget]. Over S from 16 MiB to 20 GiB every answer is
    /// the largest rung whose state the budget holds, the base when none above
    /// it does (it holds only from S = 2 × state(4096): the cache then keeps the
    /// shorter sessions, `holds` false), and the answer never falls as S
    /// grows. A budget of no byte holds no state at all, by name.
    #[test]
    fn within_cache_answers_a_rung() {
        let none = gb10_cache(0);
        assert!(
            matches!(
                none,
                Err(CtxError::NoCacheState {
                    state: 31_832,
                    ram: 0
                })
            ),
            "{none:?}"
        );
        let mut before = 0;
        for slack in (1..).map(|i| i * GRID).take_while(|&s| s <= 20 << 30) {
            let got = gb10_cache(slack).expect("a bound");
            let want = GB10_RUNGS
                .into_iter()
                .find(|&r| gb10_budget(r) <= slack)
                .unwrap_or(AT);
            assert_eq!(got.ctx, want, "slack {slack}");
            assert_eq!(got.holds, slack >= gb10_budget(AT), "slack {slack}");
            assert!(
                got.ctx >= before,
                "slack {slack}: {} after {before}",
                got.ctx
            );
            before = got.ctx;
        }
    }

    /// The prompt cache's bound moves only across a rung's budget [FAIL-first:
    /// the largest-holding bound moves at every step of the budget]. The
    /// answer is the same between two consecutive rungs' budgets (2 × state of
    /// a rung: 0.26, 0.52, 1.04, 2.09, 4.17, 8.34 and 16.69 GB), so the S where
    /// it differs from the answer a spread of #46's two readings higher lie
    /// within one spread under those seven budgets, at most seven spreads in
    /// all (4.16 GB together [derived]).
    #[test]
    fn within_cache_moves_only_across_a_rungs_budget() {
        let mut flipped = 0;
        for slack in (1..).map(|i| i * GRID).take_while(|&s| s <= 20 << 30) {
            let (low, high) = (
                gb10_cache(slack).expect("a bound"),
                gb10_cache(slack + DRIFT).expect("a bound"),
            );
            if low.ctx != high.ctx {
                flipped += GRID;
                assert!(
                    GB10_RUNGS
                        .iter()
                        .any(|&r| slack < gb10_budget(r) && gb10_budget(r) <= slack + DRIFT),
                    "slack {slack}: {} and {} with no rung's budget between",
                    low.ctx,
                    high.ctx
                );
            }
        }
        assert!(
            flipped <= GB10_RUNGS.len() as u64 * DRIFT,
            "{flipped} B of slack answer differently across the reading"
        );
    }

    /// At #46's budget, 6,258,528,616 B (S = 12.5 GB), and a spread of the
    /// readings either side of it, the bound is 131,072 [FAIL-first: the
    /// largest-holding bound answers 196,608 at it].
    #[test]
    fn within_cache_holds_still_at_the_reports_budget() {
        let report = 2 * 6_258_528_616;
        for slack in [report - DRIFT, report, report + DRIFT] {
            let got = gb10_cache(slack).expect("a bound");
            assert_eq!((got.ctx, got.holds), (131_072, true), "slack {slack}");
        }
    }

    /// The grid search's rows: no trained context is no answer, a trained
    /// context at or under the floor is its own, a floor that does not fit
    /// is no answer, the trained context where it fits (probed once, after
    /// the floor), else the largest grid step below it that does.
    #[test]
    fn searched_rows() {
        const GRAN: usize = 1024;
        let calls = Cell::new(0usize);
        let upto = |edge: usize| {
            let calls = &calls;
            move |c: usize| {
                calls.set(calls.get() + 1);
                Ok::<_, Infallible>(c <= edge)
            }
        };
        let run = |trained: Option<usize>, floor: usize, edge: usize| {
            calls.set(0);
            let got = searched(trained, floor, GRAN, &upto(edge)).expect("infallible");
            (got, calls.get())
        };
        assert_eq!(run(None, 4096, 1 << 30), (None, 0), "no trained context");
        assert_eq!(
            run(Some(4096), 4096, 0),
            (Some(4096), 0),
            "trained at the floor"
        );
        assert_eq!(
            run(Some(2048), 4096, 0),
            (Some(2048), 0),
            "trained under it"
        );
        assert_eq!(
            run(Some(262_144), 4096, 4095),
            (None, 1),
            "the floor does not fit"
        );
        assert_eq!(
            run(Some(262_144), 4096, 262_144),
            (Some(262_144), 2),
            "the trained context fits: the floor and it, two probes"
        );
        // The card holds through `edge`: the answer is the last floor + k·GRAN
        // not past it, on the grid and under the trained context.
        for edge in [4096, 4097, 5119, 5120, 40_000, 100_000, 262_143] {
            let (got, probes) = run(Some(262_144), 4096, edge);
            let want = 4096 + (edge - 4096) / GRAN * GRAN;
            assert_eq!(got, Some(want), "the card holds through {edge}");
            assert!(probes <= 2 + 8, "{probes} probes at {edge}");
        }
        // A trained context off the grid still caps the answer.
        assert_eq!(run(Some(10_000), 4096, 9_000).0, Some(4096 + 4 * GRAN));
        assert_eq!(run(Some(10_000), 4096, 10_000).0, Some(10_000));
    }

    /// A seat's default is never past what the card holds, never under the
    /// floor, a multiple of the grid, and never lower for a smaller cache
    /// term (a q8_0 cache's default is not under the f16 one), whatever the
    /// card's room: a loaded server shows the one room its card has, here
    /// every room a cache term can leave is a row. The need is
    /// `layers · rows · per_row` against `room`, the shape of the whole-fit
    /// verdict's cache term; `layers` is the row's own, not a file's.
    #[test]
    fn searched_default_keeps_the_floor_the_grid_and_the_card() {
        // 2,048 B a row of one layer, f16: 4 KV heads × 128 × 2 planes × 2 B
        // (`Qwen3-30B-A3B-2507`'s `head_count_kv` and `key_length`,
        // `arch/qwen3moe/hparams.rs`'s `a3b` test header). A q8_0 cache holds
        // 17/32 of it.
        const PER_ROW: u64 = 4 * 128 * 2 * 2;
        const FLOOR: usize = 4096;
        const GRAN: usize = 1024;
        const TRAINED: usize = 262_144;
        let answer = |per_row: u64, layers: u64, room: u64| {
            let fits = |c: usize| Ok::<_, Infallible>(c as u64 * per_row * layers <= room);
            searched(Some(TRAINED), FLOOR, GRAN, &fits).expect("infallible")
        };
        for layers in [1u64, 12, 48] {
            for room in [0u64, 1 << 20, 1 << 28, 1 << 31, 1 << 34, 1 << 40] {
                let fits = |c: usize| c as u64 * PER_ROW * layers <= room;
                let label = format!("{layers} layers, room {room}");
                match answer(PER_ROW, layers, room) {
                    None => assert!(!fits(FLOOR), "{label}: the floor fits yet None"),
                    Some(n) => {
                        assert!(
                            (FLOOR..=TRAINED).contains(&n),
                            "{label}: {n} outside the floor and the trained context"
                        );
                        assert_eq!(n % GRAN, 0, "{label}: {n} is off the grid");
                        assert!(fits(n), "{label}: {n} is past the card");
                        assert!(
                            n == TRAINED || !fits(n + GRAN),
                            "{label}: {n} leaves a grid step of room"
                        );
                    }
                }
                assert!(
                    answer(PER_ROW * 17 / 32, layers, room) >= answer(PER_ROW, layers, room),
                    "{label}: a q8_0 cache's default is under the f16 one"
                );
            }
        }
        // Where the card has room past the floor the default is past it.
        assert_eq!(
            answer(PER_ROW, 12, 1 << 31),
            Some(87_040),
            "2 GiB, 12 layers"
        );
        assert!(answer(PER_ROW, 12, 1 << 31) > Some(FLOOR));
    }
}
