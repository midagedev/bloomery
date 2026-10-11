//! The serving seats' shared `--ctx` rules: the bisection the seats'
//! searches run ([`largest`]), the default context a slot takes past the
//! seat's base ([`default_slot_ctx`]; [`within_margin`] is its card-only
//! form, which the glm seat lifts), the prompt-cache guard that bounds it by
//! the budget the plan leaves at each context ([`within_cache`]), the search
//! over a context grid both qwen3 defaults run ([`searched`]) and the slot
//! count a `--ctx` flag buys ([`slots_of`]) — more positions on the card push
//! card experts to the host, and decode crawls, so a context the flag did not
//! name never trades away more than the plan's margin of stage-card expert
//! bytes, nor moves experts to the NVMe tier the base's plan kept off it. The
//! plan a seat probes at each context is its own (the model's placements
//! differ); these are the terms the seats already share, and every
//! model-specific input — the plan's bytes, the margin, the step, the trained
//! context — enters as an argument.

use std::cell::Cell;
use std::convert::Infallible;

/// Why a slot count was refused.
#[derive(Debug, thiserror::Error)]
pub enum CtxError {
    /// A seat's own parse refuses `--parallel 0` before the rule; the rule
    /// refuses it too, so no call can serve a count of no slot.
    #[error("--parallel 0: the server serves no slot")]
    NoSlot,
    /// No context holds one sequence state in the prompt cache's budget:
    /// at one position the state takes `state` bytes and the budget is
    /// `ram`.
    #[error(
        "the prompt cache's {ram} bytes hold no sequence state ({state} bytes at one position, \
         two recurrent copies among its terms); --cache-ram 0 turns it off"
    )]
    NoCacheState { state: u64, ram: u64 },
    /// The context the cache's bound rounded to does not hold its state: the
    /// budget at `ctx` is under the `state` there, so the budget is not
    /// monotone across the rounding.
    #[error(
        "the prompt cache's bound rounded to a context of {ctx} positions, whose state of \
         {state} bytes passes the budget of {ram} bytes there"
    )]
    CacheRounding { ctx: usize, state: u64, ram: u64 },
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
}

impl ExpertBytes {
    /// Whether this plan, a deeper context's, is within `margin` of `base`'s:
    /// at most `margin` fewer card expert bytes and no more NVMe expert
    /// bytes. The one owner of "within" for [`default_slot_ctx`] and
    /// [`within_margin`].
    fn within(self, base: Self, margin: u64) -> bool {
        base.card.saturating_sub(self.card) <= margin && self.nvme <= base.nvme
    }
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
///
/// # Panics
/// On a `step` of 0.
fn scan<E>(
    fit: usize,
    base_at: usize,
    base: ExpertBytes,
    step: usize,
    margin: u64,
    probe: &dyn Fn(usize) -> Result<ExpertBytes, E>,
) -> (usize, ExpertBytes) {
    assert!(step > 0, "a context step of 0");
    if fit <= base_at {
        return (base_at, base);
    }
    // The last context probed within, which a bisection moves only upward.
    let best = Cell::new((base_at, base));
    let within = |ctx: usize| {
        let Ok(plan) = probe(ctx) else {
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

/// The largest multiple of `step` up to `fit` whose `card_bytes` holds at
/// most `margin` fewer bytes than `base_bytes` (the plan's at `base_at`):
/// `fit` when every context to there does, `base_at` when no step past it
/// does, else the largest that does rounded down to `step` (never under
/// `base_at`). A context whose plan is refused is not within, as the fit's
/// own search reads it: a probe is never the seat's refusal. The card-only
/// form of [`default_slot_ctx`]'s scan, for a plan with no NVMe tier.
///
/// # Errors
/// None today: a probe's error is a context that is not within. The result
/// type is the seat's own so its call reads as the fit's search does.
///
/// # Panics
/// On a `step` of 0.
pub fn within_margin<E>(
    fit: usize,
    base_at: usize,
    base_bytes: u64,
    step: usize,
    margin: u64,
    card_bytes: &dyn Fn(usize) -> Result<u64, E>,
) -> Result<usize, E> {
    let base = ExpertBytes {
        card: base_bytes,
        nvme: 0,
    };
    let probe = |c: usize| card_bytes(c).map(|card| ExpertBytes { card, nvme: 0 });
    Ok(scan(fit, base_at, base, step, margin, &probe).0)
}

/// What [`default_slot_ctx`] chose and the terms it chose by: a seat prints
/// them as one line ([`SlotCtx::line`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotCtx {
    /// The context a slot takes. A seat that bounds it further (a prompt
    /// cache's room) sets it and [`SlotCtx::rule`], and leaves the scan's
    /// terms below as they are.
    pub ctx: usize,
    /// What set it: `card` (the card holds fewer than a base context, the
    /// fit is the answer) or `margin` (the scan's).
    pub rule: &'static str,
    /// The largest context a slot holds, as the caller's own search found it.
    pub fit: usize,
    /// The scan's answer: `ctx` as [`default_slot_ctx`] returns it.
    pub margin_ctx: usize,
    /// The context the margin counts against, `base` or `fit` when that is
    /// fewer, and its plan's bytes.
    pub base_ctx: usize,
    pub base: ExpertBytes,
    /// The plan's bytes at `margin_ctx`.
    pub chosen: ExpertBytes,
    /// The margin the rule held `chosen` to.
    pub margin: u64,
}

impl SlotCtx {
    /// The `ctx` line a seat prints on stderr: every context a slot's,
    /// `total` the sum of the `slots`', `base` the context the margin counts
    /// against, `lost_bytes` the card expert bytes the chosen plan holds
    /// fewer than the base's.
    #[must_use]
    pub fn line(&self, slots: usize) -> String {
        format!(
            "ctx rule={} ctx={} slots={slots} total={} fit={} margin_ctx={} base={} \
             base_card_expert_bytes={} base_nvme_expert_bytes={} card_expert_bytes={} \
             nvme_expert_bytes={} lost_bytes={} margin_bytes={}",
            self.rule,
            self.ctx,
            slots * self.ctx,
            self.fit,
            self.margin_ctx,
            self.base_ctx,
            self.base.card,
            self.base.nvme,
            self.chosen.card,
            self.chosen.nvme,
            self.base.card.saturating_sub(self.chosen.card),
            self.margin
        )
    }
}

/// The context a slot takes when the flag names none, every number a slot's:
/// `fit` when it is under `base` (rule `card`: the card holds fewer than a
/// base context), else the largest multiple of `step` up to `fit` whose plan
/// holds at most `margin` fewer card expert bytes than the plan at `base`
/// and no more NVMe expert bytes (rule `margin`) — `fit` when its plan does,
/// `base` when no step past it does. The NVMe clause stops a deeper context
/// from buying positions with experts a tier slower than the host's.
///
/// `probe` plans one context. A context whose plan is refused is not within
/// (a probe is never the seat's refusal), except the base's own, whose error
/// is the rule's.
///
/// The scan is a bisection over the multiples of `step`, after one probe at
/// `fit`, where a linear scan would take one probe a step: it assumes `probe`
/// is monotone in the context — card expert bytes never rise, NVMe expert
/// bytes never fall and a refused plan stays refused as the context deepens,
/// the solver trading experts for positions only one way — so the contexts
/// within are a prefix. A probe is a plan solve: past the probes at `base`
/// and `fit`, the `n` multiples of `step` between them cost at most
/// `⌈log2(n + 1)⌉` more.
///
/// # Errors
/// The error of the probe at `base`, or at `fit` when that is under it.
///
/// # Panics
/// On a `step` of 0.
pub fn default_slot_ctx<E>(
    fit: usize,
    base: usize,
    step: usize,
    margin: u64,
    probe: &dyn Fn(usize) -> Result<ExpertBytes, E>,
) -> Result<SlotCtx, E> {
    let base_ctx = base.min(fit);
    let at_base = probe(base_ctx)?;
    let (ctx, rule, chosen) = if fit < base {
        (fit, "card", at_base)
    } else {
        let (ctx, chosen) = scan(fit, base_ctx, at_base, step, margin, probe);
        (ctx, "margin", chosen)
    };
    Ok(SlotCtx {
        ctx,
        rule,
        fit,
        margin_ctx: ctx,
        base_ctx,
        base: at_base,
        chosen,
        margin,
    })
}

/// The context a default keeps when the prompt cache must hold one sequence
/// state there: `from` when its `state` fits the budget `ram_at` gives at
/// `from`, else the largest context under `from` that holds, rounded down
/// to a multiple of `step` (a context under `step` is its own). The budget
/// is read at the context it bounds, not at `from`: a plan's host need
/// moves with the context (a unified pool counts the positions' bytes, a
/// split moves experts), so the budget a smaller context leaves is larger.
/// `from` is tried first and alone: the budget is not monotone where the
/// plan's tables change tier, and a search that skipped it could pass a
/// holding `from` for a smaller context. The rounding keeps the bound: the
/// returned context holds, checked after it.
///
/// # Errors
/// [`CtxError::NoCacheState`] when not even one position's state holds;
/// [`CtxError::CacheRounding`] when the rounded context does not hold; the
/// first error `ram_at` returns.
pub fn within_cache<E: From<CtxError>>(
    from: usize,
    step: usize,
    state: &dyn Fn(usize) -> u64,
    ram_at: &dyn Fn(usize) -> Result<u64, E>,
) -> Result<usize, E> {
    let holds = |n: usize| Ok::<_, E>(state(n) <= ram_at(n)?);
    if holds(from)? {
        return Ok(from);
    }
    if !holds(1)? {
        return Err(CtxError::NoCacheState {
            state: state(1),
            ram: ram_at(1)?,
        }
        .into());
    }
    let c = largest(1, from.saturating_sub(1), holds)?;
    let ctx = (c / step * step).max(c.min(step));
    if !holds(ctx)? {
        return Err(CtxError::CacheRounding {
            ctx,
            state: state(ctx),
            ram: ram_at(ctx)?,
        }
        .into());
    }
    Ok(ctx)
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

/// [`searched`] over a placed plan's expert split: a context fits when its
/// plan, `card_experts(ctx)`, keeps at least the card experts the floor's
/// plan keeps — the solver's context-for-experts trade never goes below the
/// floor's split, so the search spends only the plan's own headroom. A
/// context whose plan cannot build (`None`: the card is held, a budget
/// binds) does not fit; `None` when the floor's own plan cannot build, or
/// no trained context is stated.
///
/// # Errors
/// The first error `card_experts` returns.
pub fn searched_placed<E>(
    trained: Option<usize>,
    floor: usize,
    gran: usize,
    card_experts: &dyn Fn(usize) -> Result<Option<u64>, E>,
) -> Result<Option<usize>, E> {
    let Some(at_floor) = card_experts(floor)? else {
        return Ok(None);
    };
    let fits = |ctx: usize| Ok(card_experts(ctx)?.is_some_and(|e| e >= at_floor));
    searched(trained, floor, gran, &fits)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;

    use super::*;

    /// A bisection's probe, counted: a search over `n` points takes at most
    /// `⌈log2 n⌉ + 1` of them, and a linear scan, or a loop that never
    /// closes, is a panic here rather than a hang.
    struct Probes(Cell<usize>);

    impl Probes {
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
                let probes = Probes::new();
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

    /// A card whose bytes drop by `per` for every position past `at`,
    /// `base` there: the plan's card expert bytes a slot as the margin rule
    /// counts them.
    fn dropping(base: u64, at: usize, per: u64) -> impl Fn(usize) -> Result<u64, String> {
        move |c| Ok(base - (c.saturating_sub(at) as u64) * per)
    }

    /// The margin guard's rows: the fit when every context to it holds the
    /// margin, the largest multiple of the step that does otherwise, the
    /// base when no step past it does, and a refused plan as a context that
    /// is not within.
    #[test]
    fn within_margin_rows() {
        const BASE: u64 = 1_000_000;
        const MARGIN: u64 = 1_000;
        let card = dropping(BASE, 4096, 1);
        let at = |fit: usize, step: usize| {
            within_margin(fit, 4096, BASE, step, MARGIN, &card).expect("a context")
        };
        // The drop is 1 B a position: 1,000 B of margin holds through 5096.
        assert_eq!(at(5000, 256), 5000, "every context to the fit holds");
        assert_eq!(at(5096, 256), 5096, "the fit at the margin's last byte");
        assert_eq!(at(9000, 256), 4864, "5096 rounded down to a step of 256");
        assert_eq!(at(9000, 1024), 4096, "no whole step past the base holds");
        assert_eq!(at(9000, 1), 5096, "a step of one is the margin itself");
        assert_eq!(at(4096, 256), 4096, "a fit at the base");
        // A plan the card refuses past 4600 is not within, whatever its bytes.
        let refused = |c: usize| {
            if c > 4600 {
                Err("refused".to_string())
            } else {
                card(c)
            }
        };
        let got = within_margin(9000, 4096, BASE, 256, MARGIN, &refused).expect("a context");
        assert_eq!(got, 4352, "4600 rounded down to a step of 256");
        // One byte of margin less, and the context before it is the answer.
        let got = within_margin(9000, 4096, BASE, 1, MARGIN - 1, &card).expect("a context");
        assert_eq!(got, 5095);
        // A base off the step: the last context that holds, 4200, rounds down
        // to 4096, under the base, and the base is the answer.
        let off = dropping(BASE, 4100, 1);
        let got = within_margin(9000, 4100, BASE, 256, 100, &off).expect("a context");
        assert_eq!(got, 4100);
    }

    /// The cache guard's rows: `from` when its state holds there, else the
    /// largest context whose state holds the budget there — the budget
    /// shrinking with the context as a unified pool's need grows with it —
    /// rounded down to the step, a context under the step its own; a budget
    /// that does not move with the context (`--cache-ram`) is the plain
    /// bound; and a search no longer than a halving takes.
    #[test]
    fn within_cache_rows() {
        let state = |n: usize| 3 * n as u64 + 10;
        // Holds while 3n + 10 <= 1000 - 2n: through 198.
        let shrinking = |n: usize| Ok::<_, Box<dyn std::error::Error>>(1000 - 2 * n as u64);
        let rows: [(usize, usize, usize); 6] = [
            (150, 16, 150),
            (198, 16, 198),
            (199, 1, 198),
            (199, 16, 192),
            (400, 16, 192),
            (400, 256, 198),
        ];
        for (from, step, want) in rows {
            let probes = Probes::new();
            let counted = |n: usize| {
                probes.take();
                shrinking(n)
            };
            let got = within_cache(from, step, &state, &counted).expect("a context");
            assert_eq!(got, want, "from {from}, step {step}");
            assert!(state(got) <= 1000 - 2 * got as u64);
        }
        // A constant budget of 100 B holds through 30 positions; a context
        // under the step is its own.
        let constant = |_: usize| Ok::<_, Box<dyn std::error::Error>>(100u64);
        assert_eq!(
            within_cache(262_144, 256, &state, &constant).expect("a context"),
            30
        );
        assert_eq!(
            within_cache(30, 256, &state, &constant).expect("a context"),
            30
        );
        assert_eq!(
            within_cache(31, 256, &state, &constant).expect("a context"),
            30
        );
    }

    /// `from` is tried first: a budget that holds at `from` and fails in
    /// between (the plan's table changing tier at a context) keeps `from`,
    /// where a bisection over `1..=from` lands under the gap.
    #[test]
    fn within_cache_tries_from_before_the_search() {
        let state = |n: usize| n as u64;
        let gap = |n: usize| {
            Ok::<_, Box<dyn std::error::Error>>(if n <= 100 || n >= 240 { 10_000 } else { 0 })
        };
        assert_eq!(within_cache(250, 1, &state, &gap).expect("a context"), 250);
        assert_eq!(within_cache(239, 1, &state, &gap).expect("a context"), 100);
    }

    /// The cache guard's refusals, by name: no context holds one position's
    /// state, a rounding that lands where the state does not hold, and a
    /// budget read that fails.
    #[test]
    fn within_cache_refuses_by_name() {
        let state = |n: usize| 3 * n as u64 + 10;
        let none = |_: usize| Ok::<_, Box<dyn std::error::Error>>(5u64);
        let err = within_cache(1000, 16, &state, &none).expect_err("no state holds");
        assert_eq!(
            err.downcast_ref::<CtxError>().map(ToString::to_string),
            Some(
                "the prompt cache's 5 bytes hold no sequence state (13 bytes at one position, two \
                 recurrent copies among its terms); --cache-ram 0 turns it off"
                    .to_string()
            )
        );
        // Holds at 1, 5 and 7 only: the bisection over 1..=8 ends at 7,
        // which the step of 4 rounds to 4, where the state does not hold.
        let state = |n: usize| n as u64;
        let sparse = |n: usize| {
            Ok::<_, Box<dyn std::error::Error>>(if matches!(n, 1 | 5 | 7) { 100 } else { 0 })
        };
        let err = within_cache(9, 4, &state, &sparse).expect_err("a rounding past the bound");
        assert_eq!(
            err.downcast_ref::<CtxError>().map(ToString::to_string),
            Some(
                "the prompt cache's bound rounded to a context of 4 positions, whose state of 4 \
                 bytes passes the budget of 0 bytes there"
                    .to_string()
            )
        );
        let refused = |n: usize| -> Result<u64, Box<dyn std::error::Error>> {
            if n < 100 {
                Ok(1_000)
            } else {
                Err("refused".into())
            }
        };
        let err = within_cache(200, 1, &state, &refused).expect_err("a plan refused");
        assert_eq!(err.to_string(), "refused");

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
            })
        }
    }

    /// The default over `probe`, with the probes it took.
    fn run(
        fit: usize,
        step: usize,
        margin: u64,
        probe: impl Fn(usize) -> Result<ExpertBytes, String>,
    ) -> (Result<SlotCtx, String>, usize) {
        let probes = Probes::new();
        let counted = |c: usize| {
            probes.take();
            probe(c)
        };
        let got = default_slot_ctx(fit, AT, step, margin, &counted);
        (got, probes.0.get())
    }

    /// The probes a default may take over a context range: the base's, the
    /// fit's and a bisection's `⌈log2(n + 1)⌉` over the `n` multiples of
    /// `step` between them, counted here one at a time.
    fn probe_bound(fit: usize, step: usize) -> usize {
        let n = (1..)
            .map(|i| (AT / step + i) * step)
            .take_while(|&c| c < fit)
            .count();
        2 + (n + 1).next_power_of_two().ilog2() as usize
    }

    /// A fit under the base is the answer, off the step as it is, by the
    /// `card` rule: the one probe is the base's, which is the chosen plan's
    /// too; a fit at the base is the margin rule's, with no step past it.
    #[test]
    fn default_ctx_under_the_base_is_the_fit_by_the_card_rule() {
        let (got, probes) = run(3000, STEP, MARGIN, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule, got.base_ctx), (3000, "card", 3000));
        assert_eq!(got.base, got.chosen, "the plan at the fit is both");
        assert_eq!(probes, 1);
        let (got, probes) = run(AT - 1, STEP, MARGIN, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule), (AT - 1, "card"));
        assert_eq!(probes, 1);
        let (got, probes) = run(AT, STEP, MARGIN, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule, got.base_ctx), (AT, "margin", AT));
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
            (9000, 256, 4864, "5096 rounded down to a step of 256"),
            (9000, 1024, 4096, "no whole step past the base holds"),
            (
                9000,
                100,
                5000,
                "a multiple of 100 itself, off the base's 4096",
            ),
            (262_144, 256, 4864, "the wide range"),
        ];
        for (fit, step, want, why) in rows {
            let (got, probes) = run(fit, step, MARGIN, plan(CARD, AT, 1, |_| 0));
            let got = got.expect("a plan");
            assert_eq!((got.ctx, got.rule), (want, "margin"), "{why}");
            assert_eq!(got.margin_ctx, want, "{why}");
            assert_eq!(got.base_ctx, AT, "{why}");
            let at = |c| plan(CARD, AT, 1, |_| 0)(c).expect("a plan");
            assert_eq!(got.base, at(AT), "{why}");
            assert_eq!(
                got.chosen,
                at(want),
                "the plan at the chosen context: {why}"
            );
            assert!(probes <= probe_bound(fit, step), "{why}: {probes} probes");
        }
    }

    /// Every context within is the fit — off the step, and at the margin's
    /// last byte (at most `margin` fewer is within) — after the base's probe
    /// and the fit's alone; one byte of margin less and the answer is the
    /// last step.
    #[test]
    fn default_ctx_every_context_within_is_the_fit() {
        let lost_at_fit = 9000 - AT as u64;
        let (got, probes) = run(9000, STEP, lost_at_fit, plan(CARD, AT, 1, |_| 0));
        let got = got.expect("a plan");
        assert_eq!((got.ctx, got.rule), (9000, "margin"));
        assert_eq!(got.chosen.card, CARD - lost_at_fit);
        assert_eq!(probes, 2, "the base and the fit");
        let (got, _) = run(9000, STEP, lost_at_fit - 1, plan(CARD, AT, 1, |_| 0));
        assert_eq!(
            got.expect("a plan").ctx,
            8960,
            "8999 rounded down to a step"
        );
    }

    /// A plan the card refuses past a context is not within there, whatever
    /// its bytes, and a refused fit is not the rule's error: the scan goes on
    /// below it.
    #[test]
    fn default_ctx_refused_plan_past_the_base_is_not_within() {
        let card = plan(CARD, AT, 1, |_| 0);
        let refused_past = |at: usize| {
            let card = plan(CARD, AT, 1, |_| 0);
            move |c: usize| {
                if c > at {
                    Err("refused".to_string())
                } else {
                    card(c)
                }
            }
        };
        // The margin holds through 5096; the card refuses past 4600.
        let (got, _) = run(9000, STEP, MARGIN, refused_past(4600));
        assert_eq!(
            got.expect("a plan").ctx,
            4352,
            "4600 rounded down to a step"
        );
        // A refusal at the fit alone leaves the margin's answer.
        let (got, _) = run(9000, STEP, MARGIN, refused_past(8999));
        let want = run(9000, STEP, MARGIN, &card).0.expect("a plan");
        assert_eq!(
            got.expect("a plan"),
            want,
            "the fit's refusal is not within"
        );
        assert_eq!(want.ctx, 4864);
    }

    /// A whole-card load — every expert on the card, nothing on the NVMe tier,
    /// at every context — is within a margin of none: the fit, after two
    /// probes, however wide the range.
    #[test]
    fn default_ctx_whole_card_load_is_the_fit() {
        for fit in [5000, 9000, 262_144] {
            let (got, probes) = run(fit, STEP, 0, |_| {
                Ok(ExpertBytes {
                    card: CARD,
                    nvme: 0,
                })
            });
            let got = got.expect("a plan");
            assert_eq!((got.ctx, got.rule), (fit, "margin"), "fit {fit}");
            assert_eq!(probes, 2, "fit {fit}: the base and the fit");
        }
    }

    /// NVMe expert bytes growing past the base's stop the default below the
    /// context where they do, though the card's margin holds there: the
    /// base's own NVMe bytes are what it keeps (no more than those, not
    /// none), and bytes that stay or fall are within.
    #[test]
    fn default_ctx_stops_where_nvme_grows() {
        let grows_past_6000 = |base: u64| move |c: usize| base + c.saturating_sub(6000) as u64 * 10;
        // The margin never binds; the NVMe bytes grow at 6001 and the answer
        // is the last multiple of 256 under it.
        for base in [0, 500] {
            let (got, probes) = run(
                262_144,
                STEP,
                1 << 40,
                plan(CARD, AT, 1, grows_past_6000(base)),
            );
            let got = got.expect("a plan");
            assert_eq!((got.ctx, got.rule), (5888, "margin"), "base NVMe {base}");
            assert_eq!(
                (got.base.nvme, got.chosen.nvme),
                (base, base),
                "the base's NVMe bytes are the bound"
            );
            assert!(probes <= probe_bound(262_144, STEP), "{probes} probes");
        }
        // The same NVMe bytes at every context, or fewer, are within.
        let (got, _) = run(9000, STEP, 1 << 40, plan(CARD, AT, 1, |_| 500));
        assert_eq!(got.expect("a plan").ctx, 9000, "equal NVMe bytes");
        let fewer = |c: usize| if c <= AT { 500 } else { 400 };
        let (got, _) = run(9000, STEP, 1 << 40, plan(CARD, AT, 1, fewer));
        assert_eq!(got.expect("a plan").ctx, 9000, "fewer NVMe bytes");
    }

    /// The base's own plan refused is the rule's error, by the card rule as by
    /// the margin's, and nothing is probed past it.
    #[test]
    fn default_ctx_base_probe_error_is_returned() {
        let card = plan(CARD, AT, 1, |_| 0);
        let base_refused = |c: usize| {
            if c == AT {
                Err("no base".to_string())
            } else {
                card(c)
            }
        };
        let (got, probes) = run(9000, STEP, MARGIN, base_refused);
        assert_eq!((got, probes), (Err("no base".to_string()), 1));
        let all_refused = |_: usize| Err::<ExpertBytes, _>("none".to_string());
        let (got, probes) = run(3000, STEP, MARGIN, all_refused);
        assert_eq!((got, probes), (Err("none".to_string()), 1), "the card rule");
    }

    /// A step of 0 is refused by name, not by the divide it would reach.
    #[test]
    #[should_panic(expected = "a context step of 0")]
    fn default_ctx_step_of_zero_is_refused_by_name() {
        let _ = run(9000, 0, MARGIN, plan(CARD, AT, 1, |_| 0));
    }

    /// The `ctx` line the seats print: its field names are what the gates
    /// read, in this order — the margin rule's line (an example plan whose
    /// card loses 48 KiB a position past the base, for a margin of 1 GiB) and
    /// the card rule's.
    #[test]
    fn default_ctx_line_names_its_terms() {
        let probe = plan(9_000_000_000, AT, 49_152, |_| 0);
        let got = default_slot_ctx(262_144, AT, STEP, 1 << 30, &probe).expect("a plan");
        assert_eq!(
            got.line(2),
            "ctx rule=margin ctx=25856 slots=2 total=51712 fit=262144 margin_ctx=25856 \
             base=4096 base_card_expert_bytes=9000000000 base_nvme_expert_bytes=0 \
             card_expert_bytes=7930452480 nvme_expert_bytes=0 lost_bytes=1069547520 \
             margin_bytes=1073741824"
        );
        let got = default_slot_ctx(3000, AT, STEP, 1 << 30, &probe).expect("a plan");
        assert_eq!(
            got.line(1),
            "ctx rule=card ctx=3000 slots=1 total=3000 fit=3000 margin_ctx=3000 base=3000 \
             base_card_expert_bytes=9000000000 base_nvme_expert_bytes=0 \
             card_expert_bytes=9000000000 nvme_expert_bytes=0 lost_bytes=0 \
             margin_bytes=1073741824"
        );
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

    /// The placed search keeps the floor's card experts: a context fits
    /// while its plan holds as many as the floor's does; a plan that cannot
    /// build does not fit, and a floor whose plan cannot build is no answer,
    /// probed once.
    #[test]
    fn searched_placed_rows() {
        const GRAN: usize = 1024;
        let calls = Cell::new(0usize);
        // 100 experts to 20,000 positions, 99 to 40,000, none after: a plan
        // that cannot build past 60,000.
        let experts = |c: usize| {
            calls.set(calls.get() + 1);
            Ok::<_, Infallible>(if c <= 20_000 {
                Some(100)
            } else if c <= 40_000 {
                Some(99)
            } else if c <= 60_000 {
                Some(0)
            } else {
                None
            })
        };
        let run = |trained: Option<usize>, floor: usize| {
            calls.set(0);
            let got = searched_placed(trained, floor, GRAN, &experts).expect("infallible");
            (got, calls.get())
        };
        // The floor's split is 100 experts: the answer is the last grid step
        // that keeps them, not the trained context, not one that keeps 99.
        assert_eq!(run(Some(262_144), 4096).0, Some(4096 + 15 * GRAN), "19,456");
        // The trained context inside the split is its own answer.
        assert_eq!(run(Some(20_000), 4096).0, Some(20_000));
        assert_eq!(run(Some(10_000), 4096).0, Some(10_000));
        // A floor past the split (99 experts) holds to 99, not to 100.
        assert_eq!(
            run(Some(262_144), 30_000).0,
            Some(30_000 + 9 * GRAN),
            "39,216"
        );
        // No plan at the floor: no answer, one probe.
        assert_eq!(run(Some(262_144), 70_000), (None, 1));
        // No trained context: no answer, after the floor's one probe.
        assert_eq!(run(None, 4096), (None, 1));
    }
}
