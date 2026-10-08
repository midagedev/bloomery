//! The serving seats' shared `--ctx` rules: the bisection the seats'
//! searches run ([`largest`]), the expert-margin guard that bounds a default
//! context past the seat's floor ([`within_margin`], qwen38's margin rule,
//! which the glm seat lifts), the search over a context grid both qwen3
//! defaults run ([`searched`]) and the slot count a `--ctx` flag buys
//! ([`slots_of`]) — more positions on the card push card experts to the
//! host, and decode crawls, so a context the flag did not name never trades
//! away more than the plan's margin of stage-card expert bytes. The plan a
//! seat probes at each context is its own (the model's placements differ);
//! these are the terms the seats already share, and every model-specific
//! input — the plan's bytes, the margin, the step, the trained context —
//! enters as an argument.

/// Why a slot count was refused.
#[derive(Debug, thiserror::Error)]
pub enum CtxError {
    /// A seat's own parse refuses `--parallel 0` before the rule; the rule
    /// refuses it too, so no call can serve a count of no slot.
    #[error("--parallel 0: the server serves no slot")]
    NoSlot,
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

/// The largest multiple of `step` up to `fit` whose `card_bytes` holds at
/// most `margin` fewer bytes than `base_bytes` (the plan's at `base_at`):
/// `fit` when every context to there does, `base_at` when no step past it
/// does, else the largest that does rounded down to `step` (never under
/// `base_at`). A context whose plan is refused is not within, as the fit's
/// own search reads it: a probe is never the seat's refusal.
///
/// # Errors
/// None today: a probe's error is a context that is not within. The result
/// type is the seat's own so its call reads as the fit's search does.
pub fn within_margin<E>(
    fit: usize,
    base_at: usize,
    base_bytes: u64,
    step: usize,
    margin: u64,
    card_bytes: &dyn Fn(usize) -> Result<u64, E>,
) -> Result<usize, E> {
    let within =
        |c: usize| Ok::<_, E>(card_bytes(c).is_ok_and(|b| base_bytes.saturating_sub(b) <= margin));
    let c = largest(base_at, fit, within)?;
    if c == fit {
        return Ok(c);
    }
    Ok((c / step * step).max(base_at))
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
