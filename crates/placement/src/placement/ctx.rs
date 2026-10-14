//! What a set `--ctx` is held to, and the bisection a search runs.
//!
//! A context the flag names is checked per slot against the model's limit
//! ([`CtxLimit`], [`check_slot_ctx`]) before any plan: past it the load
//! stops by name ([`PastLimit`]), never on a context cut to fit. The
//! context an unset flag serves is [`super::ctx_ask::choose`], not a search
//! in this module.
//!
//! [`slots_of`] is the resident sequences a seat serves and the word for
//! what set the count. [`largest`] is the bisection: the greatest `c` in a
//! range for which a monotone probe holds.

/// Why a slot count was refused.
#[derive(Debug, thiserror::Error)]
pub enum CtxError {
    /// A seat's own parse refuses `--parallel 0` before the rule; the rule
    /// refuses it too, so no call can serve a count of no slot.
    #[error("--parallel 0: the server serves no slot")]
    NoSlot,
}

/// The most positions one slot may be asked for: the least of the file's
/// trained context and, where the engine has them, a kernel's position bound
/// and the bound the gates have checked up to, each with the word its
/// refusal names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtxLimit {
    /// The file's `context_length`.
    pub trained: u64,
    /// A position bound a kernel or the cache's index type imposes.
    pub kernel: Option<(u64, &'static str)>,
    /// The deepest context the gates have checked the model at.
    pub verified: Option<(u64, &'static str)>,
}

impl CtxLimit {
    /// The least bound and its word: `trained`, or the bound's own; of equal
    /// bounds the earlier word (trained, kernel, verified).
    pub fn limit(&self) -> (u64, &'static str) {
        let mut least = (self.trained, "trained");
        for bound in [self.kernel, self.verified].into_iter().flatten() {
            if bound.0 < least.0 {
                least = bound;
            }
        }
        least
    }
}

/// A context asked past the model's limit ([`check_slot_ctx`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "--ctx asks {ctx} positions a slot, past this model's limit of {limit} ({bound}; the file \
     was trained at {trained}): the load stops; ask {limit} or fewer a slot"
)]
pub struct PastLimit {
    /// The positions a slot was asked for.
    pub ctx: u64,
    /// [`CtxLimit::limit`]'s bound.
    pub limit: u64,
    /// The word that names the bound.
    pub bound: &'static str,
    /// The file's trained context.
    pub trained: u64,
}

/// The limit when a slot's `ctx` is within it, else [`PastLimit`]: the one
/// comparison every seat's set `--ctx` goes through, never a cut.
///
/// # Errors
/// [`PastLimit`] when `ctx` is past [`CtxLimit::limit`].
pub fn check_slot_ctx(ctx: u64, bounds: &CtxLimit) -> Result<u64, PastLimit> {
    let (limit, bound) = bounds.limit();
    if ctx > limit {
        return Err(PastLimit {
            ctx,
            limit,
            bound,
            trained: bounds.trained,
        });
    }
    Ok(limit)
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
    mut ok: impl FnMut(usize) -> Result<bool, E>,
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
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

    /// The limit is the least bound with its word (of equal bounds the
    /// earlier word); a slot at it gets it back, one past it is
    /// [`PastLimit`] naming the bound, its word and the trained context.
    #[test]
    fn a_slot_past_the_least_bound_is_refused_by_its_name() {
        type Row = (
            u64,
            Option<(u64, &'static str)>,
            Option<(u64, &'static str)>,
            (u64, &'static str),
        );
        let rows: [Row; 5] = [
            (262_144, None, None, (262_144, "trained")),
            (1 << 20, Some((1 << 31, "i32")), None, (1 << 20, "trained")),
            (
                1 << 20,
                Some((1 << 31, "i32")),
                Some((16_384, "ORACLE_POSITIONS")),
                (16_384, "ORACLE_POSITIONS"),
            ),
            (
                u64::MAX,
                Some((u64::from(u32::MAX), "KERNEL_POSITIONS")),
                None,
                (u64::from(u32::MAX), "KERNEL_POSITIONS"),
            ),
            (
                4096,
                Some((4096, "i32")),
                Some((4096, "ORACLE_POSITIONS")),
                (4096, "trained"),
            ),
        ];
        for (trained, kernel, verified, (limit, source)) in rows {
            let bounds = CtxLimit {
                trained,
                kernel,
                verified,
            };
            assert_eq!(bounds.limit(), (limit, source));
            assert_eq!(check_slot_ctx(limit, &bounds), Ok(limit));
            let past = check_slot_ctx(limit + 1, &bounds).expect_err("one past the limit");
            assert_eq!(
                past,
                PastLimit {
                    ctx: limit + 1,
                    limit,
                    bound: source,
                    trained,
                }
            );
            assert_eq!(
                past.to_string(),
                format!(
                    "--ctx asks {} positions a slot, past this model's limit of {limit} ({source}; \
                     the file was trained at {trained}): the load stops; ask {limit} or \
                     fewer a slot",
                    limit + 1
                )
            );
        }
    }
}
