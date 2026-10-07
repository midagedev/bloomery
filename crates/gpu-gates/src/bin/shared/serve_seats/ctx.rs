//! The serving seats' shared `--ctx` rules: the bisection the seats'
//! searches run ([`largest`]), the expert-margin guard that bounds a default
//! context past the seat's floor ([`within_margin`], qwen38's margin rule,
//! which the glm seat lifts), and the slot count a `--ctx` flag buys
//! ([`slots_of`]) — more positions on the card push card experts to the
//! host, and decode crawls, so a context the flag did not name never trades
//! away more than the plan's
//! [`MARGIN`](model::placement::workstation::MARGIN) of stage-card expert
//! bytes. The plan a seat probes at each context is its own (the model's
//! placements differ); these are the terms the seats already share.

use bloomery_gpu_gates::GateError;
use model::placement::workstation::MARGIN;

/// The resident sequences a seat serves and what set the count, for its
/// `parallel` line's `from` word: `parallel` (`--parallel`) as given
/// (`flag`); unset beside a set `--ctx`, one slot at the whole context —
/// a `--ctx-size` is what one request gets, as llama-server reads it
/// (`ctx`); unset with no context named, the seat's own default
/// (`seat_default`) over its automatic context (`default`). A seat's own
/// parse refuses `--parallel 0` before this; it is refused here too, by
/// name, so no call can serve a count of no slot.
pub fn slots_of(
    parallel: Option<usize>,
    ctx_set: bool,
    seat_default: usize,
) -> Result<(usize, &'static str), GateError> {
    let (slots, from) = match parallel {
        Some(n) => (n, "flag"),
        None if ctx_set => (1, "ctx"),
        None => (seat_default, "default"),
    };
    if slots == 0 {
        return Err("--parallel 0: the server serves no slot".into());
    }
    Ok((slots, from))
}

/// The largest `c` in `lo..=hi` for which `ok` holds, `ok(lo)` given: `ok`
/// holds below a point and not above it.
pub fn largest(
    mut lo: usize,
    mut hi: usize,
    ok: impl Fn(usize) -> Result<bool, GateError>,
) -> Result<usize, GateError> {
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
/// most [`MARGIN`](model::placement::workstation::MARGIN) fewer bytes than
/// `base_bytes` (the plan's at `base_at`): `fit` when every context to
/// there does, `base_at` when no step past it does, else the largest that
/// does rounded down to `step` (never under `base_at`). A context whose plan
/// is refused is not within, as the fit's own search reads it: a probe is
/// never the seat's refusal.
pub fn within_margin(
    fit: usize,
    base_at: usize,
    base_bytes: u64,
    step: usize,
    card_bytes: &dyn Fn(usize) -> Result<u64, GateError>,
) -> Result<usize, GateError> {
    let within = |c: usize| Ok(card_bytes(c).is_ok_and(|b| base_bytes.saturating_sub(b) <= MARGIN));
    let c = largest(base_at, fit, within)?;
    if c == fit {
        return Ok(c);
    }
    Ok((c / step * step).max(base_at))
}
