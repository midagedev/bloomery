//! The expert-stream split rule: for one layer of one prompt unit, which host
//! experts the card pool admits, which a transient ring streams over the copy
//! lane, and which the host union keeps — a pure function of the unit's routed
//! counts and constants measured at load. No machine, card or model name
//! enters the rule; every constant is an input.
//!
//! **Two rates.** A copy gets one of two rates. Alone — the pick phase, while
//! the walk thread waits for the staging thread — it runs at the lane's
//! measured rate `lane_b_per_us`. Beside the union — every copy a pick leaves
//! behind it, and every copy the ring makes — the union's expert-weight reads
//! hold the host's DRAM for their share of its own window, and a copy beside
//! it runs at [`beside_b_per_us`]: the lane's rate less the union's burst
//! share [`union_burst_share`], a pure function of the unit's own counts
//! (its listed host experts and their columns) and the union's costs W and c.
//! The rule prices every copy at the rate it gets, never at the lane's alone.
//!
//! An expert routed `m` columns of the unit leaves the host union when it
//! costs more there than on the lane plus the card:
//! `max(W, c·m) > b/r + f + k·m`, for the host union's fixed cost W and
//! per-column cost c, the lane's beside-union rate r, the expert's b bytes
//! and the card's fixed cost f and per-column cost k. [`m_star`] is that
//! inequality's crossover on its per-column branch, `(b/r + f)/(c − k)`;
//! while W is at most `b/r + f` the experts that stream are exactly those
//! routed past it. [`m_min`] is the least unit width that streams: a column
//! spreads its top-k picks over the choice set, so the average expert's
//! count `top_k·m/experts` first clears [`m_star`] at
//! `m_min = m_star·experts/top_k`. Below it the ring stays dark and the pool
//! admits alone. Both lists fill in the one ranking [`rank_key`], so the
//! same counts give the same split. [`stream_tail`] is the ring's half
//! alone, over a host set whose admits are already taken; [`split`] takes
//! its tail through it, so the stream inequality has one caller.
//!
//! **The walk.** A split pick's admits and its backlog bound come from one
//! balance over the two chains a layer runs after its pick returns: the
//! card's — the admits' copies beside the union, then the route's tail τ
//! ([`Constants::card_tail_us`]) — and the host union's, each admitted
//! expert's cost leaving it. The tail is a level per unit width, not a
//! family constant: [`unit_constants`] prices it per card pick, a rate the
//! family passes, from the unit's own counts. [`admit_walk`] walks the host
//! experts hottest first, admitting while the card chain stays under the
//! union, and returns the floor (the coldest admit's count, a
//! rank-contiguous set) and the backlog bound: exactly the jobs whose copies
//! fit inside the union's shadow, the rest staging alone ahead of it. In a
//! call not kept, an admit in its last unit whose way back the lane cannot
//! take inside the walk is priced at two copies ([`ReturnCost`]).
//! [`walk_gate`] is the least unit width whose picks could pay the walk's
//! first admit: a prompt narrower than it admits nothing whatever its
//! counts, so a caller gates on it before it counts.

use std::cmp::Reverse;
use std::fmt;

/// The rule's constants: the machine's measured rates and the layer's
/// geometry, every field's unit in its name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constants {
    /// The copy lane's host→card rate [B/µs], measured alone: the pick
    /// phase's rate, and the base the beside-union rate takes from.
    pub lane_b_per_us: f64,
    /// The host union's per-column cost of one expert [µs/col].
    pub host_us_per_col: f64,
    /// The host union's fixed cost of one expert [µs].
    pub host_us_fixed: f64,
    /// The card's fixed cost of one streamed expert [µs].
    pub card_us_fixed: f64,
    /// The card's per-column cost of one streamed expert [µs/col].
    pub card_us_per_col: f64,
    /// The union's weight-listing burst share of its own window: the
    /// fraction of the union's window its expert-weight reads hold the
    /// host's DRAM, during which a copy beside it gets nothing of the lane's
    /// rate. A load's family row carries 0 (no unit known); the paths that
    /// know a unit's counts set it through [`union_burst_share`].
    pub union_burst_share: f64,
    /// The card route's tail past a layer's last copy [µs]: the work the
    /// pipeline cannot hide behind the copies, the walk's card chain ends
    /// with it. A family row carries 0; [`unit_constants`] sets the unit's
    /// own, a family rate times the unit's card picks. The ring's rule and
    /// [`walk_gate`] never read it.
    pub card_tail_us: f64,
    /// One routed expert's bytes in the layer [B].
    pub expert_b: u64,
    /// The router's choice set: routed experts in the layer.
    pub experts: usize,
    /// The router's picks per column.
    pub top_k: usize,
}

/// The rule for one expert routed `count` columns of the unit: it streams
/// when the host union's cost is above the lane's and the card's together —
/// the lane's copy priced beside the union, where it runs.
fn streams(k: &Constants, count: u32) -> bool {
    let m = f64::from(count);
    let host = k.host_us_fixed.max(k.host_us_per_col * m);
    let lane_card =
        k.expert_b as f64 / beside_b_per_us(k) + k.card_us_fixed + k.card_us_per_col * m;
    host > lane_card
}

/// The union's weight-listing burst share of its own window for a unit that
/// routes `cols` columns over `listed` host experts: the union's fixed
/// weight-reading cost against its whole cost, the share of its window a
/// copy beside it loses. Pure in the unit's counts and the union's costs;
/// 0 with nothing listed (no weight reads to lose the lane to).
#[must_use]
pub fn union_burst_share(cols: u64, listed: u64, k: &Constants) -> f64 {
    let fixed = k.host_us_fixed * listed as f64;
    let whole = fixed + k.host_us_per_col * cols as f64;
    if whole <= 0.0 {
        return 0.0;
    }
    fixed / whole
}

/// A unit's constants for the walk: `k` with the unit's own burst share
/// ([`union_burst_share`] over the host experts' columns and the listed ones
/// among them) and its card tail, `card_tail_us_per_pick` [µs per card pick]
/// times the unit's card picks — the routed picks `counts` holds less the
/// columns the `host` experts take, the host set as the pick finds it, before
/// its admits move. The route's tail is a level per unit
/// width, so it is priced from the unit's counts and never set once for the
/// family; `card_tail_us_per_pick` is the family's rate, 0 where no record
/// has priced its route.
///
/// Refused by name: counts of another length than the layer's experts, a
/// host list naming an expert outside the layer, or one that takes more
/// columns than the unit routed (an expert named twice), and a rate that is
/// not finite and 0 or more.
///
/// # Errors
/// Every refusal names its input; see [`SplitError`].
pub fn unit_constants(
    k: &Constants,
    card_tail_us_per_pick: f64,
    counts: &[u32],
    host: &[u32],
) -> Result<Constants, SplitError> {
    if !(card_tail_us_per_pick.is_finite() && card_tail_us_per_pick >= 0.0) {
        return Err(SplitError::Param {
            name: "card_tail_us_per_pick",
            range: "finite, 0 or more",
        });
    }
    let total = check_counts(counts, k)?;
    let mut listed = 0u64;
    let mut cols = 0u64;
    for &id in host {
        let count = *counts.get(id as usize).ok_or(SplitError::HostOutOfRange {
            id,
            experts: k.experts,
        })?;
        listed += u64::from(count > 0);
        cols += u64::from(count);
    }
    let card_picks = total.checked_sub(cols).ok_or(SplitError::Param {
        name: "host",
        range: "each expert once",
    })?;
    let mut unit = *k;
    unit.union_burst_share = union_burst_share(cols, listed, &unit);
    unit.card_tail_us = card_tail_us_per_pick * card_picks as f64;
    Ok(unit)
}

/// The beside-union copy rate [B/µs]: the lane's measured rate less the
/// union's burst share of it — the rate every copy the pick leaves behind
/// it, and every ring copy, gets.
#[must_use]
pub fn beside_b_per_us(k: &Constants) -> f64 {
    k.lane_b_per_us * (1.0 - k.union_burst_share)
}

/// The stream floor: the count past which the rule's per-column branch
/// streams, `c·m > b/r + f + k·m` solved for m, `(b/r + f)/(c − k)` at the
/// beside-union rate r. A card column that costs at least the host's never
/// pays its fixed cost back: the floor is then infinite. In columns.
#[must_use]
pub fn m_star(k: &Constants) -> f64 {
    let per_col = k.host_us_per_col - k.card_us_per_col;
    if per_col <= 0.0 {
        return f64::INFINITY;
    }
    (k.expert_b as f64 / beside_b_per_us(k) + k.card_us_fixed) / per_col
}

/// The least unit width that streams: a column's top-k picks spread over the
/// choice set put `top_k·m/experts` columns on the average expert, and the
/// unit's mass first clears [`m_star`] when that mean does. An infinite floor
/// gives `u64::MAX`: no unit streams. In columns.
#[must_use]
pub fn m_min(k: &Constants) -> u64 {
    (m_star(k) * k.experts as f64 / k.top_k as f64).ceil() as u64
}

/// The one ranking the pool's admits and the ring's tail share: count
/// descending, id ascending — the order [`split`] fills both lists in.
#[must_use]
pub fn rank_key(count: u32, id: u32) -> (Reverse<u32>, u32) {
    (Reverse(count), id)
}

/// What an admit's way back costs the walk ([`admit_walk`]): a call not kept
/// copies, at its end or in its last unit's walk, each expert the call sent to
/// the host back into the slot of the one admitted in its place, so an admit
/// in that call's last unit puts a second copy on the lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnCost {
    /// No return copy the walk pays: a call that keeps its placement, an
    /// earlier unit's admit (its return runs in the last unit), or a last
    /// unit whose lane keeps up with its returns inside the walk's waits for
    /// the card's fronts.
    Hidden,
    /// The return copy lands at the call's end, alone on the lane: the walk
    /// owes returns it has not issued, or no front follows this layer's.
    /// The admit is priced at two copies: its own beside the union, its
    /// return's at the lane's measured rate alone.
    Exposed,
}

/// A split pick's plan from [`admit_walk`]: the floor its admits come from
/// and the backlog bound its copies leave the staging thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmitPlan {
    /// The least count an admitted expert has in the pick's counts: the
    /// coldest admit of the walk's rank-contiguous set, so a floor
    /// reproduces the set exactly. `u32::MAX` when the walk admits nothing.
    pub floor: u32,
    /// The most of the call's jobs the staging thread may leave unstaged
    /// when the pick issues another: the jobs whose copies beside the union
    /// fit inside its shadow, the rest staging alone ahead of it. 0 when the
    /// walk admits nothing (no job ever issues at the bound).
    pub backlog: u64,
}

/// The ranked chain-balance walk: a split pick's admits and its backlog
/// bound, over the layer's routed `counts` (one a router expert) and its
/// host-tier `host` ids in any order, at the unit's constants `k` (its
/// beside-union share set, its card tail the model's own). The card chain
/// runs the admits' copies beside the union and ends at the route's tail;
/// the host chain loses each admitted expert's union cost. Walking hottest
/// first ([`rank_key`]), an expert is admitted while the card chain after it
/// stays under the union it leaves — the wall the walk balances is the
/// longer of the two chains, so past that point an admit lengthens the
/// layer. The floor is the coldest admit's count; the backlog bound leaves
/// beside the union exactly the jobs whose copies fit inside its shadow,
/// `⌈(U − τ)·r/b⌉` clamped to the caller's queue floor and to the admits
/// there are. Under [`ReturnCost::Exposed`] each admit's card chain carries
/// its return copy too, `b / lane_b_per_us`; the shadow bound counts the
/// pick's own jobs, as before.
///
/// Refused by name: the constants the derivations cannot read, counts of
/// another length than the layer's experts, and a host list naming an
/// expert outside the layer or twice.
///
/// # Errors
/// Every refusal names its input; see [`SplitError`].
pub fn admit_walk(
    counts: &[u32],
    host: &[u32],
    k: &Constants,
    queue_floor: u64,
    ret: ReturnCost,
) -> Result<AdmitPlan, SplitError> {
    check_constants(k)?;
    check_counts(counts, k)?;
    check_host(host, k, &mut Vec::new())?;
    let copy_us = k.expert_b as f64 / beside_b_per_us(k);
    let admit_us = match ret {
        ReturnCost::Hidden => copy_us,
        ReturnCost::Exposed => copy_us + k.expert_b as f64 / k.lane_b_per_us,
    };
    let host_us = |m: u32| k.host_us_fixed.max(k.host_us_per_col * f64::from(m));
    let mut ranked: Vec<u32> = host.to_vec();
    ranked.retain(|&id| counts[id as usize] > 0);
    ranked.sort_unstable_by_key(|&id| rank_key(counts[id as usize], id));
    let union = ranked
        .iter()
        .map(|&id| host_us(counts[id as usize]))
        .sum::<f64>();
    let mut card = k.card_tail_us;
    let mut left = union;
    let mut admits: u64 = 0;
    let mut plan = AdmitPlan {
        floor: u32::MAX,
        backlog: 0,
    };
    for &id in &ranked {
        let count = counts[id as usize];
        let card_next = card + admit_us;
        let left_next = left - host_us(count);
        if card_next > left_next {
            break;
        }
        card = card_next;
        left = left_next;
        plan.floor = count;
        admits += 1;
    }
    if plan.floor == u32::MAX {
        return Ok(plan);
    }
    // The shadow bound: the jobs whose copies fit inside the union the walk
    // left, the rest staging alone ahead of it — never fewer than the
    // machine's queue floor, never more than the admits there are.
    let shadow = (left - k.card_tail_us).max(0.0) * beside_b_per_us(k) / k.expert_b as f64;
    let bound = shadow
        .ceil()
        .clamp(queue_floor.min(admits) as f64, admits as f64);
    plan.backlog = bound as u64;
    Ok(plan)
}

/// The least unit width whose picks could pay the walk's first admit: an
/// admit needs the union to cost at least one copy beside it (the first
/// admit's copy and the card's tail ahead of it, the union less the expert it
/// loses), and a unit of `m` columns costs the union at most `top_k·m` of
/// both the fixed and the per-column kind — an expert listed at most once a
/// column. The tail is not an input: it is the rate times the card picks,
/// and a unit whose every pick sits on the host has none, where the union is
/// the largest, so a bound that never skips a unit the walk would admit from
/// takes it at 0. The copy is priced at `k`'s beside-union rate; a unit's own
/// share only slows it, so a `k` with no share (the family's) under-states
/// the gate and never over-states it. A prompt narrower than the gate admits
/// nothing whatever its counts, so a caller gates before it counts. In
/// columns.
#[must_use]
pub fn walk_gate(k: &Constants) -> u32 {
    let copy_us = k.expert_b as f64 / beside_b_per_us(k);
    let per_pick = (k.host_us_fixed + k.host_us_per_col) * k.top_k as f64;
    if per_pick <= 0.0 {
        return 1;
    }
    let gate = (copy_us / per_pick).ceil();
    if !(gate.is_finite() && gate >= 1.0 && gate < f64::from(u32::MAX)) {
        return u32::MAX;
    }
    gate as u32
}

/// One layer's share of a prompt unit: what leaves the host union, and what
/// stays. [`split`] clears and fills a caller-held `Split`, so a walk reuses
/// one across layers and units.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Split {
    /// The host experts the card pool admits, hottest first by [`rank_key`]:
    /// persistent for the model's life after the call that moves them.
    pub admit: Vec<u32>,
    /// The host experts the transient ring streams this unit, by [`rank_key`]
    /// after every admit: borrowed card slots, returned at the unit's end.
    pub stream: Vec<u32>,
    /// The columns the host union keeps: the counts of every host expert
    /// neither list took.
    pub host_columns: u64,
}

/// The ring's share of a host set whose admits are already taken: what it
/// streams this unit, and what stays. [`stream_tail`] clears and fills a
/// caller-held `Tail`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tail {
    /// The host experts the transient ring streams this unit, by
    /// [`rank_key`].
    pub stream: Vec<u32>,
    /// The columns the host union keeps: the counts of every host expert the
    /// ring did not take.
    pub host_columns: u64,
}

/// A split the rule refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SplitError {
    /// A constant outside its range.
    Param {
        name: &'static str,
        range: &'static str,
    },
    /// Counts of another length than the layer's experts.
    CountsShape { len: usize, want: usize },
    /// A host list naming an expert outside the layer's experts.
    HostOutOfRange { id: u32, experts: usize },
    /// A host list naming an expert twice.
    HostDuplicate { id: u32 },
    /// Counts that do not sum to whole columns at the router's top-k.
    CountsSum { sum: u64, top_k: usize },
    /// A count above its unit's width: a column routes an expert at most once.
    CountPastUnit { id: u32, count: u32, unit: u64 },
    /// Counts that do not sum to the given unit's columns at the router's
    /// top-k.
    CountsUnit { sum: u64, unit: u64, top_k: usize },
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            SplitError::Param { name, range } => {
                write!(f, "split rule constant {name} must be {range}")
            }
            SplitError::CountsShape { len, want } => write!(
                f,
                "split counts of another length than the layer's experts: {len} of {want}"
            ),
            SplitError::HostOutOfRange { id, experts } => write!(
                f,
                "split host list names expert {id} outside the layer's {experts}"
            ),
            SplitError::HostDuplicate { id } => {
                write!(f, "split host list names expert {id} twice")
            }
            SplitError::CountsSum { sum, top_k } => write!(
                f,
                "split counts sum to {sum}, not whole columns at top-k {top_k}"
            ),
            SplitError::CountPastUnit { id, count, unit } => write!(
                f,
                "split count {count} for expert {id} passes its unit's {unit} columns"
            ),
            SplitError::CountsUnit { sum, unit, top_k } => write!(
                f,
                "split counts sum to {sum}, not {unit} columns at top-k {top_k}"
            ),
        }
    }
}

impl std::error::Error for SplitError {}

/// Split one layer of a prompt unit. `counts` holds one count per expert of
/// the layer (the unit's routed picks), `host` the layer's host-tier ids in
/// any order, `pool` the card slots the call can fill, and `k` the
/// [`Constants`] the load measured. The pool admits the hottest routed host
/// experts, at most `pool` of them by [`rank_key`]; the ring streams the rest
/// the rule sends to the card, only when the unit's width — the counts' sum
/// over the router's top-k — clears [`m_min`]; the union keeps every other
/// host expert, its columns summed into `host_columns`. A `pool` of zero
/// streams nothing and admits nothing.
///
/// The call clears `out` first; a refusal leaves it empty. Refused by name:
/// constants out of range, counts of another length than the layer's experts,
/// counts that do not sum to whole columns or count one expert past the
/// unit's width, and a host list naming an expert outside the layer or twice.
///
/// # Errors
/// Every refusal names its input; see [`SplitError`].
pub fn split(
    counts: &[u32],
    host: &[u32],
    pool: usize,
    k: &Constants,
    out: &mut Split,
) -> Result<(), SplitError> {
    out.admit.clear();
    out.stream.clear();
    out.host_columns = 0;
    check_constants(k)?;
    let sum = check_counts(counts, k)?;
    if !sum.is_multiple_of(k.top_k as u64) {
        return Err(SplitError::CountsSum {
            sum,
            top_k: k.top_k,
        });
    }
    let unit = sum / k.top_k as u64;
    check_unit(counts, unit)?;
    check_host(host, k, &mut out.admit)?;
    // A count of zero names an expert the unit never routes: neither list
    // takes it, and its columns are none.
    out.admit.retain(|&id| counts[id as usize] > 0);
    out.admit
        .sort_unstable_by_key(|&id| rank_key(counts[id as usize], id));
    if pool == 0 {
        out.host_columns = out
            .admit
            .iter()
            .map(|&id| u64::from(counts[id as usize]))
            .sum();
        out.admit.clear();
        return Ok(());
    }
    let admits = pool.min(out.admit.len());
    let mut tail = Tail {
        stream: std::mem::take(&mut out.stream),
        host_columns: 0,
    };
    let tailed = stream_tail(counts, &out.admit[admits..], unit, k, &mut tail);
    out.stream = tail.stream;
    if let Err(err) = tailed {
        out.admit.clear();
        return Err(err);
    }
    out.admit.truncate(admits);
    out.host_columns = tail.host_columns;
    Ok(())
}

/// The ring's half of the rule over a host set whose admits are already
/// taken: `counts` holds one count per expert of the layer over a unit of
/// `m` columns, `host` the host-tier ids left after the admits, in any
/// order, and `k` the [`Constants`] the load measured. The ring streams
/// every routed host expert the rule sends to the card, by [`rank_key`],
/// only when `m` clears [`m_min`]; the union keeps every other one, its
/// columns summed into `host_columns`. Nothing is admitted, and no capacity
/// bounds the ring: its slots are transient.
///
/// The call clears `out` first; a refusal leaves it empty. Refused by name:
/// constants out of range, counts of another length than the layer's
/// experts, counts that do not sum to `m` columns or count one expert past
/// `m`, and a host list naming an expert outside the layer or twice.
///
/// # Errors
/// Every refusal names its input; see [`SplitError`].
pub fn stream_tail(
    counts: &[u32],
    host: &[u32],
    m: u64,
    k: &Constants,
    out: &mut Tail,
) -> Result<(), SplitError> {
    out.stream.clear();
    out.host_columns = 0;
    check_constants(k)?;
    let sum = check_counts(counts, k)?;
    if m.checked_mul(k.top_k as u64) != Some(sum) {
        return Err(SplitError::CountsUnit {
            sum,
            unit: m,
            top_k: k.top_k,
        });
    }
    check_unit(counts, m)?;
    check_host(host, k, &mut out.stream)?;
    let ring = m >= m_min(k);
    let mut kept = 0;
    out.stream.retain(|&id| {
        let count = counts[id as usize];
        let streamed = ring && count > 0 && streams(k, count);
        if !streamed {
            kept += u64::from(count);
        }
        streamed
    });
    out.stream
        .sort_unstable_by_key(|&id| rank_key(counts[id as usize], id));
    out.host_columns = kept;
    Ok(())
}

/// Refuse counts of another length than the layer's experts; their sum.
fn check_counts(counts: &[u32], k: &Constants) -> Result<u64, SplitError> {
    if counts.len() != k.experts {
        return Err(SplitError::CountsShape {
            len: counts.len(),
            want: k.experts,
        });
    }
    Ok(counts.iter().map(|&c| u64::from(c)).sum())
}

/// Refuse a count past the unit's width.
fn check_unit(counts: &[u32], unit: u64) -> Result<(), SplitError> {
    match counts
        .iter()
        .enumerate()
        .find(|&(_, &c)| u64::from(c) > unit)
    {
        Some((id, &count)) => Err(SplitError::CountPastUnit {
            id: u32::try_from(id)
                .expect("an expert id: the constants bound experts by the id type"),
            count,
            unit,
        }),
        None => Ok(()),
    }
}

/// Refuse a host list naming an expert outside the layer or twice; on
/// success `by_id` holds the list sorted by id, on a refusal it is empty.
fn check_host(host: &[u32], k: &Constants, by_id: &mut Vec<u32>) -> Result<(), SplitError> {
    by_id.clear();
    if let Some(&id) = host.iter().find(|&&id| id as usize >= k.experts) {
        return Err(SplitError::HostOutOfRange {
            id,
            experts: k.experts,
        });
    }
    by_id.extend_from_slice(host);
    by_id.sort_unstable();
    if let Some(id) = by_id
        .windows(2)
        .find_map(|pair| (pair[0] == pair[1]).then_some(pair[0]))
    {
        by_id.clear();
        return Err(SplitError::HostDuplicate { id });
    }
    Ok(())
}

/// Refuse the constants the derivations cannot read: every rate and cost
/// finite, the two that divide positive, the expert's bytes present, and the
/// router's shape whole.
fn check_constants(k: &Constants) -> Result<(), SplitError> {
    let param = |name, range| Err(SplitError::Param { name, range });
    if !(k.lane_b_per_us.is_finite() && k.lane_b_per_us > 0.0) {
        return param("lane_b_per_us", "finite, above 0");
    }
    if !(k.host_us_per_col.is_finite() && k.host_us_per_col > 0.0) {
        return param("host_us_per_col", "finite, above 0");
    }
    if !(k.host_us_fixed.is_finite() && k.host_us_fixed >= 0.0) {
        return param("host_us_fixed", "finite, 0 or more");
    }
    if !(k.card_us_fixed.is_finite() && k.card_us_fixed >= 0.0) {
        return param("card_us_fixed", "finite, 0 or more");
    }
    if !(k.card_us_per_col.is_finite() && k.card_us_per_col >= 0.0) {
        return param("card_us_per_col", "finite, 0 or more");
    }
    if !(k.union_burst_share.is_finite() && k.union_burst_share >= 0.0 && k.union_burst_share < 1.0)
    {
        return param("union_burst_share", "finite, 0 or more, under 1");
    }
    if !(k.card_tail_us.is_finite() && k.card_tail_us >= 0.0) {
        return param("card_tail_us", "finite, 0 or more");
    }
    if k.expert_b == 0 {
        return param("expert_b", "1 or more");
    }
    if k.experts == 0 {
        return param("experts", "1 or more");
    }
    if k.experts > u32::MAX as usize {
        return param("experts", "at most the id type holds");
    }
    if k.top_k == 0 || k.top_k > k.experts {
        return param("top_k", "1 or more, at most experts");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constants whose derivations land on exact numbers: an expert of
    /// 50000 B over a lane of 1000 B/µs copies in 50, the card's fixed cost
    /// adds 10, and the host's 10.5 a column against the card's 0.5 puts
    /// m* = (50 + 10)/(10.5 − 0.5) = 6; eight experts routed two a column put
    /// m_min at 6 × 8/2 = 24. W = 2 is under b/r + f = 60. No burst share:
    /// the beside-union rate is the lane's own, so every number stays exact.
    fn plain() -> Constants {
        Constants {
            lane_b_per_us: 1000.0,
            host_us_per_col: 10.5,
            host_us_fixed: 2.0,
            card_us_fixed: 10.0,
            card_us_per_col: 0.5,
            union_burst_share: 0.0,
            card_tail_us: 0.0,
            expert_b: 50000,
            experts: 8,
            top_k: 2,
        }
    }

    /// The A6000 plan (a) calibration: 512 routed experts, ten picks a
    /// column, the lane's measured rate, the recalibrated tile-path host
    /// union (W 19.95, c 3.67), the common layer kind's expert bytes, the
    /// card's per-expert costs (13 + 0.16 a column), the calibration shape's
    /// burst share and a flat route tail of 4690, which the tests that set
    /// their own tail overwrite.
    /// beside = 21190 × (1 − 0.381) = 13116.6,
    /// m* = (3072000/13116.6 + 13)/(3.67 − 0.16) = 247.207/3.51 = 70.429,
    /// m_min = ceil(70.429 × 51.2) = ceil(3605.98) = 3606.
    fn a6000() -> Constants {
        Constants {
            lane_b_per_us: 21190.0,
            host_us_per_col: 3.67,
            host_us_fixed: 19.95,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            union_burst_share: 0.381,
            card_tail_us: 4690.0,
            expert_b: 3072000,
            experts: 512,
            top_k: 10,
        }
    }

    /// The 3090 plan (a) calibration: the A6000's link, union and share, the
    /// card's costs at the design's ×1.25 (16.25 + 0.2 a column).
    /// m* = (234.207 + 16.25)/(3.67 − 0.2) = 250.457/3.47 = 72.178,
    /// m_min = ceil(72.178 × 51.2) = ceil(3695.5) = 3696.
    fn a3090() -> Constants {
        Constants {
            card_us_fixed: 16.25,
            card_us_per_col: 0.2,
            ..a6000()
        }
    }

    /// The 5090 calibration: the same expert bytes and routing at the A6000's
    /// share, the pageable staging rate that box holds, its fitted host union
    /// (W 38.4, c 6.2) and the card's costs at the design's ×0.6
    /// (7.8 + 0.096 a column).
    /// beside = 21200 × 0.619 = 13122.8,
    /// m* = (3072000/13122.8 + 7.8)/(6.2 − 0.096) = 241.897/6.104 = 39.629,
    /// m_min = ceil(39.629 × 51.2) = ceil(2029.01) = 2030.
    fn their_5090() -> Constants {
        Constants {
            lane_b_per_us: 21200.0,
            host_us_per_col: 6.2,
            host_us_fixed: 38.4,
            card_us_fixed: 7.8,
            card_us_per_col: 0.096,
            expert_b: 3072000,
            experts: 512,
            top_k: 10,
            ..a6000()
        }
    }

    /// A count at or below the floor never streams, whatever the pool: the
    /// rule's inequality is strict, so a count exactly at m* stays with the
    /// union (mutants: the floor ignored, the inequality inclusive).
    #[test]
    fn a_count_at_or_below_the_floor_never_streams() {
        let k = plain();
        // 110 picks over two a column: a 55-column unit, past m_min 24, so
        // the ring is lit and only the floor speaks.
        let counts = [10, 10, 40, 30, 7, 6, 5, 2];
        let mut out = Split::default();
        split(&counts, &[6, 2, 5, 3, 4], 1, &k, &mut out).unwrap();
        assert_eq!(out.admit, [2], "the hottest host expert fills the pool");
        assert_eq!(out.stream, [3, 4], "the counts past the floor of 6 stream");
        assert_eq!(
            out.host_columns,
            6 + 5,
            "the count at 6 and the one under stay"
        );
    }

    /// [`m_star`] is the rule's own crossover: while W is at most b/r + f, a
    /// count streams exactly when it is past m*, on every calibration
    /// (mutants: the card's fixed cost subtracted, its per-column cost
    /// dropped).
    #[test]
    fn the_floor_is_the_rules_crossover() {
        for k in [plain(), a6000(), a3090(), their_5090()] {
            let floor = m_star(&k);
            for count in 0..=4096 {
                assert_eq!(
                    streams(&k, count),
                    f64::from(count) > floor,
                    "count {count} against m* {floor}"
                );
            }
        }
    }

    /// The union's fixed cost is the rule's too: when W is above b/r + f, a
    /// count under m* costs the union more than the lane and the card, and
    /// streams (mutant: W dropped from the host side).
    #[test]
    fn the_union_floor_streams_a_count_under_the_column_floor() {
        let mut k = plain();
        k.host_us_fixed = 100.0; // above b/r + f + k·m = 60 + 0.5·m under 80 columns
        let counts = [10, 10, 40, 30, 7, 6, 5, 2];
        let mut out = Split::default();
        split(&counts, &[7, 6, 2, 5, 3, 4], 1, &k, &mut out).unwrap();
        assert_eq!(out.admit, [2]);
        assert_eq!(out.stream, [3, 4, 5, 6, 7]);
        assert_eq!(out.host_columns, 0);
    }

    /// A card column that costs at least the host's never pays back the copy:
    /// the floor is infinite, m_min saturates and nothing streams however wide
    /// the unit (mutant: the guard dropped, a negative floor).
    #[test]
    fn a_card_column_at_or_past_the_hosts_streams_nothing() {
        let mut k = plain();
        k.card_us_per_col = 11.0;
        assert_eq!(m_star(&k), f64::INFINITY);
        assert_eq!(m_min(&k), u64::MAX);
        let counts = [10, 10, 40, 30, 7, 6, 5, 2];
        let mut out = Split::default();
        split(&counts, &[6, 2, 5, 3, 4], 1, &k, &mut out).unwrap();
        assert_eq!(out.admit, [2]);
        assert!(out.stream.is_empty());
        assert_eq!(out.host_columns, 30 + 7 + 6 + 5);
    }

    /// A capacity of zero streams nothing and admits nothing: the union keeps
    /// every host expert it serves (mutant: the ring lit with no pool).
    #[test]
    fn a_capacity_of_zero_streams_and_admits_nothing() {
        let k = plain();
        let counts = [10, 10, 40, 30, 7, 6, 5, 2];
        let mut out = Split::default();
        split(&counts, &[6, 2, 5, 3, 4], 0, &k, &mut out).unwrap();
        assert!(out.admit.is_empty());
        assert!(out.stream.is_empty());
        assert_eq!(out.host_columns, 40 + 30 + 7 + 6 + 5);
    }

    /// Both lists fill in the one ranking, count descending then id ascending,
    /// whatever order the host list arrives in (mutant: the ranking by id
    /// alone).
    #[test]
    fn both_lists_rank_count_descending_then_id_ascending() {
        assert!(rank_key(9, 5) < rank_key(7, 0));
        assert!(rank_key(7, 3) < rank_key(7, 9));
        let k = plain();
        // A 30-column unit, past m_min 24: two admits, then the ring's tail
        // in the same order, the count of 6 at the floor kept.
        let counts = [7, 9, 0, 7, 9, 7, 6, 15];
        let mut out = Split::default();
        split(&counts, &[6, 5, 4, 3, 2, 1, 0], 2, &k, &mut out).unwrap();
        assert_eq!(out.admit, [1, 4]);
        assert_eq!(out.stream, [0, 3, 5]);
        assert_eq!(out.host_columns, 6);
    }

    /// Bad input is refused by name and leaves the split empty: constants the
    /// derivations cannot read, counts of another length, counts that do not
    /// sum to whole columns or pass the unit, and a host id outside the layer
    /// or named twice, a never-routed one too (mutant: any check dropped).
    #[test]
    fn bad_input_is_refused_by_name() {
        let k = plain();
        let counts = [10, 10, 40, 30, 7, 6, 5, 2];
        let mut out = Split {
            admit: vec![7, 7],
            stream: vec![7],
            host_columns: 99,
        };
        let mut refuses = |k: &Constants, counts: &[u32], host: &[u32], want: SplitError| {
            assert_eq!(split(counts, host, 2, k, &mut out), Err(want));
            assert!(
                out.admit.is_empty() && out.stream.is_empty() && out.host_columns == 0,
                "a refusal leaves the split empty"
            );
        };
        let param = |name, range| SplitError::Param { name, range };
        for (bad, want) in [
            (
                Constants {
                    lane_b_per_us: f64::NAN,
                    ..k
                },
                param("lane_b_per_us", "finite, above 0"),
            ),
            (
                Constants {
                    lane_b_per_us: 0.0,
                    ..k
                },
                param("lane_b_per_us", "finite, above 0"),
            ),
            (
                Constants {
                    host_us_per_col: 0.0,
                    ..k
                },
                param("host_us_per_col", "finite, above 0"),
            ),
            (
                Constants {
                    host_us_fixed: f64::NAN,
                    ..k
                },
                param("host_us_fixed", "finite, 0 or more"),
            ),
            (
                Constants {
                    card_us_fixed: f64::INFINITY,
                    ..k
                },
                param("card_us_fixed", "finite, 0 or more"),
            ),
            (
                Constants {
                    card_us_per_col: -0.1,
                    ..k
                },
                param("card_us_per_col", "finite, 0 or more"),
            ),
            (
                Constants { expert_b: 0, ..k },
                param("expert_b", "1 or more"),
            ),
            (Constants { experts: 0, ..k }, param("experts", "1 or more")),
            (
                Constants { top_k: 0, ..k },
                param("top_k", "1 or more, at most experts"),
            ),
            (
                Constants { top_k: 9, ..k },
                param("top_k", "1 or more, at most experts"),
            ),
        ] {
            refuses(&bad, &counts, &[2], want);
        }
        refuses(
            &k,
            &counts[..7],
            &[2],
            SplitError::CountsShape { len: 7, want: 8 },
        );
        refuses(
            &k,
            &[10, 10, 40, 30, 7, 6, 5, 1],
            &[2],
            SplitError::CountsSum { sum: 109, top_k: 2 },
        );
        refuses(
            &k,
            &[4, 4, 4, 4, 60, 3, 2, 1],
            &[2],
            SplitError::CountPastUnit {
                id: 4,
                count: 60,
                unit: 41,
            },
        );
        refuses(
            &k,
            &counts,
            &[8],
            SplitError::HostOutOfRange { id: 8, experts: 8 },
        );
        refuses(&k, &counts, &[3, 3], SplitError::HostDuplicate { id: 3 });
        let unrouted = [10, 10, 40, 30, 7, 0, 5, 8];
        refuses(
            &k,
            &unrouted,
            &[2, 5, 5],
            SplitError::HostDuplicate { id: 5 },
        );
    }

    /// The A6000 shape: 512 experts routed ten a column, 301 on the card and
    /// 211 on the host, a pool of 301 − 148 = 153 (the card's experts past
    /// `mid-p148-s1`'s pinned seed). At 512 columns every expert carries
    /// 512 × 10/512 = 10 and the unit is under m_min 3606, so the pool admits
    /// alone: ids 301..454, the other 58 keep 58 × 10 = 580 columns on the
    /// union. A 512-column unit that routes every host expert 24 times, under
    /// the floor of 70.429 (211 × 24 = 5064 picks, 56 on the card make
    /// 5120 = 512 × 10), is still under m_min: admit alone, 58 × 24 = 1392
    /// columns kept. At 4096 each carries 80, past the floor, and the unit
    /// clears m_min: the same admits, the ring takes the 58 past them
    /// (mutants: the unit gate dropped, m_min without the router's top-k).
    #[test]
    fn the_a6000_shape_admits_alone_at_512_and_streams_its_tail_at_4096() {
        let k = a6000();
        let host: Vec<u32> = (301..512).collect();
        let mut out = Split::default();
        split(&[10; 512], &host, 153, &k, &mut out).unwrap();
        assert_eq!(out.admit, (301..454).collect::<Vec<_>>());
        assert_eq!(out.stream, [], "512 columns is under m_min: admit alone");
        assert_eq!(out.host_columns, 580);
        let mut hot = [0; 512];
        hot[..56].fill(1);
        hot[301..].fill(24);
        split(&hot, &host, 153, &k, &mut out).unwrap();
        assert_eq!(out.admit, (301..454).collect::<Vec<_>>());
        assert_eq!(out.stream, [], "past the floor, still under m_min");
        assert_eq!(out.host_columns, 1392);
        split(&[80; 512], &host, 153, &k, &mut out).unwrap();
        assert_eq!(out.admit, (301..454).collect::<Vec<_>>());
        assert_eq!(out.stream, (454..512).collect::<Vec<_>>());
        assert_eq!(out.host_columns, 0);
    }

    /// The 5090 shape: the same 512 experts and routing, 5482/48 → 114 on the
    /// card and 398 on the host, a pool of its 2746 churn slots / 48 → 57. At
    /// 4096 columns every expert carries 80, past the floor of 39.629, and the
    /// unit clears m_min 2030: the pool admits ids 114..171 and the ring
    /// streams all 341 host experts past them, 341/398 of the host tier; the
    /// union keeps none (mutants: the rule's inequality reversed, m_min
    /// without the router's top-k).
    #[test]
    fn the_5090_shape_streams_nearly_all_at_4096() {
        let k = their_5090();
        let host: Vec<u32> = (114..512).collect();
        let mut out = Split::default();
        split(&[80; 512], &host, 57, &k, &mut out).unwrap();
        assert_eq!(out.admit, (114..171).collect::<Vec<_>>());
        assert_eq!(out.stream, (171..512).collect::<Vec<_>>());
        assert_eq!(out.host_columns, 0);
    }

    /// m* and m_min on all three machines' calibrations, each derived in its
    /// constants' comment (mutants: the card's fixed cost subtracted, the
    /// mean without the choice set's spread).
    ///
    /// PIN(2026-10-08): m* 23.096, 23.709, 25.017 and m_min 1183, 1214, 1281
    /// before the rule priced its copies beside the union; now m* 70.429,
    /// 72.178, 39.629 and m_min 3606, 3696, 2030, the values the constants'
    /// comments derive at the beside-union rate (A6000: (3072000/13116.6 +
    /// 13)/(3.67 - 0.16) = 70.429).
    #[test]
    fn m_min_on_all_three_machines() {
        assert!((m_star(&a6000()) - 70.429).abs() < 0.001);
        assert!((m_star(&a3090()) - 72.178).abs() < 0.001);
        assert!((m_star(&their_5090()) - 39.629).abs() < 0.001);
        for (name, k, want) in [
            ("A6000", a6000(), 3606),
            ("3090", a3090(), 3696),
            ("5090", their_5090(), 2030),
        ] {
            assert_eq!(m_min(&k), want, "{name}");
        }
    }

    /// The ring's half alone streams only from the host set it is given: an
    /// empty one streams nothing and keeps no columns, however wide the unit
    /// (mutant: the tail read over every expert of the layer).
    #[test]
    fn stream_tail_with_an_empty_host_set_streams_nothing() {
        let k = plain();
        let mut out = Tail::default();
        stream_tail(&[10, 10, 40, 30, 7, 6, 5, 2], &[], 55, &k, &mut out).unwrap();
        assert_eq!(out, Tail::default());
    }

    /// Below m_min the ring's half streams nothing: a 20-column unit, under
    /// m_min 24, whose host experts all carry 7, past the floor of 6, keeps
    /// all 5 × 7 + 5 = 40 of their columns on the union (mutant: the unit
    /// gate dropped).
    #[test]
    fn stream_tail_below_m_min_streams_nothing() {
        let k = plain();
        let counts = [7, 7, 7, 7, 7, 0, 5, 0];
        let mut out = Tail::default();
        stream_tail(&counts, &[6, 4, 3, 2, 1, 0], 20, &k, &mut out).unwrap();
        assert!(out.stream.is_empty());
        assert_eq!(out.host_columns, 40);
    }

    /// The ring's half over the host set a split's admits leave is that
    /// split's tail, list and columns, in the ranking's order whatever order
    /// the set arrives in. The first case's tail ranks 2 (12), 1 (9), 0 (8),
    /// 5 (7) past the floor of 6 and keeps 6's 4 columns: not the id order
    /// (mutants: split handing the admits to the tail too, the tail left in
    /// id order).
    #[test]
    fn stream_tail_is_splits_tail_on_the_post_admit_host_set() {
        let hot: Vec<u32> = (0..512)
            .map(|id| match id {
                0..56 => 1,
                301.. => 24,
                _ => 0,
            })
            .collect();
        let cases: [(Constants, Vec<u32>, Vec<u32>, usize); 5] = [
            (
                plain(),
                vec![8, 9, 12, 0, 20, 7, 4, 0],
                vec![6, 5, 4, 2, 1, 0],
                1,
            ),
            (
                plain(),
                vec![10, 10, 40, 30, 7, 6, 5, 2],
                vec![6, 2, 5, 3, 4],
                1,
            ),
            (a6000(), vec![80; 512], (301..512).rev().collect(), 153),
            (a6000(), hot, (301..512).collect(), 153),
            (their_5090(), vec![80; 512], (114..512).rev().collect(), 57),
        ];
        for (k, counts, host, pool) in cases {
            let mut whole = Split::default();
            split(&counts, &host, pool, &k, &mut whole).unwrap();
            let left: Vec<u32> = host
                .iter()
                .copied()
                .filter(|id| !whole.admit.contains(id))
                .collect();
            let unit = counts.iter().map(|&c| u64::from(c)).sum::<u64>() / k.top_k as u64;
            let mut tail = Tail::default();
            stream_tail(&counts, &left, unit, &k, &mut tail).unwrap();
            assert_eq!(tail.stream, whole.stream, "unit {unit}");
            assert_eq!(tail.host_columns, whole.host_columns, "unit {unit}");
        }
        let mut first = Tail::default();
        stream_tail(
            &[8, 9, 12, 0, 20, 7, 4, 0],
            &[6, 5, 2, 1, 0],
            30,
            &plain(),
            &mut first,
        )
        .unwrap();
        assert_eq!(first.stream, [2, 1, 0, 5]);
        assert_eq!(first.host_columns, 4);
    }

    /// The ring's half refuses what [`split`] refuses, by name, and a count
    /// sum that is not the given unit's; a refusal leaves the tail empty
    /// (mutant: any check dropped).
    #[test]
    fn stream_tail_refuses_by_name() {
        let k = plain();
        let counts = [10, 10, 40, 30, 7, 6, 5, 2];
        let mut out = Tail {
            stream: vec![7],
            host_columns: 99,
        };
        let mut refuses =
            |k: &Constants, counts: &[u32], host: &[u32], m: u64, want: SplitError| {
                assert_eq!(stream_tail(counts, host, m, k, &mut out), Err(want));
                assert!(
                    out.stream.is_empty() && out.host_columns == 0,
                    "a refusal leaves the tail empty"
                );
            };
        refuses(
            &Constants {
                host_us_fixed: f64::NAN,
                ..k
            },
            &counts,
            &[2],
            55,
            SplitError::Param {
                name: "host_us_fixed",
                range: "finite, 0 or more",
            },
        );
        refuses(
            &k,
            &counts[..7],
            &[2],
            55,
            SplitError::CountsShape { len: 7, want: 8 },
        );
        refuses(
            &k,
            &counts,
            &[2],
            54,
            SplitError::CountsUnit {
                sum: 110,
                unit: 54,
                top_k: 2,
            },
        );
        refuses(
            &k,
            &counts,
            &[2],
            u64::MAX,
            SplitError::CountsUnit {
                sum: 110,
                unit: u64::MAX,
                top_k: 2,
            },
        );
        refuses(
            &k,
            &[4, 4, 4, 4, 60, 3, 2, 1],
            &[2],
            41,
            SplitError::CountPastUnit {
                id: 4,
                count: 60,
                unit: 41,
            },
        );
        refuses(
            &k,
            &counts,
            &[8],
            55,
            SplitError::HostOutOfRange { id: 8, experts: 8 },
        );
        refuses(
            &k,
            &counts,
            &[3, 3],
            55,
            SplitError::HostDuplicate { id: 3 },
        );
        let unrouted = [10, 10, 40, 30, 7, 0, 5, 8];
        refuses(
            &k,
            &unrouted,
            &[2, 5, 5],
            55,
            SplitError::HostDuplicate { id: 5 },
        );
    }

    /// Constants of 8 experts routed two a column whose first admit's copy
    /// is 1000 (a 10^6 B expert over a lane of 1000 B a microsecond), no burst share and
    /// no tail, for the gate and the unit pricing: W 50, c 10.
    fn gate_shape() -> Constants {
        Constants {
            lane_b_per_us: 1000.0,
            host_us_per_col: 10.0,
            host_us_fixed: 50.0,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            union_burst_share: 0.0,
            card_tail_us: 0.0,
            expert_b: 1_000_000,
            experts: 8,
            top_k: 2,
        }
    }

    /// The lever-2 log's layer-1 shape (4137 columns over 212 listed host
    /// experts) as a counts vector: the admits' worth uniform from the
    /// today-floor 25 to the layer's hottest count 52, the union's listed
    /// experts at the layer's mean count. For the share and the walk tests,
    /// [derived] from that log's `stat prompt lb` row.
    fn layer1_shape() -> (Vec<u32>, Vec<u32>) {
        let mut counts = vec![0u32; 512];
        for (i, c) in counts[..122].iter_mut().enumerate() {
            *c = u32::try_from(25 + (i * 27) / 122).expect("a count");
        }
        for c in counts[122..334].iter_mut() {
            *c = 27;
        }
        let host: Vec<u32> = (0..334).collect();
        (counts, host)
    }

    /// The union's burst share at the calibration shape sits inside its
    /// measured band, and the beside-union rate is the lane's less it: at
    /// layer 1, W x 212 over W x 212 + c x 4137 (mutant: the share without
    /// the per-column term, which saturates it to 1).
    #[test]
    fn the_burst_share_sits_in_its_band_and_cuts_the_lane() {
        let k = a6000();
        let s = union_burst_share(4137, 212, &k);
        assert!((0.2..0.6).contains(&s), "the share at layer 1 is {s}");
        let cold = union_burst_share(0, 0, &k);
        assert_eq!(cold, 0.0, "nothing listed: no weight reads to lose to");
        let mut unit = k;
        unit.union_burst_share = s;
        assert!((beside_b_per_us(&unit) - k.lane_b_per_us * (1.0 - s)).abs() < 1e-9);
    }

    /// The stream floor prices its copy beside the union, where it runs: at
    /// the recalibrated costs and share, m* = 70.429, past the alone-rate
    /// floor the lane's own rate gives, 46.3 (mutant: the floor at the
    /// lane's rate alone).
    #[test]
    fn the_stream_floor_prices_its_copy_beside_the_union() {
        let k = a6000();
        assert!((m_star(&k) - 70.429).abs() < 0.001);
        let alone = k.expert_b as f64 / k.lane_b_per_us + k.card_us_fixed;
        assert!(
            m_star(&k) > alone / (k.host_us_per_col - k.card_us_per_col),
            "the beside-union rate must price dearer than the lane's own"
        );
    }

    /// The walk admits past the balance the beside-union rate stops at: the
    /// same layer-1 shape at two shares, the lane's own (a copy alone, the
    /// mutant the rule must not price at) and the calibration's — the dearer
    /// copies cross the union's saving sooner, so the floor the beside rate
    /// returns is the higher and admits the fewer.
    #[test]
    fn the_walk_stops_sooner_at_the_beside_union_rate() {
        let (counts, host) = layer1_shape();
        let mut alone = a6000();
        alone.union_burst_share = 0.0;
        let beside = a6000();
        let fast = admit_walk(&counts, &host, &alone, 4, ReturnCost::Hidden).unwrap();
        let slow = admit_walk(&counts, &host, &beside, 4, ReturnCost::Hidden).unwrap();
        assert!(
            fast.floor < slow.floor,
            "the alone rate admits past the balance: {} against {}",
            fast.floor,
            slow.floor
        );
        assert!(fast.backlog >= slow.backlog);
    }

    /// The card's tail sits at the head of the walk's card chain, and the
    /// backlog bound never passes the admits there are: the same four hot
    /// experts — each saving the union 300, each copy 100 — admit three with
    /// no tail (the chain from 0) and two with a 200 tail, the bound riding
    /// to the admits both times. A walk that drops the tail admits the
    /// third expert the tail's weight refused (mutant: the chain from 0
    /// whatever the tail).
    #[test]
    fn the_backlog_bound_fits_the_unions_shadow_past_the_tail() {
        let k = Constants {
            lane_b_per_us: 1000.0,
            host_us_per_col: 10.0,
            host_us_fixed: 50.0,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            union_burst_share: 0.0,
            card_tail_us: 0.0,
            expert_b: 100000,
            experts: 8,
            top_k: 2,
        };
        let counts = [30, 30, 30, 30, 0, 0, 0, 0];
        let host: Vec<u32> = (0..8).collect();
        let none = admit_walk(&counts, &host, &k, 1, ReturnCost::Hidden).unwrap();
        assert_eq!(none.floor, 30);
        assert_eq!(none.backlog, 3, "the bound rides to the admits");
        let mut tailed = k;
        tailed.card_tail_us = 200.0;
        let some = admit_walk(&counts, &host, &tailed, 1, ReturnCost::Hidden).unwrap();
        assert_eq!(some.floor, 30);
        assert_eq!(some.backlog, 2, "the tail's weight refuses the third admit");
        // The shadow's own tail term: at a stop the union the walk leaves
        // already covers the admits' copies (the chain crossed under it), so
        // the bound is the admits — the box's stamps read whether it held.
        assert!(some.backlog <= 2 && none.backlog <= 3);
    }

    /// A share at or past the whole window, or one no number reads, is
    /// refused by name: no copy is priced at a rate the union cannot leave
    /// it (mutant: the range check dropped).
    #[test]
    fn a_share_the_union_cannot_leave_is_refused_by_name() {
        let (counts, host) = layer1_shape();
        for bad in [1.0, 1.4, f64::NAN, f64::INFINITY, -0.1] {
            let mut k = a6000();
            k.union_burst_share = bad;
            assert_eq!(
                admit_walk(&counts, &host, &k, 4, ReturnCost::Hidden),
                Err(SplitError::Param {
                    name: "union_burst_share",
                    range: "finite, 0 or more, under 1"
                }),
                "the share {bad}"
            );
        }
    }

    /// The fit's form: an expert over W/c costs the union its columns alone,
    /// one under it costs W — the max form's two branches. Two hot experts
    /// at 40 columns (over W/c = 5) and six cold at 4 (under it): the colds'
    /// saving is W each, too little for the card chain's third copy, so the
    /// walk admits the two hot alone; a sum form (W added to every expert's
    /// columns too) inflates the colds' saving and buys a third admit
    /// (mutant: the host cost W + c·m).
    #[test]
    fn the_walk_prices_the_max_form_the_fit_selected() {
        let k = Constants {
            lane_b_per_us: 1000.0,
            host_us_per_col: 10.0,
            host_us_fixed: 50.0,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            union_burst_share: 0.0,
            card_tail_us: 0.0,
            expert_b: 100000,
            experts: 8,
            top_k: 2,
        };
        let counts = [40, 40, 4, 4, 4, 4, 4, 4];
        let host: Vec<u32> = (0..8).collect();
        let plan = admit_walk(&counts, &host, &k, 1, ReturnCost::Hidden).unwrap();
        // A copy is 100, the union 2 x 400 + 6 x 50 = 1100: the third
        // admit's 300 passes 250 only under the sum form's 90-a-cold.
        assert_eq!(plan.floor, 40, "the two hottest alone");
    }

    /// An admit whose return lands at the call's end costs the walk two
    /// copies: its own beside the union and its return's at the lane's rate
    /// alone. Five host experts of 1000, 800, 600, 400 and 200 against a copy
    /// of 120: the hidden walk admits three (360 <= 600, then 480 > 200), the
    /// exposed one two (720 > 600). At a burst share of 0.5 the copy beside
    /// the union is 200 and the return alone 100: one admit of 1000 against a
    /// union of 1350 fits at 300 <= 350, where a return priced beside the
    /// union (400) would admit nothing (mutants: the exposed return priced as
    /// a hidden one, today's price; the return at the beside-union rate).
    #[test]
    fn an_exposed_return_prices_an_admit_at_two_copies() {
        let k = Constants {
            lane_b_per_us: 1000.0,
            host_us_per_col: 10.0,
            host_us_fixed: 50.0,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            union_burst_share: 0.0,
            card_tail_us: 0.0,
            expert_b: 120000,
            experts: 8,
            top_k: 2,
        };
        let counts = [100, 80, 60, 40, 20, 0, 0, 0];
        let host: Vec<u32> = (0..8).collect();
        let hidden = admit_walk(&counts, &host, &k, 1, ReturnCost::Hidden).unwrap();
        let exposed = admit_walk(&counts, &host, &k, 1, ReturnCost::Exposed).unwrap();
        assert_eq!(hidden.floor, 60, "three admits when the return is hidden");
        assert_eq!(exposed.floor, 80, "two when each admit carries its return");
        assert_eq!((hidden.backlog, exposed.backlog), (3, 2));
        let shared = Constants {
            union_burst_share: 0.5,
            expert_b: 100000,
            ..k
        };
        let counts = [100, 35, 0, 0, 0, 0, 0, 0];
        let one = admit_walk(&counts, &host, &shared, 1, ReturnCost::Exposed).unwrap();
        assert_eq!(one.floor, 100, "the return priced at the lane's rate alone");
    }

    /// The walk's gate is the first admit's copy over a pick's worth of the
    /// union, with no tail: at the 8-expert shape's copy of 1000 (a
    /// 10^6 B expert over a lane of 1000 B a microsecond) and (50 + 10) × 2 = 120 a
    /// column the gate is ⌈1000/120⌉ = 9, whatever tail the constants carry
    /// (mutant: the gate reading the tail, which 4690 would lift to 48).
    ///
    /// PIN(2026-10-08): the gate pinned 21 at the family's flat 4690 µs tail
    /// ((234.2 + 4690)/(19.95 + 3.67) × 10); the tail is now a level per
    /// unit width the walk prices from the unit's counts, which a gate before
    /// the counts cannot read, so the gate is derived at the tail's least
    /// value, 0, and is 1 at the A6000 shape.
    #[test]
    fn the_walks_gate_is_the_copy_over_a_picks_union() {
        let mut k = gate_shape();
        assert_eq!(walk_gate(&k), 9);
        k.card_tail_us = 4690.0;
        assert_eq!(walk_gate(&k), 9, "the tail is not an input of the gate");
        let a = a6000();
        // 234.2 / ((19.95 + 3.67) x 10) = 0.99 -> 1.
        assert_eq!(walk_gate(&a), 1);
    }

    /// The gate never skips a unit the walk would admit from: over every
    /// unit width under it, a thousand units each at random picks over 512 experts and a
    /// random host set admit nothing at the worst tail (none), and over the
    /// widths past it some unit admits (the gate is not vacuous). A walk gate
    /// that read the 4690 tail would skip the widths 9..48 these units
    /// admit at (mutant: the gate with a tail).
    #[test]
    fn the_walks_gate_never_skips_a_unit_the_walk_admits_from() {
        let k = Constants {
            experts: 512,
            ..gate_shape()
        };
        let gate = walk_gate(&k);
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move |n: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % n
        };
        let mut admitted_past = 0;
        for m in 1..=3 * gate as usize {
            for _ in 0..1000 {
                let mut counts = vec![0u32; 512];
                for _ in 0..m {
                    let a = next(512) as usize;
                    let b = (a + 1 + next(511) as usize) % 512;
                    counts[a] += 1;
                    counts[b] += 1;
                }
                let host: Vec<u32> = (0..512).filter(|_| next(4) != 0).collect();
                let mut unit = unit_constants(&k, 0.0, &counts, &host).unwrap();
                // The gate's copy is at no share, the walk's best case: a
                // unit's own share only slows the copy.
                unit.union_burst_share = 0.0;
                let plan = admit_walk(&counts, &host, &unit, 1, ReturnCost::Hidden).unwrap();
                let admits = plan.floor != u32::MAX;
                if m < gate as usize {
                    assert!(
                        !admits,
                        "a unit of {m} columns admits under the gate {gate}"
                    );
                } else {
                    admitted_past += usize::from(admits);
                }
            }
        }
        assert!(
            admitted_past > 0,
            "no unit past the gate admits: a vacuous gate"
        );
    }

    /// A unit's tail is the rate times its card picks and its share the
    /// union's own: of 40 routed picks, 4 + 4 sit on two listed host experts
    /// (a third host expert is unrouted) and 12 + 20 on the card, so 32 card
    /// picks at 0.5 each make 16 (mutant: the tail from the host's columns,
    /// 8 picks, not the card's).
    #[test]
    fn a_units_tail_is_the_rate_times_its_card_picks() {
        let k = Constants {
            card_tail_us: 7.0,
            union_burst_share: 0.9,
            ..gate_shape()
        };
        let counts = [4, 4, 0, 0, 12, 20, 0, 0];
        let host = [0, 1, 2];
        let unit = unit_constants(&k, 0.5, &counts, &host).unwrap();
        assert_eq!(unit.card_tail_us, 16.0, "32 card picks at 0.5");
        // W x 2 listed over W x 2 + c x 8 columns = 100/180.
        assert!((unit.union_burst_share - 100.0 / 180.0).abs() < 1e-12);
        assert_eq!(unit.lane_b_per_us, k.lane_b_per_us);
        let none = unit_constants(&k, 0.0, &counts, &host).unwrap();
        assert_eq!(none.card_tail_us, 0.0, "a family with no rate has no tail");
    }

    /// Every input the pricing cannot read is refused by name (mutants: the
    /// host bound unchecked, the subtraction unchecked, the rate unchecked).
    #[test]
    fn a_units_pricing_refuses_by_name() {
        let k = gate_shape();
        let counts = [4, 4, 0, 0, 12, 20, 0, 0];
        assert_eq!(
            unit_constants(&k, 0.5, &counts, &[0, 8]),
            Err(SplitError::HostOutOfRange { id: 8, experts: 8 })
        );
        assert_eq!(
            unit_constants(&k, 0.5, &counts[..4], &[0]),
            Err(SplitError::CountsShape { len: 4, want: 8 })
        );
        let twice = unit_constants(&k, 0.5, &[40, 0, 0, 0, 0, 0, 0, 0], &[0, 0]);
        assert_eq!(
            twice,
            Err(SplitError::Param {
                name: "host",
                range: "each expert once"
            })
        );
        for bad in [-0.5, f64::NAN, f64::INFINITY] {
            assert_eq!(
                unit_constants(&k, bad, &counts, &[0]),
                Err(SplitError::Param {
                    name: "card_tail_us_per_pick",
                    range: "finite, 0 or more"
                }),
                "the rate {bad}"
            );
        }
    }

    /// The 512-column unit of the A6000 plan (a) Q4 file, as the
    /// walk sees it before a pick: 130 listed host experts, the first 20 the
    /// admits' worth (110 down to 50 columns), the next 30 under them (50
    /// down to 6), the rest 5 each, 2863 columns on the host; the 382 card
    /// experts take the other 2257 of the unit's 5120 picks. [derived] from
    /// the 512-column calls' per-layer medians (129.5 listed host experts,
    /// 1581 kept columns, the admits' columns past it); a stand, not a
    /// record.
    fn unit512() -> (Vec<u32>, Vec<u32>) {
        let mut counts = vec![0u32; 512];
        for (i, c) in counts[..20].iter_mut().enumerate() {
            *c = u32::try_from(110 - (i * 60) / 19).expect("a count");
        }
        for (i, c) in counts[20..50].iter_mut().enumerate() {
            *c = u32::try_from(50 - (i * 44) / 29).expect("a count");
        }
        counts[50..130].fill(5);
        let host_cols: u32 = counts[..130].iter().sum();
        let card = 5120 - host_cols;
        for (j, c) in counts[130..].iter_mut().enumerate() {
            *c = card / 382 + u32::from((j as u32) < card % 382);
        }
        assert_eq!(counts.iter().sum::<u32>(), 5120);
        (counts, (0..130).collect())
    }

    /// The admits the walk makes over `unit512`'s host set at constants `k`:
    /// the host experts at or over its floor.
    fn walk_admits(counts: &[u32], host: &[u32], k: &Constants) -> usize {
        let plan = admit_walk(counts, host, k, 1, ReturnCost::Hidden).unwrap();
        host.iter()
            .filter(|&&id| counts[id as usize] >= plan.floor)
            .count()
    }

    /// The ring's rule never reads the card's tail: m*, m_min and the tail's
    /// split are the same at no tail, at the flat 4690 and at a unit's
    /// priced one (the negative witness of the tail's pricing: the ring's
    /// floor moves with nothing the walk's tail sets; mutant: the floor
    /// reading the tail as a fixed cost).
    #[test]
    fn the_rings_rule_never_reads_the_tail() {
        let (counts, host) = unit512();
        let base = a6000();
        let priced = unit_constants(&base, 0.125, &counts, &host).unwrap();
        let mut zero = priced;
        zero.card_tail_us = 0.0;
        let mut flat = priced;
        flat.card_tail_us = 4690.0;
        let (mut a, mut b, mut c) = (Split::default(), Split::default(), Split::default());
        for (k, out) in [(&zero, &mut a), (&flat, &mut b), (&priced, &mut c)] {
            assert_eq!(m_star(k), m_star(&zero));
            assert_eq!(m_min(k), m_min(&zero));
            split(&counts, &host, 22, k, out).unwrap();
        }
        assert_eq!((&a, &b), (&b, &c));
    }

    /// The tail priced per card pick at a 512-column unit lands the
    /// 989-class (a call's 989 admits over 48 layers, 20.6 a layer): the
    /// walk admits 22 here, its tail 0.125 x 2257 = 282.125. The flat
    /// 4690 the family carried before stops it at 11, under the class, and
    /// a tail of 0 over-admits at 23 (mutants: the tail left at the family's
    /// 4690, the tail left at 0 whatever the rate).
    #[test]
    fn the_priced_tail_admits_the_class_a_flat_tail_cannot() {
        const CLASS: f64 = 989.0 / 48.0;
        let (counts, host) = unit512();
        let k = a6000();
        let priced = unit_constants(&k, 0.125, &counts, &host).unwrap();
        let at = walk_admits(&counts, &host, &priced);
        let mut flat = priced;
        flat.card_tail_us = 4690.0;
        let under = walk_admits(&counts, &host, &flat);
        let mut zero = priced;
        zero.card_tail_us = 0.0;
        let over = walk_admits(&counts, &host, &zero);
        assert!(
            at as f64 >= CLASS,
            "the priced tail admits {at}, under the class {CLASS}"
        );
        assert!(
            (under as f64) < CLASS,
            "a flat tail admits {under}, the class {CLASS}"
        );
        assert!(
            over > at,
            "no tail admits {over}, no more than the priced {at}"
        );
        assert_eq!((at, under, over), (22, 11, 23));
        assert_eq!(priced.card_tail_us, 0.125 * 2257.0);
    }
}
