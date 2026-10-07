//! The expert-stream split rule: for one layer of one prompt unit, which host
//! experts the card pool admits, which a transient ring streams over the copy
//! lane, and which the host union keeps — a pure function of the unit's routed
//! counts and constants measured at load. No machine, card or model name
//! enters the rule; every constant is an input.
//!
//! An expert routed `m` columns of the unit leaves the host union when it
//! costs more there than on the lane plus the card:
//! `max(W, c·m) > b/r + f + k·m`, for the host union's fixed cost W and
//! per-column cost c, the lane's rate r, the expert's b bytes and the card's
//! fixed cost f and per-column cost k. [`m_star`] is that inequality's
//! crossover on its per-column branch, `(b/r + f)/(c − k)`; while W is at
//! most `b/r + f` the experts that stream are exactly those routed past it.
//! [`m_min`] is the least unit width that streams: a column spreads its top-k
//! picks over the choice set, so the average expert's count `top_k·m/experts`
//! first clears [`m_star`] at `m_min = m_star·experts/top_k`. Below it the
//! ring stays dark and the pool admits alone. Both lists fill in the one
//! ranking [`rank_key`], so the same counts give the same split.
//! [`stream_tail`] is the ring's half alone, over a host set whose admits
//! are already taken; [`split`] takes its tail through it, so the stream
//! inequality has one caller.

use std::cmp::Reverse;
use std::fmt;

/// The rule's constants: the machine's measured rates and the layer's
/// geometry, every field's unit in its name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constants {
    /// The copy lane's host→card rate [B/µs].
    pub lane_b_per_us: f64,
    /// The host union's per-column cost of one expert [µs/col].
    pub host_us_per_col: f64,
    /// The host union's fixed cost of one expert [µs].
    pub host_us_fixed: f64,
    /// The card's fixed cost of one streamed expert [µs].
    pub card_us_fixed: f64,
    /// The card's per-column cost of one streamed expert [µs/col].
    pub card_us_per_col: f64,
    /// One routed expert's bytes in the layer [B].
    pub expert_b: u64,
    /// The router's choice set: routed experts in the layer.
    pub experts: usize,
    /// The router's picks per column.
    pub top_k: usize,
}

/// The rule for one expert routed `count` columns of the unit: it streams
/// when the host union's cost is above the lane's and the card's together.
fn streams(k: &Constants, count: u32) -> bool {
    let m = f64::from(count);
    let host = k.host_us_fixed.max(k.host_us_per_col * m);
    let lane_card = k.expert_b as f64 / k.lane_b_per_us + k.card_us_fixed + k.card_us_per_col * m;
    host > lane_card
}

/// The stream floor: the count past which the rule's per-column branch
/// streams, `c·m > b/r + f + k·m` solved for m, `(b/r + f)/(c − k)`. A card
/// column that costs at least the host's never pays its fixed cost back: the
/// floor is then infinite. In columns.
#[must_use]
pub fn m_star(k: &Constants) -> f64 {
    let per_col = k.host_us_per_col - k.card_us_per_col;
    if per_col <= 0.0 {
        return f64::INFINITY;
    }
    (k.expert_b as f64 / k.lane_b_per_us + k.card_us_fixed) / per_col
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
    /// m_min at 6 × 8/2 = 24. W = 2 is under b/r + f = 60.
    fn plain() -> Constants {
        Constants {
            lane_b_per_us: 1000.0,
            host_us_per_col: 10.5,
            host_us_fixed: 2.0,
            card_us_fixed: 10.0,
            card_us_per_col: 0.5,
            expert_b: 50000,
            experts: 8,
            top_k: 2,
        }
    }

    /// The A6000 plan (a) calibration: 512 routed experts, ten picks a
    /// column, the lane's rate beside the running union, the no-tile host
    /// union (W 23.36, c 7.0), the common layer kind's expert bytes and the
    /// card's per-expert costs (13 + 0.16 a column).
    /// m* = (3072000/21190 + 13)/(7 − 0.16) = 157.974/6.84 = 23.096,
    /// m_min = ceil(23.096 × 512/10) = ceil(1182.49) = 1183.
    fn a6000() -> Constants {
        Constants {
            lane_b_per_us: 21190.0,
            host_us_per_col: 7.0,
            host_us_fixed: 23.36,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            expert_b: 3072000,
            experts: 512,
            top_k: 10,
        }
    }

    /// The 3090 plan (a) calibration: the A6000's link and host, the card's
    /// costs at the design's ×1.25 (16.25 + 0.2 a column).
    /// m* = (3072000/21190 + 16.25)/(7 − 0.2) = 161.224/6.8 = 23.709,
    /// m_min = ceil(23.709 × 51.2) = ceil(1213.92) = 1214.
    fn a3090() -> Constants {
        Constants {
            card_us_fixed: 16.25,
            card_us_per_col: 0.2,
            ..a6000()
        }
    }

    /// The 5090 calibration: the same expert bytes and routing, the pageable
    /// staging rate that box holds, its fitted host union (W 38.4, c 6.2) and
    /// the card's costs at the design's ×0.6 (7.8 + 0.096 a column).
    /// m* = (3072000/21200 + 7.8)/(6.2 − 0.096) = 152.706/6.104 = 25.017,
    /// m_min = ceil(25.017 × 51.2) = ceil(1280.89) = 1281.
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
    /// 512 × 10/512 = 10 and the unit is under m_min 1183, so the pool admits
    /// alone: ids 301..454, the other 58 keep 58 × 10 = 580 columns on the
    /// union. A 512-column unit that routes every host expert 24 times, past
    /// the floor of 23.096 (211 × 24 = 5064 picks, 56 on the card make
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
    /// 4096 columns every expert carries 80, past the floor of 25.017, and the
    /// unit clears m_min 1281: the pool admits ids 114..171 and the ring
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
    /// constants' comment: m_min 1183, 1214 and 1281 (mutants: the card's
    /// fixed cost subtracted, the mean without the choice set's spread).
    #[test]
    fn m_min_on_all_three_machines() {
        assert!((m_star(&a6000()) - 23.096).abs() < 0.001);
        assert!((m_star(&a3090()) - 23.709).abs() < 0.001);
        assert!((m_star(&their_5090()) - 25.017).abs() < 0.001);
        for (name, k, want) in [
            ("A6000", a6000(), 1183),
            ("3090", a3090(), 1214),
            ("5090", their_5090(), 1281),
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
}
