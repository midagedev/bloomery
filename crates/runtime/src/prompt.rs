//! The prompt walk's common owner: what every engine that feeds a prompt as
//! batches decides the same way, once — the feed's two modes
//! ([`PrefillMode`]), how a call is cut into batches ([`batches`],
//! [`call_batches`]) and chunks ([`chunks`]), how a call's batches are
//! walked together ([`group_sets`], [`groups`], [`check_group`],
//! [`refuse_dense_after_routed`]) and the call's end ([`end_of`],
//! [`call_end`]).
//!
//! Host only, and no device error type: a refusal comes back as a
//! [`Refusal`], whose detail the engine crate puts in its own error beside
//! the name it prints, so each engine's error text stays its own.

use std::fmt;
use std::ops::Range;

/// How a prompt is fed: in batches or one decode step per id — the
/// same-binary arm, which is the decode step and not a second
/// implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Batch,
    Steps,
}

impl PrefillMode {
    /// The mode [`PrefillMode::name`] names; `None` for any other word.
    #[must_use]
    pub fn from_name(name: &str) -> Option<PrefillMode> {
        [PrefillMode::Batch, PrefillMode::Steps]
            .into_iter()
            .find(|m| m.name() == name)
    }

    /// The name a `load` line prints and a `--prefill` flag or
    /// `BLOOMERY_PREFILL` takes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PrefillMode::Batch => "batch",
            PrefillMode::Steps => "steps",
        }
    }
}

/// A prompt call or a group refused by name: the detail of the caller's own
/// error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

/// The lever whose value [`check_group`] refuses: the name an error of it
/// prints.
pub const GROUP_LEVER: &str = "BLOOMERY_PREFILL_GROUP";

/// The batches of a run of `n` positions from `first`: `⌈n / t_max⌉` of
/// near-equal size, the first `n mod k` one position longer. Each layer
/// reads every host expert its batch's tokens route to once, so a short last
/// batch would pay that read for few tokens. Panics on a `t_max` of 0.
#[must_use]
pub fn batches(first: usize, n: usize, t_max: usize) -> Vec<Range<usize>> {
    assert!(t_max > 0, "a batch holds at least one position");
    let k = n.div_ceil(t_max);
    let mut out = Vec::with_capacity(k);
    let mut p = first;
    for j in 0..k {
        let len = n / k + usize::from(j < n % k);
        out.push(p..p + len);
        p += len;
    }
    out
}

/// The batches of a call from `from` to `to` whose checkpoint marks are
/// `marks`: each run between two marks cut by [`batches`] at `t_max`
/// positions. A body that takes no checkpoint passes the call's end as the
/// one mark.
#[must_use]
pub fn call_batches(from: u32, to: u32, marks: &[u32], t_max: usize) -> Vec<Range<u32>> {
    let pos = |p: usize| u32::try_from(p).expect("a position inside a u32 call");
    let mut out = Vec::new();
    let mut at = from;
    for &mark in marks.iter().filter(|&&k| k > from && k <= to) {
        out.extend(
            batches(at as usize, (mark - at) as usize, t_max)
                .into_iter()
                .map(|r| pos(r.start)..pos(r.end)),
        );
        at = mark;
    }
    out
}

/// The chunks of a batch of `t` tokens: runs of `chunk` from its first, the
/// last one shorter; `(first token, tokens)` each. Panics on a `chunk` of 0.
pub fn chunks(t: usize, chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..t).step_by(chunk).map(move |c0| (c0, chunk.min(t - c0)))
}

/// The call's end, `from + n`, refused by name for no id or past `u32`.
pub fn end_of(from: u32, n: usize) -> Result<u32, Refusal> {
    u32::try_from(n)
        .ok()
        .and_then(|n| from.checked_add(n))
        .filter(|&to| to > from)
        .ok_or_else(|| Refusal(format!("a prompt of {n} ids from {from}")))
}

/// The end of a call of `n` ids from `from` over stores of `ctx` positions:
/// [`end_of`], refused by name too when it passes the stores, so the model
/// then stands where it stood.
pub fn call_end(from: u32, n: usize, ctx: usize) -> Result<u32, Refusal> {
    let to = end_of(from, n)?;
    if to as usize > ctx {
        return Err(Refusal(format!(
            "a prompt of {n} ids from position {from} ends at {to}, past the {ctx} positions \
             the stores hold (the load's ctx)"
        )));
    }
    Ok(to)
}

/// `group`, the batches a group holds (a lever's value; 1 runs each batch
/// alone, every layer of it before the next batch's first), refused by name
/// with its value unless it is from 1 to `max`. The error prints as
/// [`GROUP_LEVER`].
pub fn check_group(group: usize, max: usize) -> Result<(), Refusal> {
    if (1..=max).contains(&group) {
        Ok(())
    } else {
        Err(Refusal(format!(
            "{group} batches, where a group holds 1 to {max}"
        )))
    }
}

/// Batches a prompt group holds at most under a group lever of `g`: `g`, and
/// one more from 2 on — a call's lone last batch joins the group before it
/// ([`groups`]). The prompt batch's per-unit buffers are made for this many.
#[must_use]
pub const fn group_sets(g: usize) -> usize {
    if g >= 2 { g + 1 } else { 1 }
}

/// The groups of a call of `k` batches under a lever of `g`: runs of `g`
/// consecutive batches, where a lone last batch joins the run before it —
/// a group of one runs no route under another batch's union. A call of one
/// batch is one group of one.
#[must_use]
pub fn groups(k: usize, g: usize) -> Vec<Range<usize>> {
    let g = g.max(1);
    let mut out: Vec<Range<usize>> = (0..k).step_by(g).map(|s| s..(s + g).min(k)).collect();
    if g >= 2
        && out.len() >= 2
        && out.last().is_some_and(|r| r.len() == 1)
        && let Some(tail) = out.pop()
        && let Some(prev) = out.last_mut()
    {
        prev.end = tail.end;
    }
    out
}

/// Refused by name for a group of 2 or more on a load with a dense layer
/// past a routed one: a dense front writes the shared buffers the previous
/// routed item's back still reads (the shared-buffer rule holds only for a
/// dense prefix). `host_leg` is each layer's, in layer order: whether a host
/// tier serves its experts.
pub fn refuse_dense_after_routed(
    host_leg: impl IntoIterator<Item = bool>,
    g: usize,
) -> Result<(), Refusal> {
    if g < 2 {
        return Ok(());
    }
    let mut routed = false;
    for (l, leg) in host_leg.into_iter().enumerate() {
        if leg {
            routed = true;
        } else if routed {
            return Err(Refusal(format!(
                "a prompt group of {g} batches on a load whose dense layer {l} comes after a \
                 routed one: a group's units share buffers only behind a dense prefix"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call's batches cut into groups: runs of `g`, a lone last batch
    /// joining the run before it (never at a lever of 1, never a call of one
    /// batch), and no group past `group_sets(g)` batches.
    #[test]
    fn groups_join_a_lone_last_batch() {
        assert_eq!(groups(1, 2), vec![0..1]);
        assert_eq!(groups(2, 2), vec![0..2]);
        assert_eq!(groups(3, 2), vec![0..3]);
        assert_eq!(groups(5, 2), vec![0..2, 2..5]);
        assert_eq!(groups(4, 2), vec![0..2, 2..4]);
        assert_eq!(groups(5, 4), vec![0..5]);
        assert_eq!(groups(6, 4), vec![0..4, 4..6]);
        assert_eq!(groups(3, 1), vec![0..1, 1..2, 2..3]);
        assert_eq!(groups(0, 2), Vec::<Range<usize>>::new());
        assert_eq!((group_sets(1), group_sets(2), group_sets(8)), (1, 3, 9));
        for g in 1..=8 {
            for k in 1..=20 {
                let cut = groups(k, g);
                assert_eq!(cut.first().map(|r| r.start), Some(0));
                assert_eq!(cut.last().map(|r| r.end), Some(k));
                assert!(cut.windows(2).all(|w| w[0].end == w[1].start));
                assert!(
                    cut.iter()
                        .all(|r| !r.is_empty() && r.len() <= group_sets(g)),
                    "{k} batches at {g}: {cut:?}"
                );
            }
        }
    }

    /// A run of positions cut into `⌈n / t_max⌉` near-equal batches, the
    /// longer ones first, none past `t_max`.
    #[test]
    fn batches_are_near_equal() {
        assert_eq!(batches(7, 0, 512), Vec::<Range<usize>>::new());
        assert_eq!(batches(7, 512, 512), vec![7..519]);
        assert_eq!(batches(0, 513, 512), vec![0..257, 257..513]);
        assert_eq!(batches(0, 4096, 512).len(), 8);
        assert_eq!(batches(0, 1025, 512), vec![0..342, 342..684, 684..1025]);
        for n in 1..=2000 {
            let cut = batches(3, n, 512);
            assert_eq!(cut.len(), n.div_ceil(512));
            assert_eq!(cut.first().map(|r| r.start), Some(3));
            assert_eq!(cut.last().map(|r| r.end), Some(3 + n));
            assert!(cut.windows(2).all(|w| w[0].end == w[1].start));
            let lens: Vec<usize> = cut.iter().map(Range::len).collect();
            assert!(lens.iter().all(|&l| (1..=512).contains(&l)));
            assert!(lens.windows(2).all(|w| w[0] >= w[1] && w[0] - w[1] <= 1));
        }
    }

    /// A call's runs between its marks are cut apart: a mark never falls
    /// inside a batch; marks outside the call are ignored.
    #[test]
    fn call_batches_cut_at_the_marks() {
        assert_eq!(call_batches(0, 700, &[700], 512), vec![0..350, 350..700]);
        assert_eq!(
            call_batches(10, 1100, &[0, 600, 1100, 2000], 512),
            vec![10..305, 305..600, 600..1100]
        );
        assert_eq!(call_batches(5, 6, &[6], 512), vec![5..6]);
        assert_eq!(call_batches(5, 9, &[], 512), Vec::<Range<u32>>::new());
    }

    /// Chunks of a batch: runs of the chunk size, the last one shorter.
    #[test]
    fn chunks_cover_a_batch() {
        let c: Vec<_> = chunks(19, 8).collect();
        assert_eq!(c, vec![(0, 8), (8, 8), (16, 3)]);
        assert_eq!(chunks(8, 8).collect::<Vec<_>>(), vec![(0, 8)]);
        assert_eq!(chunks(0, 8).count(), 0);
        assert_eq!(
            chunks(40, 16).collect::<Vec<_>>(),
            vec![(0, 16), (16, 16), (32, 8)]
        );
    }

    /// The call's end is refused for no id, for a sum past `u32` and for a
    /// count past `u32`, each naming the call.
    #[test]
    fn end_of_refuses_by_name() {
        assert_eq!(end_of(5, 3), Ok(8));
        let text = |r: Result<u32, Refusal>| r.unwrap_err().to_string();
        assert_eq!(text(end_of(5, 0)), "a prompt of 0 ids from 5");
        assert_eq!(
            text(end_of(u32::MAX, 1)),
            format!("a prompt of 1 ids from {}", u32::MAX)
        );
        assert!(text(end_of(0, usize::MAX)).starts_with("a prompt of "));
        assert_eq!(call_end(5, 3, 8), Ok(8));
        assert_eq!(
            call_end(5, 4, 8).unwrap_err().to_string(),
            "a prompt of 4 ids from position 5 ends at 9, past the 8 positions the stores hold \
             (the load's ctx)"
        );
    }

    /// A group is refused by name with its value outside 1 to the most.
    #[test]
    fn group_lever_is_checked() {
        assert!(check_group(1, 8).is_ok() && check_group(8, 8).is_ok());
        assert_eq!(
            check_group(0, 8).unwrap_err().to_string(),
            "0 batches, where a group holds 1 to 8"
        );
        assert!(check_group(9, 8).is_err());
    }

    /// Groups of 2 or more need a dense prefix; a lone batch group never
    /// refuses.
    #[test]
    fn dense_after_routed_is_refused() {
        let prefix = [false, true, true];
        assert!(refuse_dense_after_routed(prefix, 4).is_ok());
        let late = [false, true, false, true];
        assert!(refuse_dense_after_routed(late, 1).is_ok());
        let e = refuse_dense_after_routed(late, 2).unwrap_err().to_string();
        assert!(e.starts_with("a prompt group of 2 batches on a load whose dense layer 2 "));
        assert!(refuse_dense_after_routed([true, true], 3).is_ok());
        assert!(refuse_dense_after_routed([false, false], 3).is_ok());
    }

    /// The feed's two words round-trip, and no other word is a mode.
    #[test]
    fn prefill_mode_names() {
        for m in [PrefillMode::Batch, PrefillMode::Steps] {
            assert_eq!(PrefillMode::from_name(m.name()), Some(m));
        }
        assert_eq!(PrefillMode::from_name("both"), None);
    }
}
