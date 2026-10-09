//! Which pages of a routed stack a drop names. After a reader has read some
//! of a paged stack's experts through the model file's mapping and consumed
//! them, the NVMe tier drops their pages from the process's page tables and
//! from the page cache (`bloomery-gpu`'s `host::nvtier`, which makes the
//! syscalls). Which pages go is decided here, from the books alone — the
//! experts the host holds and the experts the reader read — never from what
//! `mincore` finds: a page goes when it holds a byte of an expert the reader
//! read, and no byte the host keeps — a held expert's, or a byte outside
//! every expert of the stack (the tensors beside it, a gap). So the runs
//! round inward at a held neighbour and at the stack's ends, and outward
//! into a neighbour the host does not hold.

use std::ops::Range;

/// What the books say of one expert's bytes when a drop is planned
/// ([`drop_runs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// The host keeps its pages (the plan's host segment): no page that holds
    /// a byte of it is named.
    Held,
    /// Read through the mapping and not held: every page that holds a byte of
    /// it is named, but one that holds a byte the host keeps.
    Read,
    /// Neither held nor read: named only where it shares a page with a read
    /// expert and that page holds no byte the host keeps.
    Rest,
}

/// Books a drop cannot be planned from ([`drop_runs`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DropError {
    #[error("a drop over {parts} experts' byte ranges with {marks} marks")]
    Count { parts: usize, marks: usize },
    #[error("a drop in pages of 0 bytes")]
    Page,
    /// The parts are each a nonempty range, in ascending order, apart.
    #[error(
        "expert {id}'s bytes {at:?} are empty or start before {prev}, the end of the one before"
    )]
    Order {
        id: usize,
        at: Range<u64>,
        prev: u64,
    },
}

/// The byte runs `[start, end)` a drop over one stack names, into `out`
/// (cleared first; its capacity reused): `parts[id]` expert `id`'s bytes in
/// the file, ascending and apart, `marks[id]` what the books say of it, and
/// `page` the page in bytes. Each run is whole pages; the runs ascend, apart
/// and merged — experts read one after the other give one run. A page is in
/// a run exactly when it holds a byte of a [`Mark::Read`] expert and no byte
/// of a [`Mark::Held`] one or outside every part. Refused by name: a mark
/// count other than the parts', a page of 0 bytes, and parts that are empty,
/// out of order or overlapping.
pub fn drop_runs(
    parts: &[Range<u64>],
    marks: &[Mark],
    page: u64,
    out: &mut Vec<Range<u64>>,
) -> Result<(), DropError> {
    out.clear();
    check(parts, marks, page)?;
    for (k, part) in parts.iter().enumerate() {
        if marks[k] != Mark::Read {
            continue;
        }
        let (down, up) = (part.start / page * page, part.end.div_ceil(page) * page);
        let lo = if kept_before(parts, marks, k, down) {
            down + page
        } else {
            down
        };
        let hi = if kept_after(parts, marks, k, up) {
            up - page
        } else {
            up
        };
        if lo >= hi {
            continue;
        }
        // The runs come in ascending order of their starts: a nonempty run
        // starts at or before its expert's last page, which the next read
        // expert's run starts at or after.
        match out.last_mut() {
            Some(last) if last.end >= lo => last.end = last.end.max(hi),
            _ => out.push(lo..hi),
        }
    }
    Ok(())
}

/// The books against the parts: one mark a part, a page of some bytes, and
/// parts nonempty, ascending and apart.
fn check(parts: &[Range<u64>], marks: &[Mark], page: u64) -> Result<(), DropError> {
    if parts.len() != marks.len() {
        return Err(DropError::Count {
            parts: parts.len(),
            marks: marks.len(),
        });
    }
    if page == 0 {
        return Err(DropError::Page);
    }
    let mut prev = 0;
    for (id, at) in parts.iter().enumerate() {
        if at.start >= at.end || at.start < prev {
            return Err(DropError::Order {
                id,
                at: at.clone(),
                prev,
            });
        }
        prev = at.end;
    }
    Ok(())
}

/// Whether bytes `from .. parts[k].start` — the part of `k`'s first page
/// before it — hold a byte the host keeps: a held expert's, or one outside
/// every part.
fn kept_before(parts: &[Range<u64>], marks: &[Mark], k: usize, from: u64) -> bool {
    let mut edge = parts[k].start;
    for i in (0..k).rev() {
        if edge <= from {
            return false;
        }
        if parts[i].end < edge || marks[i] == Mark::Held {
            return true;
        }
        edge = parts[i].start;
    }
    edge > from
}

/// Whether bytes `parts[k].end .. to` — the part of `k`'s last page after
/// it — hold a byte the host keeps: a held expert's, or one outside every
/// part.
fn kept_after(parts: &[Range<u64>], marks: &[Mark], k: usize, to: u64) -> bool {
    let mut edge = parts[k].end;
    for i in k + 1..parts.len() {
        if edge >= to {
            return false;
        }
        if parts[i].start > edge || marks[i] == Mark::Held {
            return true;
        }
        edge = parts[i].end;
    }
    edge < to
}

/// The first bytes of `runs` (byte runs, ascending and apart, as
/// [`drop_runs`] writes them) that `pages` covers (page ranges of `page`
/// bytes, ascending and apart — a host set's runs of one file); `None` when
/// no page of `pages` lies in a run.
#[must_use]
pub fn overlap_of(runs: &[Range<u64>], pages: &[Range<u64>], page: u64) -> Option<Range<u64>> {
    let (mut i, mut j) = (0, 0);
    while i < runs.len() && j < pages.len() {
        let held = pages[j].start * page..pages[j].end * page;
        let (a, b) = (runs[i].start.max(held.start), runs[i].end.min(held.end));
        if a < b {
            return Some(a..b);
        }
        if runs[i].end <= held.end {
            i += 1;
        } else {
            j += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{DropError, Mark, drop_runs, overlap_of};
    use std::ops::Range;

    const PAGE: u64 = 4096;

    /// `n` experts of `per` bytes each from byte `base`, one after the
    /// other: a stack's parts as the file lays them out.
    fn stack(base: u64, per: u64, n: usize) -> Vec<Range<u64>> {
        (0..n as u64)
            .map(|i| base + i * per..base + (i + 1) * per)
            .collect()
    }

    fn runs(parts: &[Range<u64>], marks: &[Mark]) -> Vec<Range<u64>> {
        let mut out = Vec::new();
        drop_runs(parts, marks, PAGE, &mut out).expect("books that fit the parts");
        out
    }

    fn up(b: u64) -> u64 {
        b.div_ceil(PAGE) * PAGE
    }

    fn down(b: u64) -> u64 {
        b / PAGE * PAGE
    }

    /// A read expert between two held ones keeps both edge pages: its run
    /// is the whole pages inside its own bytes, the pages it shares with the
    /// held neighbours stay. Mutant: outward rounding at a held edge — both
    /// edge pages named — reds the equality.
    #[test]
    fn a_run_between_two_held_ids_keeps_both_edge_pages() {
        // Three pages and 100 B an expert from an unaligned base: every
        // expert's first and last page is shared with its neighbour's.
        let parts = stack(PAGE + 40, 3 * PAGE + 100, 3);
        let mid = parts[1].clone();
        let r = runs(&parts, &[Mark::Held, Mark::Read, Mark::Held]);
        assert_eq!(r, vec![up(mid.start)..down(mid.end)]);
        assert!(
            r[0].start > mid.start && r[0].end < mid.end,
            "{r:?} in {mid:?}"
        );
    }

    /// Read experts one after the other give one run, from the first whole
    /// page past the held expert before them to the last before the held one
    /// after: the pages two of them share go with them. Mutant: no merge — a
    /// run an expert — reds the equality.
    #[test]
    fn adjacent_non_held_ids_merge_into_one_run() {
        let parts = stack(PAGE + 40, 3 * PAGE + 100, 6);
        let mut marks = vec![Mark::Read; 6];
        marks[0] = Mark::Held;
        marks[5] = Mark::Held;
        assert_eq!(
            runs(&parts, &marks),
            vec![up(parts[1].start)..down(parts[4].end)]
        );
        // A rest expert smaller than a page between two read ones joins
        // their runs through the page the three share; the last run ends
        // inward at the stack's end.
        let parts = stack(PAGE + 40, 3 * PAGE + 100, 2)
            .into_iter()
            .chain(stack(7 * PAGE + 240, 160, 1))
            .chain(stack(7 * PAGE + 400, 3 * PAGE, 1))
            .collect::<Vec<_>>();
        let r = runs(&parts, &[Mark::Held, Mark::Read, Mark::Rest, Mark::Read]);
        assert_eq!(r, vec![up(parts[1].start)..down(parts[3].end)]);
    }

    /// Over books drawn from a fixed seed — stacks of any expert size
    /// against the page, from any base, held, read and rest experts mixed —
    /// a held expert's every byte, and every byte outside the stack, stays
    /// outside every run; the runs are whole pages, ascending and apart; and
    /// a page is in a run exactly when it holds a read byte and no kept one.
    /// Mutant: a held expert's range named reds the first clause.
    #[test]
    fn a_held_ids_every_byte_stays_outside_every_dropped_range() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for case in 0..2000 {
            let n = 1 + next(24) as usize;
            let per = 1 + next(5 * PAGE);
            let base = next(3 * PAGE);
            let parts = stack(base, per, n);
            let marks: Vec<Mark> = (0..n)
                .map(|_| match next(3) {
                    0 => Mark::Held,
                    1 => Mark::Read,
                    _ => Mark::Rest,
                })
                .collect();
            let r = runs(&parts, &marks);
            let end = base + n as u64 * per;
            for w in r.windows(2) {
                assert!(w[0].end < w[1].start, "case {case}: {r:?} not apart");
            }
            for run in &r {
                assert!(
                    run.start < run.end && run.start % PAGE == 0 && run.end % PAGE == 0,
                    "case {case}: {run:?} is not whole pages"
                );
                assert!(
                    base <= run.start && run.end <= end,
                    "case {case}: {run:?} leaves the stack {base}..{end}"
                );
                for (p, m) in parts.iter().zip(&marks) {
                    assert!(
                        *m != Mark::Held || p.end <= run.start || run.end <= p.start,
                        "case {case}: {run:?} names held bytes {p:?}"
                    );
                }
            }
            for pg in base / PAGE..end.div_ceil(PAGE) {
                let (a, b) = (pg * PAGE, (pg + 1) * PAGE);
                let holds = |want: Mark| {
                    parts
                        .iter()
                        .zip(&marks)
                        .any(|(p, &m)| m == want && p.start < b && a < p.end)
                };
                let kept = a < base || b > end || holds(Mark::Held);
                let named = r.iter().any(|run| run.start <= a && b <= run.end);
                assert_eq!(
                    named,
                    holds(Mark::Read) && !kept,
                    "case {case}: page {pg} of {parts:?} under {marks:?}"
                );
            }
        }
    }

    /// A read expert smaller than a page names no page when that page holds
    /// a held neighbour's bytes, or bytes before the stack; between rest
    /// neighbours that fill the page with it, the page goes.
    #[test]
    fn a_part_smaller_than_a_page_names_no_page() {
        let tiny = stack(PAGE + 100, 512, 3);
        assert_eq!(runs(&tiny, &[Mark::Held, Mark::Read, Mark::Held]), vec![]);
        assert_eq!(runs(&stack(PAGE + 100, 512, 1), &[Mark::Read]), vec![]);
        let filled = stack(PAGE, 512, 8);
        let mut marks = vec![Mark::Rest; 8];
        marks[2] = Mark::Read;
        assert_eq!(runs(&filled, &marks), vec![PAGE..2 * PAGE]);
    }

    /// Books that do not fit their parts are refused by name: a mark count
    /// other than the parts', a page of 0 bytes, an empty part and parts
    /// that overlap.
    #[test]
    fn books_that_do_not_fit_the_parts_are_refused_by_name() {
        let mut out = stack(0, PAGE, 1);
        let parts = stack(0, PAGE, 2);
        assert_eq!(
            drop_runs(&parts, &[Mark::Read], PAGE, &mut out),
            Err(DropError::Count { parts: 2, marks: 1 })
        );
        assert!(out.is_empty(), "a refusal leaves no run behind");
        assert_eq!(
            drop_runs(&parts, &[Mark::Read; 2], 0, &mut out),
            Err(DropError::Page)
        );
        assert_eq!(
            drop_runs(&[0..10, 10..10], &[Mark::Read; 2], PAGE, &mut out),
            Err(DropError::Order {
                id: 1,
                at: 10..10,
                prev: 10
            })
        );
        assert_eq!(
            drop_runs(&[0..10, 5..20], &[Mark::Read; 2], PAGE, &mut out),
            Err(DropError::Order {
                id: 1,
                at: 5..20,
                prev: 10
            })
        );
    }

    /// A host set's page that lies inside a run is found, with the bytes the
    /// two share; pages that only touch a run are not.
    #[test]
    fn a_held_page_inside_a_run_is_found() {
        let held = |pages: &[(u64, u64)]| pages.iter().map(|&(a, b)| a..b).collect::<Vec<_>>();
        let runs = [2 * PAGE..5 * PAGE, 9 * PAGE..10 * PAGE];
        assert_eq!(
            overlap_of(&runs, &held(&[(0, 2), (5, 9), (10, 12)]), PAGE),
            None
        );
        assert_eq!(
            overlap_of(&runs, &held(&[(0, 3)]), PAGE),
            Some(2 * PAGE..3 * PAGE)
        );
        assert_eq!(
            overlap_of(&runs, &held(&[(6, 7), (9, 11)]), PAGE),
            Some(9 * PAGE..10 * PAGE)
        );
        assert_eq!(overlap_of(&[], &held(&[(0, 12)]), PAGE), None);
    }
}
