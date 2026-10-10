//! Where a saved sequence state's bytes go, as pure arithmetic: the one 2-D
//! copy each positional store takes between the card and the host
//! ([`Strip`]), the prefix of a state a resume puts back ([`plan_prefix`]),
//! and the pinned host extents a state is staged in ([`Extents`],
//! [`Layout`]). The copies and the pinned memory are
//! `bloomery_gpu::checkpoint`'s; the numbers they run with are these.
//!
//! A positional store is `planes` runs of `len` values on the card, the
//! first at value 0 and each `stride` values past the one before; on the
//! host the same runs lie back to back, `len` apart. A run copies as one
//! strip: `planes` rows of `len` values, the card's rows `stride` apart and
//! the host's `len` apart. The prefix of `n` positions of a store whose runs
//! grow with the positions ([`Per`]) is the same strip narrowed to the
//! values of `n` positions, the host's rows still `len` apart (the saved
//! state's), so a resume of `n` positions moves exactly the values of those
//! positions and nothing past them.

use std::fmt;

/// How a run's length follows the positions a state holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Per {
    /// Not known: only the whole run is ever copied.
    Whole,
    /// `values` for each position.
    Position { values: usize },
    /// `values` for each pool of `every` positions, the last pool partial.
    Pool { every: usize, values: usize },
}

impl Per {
    /// The values a plane holds at `n` positions; [`SaveMapError::Whole`] for
    /// a run of unknown growth, [`SaveMapError::Geometry`] for a pool of no
    /// positions or a count past `usize`.
    pub fn len_at(self, n: usize) -> Result<usize, SaveMapError> {
        match self {
            Per::Whole => Err(SaveMapError::Whole),
            Per::Position { values } => n
                .checked_mul(values)
                .ok_or_else(|| SaveMapError::geometry(format!("{n} positions × {values} values"))),
            Per::Pool { every: 0, .. } => {
                Err(SaveMapError::geometry("a pool of 0 positions".to_owned()))
            }
            Per::Pool { every, values } => n.div_ceil(every).checked_mul(values).ok_or_else(|| {
                SaveMapError::geometry(format!("{n} positions in pools of {every}"))
            }),
        }
    }
}

/// One positional store of a state: `planes` runs of `len` values, `stride`
/// values apart on the card, growing with the positions as `per` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub planes: usize,
    pub stride: usize,
    pub len: usize,
    pub per: Per,
}

impl Run {
    /// The values the store's runs hold in all.
    #[must_use]
    pub fn values(&self) -> usize {
        self.planes * self.len
    }

    /// Refused by name when the runs overlap on the card.
    pub fn check(&self) -> Result<(), SaveMapError> {
        if self.planes > 1 && self.stride < self.len {
            return Err(SaveMapError::geometry(format!(
                "{} runs of {} values {} apart overlap",
                self.planes, self.len, self.stride
            )));
        }
        Ok(())
    }
}

/// One 2-D copy of store `run`: `rows` runs of `width` values, the card's
/// `dev_pitch` values apart and the host's `host_pitch` apart, the host's first
/// at value `host_off` of the state's rows. The card's first is the store's
/// own start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strip {
    /// Which store of the state's list the copy moves.
    pub run: usize,
    pub host_off: usize,
    pub width: usize,
    pub rows: usize,
    pub dev_pitch: usize,
    pub host_pitch: usize,
}

impl Strip {
    /// The values the copy moves.
    #[must_use]
    pub fn values(&self) -> usize {
        self.width * self.rows
    }

    /// The `(host value, card value, values)` of each run, as a copy a run
    /// would make: what the strip stands for, one plane at a time.
    pub fn planes(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (0..self.rows).map(|p| {
            (
                self.host_off + p * self.host_pitch,
                p * self.dev_pitch,
                self.width,
            )
        })
    }
}

/// `run` as one strip of `width` values a plane, its host runs at value
/// `host_off`; `None` when there is nothing to copy.
fn strip(index: usize, run: &Run, host_off: usize, width: usize) -> Option<Strip> {
    if width == 0 || run.planes == 0 {
        return None;
    }
    // One row has no pitch to speak of: it is the row's own width.
    let (dev_pitch, host_pitch) = if run.planes == 1 {
        (width, width)
    } else {
        (run.stride, run.len)
    };
    Some(Strip {
        run: index,
        host_off,
        width,
        rows: run.planes,
        dev_pitch,
        host_pitch,
    })
}

/// The strips that move every store's whole runs, in the stores' order, the
/// host runs laid back to back: the plane-by-plane copies of a save or a
/// resume, `planes` calls a store, as `planes` rows of one call.
pub fn plan_whole(runs: &[Run]) -> Result<Vec<Strip>, SaveMapError> {
    let mut at = 0;
    let mut out = Vec::new();
    for (i, r) in runs.iter().enumerate() {
        r.check()?;
        out.extend(strip(i, r, at, r.len));
        at += r.values();
    }
    Ok(out)
}

/// The strips that move the first `n` positions of every store, the host
/// runs where the whole state laid them ([`plan_whole`]'s offsets): each
/// store's width is its values at `n` positions, never past its run. Refused
/// by name for a store of unknown growth ([`Per::Whole`]) and one whose `n`
/// positions hold more values than the saved run.
pub fn plan_prefix(runs: &[Run], n: usize) -> Result<Vec<Strip>, SaveMapError> {
    let mut at = 0;
    let mut out = Vec::new();
    for (i, r) in runs.iter().enumerate() {
        r.check()?;
        let width = r.per.len_at(n)?;
        if width > r.len {
            return Err(SaveMapError::geometry(format!(
                "{n} positions hold {width} values a plane; the saved run holds {}",
                r.len
            )));
        }
        if r.planes > 1 && r.stride < width {
            return Err(SaveMapError::geometry(format!(
                "{} runs of {width} values {} apart overlap",
                r.planes, r.stride
            )));
        }
        out.extend(strip(i, r, at, width));
        at += r.values();
    }
    Ok(out)
}

/// A name for what a refused geometry or prefix was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveMapError {
    /// A store whose runs carry no growth rule was asked for a prefix.
    Whole,
    /// Runs, pitches or counts that disagree.
    Geometry(String),
}

impl SaveMapError {
    fn geometry(detail: String) -> SaveMapError {
        SaveMapError::Geometry(detail)
    }
}

impl fmt::Display for SaveMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SaveMapError::Whole => f.write_str(
                "a prefix of a store whose runs carry no growth rule (built with `Per::Whole`)",
            ),
            SaveMapError::Geometry(d) => f.write_str(d),
        }
    }
}

impl std::error::Error for SaveMapError {}

/// Where a state's two host arrays lie in one staging extent: the rows
/// (`u16`) from byte 0, the recurrent stores (`f32`) from the next
/// [`Layout::ALIGN`] boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The byte the recurrent stores start at.
    pub recurrent_at: usize,
    /// The extent's bytes in all.
    pub total: usize,
}

impl Layout {
    /// The alignment of the recurrent array's start: a cache line.
    pub const ALIGN: usize = 64;

    /// The layout of `rows` `u16` and `recurrent` `f32` values; `None` when
    /// the bytes pass `usize`.
    #[must_use]
    pub fn of(rows: usize, recurrent: usize) -> Option<Layout> {
        let rows_bytes = rows.checked_mul(2)?;
        let recurrent_at = rows_bytes.checked_next_multiple_of(Layout::ALIGN)?;
        let total = recurrent_at.checked_add(recurrent.checked_mul(4)?)?;
        Some(Layout {
            recurrent_at,
            total,
        })
    }
}

/// First-fit extents of one pinned slab of `size` bytes, each a multiple of
/// `align` bytes at a multiple of `align`: what a staging pool lends a state.
/// Freed extents rejoin their neighbours. The slab's memory is not here, only
/// which of its bytes are lent.
#[derive(Clone, Debug)]
pub struct Extents {
    size: usize,
    align: usize,
    /// The free extents as `(offset, len)`, ascending and never touching.
    free: Vec<(usize, usize)>,
}

impl Extents {
    /// A slab of `size` bytes, all free. `size` is rounded down to a
    /// multiple of `align`.
    ///
    /// # Panics
    ///
    /// When `align` is 0 or not a power of two: an extent's alignment names
    /// a hardware boundary.
    #[must_use]
    pub fn new(size: usize, align: usize) -> Extents {
        assert!(
            align.is_power_of_two(),
            "extents aligned to {align} bytes: a power of two"
        );
        let size = size / align * align;
        Extents {
            size,
            align,
            free: if size == 0 {
                Vec::new()
            } else {
                vec![(0, size)]
            },
        }
    }

    /// The slab's bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The bytes free, in all.
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        self.free.iter().map(|&(_, l)| l).sum()
    }

    /// The bytes an extent of `len` takes: `len` up to a multiple of `align`.
    #[must_use]
    pub fn take_of(&self, len: usize) -> Option<usize> {
        len.max(1).checked_next_multiple_of(self.align)
    }

    /// The offset of a new extent of `len` bytes (rounded up, [`Extents::take_of`]),
    /// the first free one that holds it; `None` when no free extent does.
    pub fn take(&mut self, len: usize) -> Option<usize> {
        let need = self.take_of(len)?;
        let i = self.free.iter().position(|&(_, l)| l >= need)?;
        let (off, l) = self.free[i];
        if l == need {
            self.free.remove(i);
        } else {
            self.free[i] = (off + need, l - need);
        }
        Some(off)
    }

    /// The extent at `off` of `len` bytes (the `len` it was taken with)
    /// given back.
    ///
    /// # Panics
    ///
    /// When the extent is not a lent one of this slab: outside it, at an
    /// offset off the alignment, or overlapping a free extent (a double
    /// give). A staging pool that gave memory back twice would lend it to
    /// two states.
    pub fn give(&mut self, off: usize, len: usize) {
        let need = self
            .take_of(len)
            .unwrap_or_else(|| panic!("an extent of {len} bytes passes usize"));
        assert!(
            off.is_multiple_of(self.align) && off.checked_add(need).is_some_and(|e| e <= self.size),
            "an extent of {need} bytes at {off} lies outside a slab of {}",
            self.size
        );
        let i = self.free.partition_point(|&(o, _)| o < off);
        let before = i.checked_sub(1).map(|j| self.free[j]);
        let after = self.free.get(i).copied();
        assert!(
            before.is_none_or(|(o, l)| o + l <= off) && after.is_none_or(|(o, _)| off + need <= o),
            "an extent of {need} bytes at {off} overlaps a free extent: given back twice"
        );
        let mut at = off;
        let mut len = need;
        let mut at_i = i;
        if let Some((o, l)) = before
            && o + l == off
        {
            at = o;
            len += l;
            at_i -= 1;
            self.free.remove(at_i);
        }
        if let Some((o, l)) = after
            && at + len == o
        {
            len += l;
            self.free.remove(at_i);
        }
        self.free.insert(at_i, (at, len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs() -> Vec<Run> {
        let pos = |planes, stride, n: usize, values: usize| Run {
            planes,
            stride,
            len: n * values,
            per: Per::Position { values },
        };
        vec![
            // K and V of a layer: 4 heads of 128 values a position, a context of 64.
            pos(4, 64 * 128, 40, 128),
            pos(4, 64 * 128, 40, 128),
            // The raw keys: one run.
            Run {
                planes: 1,
                stride: 0,
                len: 40 * 16,
                per: Per::Position { values: 16 },
            },
            // The pooled keys: one run, a pool of 8 positions, 2 values a pool.
            Run {
                planes: 1,
                stride: 0,
                len: 40usize.div_ceil(8) * 2,
                per: Per::Pool {
                    every: 8,
                    values: 2,
                },
            },
        ]
    }

    /// The plane-by-plane copies a save ran before one call a store: each
    /// run's `(host value, card value, values)` in order.
    fn by_plane(runs: &[Run]) -> Vec<(usize, usize, usize)> {
        let mut at = 0;
        let mut out = Vec::new();
        for r in runs {
            for p in 0..r.planes {
                if r.len > 0 {
                    out.push((at, p * r.stride, r.len));
                }
                at += r.len;
            }
        }
        out
    }

    /// One strip a store moves the very runs, in the very order, the
    /// plane-by-plane copies did: the bytes are equal, call for call.
    #[test]
    fn a_strip_is_its_planes_and_moves_the_same_bytes() {
        let runs = runs();
        let strips = plan_whole(&runs).expect("a plan");
        assert_eq!(strips.len(), runs.len(), "one call a store");
        let planes: Vec<_> = strips.iter().flat_map(Strip::planes).collect();
        assert_eq!(planes, by_plane(&runs));
        let values: usize = strips.iter().map(Strip::values).sum();
        assert_eq!(values, runs.iter().map(Run::values).sum::<usize>());
        assert_eq!(
            values,
            by_plane(&runs).iter().map(|&(_, _, l)| l).sum::<usize>()
        );
    }

    /// A prefix of `n` positions moves the values of those positions and no
    /// others: per plane `n` rows of the store's row width, the pools those
    /// positions reach, the host's runs still where the whole state lays
    /// them.
    #[test]
    fn a_prefix_moves_only_the_values_of_its_positions() {
        let runs = runs();
        let whole = plan_whole(&runs).expect("a plan");
        for n in [1usize, 7, 8, 9, 31, 40] {
            let strips = plan_prefix(&runs, n).expect("a plan");
            let widths: Vec<usize> = strips.iter().map(|s| s.width).collect();
            assert_eq!(
                widths,
                [n * 128, n * 128, n * 16, n.div_ceil(8) * 2],
                "n = {n}"
            );
            for (p, w) in strips.iter().zip(&whole) {
                assert_eq!(
                    (p.host_off, p.rows),
                    (w.host_off, w.rows),
                    "the host's runs stay where the whole state laid them"
                );
                if p.rows > 1 {
                    assert_eq!((p.host_pitch, p.dev_pitch), (w.host_pitch, w.dev_pitch));
                }
            }
            let moved: usize = strips.iter().map(Strip::values).sum();
            let expect: usize = runs
                .iter()
                .map(|r| r.planes * r.per.len_at(n).expect("a rule"))
                .sum();
            assert_eq!(moved, expect, "n = {n}");
        }
    }

    /// A prefix as long as the state is the whole copy.
    #[test]
    fn a_prefix_of_every_position_is_the_whole() {
        let runs = runs();
        assert_eq!(plan_prefix(&runs, 40), plan_whole(&runs));
    }

    /// Refused by name: a store with no growth rule, a prefix past the saved
    /// run, a pool of no positions, runs that overlap.
    #[test]
    fn refusals_are_named() {
        let mut r = runs();
        r[0].per = Per::Whole;
        assert_eq!(plan_prefix(&r, 5), Err(SaveMapError::Whole));
        assert!(plan_whole(&r).is_ok(), "a whole copy needs no rule");
        let e = plan_prefix(&runs(), 41).expect_err("past the run");
        assert!(e.to_string().contains("saved run holds"), "{e}");
        assert!(
            Per::Pool {
                every: 0,
                values: 1
            }
            .len_at(3)
            .is_err()
        );
        let mut r = runs();
        r[0].stride = 10;
        assert!(
            plan_whole(&r)
                .expect_err("overlap")
                .to_string()
                .contains("overlap")
        );
    }

    /// An empty run is no copy.
    #[test]
    fn an_empty_run_is_no_strip() {
        let r = Run {
            planes: 2,
            stride: 8,
            len: 0,
            per: Per::Position { values: 4 },
        };
        assert_eq!(plan_whole(&[r]).expect("a plan"), vec![]);
        assert_eq!(plan_prefix(&[r], 0).expect("a plan"), vec![]);
    }

    #[test]
    fn layout_aligns_the_recurrent_array() {
        let l = Layout::of(5, 3).expect("a layout");
        assert_eq!((l.recurrent_at, l.total), (64, 64 + 12));
        let l = Layout::of(32, 1).expect("a layout");
        assert_eq!((l.recurrent_at, l.total), (64, 68));
        assert_eq!(Layout::of(usize::MAX, 0), None);
    }

    /// Lent extents never overlap, a given-back one is lent again, and
    /// neighbours rejoin: a slab of four extents taken and given back in any
    /// order is one free extent again.
    #[test]
    fn extents_lend_first_fit_and_coalesce() {
        let mut e = Extents::new(4096, 64);
        let a = e.take(100).expect("a");
        let b = e.take(64).expect("b");
        let c = e.take(1000).expect("c");
        assert_eq!((a, b, c), (0, 128, 192), "first fit, rounded to 64");
        assert_eq!(e.free_bytes(), 4096 - 128 - 64 - 1024);
        e.give(b, 64);
        assert_eq!(e.take(64), Some(b), "the hole is lent again");
        e.give(b, 64);
        e.give(a, 100);
        e.give(c, 1000);
        assert_eq!(e.free_bytes(), 4096);
        assert_eq!(e.take(4096), Some(0), "one extent again");
        assert_eq!(e.take(1), None, "nothing left");
        let mut e = Extents::new(256, 64);
        assert_eq!(e.take(257), None, "more than the slab");
    }

    #[test]
    #[should_panic(expected = "given back twice")]
    fn an_extent_given_twice_is_a_named_panic() {
        let mut e = Extents::new(256, 64);
        let a = e.take(64).expect("a");
        e.give(a, 64);
        e.give(a, 64);
    }

    #[test]
    #[should_panic(expected = "lies outside")]
    fn an_extent_outside_the_slab_is_a_named_panic() {
        let mut e = Extents::new(256, 64);
        e.give(192, 128);
    }
}
