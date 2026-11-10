//! A load's card buffers that no tensor row describes, as named parts: each
//! part its buffers in allocation order, a buffer its element count and the
//! bytes an element. The one owner of what a family's buffers past its
//! uploads take on the card (a prompt batch, a selection's scores): the
//! family gives only the function of its shapes that lists its parts; the
//! plan counts their sum ([`Parts::card_bytes`], the plan's allocator model
//! at the card's granule, or [`Parts::bytes`]) in a card term
//! ([`CardTotals::grow_prompt`](super::CardTotals::grow_prompt)), and the
//! card body holds what it made to the same parts ([`Parts::check`]), which
//! refuses by name, listing every part apart, so one run shows all drift.
//!
//! It lives here, in the pure crate, because both sides reach it: the
//! planner (`model`, which re-exports this module) and every card body
//! (through `model::placement`), and because the per-buffer rounding is the
//! plan's own allocator model ([`allocator_bytes`]), not a second one.

use std::fmt;
use std::num::NonZeroU64;

use super::allocator_bytes;

/// One buffer of a part: `len` elements of `elem` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub len: u64,
    pub elem: u64,
}

impl Buffer {
    /// The buffer's bytes; panics by name past u64, a shape no card holds.
    #[must_use]
    pub fn bytes(self) -> u64 {
        self.len
            .checked_mul(self.elem)
            .unwrap_or_else(|| panic!("a buffer of {} × {} B passes u64", self.len, self.elem))
    }
}

/// One named part: its buffers in the order the body allocates them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    name: &'static str,
    buffers: Vec<Buffer>,
}

impl Part {
    /// A part of no buffer yet.
    #[must_use]
    pub fn new(name: &'static str) -> Part {
        Part {
            name,
            buffers: Vec::new(),
        }
    }

    /// The part with one more buffer of `len` elements of `elem` bytes.
    #[must_use]
    pub fn with(mut self, len: usize, elem: usize) -> Part {
        self.buffers.push(Buffer {
            len: len as u64,
            elem: elem as u64,
        });
        self
    }

    /// The part with one more buffer of `len` four-byte elements (f32, u32).
    #[must_use]
    pub fn word(self, len: usize) -> Part {
        self.with(len, 4)
    }

    /// The part with one more buffer of `bytes` bytes, a buffer whose owner
    /// sizes it in bytes (a q8_1 form's planes, a selection's scores).
    #[must_use]
    pub fn bytes_buffer(mut self, bytes: u64) -> Part {
        self.buffers.push(Buffer {
            len: bytes,
            elem: 1,
        });
        self
    }

    /// The part's name, as the check names it.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The part's buffers, in allocation order.
    #[must_use]
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Element count of buffer `i`, for a body that allocates from it;
    /// refused by name past the part's buffers or past usize.
    pub fn len(&self, i: usize) -> Result<usize, String> {
        let b = self.buffers.get(i).ok_or_else(|| {
            format!(
                "part {} has {} buffers, not a buffer {i}",
                self.name,
                self.buffers.len()
            )
        })?;
        usize::try_from(b.len)
            .map_err(|_| format!("buffer {i} of part {} holds {} elements", self.name, b.len))
    }

    /// The part's bytes, the buffers summed as they are, unrounded.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buffers.iter().fold(0u64, |s, b| {
            s.checked_add(b.bytes())
                .unwrap_or_else(|| panic!("part {} passes u64 bytes", self.name))
        })
    }
}

/// A load's named parts, in allocation order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parts {
    parts: Vec<Part>,
}

impl Parts {
    /// `parts` in order; panics by name on two parts of one name, which the
    /// check could not tell apart.
    #[must_use]
    pub fn new(parts: Vec<Part>) -> Parts {
        for (i, p) in parts.iter().enumerate() {
            assert!(
                parts[..i].iter().all(|q| q.name != p.name),
                "two parts named {}",
                p.name
            );
        }
        Parts { parts }
    }

    /// These parts with `part` in place of the part of its name, or after
    /// the last when none has it: the parts at a lever the formula's first
    /// count did not take (a larger group's units).
    #[must_use]
    pub fn with_part(mut self, part: Part) -> Parts {
        match self.parts.iter_mut().find(|p| p.name == part.name) {
            Some(p) => *p = part,
            None => self.parts.push(part),
        }
        self
    }

    /// The part named `name`.
    #[must_use]
    pub fn part(&self, name: &str) -> Option<&Part> {
        self.parts.iter().find(|p| p.name == name)
    }

    /// Every part, in order.
    #[must_use]
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// The parts' bytes summed as they are, unrounded.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.parts.iter().fold(0u64, |s, p| {
            s.checked_add(p.bytes())
                .unwrap_or_else(|| panic!("the parts pass u64 bytes"))
        })
    }

    /// The bytes the card's allocator takes for every buffer of the parts,
    /// in order, on a card of allocation granule `granule` (the granule the
    /// card reports): the plan's allocator model ([`allocator_bytes`]), a
    /// buffer of a granule or more in whole granules of its own, a smaller
    /// one packed into shared granules.
    #[must_use]
    pub fn card_bytes(&self, granule: NonZeroU64) -> u64 {
        allocator_bytes(
            granule,
            self.parts
                .iter()
                .flat_map(|p| p.buffers.iter().map(|b| b.bytes())),
        )
    }

    /// Refused by name, every part apart listed, unless the bytes the body
    /// made (`made`, by part name) are the parts' own: a part with no made
    /// figure, a made figure of no part and a part of other bytes are each
    /// apart. `what` names the body.
    pub fn check(
        &self,
        what: &'static str,
        made: &[(&'static str, u64)],
    ) -> Result<(), PartsApart> {
        let mut apart: Vec<Apart> = Vec::new();
        for p in &self.parts {
            let m = made.iter().find(|(n, _)| *n == p.name).map(|&(_, b)| b);
            if m != Some(p.bytes()) {
                apart.push(Apart {
                    name: p.name,
                    made: m,
                    planned: Some(p.bytes()),
                });
            }
        }
        for &(n, b) in made {
            if self.part(n).is_none() {
                apart.push(Apart {
                    name: n,
                    made: Some(b),
                    planned: None,
                });
            }
        }
        if apart.is_empty() {
            return Ok(());
        }
        Err(PartsApart {
            what,
            made: made.iter().map(|&(_, b)| u128::from(b)).sum(),
            planned: u128::from(self.bytes()),
            apart,
        })
    }
}

/// One part whose made bytes are not its formula's: `None` where one side
/// has no such part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Apart {
    pub name: &'static str,
    pub made: Option<u64>,
    pub planned: Option<u64>,
}

impl fmt::Display for Apart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = |b: Option<u64>| b.map_or_else(|| "none".to_owned(), |b| format!("{b} B"));
        write!(
            f,
            "{} made {} planned {}",
            self.name,
            side(self.made),
            side(self.planned)
        )?;
        if let (Some(m), Some(p)) = (self.made, self.planned) {
            write!(f, " ({:+} B)", i128::from(m) - i128::from(p))?;
        }
        Ok(())
    }
}

/// A body's made card buffers that are not its parts' formula
/// ([`Parts::check`]): every part apart, and both sums.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "{what} made {made} B of buffers where its parts' formula counts {planned} B; apart: {} \
     (placement::parts)",
    joined(.apart)
)]
pub struct PartsApart {
    pub what: &'static str,
    pub made: u128,
    pub planned: u128,
    pub apart: Vec<Apart>,
}

fn joined(apart: &[Apart]) -> String {
    apart
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two() -> Parts {
        Parts::new(vec![
            Part::new("a").word(3).with(2, 8),
            Part::new("b").bytes_buffer(5),
        ])
    }

    /// The sums: 3·4 + 2·8 = 28 B in part a, 5 in part b, 33 in all; on a
    /// granule of 64 KiB the three small buffers share one granule, and one
    /// buffer of a granule and one byte takes two.
    #[test]
    fn the_parts_sum_and_round_as_the_plans_allocator() {
        let p = two();
        assert_eq!((p.part("a").map(Part::bytes), p.bytes()), (Some(28), 33));
        let g = NonZeroU64::new(64 << 10).expect("not zero");
        assert_eq!(p.card_bytes(g), 64 << 10);
        let big = Parts::new(vec![Part::new("big").bytes_buffer((64 << 10) + 1)]);
        assert_eq!(big.card_bytes(g), 128 << 10);
    }

    /// Made equal to the parts passes; each kind of drift is listed, all at
    /// once, by name.
    #[test]
    fn the_check_lists_every_part_apart() {
        let p = two();
        assert_eq!(p.check("body", &[("a", 28), ("b", 5)]), Ok(()));
        let e = p
            .check("body", &[("a", 30), ("c", 1)])
            .expect_err("a is apart, b is missing, c is no part");
        assert_eq!(
            e.apart,
            vec![
                Apart {
                    name: "a",
                    made: Some(30),
                    planned: Some(28)
                },
                Apart {
                    name: "b",
                    made: None,
                    planned: Some(5)
                },
                Apart {
                    name: "c",
                    made: Some(1),
                    planned: None
                },
            ]
        );
        assert_eq!(
            e.to_string(),
            "body made 31 B of buffers where its parts' formula counts 33 B; apart: a made 30 B \
             planned 28 B (+2 B); b made none planned 5 B; c made 1 B planned none \
             (placement::parts)"
        );
    }

    /// A part of a name the parts hold takes its place; another name joins
    /// them last.
    #[test]
    fn with_part_replaces_by_name() {
        let p = two().with_part(Part::new("b").bytes_buffer(7));
        assert_eq!((p.parts().len(), p.bytes()), (2, 35));
        let p = p.with_part(Part::new("c").word(1));
        assert_eq!(p.parts().last().map(Part::name), Some("c"));
    }

    #[test]
    #[should_panic(expected = "two parts named a")]
    fn two_parts_of_one_name_are_refused() {
        let _ = Parts::new(vec![Part::new("a"), Part::new("a")]);
    }
}
