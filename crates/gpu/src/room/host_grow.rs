//! The host side of the K/V rooms' growth, the common owner for every body: before a fit backs
//! host pages (a ring shadow's, a spill room's), the host K/V the process holds
//! ([`super::host::host_backed`]) and the fit's host asks are read against the plan's books
//! ([`HostGrow::due`]); past what the plan charged and the yields done hold, the host
//! residency machine gives up the next host experts in the plan's yield order
//! ([`HostYielder::yield_experts`]: their pages unlocked where the load locked them, then
//! dropped from the mapping and the page cache, so the next read of one faults it in from the
//! file as a paged expert's does), and the yield is counted and kept for the caller's record
//! ([`HostGrowBook::take_moves`]). The arithmetic is `placement::host_grow`'s; nothing here
//! names a model, and a body takes part with its rooms' host asks and the two handles
//! ([`KvRoomBody::host_asks`], [`KvRoomBody::host_grow`]) beside a book made at load
//! ([`HostGrowBook::at_load`]). Never inside a capture: it runs where [`super::kv::fit_rooms`]
//! runs.

use ::model::placement::{HostGrow, Plan, Yield, YieldAsk, YieldCursor};

use super::kv::KvRoomBody;
use crate::GpuError;

const WHAT: &str = "host room";

/// The host residency machine's move a yield asks for: the pages of `yields` given up.
pub trait HostYielder {
    /// Give up the host pages of `yields`, the plan's yields from index `at` on (each expert's
    /// bytes over its layer's routed stacks): unlocked first where the load locked them, then
    /// dropped from this process's mapping and the page cache. The bytes dropped, whole pages.
    ///
    /// # Errors
    ///
    /// The machine's refusal by name: a run it cannot release.
    fn yield_experts(&mut self, at: usize, yields: &[Yield]) -> Result<u64, GpuError>;
}

/// One yield a fit asked for: the positions the fit reached, the shortfall past the books, the
/// experts and their bytes given up, the bytes their pages dropped, and the host K/V bytes the
/// yields left to come still cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostYieldMove {
    pub positions: usize,
    pub shortfall: u64,
    pub experts: usize,
    pub bytes: u64,
    pub dropped: u64,
    pub left: u64,
}

/// The tallies of a load's yields since it opened: the yields made, the experts and their bytes
/// given up, the bytes their pages dropped, the positions the last yield's fit reached (0 before
/// one), and the host expert bytes left to yield.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostGrowStats {
    pub yields: u64,
    pub experts: u64,
    pub bytes: u64,
    pub dropped: u64,
    pub positions: u64,
    pub left: u64,
}

/// A load's books for its host K/V's growth: the plan's [`HostGrow`], the yields done, the room
/// page a yield adds past the shortfall, the tallies and the moves not yet printed.
#[derive(Debug)]
pub struct HostGrowBook {
    grow: HostGrow,
    at: YieldCursor,
    page: u64,
    stats: HostGrowStats,
    moves: Vec<HostYieldMove>,
}

impl HostGrowBook {
    /// The books of a load whose plan holds `grow`, its rooms' host page `page`; `None` for a
    /// plan with no yield (its K/V charged at the cap, or a room that holds it).
    #[must_use]
    pub fn of(grow: &HostGrow, page: u64) -> Option<HostGrowBook> {
        (!grow.yields.is_empty()).then(|| HostGrowBook {
            grow: grow.clone(),
            at: YieldCursor::default(),
            page,
            stats: HostGrowStats {
                left: grow.yield_bytes,
                ..HostGrowStats::default()
            },
            moves: Vec::new(),
        })
    }

    /// The books of a load of `plan` whose rooms take host pages on visible device `ordinal`
    /// ([`crate::host_page`]), read once at load; `None` for a plan with no yield.
    ///
    /// # Errors
    ///
    /// The device's page unreadable.
    pub fn at_load(plan: &Plan<'_>, ordinal: usize) -> Result<Option<HostGrowBook>, GpuError> {
        if plan.host.grow.yields.is_empty() {
            return Ok(None);
        }
        let page = super::host::page_for(ordinal)?;
        Ok(HostGrowBook::of(&plan.host.grow, page as u64))
    }

    /// The tallies so far.
    #[must_use]
    pub fn stats(&self) -> HostGrowStats {
        self.stats
    }

    /// The yields done.
    #[must_use]
    pub fn cursor(&self) -> YieldCursor {
        self.at
    }

    /// The moves made since the last call, oldest first.
    pub fn take_moves(&mut self) -> Vec<HostYieldMove> {
        std::mem::take(&mut self.moves)
    }

    /// The bytes of host experts left to yield.
    #[must_use]
    pub fn left(&self) -> u64 {
        self.grow.yield_bytes.saturating_sub(self.at.bytes)
    }

    /// The yields a fit through `through` positions asks for, its `asks` bytes of new host
    /// pages on top of the host K/V the process holds (`held`), made by `yielder`: none for a
    /// fit that backs no new host page or within the plan's charge and the yields done; else
    /// the fewest next yields that reach the shortfall plus a page, or every one left
    /// ([`HostGrow::due`]), counted and kept as a move. A host that already holds past what the
    /// plan charged and every yield holds is the named [`GpuError::NoRoom`] of the host room:
    /// what it holds, the yields left, and the fit it was for.
    ///
    /// # Errors
    ///
    /// As above, and the machine's refusal of a yield.
    pub fn grow_to(
        &mut self,
        (held, asks): (u64, u64),
        through: usize,
        yielder: &mut dyn HostYielder,
    ) -> Result<(), GpuError> {
        match self.grow.due((held, asks), self.at, self.page) {
            Ok(None) => Ok(()),
            Ok(Some(ask)) => self.make(ask, yielder, through),
            Err(short) => Err(GpuError::NoRoom {
                what: WHAT,
                detail: format!(
                    "{short}; the next step was the fit through {through} positions, {} B of \
                     host experts yielded so far",
                    self.at.bytes
                ),
                source: Box::new(GpuError::State {
                    what: WHAT,
                    missing: "a host expert left to yield to the K/V's growth",
                }),
                asked: usize::try_from(short.held.saturating_add(asks) - short.allowed)
                    .unwrap_or(usize::MAX),
                free: None,
                page: usize::try_from(self.page).unwrap_or(usize::MAX),
                spill: true,
            }),
        }
    }

    /// The yields of `ask`, made by `yielder` for a fit through `positions`: counted, and kept
    /// as a move.
    fn make(
        &mut self,
        ask: YieldAsk,
        yielder: &mut dyn HostYielder,
        positions: usize,
    ) -> Result<(), GpuError> {
        let yields = self.grow.yields.get(ask.from..ask.to).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "yields {}..{} of a plan's {}",
                    ask.from,
                    ask.to,
                    self.grow.yields.len()
                ),
            )
        })?;
        let dropped = yielder.yield_experts(ask.from, yields)?;
        self.at = YieldCursor {
            done: ask.to,
            bytes: self.at.bytes.saturating_add(ask.bytes),
        };
        let experts = ask.to - ask.from;
        self.stats.yields += 1;
        self.stats.experts += experts as u64;
        self.stats.bytes = self.stats.bytes.saturating_add(ask.bytes);
        self.stats.dropped = self.stats.dropped.saturating_add(dropped);
        self.stats.positions = positions as u64;
        self.stats.left = self.left();
        self.moves.push(HostYieldMove {
            positions,
            shortfall: ask.shortfall,
            experts,
            bytes: ask.bytes,
            dropped,
            left: self.left(),
        });
        Ok(())
    }
}

/// A fit through `through` positions on `body`'s rooms with its host side read first: nothing
/// on a body with no books ([`KvRoomBody::host_grow`]) or a fit that backs no new host page;
/// else the host K/V the process holds now and the fit's host asks ([`KvRoomBody::host_asks`])
/// read against the books, and the yields they ask for made ([`HostGrowBook::grow_to`]).
///
/// # Errors
///
/// As above, and the machine's refusal of a yield.
pub fn take_host_yield<B: KvRoomBody>(body: &mut B, through: usize) -> Result<(), GpuError> {
    let asks = body.host_asks(through);
    if asks == 0 {
        return Ok(());
    }
    let Some((book, yielder)) = body.host_grow() else {
        return Ok(());
    };
    book.grow_to((super::host::host_backed(), asks), through, yielder)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The yields a fake machine was asked for, `(from, ids)`.
    #[derive(Default)]
    struct Asked(Vec<(usize, Vec<u32>)>);

    impl HostYielder for Asked {
        fn yield_experts(&mut self, at: usize, yields: &[Yield]) -> Result<u64, GpuError> {
            self.0.push((at, yields.iter().map(|y| y.id).collect()));
            Ok(yields.iter().map(|y| y.bytes).sum())
        }
    }

    /// 1,000 B charged at the floor and 1,000 B of growth, of which three host experts of 300 B
    /// yield, ids 9, 8, 7 in order; a room page of 100 B.
    fn book() -> HostGrowBook {
        let mut grow = HostGrow::at_floor(8, 1_000, 1_000);
        grow.yields = [9, 8, 7]
            .into_iter()
            .map(|id| Yield {
                layer: 0,
                id,
                bytes: 300,
            })
            .collect();
        grow.yield_bytes = 900;
        grow.hold = model::placement::GrowHold::Yields;
        HostGrowBook::of(&grow, 100).expect("a book with yields")
    }

    /// Within the charge (1,100 B) nothing yields; a fit past it yields the fewest next experts
    /// that reach the shortfall plus a page, in the plan's order, and the move counts them; a
    /// host that already holds past every yield is the host room's named refusal and yields
    /// nothing; a fit the yields left cannot reach while the host holds within them yields
    /// them all. FAIL-first: a book that asks for the shortfall alone yields one expert at
    /// 1,400 B, not two; a book that refuses on the fit's asks refuses the last fit.
    #[test]
    fn a_fit_yields_the_shortfall_plus_a_page_and_past_the_yields_is_refused() {
        let mut b = book();
        let mut m = Asked::default();
        b.grow_to((1_000, 100), 64, &mut m)
            .expect("within the charge");
        assert!(m.0.is_empty() && b.take_moves().is_empty());
        b.grow_to((1_000, 400), 128, &mut m).expect("a yield");
        assert_eq!(m.0, [(0, vec![9, 8])]);
        let moved = b.take_moves();
        assert_eq!(
            moved,
            [HostYieldMove {
                positions: 128,
                shortfall: 300,
                experts: 2,
                bytes: 600,
                dropped: 600,
                left: 300,
            }]
        );
        assert_eq!(
            b.cursor(),
            YieldCursor {
                done: 2,
                bytes: 600
            }
        );
        b.grow_to((1_700, 0), 192, &mut m)
            .expect("within the yields done");
        let err = b
            .grow_to((2_001, 300), 256, &mut m)
            .expect_err("a host past every yield");
        assert!(
            matches!(
                err,
                GpuError::NoRoom {
                    what: "host room",
                    asked: 301,
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(b.stats().experts, 2, "a refused growth yields nothing");
        assert_eq!(m.0.len(), 1);
        b.grow_to((1_900, 300), 320, &mut m)
            .expect("the last yield, short of the fit's asks");
        assert_eq!(m.0, [(0, vec![9, 8]), (2, vec![7])]);
        assert_eq!(
            b.stats(),
            HostGrowStats {
                yields: 2,
                experts: 3,
                bytes: 900,
                dropped: 900,
                positions: 320,
                left: 0,
            }
        );
    }
}
