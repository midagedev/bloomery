//! The card bytes of the graphs a load captures past its load reading: the one owner of what a
//! load captures there ([`LoadCaptures`]) and of the term its plan counts for them
//! ([`CardTotals::grow_graphs`](super::CardTotals::grow_graphs)). A load reads its card once it
//! opened and captured its step; then it adds its slots, opens its drafts (every slot's verify
//! passes and the draft's own walks) and captures its rounds (every pass a round of its slots runs,
//! and each slot's own step). Each captured graph is instantiated and uploaded once, and the
//! driver keeps device bytes a node for it ([`GRAPH_NODE_BYTES`]), which no buffer of the plan
//! counts.
//!
//! A seat fixes its rule once, before any plan ([`CaptureRule`]): every plan it asks derives the
//! captures from its own slots, draft and context ([`CaptureRule::of`]), as the pass heads are, and
//! the load's capture calls read the same value, so the plan and the load cannot drift. Model-free:
//! a family gives the nodes of one captured pass and one draft walk ([`GraphNodes`]) from its own
//! launch counts.

use std::num::NonZeroU64;

use super::PlacementError;

/// The device bytes the driver keeps for one node of a graph a load instantiates and uploads
/// (`cuGraphInstantiateWithFlags`, `cuGraphUpload`): the one term of a load's captures the code
/// cannot fix. A pass of the Qwen3.8 body's 1,381 nodes holds at most 7 of the driver's steps
/// ([`GRAPH_STEP_BYTES`]), 10,630 B a node, rounded up to 10.5 KiB.
/// PIN(2026-11-10): the `bytes=` of the capture lines on the 3090 (Qwen3.8, real and fixture
/// files, two slots), each phase at most [`graph_bytes`] a graph
/// ([`tests::every_measured_capture_fits_its_graphs`]).
pub const GRAPH_NODE_BYTES: u64 = 10_752;

/// The step the driver keeps a pass's graph bytes in: every capture line's `bytes=` is a
/// multiple of it, and a pass of a few hundred nodes still takes one.
pub const GRAPH_STEP_BYTES: u64 = 2 << 20;

/// The card bytes one pass's graph of `nodes` nodes keeps: [`GRAPH_NODE_BYTES`] a node, in whole
/// [`GRAPH_STEP_BYTES`].
#[must_use]
pub const fn graph_bytes(nodes: u64) -> u64 {
    (nodes * GRAPH_NODE_BYTES).next_multiple_of(GRAPH_STEP_BYTES)
}

/// The most slots whose rounds a load captures at its open: the rounds grow as
/// (rows + 1)^slots, so a load of more slots captures a pass on its first use instead.
pub const ROUND_SLOTS: usize = 2;

/// What a slot's draft captures on it: its verify passes of 2 to `verify_rows` rows and its own
/// walks (an MTP draft's; none for a lookup draft, which has no layer of its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftCaptures {
    pub verify_rows: usize,
    pub walks: usize,
}

/// The rounds a load captures: every pass a round of its slots runs when each busy slot holds 1
/// to `rows` rows, cut within `limits` (a pass's rows and slots, the body's), and each slot's own
/// step past slot 0's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundCaptures {
    pub rows: usize,
    pub limits: (usize, usize),
}

/// The nodes of one captured graph of the load's family: a pass of the target (its verify, a
/// round's pass, a slot's step: one bound for all) and one walk of its draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphNodes {
    pub pass: u64,
    pub walk: u64,
}

/// What a seat's load captures under one draft choice: the draft every slot opens, the rows a
/// busy slot holds in a round's pass, and whether the load's rounds of several slots run as one
/// pass of the busy rows (the body's and the placement's fact).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureArm {
    pub draft: Option<DraftCaptures>,
    pub rows: usize,
    pub one_pass: bool,
}

/// What a seat's loads capture past their reading, by the facts the seat fixes before any plan: a
/// plan that drafts takes `drafted`, one that drafts no weights `undrafted` (a lookup draft's, or
/// none), each within the body's pass `limits` (a pass's rows and slots).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureRule {
    pub drafted: CaptureArm,
    pub undrafted: CaptureArm,
    pub limits: (usize, usize),
}

impl CaptureRule {
    /// What a load of `slots` slots of `ctx` positions a slot captures, drafted or not: the arm's
    /// draft on every slot, and the rounds of the arm's rows cut to `ctx` when its rounds of several
    /// slots run as one pass ([`LoadCaptures::of`]). The one value a plan counts and the load's
    /// capture calls read.
    #[must_use]
    pub fn of(&self, slots: usize, drafted: bool, ctx: usize) -> LoadCaptures {
        let arm = if drafted {
            self.drafted
        } else {
            self.undrafted
        };
        LoadCaptures::of(
            slots,
            arm.draft,
            slots > 1 && arm.one_pass,
            (arm.rows.min(ctx), self.limits),
        )
    }
}

/// What a load of `slots` resident slots captures past its load reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadCaptures {
    pub slots: usize,
    /// The draft every slot opens; `None` for a load that drafts nothing.
    pub draft: Option<DraftCaptures>,
    /// The rounds the load captures; `None` for a load that captures none.
    pub rounds: Option<RoundCaptures>,
}

impl LoadCaptures {
    /// A load that captures nothing past its reading: a plan's term of 0.
    pub const NONE: LoadCaptures = LoadCaptures {
        slots: 1,
        draft: None,
        rounds: None,
    };

    /// What a load of `slots` slots captures: `draft` on every slot, and the rounds of `rows` rows
    /// a busy slot within `limits` when its rounds run as one pass of the busy rows (`together`)
    /// and it holds at most [`ROUND_SLOTS`] slots. The one condition a seat's capture calls and its
    /// plan both read.
    #[must_use]
    pub fn of(
        slots: usize,
        draft: Option<DraftCaptures>,
        together: bool,
        (rows, limits): (usize, (usize, usize)),
    ) -> LoadCaptures {
        LoadCaptures {
            slots,
            draft,
            rounds: (together && slots <= ROUND_SLOTS).then_some(RoundCaptures { rows, limits }),
        }
    }

    /// The target passes the load captures past its reading: each slot's verify passes of 2 to
    /// the draft's rows, and with the rounds every pass of them ([`runtime::round_keys`], the
    /// count the session's capture runs) and each slot's step but slot 0's, which the load
    /// captured before its reading.
    ///
    /// # Errors
    ///
    /// [`PlacementError::Rounds`] for rounds the pass's bounds refuse.
    pub fn passes(&self) -> Result<u64, PlacementError> {
        let verify = self
            .draft
            .map_or(0, |d| self.slots * d.verify_rows.saturating_sub(1));
        let rounds = match self.rounds {
            Some(r) => {
                runtime::round_keys(self.slots, r.rows, r.limits)?.len()
                    + self.slots.saturating_sub(1)
            }
            None => 0,
        };
        Ok((verify + rounds) as u64)
    }

    /// Whether the load captures no graph past its reading: no pass and no walk.
    ///
    /// # Errors
    ///
    /// As [`LoadCaptures::passes`].
    pub fn captures_none(&self) -> Result<bool, PlacementError> {
        Ok(self.passes()? == 0 && self.walks() == 0)
    }

    /// The draft walks the load captures past its reading: the draft's walks on every slot.
    #[must_use]
    pub fn walks(&self) -> u64 {
        self.draft.map_or(0, |d| (self.slots * d.walks) as u64)
    }

    /// The device bytes the driver keeps for the load's captures past its reading, graphs of
    /// `nodes` each, rounded up to `granule`: [`LoadCaptures::passes`] passes and
    /// [`LoadCaptures::walks`] walks: each pass's graph in whole steps ([`graph_bytes`]), the
    /// walks' small graphs together ([`graph_bytes`] of all their nodes). 0 for a load that captures
    /// nothing ([`LoadCaptures::captures_none`]).
    ///
    /// # Errors
    ///
    /// As [`LoadCaptures::passes`].
    pub fn card_bytes(
        &self,
        nodes: GraphNodes,
        granule: NonZeroU64,
    ) -> Result<u64, PlacementError> {
        if self.captures_none()? {
            return Ok(0);
        }
        let passes = self.passes()? * graph_bytes(nodes.pass);
        let walks = graph_bytes(self.walks() * nodes.walk);
        Ok((passes + walks).next_multiple_of(granule.get()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn granule() -> NonZeroU64 {
        NonZeroU64::new(2 * MIB).expect("not zero")
    }

    /// Qwen3.8's MTP draft: verify passes of 2 to 4 rows, five walks a slot.
    const MTP: DraftCaptures = DraftCaptures {
        verify_rows: 4,
        walks: 5,
    };

    /// The capture lines' `bytes=` on the 3090 (Qwen3.8, two slots): one verify pass at its
    /// largest, the plain rounds' 3 passes and the drafted rounds' 24 (1,381 nodes each on the
    /// real file, 132 on the fixture), and the drafts' line less its verify passes (ten walks).
    #[test]
    fn every_measured_capture_fits_its_graphs() {
        for (what, graphs, nodes, made) in [
            ("verify rows=3", 1_u64, 1_381_u64, 14_680_064_u64),
            ("rounds rows=1", 3, 1_380, 44_040_192),
            ("rounds rows=4", 24, 1_381, 306_184_192),
            ("fixture rounds rows=1", 3, 132, 6_291_456),
        ] {
            assert!(
                graphs * graph_bytes(nodes) >= made,
                "{what}: {graphs} x {nodes} nodes made {made} B"
            );
        }
        assert!(graph_bytes(10 * 38) >= 79_691_776 - 77_594_624, "the walks");
    }

    /// Two slots drafted, rounds of 4 rows a slot in passes of 8 rows and 8 slots: 6 verify
    /// passes, 24 round passes and slot 1's step, ten walks.
    #[test]
    fn two_drafted_slots_capture_their_rounds() {
        let c = LoadCaptures::of(2, Some(MTP), true, (4, (8, 8)));
        assert_eq!(c.passes().expect("passes"), 6 + 24 + 1);
        assert_eq!(c.walks(), 10);
        let nodes = GraphNodes {
            pass: 1_440,
            walk: 48,
        };
        let bytes = 31 * graph_bytes(1_440) + graph_bytes(10 * 48);
        assert_eq!(
            c.card_bytes(nodes, granule()).expect("bytes"),
            bytes.next_multiple_of(2 * MIB)
        );
    }

    /// A rule's plan of one slot drafted captures the draft alone; of two drafted slots, their
    /// rounds too; of two slots in turns, no rounds; undrafted with no draft, nothing at one slot.
    #[test]
    fn a_rule_derives_each_plans_captures() {
        let rule = CaptureRule {
            drafted: CaptureArm {
                draft: Some(MTP),
                rows: 4,
                one_pass: true,
            },
            undrafted: CaptureArm {
                draft: None,
                rows: 1,
                one_pass: true,
            },
            limits: (8, 8),
        };
        assert_eq!(
            rule.of(1, true, 4096),
            LoadCaptures::of(1, Some(MTP), false, (4, (8, 8)))
        );
        assert_eq!(rule.of(2, true, 4096).passes().expect("passes"), 31);
        assert!(rule.of(1, false, 4096).captures_none().expect("passes"));
        let turns = CaptureRule {
            drafted: CaptureArm {
                one_pass: false,
                ..rule.drafted
            },
            ..rule
        };
        assert!(turns.of(2, true, 4096).rounds.is_none());
        assert_eq!(rule.of(2, true, 2).rounds.map(|r| r.rows), Some(2));
    }

    /// More slots than [`ROUND_SLOTS`] or rounds a slot at a time capture no rounds; a load of
    /// one plain slot captures nothing, so its term is 0.
    #[test]
    fn rounds_only_of_few_slots_together() {
        assert!(
            LoadCaptures::of(3, None, true, (1, (8, 8)))
                .rounds
                .is_none()
        );
        assert!(
            LoadCaptures::of(2, None, false, (1, (8, 8)))
                .rounds
                .is_none()
        );
        let plain = LoadCaptures::of(1, None, false, (1, (8, 8)));
        assert_eq!(plain, LoadCaptures::NONE);
        let nodes = GraphNodes { pass: 1, walk: 1 };
        assert_eq!(plain.card_bytes(nodes, granule()).expect("bytes"), 0);
        let plain_two = LoadCaptures::of(2, None, true, (1, (8, 8)));
        assert_eq!(plain_two.passes().expect("passes"), 3 + 1);
    }
}
