//! The card-byte invariant: what a run makes on a card never passes what its
//! plan counts there.
//!
//! A run's made bytes on a card are the device's free reading taken when the
//! card's context opened, before the load's first allocation, less a reading
//! taken now ([`CardMade::made`]). They are held to the card's terms
//! ([`CardTotals::terms`]) less the part of the context term the opening
//! reading already carries: that reading is taken on the open context, so the
//! context's own creation cost ([`workstation::CONTEXT_SELF`], at most the
//! card's whole context term) was paid before it and is never in the made
//! bytes, while the rest of the context term (the bundles' module images, the
//! kernels' local-memory growth, a capture's graph) is. There is no slack term:
//! a byte the plan does not count is the finding. The comparison is checked
//! after the load, after the first prompt call and after the first step
//! ([`CardPhase`]).
//!
//! The readings are the device's, every process's bytes in them, so the
//! comparison holds on a card no other process holds. A card the plan's census
//! found held by another process ([`Card::held_by`]), or an integrated device
//! (whose reading is the host's room), is not measured: the reading says so by
//! name ([`CardReading::Unmeasured`]) and is never a pass.

use std::fmt;

use super::workstation::CONTEXT_SELF;
use super::{Card, CardTotals};

/// When a run's card bytes are checked against its plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardPhase {
    /// The load done: weights, body, head, nothing captured or run.
    Load,
    /// The first prompt call done.
    FirstPrompt,
    /// The first step done.
    FirstStep,
}

impl CardPhase {
    /// The word records and refusals name the phase by.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            CardPhase::Load => "load",
            CardPhase::FirstPrompt => "first-prompt",
            CardPhase::FirstStep => "first-step",
        }
    }
}

impl fmt::Display for CardPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.word())
    }
}

/// What card `card` of a plan made by `phase`, against what the plan counts
/// there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardMade {
    /// The card's index among the plan's cards.
    pub card: usize,
    /// The name the plan gives the card.
    pub name: String,
    pub phase: CardPhase,
    /// The device's free bytes when the card's context opened.
    pub before: u64,
    /// The device's free bytes at `phase`.
    pub now: u64,
    /// The plan's terms for the card ([`CardTotals::terms`]).
    pub terms: [(&'static str, u64); 8],
    /// The part of the context term the opening reading already carried.
    pub in_reading: u64,
}

impl CardMade {
    /// Card `card` of a plan, `spec` its machine card and `totals` its totals,
    /// read `before` free bytes at its open and `now` at `phase`.
    #[must_use]
    pub fn of(
        card: usize,
        spec: &Card,
        totals: &CardTotals,
        phase: CardPhase,
        before: u64,
        now: u64,
    ) -> CardMade {
        CardMade {
            card,
            name: spec.name.clone(),
            phase,
            before,
            now,
            terms: totals.terms(),
            in_reading: totals.context_bytes.min(CONTEXT_SELF),
        }
    }

    /// The bytes the run made on the device since its context opened; 0 when
    /// the device has more free than then (the reading shows both).
    #[must_use]
    pub fn made(&self) -> u64 {
        self.before.saturating_sub(self.now)
    }

    /// The bytes the plan counts on the card past the opening reading: every
    /// term, less the context's part that reading carried. A sum past u64
    /// reads `u64::MAX`, as [`CardTotals::bytes`] does.
    #[must_use]
    pub fn planned(&self) -> u64 {
        let sum: u128 = self.terms.iter().map(|&(_, b)| u128::from(b)).sum();
        u64::try_from(sum)
            .unwrap_or(u64::MAX)
            .saturating_sub(self.in_reading)
    }

    /// The bytes the run made past the plan's count; `None` within it.
    #[must_use]
    pub fn past(&self) -> Option<u64> {
        let (made, planned) = (self.made(), self.planned());
        (made > planned).then(|| made - planned)
    }

    /// The plan's terms as one `name=bytes` list, in [`CardTotals::terms`]'s
    /// order.
    #[must_use]
    pub fn terms_text(&self) -> String {
        self.terms
            .iter()
            .map(|(name, b)| format!("{name}={b}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// A card a device-wide reading cannot measure, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardUnmeasured {
    pub card: usize,
    pub name: String,
    pub phase: CardPhase,
    pub why: String,
}

/// Why card `spec` cannot be measured by a device-wide reading: another
/// process held it when the plan's census read it, or it is an integrated
/// device, whose reading is the host's room. `None` when it can.
#[must_use]
pub fn unmeasured_why(spec: &Card, integrated: bool) -> Option<String> {
    if integrated {
        return Some(
            "an integrated device: its free reading is the host's room, not the card's".to_owned(),
        );
    }
    spec.held_by.map(|held| {
        format!("other processes hold the device ({held}), and the reading is the device's")
    })
}

/// One check of a card's bytes: what it made against its plan, or why it
/// could not be measured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardReading {
    Made(CardMade),
    Unmeasured(CardUnmeasured),
}

impl CardReading {
    /// The refusal of a run where the card is held: a card that made bytes
    /// past its plan, or one that could not be measured. `Ok` only for a
    /// card measured within its plan.
    pub fn check(&self) -> Result<(), CardRefusal> {
        match self {
            CardReading::Made(m) if m.past().is_none() => Ok(()),
            CardReading::Made(m) => Err(CardRefusal::Past(m.clone())),
            CardReading::Unmeasured(u) => Err(CardRefusal::Unmeasured(u.clone())),
        }
    }

    /// The phase the reading was taken at.
    #[must_use]
    pub fn phase(&self) -> CardPhase {
        match self {
            CardReading::Made(m) => m.phase,
            CardReading::Unmeasured(u) => u.phase,
        }
    }
}

/// A run refused by the card-byte invariant ([`CardReading::check`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardRefusal {
    /// The card made bytes past its plan.
    Past(CardMade),
    /// The card could not be measured.
    Unmeasured(CardUnmeasured),
}

impl fmt::Display for CardRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CardRefusal::Past(m) => write!(
                f,
                "card {} ({}) made {} device bytes by {} (free {} at its open, {} now), past the \
                 {} its plan counts by {}: {}, less {} of the context term the opening reading \
                 carries",
                m.card,
                m.name,
                m.made(),
                m.phase,
                m.before,
                m.now,
                m.planned(),
                m.past().unwrap_or(0),
                m.terms_text(),
                m.in_reading,
            ),
            CardRefusal::Unmeasured(u) => write!(
                f,
                "card {} ({}) at {}: its bytes are not measured against its plan: {}",
                u.card, u.name, u.phase, u.why
            ),
        }
    }
}

impl std::error::Error for CardRefusal {}

#[cfg(test)]
mod tests {
    use super::{CardMade, CardPhase, CardReading, CardRefusal, CardTotals, unmeasured_why};
    use crate::placement::workstation::{CONTEXT_SELF, RTX_3090, holder_card};

    const MIB: u64 = 1 << 20;

    fn zero() -> CardTotals {
        CardTotals {
            dense_bytes: 0,
            expert_bytes: 0,
            rounding_bytes: 0,
            experts: 0,
            kv_bytes: 0,
            stage_rows_bytes: 0,
            scratch_bytes: 0,
            context_bytes: 0,
            reserve_bytes: 0,
            headroom_bytes: 0,
        }
    }

    /// Every term at a distinct size, the context term the workstation's.
    fn full() -> CardTotals {
        CardTotals {
            dense_bytes: 1000 * MIB,
            expert_bytes: 2000 * MIB,
            rounding_bytes: 3 * MIB,
            experts: 7,
            kv_bytes: 400 * MIB,
            stage_rows_bytes: 5 * MIB,
            scratch_bytes: 64 * MIB,
            context_bytes: 512 * MIB,
            reserve_bytes: 70 * MIB,
            headroom_bytes: 1,
        }
    }

    fn made(t: &CardTotals, bytes: u64) -> CardMade {
        let free = 20_000 * MIB;
        CardMade::of(
            0,
            &holder_card(RTX_3090),
            t,
            CardPhase::Load,
            free,
            free - bytes,
        )
    }

    /// The plan's count is every term less the context's own creation cost:
    /// made bytes at the count pass, a byte more is refused, fewer pass. The
    /// edge case refuses a `>=` comparison; the one-past case and the
    /// fewer case refuse reversed operands.
    #[test]
    fn made_bytes_pass_up_to_the_plans_count_and_no_byte_more() {
        let t = full();
        let count = t.bytes() - CONTEXT_SELF;
        assert_eq!(made(&t, count).planned(), count);
        assert_eq!(made(&t, count).past(), None, "made exactly the count");
        assert_eq!(
            made(&t, count + 1).past(),
            Some(1),
            "one byte past the count"
        );
        assert_eq!(made(&t, count / 2).past(), None, "made half the count");
        assert_eq!(made(&t, 0).past(), None, "made nothing");
    }

    /// Each term alone counts in full: a plan of one term passes made bytes
    /// of that term and refuses one byte more, so no term is left out of the
    /// sum. The context term alone counts only past the reading's part.
    #[test]
    fn every_term_counts() {
        let names = full().terms().map(|(name, _)| name);
        for (i, name) in names.into_iter().enumerate() {
            let mut t = zero();
            let bytes = 600 * MIB;
            match i {
                0 => t.dense_bytes = bytes,
                1 => t.expert_bytes = bytes,
                2 => t.rounding_bytes = bytes,
                3 => t.kv_bytes = bytes,
                4 => t.stage_rows_bytes = bytes,
                5 => t.scratch_bytes = bytes,
                6 => t.context_bytes = bytes,
                7 => t.reserve_bytes = bytes,
                _ => unreachable!("eight terms"),
            }
            let count = if name == "context" {
                bytes - CONTEXT_SELF
            } else {
                bytes
            };
            assert_eq!(made(&t, count).past(), None, "{name} alone, made its count");
            assert_eq!(
                made(&t, count + 1).past(),
                Some(1),
                "{name} alone, one byte past"
            );
        }
    }

    /// A context term under the creation cost counts nothing past the
    /// reading: the reading's part is at most the whole term.
    #[test]
    fn a_small_context_term_counts_nothing_past_the_reading() {
        let mut t = zero();
        t.context_bytes = 100 * MIB;
        assert_eq!(made(&t, 0).in_reading, 100 * MIB);
        assert_eq!(made(&t, 0).past(), None);
        assert_eq!(made(&t, 1).past(), Some(1));
    }

    /// A device with more free than at the open made nothing, and the
    /// reading keeps both readings.
    #[test]
    fn more_free_than_at_the_open_made_nothing() {
        let t = zero();
        let m = CardMade::of(0, &holder_card(RTX_3090), &t, CardPhase::FirstStep, 10, 20);
        assert_eq!(m.made(), 0);
        assert_eq!((m.before, m.now), (10, 20));
        assert_eq!(CardReading::Made(m).check(), Ok(()));
    }

    /// The refusal names the card, the phase, the made bytes, both readings,
    /// the count, the excess and every term with its bytes.
    #[test]
    fn the_refusal_names_every_term() {
        let t = full();
        let count = t.bytes() - CONTEXT_SELF;
        let mut m = made(&t, count + 3);
        m.phase = CardPhase::FirstPrompt;
        let Err(e) = CardReading::Made(m).check() else {
            panic!("three bytes past the count passed");
        };
        let text = e.to_string();
        for (name, bytes) in t.terms() {
            assert!(
                text.contains(&format!("{name}={bytes}")),
                "{name} in {text}"
            );
        }
        for want in [
            "card 0 (3090)".to_owned(),
            format!("made {} device bytes by first-prompt", count + 3),
            format!("past the {count} its plan counts by 3"),
            format!("less {CONTEXT_SELF} of the context term"),
        ] {
            assert!(text.contains(&want), "{want:?} in {text}");
        }
    }

    /// A card another process held at plan time, or an integrated device, is
    /// not measured, named, and refused where the card is held.
    #[test]
    fn a_held_or_integrated_card_is_not_measured_by_name() {
        let mut card = holder_card(RTX_3090);
        assert_eq!(unmeasured_why(&card, false), None);
        let integrated = unmeasured_why(&card, true).expect("integrated is unmeasured");
        assert!(integrated.contains("integrated"), "{integrated}");
        card.held_by = Some("pid 7 (977 MiB)");
        let held = unmeasured_why(&card, false).expect("a held card is unmeasured");
        assert!(held.contains("pid 7 (977 MiB)"), "{held}");
        let r = CardReading::Unmeasured(super::CardUnmeasured {
            card: 1,
            name: card.name.clone(),
            phase: CardPhase::Load,
            why: held,
        });
        let Err(e @ CardRefusal::Unmeasured(_)) = r.check() else {
            panic!("an unmeasured card passed");
        };
        assert!(e.to_string().contains("card 1 (3090) at load"), "{e}");
    }
}
