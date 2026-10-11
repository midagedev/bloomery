//! The plan's side of an image input, free of the encoder crates so a placement module every
//! build includes can take it: the tower's reserve on the stage card, entered into a placed plan's
//! machine and a whole-fit probe's by one function ([`reserve_on`]), a plan's refusal beside it
//! ([`refusal`]), and what a plan adds per position of a sequence for a media load
//! ([`table_bytes_per_position`]). The seat that owns the rest of the vision seat
//! (`crate::vision_seat`) re-exports these.

use std::fmt::Display;

use model::placement::Machine;

/// What the plan's reserve for the tower is called, as the plan's terms name it (beside
/// `workstation`'s draft and tier-batch reserves).
pub const RESERVE: &str = "image encoder";

/// `machine` with the tower's `reserve` on its stage card, when the seat is a stage reserve. The
/// one place the reserve enters a machine: a placed plan's builder and a whole-fit probe both call
/// it, so both fit the same bytes.
pub fn reserve_on(machine: &mut Machine, reserve: Option<u64>) {
    if let Some(bytes) = reserve {
        machine.cards[0].reserves.push((RESERVE.to_owned(), bytes));
    }
}

/// A machine builder (`layers` to its [`Machine`]) with [`reserve_on`] applied to what it builds.
pub fn with_reserve(
    plain: impl Fn(usize) -> Machine,
    reserve: Option<u64>,
) -> impl Fn(usize) -> Machine {
    move |layers| {
        let mut machine = plain(layers);
        reserve_on(&mut machine, reserve);
        machine
    }
}

/// A plan's refusal, beside the reserve that was in it: with no reserve the refusal as it is.
pub fn refusal(reserve: Option<u64>, e: impl Display) -> String {
    reserve.map_or_else(
        || e.to_string(),
        |bytes| {
            format!(
                "the plan holds the image encoder's {bytes} B as the stage card's {RESERVE} \
                 reserve: {e}"
            )
        },
    )
}

/// The bytes a plan adds per position of a sequence for a media load: the sequence's own rotation
/// table (`gpu::media::SeqRope`) holds a row of `width` f32 for each of its positions, `width`
/// being the f32 a rope row turns (64 for the Qwen3-VL family's text, 256 B a position).
#[must_use]
pub const fn table_bytes_per_position(width: usize) -> u64 {
    width as u64 * size_of::<f32>() as u64
}

#[cfg(test)]
mod tests {
    use model::placement::workstation::plan_a;

    use super::*;

    const BYTES: u64 = 1_476_000_000;

    /// The whole-fit probe and the placed plan take the one reserve: the machines both build carry
    /// the same named bytes, the card sets aside exactly that much more, and none changes nothing.
    #[test]
    fn the_probe_and_the_placed_plan_see_the_same_reserve() {
        let plain = plan_a(40);
        let mut probe = plan_a(40);
        reserve_on(&mut probe, Some(BYTES));
        let placed = with_reserve(plan_a, Some(BYTES))(40);
        assert_eq!(probe, placed);
        assert_eq!(
            probe.cards[0].reserves.last(),
            Some(&(RESERVE.to_owned(), BYTES))
        );
        assert_eq!(
            probe.cards[0].reserve_bytes(),
            plain.cards[0].reserve_bytes() + BYTES
        );
        assert_eq!(
            probe.cards[0].set_aside_bytes(),
            plain.cards[0].set_aside_bytes() + BYTES
        );
        let mut none = plan_a(40);
        reserve_on(&mut none, None);
        assert_eq!(none, plain);
        assert_eq!(with_reserve(plan_a, None)(40), plain);
        assert_eq!(refusal(None, "no fit"), "no fit");
        assert_eq!(
            refusal(Some(BYTES), "no fit"),
            "the plan holds the image encoder's 1476000000 B as the stage card's image encoder \
             reserve: no fit"
        );
    }

    /// The table term: a row of the rope's width in f32 a position (256 B at 64).
    #[test]
    fn the_table_term_is_a_row_of_f32_a_position() {
        assert_eq!(table_bytes_per_position(64), 256);
        assert_eq!(table_bytes_per_position(128), 512);
        assert_eq!(table_bytes_per_position(0), 0);
    }
}
