//! The MoE join's host rules for a model's routed slot count: which device
//! instance serves a count ([`Slots::of`]), and the combine of one output
//! value ([`combine`]) in the order the device sums it. A card launcher picks
//! its entry through [`Slots::of`]; a gate's host side holds an entry's
//! output to [`combine`] bit for bit.

use std::fmt;

/// The routed slot counts the join's device entries are built for: one
/// entry per count, picked at launch, and every other count refused by
/// name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slots {
    /// Six slots a token (DeepSeek-V4.1): the original entries.
    Six,
    /// Eight slots a token (GLM-5.3-Flash): the `_8` entries.
    Eight,
}

/// A slot count no entry serves ([`Slots::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotsRefused {
    pub n_used: usize,
}

impl fmt::Display for SlotsRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} routed slots a token: the join's entries are built for 6 and 8",
            self.n_used
        )
    }
}

impl std::error::Error for SlotsRefused {}

impl Slots {
    /// The instance for `n_used` slots a token, or the count refused.
    pub fn of(n_used: usize) -> Result<Slots, SlotsRefused> {
        match n_used {
            6 => Ok(Slots::Six),
            8 => Ok(Slots::Eight),
            _ => Err(SlotsRefused { n_used }),
        }
    }

    /// The instance's slot count.
    #[must_use]
    pub const fn n(self) -> usize {
        match self {
            Slots::Six => 6,
            Slots::Eight => 8,
        }
    }
}

/// The combine's card half: `acc = fma(down_j, w_j, acc)` from zero for each
/// slot `j` whose place is on the card (`card[j]`), in ascending `j`. A slot
/// the host serves is never read. This order is the gate.
#[must_use]
pub fn card_sum<const N: usize>(down: [f32; N], w: [f32; N], card: [bool; N]) -> f32 {
    let mut acc = 0.0f32;
    for ((d, wj), on) in down.into_iter().zip(w).zip(card) {
        if on {
            acc = d.mul_add(wj, acc);
        }
    }
    acc
}

/// The combine's join: the card sum, then the host's, then the shared
/// expert's, `(acc + hsum) + shexp`.
#[must_use]
pub fn join(acc: f32, hsum: f32, shexp: f32) -> f32 {
    (acc + hsum) + shexp
}

/// One output value's combine: [`card_sum`] of the card's slots, then
/// [`join`] with the host's partial sum and the shared expert's output.
#[must_use]
pub fn combine<const N: usize>(
    down: [f32; N],
    w: [f32; N],
    card: [bool; N],
    hsum: f32,
    shexp: f32,
) -> f32 {
    join(card_sum(down, w, card), hsum, shexp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_of_takes_six_and_eight_only() {
        for n in 0..=16 {
            match Slots::of(n) {
                Ok(s) => {
                    assert!(n == 6 || n == 8, "{n} slots picked an instance");
                    assert_eq!(s.n(), n);
                }
                Err(e) => {
                    assert!(n != 6 && n != 8, "{n} slots refused");
                    assert_eq!(e, SlotsRefused { n_used: n });
                    let text = e.to_string();
                    assert!(text.starts_with(&format!("{n} routed slots")), "{text}");
                }
            }
        }
        assert_eq!(Slots::of(6), Ok(Slots::Six));
        assert_eq!(Slots::of(8), Ok(Slots::Eight));
        for n in [5, 7, 9] {
            assert_eq!(Slots::of(n), Err(SlotsRefused { n_used: n }));
        }
    }

    /// Slots 0..8 ascending: `1e8 + 1` rounds back to `1e8` before `−1e8`
    /// cancels it, so the sum is 6.5; the slots in descending order, or the
    /// host slot's NaN read, give another value.
    #[test]
    fn card_sum_is_ascending_from_zero() {
        let down = [1.0e8, 1.0, -1.0e8, 1.0, 0.5, 3.0, f32::NAN, 2.0];
        let w = [1.0f32; 8];
        let card = [true, true, true, true, true, true, false, true];
        assert_eq!(card_sum(down, w, card).to_bits(), 6.5f32.to_bits());
        let rev = |a: [f32; 8]| {
            let mut r = a;
            r.reverse();
            r
        };
        let mut card_rev = card;
        card_rev.reverse();
        assert_ne!(
            card_sum(rev(down), rev(w), card_rev).to_bits(),
            6.5f32.to_bits()
        );
    }

    /// `a·a` with `a = 1 + 2⁻¹²` is `1 + 2⁻¹¹ + 2⁻²⁴`: a fused step keeps the
    /// `2⁻²⁴` against slot 0's `−(1 + 2⁻¹¹)`, a rounded product loses it.
    #[test]
    fn card_sum_fuses_each_step() {
        let a = 1.0f32 + f32::powi(2.0, -12);
        let down = [-1.0f32, a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = 1.0f32 + f32::powi(2.0, -11);
        let w = [b, a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let card = [true; 8];
        assert_eq!(
            card_sum(down, w, card).to_bits(),
            f32::powi(2.0, -24).to_bits()
        );
    }

    /// `(1 + 1e8) − 1e8` is 0 in f32; `1 + (1e8 − 1e8)` would be 1.
    #[test]
    fn combine_adds_host_then_shared() {
        let mut down = [0.0f32; 8];
        let mut card = [false; 8];
        down[3] = 1.0;
        card[3] = true;
        let w = [1.0f32; 8];
        assert_eq!(combine(down, w, card, 1.0e8, -1.0e8).to_bits(), 0);
        assert_eq!(join(1.0, 1.0e8, -1.0e8).to_bits(), 0);
    }

    /// A slot the host serves contributes nothing, whatever it holds.
    #[test]
    fn host_slots_are_never_read() {
        let down = [f32::NAN, 2.0, f32::INFINITY, 0.5, f32::NAN, 1.0];
        let w = [f32::NAN, 0.25, 1.0, 2.0, 1.0, f32::NAN];
        let card = [false, true, false, true, false, false];
        assert_eq!(card_sum(down, w, card).to_bits(), 1.5f32.to_bits());
        assert_eq!(card_sum([f32::NAN; 8], [1.0; 8], [false; 8]).to_bits(), 0);
    }
}
