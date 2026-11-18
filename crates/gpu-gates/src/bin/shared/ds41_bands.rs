//! The bands the V4.1 fixture's walks between two placements are held to: the clarity margin and the
//! pins of the pair the tier gate's rank walk (B10) and the NVMe-tier gate's rank clause read. The
//! machinery they feed (`Pins`, `walk_pinned`) is `bloomery_gpu_gates::logits_band`'s; the real
//! tier's margin is `bloomery_gpu_gates::prompts::PLACE_MARGIN_V41`.

use bloomery_gpu_gates::logits_band::{LOGITS_CLEAR_PIN, Pins};
use bloomery_gpu_gates::prompts::PLACE_MARGIN_V41;
use bloomery_gpu_gates::tier::Tier;

/// [`PLACE_MARGIN_V41`] on the V4.1 fixture, by its method: the fixture's two placements move
/// the top-two gap by the pair's own distance, which is no card's (the walk's RMS |Δlogit|
/// reads 0.2510 on the 3090 and the A6000, 0.2479 on an RTX 5090), and
/// [`bloomery_gpu_gates::prompts::MARGIN_FLOOR`] sits under that shift. On the fixture the tier
/// gate's rank walk (B10) and the NVMe-tier gate's rank clause are one pair: both feed the same 64 prompt ids against the plan without the
/// rank, and of what the rank moves only the card's list moves bits (a host and an NVMe-paged
/// expert run one host kernel, the NVMe-tier gate's bits clause; a tiered expert is the card's,
/// B10's T1), so the two walks read the same shifts. In the
/// error model's terms the floor is 4.8 σ of a gap's shift (σ = √2 × 0.251, the two logits'
/// RMS distance), where a top-1 of any runner-up flips with probability under 1e-6 a position;
/// the walk's band is [`PLACE_PINS_V41_FIXTURE`], both read through [`place_v41`].
// PIN(2026-11-18): 2 × 0.8469, the largest |top-2 gap shift| over B10's 64 fixture positions
// (position 41 on the 3090 and the A6000; 0.6932 at position 55 on the 5090), rounded up to 1.7.
// The NVMe-tier rank walk reads the same 0.8469 at 41 (sm_86, 66 positions) and 0.6932 at 55
// (5090, 64). The 5090's runs flipped a top-1 at margin 0.6098 at position 20 in both walks (a
// third id over the reference's two), a re-lottery under this floor; at the old floor, 0.5,
// both were red with the same distance.
pub const PLACE_MARGIN_V41_FIXTURE: f32 = 1.7;

/// The band the V4.1 fixture's walk between two placements is held to, with
/// [`PLACE_MARGIN_V41_FIXTURE`] as its clarity margin ([`place_v41`]): the tier
/// gate's rank walk (B10) and the NVMe-tier gate's rank clause, one pair on the fixture — the
/// router rank's card list against the id prefix, the same 64 prompt ids fed one step a
/// position — whose readings agree to the digit on every card. The two plans run other experts
/// on the card and on the host, and the distance is the pair's, not the card's (the RMS
/// |Δlogit| reads 0.2510 on the 3090 and the A6000, 0.2479 on an RTX 5090). Twice the larger
/// reading of each, [`bloomery_gpu_gates::logits_band::LOGITS_MAX_PIN`]'s and
/// [`bloomery_gpu_gates::logits_band::LOGITS_RMS_PIN`]'s method on this pair alone;
/// the shared pins carry the real tier's distance and sit ten times over this one.
// PIN(2026-11-18): max 2 × 2.597 (RTX 5090; 2.338 on sm_86) rounded up to 5.2, rms 2 × 0.2510
// (sm_86; 0.2479 on the 5090) rounded up to 0.51, both B10's line over its 64 positions; the
// NVMe-tier rank walk reads the same max and 0.2485 (sm_86, 66 positions) and 0.2479 (5090, 64).
// FAIL-first: every layer's ranked host sum zeroed moves the real tier's rms 1.03 past 2.5, so
// it is red here at 2.4 × 0.25 = 0.6 > 0.51.
pub const PLACE_PINS_V41_FIXTURE: Pins = Pins {
    clear: LOGITS_CLEAR_PIN,
    max: 5.2,
    rms: 0.51,
};

/// The clarity margin and band a V4.1 walk between two placements reads in `tier`: the real
/// tier's [`PLACE_MARGIN_V41`] in [`Pins::SHARED`], the fixture's
/// [`PLACE_MARGIN_V41_FIXTURE`] in [`PLACE_PINS_V41_FIXTURE`]. The one choice
/// the tier gate's rank walk and the NVMe-tier gate's rank clause both make.
pub fn place_v41(tier: Tier) -> (f32, Pins) {
    match tier {
        Tier::Real => (PLACE_MARGIN_V41, Pins::SHARED),
        Tier::Fixture => (PLACE_MARGIN_V41_FIXTURE, PLACE_PINS_V41_FIXTURE),
    }
}
