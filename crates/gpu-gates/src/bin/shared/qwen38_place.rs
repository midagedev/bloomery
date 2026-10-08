//! What a Qwen3.8 (qwen4exp) binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the family's input to the common
//! unset `--place` rule. `generate_qwen3moe` and the Qwen3.8 serve seat read
//! it here.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{Place, PlaceWhy, TierRule};
use model::placement::workstation::DeviceInfo;

/// The Qwen3.8 family's input to the common unset rule (`Place::choose`):
/// one expert tier card served, no break-even yet — no sitting has shown
/// the tier not slower (docs/cards/q38bpbug-ab.card holds the measured
/// rates a break-even would be derived from).
pub const Q38_RULE: TierRule = TierRule {
    tiers: 1,
    break_even: None,
    basis: "docs/cards/q38bpbug-ab.card",
};

/// The placement a Qwen3.8 binary's `--place` names, and why, on `census`
/// (`Place::choose` by [`Q38_RULE`]): `flag` as given when set, else `a`, the
/// plan never asked. The caller prints [`PlaceWhy::record`].
pub fn choose(flag: Option<Place>, census: &[DeviceInfo]) -> Result<PlaceWhy, GateError> {
    Place::choose(flag, census, Q38_RULE, |p| {
        Err(format!(
            "--place unset: Qwen3.8 has no break-even yet, so no tier count for {}",
            p.name()
        )
        .into())
    })
}
