//! Qwen3.8's card leg: which of a layer's routed experts the card serves.
//! Every walk — the captured step, the eager pass, the ubatch walk and the
//! verify — serves every routed expert on the host tier and has no card leg,
//! so a slot map that holds a routed expert on the card is refused by name
//! ([`refuse_card_map`]) at each walk's entry, before anything moves, where
//! a walk that ran it would leave that expert's contribution out of the sum.

use crate::GpuError;
use crate::hybrid::SlotMap;

/// What the refusal names.
const WHAT: &str = "qwen4exp card leg";

/// Refused by name when `map` holds a routed expert on the card: the error
/// names `walk`, the first such layer and its count. Reads one count a
/// layer and allocates nothing unless it refuses.
pub(super) fn refuse_card_map(map: &SlotMap, walk: &str) -> Result<(), GpuError> {
    for l in map.layers() {
        for (on, device) in [
            (map.on_card(l)?, "the card"),
            (map.on_tier(l)?, "the tier card"),
        ] {
            if on > 0 {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the {walk} walk has no card leg yet: layer {l} holds {on} routed experts \
                         on {device}"
                    ),
                ));
            }
        }
    }
    Ok(())
}
