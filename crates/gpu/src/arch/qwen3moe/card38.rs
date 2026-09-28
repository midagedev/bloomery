//! Qwen3.8's card leg: which of a layer's routed experts the card serves.
//! Every walk — the captured step, the eager pass, the ubatch walk and the
//! verify — serves every routed expert on the host tier and has no card leg,
//! so a slot map that holds a routed expert on the card is refused by name
//! ([`CardLeg::refuse`]) at each walk's entry, before anything moves, where
//! a walk that ran it would leave that expert's contribution out of the sum.
//! The map's answer ([`CardLeg::of`]) is read when the map is set — at the
//! load, when a gate plants one and at reset — so a walk reads one field.

use crate::GpuError;
use crate::hybrid::SlotMap;

/// What the refusal names.
const WHAT: &str = "qwen4exp card leg";

/// The first layer of a slot map that holds routed experts on a card: the
/// layer, how many, and which card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CardLeg {
    layer: usize,
    on: usize,
    device: &'static str,
}

impl CardLeg {
    /// The first layer of `map` with a routed expert on the card or on the
    /// tier card; `None` when every routed expert is the host's. Reads one
    /// count a layer.
    pub(super) fn of(map: &SlotMap) -> Result<Option<CardLeg>, GpuError> {
        for l in map.layers() {
            for (on, device) in [
                (map.on_card(l)?, "the card"),
                (map.on_tier(l)?, "the tier card"),
            ] {
                if on > 0 {
                    return Ok(Some(CardLeg {
                        layer: l,
                        on,
                        device,
                    }));
                }
            }
        }
        Ok(None)
    }

    /// Refused by name when `leg` names a layer with a card leg: the error
    /// names `walk`, the layer and its count. Allocates nothing unless it
    /// refuses.
    pub(super) fn refuse(leg: Option<CardLeg>, walk: &str) -> Result<(), GpuError> {
        match leg {
            None => Ok(()),
            Some(CardLeg { layer, on, device }) => Err(GpuError::shape(
                WHAT,
                format!(
                    "the {walk} walk has no card leg yet: layer {layer} holds {on} routed experts \
                     on {device}"
                ),
            )),
        }
    }
}
