//! A router's shape as its buffers and launches take it: the instance row a file's routed shape
//! selects, and the experts a token keeps.

use crate::ChainError;
use model::arch::models::shape::{self, MoeShape, RouterBody, RouterInst};

/// The most tokens one ubatch takes; its slots must also fit the slots one
/// route of the backend's grouped projection holds at the file's `top_k`
/// ([`RouterDims::ubatch`]). Also the default ubatch size: the one that reads
/// each weight the fewest times per prompt and gives each expert's GEMM the
/// most columns. Its owner is the placement's planned ubatch, the size a
/// default machine counts.
pub const UBATCH: usize = model::arch::qwen35moe::place::UBATCH_PLANNED as usize;

/// A router's shape as its buffers and launches take it: the instance row a
/// file's shape selected, and the experts a token keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterDims {
    inst: RouterInst,
    used: usize,
}

impl RouterDims {
    /// The dims of `shape`: its instance row, which must be one of the qwen3moe
    /// router bodies (`qwen3moe_router_*` or `gated`'s entries), and its
    /// `top_k`. A shape no row serves, or a row another body owns, is
    /// refused by name.
    pub fn of(shape: MoeShape) -> Result<RouterDims, ChainError> {
        const WHAT: &str = "qwen3moe::router::RouterDims::of";
        let inst =
            shape::select_router(shape).map_err(|e| ChainError::shape(WHAT, e.to_string()))?;
        if !matches!(inst.body, RouterBody::Qwen3moe | RouterBody::Qwen35moe) {
            return Err(ChainError::shape(
                WHAT,
                format!(
                    "{shape:?} selects {}, which this module does not run",
                    inst.at
                ),
            ));
        }
        Ok(RouterDims {
            inst,
            used: shape.top_k as usize,
        })
    }

    /// The instance row.
    #[must_use]
    pub fn inst(&self) -> RouterInst {
        self.inst
    }

    /// Experts the router softmaxes over.
    #[must_use]
    pub fn experts(&self) -> usize {
        self.inst.experts() as usize
    }

    /// Routed experts a token keeps.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Whether a shared expert's gate is the router's last row and its slot
    /// the token's last.
    #[must_use]
    pub fn gated(&self) -> bool {
        self.inst.rule.gated
    }

    /// Logits per token: one per router row.
    #[must_use]
    pub fn logits(&self) -> usize {
        self.experts() + usize::from(self.gated())
    }

    /// Slots (an id and a weight each) per token: the routed ones, then a
    /// gated shared expert's.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.used + usize::from(self.gated())
    }

    /// The most tokens a ubatch routing may hold ([`ubatch_at`] at this
    /// router's slots a token).
    #[must_use]
    pub fn ubatch(&self, max_slots: usize) -> usize {
        ubatch_at(self.slots(), max_slots)
    }
}

/// The most tokens a ubatch of `slots` slots a token may hold: at most
/// [`UBATCH`], and every slot of them within the `max_slots` one route of the
/// backend's grouped projection holds (a dense FFN's one slot a token too).
#[must_use]
pub fn ubatch_at(slots: usize, max_slots: usize) -> usize {
    UBATCH.min(max_slots / slots)
}

#[cfg(test)]
mod tests {
    use super::{UBATCH, ubatch_at};

    /// A ubatch holds at most [`UBATCH`] tokens and no more slots than one route holds: the
    /// planned ubatch at nine slots a token fills a route of `9 · UBATCH` slots, one slot less cuts
    /// a token, and a dense FFN's one slot a token keeps the cap.
    #[test]
    fn a_ubatch_fits_one_route_and_the_cap() {
        assert_eq!(ubatch_at(9, 9 * UBATCH), UBATCH);
        assert_eq!(ubatch_at(9, 9 * UBATCH - 1), UBATCH - 1);
        assert_eq!(ubatch_at(1, 9 * UBATCH), UBATCH);
    }
}
