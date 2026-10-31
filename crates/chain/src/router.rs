//! A router's shape as its buffers and launches take it: the instance row a file's routed shape
//! selects, and the experts a token keeps.

use crate::ChainError;
use model::arch::models::shape::{self, MoeShape, RouterBody, RouterInst};

/// A router's shape as its buffers and launches take it: the instance row a
/// file's shape selected, and the experts a token keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterDims {
    inst: RouterInst,
    used: usize,
}

impl RouterDims {
    /// The dims of `shape`: its instance row, which must be one of the softmax
    /// router bodies (`qwen3moe_router_*` or `gated`'s entries), and its
    /// `top_k`. A shape no row serves, or a row another body owns, is
    /// refused by name.
    pub fn of(shape: MoeShape) -> Result<RouterDims, ChainError> {
        const WHAT: &str = "router::RouterDims::of";
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
}
