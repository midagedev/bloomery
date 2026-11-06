//! Kimi K3's body behind the traits the common hybrid seat serves it by: [`Rollback`], which takes
//! no position back, and [`TierBody`], which is the placed load's host tier.

use super::body::Body;
use crate::GpuError;
use crate::host::run::HostRun;
use crate::host::served::{ResidencyParts, TierBody};
use crate::hybrid::{Chain, Hybrid};
use crate::model::{ChainBody, Rollback};

const ROLLBACK: &str = "kimi_k3::Body::rollback";
const SERVE: &str = "kimi_k3::Body::serve_chain";

/// What the tier's accessors name when the load is the whole card's.
const WHOLE_CARD: &str = "kimi_k3::Body as a TierBody: the whole-card load (GpuModel::open) has no \
                          host tier; the hybrid seat loads by a plan (GpuModel::open_placed)";

impl Rollback for Body {
    /// Refused by name, always: the delta layers' state holds the position it stands at alone, so
    /// the body takes no position back. The session's cut keeps every held position or nothing
    /// (a reset) and never asks for another.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        Err(GpuError::shape(
            ROLLBACK,
            format!("a rollback to position {pos}: the delta layers' state takes no position back"),
        ))
    }
}

/// The placed load's host tier, which is [`Placed`](super::Placed)'s. The tier is the placed
/// load's alone, so a whole-card body (`GpuModel::open`) has none: `hybrid` and `hybrid_mut` panic
/// by name there, `residency_parts` answers `None` and `serve_chain` is a named error. The
/// hybrid seat always loads by a plan, so it holds a placed body.
impl TierBody for Body {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.placed().expect(WHOLE_CARD).hybrid()
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.host().expect(WHOLE_CARD).hybrid_mut()
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.host().and_then(|placed| placed.residency_parts())
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        match self.host() {
            Some(placed) => placed.serve_chain(chain),
            None => Err(GpuError::shape(SERVE, WHOLE_CARD)),
        }
    }
}
