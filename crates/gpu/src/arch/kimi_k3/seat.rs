//! Kimi K3's body behind the traits the common hybrid seat serves it by: [`Rollback`], which takes
//! a position back to a checkpoint, and [`TierBody`], which is the placed load's host tier.

use super::body::Body;
use crate::GpuError;
use crate::host::run::HostRun;
use crate::host::served::{ResidencyParts, TierBody};
use crate::hybrid::{Chain, Hybrid};
use crate::model::{ChainBody, Rollback};

const SERVE: &str = "kimi_k3::Body::serve_chain";

/// What the tier's accessors name when the load is the whole card's.
const WHOLE_CARD: &str = "kimi_k3::Body as a TierBody: the whole-card load (GpuModel::open) has no \
                          host tier; the hybrid seat loads by a plan (GpuModel::open_placed)";

impl Rollback for Body {
    /// A cut to `pos` ([`Body::cut`]): a position where a checkpoint stands, refused by name at
    /// any other (the delta layers' state holds the position it stands at alone); the card work
    /// waits for the next device call.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.cut(pos)
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
