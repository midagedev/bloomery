//! The host leg of a step walk as the schedules see it ([`StepLeg`]): the
//! [`Port`] of kind [`PortKind::Step`] over a tier's step port, one for every
//! architecture whose layer program hands its routed experts to the host
//! tier. A program reaches the tier through the leg ([`StepLeg::hybrid`]) for
//! its go, its wait and its [`HostTier::row_enqueued`].

use cuda_core::CudaStream;
use runtime::sched::{Overlap, Port, PortKind, Refused};

use super::{HostExperts, HostTier};
use crate::GpuError;

/// What a walk the leg refuses names.
const WHAT: &str = "host StepLeg (the host tier's step port)";

/// A step walk's host leg: the stream its chain is enqueued on and the tier
/// that serves the chain's routed experts.
pub struct StepLeg<'a, H: HostExperts> {
    stream: &'a CudaStream,
    hybrid: &'a mut HostTier<H>,
}

impl<'a, H: HostExperts> StepLeg<'a, H> {
    /// The leg of a walk enqueued on `stream` whose host legs `hybrid` serves.
    pub fn new(stream: &'a CudaStream, hybrid: &'a mut HostTier<H>) -> StepLeg<'a, H> {
        StepLeg { stream, hybrid }
    }

    /// The tier, for a layer program's go, wait and enqueue notice.
    pub fn hybrid(&mut self) -> &mut HostTier<H> {
        self.hybrid
    }

    /// The stream the walk is enqueued on.
    pub fn stream(&self) -> &'a CudaStream {
        self.stream
    }
}

impl<H: HostExperts> Port for StepLeg<'_, H> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Step;

    /// Opens the tier's step port on the point's rows
    /// ([`HostTier::open_step`]): one row of one column is the step, two rows
    /// the pair pass; any other point is refused by name there.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        self.hybrid.open_step(self.stream, o.units, o.cols)
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT,
            detail: why.to_string(),
        }
    }
}
