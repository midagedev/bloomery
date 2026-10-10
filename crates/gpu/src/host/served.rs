//! The [`HostServed`](crate::model::HostServed) every hybrid body shares: a
//! body implements [`TierBody`] — its host tier, its residency machine's
//! parts handed over as one value, and its own step around the tier's
//! service — and the blanket impl below holds the forwards the bodies wrote
//! by hand, once. A body that runs no residency machine returns `None` from
//! [`TierBody::residency_parts`] and the machine's calls keep
//! [`HostServed`]'s defaults.

use std::sync::Arc;

use cuda_core::CudaStream;
use runtime::swaprule::KeptRows;

use crate::Gpu;
use crate::GpuError;
use crate::fault::read_cards;
use crate::host::HostExperts;
use crate::host::PassKind;
use crate::host::swap::{BoundaryAt, ResetReport};
use crate::host::swap_source::ResidencyGlue;
use crate::hybrid::{Chain, HostResidency, Hybrid, Refusal};
use crate::model::HostServed;
use crate::tensor::DeviceTensor;

/// A body whose host tier is a [`Hybrid`]: what the shared [`HostServed`]
/// impl needs of it — the tier, the residency machine's parts, and the body's
/// own step around the tier's service, which stays in the body.
pub trait TierBody {
    /// The tier's host experts: the architecture's host run.
    type Experts: HostExperts;

    /// The body's host tier.
    #[must_use]
    fn hybrid(&self) -> &Hybrid<Self::Experts>;

    /// The body's host tier, mutable.
    fn hybrid_mut(&mut self) -> &mut Hybrid<Self::Experts>;

    /// The body's residency machine's parts — the glue, the tier, the slot
    /// map's card copy — as one value of disjoint borrows; `None` for a body
    /// that runs no machine.
    #[must_use]
    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, Self::Experts>>;

    /// The body's own step around the tier's service of the chain `chain` a
    /// graph replay just submitted ([`HostServed::serve_captured`]'s body):
    /// the architecture's part of the step, before or around
    /// [`Hybrid::serve_captured_of`], stays here.
    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError>;
}

/// The parts [`TierBody::residency_parts`] hands the shared impl: disjoint
/// borrows of the body, so the machine's calls go through them together.
pub struct ResidencyParts<'a, E: HostExperts> {
    /// The machine's glue, which drives and logs it.
    pub glue: &'a mut ResidencyGlue,
    /// The host tier the machine runs over.
    pub hybrid: &'a mut Hybrid<E>,
    /// The slot map's card copy, a second reference of the body's.
    pub slots: &'a Arc<DeviceTensor<u32>>,
}

impl<B: TierBody> HostServed for B {
    /// The body's own step around the tier's service
    /// ([`TierBody::serve_chain`]).
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.serve_chain(chain)
    }

    /// The tier's note on `e` ([`Hybrid::noted`]).
    fn noted(&self, e: GpuError) -> GpuError {
        self.hybrid().noted(e)
    }

    /// The tier's refusal that failed the step's service, once
    /// ([`Hybrid::take_step_refusal`]).
    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid_mut().take_step_refusal()
    }

    /// The tier's refusal poison lifted ([`Hybrid::lift_refusal`]) — the
    /// settling a reset runs is [`Hybrid::settle`]'s, the body's reset.
    fn lift_refusal(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid_mut().lift_refusal(stream)
    }

    /// The refusal the tier is poisoned by now ([`Hybrid::refusal_poison`]).
    fn refusal_poison(&self) -> Option<Refusal> {
        self.hybrid().refusal_poison()
    }

    /// The host set the placed load read in and locked, which the tier holds
    /// ([`Hybrid::residency`]).
    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid().residency()
    }

    /// The machine's boundary at `at` ([`ResidencyGlue::at_boundary`]), its
    /// report logged when a binary asked.
    fn at_boundary(&mut self, stream: &CudaStream, at: BoundaryAt) -> Result<(), GpuError> {
        match self.residency_parts() {
            Some(ResidencyParts { glue, hybrid, .. }) => glue.at_boundary(hybrid, stream, at),
            None => Ok(()),
        }
    }

    /// The pass the last boundary opened keeping `kept` rows
    /// ([`ResidencyGlue::keep_rows`]).
    fn keep_rows(&mut self, kept: KeptRows, kind: PassKind) -> Result<(), GpuError> {
        match self.residency_parts() {
            Some(ResidencyParts { glue, hybrid, .. }) => glue.keep_rows(hybrid, kept, kind),
            None => Ok(()),
        }
    }

    /// The machine reset to its seed ([`ResidencyGlue::reset`]).
    fn residency_reset(&mut self, stream: &CudaStream) -> Result<Option<ResetReport>, GpuError> {
        match self.residency_parts() {
            Some(ResidencyParts { glue, hybrid, .. }) => glue.reset(hybrid, stream),
            None => Ok(None),
        }
    }

    /// The machine stopped before anything of the body frees
    /// ([`ResidencyGlue::stop`]).
    fn stop_residency(&mut self) {
        if let Some(ResidencyParts { glue, hybrid, .. }) = self.residency_parts() {
            glue.stop(hybrid);
        }
    }

    /// The machine started once the body's pieces are sized
    /// ([`ResidencyGlue::start`]).
    fn start_residency(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        match self.residency_parts() {
            Some(ResidencyParts {
                glue,
                hybrid,
                slots,
            }) => glue.start(hybrid, gpu.context(), gpu.stream(), Arc::clone(slots)),
            None => Ok(()),
        }
    }
}

/// The fault word after a prompt group's walk on a body with a host tier: a
/// fault the group raised on the expert tier is the call's error, as the
/// stage card's is (the first layer wins), named `what`; `Ok(false)` for a
/// group with none. The last group's stage word rides the head's readback
/// ([`crate::GpuModel::run_rows`]), so with `last` it returns that the head
/// was enqueued and reads only the tier's word; an inner group reads both
/// words here.
pub fn read_fault<E: HostExperts>(
    what: &'static str,
    gpu: &Gpu,
    hybrid: &mut Hybrid<E>,
    last: bool,
) -> Result<bool, GpuError> {
    let tier = hybrid.tier_fault()?;
    if last {
        return match tier {
            Some(t) => Err(GpuError::fault(
                what,
                read_cards(&[gpu.fault()?, Some(t)]).unwrap_or(t),
            )),
            None => Ok(true),
        };
    }
    match read_cards(&[gpu.fault()?, tier]) {
        Some(fault) => Err(GpuError::fault(what, fault)),
        None => Ok(false),
    }
}
