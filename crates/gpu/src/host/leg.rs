//! The host legs of a walk as the schedules see them: [`StepLeg`], the
//! [`Port`] of kind [`PortKind::Step`] over a tier's step port, and
//! [`BatchLeg`], the [`Port`] of kind [`PortKind::Batch`] over its batch port —
//! one of each for every architecture whose layer program hands its routed
//! experts to the host tier. A program reaches the tier through the leg
//! ([`StepLeg::hybrid`], [`BatchLeg::hybrid`]): a step walk for its go, its
//! wait and its [`HostTier::row_enqueued`]; an eager batch walk for its
//! download, its serve and its upload.

use cuda_core::{CudaStream, DeviceBuffer};
use runtime::sched::{At, Overlap, Port, PortKind, Refused};

use super::batch::BatchKey;
use super::{HostExperts, HostTier};
use crate::GpuError;

/// What a walk the leg refuses names.
const WHAT: &str = "host StepLeg (the host tier's step port)";
/// What a walk the batch leg refuses names.
const WHAT_BATCH: &str = "host BatchLeg (the host tier's batch port)";

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

/// An eager batch walk's host leg: the stream its layers are enqueued on, the
/// tier whose batch port serves their routed experts, the card buffer each
/// layer's host sums land in, and the most columns a unit may hold (the
/// port's sets and the host's union were made for them). Unit `u` of layer
/// `l` over `cols` columns is the key `(l, u, 0 .. cols)` ([`BatchLeg::key`]):
/// the program's download names it, the walk's serve serves it and uploads
/// its sums.
pub struct BatchLeg<'a, H: HostExperts> {
    stream: &'a CudaStream,
    hybrid: &'a mut HostTier<H>,
    hsum: &'a mut DeviceBuffer<f32>,
    cap: usize,
    /// The open walk's columns.
    cols: usize,
}

impl<'a, H: HostExperts> BatchLeg<'a, H> {
    /// The leg of a walk enqueued on `stream` whose host legs `hybrid`'s batch
    /// port serves into `hsum`, for units of up to `cap` columns
    /// ([`HostTier::prepare_batch`] made its sets for `cap`).
    pub fn new(
        stream: &'a CudaStream,
        hybrid: &'a mut HostTier<H>,
        hsum: &'a mut DeviceBuffer<f32>,
        cap: usize,
    ) -> BatchLeg<'a, H> {
        BatchLeg {
            stream,
            hybrid,
            hsum,
            cap,
            cols: 0,
        }
    }

    /// The tier, for a layer program's download.
    pub fn hybrid(&mut self) -> &mut HostTier<H> {
        self.hybrid
    }

    /// The stream the walk is enqueued on.
    pub fn stream(&self) -> &'a CudaStream {
        self.stream
    }

    /// The host sums the last upload sent, `cols` rows of the model width.
    pub fn hsum(&self) -> &DeviceBuffer<f32> {
        self.hsum
    }

    /// `at`'s exchange key over the open walk's columns.
    #[must_use]
    pub fn key(&self, at: At) -> BatchKey {
        BatchKey {
            layer: at.layer,
            set: at.unit,
            at: 0,
            u: self.cols,
        }
    }
}

impl<H: HostExperts> Port for BatchLeg<'_, H> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Batch;

    /// One unit of `1..=cap` columns, both sets of the port free
    /// ([`HostTier::begin_group`]: the walk is eager, and its caller leaves
    /// nothing of an earlier walk in flight); any other point is refused by
    /// name.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        if o.units != 1 || !(1..=self.cap).contains(&o.cols) {
            return Err(GpuError::Shape {
                what: WHAT_BATCH,
                detail: format!(
                    "{} units of {} columns; the batch leg walks one unit of 1..={} columns",
                    o.units, o.cols, self.cap
                ),
            });
        }
        self.cols = o.cols;
        self.hybrid.begin_group()
    }

    /// Wait for `at`'s download, serve its host experts in one union call and
    /// enqueue the upload of their sums into the leg's `hsum`.
    fn serve(&mut self, at: At) -> Result<(), GpuError> {
        let key = self.key(at);
        self.hybrid.serve_key(key)?;
        self.hybrid.enqueue_upload(self.stream, self.hsum, key)
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT_BATCH,
            detail: why.to_string(),
        }
    }
}
