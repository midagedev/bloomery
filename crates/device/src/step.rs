//! One token's work, as the gates pin it.

use crate::DeviceError;

/// One token's work as the gates pin it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayShape {
    /// A captured graph, replayed: the node count of the CUDA chain.
    Graph {
        /// The graph's nodes.
        nodes: usize,
    },
    /// A walk encoded into one command buffer per token: the dispatch and
    /// barrier counts of that encoding.
    Encoded {
        /// The dispatches the encoding holds.
        dispatches: usize,
        /// The barriers between them.
        barriers: usize,
    },
}

/// The step's inputs as the device reads them: the token id, the position
/// the step starts at, the lane it runs on, and whether it is the first step
/// of the prompt.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepRecord {
    /// The token the step reads.
    pub id: u32,
    /// The position the first row of the step stands at.
    pub pos0: u32,
    /// The lane the step runs on.
    pub lane: u32,
    /// Whether this is the first step of the prompt.
    pub first: u32,
}

/// One token's work, enqueued on a stream.
///
/// The CUDA side replays a captured graph; the Metal side encodes a walk
/// into one command buffer per token. The trait is the same: the work is
/// enqueued and does not wait.
pub trait TokenReplay {
    /// The stream the work runs on.
    type Stream;

    /// The shape of the work this replay enqueues.
    fn shape(&self) -> ReplayShape;

    /// Enqueue one token; does not wait.
    fn launch(&mut self, s: &Self::Stream, step: StepRecord) -> Result<(), DeviceError>;
}
