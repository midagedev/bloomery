//! The draft's head chain: the final norm, the projection
//! ([`super::HeadSource`]), the Markov loop over the block's positions and
//! the draft-to-target id map.

use cuda_core::DeviceBuffer;

use super::HeadSource;
use crate::DeviceTensor;

/// The head's weights on the card. Draft matrices are Q8_0 rows, quantized
/// from the file's bf16 on the host at load.
pub struct HeadWeights {
    /// The final norm.
    pub norm: DeviceTensor<f32>,
    /// Which projection the head runs.
    pub source: HeadSource,
    /// [`HeadSource::Own`]'s projection, one Q8_0 row per head id.
    pub own: Option<DeviceTensor<u32>>,
    /// The Markov term's input table, one row per target id (the previous
    /// token).
    pub markov_w1: DeviceTensor<u32>,
    /// The Markov term's output table, one row per head id.
    pub markov_w2: DeviceTensor<u32>,
    /// A reduced head's draft-to-target map on the card.
    pub to_target: Option<DeviceBuffer<u32>>,
    /// The confidence head's projection (hidden plus rank) and bias.
    pub conf: Option<(DeviceTensor<f32>, f32)>,
}
