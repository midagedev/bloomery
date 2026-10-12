//! The draft's head chain: the final norm, the projection
//! ([`super::HeadSource`]), the Markov loop over the block's positions and
//! the draft-to-target id map.

use cuda_core::DeviceBuffer;

use super::HeadSource;
use crate::DeviceTensor;

/// The head's weights on the card.
pub struct HeadWeights {
    /// The final norm's gain.
    pub norm: DeviceTensor<f32>,
    /// The final norm's epsilon.
    pub eps: f32,
    /// The projection: the target's matrix or the draft's own Q8_0 rows.
    pub source: HeadSource,
    /// The Markov term's input table, one row per target id (the previous
    /// token): bf16 values two to a word.
    pub markov_w1: DeviceTensor<u32>,
    /// The Markov term's output table, one row per head id: bf16 values two
    /// to a word.
    pub markov_w2: DeviceTensor<u32>,
    /// A reduced head's draft-to-target map on the card.
    pub to_target: Option<DeviceBuffer<u32>>,
    /// The confidence head's projection (hidden plus rank) and bias.
    pub conf: Option<(DeviceTensor<f32>, f32)>,
}
