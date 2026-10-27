//! The family's own layer: a dense MLP with no gate, `down(relu²(up(x)))`.

use crate::hybrid::plan::LayerPlan;
use crate::site::SiteTy;

/// A dense MLP layer: the pre-norm, the up projection to `ff` values and the down projection back.
#[derive(Clone, Debug, PartialEq)]
pub struct MlpPlan {
    pub norm: String,
    pub up: String,
    pub down: String,
    /// The layer's feed-forward width.
    pub ff: usize,
    pub up_ty: SiteTy,
    pub down_ty: SiteTy,
}

/// One layer of a nemotron_h chain.
pub type Plan = LayerPlan<MlpPlan>;
