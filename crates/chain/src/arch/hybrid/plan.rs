//! A layer's plan in a Mamba-2 and attention hybrid: every weight's name, the type of each
//! projection site and the geometry the kernels are built for, resolved once at load from the
//! file's header so a backend's ops read names and never decide a layer's kind. A layer is one of
//! three kinds ([`LayerPlan`]); the walk ([`super::walk`]) is a `match` on it.

use crate::site::SiteTy;
use model::arch::models::Mamba2;

/// One layer's plan: the kind it is, with that kind's weights. `F` is the family's feed-forward
/// plan ([`super::ops::HybridOps::Ffn`]).
#[derive(Clone, Debug, PartialEq)]
pub enum LayerPlan<F> {
    Mamba(MambaPlan),
    Attn(NopeGqaPlan),
    Ffn(F),
}

/// A Mamba-2 layer: the pre-norm, the two projections around the scan, and the scan's own vectors.
/// Every tensor name is the file's.
#[derive(Clone, Debug, PartialEq)]
pub struct MambaPlan {
    pub norm: String,
    pub ssm_in: String,
    pub ssm_out: String,
    pub a: String,
    pub d: String,
    pub dt_bias: String,
    pub conv_w: String,
    /// `None` where the file holds no conv bias (it is optional): the backend supplies zeros.
    pub conv_b: Option<String>,
    pub ssm_norm: String,
    /// The description's widths.
    pub shape: Mamba2,
    /// Conv channels, `d_inner + 2 · n_groups · d_state`: the width of the conv bias and of the
    /// `xBC` view of `ssm_in`'s output.
    pub conv_dim: usize,
    pub in_ty: SiteTy,
    pub out_ty: SiteTy,
}

/// A grouped-query attention layer with no rope, no q/k norm, no output norm and no output gate,
/// over heads of [`crate::linear::HEAD`] values.
#[derive(Clone, Debug, PartialEq)]
pub struct NopeGqaPlan {
    pub norm: String,
    pub q: String,
    pub k: String,
    pub v: String,
    pub o: String,
    /// Query heads.
    pub n_head: usize,
    /// Key and value heads; `n_head` is a multiple of it.
    pub n_kv: usize,
    pub q_ty: SiteTy,
    pub k_ty: SiteTy,
    pub v_ty: SiteTy,
    pub o_ty: SiteTy,
}

/// How many layers of each kind a chain's plans hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub mamba: usize,
    pub attn: usize,
    pub ffn: usize,
}

impl Counts {
    /// The layers of each kind in `plans`.
    #[must_use]
    pub fn of<F>(plans: &[LayerPlan<F>]) -> Counts {
        plans.iter().fold(Counts::default(), |mut n, p| {
            match p {
                LayerPlan::Mamba(_) => n.mamba += 1,
                LayerPlan::Attn(_) => n.attn += 1,
                LayerPlan::Ffn(_) => n.ffn += 1,
            }
            n
        })
    }
}
