//! One decode step: the layer kinds and the enqueue of the whole chain, the chain crate's walk
//! ([`chain::hybrid::walk`]) over [`Program`]'s ops (async throughout: nothing here
//! allocates, frees or synchronizes, so it is both the eager body and what a capture records).

use super::body::Body;
use super::program::Program;
use crate::arch::hybrid::{Cx, GqaLayer, MambaLayer, Site};
use crate::head::Head;
use crate::site::SiteTy;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use chain::hybrid::walk::Walk;

const WHAT: &str = "nemotron_h::Body::enqueue_chain";

/// The file types a site launches: Q5_0 through its packed 32-value gemv,
/// Q8_0 through the f32 one, Q4_K and Q6_K through theirs on the q8_1 rows.
pub(super) const SITE_TYPES: &[SiteTy] = &[SiteTy::Q5_0, SiteTy::Q8_0, SiteTy::Q4K, SiteTy::Q6K];

pub(super) struct MlpLayer {
    pub(super) norm: String,
    pub(super) up: Site,
    pub(super) down: Site,
    pub(super) ff: usize,
}

/// A layer, by the kind the file's per-layer arrays give it.
pub(super) enum Layer {
    Mamba(MambaLayer),
    Attn(GqaLayer),
    Mlp(MlpLayer),
}

impl Layer {
    pub(super) fn sites(&self) -> Vec<&Site> {
        match self {
            Layer::Mamba(m) => m.sites().to_vec(),
            Layer::Attn(a) => a.sites(),
            Layer::Mlp(m) => vec![&m.up, &m.down],
        }
    }
}

impl Body {
    /// The whole chain: the embedding, every layer, then `head`, as the walk of the body's plans
    /// orders the ops of [`Program`]. The residual stream alternates between the head's input and
    /// the body's own buffer ([`Program`]).
    pub(super) fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
    ) -> Result<(), GpuError> {
        let cx = Cx {
            what: WHAT,
            gpu,
            w,
            k: &self.k,
            sh: self.sh,
        };
        let in_head = self.layers.len().is_multiple_of(2);
        let mut program = Program {
            cx,
            embed: &self.embed,
            layers: &mut self.layers,
            s: &mut self.s,
            acts: &mut self.acts,
            r: &mut self.r,
            head,
            in_head,
        };
        Walk::new(&mut program, &self.plans, 1).run(1)
    }
}
