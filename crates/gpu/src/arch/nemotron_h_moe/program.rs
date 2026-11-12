//! The family's layer program ([`Program`]): the ops ([`HybridOps`]) the chain crate's walk
//! ([`Walk`](chain::hybrid::walk::Walk)) orders over one decode step. The Mamba-2 and attention
//! ops are the hybrid trunk's ([`Trunk`]); the FFN ops are the MoE layer's ([`super::moe`]). Each
//! runs the layer the walk's [`Step`] names. A placed load runs the same ops per layer around its
//! host leg ([`super::placed::StepWalk`]).

use super::body::Layer;
use super::moe::{MoeKernels, MoeLayer, MoeScratch, MoeShape, moe_core, moe_in, moe_out};
use crate::arch::hybrid::{GqaLayer, MambaLayer, Trunk, WORD_POS, WORD_TOKEN, mismatch, rails};
use crate::flash_gqa::GROUP_16;
use crate::q38::{EmbedArgs, EmbedRows, Q38Kernels};
use crate::{GpuError, Window};
use chain::arch::nemotron_h_moe::plan::MoePlan;
use chain::hybrid::ops::{HybridOps, Step};
use chain::hybrid::plan::{MambaPlan, NopeGqaPlan};

const WHAT: &str = "nemotron_h_moe::Program";

/// One decode step's walk over the body's layers, the MoE scratch and the trunk's scratch and
/// residual stream.
pub(super) struct Program<'a> {
    pub(super) t: Trunk<'a>,
    /// The Q8_0 token table's name.
    pub(super) embed: &'a str,
    pub(super) q38: &'a Q38Kernels,
    pub(super) mk: &'a MoeKernels,
    pub(super) moe: MoeShape,
    pub(super) layers: &'a mut [Layer],
    pub(super) ms: &'a mut MoeScratch,
}

fn mamba_at(layers: &mut [Layer], l: usize) -> Result<&mut MambaLayer, GpuError> {
    match layers.get_mut(l) {
        Some(Layer::Mamba(m)) => Ok(m),
        _ => Err(mismatch(WHAT, l, "Mamba-2 layer")),
    }
}

fn attn_at(layers: &mut [Layer], l: usize) -> Result<&mut GqaLayer, GpuError> {
    match layers.get_mut(l) {
        Some(Layer::Attn(a)) => Ok(a),
        _ => Err(mismatch(WHAT, l, "attention layer")),
    }
}

fn moe_at(layers: &mut [Layer], l: usize) -> Result<&mut MoeLayer, GpuError> {
    match layers.get_mut(l) {
        Some(Layer::Moe(m)) => Ok(m),
        _ => Err(mismatch(WHAT, l, "MoE layer")),
    }
}

impl HybridOps for Program<'_> {
    type Error = GpuError;
    type Ffn = MoePlan;

    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        let t = &mut self.t;
        let (gpu, w) = (t.cx.gpu, t.cx.w);
        let (qs, d) = w.q8_planes(t.cx.what, self.embed)?;
        let s = &mut *t.s;
        let ids = Window::<u32>::of(&s.words, WORD_TOKEN * size_of::<u32>(), 1)?;
        let pos0 = Window::<u32>::of(&s.words, WORD_POS * size_of::<u32>(), 1)?;
        let first = if t.in_head {
            &mut *t.head.input_mut()
        } else {
            &mut *t.r
        };
        self.q38.enqueue_embed_rows(
            gpu.stream(),
            EmbedArgs {
                table: EmbedRows::Q8_0 { qs, d },
                ids: &ids,
                pos0: &pos0,
                first: 0,
                fault: gpu.unlabelled_sink(),
                y: first,
                pos: &mut s.buf.pos,
                n_keys: &mut s.buf.n_keys,
            },
        )
    }

    fn mamba_in(&mut self, s: Step, _: &MambaPlan) -> Result<(), GpuError> {
        let m = mamba_at(self.layers, s.layer)?;
        self.t.mamba_in(s.layer, m)
    }

    fn mamba_core(&mut self, s: Step, _: &MambaPlan) -> Result<(), GpuError> {
        let m = mamba_at(self.layers, s.layer)?;
        self.t.mamba_core(s.layer, m)
    }

    fn mamba_out(&mut self, s: Step, _: &MambaPlan) -> Result<(), GpuError> {
        let m = mamba_at(self.layers, s.layer)?;
        self.t.mamba_out(s.layer, m)
    }

    fn attn_in(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        self.t.attn_in(s.layer, a)
    }

    fn attn_core(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        self.t.attn_core::<GROUP_16>(s.layer, a)
    }

    fn attn_out(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        self.t.attn_out(s.layer, a)
    }

    fn ffn_in(&mut self, s: Step, _: &MoePlan) -> Result<(), GpuError> {
        let m = moe_at(self.layers, s.layer)?;
        let t = &mut self.t;
        let (src, _) = rails(t.in_head, &mut *t.head, &mut *t.r);
        moe_in(
            &t.cx,
            (self.mk, self.moe),
            (s.layer, m),
            (&mut *t.s, &mut *self.ms, &mut *t.acts),
            src,
        )
    }

    fn ffn_core(&mut self, s: Step, _: &MoePlan) -> Result<(), GpuError> {
        let m = moe_at(self.layers, s.layer)?;
        let t = &mut self.t;
        moe_core(
            &t.cx,
            (self.mk, self.moe),
            (s.layer, m),
            (&mut *t.s, &mut *self.ms, &mut *t.acts),
        )
    }

    fn ffn_out(&mut self, s: Step, _: &MoePlan) -> Result<(), GpuError> {
        let m = moe_at(self.layers, s.layer)?;
        let t = &mut self.t;
        let (src, dst) = rails(t.in_head, &mut *t.head, &mut *t.r);
        moe_out(
            &t.cx,
            (self.mk, self.moe),
            (s.layer, m),
            (&mut *t.s, &mut *self.ms, &mut *t.acts),
            (src, dst),
        )?;
        t.flip();
        Ok(())
    }

    fn head(&mut self, _: usize) -> Result<(), GpuError> {
        self.t.enqueue_head()
    }
}
