//! The family's layer program ([`Program`]): the ops ([`HybridOps`]) the chain crate's walk
//! ([`Walk`](chain::hybrid::walk::Walk)) orders over one decode step. The Mamba-2 and attention
//! ops are the hybrid trunk's ([`Trunk`]); the FFN ops are the MLP layer's. Each runs the layer the
//! walk's [`Step`] names.

use super::step::{Layer, MlpLayer};
use crate::arch::hybrid::{
    GqaLayer, MambaLayer, Trunk, WORD_POS, WORD_TOKEN, mismatch, norm, rails, residual,
};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::GROUP_5;
use crate::weights::DevWeight;
use crate::{GpuError, Window};
use chain::arch::nemotron_h::plan::MlpPlan;
use chain::hybrid::ops::{HybridOps, Step};
use chain::hybrid::plan::{MambaPlan, NopeGqaPlan};

const WHAT: &str = "nemotron_h::Program";

/// One decode step's walk over the body's layers and the trunk's scratch and residual stream.
pub(super) struct Program<'a> {
    pub(super) t: Trunk<'a>,
    /// The Q5_0 token table's name.
    pub(super) embed: &'a str,
    pub(super) layers: &'a mut [Layer],
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

fn mlp_at(layers: &[Layer], l: usize) -> Result<&MlpLayer, GpuError> {
    match layers.get(l) {
        Some(Layer::Mlp(m)) => Ok(m),
        _ => Err(mismatch(WHAT, l, "MLP layer")),
    }
}

impl HybridOps for Program<'_> {
    type Error = GpuError;
    type Ffn = MlpPlan;

    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        let t = &mut self.t;
        let (gpu, w) = (t.cx.gpu, t.cx.w);
        let Some(DevWeight::Q5_0 { w: table, .. }) = w.get(self.embed) else {
            return Err(GpuError::tensor(WHAT, self.embed, "a Q5_0 table"));
        };
        let s = &mut *t.s;
        let ids = Window::<u32>::of(&s.words, WORD_TOKEN * size_of::<u32>(), 1)?;
        let pos0 = Window::<u32>::of(&s.words, WORD_POS * size_of::<u32>(), 1)?;
        let hin = t.head.input_mut();
        let first = if t.in_head { hin } else { &mut *t.r };
        gpu.elem().enqueue_embed_rows_q5_0(
            gpu.stream(),
            EmbedRowsArgs {
                w: table,
                ids: &ids,
                pos0: &pos0,
                first: 0,
                y: first,
                pos: &mut s.buf.pos,
                n_keys: &mut s.buf.n_keys,
            },
            t.cx.sh.hidden,
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
        self.t.attn_core::<GROUP_5>(s.layer, a)
    }

    fn attn_out(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        self.t.attn_out(s.layer, a)
    }

    fn ffn_in(&mut self, s: Step, _: &MlpPlan) -> Result<(), GpuError> {
        let m = mlp_at(self.layers, s.layer)?;
        let t = &mut self.t;
        let (src, _) = rails(t.in_head, &mut *t.head, &mut *t.r);
        let b = &mut t.s.buf;
        norm(&t.cx, &m.norm, src, &mut b.xn)?;
        m.up.project(&t.cx.gqa(), t.acts, s.layer, &b.xn, &mut b.up)
    }

    fn ffn_core(&mut self, s: Step, _: &MlpPlan) -> Result<(), GpuError> {
        let m = mlp_at(self.layers, s.layer)?;
        let b = &mut self.t.s.buf;
        let gpu = self.t.cx.gpu;
        gpu.elem()
            .enqueue_relu_sqr(gpu.stream(), &b.up, m.ff, &mut b.act)
    }

    fn ffn_out(&mut self, s: Step, _: &MlpPlan) -> Result<(), GpuError> {
        let m = mlp_at(self.layers, s.layer)?;
        let t = &mut self.t;
        let (src, dst) = rails(t.in_head, &mut *t.head, &mut *t.r);
        let b = &mut t.s.buf;
        m.down
            .project(&t.cx.gqa(), t.acts, s.layer, &b.act, &mut b.delta)?;
        residual(&t.cx, src, &b.delta, dst)?;
        t.flip();
        Ok(())
    }

    fn head(&mut self, _: usize) -> Result<(), GpuError> {
        self.t.enqueue_head()
    }
}
