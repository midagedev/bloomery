//! The family's layer program ([`Program`]): the ops ([`HybridOps`]) the chain crate's walk
//! ([`Walk`](chain::arch::hybrid::walk::Walk)) orders over one decode step. Each op is one of the
//! launch groups of the hybrid trunk ([`crate::arch::hybrid`]) or of the MLP layer, the layer it
//! runs the one the walk's [`Step`] names.
//!
//! The residual stream alternates between the head's input and the body's own buffer, each layer
//! writing the one it did not read, so the last layer's sum is the head's input whatever the layer
//! count's parity: the last op of each layer (`mamba_out`, `attn_out`, `ffn_out`) flips the side.

use super::step::{Layer, MlpLayer};
use crate::arch::hybrid::{
    self, Acts, Cx, GqaLayer, MambaLayer, Scratch, WORD_POS, WORD_TOKEN, norm, residual,
};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::GROUP_5;
use crate::head::Head;
use crate::weights::DevWeight;
use crate::{GpuError, Window};
use chain::arch::hybrid::ops::{HybridOps, Step};
use chain::arch::hybrid::plan::{MambaPlan, NopeGqaPlan};
use chain::arch::nemotron_h::plan::MlpPlan;
use cuda_core::DeviceBuffer;

const WHAT: &str = "nemotron_h::Program";

/// One decode step's walk over the body's layers, scratch and residual buffers.
pub(super) struct Program<'a> {
    pub(super) cx: Cx<'a>,
    /// The Q5_0 token table's name.
    pub(super) embed: &'a str,
    pub(super) layers: &'a mut [Layer],
    pub(super) s: &'a mut Scratch,
    pub(super) acts: &'a mut Acts,
    /// The residual stream's second buffer.
    pub(super) r: &'a mut DeviceBuffer<f32>,
    pub(super) head: &'a mut Head,
    /// Whether the rows the next layer reads are in the head's input.
    pub(super) in_head: bool,
}

/// The residual stream as (the rows a layer reads, the buffer it writes): the head's input and the
/// body's own buffer, in the order `in_head` says.
fn rails<'b>(
    in_head: bool,
    head: &'b mut Head,
    r: &'b mut DeviceBuffer<f32>,
) -> (&'b DeviceBuffer<f32>, &'b mut DeviceBuffer<f32>) {
    let hin = head.input_mut();
    if in_head { (&*hin, r) } else { (&*r, hin) }
}

/// A plan part that names a kind of layer the body holds another of: the plans and the layers are
/// made together at load, so this cannot meet a loaded body.
fn mismatch(layer: usize, runs: &str) -> GpuError {
    GpuError::shape(
        WHAT,
        format!("layer {layer} is not the {runs} its plan names"),
    )
}

fn mamba_at(layers: &mut [Layer], l: usize) -> Result<&mut MambaLayer, GpuError> {
    match layers.get_mut(l) {
        Some(Layer::Mamba(m)) => Ok(m),
        _ => Err(mismatch(l, "Mamba-2 layer")),
    }
}

fn attn_at(layers: &mut [Layer], l: usize) -> Result<&mut GqaLayer, GpuError> {
    match layers.get_mut(l) {
        Some(Layer::Attn(a)) => Ok(a),
        _ => Err(mismatch(l, "attention layer")),
    }
}

fn mlp_at(layers: &[Layer], l: usize) -> Result<&MlpLayer, GpuError> {
    match layers.get(l) {
        Some(Layer::Mlp(m)) => Ok(m),
        _ => Err(mismatch(l, "MLP layer")),
    }
}

impl HybridOps for Program<'_> {
    type Error = GpuError;
    type Ffn = MlpPlan;

    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let table = match w.get(self.embed) {
            Some(DevWeight::Q5_0 { w: table, .. }) => table,
            _ => return Err(GpuError::tensor(self.cx.what, self.embed, "a Q5_0 table")),
        };
        let t = &mut *self.s;
        let ids = Window::<u32>::of(&t.words, WORD_TOKEN * size_of::<u32>(), 1)?;
        let pos0 = Window::<u32>::of(&t.words, WORD_POS * size_of::<u32>(), 1)?;
        let hin = self.head.input_mut();
        let first = if self.in_head { hin } else { &mut *self.r };
        gpu.elem().enqueue_embed_rows_q5_0(
            gpu.stream(),
            EmbedRowsArgs {
                w: table,
                ids: &ids,
                pos0: &pos0,
                first: 0,
                y: first,
                pos: &mut t.buf.pos,
                n_keys: &mut t.buf.n_keys,
            },
            self.cx.sh.hidden,
        )
    }

    fn mamba_in(&mut self, s: Step, _: &MambaPlan) -> Result<(), GpuError> {
        let m = mamba_at(self.layers, s.layer)?;
        let (src, _) = rails(self.in_head, &mut *self.head, &mut *self.r);
        hybrid::mamba_in(
            &self.cx,
            (s.layer, &*m),
            (&mut *self.s, &mut *self.acts),
            src,
        )
    }

    fn mamba_core(&mut self, s: Step, _: &MambaPlan) -> Result<(), GpuError> {
        let m = mamba_at(self.layers, s.layer)?;
        hybrid::mamba_core(&self.cx, (s.layer, m), &mut *self.s)
    }

    fn mamba_out(&mut self, s: Step, _: &MambaPlan) -> Result<(), GpuError> {
        let m = mamba_at(self.layers, s.layer)?;
        let (src, dst) = rails(self.in_head, &mut *self.head, &mut *self.r);
        hybrid::mamba_out(
            &self.cx,
            (s.layer, &*m),
            (&mut *self.s, &mut *self.acts),
            (src, dst),
        )?;
        self.in_head = !self.in_head;
        Ok(())
    }

    fn attn_in(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        let (src, _) = rails(self.in_head, &mut *self.head, &mut *self.r);
        hybrid::gqa_in(
            &self.cx.gqa(),
            (s.layer, &*a),
            (&mut self.s.gqa(), &mut *self.acts),
            src,
        )
    }

    fn attn_core(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        hybrid::gqa_core::<GROUP_5>(&self.cx.gqa(), (s.layer, a), &mut self.s.gqa())
    }

    fn attn_out(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), GpuError> {
        let a = attn_at(self.layers, s.layer)?;
        let (src, dst) = rails(self.in_head, &mut *self.head, &mut *self.r);
        hybrid::gqa_out(
            &self.cx.gqa(),
            (s.layer, &*a),
            (&mut self.s.gqa(), &mut *self.acts),
            (src, dst),
        )?;
        self.in_head = !self.in_head;
        Ok(())
    }

    fn ffn_in(&mut self, s: Step, _: &MlpPlan) -> Result<(), GpuError> {
        let m = mlp_at(self.layers, s.layer)?;
        let (src, _) = rails(self.in_head, &mut *self.head, &mut *self.r);
        let b = &mut self.s.buf;
        norm(&self.cx, &m.norm, src, &mut b.xn)?;
        m.up.project(&self.cx.gqa(), self.acts, s.layer, &b.xn, &mut b.up)
    }

    fn ffn_core(&mut self, s: Step, _: &MlpPlan) -> Result<(), GpuError> {
        let m = mlp_at(self.layers, s.layer)?;
        let b = &mut self.s.buf;
        let gpu = self.cx.gpu;
        gpu.elem()
            .enqueue_relu_sqr(gpu.stream(), &b.up, m.ff, &mut b.act)
    }

    fn ffn_out(&mut self, s: Step, _: &MlpPlan) -> Result<(), GpuError> {
        let m = mlp_at(self.layers, s.layer)?;
        let (src, dst) = rails(self.in_head, &mut *self.head, &mut *self.r);
        let b = &mut self.s.buf;
        m.down
            .project(&self.cx.gqa(), self.acts, s.layer, &b.act, &mut b.delta)?;
        residual(&self.cx, src, &b.delta, dst)?;
        self.in_head = !self.in_head;
        Ok(())
    }

    fn head(&mut self, _: usize) -> Result<(), GpuError> {
        self.head.enqueue(self.cx.gpu, self.cx.w)
    }
}
