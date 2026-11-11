//! The family's layer program ([`Program`]): the ops ([`HybridOps`]) the chain crate's walk
//! ([`Walk`](chain::hybrid::walk::Walk)) orders over one decode step. Each op is one of the launch
//! groups of the hybrid trunk ([`crate::arch::hybrid`]) or of the MoE layer ([`super::moe`]), the
//! layer it runs the one the walk's [`Step`] names.
//!
//! The residual stream alternates between the head's input and the body's own buffer, each layer
//! writing the one it did not read, so the last layer's sum is the head's input whatever the layer
//! count's parity: the last op of each layer (`mamba_out`, `attn_out`, `ffn_out`) flips the side.
//! A placed load runs the same ops per layer around its host leg ([`super::placed::StepWalk`]).

use super::body::Layer;
use super::moe::{MoeKernels, MoeLayer, MoeScratch, MoeShape, moe_core, moe_in, moe_out};
use crate::arch::hybrid::{self, Acts, Cx, GqaLayer, MambaLayer, Scratch, WORD_POS, WORD_TOKEN};
use crate::flash_gqa::GROUP_16;
use crate::head::Head;
use crate::q38::{EmbedArgs, EmbedRows, Q38Kernels};
use crate::{GpuError, Window};
use chain::arch::nemotron_h_moe::plan::MoePlan;
use chain::hybrid::ops::{HybridOps, Step};
use chain::hybrid::plan::{MambaPlan, NopeGqaPlan};
use cuda_core::DeviceBuffer;

const WHAT: &str = "nemotron_h_moe::Program";

/// One decode step's walk over the body's layers, scratch and residual buffers.
pub(super) struct Program<'a> {
    pub(super) cx: Cx<'a>,
    /// The Q8_0 token table's name.
    pub(super) embed: &'a str,
    pub(super) q38: &'a Q38Kernels,
    pub(super) mk: &'a MoeKernels,
    pub(super) moe: MoeShape,
    pub(super) layers: &'a mut [Layer],
    pub(super) s: &'a mut Scratch,
    pub(super) ms: &'a mut MoeScratch,
    pub(super) acts: &'a mut Acts,
    /// The residual stream's second buffer.
    pub(super) r: &'a mut DeviceBuffer<f32>,
    pub(super) head: &'a mut Head,
    /// Whether the rows the next layer reads are in the head's input.
    pub(super) in_head: bool,
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

fn moe_at(layers: &mut [Layer], l: usize) -> Result<&mut MoeLayer, GpuError> {
    match layers.get_mut(l) {
        Some(Layer::Moe(m)) => Ok(m),
        _ => Err(mismatch(l, "MoE layer")),
    }
}

/// The residual stream as (the rows a layer reads, the buffer it writes): the head's input and the
/// body's own buffer, in the order `in_head` says.
pub(super) fn rails<'b>(
    in_head: bool,
    head: &'b mut Head,
    r: &'b mut DeviceBuffer<f32>,
) -> (&'b DeviceBuffer<f32>, &'b mut DeviceBuffer<f32>) {
    let hin = head.input_mut();
    if in_head { (&*hin, r) } else { (&*r, hin) }
}

impl HybridOps for Program<'_> {
    type Error = GpuError;
    type Ffn = MoePlan;

    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        let (gpu, w) = (self.cx.gpu, self.cx.w);
        let (qs, d) = w.q8_planes(self.cx.what, self.embed)?;
        let t = &mut *self.s;
        let ids = Window::<u32>::of(&t.words, WORD_TOKEN * size_of::<u32>(), 1)?;
        let pos0 = Window::<u32>::of(&t.words, WORD_POS * size_of::<u32>(), 1)?;
        let first = if self.in_head {
            &mut *self.head.input_mut()
        } else {
            &mut *self.r
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
                pos: &mut t.buf.pos,
                n_keys: &mut t.buf.n_keys,
            },
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
        hybrid::gqa_core::<GROUP_16>(&self.cx.gqa(), (s.layer, a), &mut self.s.gqa())
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

    fn ffn_in(&mut self, s: Step, _: &MoePlan) -> Result<(), GpuError> {
        let m = moe_at(self.layers, s.layer)?;
        let (src, _) = rails(self.in_head, &mut *self.head, &mut *self.r);
        moe_in(
            &self.cx,
            (self.mk, self.moe),
            (s.layer, m),
            (&mut *self.s, &mut *self.ms, &mut *self.acts),
            src,
        )
    }

    fn ffn_core(&mut self, s: Step, _: &MoePlan) -> Result<(), GpuError> {
        let m = moe_at(self.layers, s.layer)?;
        moe_core(
            &self.cx,
            (self.mk, self.moe),
            (s.layer, m),
            (&mut *self.s, &mut *self.ms, &mut *self.acts),
        )
    }

    fn ffn_out(&mut self, s: Step, _: &MoePlan) -> Result<(), GpuError> {
        let m = moe_at(self.layers, s.layer)?;
        let (src, dst) = rails(self.in_head, &mut *self.head, &mut *self.r);
        moe_out(
            &self.cx,
            (self.mk, self.moe),
            (s.layer, m),
            (&mut *self.s, &mut *self.ms, &mut *self.acts),
            (src, dst),
        )?;
        self.in_head = !self.in_head;
        Ok(())
    }

    fn head(&mut self, _: usize) -> Result<(), GpuError> {
        self.head.enqueue(self.cx.gpu, self.cx.w)
    }
}
