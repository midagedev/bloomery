//! The launches of one decode step: the layer kinds, the MLP layer, and the
//! enqueue of the whole chain over the hybrid trunk's Mamba-2 and
//! attention layers ([`crate::arch::hybrid`]) (async throughout: nothing here
//! allocates, frees or synchronizes, so it is both the eager body and what a
//! capture records).

use super::body::Body;
use crate::arch::hybrid::{
    Acts, Cx, GqaLayer, MambaLayer, Scratch, Site, WORD_POS, WORD_TOKEN, gqa_layer, mamba_layer,
    norm, residual,
};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::GROUP_5;
use crate::head::Head;
use crate::site::SiteTy;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, Window};
use cuda_core::DeviceBuffer;

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

/// One MLP layer: `down(relu²(up(x)))`, no gate.
fn mlp_layer(
    cx: &Cx<'_>,
    (l, m): (usize, &MlpLayer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let b = &mut s.buf;
    let site_cx = cx.gqa();
    norm(cx, &m.norm, src, &mut b.xn)?;
    m.up.project(&site_cx, acts, l, &b.xn, &mut b.up)?;
    cx.gpu
        .elem()
        .enqueue_relu_sqr(cx.gpu.stream(), &b.up, m.ff, &mut b.act)?;
    m.down.project(&site_cx, acts, l, &b.act, &mut b.delta)?;
    residual(cx, src, &b.delta, dst)
}

impl Body {
    /// The whole chain: the embedding, every layer, then `head`. The residual
    /// stream alternates between the head's input and the body's own buffer,
    /// each layer writing the one it did not read, so the last layer's sum is
    /// the head's input whatever the layer count's parity.
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
        let Some(DevWeight::Q5_0 { w: table, .. }) = w.get(&self.embed) else {
            return Err(GpuError::tensor(WHAT, &self.embed, "a Q5_0 table"));
        };
        let hin = head.input_mut();
        let mut in_head = self.layers.len().is_multiple_of(2);
        {
            let t = &mut self.s;
            let ids = Window::<u32>::of(&t.words, WORD_TOKEN * size_of::<u32>(), 1)?;
            let pos0 = Window::<u32>::of(&t.words, WORD_POS * size_of::<u32>(), 1)?;
            let first = if in_head { &mut *hin } else { &mut self.r };
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
                self.sh.hidden,
            )?;
        }
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let (src, dst) = if in_head {
                (&*hin, &mut self.r)
            } else {
                (&self.r as &DeviceBuffer<f32>, &mut *hin)
            };
            match layer {
                Layer::Mamba(m) => {
                    mamba_layer(&cx, (l, m), (&mut self.s, &mut self.acts), (src, dst))?;
                }
                Layer::Attn(a) => gqa_layer::<GROUP_5>(
                    &cx.gqa(),
                    (l, a),
                    (self.s.gqa(), &mut self.acts),
                    (src, dst),
                )?,
                Layer::Mlp(m) => mlp_layer(&cx, (l, m), (&mut self.s, &mut self.acts), (src, dst))?,
            }
            in_head = !in_head;
        }
        head.enqueue(gpu, w)
    }
}
