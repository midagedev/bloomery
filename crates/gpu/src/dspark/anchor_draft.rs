//! A fused DSpark draft over [`CtxDenseLayers`] whose head is the anchor form of the recurrent
//! Markov head ([`RnnTail::ANCHOR`]), its correction folded through the target's output matrix:
//! Inkling 2's draft. [`AnchorDraft::load`] reads every weight by its statement name through a
//! [`Bf16Source`], so the forms a draft ships in (a checkpoint folder's BF16, a GGUF's Q8_0 blocks
//! and F32 vectors) build the same body: each matrix lands as Q8_0 rows ([`Bf16Source::q8_rows`]),
//! each vector as f32 ([`Bf16Source::f32s`]).
//!
//! The statement names: `fuse_weight` (a row of tap logits per mix), `layers.N.dflare_norm.weight`
//! and `markov_head.dflare_norm.weight` (each mix's gain, the layers' in order and the head's
//! last), `norm.weight`, and the head's `markov_head.{hidden_out, markov_w1, rnn_proj}`.

use cuda_core::DeviceBuffer;
use runtime::dspark::DsparkShape;

use super::ctx_dense::{AnchorEmbed, CtxDenseLayers, CtxSpec};
use super::dense::Bf16Source;
use super::fold::fold;
use super::fuse::mix_weights;
use super::head::{HeadTail, HeadWeights, RnnTail};
use super::{DsparkBody, EmbedSource, FuseWeights, HeadSource, TargetHead};
use crate::{DeviceTensor, Gpu, GpuError};

/// What the draft's statement gives the load beyond its tensors.
pub struct AnchorDraft<'a> {
    /// The draft's shape: its taps, block and fused context.
    pub shape: &'a DsparkShape,
    /// The layers' statement.
    pub layers: CtxSpec,
    /// The draft's hidden width (the target's).
    pub hidden: usize,
    /// The recurrent head's rank.
    pub rank: usize,
    /// The target's output rows: `markov_w1` holds one per id.
    pub padded_vocab: usize,
    /// The target's final soft cap, on the head's sum.
    pub softcap: f32,
}

impl AnchorDraft<'_> {
    /// The draft's body on `gpu` from `src`, the block's anchor row from `anchor` and the head's
    /// projection and fold through `output`, the target's matrix in place. Each tensor is refused by
    /// name at a shape other than the statement's. Load-time only; blocks.
    pub fn load(
        self,
        gpu: &Gpu,
        src: &dyn Bf16Source,
        anchor: AnchorEmbed,
        output: TargetHead,
    ) -> Result<DsparkBody<CtxDenseLayers>, GpuError> {
        let s = gpu.stream();
        let (n, hidden, rank) = (self.layers.layers, self.hidden, self.rank);
        let eps = self.layers.eps;
        let layers = CtxDenseLayers::load(gpu, src, self.layers, anchor)?;

        let taps = self.shape.taps.len();
        let logits = src.f32s("fuse_weight", &[n + 1, taps])?;
        let mut gains = Vec::with_capacity((n + 1) * hidden);
        for l in 0..n {
            gains.extend(src.f32s(&format!("layers.{l}.dflare_norm.weight"), &[hidden])?);
        }
        gains.extend(src.f32s("markov_head.dflare_norm.weight", &[hidden])?);
        let fuse = FuseWeights {
            mix: DeviceBuffer::from_host(s, &mix_weights(&logits, taps)?)?,
            gains: DeviceBuffer::from_host(s, &gains)?,
            eps,
        };

        let hidden_out = src.f32s("markov_head.hidden_out.weight", &[hidden, rank])?;
        let hidden_b = src.f32s("markov_head.hidden_out.bias", &[hidden])?;
        let folded = fold(gpu, &output, &hidden_out, &hidden_b, hidden, rank)?;
        let joint_in = 2 * rank + 2 * hidden;
        let w1 = src.q8_rows(
            gpu,
            "markov_head.markov_w1.weight",
            self.padded_vocab,
            rank,
            &|r| r,
        )?;
        let joint = src.q8_rows(
            gpu,
            "markov_head.rnn_proj.weight",
            3 * rank,
            joint_in,
            &|r| r,
        )?;
        let joint_b = src.f32s("markov_head.rnn_proj.bias", &[3 * rank])?;
        let norm = src.f32s("norm.weight", &[hidden])?;
        let head = HeadWeights {
            norm: DeviceTensor::upload(s, &norm, 1, hidden)?,
            eps,
            source: HeadSource::Target(output),
            tail: HeadTail::MarkovRnn(RnnTail {
                w1,
                w2: folded.w2,
                joint,
                joint_b: DeviceBuffer::from_host(s, &joint_b)?,
                softcap: Some(self.softcap),
                row: RnnTail::ANCHOR,
                c: Some(folded.c),
            }),
        };
        DsparkBody::fused(gpu, self.shape, layers, fuse, head, EmbedSource::Target)
    }
}
