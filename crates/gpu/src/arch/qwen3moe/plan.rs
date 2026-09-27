//! A layer's plan: what one layer of a qwen3moe-family chain mixes with and
//! how its experts run, resolved once at load from the file — every weight
//! name, the quantization of each mixed site, the kernels' shape — so the
//! enqueue reads names and never decides a layer's kind. One layer body
//! (`dispatch::layer`) runs every plan: the mixer is a `match` on
//! [`MixerPlan`], the shared expert an `Option` of [`MoePlan`].
//!
//! Qwen3-30B-A3B's layers are all [`MixerPlan::Gqa`] at head 128 with no
//! shared expert. Qwen3.6-35B-A3B interleaves [`MixerPlan::Delta`] (gated
//! delta rule) with gated GQA at head 256, each followed by 256 routed
//! experts and a sigmoid-gated shared expert folded in as a ninth slot of
//! joined stacks. A Qwen3.6 layer's kind comes from its [`LayerSpec`] — the
//! file's tensors, cross-checked with its interval key by the reader —
//! never from the layer's number.

use crate::GpuError;
use crate::linear::{self, LinearShape};
use model::arch::models::{
    Act, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer, Moe, RopeMode, Score,
};

/// The two K-quants a mixed site comes in: a value projection, an experts'
/// down stack or a delta layer's q·k·v projection is Q4_K on some layers
/// and Q6_K on the rest; every other projection is Q4_K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kq {
    Q4K,
    Q6K,
}

/// One layer's plan.
pub(super) struct LayerPlan {
    pub(super) mixer: MixerPlan,
    pub(super) ffn: MoePlan,
}

/// What a layer mixes its tokens with.
pub(super) enum MixerPlan {
    /// Grouped-query attention over the layer's own K/V planes.
    Gqa(GqaPlan),
    /// The gated delta rule over the layer's recurrent state and conv ring.
    Delta(DeltaPlan),
}

/// The attention geometries a kernel set is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GqaKind {
    /// Qwen3: head 128, the whole head turned by NEOX, no output gate.
    Neox128,
    /// Qwen3.6: head 256, the first 64 values turned by NEOX, `attn_q`
    /// writing `[q | gate]` per head, the flash output multiplied by
    /// `σ(gate)` in the output projection's quantizer.
    Gated256,
}

/// A GQA layer's names, types and geometry.
pub(super) struct GqaPlan {
    pub(super) kind: GqaKind,
    pub(super) attn_norm: String,
    pub(super) attn_q: String,
    pub(super) attn_k: String,
    pub(super) attn_v: String,
    pub(super) attn_q_norm: String,
    pub(super) attn_k_norm: String,
    pub(super) attn_output: String,
    pub(super) v_ty: Kq,
}

/// A gated-delta-rule layer's names, types and head counts.
pub(super) struct DeltaPlan {
    pub(super) attn_norm: String,
    /// The q·k·v channels' projection (`attn_qkv`), Q4_K or Q6_K.
    pub(super) qkv: String,
    pub(super) qkv_ty: Kq,
    /// The output gate's projection `z` (`attn_gate`).
    pub(super) gate: String,
    /// The β and α projections, one value per value head.
    pub(super) beta: String,
    pub(super) alpha: String,
    /// The conv taps, `[C][CONV_TAPS]` F32.
    pub(super) conv: String,
    pub(super) dt_bias: String,
    pub(super) ssm_a: String,
    /// The gated norm's per-head gain.
    pub(super) ssm_norm: String,
    /// The output projection back to the residual.
    pub(super) ssm_out: String,
    pub(super) shape: LinearShape,
}

/// A routed FFN's names and types.
pub(super) struct MoePlan {
    pub(super) ffn_norm: String,
    /// The router weight: the file's, or the joined one with the shared
    /// expert's gate as its last row.
    pub(super) ffn_gate_inp: String,
    pub(super) ffn_gate_exps: String,
    pub(super) ffn_up_exps: String,
    pub(super) ffn_down_exps: String,
    pub(super) down_ty: Kq,
    /// The shared expert, when the stacks above are the joined ones.
    pub(super) shared: Option<SharedPlan>,
}

/// A shared expert folded into the routed stacks as their last expert
/// (`router::gated::SHARED`, the routed count) in every token's last slot,
/// weighted by the sigmoid of the router's last row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SharedPlan;

impl LayerPlan {
    /// The GQA plan, or a named refusal for a delta layer: what a path that
    /// runs attention only (qwen3moe's ubatch) takes.
    pub(super) fn gqa(&self, what: &'static str, l: usize) -> Result<&GqaPlan, GpuError> {
        match &self.mixer {
            MixerPlan::Gqa(g) => Ok(g),
            MixerPlan::Delta(_) => Err(GpuError::shape(
                what,
                format!("layer {l} is a delta-rule layer; this path runs attention layers only"),
            )),
        }
    }

    /// Slots a token takes in the routed FFN: the routed ones, plus one for
    /// a folded shared expert.
    pub(super) fn slots(&self, n_used: usize) -> usize {
        n_used + usize::from(self.ffn.shared.is_some())
    }
}

/// A Qwen3.6 layer's mixer kind, from its description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind35 {
    Gqa,
    Delta(LinearShape),
}

/// The Qwen3.6 geometry the chain's kernels are built for: 16/2 query/key
/// heads of 256, 64 values turned, 16/32 delta heads of 128 with a 4-tap
/// conv, 256 routed experts of 512 with eight used and a sigmoid-gated
/// shared expert of 512.
pub(super) mod q35 {
    pub(in super::super) const HEADS: u32 = 16;
    pub(in super::super) const KV_HEADS: u32 = 2;
    pub(in super::super) const HEAD_DIM: u32 = 256;
    pub(in super::super) const ROPE_DIMS: u32 = 64;
    pub(in super::super) const EXPERTS: u32 = 256;
    pub(in super::super) const TOP_K: u32 = 8;
    pub(in super::super) const EXPERT_FF: u32 = 512;
}

/// Layer `l`'s mixer kind from its description `s`, checked against the
/// geometry the chain's kernels take; any other layer is refused by name
/// with what it holds.
pub(super) fn kind35(s: &LayerSpec, l: usize) -> Result<Kind35, GpuError> {
    const WHAT: &str = "qwen35moe plan";
    let refuse = |d: String| GpuError::shape(WHAT, format!("layer {l}: {d}"));
    let kind = match &s.mixer {
        Mixer::Gqa(g) => {
            gqa_fits(g).map_err(refuse)?;
            Kind35::Gqa
        }
        Mixer::DeltaRule(d) => Kind35::Delta(delta_shape(d).map_err(refuse)?),
        Mixer::Latent(_) => {
            return Err(refuse(
                "a latent-attention mixer, which no kernel here runs".into(),
            ));
        }
    };
    match &s.ffn {
        Ffn::Moe(m) => moe_fits(m).map_err(refuse)?,
        Ffn::Dense { .. } => return Err(refuse("a dense FFN, which no kernel here runs".into())),
    }
    Ok(kind)
}

/// Every layer's kind of a Qwen3.6 description, in order.
pub(super) fn kinds35(layers: &[LayerSpec]) -> Result<Vec<Kind35>, GpuError> {
    layers
        .iter()
        .enumerate()
        .map(|(l, s)| kind35(s, l))
        .collect()
}

/// Ok when `g` is the gated attention the head-256 kernels run.
fn gqa_fits(g: &Gqa) -> Result<(), String> {
    use q35::{HEAD_DIM, HEADS, KV_HEADS, ROPE_DIMS};
    let rope_ok = match g.rope.mode {
        RopeMode::Neox => true,
        // A text-only position gives every section the same position, so
        // the sections' pairs turn as NEOX's when they cover the rotated
        // values' pairs.
        RopeMode::Imrope { sections } => {
            sections.iter().map(|&s| u64::from(s)).sum::<u64>() == u64::from(g.rope.dims / 2)
        }
        RopeMode::NormTail => false,
    };
    let fits = g.heads == HEADS
        && g.kv_heads == KV_HEADS
        && g.head_dim == HEAD_DIM
        && g.rope.dims == ROPE_DIMS
        && g.rope.yarn.is_none()
        && rope_ok
        && g.qk_norm
        && g.out_gate
        && g.select.is_none();
    if fits {
        Ok(())
    } else {
        Err(format!(
            "attention {g:?}; the kernels take {HEADS}/{KV_HEADS} heads of {HEAD_DIM}, the first \
             {ROPE_DIMS} values turned by NEOX (IMROPE sections covering them), q/k norms, the \
             output gate and no key selection"
        ))
    }
}

/// The delta kernels' launch shape of `d`, or why they do not run it.
fn delta_shape(d: &DeltaRule) -> Result<LinearShape, String> {
    let map = match d.kind {
        DeltaKind::Gdn {
            khead_map: KHeadMap::Tiled,
            gate: GdnGate::Silu,
        } => linear::KHeadMap::Tiled,
        DeltaKind::Gdn {
            gate: GdnGate::Sigmoid,
            ..
        } => {
            return Err("a sigmoid-gated GDN layer, which this body does not run".into());
        }
        DeltaKind::Kda { .. } => {
            return Err("a Kimi delta rule, which no kernel here runs".into());
        }
    };
    let shape = LinearShape {
        n_k: d.k_heads as usize,
        n_v: d.v_heads as usize,
        map,
    };
    let fits = d.d as usize == linear::HEAD
        && d.conv as usize == linear::CONV_TAPS
        && shape.n_k > 0
        && shape.n_v.is_multiple_of(shape.n_k);
    if fits {
        Ok(shape)
    } else {
        Err(format!(
            "delta rule {d:?}; the kernels take heads of {} and a {}-tap conv",
            linear::HEAD,
            linear::CONV_TAPS
        ))
    }
}

/// Ok when `m` is the routed FFN the gated router and the joined stacks run.
fn moe_fits(m: &Moe) -> Result<(), String> {
    use q35::{EXPERT_FF, EXPERTS, TOP_K};
    let swiglu = matches!(m.act, Act::SwiGlu { limit: None });
    let router = m.router.score == Score::Softmax
        && m.router.norm
        && !m.router.bias
        && !m.router.hash
        && m.router.scale == 1.0;
    let shared = m.shared.is_some_and(|s| {
        s.ff == EXPERT_FF && s.sigmoid_gate && matches!(s.act, Act::SwiGlu { limit: None })
    });
    if m.experts == EXPERTS
        && m.top_k == TOP_K
        && m.expert_ff == EXPERT_FF
        && swiglu
        && router
        && shared
    {
        Ok(())
    } else {
        Err(format!(
            "experts {m:?}; the kernels take {EXPERTS} SwiGLU experts of {EXPERT_FF}, {TOP_K} used, \
             softmax renormalized at scale 1, and a sigmoid-gated shared expert of {EXPERT_FF}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Kind35, kinds35, q35};
    use crate::linear::{KHeadMap as LinearMap, LinearShape};
    use model::arch::models::{
        Act, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer, Moe, Residual,
        Rope, RopeMode, Router, Score, Shared,
    };

    fn gqa() -> Mixer {
        Mixer::Gqa(Gqa {
            heads: q35::HEADS,
            kv_heads: q35::KV_HEADS,
            head_dim: q35::HEAD_DIM,
            rope: Rope {
                mode: RopeMode::Imrope {
                    sections: [11, 11, 10, 0],
                },
                dims: q35::ROPE_DIMS,
                base: 1e7,
                yarn: None,
            },
            qk_norm: true,
            out_gate: true,
            select: None,
        })
    }

    fn delta() -> Mixer {
        Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Silu,
            },
            k_heads: 16,
            v_heads: 32,
            d: 128,
            conv: 4,
        })
    }

    fn layer(mixer: Mixer) -> LayerSpec {
        LayerSpec {
            mixer,
            ffn: Ffn::Moe(Moe {
                experts: q35::EXPERTS,
                top_k: q35::TOP_K,
                expert_ff: q35::EXPERT_FF,
                act: Act::SwiGlu { limit: None },
                router: Router {
                    score: Score::Softmax,
                    bias: false,
                    norm: true,
                    scale: 1.0,
                    hash: false,
                },
                shared: Some(Shared {
                    ff: q35::EXPERT_FF,
                    act: Act::SwiGlu { limit: None },
                    sigmoid_gate: true,
                }),
            }),
            residual: Residual::Plain,
            extras: Vec::new(),
        }
    }

    /// The plan's kinds follow the description layer by layer, wherever its
    /// attention layers sit: a description whose attention layers are 0 and
    /// 5 of 8 (not every fourth) gets exactly those two, and the rest the
    /// delta rule at the description's heads.
    #[test]
    fn kinds_come_from_the_description_not_the_layer_number() {
        let attn = [0usize, 5];
        let layers: Vec<LayerSpec> = (0..8)
            .map(|l| layer(if attn.contains(&l) { gqa() } else { delta() }))
            .collect();
        let got = kinds35(&layers).unwrap_or_else(|e| panic!("{e}"));
        let shape = LinearShape {
            n_k: 16,
            n_v: 32,
            map: LinearMap::Tiled,
        };
        let want: Vec<Kind35> = (0..8)
            .map(|l| {
                if attn.contains(&l) {
                    Kind35::Gqa
                } else {
                    Kind35::Delta(shape)
                }
            })
            .collect();
        assert_eq!(got, want);
    }

    /// A layer the kernels were not built for is refused by name with its
    /// index: a head of 128, and an FFN without its shared expert.
    #[test]
    fn a_layer_the_kernels_do_not_take_is_refused_by_name() {
        let mut narrow = layer(gqa());
        if let Mixer::Gqa(g) = &mut narrow.mixer {
            g.head_dim = 128;
        }
        let mut bare = layer(delta());
        if let Ffn::Moe(m) = &mut bare.ffn {
            m.shared = None;
        }
        for (at, bad) in [(1usize, narrow), (2, bare)] {
            let mut layers = vec![layer(delta()), layer(delta()), layer(delta())];
            layers[at] = bad;
            match kinds35(&layers) {
                Err(e) => assert!(e.to_string().contains(&format!("layer {at}:")), "{e}"),
                Ok(k) => panic!("layer {at} accepted: {k:?}"),
            }
        }
    }
}
