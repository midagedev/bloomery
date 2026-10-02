//! A layer's plan: what one layer of a qwen3moe-family chain mixes with and
//! how its experts run, resolved once at load from the file — every weight
//! name, the quantization of each mixed site, the kernels' shape — so the
//! enqueue reads names and never decides a layer's kind. One layer body
//! (`dispatch::layer`) runs every plan: the mixer is a `match` on
//! [`MixerPlan`], how a token's FFN slots are picked a `match` on
//! [`FfnRoute`].
//!
//! Qwen3-30B-A3B's layers are all [`MixerPlan::Gqa`] at head 128 with no
//! shared expert. Qwen3.6-35B-A3B interleaves [`MixerPlan::Delta`] (gated
//! delta rule) with gated GQA at head 256, each followed by 256 routed
//! experts and a sigmoid-gated shared expert folded in as one more slot of
//! joined stacks. Qwen3.5-27B has the same mixers and a dense SwiGLU FFN,
//! which runs as a stack of one expert, every token's one slot on it at
//! weight 1 ([`FfnRoute::Dense`]): the routed FFN's launches without the
//! router. A layer's kind comes from its [`LayerSpec`] — the file's tensors,
//! cross-checked with its interval key by the reader — never from the
//! layer's number.

use super::router::RouterDims;
use crate::GpuError;
use crate::linear::{self, LinearShape};
use model::arch::coverage::turns_as_neox;
use model::arch::models::shape::{AttnShape, GroupRule, MoeShape, select_gqa};
use model::arch::models::{
    Act, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer, Moe,
};

pub(super) use crate::site::{Form, SiteTy};

/// One layer's plan.
pub(super) struct LayerPlan {
    pub(super) mixer: MixerPlan,
    pub(super) ffn: FfnPlan,
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

/// How the head-256 flash entries cover a key head's group of query heads:
/// the shape table's row (`models::shape::GQA`) the layer selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Flash {
    /// One block of eight query heads a key head (the `_256` entries; and
    /// the head-128 entries' only form).
    Group,
    /// Blocks of four query heads, `group / 4` a key head (the `_256_p4`
    /// entries): the scalar and the tensor-core segment pass.
    Quads,
    /// Blocks of two query heads, `group / 2` a key head (the `_256_p2`
    /// entries): the scalar segment pass only.
    Pairs,
}

/// A GQA layer's names, types and geometry.
pub(super) struct GqaPlan {
    pub(super) kind: GqaKind,
    pub(super) flash: Flash,
    pub(super) attn_norm: String,
    pub(super) attn_q: String,
    pub(super) attn_k: String,
    pub(super) attn_v: String,
    pub(super) attn_q_norm: String,
    pub(super) attn_k_norm: String,
    pub(super) attn_output: String,
    /// The file's types of `attn_q`, `attn_k`, `attn_v` and `attn_output`.
    pub(super) q_ty: SiteTy,
    pub(super) k_ty: SiteTy,
    pub(super) v_ty: SiteTy,
    pub(super) o_ty: SiteTy,
}

/// A gated-delta-rule layer's names, types and head counts.
pub(super) struct DeltaPlan {
    pub(super) attn_norm: String,
    /// The q·k·v channels' projection (`attn_qkv`).
    pub(super) qkv: String,
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
    /// The file's types of `attn_qkv`, `attn_gate`, β, α and `ssm_out`.
    pub(super) qkv_ty: SiteTy,
    pub(super) gate_ty: SiteTy,
    pub(super) beta_ty: SiteTy,
    pub(super) alpha_ty: SiteTy,
    pub(super) out_ty: SiteTy,
}

/// An FFN's names and types: the three stacks every slot reads (a routed
/// layer's experts, or a dense layer's matrices as one expert) and how a
/// token's slots are picked.
pub(super) struct FfnPlan {
    pub(super) ffn_norm: String,
    pub(super) route: FfnRoute,
    pub(super) gate: String,
    pub(super) up: String,
    pub(super) down: String,
    /// The file's types of the three stacks.
    pub(super) gate_ty: SiteTy,
    pub(super) up_ty: SiteTy,
    pub(super) down_ty: SiteTy,
}

/// How a token's FFN slots are picked.
pub(super) enum FfnRoute {
    /// By the router.
    Router {
        /// The router weight: the file's, or the joined one with the shared
        /// expert's gate as its last row.
        gate_inp: String,
        /// The shared expert, when the stacks are the joined ones.
        shared: Option<SharedPlan>,
    },
    /// A dense FFN: one slot a token, on expert 0 of the one-expert stacks,
    /// at weight 1 — the arena's fixed dense route (`scratch::Route::Dense`).
    Dense,
}

/// A shared expert folded into the routed stacks as their last expert (id
/// the routed count) in every token's last slot, weighted by the sigmoid of
/// the router's last row.
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

    /// Slots a token takes in the FFN: the routed ones, plus one for a
    /// folded shared expert; a dense FFN's one.
    pub(super) fn slots(&self, n_used: usize) -> usize {
        match &self.ffn.route {
            FfnRoute::Router { shared, .. } => n_used + usize::from(shared.is_some()),
            FfnRoute::Dense => 1,
        }
    }
}

/// Whether every one of `tys` is Q4_K: a fused group of Q4_K launches
/// (one launch over several matrices, or a projection with its neighbour
/// folded in) runs only then.
fn all_q4k(tys: &[SiteTy]) -> bool {
    tys.iter().all(|t| *t == SiteTy::Q4K)
}

impl GqaPlan {
    /// The gemv arm's fused q·k·v: Q4_K q and k with a K-quant v (a Q6_K v
    /// in its own gemv); otherwise each projection launches alone.
    pub(super) fn qkv_fused(&self) -> bool {
        all_q4k(&[self.q_ty, self.k_ty]) && self.v_ty.kquant()
    }

    /// The gemv arm's output projection with the residual add folded in: a
    /// Q4_K `attn_output`.
    pub(super) fn o_fused(&self) -> bool {
        all_q4k(&[self.o_ty])
    }
}

impl DeltaPlan {
    /// The gemv arm's two input launches: a K-quant q·k·v projection
    /// (a Q6_K one in its own gemv) and Q4_K `z`, β and α; otherwise each
    /// projection launches alone.
    pub(super) fn input_fused(&self) -> bool {
        self.qkv_ty.kquant() && all_q4k(&[self.gate_ty, self.beta_ty, self.alpha_ty])
    }

    /// The gemv arm's output projection with the residual add folded in: a
    /// Q4_K `ssm_out`.
    pub(super) fn out_fused(&self) -> bool {
        all_q4k(&[self.out_ty])
    }
}

impl FfnPlan {
    /// The gemv arm's gate·up·SwiGLU in one launch: Q4_K gate and up;
    /// otherwise each launches alone and the SwiGLU after them.
    pub(super) fn gate_up_fused(&self) -> bool {
        all_q4k(&[self.gate_ty, self.up_ty])
    }
}

/// A Body35 layer's mixer kind, from its description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind35 {
    Gqa(Flash),
    Delta(LinearShape),
}

/// The geometry the chain's kernels are built for: query heads of 256 with
/// 64 values turned, in groups the shape table's head-256 rows of eight or
/// of pairs take; delta heads of 128 with a 4-tap conv; experts of 512 with
/// a sigmoid-gated shared expert of 512, or a dense FFN of whole K-quant
/// super-blocks. The routed experts' count and the experts a token keeps are
/// the router instance's the file's shape selects ([`RouterDims::of`]); the
/// head counts are the file's.
pub(super) mod q35 {
    pub(in super::super) const HEAD_DIM: u32 = 256;
    pub(in super::super) const ROPE_DIMS: u32 = 64;
    pub(in super::super) const EXPERT_FF: u32 = 512;
    /// Values of a K-quant super-block: a dense width's unit.
    pub(in super::super) const SUPER_BLOCK: u32 = 256;
}

/// Layer `l`'s mixer kind from its description `s`, checked against the
/// geometry the chain's kernels take; any other layer is refused by name
/// with what it holds.
pub(super) fn kind35(s: &LayerSpec, l: usize) -> Result<Kind35, GpuError> {
    const WHAT: &str = "qwen35moe plan";
    let refuse = |d: String| GpuError::shape(WHAT, format!("layer {l}: {d}"));
    let kind = match &s.mixer {
        Mixer::Gqa(g) => Kind35::Gqa(gqa_fits(g).map_err(refuse)?),
        Mixer::DeltaRule(d) => Kind35::Delta(delta_shape(d).map_err(refuse)?),
        Mixer::Latent(_) => {
            return Err(refuse(
                "a latent-attention mixer, which no kernel here runs".into(),
            ));
        }
    };
    match &s.ffn {
        Ffn::Moe(m) => {
            moe_fits(m).map_err(refuse)?;
        }
        Ffn::Dense { ff, act } => dense_fits(*ff, *act).map_err(refuse)?,
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

/// The flash of `g` when it is the gated attention the head-256 kernels run:
/// the shape table's row for its group ([`select_gqa`]), one of eight or of
/// pairs.
fn gqa_fits(g: &Gqa) -> Result<Flash, String> {
    use q35::{HEAD_DIM, ROPE_DIMS};
    let rope_ok = turns_as_neox(g.rope.mode, g.rope.dims);
    let fits = g.head_dim == HEAD_DIM
        && g.rope.dims == ROPE_DIMS
        && g.rope.yarn.is_none()
        && rope_ok
        && g.qk_norm
        && g.out_gate
        && g.select.is_none();
    if !fits {
        return Err(format!(
            "attention {g:?}; the kernels take heads of {HEAD_DIM}, the first {ROPE_DIMS} values \
             turned by NEOX (IMROPE sections covering them), q/k norms, the output gate and no \
             key selection"
        ));
    }
    let row = select_gqa(AttnShape::of(g)).map_err(|e| e.to_string())?;
    match (row.pack, row.group) {
        (8, GroupRule::One) => Ok(Flash::Group),
        (4, GroupRule::Packs) => Ok(Flash::Quads),
        (2, GroupRule::Packs) => Ok(Flash::Pairs),
        _ => Err(format!(
            "attention {}/{} heads selects {}, which this body does not launch",
            g.heads, g.kv_heads, row.at
        )),
    }
}

/// Ok when a dense FFN of `ff` values with `act` is the one the body runs:
/// SwiGLU with no limit through the routed launches at one slot, `ff` whole
/// K-quant super-blocks.
fn dense_fits(ff: u32, act: Act) -> Result<(), String> {
    if ff > 0 && ff.is_multiple_of(q35::SUPER_BLOCK) && matches!(act, Act::SwiGlu { limit: None }) {
        Ok(())
    } else {
        Err(format!(
            "a dense FFN of {ff} values with {act:?}; the body runs SwiGLU without a limit over \
             whole super-blocks of {}",
            q35::SUPER_BLOCK
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

/// The router `m` runs on when it is the routed FFN the gated router and the
/// joined stacks run: its instance and pick count from the file's shape
/// ([`RouterDims::of`], which refuses a count or rule no instance serves).
pub(super) fn moe_fits(m: &Moe) -> Result<RouterDims, String> {
    use q35::EXPERT_FF;
    let router = RouterDims::of(MoeShape::of(m)).map_err(|e| e.to_string())?;
    let swiglu = matches!(m.act, Act::SwiGlu { limit: None });
    let plain = !m.router.hash && m.router.scale == 1.0;
    let shared = m.shared.is_some_and(|s| {
        s.ff == EXPERT_FF && s.sigmoid_gate && matches!(s.act, Act::SwiGlu { limit: None })
    });
    if router.gated() && m.expert_ff == EXPERT_FF && swiglu && plain && shared {
        Ok(router)
    } else {
        Err(format!(
            "experts {m:?}; the kernels take SwiGLU experts of {EXPERT_FF}, softmax renormalized \
             at scale 1, and a sigmoid-gated shared expert of {EXPERT_FF}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Flash, Kind35, kinds35, q35};
    use crate::linear::{KHeadMap as LinearMap, LinearShape};
    use model::arch::models::{
        Act, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap, LayerSpec, Mixer, Moe, Residual,
        Rope, RopeMode, Router, Score, Shared,
    };

    /// Qwen3.6's 16/2 heads.
    fn gqa() -> Mixer {
        gqa_of(16, 2)
    }

    fn gqa_of(heads: u32, kv_heads: u32) -> Mixer {
        Mixer::Gqa(Gqa {
            heads,
            kv_heads,
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
                experts: 256,
                top_k: 8,
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
                    Kind35::Gqa(Flash::Group)
                } else {
                    Kind35::Delta(shape)
                }
            })
            .collect();
        assert_eq!(got, want);
    }

    /// The routed shape comes from the file: the wide instance's 512/10
    /// and any pick count up to the lane picks run, an expert count no
    /// instance is built for is refused by name with its index.
    #[test]
    fn the_routed_shape_selects_its_instance() {
        let with = |experts: u32, top_k: u32| {
            let mut l = layer(delta());
            if let Ffn::Moe(m) = &mut l.ffn {
                m.experts = experts;
                m.top_k = top_k;
            }
            l
        };
        for (experts, top_k) in [(256u32, 8u32), (512, 10), (256, 1), (512, 32)] {
            let got = kinds35(&[with(experts, top_k)]);
            assert!(got.is_ok(), "{experts}/{top_k}: {got:?}");
        }
        for (experts, top_k) in [(384u32, 8u32), (256, 33), (256, 0)] {
            match kinds35(&[layer(delta()), with(experts, top_k)]) {
                Err(e) => assert!(e.to_string().contains("layer 1:"), "{e}"),
                Ok(k) => panic!("{experts}/{top_k} accepted: {k:?}"),
            }
        }
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

    /// Qwen3.5-27B's layers run: 24/4 heads take the pairs flash, 16/48
    /// delta heads the delta shape, and a dense FFN of 17408 the one-slot
    /// route; a group the body launches no flash for (3: neither eight nor a
    /// whole number of pairs) and a dense width of no whole super-block are
    /// refused by name.
    #[test]
    fn a_dense_qwen35_layer_runs_and_its_misfits_are_refused() {
        let dense = |mixer: Mixer, ff: u32| LayerSpec {
            ffn: Ffn::Dense {
                ff,
                act: Act::SwiGlu { limit: None },
            },
            ..layer(mixer)
        };
        let wide = Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Silu,
            },
            k_heads: 16,
            v_heads: 48,
            d: 128,
            conv: 4,
        });
        let got = kinds35(&[dense(wide.clone(), 17408), dense(gqa_of(24, 4), 17408)])
            .unwrap_or_else(|e| panic!("{e}"));
        let shape = LinearShape {
            n_k: 16,
            n_v: 48,
            map: LinearMap::Tiled,
        };
        assert_eq!(got, [Kind35::Delta(shape), Kind35::Gqa(Flash::Pairs)]);
        for (at, bad) in [
            (0usize, dense(gqa_of(24, 8), 17408)),
            (1, dense(wide, 17400)),
        ] {
            let mut layers = vec![dense(gqa_of(24, 4), 17408), dense(gqa_of(24, 4), 17408)];
            layers[at] = bad;
            match kinds35(&layers) {
                Err(e) => assert!(e.to_string().contains(&format!("layer {at}:")), "{e}"),
                Ok(k) => panic!("layer {at} accepted: {k:?}"),
            }
        }
    }
}
