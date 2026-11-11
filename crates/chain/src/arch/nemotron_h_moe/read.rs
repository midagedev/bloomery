//! A nemotron_h_moe chain's plans read from the file's header: each layer's kind from its
//! description, never from its number, and the trunk projections' types at one of the backend's
//! `allowed` list. The mixture's own tensors (the router, the stacks, the latent pair) are the
//! family's to check, their names held here.

use super::plan::{MoePlan, Plan};
use crate::ChainError;
use crate::hybrid::plan::LayerPlan;
use crate::hybrid::read::{mamba_plan, no_block, nope_gqa_plan};
use crate::site::SiteTy;
use gguf::Split;
use model::arch::models::{Act, Ffn, LayerSpec, Mixer, Moe, Score};
use model::arch::names::blk;
use model::arch::nemotron_h_moe::names::{
    ATTN_NORM, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_DOWN_SHEXP, FFN_GATE_INP, FFN_LATENT_DOWN,
    FFN_LATENT_UP, FFN_UP_EXPS, FFN_UP_SHEXP,
};

/// Layer `l`'s latent MoE of the description `m`: the dialect's mixture (the sigmoid router with
/// its selection bias, squared-ReLU experts with no norm around them, one squared-ReLU shared
/// expert with no gate), every part by its file name and the layer's own counts. The router's
/// renormalization and scale, the types the family's launches read and a count no router instance
/// is built for are the family's, refused at its load.
pub fn moe_plan(what: &'static str, l: usize, m: &Moe) -> Result<MoePlan, ChainError> {
    moe_fits(m).map_err(|d| ChainError::shape(what, format!("layer {l}: {d}")))?;
    Ok(MoePlan {
        norm: blk(l, ATTN_NORM),
        router: blk(l, FFN_GATE_INP),
        bias: blk(l, EXP_PROBS_B),
        n_expert: m.experts as usize,
        ff: m.expert_ff as usize,
        top_k: m.top_k as usize,
        latent_down: m.latent.map(|_| blk(l, FFN_LATENT_DOWN)),
        latent_up: m.latent.map(|_| blk(l, FFN_LATENT_UP)),
        up_exps: blk(l, FFN_UP_EXPS),
        down_exps: blk(l, FFN_DOWN_EXPS),
        shared_up: blk(l, FFN_UP_SHEXP),
        shared_down: blk(l, FFN_DOWN_SHEXP),
    })
}

/// Why the description `m` is not the latent MoE the family's kernels run: another score than the
/// sigmoid, no selection bias, a hash router, a norm around the experts, a gated or shared-less
/// expert beside them, or experts of another activation.
fn moe_fits(m: &Moe) -> Result<(), String> {
    let router = m.router.score == Score::Sigmoid && m.router.bias && !m.router.hash;
    let norms = !m.pre_exp_norm && m.post_exp_norm.is_none() && !m.sum_norm;
    let shared = matches!(
        m.shared,
        Some(shared) if shared.act == Act::ReluSqr && !shared.sigmoid_gate
    );
    if router && norms && shared && m.act == Act::ReluSqr {
        Ok(())
    } else {
        Err(format!(
            "experts {m:?}; the family runs the sigmoid router with a selection bias, \
             squared-ReLU experts with no norm around them, and one squared-ReLU shared expert \
             with no gate"
        ))
    }
}

/// Every layer's plan of a nemotron_h_moe chain over a `hidden`-value row, `layers` the
/// description's (`model::arch::nemotron_h_moe::spec::read`): a Mamba-2 mixer, an attention with
/// no rope or a latent mixture of routed experts with no mixer, each with no other block. Any
/// other layer is refused by name with the layer and what it holds.
pub fn layer_plans(
    file: &Split,
    what: &'static str,
    layers: &[LayerSpec],
    hidden: usize,
    allowed: &[SiteTy],
) -> Result<Vec<Plan>, ChainError> {
    layers
        .iter()
        .enumerate()
        .map(|(l, spec)| {
            let refuse = |d: String| ChainError::shape(what, format!("layer {l}: {d}"));
            match (&spec.mixer, &spec.ffn) {
                (Mixer::Mamba2(m), ffn) if no_block(ffn) => Ok(LayerPlan::Mamba(mamba_plan(
                    file, what, l, m, hidden, allowed,
                )?)),
                (Mixer::Gqa(g), ffn) if no_block(ffn) => Ok(LayerPlan::Attn(nope_gqa_plan(
                    file, what, l, g, hidden, allowed,
                )?)),
                (Mixer::None, Ffn::Moe(m)) => Ok(LayerPlan::Ffn(moe_plan(what, l, m)?)),
                (mixer, ffn) => Err(refuse(format!(
                    "{} with {ffn:?}; a nemotron_h_moe layer is a Mamba-2 or attention mixer \
                     with no feed-forward block, or a latent mixture of routed experts with no \
                     mixer",
                    mixer.what()
                ))),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::plan::{MoePlan, Plan};
    use super::layer_plans;
    use crate::hybrid::plan::{Counts, LayerPlan};
    use crate::site::SiteTy;
    use crate::test_file::{decl, header};
    use gguf::Split;
    use gguf::quant::GgmlType::{Q4_K, Q8_0};
    use model::arch::hybrid::{ATTN_K, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_IN, SSM_OUT};
    use model::arch::models::{
        Act, Ffn, Gqa, LayerSpec, Mamba2, Mixer, Moe, OutGate, Residual, Rope, RopeMode, Router,
        RouterNorm, Score, Shared,
    };
    use model::arch::names::blk;
    use model::arch::nemotron_h_moe::names::{FFN_GATE_INP, FFN_UP_EXPS};

    const WHAT: &str = "nemotron_h_moe test";
    const HIDDEN: usize = 256;
    const ALLOWED: &[SiteTy] = &[SiteTy::Nvfp4, SiteTy::Q8_0];

    fn layer(mixer: Mixer, ffn: Ffn) -> LayerSpec {
        LayerSpec {
            mixer,
            ffn,
            residual: Residual::Plain,
            extras: Vec::new(),
        }
    }

    fn none() -> Ffn {
        Ffn::Dense {
            ff: 0,
            act: Act::ReluSqr,
        }
    }

    fn mamba() -> LayerSpec {
        layer(
            Mixer::Mamba2(Mamba2 {
                d_inner: 512,
                d_state: 128,
                n_heads: 8,
                head_dim: 64,
                n_groups: 2,
                d_conv: 4,
                d_in_proj: 1544,
            }),
            none(),
        )
    }

    fn attn() -> LayerSpec {
        layer(
            Mixer::Gqa(Gqa {
                heads: 10,
                kv_heads: 2,
                head_dim: 128,
                value_dim: 128,
                rope: Rope {
                    mode: RopeMode::None,
                    dims: 0,
                    base: 0.0,
                    yarn: None,
                    factors: false,
                },
                qk_norm: false,
                o_norm: false,
                out_gate: OutGate::None,
                select: None,
                window: None,
                sinks: false,
                value_scale: None,
            }),
            none(),
        )
    }

    /// The dialect's latent mixture at `experts`/`top_k`/`ff`, in a `latent` of `LATENT` values or
    /// on the hidden row itself.
    fn moe(moe_of: impl FnOnce(&mut Moe)) -> LayerSpec {
        let mut m = Moe {
            experts: 128,
            top_k: 6,
            expert_ff: 512,
            latent: Some(160),
            pre_exp_norm: false,
            post_exp_norm: None,
            sum_norm: false,
            act: Act::ReluSqr,
            router: Router {
                score: Score::Sigmoid,
                bias: true,
                norm: RouterNorm::SelectedSum,
                scale: 5.0,
                hash: false,
            },
            shared: Some(Shared {
                ff: 5376,
                act: Act::ReluSqr,
                sigmoid_gate: false,
            }),
        };
        moe_of(&mut m);
        layer(Mixer::None, Ffn::Moe(m))
    }

    /// Layer `l`'s trunk sites for its kind (`m` or `a`), the header a MoE layer adds nothing to:
    /// its names are the description's, so no tensor of its own is read here.
    fn sites(l: usize, kind: char) -> Vec<gguf::write::TensorDecl> {
        let h = HIDDEN as u64;
        match kind {
            'm' => vec![
                decl(blk(l, SSM_IN), &[h, 1544], Q8_0),
                decl(blk(l, SSM_OUT), &[512, h], Q8_0),
            ],
            'a' => vec![
                decl(blk(l, ATTN_Q), &[h, 1280], Q8_0),
                decl(blk(l, ATTN_K), &[h, 256], Q8_0),
                decl(blk(l, ATTN_V), &[h, 256], Q8_0),
                decl(blk(l, ATTN_OUTPUT), &[1280, h], Q8_0),
            ],
            'e' => Vec::new(),
            other => panic!("kind {other}"),
        }
    }

    /// A file of the layers `kinds` names, in order.
    fn file(kinds: &str) -> Split {
        header(
            kinds
                .chars()
                .enumerate()
                .flat_map(|(l, k)| sites(l, k))
                .collect(),
        )
    }

    fn plans(kinds: &str, layers: &[LayerSpec]) -> Result<Vec<Plan>, String> {
        layer_plans(&file(kinds), WHAT, layers, HIDDEN, ALLOWED).map_err(|e| e.to_string())
    }

    /// A layer's plan is its description's kind, at any index, and a MoE layer's plan holds the
    /// file's names, the layer's own counts and the latent pair only where the description has a
    /// latent.
    #[test]
    fn kinds_come_from_the_description_not_the_layer_number() {
        let layers = [
            mamba(),
            moe(|_| {}),
            attn(),
            moe(|m| {
                m.experts = 240;
                m.top_k = 9;
                m.expert_ff = 640;
                m.latent = None;
            }),
        ];
        let got = plans("meae", &layers).unwrap_or_else(|e| panic!("{e}"));
        let kind = |p: &Plan| match p {
            LayerPlan::Mamba(_) => 'm',
            LayerPlan::Attn(_) => 'a',
            LayerPlan::Ffn(_) => 'e',
        };
        assert_eq!(got.iter().map(kind).collect::<String>(), "meae");
        assert_eq!(
            Counts::of(&got),
            Counts {
                mamba: 1,
                attn: 1,
                ffn: 2
            }
        );
        assert_eq!(
            got[1],
            LayerPlan::Ffn(MoePlan {
                norm: "blk.1.attn_norm.weight".into(),
                router: blk(1, FFN_GATE_INP),
                bias: "blk.1.exp_probs_b.bias".into(),
                n_expert: 128,
                ff: 512,
                top_k: 6,
                latent_down: Some("blk.1.ffn_latent_down.weight".into()),
                latent_up: Some("blk.1.ffn_latent_up.weight".into()),
                up_exps: blk(1, FFN_UP_EXPS),
                down_exps: "blk.1.ffn_down_exps.weight".into(),
                shared_up: "blk.1.ffn_up_shexp.weight".into(),
                shared_down: "blk.1.ffn_down_shexp.weight".into(),
            })
        );
        let LayerPlan::Ffn(p) = &got[3] else {
            panic!("layer 3 is {:?}", got[3]);
        };
        assert_eq!(
            (p.n_expert, p.ff, p.top_k, p.latent_down.is_some()),
            (240, 640, 9, false)
        );
    }

    /// A layer that is not one of the family's three kinds is refused by name with the layer: a
    /// mixture of another shape (another score, no selection bias, a hash router, a norm around
    /// the experts, a gated or shared-less expert beside them, experts of another activation), a
    /// dense feed-forward block, and a mixer with a block.
    #[test]
    fn a_layer_of_no_kind_of_the_family_is_refused_by_name() {
        let with = |edit: &dyn Fn(&mut Moe)| moe(|m| edit(m));
        let cases = [
            (
                "a softmax score",
                with(&|m: &mut Moe| {
                    m.router.score = Score::Softmax;
                    m.router.bias = false;
                    m.router.norm = RouterNorm::SelectedSum;
                }),
            ),
            (
                "no selection bias",
                with(&|m: &mut Moe| m.router.bias = false),
            ),
            ("a hash router", with(&|m: &mut Moe| m.router.hash = true)),
            (
                "a pre-experts norm",
                with(&|m: &mut Moe| m.pre_exp_norm = true),
            ),
            (
                "a post-experts norm",
                with(&|m: &mut Moe| m.post_exp_norm = Some(1e-5)),
            ),
            ("a sum norm", with(&|m: &mut Moe| m.sum_norm = true)),
            ("no shared expert", with(&|m: &mut Moe| m.shared = None)),
            (
                "a gated shared expert",
                with(&|m: &mut Moe| {
                    if let Some(s) = &mut m.shared {
                        s.sigmoid_gate = true;
                    }
                }),
            ),
            (
                "swiglu experts",
                with(&|m: &mut Moe| {
                    m.act = Act::SwiGlu { limit: None };
                    if let Some(s) = &mut m.shared {
                        s.act = Act::SwiGlu { limit: None };
                    }
                }),
            ),
            (
                "a dense block",
                layer(
                    Mixer::None,
                    Ffn::Dense {
                        ff: 1024,
                        act: Act::ReluSqr,
                    },
                ),
            ),
            (
                "a mixer with a block",
                layer(
                    mamba().mixer,
                    Ffn::Dense {
                        ff: 1024,
                        act: Act::ReluSqr,
                    },
                ),
            ),
        ];
        for (name, bad) in cases {
            let err = plans("em", &[moe(|_| {}), bad])
                .err()
                .unwrap_or_else(|| name.to_string());
            let want = if name == "a dense block" || name == "a mixer with a block" {
                "a nemotron_h_moe layer is a Mamba-2 or attention mixer"
            } else {
                "the family runs the sigmoid router with a selection bias"
            };
            assert!(
                err.starts_with("nemotron_h_moe test: layer 1: ") && err.contains(want),
                "{name}: {err}"
            );
        }
    }

    /// A trunk site of a type outside the backend's list is refused by name with the site and the
    /// list, before any later layer is read; the backend's own types read.
    #[test]
    fn a_trunk_site_outside_the_backends_list_is_refused_by_name() {
        let layers = [moe(|_| {}), attn()];
        let mut bad = sites(1, 'a');
        bad[0] = decl(blk(1, ATTN_Q), &[HIDDEN as u64, 1280], Q4_K);
        let err = layer_plans(&header(bad), WHAT, &layers, HIDDEN, ALLOWED)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains(&blk(1, "attn_q.weight"))
                && err.contains("this site launches NVFP4, Q8_0"),
            "{err}"
        );
        let ok = plans("ea", &layers).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(&ok[1], LayerPlan::Attn(a) if a.q_ty == SiteTy::Q8_0));
    }
}
