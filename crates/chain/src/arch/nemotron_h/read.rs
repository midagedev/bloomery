//! A nemotron_h chain's plans read from the file's header: each layer's kind from its description,
//! never from its number, and every site's type at one of the backend's `allowed` list.

use super::plan::{MlpPlan, Plan};
use crate::ChainError;
use crate::hybrid::plan::LayerPlan;
use crate::hybrid::read::{mamba_plan, no_block, nope_gqa_plan};
use crate::site::{SiteTy, file_site};
use gguf::Split;
use model::arch::models::{Act, Ffn, LayerSpec, Mixer};
use model::arch::names::blk;
use model::arch::nemotron_h::names::{ATTN_NORM, FFN_DOWN, FFN_UP};

/// Layer `l`'s dense MLP of `ff` values over a `hidden`-value row, its two sites' types from
/// `file`'s header at one of `allowed`.
pub fn mlp_plan(
    file: &Split,
    what: &'static str,
    l: usize,
    ff: u32,
    hidden: usize,
    allowed: &[SiteTy],
) -> Result<MlpPlan, ChainError> {
    let ff = ff as usize;
    let (up, down) = (blk(l, FFN_UP), blk(l, FFN_DOWN));
    Ok(MlpPlan {
        norm: blk(l, ATTN_NORM),
        ff,
        up_ty: file_site(file, what, &up, (ff, hidden), allowed)?,
        down_ty: file_site(file, what, &down, (hidden, ff), allowed)?,
        up,
        down,
    })
}

/// Every layer's plan of a nemotron_h chain over a `hidden`-value row, `layers` the description's
/// (`model::arch::nemotron_h::spec::read`): a Mamba-2 mixer, an attention with no rope or a
/// squared-ReLU MLP with no mixer, each with no other block. Any other layer is refused by name
/// with the layer and what it holds.
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
                (
                    Mixer::None,
                    Ffn::Dense {
                        ff,
                        act: Act::ReluSqr,
                    },
                ) if *ff > 0 => Ok(LayerPlan::Ffn(mlp_plan(
                    file, what, l, *ff, hidden, allowed,
                )?)),
                (mixer, ffn) => Err(refuse(format!(
                    "{} with {ffn:?}; a nemotron_h layer is a Mamba-2 or attention mixer with no \
                     feed-forward block, or a squared-ReLU MLP with no mixer",
                    mixer.what()
                ))),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::plan::{MlpPlan, Plan};
    use super::layer_plans;
    use crate::hybrid::plan::{Counts, LayerPlan};
    use crate::hybrid::test_file::{decl, header};
    use crate::site::SiteTy;
    use gguf::Split;
    use gguf::quant::GgmlType::{self, Q4_K, Q5_0, Q8_0};
    use model::arch::hybrid::{ATTN_K, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_IN, SSM_OUT};
    use model::arch::models::{
        Act, Ffn, Gqa, LayerSpec, Mamba2, Mixer, OutGate, Residual, Rope, RopeMode,
    };
    use model::arch::names::blk;
    use model::arch::nemotron_h::names::{FFN_DOWN, FFN_UP};

    const WHAT: &str = "nemotron_h test";
    const HIDDEN: usize = 256;
    const FF: u32 = 1024;
    const ALLOWED: &[SiteTy] = &[SiteTy::Q5_0, SiteTy::Q8_0, SiteTy::Q4K];

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

    fn mlp() -> LayerSpec {
        layer(
            Mixer::None,
            Ffn::Dense {
                ff: FF,
                act: Act::ReluSqr,
            },
        )
    }

    /// Layer `l`'s sites for its kind (`m`, `a` or `p`), the MLP's up at `up`.
    fn sites(l: usize, kind: char, up: GgmlType) -> Vec<gguf::write::TensorDecl> {
        let h = HIDDEN as u64;
        match kind {
            'm' => vec![
                decl(blk(l, SSM_IN), &[h, 1544], Q5_0),
                decl(blk(l, SSM_OUT), &[512, h], Q8_0),
            ],
            'a' => vec![
                decl(blk(l, ATTN_Q), &[h, 1280], Q5_0),
                decl(blk(l, ATTN_K), &[h, 256], Q5_0),
                decl(blk(l, ATTN_V), &[h, 256], Q8_0),
                decl(blk(l, ATTN_OUTPUT), &[1280, h], Q5_0),
            ],
            'p' => vec![
                decl(blk(l, FFN_UP), &[h, u64::from(FF)], up),
                decl(blk(l, FFN_DOWN), &[u64::from(FF), h], Q8_0),
            ],
            other => panic!("kind {other}"),
        }
    }

    /// A file of the layers `kinds` names, in order.
    fn file(kinds: &str, up: GgmlType) -> Split {
        header(
            kinds
                .chars()
                .enumerate()
                .flat_map(|(l, k)| sites(l, k, up))
                .collect(),
        )
    }

    fn plans(kinds: &str, layers: &[LayerSpec]) -> Result<Vec<Plan>, String> {
        layer_plans(&file(kinds, Q5_0), WHAT, layers, HIDDEN, ALLOWED).map_err(|e| e.to_string())
    }

    /// A layer's plan is its description's kind, at any index: Mamba-2, MLP and attention twice
    /// over read as those kinds in that order, the MLP's names, width and types are the file's,
    /// and the counts are per kind.
    #[test]
    fn kinds_come_from_the_description_not_the_layer_number() {
        let layers = [mamba(), mlp(), attn(), mamba(), mlp(), attn()];
        let got = plans("mpampa", &layers).unwrap_or_else(|e| panic!("{e}"));
        let kind = |p: &Plan| match p {
            LayerPlan::Mamba(_) => 'm',
            LayerPlan::Attn(_) => 'a',
            LayerPlan::Ffn(_) => 'p',
        };
        assert_eq!(got.iter().map(kind).collect::<String>(), "mpampa");
        assert_eq!(
            Counts::of(&got),
            Counts {
                mamba: 2,
                attn: 2,
                ffn: 2
            }
        );
        assert_eq!(
            got[4],
            LayerPlan::Ffn(MlpPlan {
                norm: "blk.4.attn_norm.weight".into(),
                up: "blk.4.ffn_up.weight".into(),
                down: "blk.4.ffn_down.weight".into(),
                ff: 1024,
                up_ty: SiteTy::Q5_0,
                down_ty: SiteTy::Q8_0,
            })
        );
        let LayerPlan::Mamba(m) = &got[3] else {
            panic!("layer 3 is {:?}", got[3]);
        };
        assert_eq!(m.ssm_in, "blk.3.ssm_in.weight");
        let LayerPlan::Attn(a) = &got[5] else {
            panic!("layer 5 is {:?}", got[5]);
        };
        assert_eq!(
            (a.n_head, a.n_kv, a.q.as_str()),
            (10, 2, "blk.5.attn_q.weight")
        );
    }

    /// A layer that is not one of the family's three kinds is refused by name with the layer: an
    /// MLP of another activation or of no width, a mixer with a feed-forward block, and a mixer the
    /// kernels do not run.
    #[test]
    fn a_layer_of_no_kind_of_the_family_is_refused_by_name() {
        let swiglu = layer(
            Mixer::None,
            Ffn::Dense {
                ff: FF,
                act: Act::SwiGlu { limit: None },
            },
        );
        let empty = layer(
            Mixer::None,
            Ffn::Dense {
                ff: 0,
                act: Act::ReluSqr,
            },
        );
        let with_block = |mixer: Mixer| {
            layer(
                mixer,
                Ffn::Dense {
                    ff: FF,
                    act: Act::ReluSqr,
                },
            )
        };
        let cases = [
            ("swiglu", swiglu),
            ("empty", empty),
            ("mamba with a block", with_block(mamba().mixer)),
            ("attention with a block", with_block(attn().mixer)),
        ];
        for (name, bad) in cases {
            let err = plans("pm", &[mlp(), bad]).err().unwrap_or_default();
            assert!(
                err.starts_with("nemotron_h test: layer 1: ")
                    && err.contains("a nemotron_h layer is a Mamba-2 or attention mixer"),
                "{name}: {err}"
            );
        }
        let err = plans("m", &[attn()])
            .err()
            .unwrap_or_else(|| "read".to_string());
        assert!(
            err.starts_with("nemotron_h test: blk.0.attn_q.weight is not in the file"),
            "{err}"
        );
    }

    /// A site of a type outside the backend's list is refused by name with the site, before any
    /// later layer is read.
    #[test]
    fn a_site_type_outside_the_backends_list_is_refused_by_name() {
        let layers = [mlp(), mlp()];
        let err = layer_plans(
            &file("pp", Q4_K),
            WHAT,
            &layers,
            HIDDEN,
            &[SiteTy::Q5_0, SiteTy::Q8_0],
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(
            err.contains("blk.0.ffn_up.weight") && err.contains("this site launches Q5_0, Q8_0"),
            "{err}"
        );
        let ok = layer_plans(&file("pp", Q4_K), WHAT, &layers, HIDDEN, ALLOWED)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(&ok[0], LayerPlan::Ffn(p) if p.up_ty == SiteTy::Q4K));
    }
}
