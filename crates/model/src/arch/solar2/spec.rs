//! The solar-open2 reader's [`ModelSpec`]. A KDA layer is a [`Mixer::DeltaRule`] of
//! `attention.head_count` key and value heads whose decay is the softplus gate and whose write
//! strength is `2 · sigmoid(ssm_beta x)`; a GQA layer is the no-rope attention of the common
//! hybrid owner with the output gate its own projection computes. Every layer routes by the
//! common noaux block over SwiGLU experts, on a plain residual. The chat surface is the file's
//! template and pre-tokenizer with the parsers the family's server runs: Solar's tool-call
//! markup and its `<|think:start|>` / `<|think:end|>` reasoning span.

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, DeltaKind, DeltaRule, Gqa, KdaGate, LayerSpec, Mixer, ModelSpec, OutGate,
    ReasoningFormat, Residual, ToolFormat,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::hybrid::{Attn, nope_gqa};
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// The activation of every routed and shared expert.
const ACT: Act = Act::SwiGlu { limit: None };

/// The scale on `sigmoid(ssm_beta x)` (`allow_neg_eigval`): the write strength reaches 2.
const BETA_SCALE: f32 = 2.0;

/// The tool-call markup the family's template teaches.
const TOOLS: Option<ToolFormat> = Some(ToolFormat::TagKv);

/// The tags of the family's reasoning span, which its template's generation prompt opens and the
/// model closes.
const REASONING: Option<ReasoningFormat> = Some(ReasoningFormat::ThinkSpan {
    open: "<|think:start|>",
    close: "<|think:end|>",
});

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, None, None)?;
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` was read from, with the chat surface `chat` (its template and
/// pre-tokenizer) and the parsers the family declares ([`TOOLS`], [`REASONING`]).
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let attn = Attn {
        n_head: hp.n_head,
        key_len: hp.head_dim,
        value_len: hp.head_dim,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = match hp.kind(l) {
            Kind::Kda => Mixer::DeltaRule(DeltaRule {
                kind: DeltaKind::Kda {
                    gate: KdaGate::Softplus,
                    beta_scale: BETA_SCALE,
                },
                k_heads: spec_u32("attention.head_count", hp.n_head)?,
                v_heads: spec_u32("attention.head_count", hp.n_head)?,
                d: spec_u32("kda.head_dim", hp.kda.head_dim)?,
                conv: spec_u32("ssm.conv_kernel", hp.kda.conv)?,
            }),
            Kind::Gqa => Mixer::Gqa(Gqa {
                out_gate: OutGate::Proj,
                ..nope_gqa(&attn, hp.kv_heads[l])?
            }),
        };
        Ok(LayerSpec {
            mixer,
            ffn: hp.experts.ffn(ACT)?,
            residual: Residual::Plain,
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::SolarOpen2,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: None,
        chat: ChatSpec {
            tools: TOOLS,
            reasoning: REASONING,
            ..chat
        },
    })
}

#[cfg(test)]
mod tests {
    use models::{Act, DeltaKind, Ffn, KdaGate, Mixer, OutGate, Rope, RopeMode, Score};

    use super::super::hparams::tests::{CONV, EFF, GQA_LAYERS, K, KV, LAYERS, keys, tensors};
    use crate::arch::coverage;
    use crate::arch::synthetic::{V, verdict};

    /// The small header of the `hparams` tests with the pre-tokenizer and a chat template: its
    /// read.
    fn read(tag: &str, tensors: &[(String, Vec<u64>)]) -> Result<super::Read, String> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("solar-open")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        verdict(
            &format!("so2-spec-{tag}"),
            "solar-open2",
            &keys(),
            &global,
            tensors,
            super::read,
        )
    }

    /// The description is the table's: a NoPE GQA layer of 4 query heads and 2 key heads of 16
    /// with the gate projection at layers 0 and 4, a KDA layer of 4 heads (softplus decay, β
    /// scaled by 2) elsewhere, and a routed block with a bias, a shared expert and plain SwiGLU on
    /// every layer. Mutant: the gate as the packed one, the KDA gate sigmoid or its β scale 1.
    #[test]
    fn the_description_is_the_tables() {
        let read = read("spec", &tensors()).expect("the header reads");
        let spec = &read.spec;
        assert_eq!(spec.arch, models::Arch::SolarOpen2);
        assert_eq!((spec.hidden, spec.vocab, spec.ctx_train), (64, 2, 4096));
        assert!(spec.mtp.is_empty() && spec.hc.is_none());
        assert_eq!(spec.layers.len(), LAYERS);
        let act = Act::SwiGlu { limit: None };
        for (l, layer) in spec.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Gqa(g) => {
                    assert!(GQA_LAYERS.contains(&l), "layer {l} is GQA");
                    assert_eq!(
                        (g.heads, g.kv_heads, g.head_dim, g.value_dim),
                        (4, KV as u32, K as u32, K as u32)
                    );
                    assert_eq!(
                        g.rope,
                        Rope {
                            mode: RopeMode::None,
                            dims: 0,
                            base: 0.0,
                            yarn: None,
                            factors: false,
                        }
                    );
                    assert_eq!(g.out_gate, OutGate::Proj);
                    assert!(!g.qk_norm && g.select.is_none() && g.window.is_none() && !g.sinks);
                    assert!(g.value_scale.is_none());
                }
                Mixer::DeltaRule(d) => {
                    assert!(!GQA_LAYERS.contains(&l), "layer {l} is KDA");
                    assert_eq!(
                        d.kind,
                        DeltaKind::Kda {
                            gate: KdaGate::Softplus,
                            beta_scale: 2.0
                        }
                    );
                    assert_eq!(
                        (d.k_heads, d.v_heads, d.d, d.conv),
                        (4, 4, K as u32, CONV as u32)
                    );
                }
                other => panic!("layer {l} mixes with {other:?}"),
            }
            let Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l} is not routed");
            };
            assert_eq!(
                (m.experts, m.top_k, m.expert_ff, m.latent),
                (8, 2, EFF as u32, None)
            );
            assert_eq!(m.act, act);
            assert_eq!(m.router.score, Score::Sigmoid);
            assert!(m.router.bias && !m.router.hash);
            assert_eq!(m.router.norm, models::RouterNorm::SelectedSum);
            assert_eq!(m.router.scale, 1.0);
            let shared = m.shared.expect("a shared expert");
            assert_eq!(
                (shared.ff, shared.act, shared.sigmoid_gate),
                (EFF as u32, act, false)
            );
            assert!(layer.extras.is_empty());
        }
        assert_eq!(spec.chat.pre, "solar-open");
        assert_eq!(spec.chat.tools, Some(models::ToolFormat::TagKv));
        assert_eq!(
            spec.chat.reasoning,
            Some(models::ReasoningFormat::ThinkSpan {
                open: "<|think:start|>",
                close: "<|think:end|>"
            })
        );
    }

    /// Only what no program runs is listed: the test description's shapes (d 16, group 2, 8 experts) have
    /// no kernel, so its softplus-decay, β-scaled KDA layer is listed by its own item, while the GQA
    /// layer's sigmoid output gate, which a program runs at any shape, is not.
    // PIN(2026-10-20): the out gate left the list when its program (`attn_out_gate_sigmoid`) landed.
    #[test]
    fn what_no_program_runs_is_listed() {
        let read = read("coverage", &tensors()).expect("the header reads");
        let items: Vec<String> = coverage::check(&read.spec, &read.tensors)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        let listed = |needle: &str| items.iter().any(|f| f.contains(needle));
        let needle = "delta rule KDA: d 16, 4 heads, conv 4, softplus gate, beta scaled by 2";
        assert!(listed(needle), "{needle:?} not listed: {items:#?}");
        let gate = "attention output gate";
        assert!(!listed(gate), "{gate:?} listed: {items:#?}");
    }

    /// A tensor the table does not hold — a fused projection, a next-token block, a block past the
    /// model — fails the file with its name, never a plausible read.
    #[test]
    fn a_tensor_outside_the_table_fails_the_file() {
        for name in [
            "blk.0.attn_qkv.weight",
            "blk.8.attn_q.weight",
            "blk.7.nextn.eh_proj.weight",
        ] {
            let mut t = tensors();
            t.push((name.to_string(), vec![1]));
            let err = read("outside", &t).expect_err("the file is refused");
            assert!(err.contains(name), "{name} not named: {err}");
        }
    }
}
