//! The Kimi K3 reader's [`ModelSpec`]. A KDA layer is a [`Mixer::DeltaRule`] of
//! `attention.head_count` key and value heads of `kda.head_dim` values whose output gate is one
//! full-width projection; a latent layer caches the `kv_lora_rank` latent and the
//! `rope.dimension_count` key values no rope turns a position, absorbs the per-head up
//! projections (`attn_k_b`, `attn_v_b`) and gates its heads' outputs by a projection of the layer
//! input (`src/models/kimi-k3.cpp`). The first layer is a dense `situ` block and the rest the
//! sigmoid router over `situ` experts that run in a latent, their weighted sum normed before the
//! up projection, beside the shared stack that reads the hidden row. Every sublayer and the head
//! read the cross-layer residual's mix ([`Residual::AttnRes`]); no stream is carried beside the
//! running sum. The file's tokenizer is `kimi-k2`'s.

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, DeltaKind, DeltaRule, Ffn, Latent, LatentOut, LatentUp, LayerSpec, Mixer,
    ModelSpec, Moe, OutGate, Residual, Source,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// The pre-tokenizer the file names: Kimi K2's, which K3 keeps.
pub const PRE: &str = "kimi-k2";

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, None, None)?;
    if chat.pre != PRE {
        return Err(PlacementError::Metadata {
            key: "tokenizer.ggml.pre".to_string(),
            detail: format!("is {:?}; a Kimi K3 file's is {PRE:?}", chat.pre),
        });
    }
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` was read from, with the chat surface `chat`.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let act = Act::Situ {
        gate_beta: hp.situ.gate_beta,
        value_beta: hp.situ.value_beta,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = match hp.kind(l) {
            Kind::Kda => Mixer::DeltaRule(DeltaRule {
                kind: DeltaKind::KdaFullGate { gate: hp.gate },
                k_heads: spec_u32("attention.head_count", hp.n_head)?,
                v_heads: spec_u32("attention.head_count", hp.n_head)?,
                d: spec_u32("kda.head_dim", hp.kda.head_dim)?,
                conv: spec_u32("ssm.conv_kernel", hp.kda.conv)?,
            }),
            Kind::Mla => {
                let m = &hp.mla;
                Mixer::Latent(Latent {
                    heads: spec_u32("attention.head_count", hp.n_head)?,
                    q_lora: spec_u32("attention.q_lora_rank", m.q_lora)?,
                    latent: spec_u32("attention.kv_lora_rank", m.kv_lora)?,
                    up: LatentUp::Absorbed {
                        qk: spec_u32("attention.key_length_mla", m.head_k)?,
                        v: spec_u32("attention.value_length_mla", m.head_v)?,
                    },
                    rope: None,
                    q_head_norm: false,
                    ssmax: false,
                    pe: spec_u32("rope.dimension_count", m.pe)?,
                    cache: Source::Own(()),
                    out_gate: OutGate::Proj,
                    out: LatentOut::Plain,
                    window: None,
                    sinks: false,
                    compress: None,
                    select: None,
                })
            }
        };
        let ffn = if hp.dense(l) {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.dense_ff)?,
                act,
            }
        } else {
            Ffn::Moe(Moe {
                latent: Some(spec_u32("expert_latent_length", hp.latent)?),
                sum_norm: true,
                ..hp.experts.moe(act)?
            })
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::AttnRes {
                block: spec_u32("attn_res.block_size", hp.attn_res_block)?,
            },
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::KimiK3,
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
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::{Act, DeltaKind, Ffn, LatentUp, Mixer, OutGate, Residual, Score, Selector};

    use super::super::hparams::tests::{keys, tensors, typed};
    use crate::arch::synthetic::{V, header_typed};
    use crate::placement::PlacementError;

    /// The small header of the `hparams` tests with `tokenizer.ggml.pre` `pre`, its read.
    fn read_with(
        tag: &str,
        pre: &'static str,
        tensors: &[(String, Vec<u64>)],
    ) -> Result<super::Read, PlacementError> {
        let global = [
            ("tokenizer.ggml.pre", V::Str(pre)),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_typed(tag, "kimi-k3", &keys(), &global, &typed(tensors, &[]));
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split);
        let _ = std::fs::remove_file(&path);
        read
    }

    /// The description is the table's: a Kimi delta rule of 2 heads of 128 with the full-width
    /// output gate and the sigmoid decay bounded at -5 on layers 0, 1 and 3; a latent layer on 2
    /// caching 512 latent values and 64 unrotated ones, absorbing 192-wide queries into 128-wide
    /// values, gated by a projection and rotating nothing; the first layer a dense `situ` block
    /// and the rest the sigmoid router over `situ` experts in a latent of 256 with their sum
    /// normed, beside an ungated shared stack of 512; every layer on the cross-layer residual with
    /// a bank every 2 layers and no hyper-connection streams. Mutant: the gate kind of the delta
    /// layers, the latent or the sum norm left off the routed block, or the residual another kind.
    #[test]
    fn the_description_is_the_tables() {
        let read = read_with("kimi-k3-spec", "kimi-k2", &tensors()).expect("the header reads");
        let spec = &read.spec;
        assert_eq!(spec.arch, models::Arch::KimiK3);
        assert_eq!((spec.hidden, spec.vocab, spec.ctx_train), (512, 2, 4096));
        assert!(spec.mtp.is_empty() && spec.engram.is_none() && spec.encoder.is_none());
        assert!(spec.hc.is_none() && spec.logit_epilogue.is_none());
        let act = Act::Situ {
            gate_beta: 4.0,
            value_beta: 25.0,
        };
        assert_eq!(spec.layers.len(), 4);
        for (l, layer) in spec.layers.iter().enumerate() {
            assert_eq!(layer.residual, Residual::AttnRes { block: 2 }, "layer {l}");
            assert!(layer.extras.is_empty());
            match &layer.mixer {
                Mixer::DeltaRule(d) => {
                    assert!([0, 1, 3].contains(&l), "layer {l}");
                    assert_eq!(
                        d.kind,
                        DeltaKind::KdaFullGate {
                            gate: models::KdaGate::Sigmoid { lower_bound: -5.0 }
                        }
                    );
                    assert_eq!((d.k_heads, d.v_heads, d.d, d.conv), (2, 2, 128, 4));
                }
                Mixer::Latent(a) => {
                    assert_eq!(l, 2);
                    assert_eq!((a.heads, a.q_lora, a.latent, a.pe), (2, 256, 512, 64));
                    assert_eq!(a.up, LatentUp::Absorbed { qk: 192, v: 128 });
                    assert_eq!(a.out_gate, OutGate::Proj);
                    assert!(a.rope.is_none() && !a.q_head_norm && !a.sinks && !a.ssmax);
                    assert!(a.window.is_none() && a.compress.is_none());
                    assert_eq!(a.select, None::<Selector>);
                }
                other => panic!("layer {l}: {}", other.what()),
            }
            if l == 0 {
                assert_eq!(layer.ffn, Ffn::Dense { ff: 768, act });
                continue;
            }
            let Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l} is not routed");
            };
            assert_eq!(
                (m.experts, m.top_k, m.expert_ff, m.latent),
                (4, 2, 256, Some(256))
            );
            assert!(m.sum_norm && !m.pre_exp_norm && m.post_exp_norm.is_none());
            assert_eq!(m.act, act);
            assert_eq!(m.router.score, Score::Sigmoid);
            assert!(m.router.bias && !m.router.hash);
            assert_eq!(m.router.norm, models::RouterNorm::SelectedSum);
            assert_eq!(m.router.scale, 1.0);
            let shared = m.shared.expect("a shared stack");
            assert_eq!(
                (shared.ff, shared.act, shared.sigmoid_gate),
                (512, act, false)
            );
        }
        assert_eq!(spec.chat.pre, "kimi-k2");
        assert!(spec.chat.tools.is_none() && spec.chat.reasoning.is_none());
        assert!(read.defaults.is_empty(), "{:?}", read.defaults);
    }

    /// The classifier lists every name no role claims: the low-rank gate pair a Kimi linear file
    /// carries, a fused projection and a next-token tensor are not Kimi K3's, and the file is
    /// refused with them named. Mutant: the low-rank pair left among the delta stems.
    #[test]
    fn a_tensor_with_no_role_refuses_the_file_by_name() {
        let mut t = tensors();
        t.push(("blk.1.ssm_g_a.weight".into(), vec![512, 128]));
        t.push(("blk.1.ssm_g_b.weight".into(), vec![128, 256]));
        t.push(("blk.3.nextn.eh_proj.weight".into(), vec![1024, 512]));
        match read_with("kimi-k3-spec-unclassified", "kimi-k2", &t) {
            Ok(_) => panic!("read"),
            Err(PlacementError::Unclassified { names }) => {
                assert_eq!(
                    names,
                    [
                        "blk.1.ssm_g_a.weight",
                        "blk.1.ssm_g_b.weight",
                        "blk.3.nextn.eh_proj.weight"
                    ]
                );
            }
            Err(other) => panic!("{other}"),
        }
    }

    /// A pre-tokenizer other than Kimi K2's is refused by its key; the roles are the table's: the
    /// final mix's scores sit with the head, a layer's with its sublayer, the latent pair and the
    /// routed norm with the router.
    #[test]
    fn the_roles_and_the_pre_tokenizer_are_the_tables() {
        match read_with("kimi-k3-spec-pre", "deepseek-v3", &tensors()) {
            Ok(_) => panic!("read"),
            Err(e) => assert!(e.to_string().contains("tokenizer.ggml.pre"), "{e}"),
        }
        let read = read_with("kimi-k3-spec-roles", "kimi-k2", &tensors()).expect("reads");
        let role = |name: &str| {
            read.tensors
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| (t.role, t.layer))
        };
        use crate::placement::Role;
        assert_eq!(role("output_res_score.weight"), Some((Role::Head, None)));
        assert_eq!(
            role("blk.2.attn_res_score.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role("blk.2.ffn_res_score.weight"),
            Some((Role::FfnNorm, Some(2)))
        );
        for stem in ["ffn_routed_down", "ffn_routed_up", "ffn_routed_norm"] {
            assert_eq!(
                role(&format!("blk.1.{stem}.weight")),
                Some((Role::Router, Some(1))),
                "{stem}"
            );
        }
        assert_eq!(
            role("blk.1.ffn_gate_exps.weight"),
            Some((Role::RoutedExperts, Some(1)))
        );
    }
}
