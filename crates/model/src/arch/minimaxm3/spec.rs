//! The minimax-m3 reader's [`ModelSpec`]. Every layer is GQA with the per-head q/k norm and a
//! partial NeoX rope on a plain residual. A dense block is `swigluoai`. A routed block is the
//! sigmoid noaux_tc router over `swigluoai` experts beside an ungated shared one: the selection
//! bias steers the choice only, the kept scores are renormalized and scaled by
//! `expert_weights_scale`. A sparse-attention layer attends the blocks its indexer picks
//! ([`Selector::BlockMax`]), whichever block it runs; a full-attention layer attends every
//! position. The chat surface carries no tool-call parser; its reasoning span is the pair the
//! template's generation prompt opens and the model closes.

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, OutGate, ReasoningFormat, Residual,
    Rope, RopeMode, Selector,
};

use super::hparams::{Attn, FfnKind, Hparams, SWIGLU_ALPHA, SWIGLU_LIMIT};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// The activation of every dense, routed and shared block.
const ACT: Act = Act::SwigluOai {
    alpha: SWIGLU_ALPHA,
    limit: SWIGLU_LIMIT,
};

/// The tags of the family's reasoning span, which its template's generation prompt opens and the
/// model closes.
const REASONING: Option<ReasoningFormat> = Some(ReasoningFormat::ThinkSpan {
    open: "<mm:think>",
    close: "</mm:think>",
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

/// The description of the file `hp` was read from, with the chat surface `chat`.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let rope = Rope {
        mode: RopeMode::Neox,
        dims: spec_u32("rope.dimension_count", hp.rope.dims)?,
        base: hp.rope.base,
        yarn: None,
        factors: false,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let kind = hp.kind(l);
        let select = if kind.attn == Attn::Sparse {
            let m = &hp.msa;
            Some(Selector::BlockMax {
                heads: spec_u32("attention.indexer.head_count", m.heads)?,
                d: spec_u32("attention.indexer.key_length", m.d)?,
                top_k: spec_u32("attention.indexer.top_k", m.top_k)?,
                block: spec_u32("attention.indexer.block_size", m.block)?,
                local: spec_u32("attention.indexer.local_blocks", m.local)?,
                rope,
            })
        } else {
            None
        };
        let mixer = Mixer::Gqa(Gqa {
            heads: spec_u32("attention.head_count", hp.n_head)?,
            kv_heads: spec_u32("attention.head_count_kv", hp.n_head_kv)?,
            head_dim: spec_u32("attention.key_length", hp.head_dim)?,
            // The reader holds the file to one width for q, k and v.
            value_dim: spec_u32("attention.key_length", hp.head_dim)?,
            rope,
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::None,
            select,
            window: None,
            sinks: false,
            value_scale: None,
        });
        let ffn = match kind.ffn {
            FfnKind::Dense => {
                // The reader read the dense width whenever a layer is dense.
                let ff = hp.dense_ff.ok_or_else(|| PlacementError::Metadata {
                    key: "feed_forward_length".to_string(),
                    detail: format!("is absent, and layer {l} is a dense block"),
                })?;
                Ffn::Dense {
                    ff: spec_u32("feed_forward_length", ff)?,
                    act: ACT,
                }
            }
            FfnKind::Routed => hp.experts.ffn(ACT)?,
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Plain,
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::MiniMaxM3,
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
            reasoning: REASONING,
            ..chat
        },
    })
}

#[cfg(test)]
mod tests {
    use models::{Act, Ffn, Mixer, OutGate, Score, Selector};

    use super::super::hparams::tests::{keys, tensors};
    use super::{SWIGLU_ALPHA, SWIGLU_LIMIT};
    use crate::arch::coverage;
    use crate::arch::synthetic::{V, header_shaped};

    /// The small header of the `hparams` tests with `tokenizer.ggml.pre` `minimax-m2` and a chat
    /// template: its read and the coverage check's items.
    fn read(tag: &str) -> (super::Read, Vec<String>) {
        let global = [
            ("tokenizer.ggml.pre", V::Str("minimax-m2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "minimax-m3", &keys(), &global, &tensors());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        let read = read.expect("the header reads");
        let items = coverage::check(&read.spec, &read.tensors)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        (read, items)
    }

    /// The description is the table's: GQA 8/2 x 16 with the q/k norm and the partial NeoX rope
    /// on every layer; the first two dense, the rest sparse with a block-max selector of the
    /// indexer's keys, the sigmoid router with its bias and an ungated shared expert, and
    /// `swigluoai` in every block. Mutant: the selector on a dense layer, the router bias off, or
    /// the activation's constants swapped.
    #[test]
    fn the_description_is_the_tables() {
        let (read, _) = read("mm3-spec");
        let spec = &read.spec;
        assert_eq!(spec.arch, models::Arch::MiniMaxM3);
        assert_eq!((spec.hidden, spec.vocab, spec.ctx_train), (64, 2, 4096));
        assert!(spec.mtp.is_empty());
        assert_eq!(
            spec.chat.reasoning,
            Some(models::ReasoningFormat::ThinkSpan {
                open: "<mm:think>",
                close: "</mm:think>"
            })
        );
        let act = Act::SwigluOai {
            alpha: SWIGLU_ALPHA,
            limit: SWIGLU_LIMIT,
        };
        assert_eq!((SWIGLU_ALPHA, SWIGLU_LIMIT), (1.702, 7.0));
        assert_eq!(spec.layers.len(), 5);
        for (l, layer) in spec.layers.iter().enumerate() {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("layer {l} is not GQA");
            };
            assert_eq!(
                (g.heads, g.kv_heads, g.head_dim, g.value_dim),
                (8, 2, 16, 16)
            );
            assert_eq!((g.rope.dims, g.rope.base), (8, 5_000_000.0));
            assert!(g.qk_norm && g.out_gate == OutGate::None && g.window.is_none() && !g.sinks);
            assert!(g.value_scale.is_none());
            if l < 2 {
                assert_eq!(g.select, None, "layer {l} attends every position");
                assert_eq!(layer.ffn, Ffn::Dense { ff: 48, act });
                continue;
            }
            let Some(Selector::BlockMax {
                heads,
                d,
                top_k,
                block,
                local,
                rope,
            }) = g.select.clone()
            else {
                panic!("layer {l} has no block-max selector");
            };
            assert_eq!((heads, d, top_k, block, local), (2, 16, 3, 4, 1));
            assert_eq!(rope, g.rope);
            let Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l} is not routed");
            };
            assert_eq!(
                (m.experts, m.top_k, m.expert_ff, m.latent),
                (8, 2, 24, None)
            );
            assert_eq!(m.act, act);
            assert_eq!(m.router.score, Score::Sigmoid);
            assert!(m.router.bias && !m.router.hash);
            assert_eq!(m.router.norm, models::RouterNorm::SelectedSum);
            assert_eq!(m.router.scale, 2.0);
            let shared = m.shared.expect("a shared expert");
            assert_eq!(
                (shared.ff, shared.act, shared.sigmoid_gate),
                (24, act, false)
            );
        }
    }

    /// Only the features no program runs are listed: the swigluoai blocks, the block-max
    /// indexer, the sigmoid router and the program itself — and the chat surface's template
    /// tool-call parser. Each is a coverage item, so a load refuses the file by name rather than
    /// running it wrong. The `minimax-m2` pre-tokenizer is run by the tokenizer crate and is no item.
    #[test]
    fn what_no_program_runs_is_listed() {
        let (_, items) = read("mm3-coverage");
        let listed = |needle: &str| items.iter().any(|f| f.contains(needle));
        for needle in [
            "swigluoai",
            "block-max indexer: 2 heads x 16, blocks of 4 keep 3 (1 local)",
        ] {
            assert!(listed(needle), "{needle:?} not listed: {items:#?}");
        }
        assert!(
            !listed("pre-tokenizer"),
            "a pre-tokenizer is listed: {items:#?}"
        );
    }
}
