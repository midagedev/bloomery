//! The minimax-m3 reader's [`ModelSpec`]. Every layer is GQA with the per-head q/k norm and a
//! partial NeoX rope on a plain residual. A dense block is `swigluoai`. A routed block is the
//! sigmoid noaux_tc router over `swigluoai` experts beside an ungated shared one: the selection
//! bias steers the choice only, the kept scores are renormalized and scaled by
//! `expert_weights_scale`. A sparse-attention layer attends the blocks its indexer picks
//! ([`Selector::BlockMax`]), whichever block it runs; a full-attention layer attends every
//! position. The chat surface carries no tool-call parser; its reasoning span is the pair the
//! template's generation prompt opens and the model closes.

use gguf::Split;
use models::shape::RouterBody;
use models::{
    Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, Need, OutGate, ReasoningFormat,
    Residual, Rope, RopeMode, Selector,
};

use super::hparams::{Attn, FfnKind, Hparams, SWIGLU_ALPHA, SWIGLU_LIMIT};
use super::roles;
use crate::arch::coverage::{self, Available, Program};
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

/// What the minimax-m3 body runs that the common table ([`coverage::AVAILABLE`]) does not list
/// for it: one row per kernel instance, citing the constant or function that owns it.
pub const AVAILABLE: &[Available] = &[
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "gpu/src/flash_gqa.rs GROUP_16 (the sparse layers over the picked blocks, \
             enqueue_pass_p16_blocks)",
        runs: coverage::gqa_group_16,
    },
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "gpu-deepseek41/src/shared_routed.rs (the ungated shared expert beside the swigluoai ones)",
        runs: coverage::ungated_shared,
    },
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "gpu/src/rope_neox.rs head_norm_neox_append_128_partial (HEAD, ROT_PARTIAL: 64 of 128 dims)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 128,
                    mode: RopeMode::Neox,
                    dims: 64
                }
            )
        },
    },
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "gpu-minimaxm3/src/attn.rs MsaShape::check (HEADS, DIM, BLOCK, ROT; top_k up to \
             MAX_TOP_K, 32, and local up to top_k: runtime::msa::Msa::new)",
        runs: |n| {
            matches!(
                n,
                Need::BlockMax {
                    heads: 4,
                    d: 128,
                    top_k,
                    block: 128,
                    local,
                    rope: 64,
                } if *top_k <= 32 && runtime::msa::Msa::new(*top_k, 128, *local).is_ok()
            )
        },
    },
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "models/src/shape.rs ROUTERS, the BiasedSigmoid body at 128 experts with the top 4 \
             (gpu-deepseek41 router sigmoid_128_4, gpu-minimaxm3/src/ffn.rs)",
        runs: |n| {
            matches!(n, Need::Router(s) if (s.experts, s.top_k) == (128, 4))
                && coverage::routes(n, &[RouterBody::BiasedSigmoid])
        },
    },
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "gpu-deepseek41/src/shared_routed.rs dense (the leading blocks under Act::SwigluOai; \
             the down reads h as q8_1 blocks of 32)",
        runs: |n| matches!(n, Need::SwigluOai { ff } if ff.is_multiple_of(32)),
    },
    Available {
        programs: &[Program::MiniMaxM3Body],
        at: "gpu-deepseek41/src/shared_routed.rs rule_of (the routed and shared experts under \
             Act::SwigluOai; host.rs HostLayer::build_oai)",
        runs: |n| {
            matches!(
                n,
                Need::ExpertAct {
                    act: Act::SwigluOai { .. },
                    ..
                }
            )
        },
    },
];

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

    use super::super::hparams::tests::{keys, real, tensors};
    use super::{SWIGLU_ALPHA, SWIGLU_LIMIT, spec_of};
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

    /// The selector row is the kernels' geometry: 4 index heads x 128 over blocks of
    /// 128 roped over 64, keeping 1 to 32 blocks with 1 to `top_k` of them local. Any other
    /// geometry or count is an item.
    #[test]
    fn the_selector_row_is_the_kernels_geometry() {
        use models::Need;

        use super::AVAILABLE;
        let need = |heads, d, top_k, block, local, rope| Need::BlockMax {
            heads,
            d,
            top_k,
            block,
            local,
            rope,
        };
        let runs = |n: &Need| AVAILABLE.iter().any(|a| (a.runs)(n));
        for n in [need(4, 128, 16, 128, 1, 64), need(4, 128, 32, 128, 32, 64)] {
            assert!(runs(&n), "{n} is the kernels'");
        }
        for n in [
            need(2, 128, 16, 128, 1, 64),
            need(4, 64, 16, 128, 1, 64),
            need(4, 128, 33, 128, 1, 64),
            need(4, 128, 0, 128, 0, 64),
            need(4, 128, 16, 64, 1, 64),
            need(4, 128, 16, 128, 0, 64),
            need(4, 128, 16, 128, 17, 64),
            need(4, 128, 16, 128, 1, 128),
        ] {
            assert!(!runs(&n), "{n} is no kernel's");
        }
    }

    /// The rope, router and dense rows are the kernels' instances: the q/k-normed NeoX rope over
    /// 64 of 128 values, the 128-expert top-4 router (not the body table's 128/top-6 row) and a
    /// dense block whose width is whole q8_1 blocks of 32. Mutants: the rope's norm or mode
    /// dropped, the router's `(128, 4)` check dropped, or the dense width held to 256 again.
    #[test]
    fn the_rows_are_the_kernels_instances() {
        use models::shape::{MoeShape, rules};
        use models::{Need, RopeMode};

        use super::AVAILABLE;
        let runs = |n: &Need| AVAILABLE.iter().any(|a| (a.runs)(n));
        let rope = |qk_norm, mode| Need::QkRope {
            qk_norm,
            head: 128,
            mode,
            dims: 64,
        };
        let router = |experts, top_k| {
            Need::Router(MoeShape {
                rule: rules::BIASED_SIGMOID,
                experts,
                top_k,
            })
        };
        let dense = |ff| Need::SwigluOai { ff };
        for n in [
            rope(true, RopeMode::Neox),
            router(128, 4),
            dense(128),
            dense(12288),
        ] {
            assert!(runs(&n), "{n} is the kernels'");
        }
        for n in [
            rope(false, RopeMode::Neox),
            rope(true, RopeMode::None),
            rope(true, RopeMode::NormTail),
            router(128, 6),
            dense(48),
        ] {
            assert!(!runs(&n), "{n} is no kernel's");
        }
    }

    /// The released file's shape ([`real`]): every need its layers make is a row of the program's
    /// (the head-128 group-16 flash, the q/k norm with the NeoX rope over 64 of 128 values, the
    /// block-max indexer of 4 heads x 128 over blocks of 128, the 128-expert top-4 router, the
    /// swigluoai dense block and experts, the ungated shared expert), so none is an item.
    /// Mutant: one of those rows dropped, or the program taken off it.
    #[test]
    fn the_released_shape_is_covered() {
        let (small, _) = read("mm3-released");
        let spec = spec_of(&real(), small.spec.chat.clone()).expect("the released description");
        let needs: Vec<String> = models::needs(&spec)
            .iter()
            .map(|(n, _)| n.to_string())
            .collect();
        assert!(
            needs
                .iter()
                .any(|n| n.starts_with("block-max indexer: 4 heads x 128"))
        );
        let items: Vec<String> = coverage::check(&spec, &small.tensors)
            .into_iter()
            .map(|u| u.feature)
            .filter(|f| needs.contains(f))
            .collect();
        assert!(items.is_empty(), "{items:#?}");
    }

    /// What no kernel is built for stays listed: the small header's head-16 flash and rope, its
    /// 2 x 16 indexer over blocks of 4, its 8-expert router and its dense block 48 wide, besides the
    /// chat surface's template tool-call parser. Each is a coverage item, so a load refuses the file
    /// by name rather than running it wrong. The `minimax-m2` pre-tokenizer is run by the tokenizer
    /// crate and is no item.
    #[test]
    fn what_no_program_runs_is_listed() {
        let (_, items) = read("mm3-coverage");
        let listed = |needle: &str| items.iter().any(|f| f.contains(needle));
        for needle in [
            "GQA flash, head 16, group 4",
            "per-head QK norm plus rope: head 16",
            "block-max indexer: 2 heads x 16, blocks of 4 keep 3 (1 local)",
            "router: sigmoid, 8 experts, top 2",
            "dense swigluoai layer: ff 48",
        ] {
            assert!(listed(needle), "{needle:?} not listed: {items:#?}");
        }
        assert!(
            !listed("pre-tokenizer"),
            "a pre-tokenizer is listed: {items:#?}"
        );
    }
}
