//! The mimo2 reader's [`ModelSpec`]. Every block is a GQA layer over two
//! head widths — the score head `attention.key_length` wide, the value head
//! `attention.value_length` wide — with the KV-head count and the rope base
//! its own: a full-attention layer's θ (`rope.freq_base`), a window layer's
//! (`rope.freq_base_swa`, another base). The routed blocks run the sigmoid
//! noaux_tc router: the selection bias steers the choice only, the kept
//! unbiased scores are renormalized (`build_moe_ffn`, src/llama-graph.cpp)
//! and scaled by `expert_weights_scale`. A next-token block is a dense
//! block that reads the two norms' projection; its tensors are carried and
//! not run. The tool calls are [`TOOLS`], the `<think>` span is read.

use gguf::Split;
use models::{
    Act, Arch, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, Moe, ReasoningFormat, Residual, Rope,
    RopeMode, Router, Score, ToolFormat,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::{ModelTensors, PlacementError};

/// The tool-call markup the family's template teaches: `<tool_call>` with
/// `<function=…>` and `<parameter=…>`, the parser `crates/serve` runs.
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::QwenXml);

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(ReasoningFormat::ThinkSpan))?;
    let spec = spec_of(&hp, &tensors, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` and `tensors` were read from.
pub fn spec_of(
    hp: &Hparams,
    tensors: &ModelTensors,
    chat: models::ChatSpec,
) -> Result<ModelSpec, PlacementError> {
    let bias = |l: usize| {
        let name = format!("blk.{l}.{}", roles::SELECTION_BIAS);
        tensors.tensors.iter().any(|t| t.name == name)
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let kv = spec_u32("attention.head_count_kv", hp.kv_heads[l])?;
        let mixer = Mixer::Gqa(Gqa {
            heads: spec_u32("attention.head_count", hp.n_head)?,
            kv_heads: kv,
            head_dim: spec_u32("attention.key_length", hp.head_k)?,
            value_dim: spec_u32("attention.value_length", hp.head_v)?,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
                base: match hp.kinds[l] {
                    Kind::Full => hp.rope_base,
                    Kind::Swa => hp.rope_base_swa,
                },
                yarn: None,
            },
            qk_norm: false,
            out_gate: false,
            select: None,
            // The window layers attend the last `sliding_window` positions;
            // their sinks are the `attn_sinks` tensor where the file has it.
            window: match hp.kinds[l] {
                Kind::Full => None,
                Kind::Swa => Some(spec_u32("attention.sliding_window", hp.window)?),
            },
            sinks: tensors
                .tensors
                .iter()
                .any(|t| t.name == format!("blk.{l}.attn_sinks.weight")),
            value_scale: Some(hp.value_scale),
        });
        let routed = l < hp.n_trunk
            && tensors
                .tensors
                .iter()
                .any(|t| t.name == format!("blk.{l}.ffn_gate_inp.weight"));
        let ffn = if routed {
            Ffn::Moe(Moe {
                experts: spec_u32("expert_count", hp.n_expert)?,
                top_k: spec_u32("expert_used_count", hp.n_used)?,
                expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff)?,
                act: Act::SwiGlu { limit: None },
                router: Router {
                    score: Score::Sigmoid,
                    bias: bias(l),
                    // The builder passes the renormalization as a constant
                    // (mimo2.cpp:218).
                    norm: true,
                    scale: hp.weights_scale,
                    hash: false,
                },
                shared: None,
            })
        } else {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.dense_ff.unwrap_or(0))?,
                act: Act::SwiGlu { limit: None },
            }
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Plain,
            extras: Vec::new(),
        })
    };
    let layers = (0..hp.n_trunk).map(layer).collect::<Result<Vec<_>, _>>()?;
    let mtp = (hp.n_trunk..hp.n_layer)
        .map(layer)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ModelSpec {
        arch: Arch::MiMo2,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp,
        hc: None,
        engram: None,
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::Ffn;

    use super::super::hparams::tests::{keys, tensors};
    use super::super::roles::SELECTION_BIAS;
    use crate::arch::coverage;
    use crate::arch::synthetic::V;

    /// The small header of the `hparams` tests, with `tokenizer.ggml.pre`
    /// `qwen2` and a chat template, and the tensors `keep` keeps; its read
    /// and the coverage check's items.
    fn read(tag: &str, keep: impl Fn(&str) -> bool) -> (super::Read, Vec<String>) {
        let tensors: Vec<(String, Vec<u64>)> =
            tensors().into_iter().filter(|(t, _)| keep(t)).collect();
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = crate::arch::synthetic::header_shaped(tag, "mimo2", &keys(), &global, &tensors);
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

    /// A MiMo file's chat surface is one the tree runs: the tokenizer's
    /// `qwen2` pre-tokenizer and the `<tool_call>` parser the family
    /// declares are not coverage items.
    #[test]
    fn the_chat_surface_is_covered() {
        let (read, items) = read("mimo2-chat", |_| true);
        assert_eq!(read.spec.chat.tools, super::TOOLS);
        for covered in [
            "pre-tokenizer qwen2",
            "a tool-call parser for this template",
        ] {
            assert!(
                !items.iter().any(|f| f == covered),
                "{covered:?} listed: {items:?}"
            );
        }
    }

    /// The selection bias is optional, as the loader loads it: a file
    /// without it reads, and its routers run with none.
    #[test]
    fn the_selection_bias_is_optional() {
        let bias = |r: &super::Read| -> Vec<bool> {
            r.spec
                .layers
                .iter()
                .chain(&r.spec.mtp)
                .filter_map(|l| match &l.ffn {
                    Ffn::Moe(m) => Some(m.router.bias),
                    _ => None,
                })
                .collect()
        };
        let (with, _) = read("mimo2-bias", |_| true);
        let (without, _) = read("mimo2-nobias", |t| !t.ends_with(SELECTION_BIAS));
        assert_eq!(bias(&with), [true; 3]);
        assert_eq!(bias(&without), [false; 3]);
    }

    /// The spec carries what changes the arithmetic: a window of
    /// `sliding_window` positions and the sinks on the window layers, none
    /// on the full ones, and the value scale on every layer; the coverage
    /// list names the three.
    #[test]
    fn the_window_sinks_and_value_scale_are_in_the_spec() {
        use models::Mixer;
        let (read, items) = read("mimo2-attn", |_| true);
        let mut windows = 0;
        for layer in &read.spec.layers {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("a mimo2 layer that is not GQA");
            };
            assert_eq!(g.sinks, g.window.is_some(), "{g:?}");
            assert_eq!(g.value_scale, Some(0.707), "{g:?}");
            windows += usize::from(g.window == Some(8));
        }
        assert!(windows > 0, "no window layer in the small header");
        for listed in [
            "GQA flash, head 16, value 8, group 4, window 8, sinks, value scale",
            "GQA flash, head 16, value 8, group 2, value scale",
        ] {
            assert!(
                items.iter().any(|f| f.starts_with(listed)),
                "{listed:?} not listed: {items:?}"
            );
        }
    }

    /// A mimo2 file is listed against the whole tree: the flash at its head
    /// width and each group, the rope without a QK norm, the router at its
    /// width, and the program itself are items; the dense block (the glm5next
    /// body's row takes any width), the qwen2 tokenizer and the tool parser
    /// are not. The synthetic header's stacks are F32, which every card
    /// reads, so the routed-format item is the file gate's to pin.
    #[test]
    fn the_coverage_list_is_the_work_queue() {
        let (_, items) = read("mimo2-cov", |_| true);
        for listed in [
            "GQA flash, head 16, value 8, group 2",
            "GQA flash, head 16, value 8, group 4",
            "rope without a QK norm",
            "router: sigmoid, 8 experts, top 2, with a selection bias",
            "a layer program for mimo2",
        ] {
            assert!(
                items.iter().any(|f| f.starts_with(listed)),
                "{listed:?} not listed: {items:?}"
            );
        }
        for covered in [
            "pre-tokenizer qwen2",
            "a tool-call parser",
            "dense SwiGLU layer",
        ] {
            assert!(
                !items.iter().any(|f| f.contains(covered)),
                "{covered:?} listed: {items:?}"
            );
        }
    }
}
