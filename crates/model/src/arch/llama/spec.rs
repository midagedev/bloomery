//! The llama reader's [`ModelSpec`]: the file's [`Hparams`] and tensor roles
//! as the typed model description. Every layer is a GQA layer on the pairwise
//! rope (dividing by the file's frequency factors when it carries them) with a
//! SwiGLU feed-forward block of `feed_forward_length`, on a plain residual; the
//! per-head q/k norm is what a layer's tensors say, and nothing gates or selects
//! the attention.

use std::collections::HashSet;

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, OutGate, Residual, Rope, RopeMode,
};

use super::hparams::{Hparams, RopeMode as HpRopeMode};
use super::roles;
use crate::arch::{Read, chat_from_template, names, spec_u32, spm_pre_default};
use crate::placement::{ModelTensors, PlacementError};

/// `split`'s description and roles, and the defaults the key reads took:
/// [`Hparams::read`], then `roles::classify`, then `spec_of`. The `llama`
/// string is the architecture of several model families, so the chat surface's
/// parsers are the ones the file's template teaches (`chat_from_template`).
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_from_template(split)?;
    let spec = spec_of(&hp, &tensors, chat)?;
    let mut defaults = hp.defaults;
    defaults.extend(spm_pre_default(split));
    Ok(Read {
        spec,
        tensors,
        defaults,
    })
}

/// The description of the file `hp` and `tensors` were read from, with the
/// chat surface `chat`.
pub(crate) fn spec_of(
    hp: &Hparams,
    tensors: &ModelTensors,
    chat: ChatSpec,
) -> Result<ModelSpec, PlacementError> {
    let has: HashSet<&str> = tensors.tensors.iter().map(|t| t.name.as_str()).collect();
    let rope = Rope {
        mode: match hp.rope.mode {
            HpRopeMode::Norm => RopeMode::NormTail,
        },
        dims: spec_u32("rope.dimension_count", hp.rope.dims)?,
        base: hp.rope.base,
        yarn: None,
        factors: hp.rope.factors,
    };
    let heads = spec_u32("attention.head_count", hp.n_head)?;
    let kv_heads = spec_u32("attention.head_count_kv", hp.n_head_kv)?;
    // The reader holds the file to one width for q, k and v.
    let head_dim = spec_u32("attention.key_length", hp.head_dim)?;
    let ff = spec_u32("feed_forward_length", hp.n_ff)?;
    let layers = (0..hp.n_layer)
        .map(|l| LayerSpec {
            mixer: Mixer::Gqa(Gqa {
                heads,
                kv_heads,
                head_dim,
                value_dim: head_dim,
                rope,
                qk_norm: has.contains(names::attn_q_norm(l).as_str())
                    && has.contains(names::attn_k_norm(l).as_str()),
                o_norm: false,
                out_gate: OutGate::None,
                select: None,
                window: None,
                sinks: false,
                value_scale: None,
            }),
            ffn: Ffn::Dense {
                ff,
                act: Act::SwiGlu { limit: None },
            },
            residual: Residual::Plain,
            extras: Vec::new(),
        })
        .collect::<Vec<_>>();
    Ok(ModelSpec {
        arch: Arch::Llama,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
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
    use super::super::nanbeige_template::TEMPLATE as NANBEIGE_TEMPLATE;
    use super::*;
    use crate::arch::synthetic::{V, header_shaped, verdict};
    use crate::arch::{coverage, spec};
    use crate::placement::Role;
    use models::{ReasoningFormat, ToolFormat};

    const LAYERS: usize = 3;
    const E: u64 = 32;
    const HEADS: u64 = 4;
    const KV: u64 = 2;
    const D: u64 = 8;
    const FF: u64 = 48;
    /// The token list the synthetic header writes.
    const VOCAB: u64 = 2;

    fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(LAYERS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::U32(KV as u32)),
            ("attention.key_length", V::U32(D as u32)),
            ("attention.value_length", V::U32(D as u32)),
            ("rope.dimension_count", V::U32(D as u32)),
            ("rope.freq_base", V::F32(500_000.0)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
        ]
    }

    /// The tokenizer keys of a SentencePiece file whose `tokenizer.ggml.pre`
    /// is not a name of this tree's pre-tokenizers.
    fn global() -> Vec<(&'static str, V)> {
        vec![
            ("tokenizer.ggml.model", V::Str("llama")),
            ("tokenizer.ggml.pre", V::Str("default")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ]
    }

    /// [`global`] with `tokenizer.chat_template` set to `template`.
    fn global_with_template(template: &'static str) -> Vec<(&'static str, V)> {
        let mut global = global();
        let slot = global
            .iter_mut()
            .find(|(key, _)| *key == "tokenizer.chat_template")
            .expect("the template key");
        slot.1 = V::Str(template);
        global
    }

    fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut out = vec![
            ("token_embd.weight".to_string(), vec![E, VOCAB]),
            ("output_norm.weight".to_string(), vec![E]),
            ("output.weight".to_string(), vec![E, VOCAB]),
        ];
        for l in 0..LAYERS {
            for (name, dims) in [
                ("attn_norm", vec![E]),
                ("attn_q", vec![E, HEADS * D]),
                ("attn_k", vec![E, KV * D]),
                ("attn_v", vec![E, KV * D]),
                ("attn_output", vec![HEADS * D, E]),
                ("ffn_norm", vec![E]),
                ("ffn_gate", vec![E, FF]),
                ("ffn_up", vec![E, FF]),
                ("ffn_down", vec![FF, E]),
            ] {
                out.push((format!("blk.{l}.{name}.weight"), dims));
            }
        }
        out
    }

    fn load(
        tag: &str,
        kv: &[(&str, V)],
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Read, String> {
        verdict(
            &format!("llama-spec-{tag}"),
            "llama",
            kv,
            global,
            tensors,
            read,
        )
    }

    /// The description of a header with the Llama shape: dense GQA layers on
    /// the pairwise rope, no gate and no q/k norm, the chat surface bound.
    #[test]
    fn the_description_is_a_dense_pairwise_rope_trunk() {
        let read = load("shape", &keys(), &global(), &tensors()).expect("the header reads");
        let s = &read.spec;
        assert_eq!(s.arch, Arch::Llama);
        assert_eq!(s.arch.name(), "llama");
        assert_eq!((s.hidden, s.vocab, s.ctx_train), (E as u32, 2, 4096));
        assert_eq!(s.layers.len(), LAYERS);
        for layer in &s.layers {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("a layer mixes with {:?}", layer.mixer);
            };
            assert_eq!(
                (g.heads, g.kv_heads, g.head_dim, g.value_dim),
                (HEADS as u32, KV as u32, D as u32, D as u32)
            );
            assert_eq!(g.group(), 2);
            assert_eq!(g.rope.mode, RopeMode::NormTail);
            assert_eq!((g.rope.dims, g.rope.base), (D as u32, 500_000.0));
            assert!(g.rope.yarn.is_none() && !g.rope.factors);
            assert!(!g.qk_norm);
            assert_eq!(g.out_gate, OutGate::None);
            assert!(g.select.is_none() && g.window.is_none() && !g.sinks);
            assert!(g.value_scale.is_none());
            assert_eq!(
                layer.ffn,
                Ffn::Dense {
                    ff: FF as u32,
                    act: Act::SwiGlu { limit: None }
                }
            );
            assert_eq!(layer.residual, Residual::Plain);
            assert!(layer.extras.is_empty());
        }
        assert!(s.mtp.is_empty() && s.hc.is_none() && s.engram.is_none() && s.encoder.is_none());
        assert_eq!(s.chat.pre, "llama");
        assert_eq!(s.chat.template.as_deref(), Some("{{ messages }}"));
        // The SentencePiece rule's rewrite of the file's pre-tokenizer is recorded.
        assert_eq!(
            read.defaults,
            [
                "tokenizer.ggml.pre = llama (tokenizer.ggml.model llama; the file says Some(\"default\"))"
            ]
        );
        let role_of = |name: &str| {
            read.tensors
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| (t.role, t.layer))
        };
        assert_eq!(
            role_of("blk.2.attn_k.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role_of("blk.2.ffn_norm.weight"),
            Some((Role::FfnNorm, Some(2)))
        );
        assert_eq!(
            role_of("blk.1.ffn_up.weight"),
            Some((Role::DenseFfn, Some(1)))
        );
        assert_eq!(role_of("output.weight"), Some((Role::Head, None)));
        assert_eq!(
            role_of("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!((read.tensors.experts, read.tensors.experts_used), (0, 0));
    }

    /// A template that writes the Qwen XML call markup and opens a `<think>`
    /// span (Nanbeige4.3's, as its header carries it) names both parsers.
    #[test]
    fn a_template_with_the_qwen_xml_markup_and_a_think_span_names_both() {
        let global = global_with_template(NANBEIGE_TEMPLATE);
        let read = load("template-qwen", &keys(), &global, &tensors()).expect("the header reads");
        let chat = &read.spec.chat;
        assert_eq!(chat.template.as_deref(), Some(NANBEIGE_TEMPLATE));
        assert_eq!(
            (chat.tools, chat.reasoning),
            (Some(ToolFormat::QwenXml), Some(ReasoningFormat::THINK))
        );
    }

    /// A Llama-3 style template writes neither markup: the description names
    /// no parser, and the serve layer then refuses a tool request by name.
    #[test]
    fn a_template_without_either_markup_names_no_parser() {
        const LLAMA3: &str = concat!(
            r#"{% set loop_messages = messages %}{% for message in loop_messages %}"#,
            r#"{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"#,
            r#" + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}"#,
            r#"{{ bos_token + content }}{% else %}{{ content }}{% endif %}{% endfor %}"#,
            r#"{% if add_generation_prompt %}"#,
            r#"{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"#,
        );
        let global = global_with_template(LLAMA3);
        let read = load("template-llama3", &keys(), &global, &tensors()).expect("the header reads");
        let chat = &read.spec.chat;
        assert_eq!((chat.tools, chat.reasoning), (None, None));
    }

    /// The file's own `llama` pre-tokenizer is no rewrite, and the registry
    /// reaches the reader by the architecture string.
    #[test]
    fn a_llama_pre_is_no_default_and_the_registry_reads_the_file() {
        let mut global = global();
        global[1].1 = V::Str("llama");
        let read = load("pre", &keys(), &global, &tensors()).expect("the header reads");
        assert!(read.defaults.is_empty(), "{:?}", read.defaults);
        let path = header_shaped("llama-spec-registry", "llama", &keys(), &global, &tensors());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let through = spec(&split).map(|r| r.spec.arch);
        drop(split);
        std::fs::remove_file(&path).expect("remove the synthetic header");
        assert_eq!(
            through.expect("the registry reads a llama file"),
            Arch::Llama
        );
    }

    /// A layer's q/k norm is what its tensors say: both are the description's
    /// `qk_norm`, one without the other is refused by name.
    #[test]
    fn qk_norm_is_the_tensors_pair() {
        let mut t = tensors();
        for l in 0..LAYERS {
            t.push((format!("blk.{l}.attn_q_norm.weight"), vec![D]));
            t.push((format!("blk.{l}.attn_k_norm.weight"), vec![D]));
        }
        let read = load("norms", &keys(), &global(), &t).expect("both norms on every layer");
        for layer in &read.spec.layers {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("a layer mixes with {:?}", layer.mixer);
            };
            assert!(g.qk_norm);
        }
        t.retain(|(n, _)| n != "blk.1.attn_k_norm.weight");
        let err = load("one-norm", &keys(), &global(), &t).expect_err("a q norm alone");
        assert!(
            err.starts_with("tensor blk.1.attn_k_norm.weight: is not in the file, and blk.1.attn_q_norm.weight is;"),
            "{err}"
        );
    }

    /// The rope's factors are the file's own tensor, one value a pair, read once by the load
    /// (`Unread`, on no layer): a file with the table has a rope on every layer that divides by
    /// it and the description needs the factors, and a file without it has neither.
    #[test]
    fn the_factors_are_the_files_table() {
        let mut t = tensors();
        t.push(("rope_freqs.weight".to_string(), vec![D / 2]));
        let read = load("factors", &keys(), &global(), &t).expect("a file with a factor table");
        for layer in &read.spec.layers {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("a layer mixes with {:?}", layer.mixer);
            };
            assert!(g.rope.factors);
        }
        assert!(
            models::needs(&read.spec)
                .iter()
                .any(|(n, _)| matches!(n, models::Need::RopeFactors))
        );
        let table = read
            .tensors
            .tensors
            .iter()
            .find(|t| t.name == "rope_freqs.weight")
            .expect("the table is classified");
        assert_eq!((table.role, table.layer), (Role::Unread, None));
        let bare = load("no-factors", &keys(), &global(), &tensors()).expect("a file with none");
        assert!(
            !models::needs(&bare.spec)
                .iter()
                .any(|(n, _)| matches!(n, models::Need::RopeFactors))
        );
    }

    /// Each file this chain does not run is refused by the name of what is
    /// wrong with it.
    #[test]
    fn a_file_this_chain_does_not_run_is_refused_by_name() {
        let without = |name: &str| {
            let mut t = tensors();
            t.retain(|(n, _)| n != name);
            t
        };
        let plus = |name: &str, dims: Vec<u64>| {
            let mut t = tensors();
            t.push((name.to_string(), dims));
            t
        };
        let reshaped = |name: &str, dims: Vec<u64>| {
            let mut t = tensors();
            t.iter_mut().find(|(n, _)| n == name).expect(name).1 = dims;
            t
        };
        let cases = [
            (
                without("blk.2.ffn_down.weight"),
                "tensor blk.2.ffn_down.weight: is not in the file (1 tensor(s) a dense llama trunk needs are missing",
            ),
            (
                plus("blk.0.attn_q.bias", vec![HEADS * D]),
                "1 tensors have no role: blk.0.attn_q.bias",
            ),
            (
                plus("rope_freqs.weight", vec![D]),
                "tensor rope_freqs.weight: has dims [8]; the rope turns 8 values of a head, 4 pairs",
            ),
            (
                plus("blk.1.ffn_gate_inp.weight", vec![E, 8]),
                "1 tensors have no role: blk.1.ffn_gate_inp.weight",
            ),
            (
                reshaped("blk.1.attn_k.weight", vec![E, KV * D + 1]),
                "tensor blk.1.attn_k.weight: has other dims than the keys give the tensor",
            ),
            (
                reshaped("token_embd.weight", vec![E + 1, VOCAB]),
                "tensor token_embd.weight: has other dims than the keys give the tensor",
            ),
        ];
        let wrong: Vec<String> = cases
            .into_iter()
            .enumerate()
            .filter_map(|(i, (t, want))| {
                match load(&format!("refuse{i}"), &keys(), &global(), &t) {
                    Ok(_) => Some(format!("case {i} read; expected \"{want}\"")),
                    Err(e) if e.starts_with(want) => None,
                    Err(e) => Some(format!("case {i}: {e}")),
                }
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A file with no `output.weight` reads: its head is tied to the token
    /// embedding (Llama-3.2's folders, `tie_word_embeddings`), which the
    /// readers of [`Role::Head`] resolve, and no tensor of the table claims
    /// the head's name.
    #[test]
    fn a_tied_head_is_the_embedding_and_reads() {
        let mut t = tensors();
        t.retain(|(n, _)| n != "output.weight");
        let read = load("tied", &keys(), &global(), &t).expect("a tied head reads");
        assert_eq!(read.spec.layers.len(), LAYERS);
        assert!(
            !read
                .tensors
                .tensors
                .iter()
                .any(|x| x.name == "output.weight"),
            "the tied head claims no tensor of its own"
        );
    }

    /// The coverage list of a llama file names no pre-tokenizer (a SentencePiece
    /// file's is `llama`, whatever its `tokenizer.ggml.pre` says) and no layer
    /// program (the Qwen3.5 body is the architecture's).
    #[test]
    fn the_coverage_list_holds_no_pre_tokenizer_and_no_program() {
        let read = load("coverage", &keys(), &global(), &tensors()).expect("the header reads");
        let list = coverage::check(&read.spec, &read.tensors);
        let features: Vec<&str> = list.iter().map(|u| u.feature.as_str()).collect();
        assert!(
            features
                .iter()
                .all(|f| !f.starts_with("pre-tokenizer") && !f.starts_with("a layer program")),
            "{features:?}"
        );
    }
}
