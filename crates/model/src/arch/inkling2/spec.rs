//! The Inkling 2 reader's [`ModelSpec`]: every layer a [`Mixer::RelBias`] of its head counts and
//! window (the pattern's windowless layers attend every position), the routed `situ` experts
//! behind the biased sigmoid router with no shared one, the softmax mix of the banked block
//! inputs as every sublayer's input ([`Residual::AttnResMix`], one attention query row windowed
//! and three windowless), the head's softcap and unpadded column cut, and the loop
//! ([`models::LoopSpec`]) where the file carries one. The file's tokenizer names one
//! pre-tokenizer under either of its two spellings; the parsers are the template's (no markup
//! of this family's own is read yet).

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Ffn, LayerSpec, LogitEpilogue, LoopSpec, Mixer, ModelSpec, Moe, OutGate,
    RelBias, Residual,
};

use super::hparams::{self, Hparams};
use super::names;
use super::roles;
use crate::arch::{Read, chat_from_template, spec_u32};
use crate::placement::PlacementError;

/// The pre-tokenizers a file of this family names: llama.cpp's `inkling2` is the GPT-4o regex
/// (`llama-vocab.cpp`), and the two released files spell it each way. The tokenizer round holds
/// our ids to the reference's on both.
const PRES: &[&str] = &["inkling2", "gpt-4o"];

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_from_template(split)?;
    if !PRES.contains(&chat.pre.as_str()) {
        return Err(PlacementError::Metadata {
            key: "tokenizer.ggml.pre".to_string(),
            detail: format!(
                "is {:?}; an Inkling 2 file names the GPT-4o pre-tokenizer, spelled {:?}",
                chat.pre, PRES
            ),
        });
    }
    // The row's facts no key states: the router's norm and f32 logits, the value norm, the mix's
    // fixed epsilon, the convolutions' scale, the sink denominator and the head's cut.
    let rules = crate::arch::rules_of(split)?;
    let bias: Vec<bool> = (0..hp.n_layer)
        .map(|l| split.find(&names::exp_probs_b(l)).is_some())
        .collect();
    let spec = spec_of(&hp, bias, chat)?;
    let mut defaults = hp.defaults;
    defaults.push(rules.router_norm_line());
    defaults.extend(rules.facts_lines());
    Ok(Read {
        spec,
        tensors,
        defaults,
    })
}

/// The description of the file `hp` was read from, its layers' router bias `bias`, with the chat
/// surface `chat`.
pub fn spec_of(hp: &Hparams, bias: Vec<bool>, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let act = Act::Situ {
        gate_beta: hp.situ.gate_beta,
        value_beta: hp.situ.value_beta,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let windowed = hp.windowed[l];
        Ok(LayerSpec {
            mixer: Mixer::RelBias(RelBias {
                heads: spec_u32("attention.head_count", hp.n_head[l])?,
                kv_heads: spec_u32("attention.head_count_kv", hp.n_head_kv[l])?,
                head_dim: spec_u32("attention.key_length", hp.head_dim)?,
                window: windowed.then_some(spec_u32("attention.sliding_window", hp.swa)?),
                out_gate: OutGate::Proj,
                d_rel: spec_u32("attention.relative_length", hp.d_rel)?,
                extent: spec_u32("attention.relative_extent", hp.extent)?,
                qk_clip: hp.qk_clip,
                rel_clip: hp.rel_clip,
                log_floor: hp.log_floor,
                log_alpha: hp.log_alpha,
                taps: spec_u32("shortconv.l_cache", hp.conv)?,
                dynamic: !windowed,
            }),
            ffn: Ffn::Moe(Moe {
                experts: spec_u32("expert_count", hp.n_expert)?,
                top_k: spec_u32("expert_used_count", hp.n_used[l])?,
                expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff[l])?,
                latent: None,
                pre_exp_norm: false,
                post_exp_norm: None,
                sum_norm: false,
                act,
                router: hparams::router(bias[l], hp.weights_scale),
                shared: None,
            }),
            residual: Residual::AttnResMix {
                block: spec_u32("attn_res.block_size", hp.block)?,
                queries: u32::from(!windowed) * 2 + 1,
            },
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::Inkling2,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab_size", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: Some(LogitEpilogue {
            scale: 1.0,
            softcap: hp.softcap,
            cut: Some(spec_u32("vocab_size_unpadded", hp.n_vocab_unpadded)?),
        }),
        loop_: match hp.loop_ {
            None => None,
            Some(loop_) => Some(LoopSpec {
                core_start: spec_u32("loop.core_start", loop_.core_start)?,
                core_end: spec_u32("loop.core_end", loop_.core_end)?,
                passes: spec_u32("num_loops", loop_.passes)?,
                gate_rank: spec_u32("loop.gate_rank", loop_.gate_rank)?,
            }),
        },
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::{Mixer, Residual, RouterNorm, Score};

    use super::super::hparams::tests::{keys, master_keys, master_tensors, tensors};
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{PlacementError, Role};

    /// The small header of the `hparams` tests with `tokenizer.ggml.pre` `pre` and its read.
    fn read_with(
        tag: &str,
        pre: &'static str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<super::Read, PlacementError> {
        let global = [
            ("tokenizer.ggml.pre", V::Str(pre)),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "inkling2", kv, &global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split);
        let _ = std::fs::remove_file(&path);
        read
    }

    /// [`read_with`] under the master dialect's architecture word.
    fn read_master_with(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<super::Read, PlacementError> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("inkling2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, crate::arch::INKLING2_LOOP, kv, &global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split);
        let _ = std::fs::remove_file(&path);
        read
    }

    /// The description is the table's: every layer a relative-position bias mixer of 4 heads
    /// over 2 of 16 values, windowed at 16 or attending every position with input-dependent
    /// conv taps, the output gate a projection, the bias bank 4 x 32, the clips 22.627 and 8,
    /// the log-scaled positions at floor 8192 alpha 0.1, the convs 4 taps; the routed block 8
    /// experts of 32 at the top 2 behind the biased sigmoid router scaled 8 with `situ` 4/25 and
    /// no shared expert; the residual the block mix of block 2, one attention query windowed and
    /// three windowless; the head softcaps at 30 and cuts to the unpadded 2 of 4; the loop the
    /// core [2, 4) three times at rank 8. Mutant: a field of the table left off the reader.
    #[test]
    fn the_description_is_the_tables() {
        let read = read_with("inkling2-spec", "inkling2", &keys(true), &tensors(true))
            .expect("the header reads");
        let spec = &read.spec;
        assert_eq!(spec.arch, models::Arch::Inkling2);
        assert_eq!((spec.hidden, spec.vocab, spec.ctx_train), (64, 4, 4096));
        assert_eq!(
            spec.loop_,
            Some(models::LoopSpec {
                core_start: 2,
                core_end: 4,
                passes: 3,
                gate_rank: 8,
            })
        );
        assert_eq!(
            spec.logit_epilogue,
            Some(models::LogitEpilogue {
                scale: 1.0,
                softcap: 30.0,
                cut: Some(2),
            })
        );
        assert!(spec.mtp.is_empty() && spec.hc.is_none() && spec.engram.is_none());
        assert!(spec.encoder.is_none());
        assert_eq!(spec.layers.len(), 6);
        for (l, layer) in spec.layers.iter().enumerate() {
            let windowed = l != 2;
            let Mixer::RelBias(a) = &layer.mixer else {
                panic!("layer {l} is not a rel-bias mixer");
            };
            assert_eq!(
                (a.heads, a.kv_heads, a.head_dim, a.window, a.out_gate),
                (4, 2, 16, windowed.then_some(16), models::OutGate::Proj)
            );
            assert_eq!(
                (a.d_rel, a.extent, a.taps, a.dynamic),
                (4, 32, 4, !windowed)
            );
            assert_eq!((a.qk_clip, a.rel_clip), (22.627_417, 8.0));
            assert_eq!((a.log_floor, a.log_alpha), (8192, 0.1));
            assert_eq!(
                layer.residual,
                Residual::AttnResMix {
                    block: 2,
                    queries: u32::from(!windowed) * 2 + 1,
                },
                "layer {l}"
            );
            let models::Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l} is not routed");
            };
            assert_eq!(
                (m.experts, m.top_k, m.expert_ff, m.shared),
                (8, 2, 32, None)
            );
            assert_eq!(
                m.act,
                models::Act::Situ {
                    gate_beta: 4.0,
                    value_beta: 25.0,
                }
            );
            assert!(!m.pre_exp_norm && !m.sum_norm && m.post_exp_norm.is_none());
            assert_eq!(m.router.score, Score::Sigmoid);
            assert!(m.router.bias && !m.router.hash);
            assert_eq!(m.router.norm, RouterNorm::SelectedSum);
            assert_eq!(m.router.scale, 8.0);
            assert!(layer.extras.is_empty(), "layer {l}");
        }
        assert_eq!(spec.chat.pre, "inkling2");
        assert!(spec.chat.tools.is_none() && spec.chat.reasoning.is_none());
        // The defaults are the description row's facts, every one with its source: the file
        // carries every key the reader reads.
        assert_eq!(
            read.defaults,
            [
                "router norm = selected-sum (B:65: expert_weights_norm_eps = 1e-20 unconditional)",
                "router logits = f32 (expert_router_f32) (B:64: the router scores run in f32)",
                "router bias = selection only (M:690-696: exp_probs_b steers the choice, the kept weights are the unbiased scores)",
                "V = gainless per-head RMS, always (B:406: the value rows are normed whatever the file's `value_norm` key says)",
                "mix eps = 1e-6 fixed, scale 1/sqrt(hidden) (B:217,222: the block attention residual's states are normed at a fixed epsilon and scored at 1/sqrt(hidden))",
                "conv output = (x + conv(x)) / sqrt(taps + 1), twice the taps where the k/v convs are dynamic (B:282: every convolution's scale counts its taps (and the dynamic taps))",
                "softmax denominator = 1 + sum(exp(s - t)) (B:429-438: a per-(head, query) sink logit t adds a zero column)",
                "head = softcap then the unpadded columns (B:503-510: the softcap runs on the padded logits, the cut after it)",
                "flash attention = off (B:88, llama-graph.cpp:3201: use_flash_attn needs no kq_b, and every layer carries one)",
            ]
        );
    }

    /// A pre-tokenizer neither spelling names is refused by its key, and the roles are the
    /// table's: the embedding's norm with the embedding, the final mix's query with the head,
    /// the loop's message queries with the attention rows.
    #[test]
    fn the_roles_and_the_pre_tokenizer_are_the_tables() {
        match read_with(
            "inkling2-spec-pre",
            "deepseek-v3",
            &keys(false),
            &tensors(false),
        ) {
            Ok(_) => panic!("read"),
            Err(e) => assert!(e.to_string().contains("tokenizer.ggml.pre"), "{e}"),
        }
        read_with(
            "inkling2-spec-pre-4o",
            "gpt-4o",
            &keys(false),
            &tensors(false),
        )
        .expect("the GPT-4o spelling reads");
        let read = read_with(
            "inkling2-spec-roles",
            "inkling2",
            &keys(true),
            &tensors(true),
        )
        .expect("the header reads");
        let role = |name: &str| {
            read.tensors
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| (t.role, t.layer))
        };
        assert_eq!(
            role("token_embd_norm.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output_res_query.weight"), Some((Role::Head, None)));
        assert_eq!(role("loop_res_query.weight"), Some((Role::Attention, None)));
        assert_eq!(
            role("blk.2.loop_v_gain.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role("blk.1.attn_res_query.weight"),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.1.ffn_post_conv.weight"),
            Some((Role::FfnNorm, Some(1)))
        );
        assert_eq!(
            role("blk.1.ffn_gate_up_exps.weight"),
            Some((Role::RoutedExperts, Some(1)))
        );
    }

    /// The two dialects' stems do not mix: a master stem in an unsloth file, an unsloth stem in
    /// a master file and any loop tensor in a loopless one are refused by the tensor's name —
    /// the tensor checks' misplaced refusal or the classifier's unclassified list, whichever
    /// meets it first. No such file is on the box; the synthetic header is the proof. Mutant: a
    /// dialect's stems given to both kinds.
    #[test]
    fn the_dialects_stems_do_not_mix() {
        let mut master_in_unsloth = tensors(true);
        master_in_unsloth.push(("blk.2.loop_msg_query.weight".into(), vec![64, 4]));
        let e = read_with("mixed-master", "inkling2", &keys(true), &master_in_unsloth)
            .expect_err("a master stem in an unsloth file");
        assert!(e.to_string().contains("blk.2.loop_msg_query.weight"), "{e}");
        let mut unsloth_in_master = master_tensors();
        unsloth_in_master.push(("blk.2.loop_attn_down.weight".into(), vec![128, 8]));
        let e = read_master_with("mixed-unsloth", &master_keys(), &unsloth_in_master)
            .expect_err("an unsloth stem in a master file");
        assert!(e.to_string().contains("blk.2.loop_attn_down.weight"), "{e}");
        let mut stray = tensors(false);
        stray.push(("blk.2.loop_attn_down.weight".into(), vec![128, 8]));
        let e = read_with("stray-loop", "inkling2", &keys(false), &stray)
            .expect_err("a loop tensor in a loopless file");
        assert!(e.to_string().contains("blk.2.loop_attn_down.weight"), "{e}");
    }
}
