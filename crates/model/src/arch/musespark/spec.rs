//! The `muse-spark` reader's [`ModelSpec`]. Every layer is grouped-query
//! attention with a per-head q/k norm and a projection-written output gate,
//! roped in NORM pairs on the sliding layers and NoPE on the full ones, over
//! a sigmoid-routed mixture of SiLU-gated experts in a latent: the layer's
//! input projected down before them and their weighted sum projected back up
//! after, the router reading the full stream (its weights pre-scaled by the
//! MetaP output multiplier at conversion — the reader multiplies nothing).
//! The block norms carry a delta folded at conversion; the residual mixes
//! the stream and the branch by per-channel α/β vectors, each sublayer's
//! output normed at `post_norm_epsilon` first ([`Residual::AlphaBeta`]); the
//! experts run in the latent between a weighted norm and a scaled-then-normed
//! one ([`Moe::pre_exp_norm`], [`Moe::post_exp_norm`]); the heads' outputs
//! are normed per head before their gate ([`Gqa::o_norm`]); and the head
//! scales then softcaps its logits ([`LogitEpilogue`]).
//!
//! The chat surface: the pre-tokenizer `gpt-4o` (GPT4O's one regex, which
//! this tree runs as `minimax-m2`'s), the file's own template, and Muse's
//! recipient-protocol tool parser.

use gguf::{Split, Value};
use models::{
    Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, LogitEpilogue, Mixer, ModelSpec, Moe, OutGate,
    Residual, Rope, RopeMode, Router, Score, ToolFormat,
};

use super::hparams::Hparams;
use super::roles;
use crate::arch::{Read, spec_u32, template_of};
use crate::placement::PlacementError;

/// `tokenizer.ggml.pre`'s value, the only one this family reads (GPT4O's
/// regex, run by the tokenizer as the `minimax-m2` name's).
const PRE: &str = "gpt-4o";

/// `split`'s description and roles.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let spec = spec_of(&hp, chat(split)?)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The file's tokenizer and template. The pre-tokenizer is `gpt-4o`; another
/// name, or none, is refused by name.
fn chat(split: &Split) -> Result<ChatSpec, PlacementError> {
    let key = "tokenizer.ggml.pre";
    match split.value(key).map(Value::as_str) {
        Some(Some(PRE)) => {}
        None => {
            return Err(PlacementError::Metadata {
                key: key.to_string(),
                detail: "is absent".to_string(),
            });
        }
        Some(None) => {
            return Err(PlacementError::Metadata {
                key: key.to_string(),
                detail: "is not a string".to_string(),
            });
        }
        Some(Some(other)) => {
            return Err(PlacementError::Metadata {
                key: key.to_string(),
                detail: format!("is {other:?}, not {PRE}"),
            });
        }
    }
    Ok(ChatSpec {
        pre: PRE.to_string(),
        template: template_of(split)?,
        tools: Some(ToolFormat::MuseSpark),
        reasoning: None,
    })
}

/// The description of the file `hp` was read from.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let sliding = hp.sliding[l];
        Ok(LayerSpec {
            mixer: Mixer::Gqa(Gqa {
                heads: spec_u32("attention.head_count", hp.n_head)?,
                kv_heads: spec_u32("attention.head_count_kv", hp.n_head_kv)?,
                head_dim: spec_u32("attention.key_length", hp.head_dim)?,
                value_dim: spec_u32("attention.value_length", hp.head_dim)?,
                // Rope is the sliding layers' own: the full layers are NoPE.
                rope: Rope {
                    mode: if sliding {
                        RopeMode::NormTail
                    } else {
                        RopeMode::None
                    },
                    dims: if sliding {
                        spec_u32("rope.dimension_count", hp.rope_dims)?
                    } else {
                        0
                    },
                    base: if sliding { hp.rope_base } else { 0.0 },
                    yarn: None,
                    factors: false,
                },
                qk_norm: true,
                o_norm: true,
                out_gate: OutGate::Proj,
                select: None,
                window: sliding.then_some(spec_u32("attention.sliding_window", hp.window)?),
                sinks: false,
                value_scale: None,
            }),
            ffn: Ffn::Moe(Moe {
                experts: spec_u32("expert_count", hp.n_expert)?,
                top_k: spec_u32("expert_used_count", hp.n_used)?,
                expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff[l])?,
                latent: Some(spec_u32("moe_latent_size", hp.latent)?),
                pre_exp_norm: true,
                post_exp_norm: Some(hp.post_eps),
                act: Act::SwiGlu { limit: None },
                router: Router {
                    score: Score::Sigmoid,
                    bias: true,
                    norm: models::RouterNorm::from_legacy_bool(hp.weights_norm),
                    scale: hp.weights_scale,
                    hash: false,
                },
                shared: None,
            }),
            residual: Residual::AlphaBeta { eps: hp.post_eps },
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::MuseSpark,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: Some(LogitEpilogue {
            scale: hp.logit_scale,
            softcap: hp.softcap,
        }),
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::RopeMode;

    use super::super::names;
    use crate::arch::coverage;
    use crate::arch::synthetic::{Ty, V};
    use crate::arch::{Arch, spec};

    /// Five layers, the second and the last full: both window kinds and the
    /// transition between them. The geometry is the file's own (the fixture's
    /// miniatures aside): head 64, 16 query heads over 2 KV heads, 16 experts
    /// with the top 8 in a 512-value latent, the window 2048.
    const N: usize = 5;
    const SLIDING: [bool; N] = [true, false, true, true, false];
    const EMBD: u64 = 256;
    const HEADS: u64 = 16;
    const KV: u64 = 2;
    const HEAD: u64 = 64;
    const WINDOW: u32 = 2048;
    const EXPERTS: u64 = 16;
    const USED: u32 = 8;
    const FF: u64 = 256;
    const LATENT: u64 = 512;
    const BASE: f32 = 5e5;

    fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(N as u32)),
            ("embedding_length", V::U32(EMBD as u32)),
            ("context_length", V::U32(128)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::U32(KV as u32)),
            ("attention.key_length", V::U32(HEAD as u32)),
            ("attention.value_length", V::U32(HEAD as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.sliding_window", V::U32(WINDOW)),
            (
                "attention.sliding_window_pattern",
                V::Bools(SLIDING.to_vec()),
            ),
            ("rope.freq_base_swa", V::F32(BASE)),
            ("post_norm_epsilon", V::F32(1e-8)),
            ("logit_scale", V::F32(0.1767767)),
            ("final_logit_softcapping", V::F32(20.0)),
            ("expert_count", V::U32(EXPERTS as u32)),
            ("expert_used_count", V::U32(USED)),
            ("expert_feed_forward_length", V::U32(FF as u32)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_norm", V::Bool(true)),
            ("expert_weights_scale", V::F32(1.0)),
            ("moe_latent_size", V::U32(LATENT as u32)),
        ]
    }

    /// One layer's 23 tensors, each at the type the file's own carries.
    fn layer(l: usize, out: &mut Vec<(String, Vec<u64>, Ty)>) {
        let (e, d) = (EMBD, HEAD);
        let (q, kv, x, w, lat) = (HEADS * d, KV * d, EXPERTS, FF, LATENT);
        for (stem, dims, ty) in [
            (names::ATTN_NORM, vec![e], Ty::F32),
            (names::ATTN_Q, vec![e, q], Ty::Q6K),
            (names::ATTN_K, vec![e, kv], Ty::Q8_0),
            (names::ATTN_V, vec![e, kv], Ty::Q8_0),
            (names::ATTN_Q_NORM, vec![d], Ty::F32),
            (names::ATTN_K_NORM, vec![d], Ty::F32),
            (names::ATTN_GATE, vec![e, q], Ty::Q8_0),
            (names::ATTN_OUTPUT, vec![q, e], Ty::Q6K),
            (names::ATTN_RESID_ALPHA, vec![e], Ty::F32),
            (names::ATTN_RESID_BETA, vec![e], Ty::F32),
            (names::FFN_NORM, vec![e], Ty::F32),
            (names::FFN_LATENT_DOWN, vec![e, lat], Ty::Q8_0),
            (names::FFN_PRE_EXP_NORM, vec![lat], Ty::F32),
            (names::FFN_GATE_INP, vec![e, x], Ty::F32),
            (names::EXP_PROBS_B, vec![x], Ty::F32),
            (names::FFN_GATE_EXPS, vec![lat, w, x], Ty::Iq2Xxs),
            (names::FFN_UP_EXPS, vec![lat, w, x], Ty::Iq2Xxs),
            (names::FFN_DOWN_EXPS, vec![w, lat, x], Ty::Iq2Xxs),
            (names::FFN_POST_EXP_NORM, vec![lat], Ty::F32),
            (names::FFN_LATENT_UP, vec![lat, e], Ty::Q8_0),
            (names::POST_FFW_NORM, vec![e], Ty::F32),
            (names::FFN_RESID_ALPHA, vec![e], Ty::F32),
            (names::FFN_RESID_BETA, vec![e], Ty::F32),
        ] {
            out.push((crate::arch::names::blk(l, stem), dims, ty));
        }
    }

    fn tensors() -> Vec<(String, Vec<u64>, Ty)> {
        let mut out = vec![
            (names::token_embd(), vec![EMBD, 2], Ty::Q4K),
            (names::output_norm(), vec![EMBD], Ty::F32),
            (names::output(), vec![EMBD, 2], Ty::Q6K),
        ];
        for l in 0..N {
            layer(l, &mut out);
        }
        out
    }

    const GLOBAL: [(&str, V); 2] = [
        ("tokenizer.ggml.pre", V::Str("gpt-4o")),
        ("tokenizer.chat_template", V::Str("{{ bos }}{{ messages }}")),
    ];

    fn load(
        tag: &str,
        kv: Vec<(&str, V)>,
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>, Ty)],
    ) -> Result<crate::arch::Read, String> {
        load_with(tag, "muse-spark", kv, global, tensors)
    }

    fn load_with(
        tag: &str,
        arch: &str,
        kv: Vec<(&str, V)>,
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>, Ty)],
    ) -> Result<crate::arch::Read, String> {
        let path = crate::arch::synthetic::header_typed(tag, arch, &kv, global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    fn set_key(kv: &mut [(&str, V)], key: &str, value: V) {
        let row = kv.iter_mut().find(|(k, _)| *k == key).expect(key);
        row.1 = value;
    }

    /// The sliding layers rope in NORM pairs at the sliding base inside the
    /// window and the full ones are NoPE with no window; every layer norms q
    /// and k per head, norms its output per head before its own-projection
    /// gate, routes in the latent between the expert norms and joins by the
    /// alpha/beta mixes; the head's logits scale then softcap; the rope width
    /// is the head, taken as the default it is. The coverage check is empty:
    /// a program runs the architecture and an `AVAILABLE` row covers every
    /// need, each below named by the row that covers it. A reader that ropes
    /// every layer, hides the full layers' q/k norm behind the absent rope,
    /// or states the plain residual, no output norm, no expert norm or no
    /// logit epilogue, fails this.
    #[test]
    fn the_sliding_layers_rope_and_the_full_ones_do_not() {
        let read = load("muse-ok", keys(), &GLOBAL, &tensors()).expect("the header reads");
        let s = &read.spec;
        assert_eq!(s.arch, models::Arch::MuseSpark);
        assert_eq!((s.hidden, s.vocab, s.ctx_train), (EMBD as u32, 2, 128));
        assert_eq!(s.layers.len(), N);
        assert!(s.mtp.is_empty() && s.hc.is_none() && s.engram.is_none() && s.encoder.is_none());
        for (l, layer) in s.layers.iter().enumerate() {
            let models::Mixer::Gqa(g) = &layer.mixer else {
                panic!("layer {l}: {:?}", layer.mixer);
            };
            if SLIDING[l] {
                assert_eq!(
                    (g.rope.mode, g.rope.dims, g.rope.base),
                    (RopeMode::NormTail, HEAD as u32, BASE),
                    "layer {l}"
                );
                assert_eq!(g.window, Some(WINDOW), "layer {l}");
            } else {
                assert_eq!(
                    (g.rope.mode, g.rope.dims, g.rope.base),
                    (RopeMode::None, 0, 0.0),
                    "layer {l}"
                );
                assert_eq!(g.window, None, "layer {l}");
            }
            assert_eq!(
                (g.heads, g.kv_heads, g.head_dim, g.value_dim),
                (HEADS as u32, KV as u32, HEAD as u32, HEAD as u32),
                "layer {l}"
            );
            assert!(g.qk_norm && g.o_norm && g.out_gate == models::OutGate::Proj);
            assert_eq!(
                layer.residual,
                models::Residual::AlphaBeta { eps: 1e-8 },
                "layer {l}"
            );
            let models::Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l}: {:?}", layer.ffn);
            };
            assert_eq!(
                (m.experts, m.top_k, m.expert_ff, m.latent),
                (EXPERTS as u32, USED, FF as u32, Some(LATENT as u32)),
                "layer {l}"
            );
            assert_eq!(
                (m.pre_exp_norm, m.post_exp_norm),
                (true, Some(1e-8)),
                "layer {l}"
            );
            assert_eq!(m.act, models::Act::SwiGlu { limit: None });
            assert_eq!(m.router.score, models::Score::Sigmoid);
            assert!(m.router.bias && !m.router.hash);
            assert_eq!(m.router.norm, models::RouterNorm::SelectedSum);
            assert_eq!(m.router.scale, 1.0);
            assert!(m.shared.is_none());
        }
        assert_eq!(
            s.logit_epilogue,
            Some(models::LogitEpilogue {
                scale: 0.1767767,
                softcap: 20.0,
            })
        );
        assert_eq!(s.chat.pre, "gpt-4o");
        assert_eq!(s.chat.tools, Some(models::ToolFormat::MuseSpark));
        assert!(s.chat.reasoning.is_none() && s.chat.template.is_some());
        assert_eq!(
            read.defaults,
            ["rope.dimension_count = 64 (the head, llama-hparams.cpp)"]
        );
        // The full layers' q/k norm is a need of its own, not the rope's
        // passenger (the NoPE fix in `models::needs`), and so is every other
        // new fact: the output norm beside the gate, the expert norms in the
        // latent, the residual's mixes and the head's epilogue.
        let needs = models::needs(s);
        for (l, want) in [
            (
                0,
                "per-head QK norm plus rope: head 64, NORM tail, 64 of 64 dims",
            ),
            (1, "per-head QK norm, no rope: head 64"),
            (
                0,
                "a gainless per-head RMS norm on the attention output: head 64",
            ),
            (
                0,
                "a weighted RMS norm on the routed experts' input (ffn_pre_exp_norm)",
            ),
            (
                0,
                "the routed experts' output scaled by ffn_post_exp_norm, then normed at 0.00000001",
            ),
            (
                0,
                "the alpha/beta residual mix, each sublayer's output normed at 0.00000001 first",
            ),
        ] {
            assert!(
                needs
                    .iter()
                    .any(|(n, layer)| *layer == Some(l) && n.to_string() == want),
                "layer {l}: {want} not in the needs"
            );
        }
        // Every need the reader states has a row of the musespark body's
        // program behind it: the head-64 flash (windowed and windowless), the
        // q/k norms with and without the rope, the own-projection output
        // gate, the output norm, the router at the file's and the fixture's
        // pair, the latent experts, the expert norms, the alpha/beta mix and
        // the head's epilogue. Mutant: any row removed, its need is an item
        // again below.
        let musespark = coverage::AVAILABLE
            .iter()
            .filter(|a| {
                a.programs
                    .contains(&coverage::program_of(s.arch).expect("a program"))
            })
            .collect::<Vec<_>>();
        for (l, want) in [
            (Some(0u32), "GQA flash, head 64, group 8, window 2048"),
            (Some(1), "GQA flash, head 64, group 8"),
            (
                Some(0),
                "per-head QK norm plus rope: head 64, NORM tail, 64 of 64 dims",
            ),
            (Some(1), "per-head QK norm, no rope: head 64"),
            (
                Some(0),
                "attention output gate: sigmoid of its own projection",
            ),
            (
                Some(0),
                "a gainless per-head RMS norm on the attention output: head 64",
            ),
            (
                Some(0),
                "router: sigmoid, 16 experts, top 8, with a selection bias",
            ),
            (
                Some(0),
                "routed experts in a 512-value latent (ffn_latent_down before them, ffn_latent_up \
                 after their sum)",
            ),
            (
                Some(0),
                "a weighted RMS norm on the routed experts' input (ffn_pre_exp_norm)",
            ),
            (
                Some(0),
                "the routed experts' output scaled by ffn_post_exp_norm, then normed at 0.00000001",
            ),
            (
                Some(0),
                "the alpha/beta residual mix, each sublayer's output normed at 0.00000001 first",
            ),
            (
                None,
                "logit epilogue: scale 0.1767767, then the tanh softcap at 20",
            ),
        ] {
            assert!(
                needs.iter().any(|(n, layer)| match l {
                    Some(l) => *layer == Some(l) && n.to_string() == want,
                    None => layer.is_none() && n.to_string() == want,
                }),
                "{want}: not in the needs"
            );
            let covered = musespark.iter().any(|a| {
                needs.iter().any(|(n, layer)| match l {
                    Some(l) => *layer == Some(l) && n.to_string() == want && (a.runs)(n),
                    None => layer.is_none() && n.to_string() == want && (a.runs)(n),
                })
            });
            assert!(covered, "{want}: no musespark row runs it");
        }
        // The program runs the file whole: the check lists nothing.
        let items = coverage::check(s, &read.tensors);
        assert!(
            items.is_empty(),
            "the muse-spark file has unrun parts: {items:?}"
        );
        assert_eq!(read.tensors.layers, N);
        assert_eq!(
            (read.tensors.experts, read.tensors.experts_used),
            (EXPERTS, USED as u64)
        );
    }

    /// Each key the reader takes is required by name; llama.cpp's defaults
    /// are not taken (the rope width's is, and is recorded).
    #[test]
    fn a_missing_key_is_named() {
        for (key, _) in keys() {
            let kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
            let err = load("muse-key", kv, &GLOBAL, &tensors()).expect_err(key);
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    /// A key of another value than the model's is refused by name: another
    /// gating func, a nonpositive softcap, logit scale, post-norm epsilon,
    /// weights scale or sliding rope base, a value length other than the key
    /// length, a used count past the routed count, a layer with no expert
    /// width, a pattern of another length or of no bools, a rope width that
    /// is odd or past the head, and key heads that do not divide the query
    /// heads.
    #[test]
    fn a_key_of_another_value_is_named() {
        for key in [
            "rope.freq_base_swa",
            "post_norm_epsilon",
            "logit_scale",
            "final_logit_softcapping",
            "expert_weights_scale",
            "attention.layer_norm_rms_epsilon",
        ] {
            for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
                let mut kv = keys();
                set_key(&mut kv, key, V::F32(value));
                let err = load("muse-float", kv, &GLOBAL, &tensors()).expect_err(key);
                assert!(err.contains(key), "{key}: {err}");
            }
        }
        for (key, value, want) in [
            ("expert_gating_func", V::U32(1), "is 1"),
            ("expert_gating_func", V::U32(3), "is 3"),
            ("expert_gating_func", V::U32(2), ""), // the sigmoid itself passes
            (
                "attention.value_length",
                V::U32(128),
                "is 128, the key length 64",
            ),
            (
                "attention.head_count_kv",
                V::U32(3),
                "is 3 at layer 0 for 16 query heads",
            ),
            ("expert_used_count", V::U32(17), "is 17 of expert_count 16"),
            (
                "attention.sliding_window_pattern",
                V::Bools(vec![true; N - 1]),
                "has 4 values for 5 layers",
            ),
            (
                "attention.sliding_window_pattern",
                V::I32s(vec![1; N]),
                "has no bool at layer 0",
            ),
            (
                "expert_feed_forward_length",
                V::I32s(vec![256, 0, 256, 256, 256]),
                "is 0 at layer 1",
            ),
        ] {
            let mut kv = keys();
            set_key(&mut kv, key, value);
            let read = load("muse-other", kv, &GLOBAL, &tensors());
            if want.is_empty() {
                assert!(read.is_ok(), "{key}: {read:?}");
            } else {
                let err = read.expect_err(key);
                assert!(err.contains(key) && err.contains(want), "{key}: {err}");
            }
        }
        // A rope width key of the file's own is read: an odd one, or one past
        // the head, is refused.
        for dims in [63u32, 66] {
            let mut kv = keys();
            kv.push(("rope.dimension_count", V::U32(dims)));
            let err = load("muse-rot", kv, &GLOBAL, &tensors()).expect_err("a rope width");
            assert!(
                err.contains("rope.dimension_count")
                    && err.contains(&format!("is {dims}, for a head of 64")),
                "{err}"
            );
        }
    }

    /// One expert width serves the step's launches and the plan's sizing, so
    /// a file whose layers differ is refused at the first layer that does,
    /// its tensors agreeing with its own keys (the widths whole 256-value
    /// blocks): a reader that took layer 0's width for every layer would run
    /// layer 2 at the wrong one.
    #[test]
    fn an_expert_width_that_differs_by_layer_is_named() {
        let mut kv = keys();
        set_key(
            &mut kv,
            "expert_feed_forward_length",
            V::I32s(vec![256, 256, 512, 256, 256]),
        );
        let mut t = tensors();
        for (stem, at) in [
            (names::FFN_GATE_EXPS, 1),
            (names::FFN_UP_EXPS, 1),
            (names::FFN_DOWN_EXPS, 0),
        ] {
            let name = crate::arch::names::blk(2, stem);
            let row = t.iter_mut().find(|(n, _, _)| *n == name).expect(&name);
            row.1[at] = 512;
        }
        let err = load("muse-ff", kv, &GLOBAL, &t).expect_err("layers of two widths");
        assert!(
            err.contains("expert_feed_forward_length")
                && err.contains("is 512 at layer 2, 256 at layer 0"),
            "{err}"
        );
    }

    /// A tensor missing is named with the count of the others, a tensor of
    /// other dims names its want, and a tensor no table holds is named.
    #[test]
    fn a_tensor_missing_or_of_other_dims_is_named() {
        let mut t = tensors();
        t.retain(|(n, _, _)| n != "blk.1.ffn_latent_up.weight");
        t.retain(|(n, _, _)| n != "blk.3.ffn_norm.weight");
        let err = load("muse-gone", keys(), &GLOBAL, &t).expect_err("two tensors gone");
        assert!(
            err.contains("blk.1.ffn_latent_up.weight")
                && err.contains("blk.3.ffn_norm.weight")
                && err.contains("2 tensor(s) the model needs are missing"),
            "{err}"
        );
        let mut t = tensors();
        let q = t
            .iter_mut()
            .find(|(n, _, _)| n == "blk.2.attn_q.weight")
            .unwrap();
        q.1 = vec![EMBD, HEADS * 4];
        let err = load("muse-dims", keys(), &GLOBAL, &t).expect_err("attn_q at other dims");
        assert!(
            err.contains("blk.2.attn_q.weight") && err.contains(&format!("want [{EMBD}, 1024]")),
            "{err}"
        );
        let mut t = tensors();
        t.push((
            "blk.0.ffn_down_shexp.weight".to_string(),
            vec![24, EMBD],
            Ty::F32,
        ));
        let err = load("muse-extra", keys(), &GLOBAL, &t).expect_err("a shared expert");
        assert!(err.contains("ffn_down_shexp.weight"), "{err}");
    }

    /// A pre-tokenizer other than `gpt-4o`, or none, is refused by name.
    #[test]
    fn a_pre_tokenizer_other_than_gpt_4o_is_named() {
        for global in [
            vec![("tokenizer.ggml.pre", V::Str("qwen2"))],
            vec![("tokenizer.ggml.chat_template", V::Str("x"))],
            vec![("tokenizer.ggml.pre", V::U32(4))],
        ] {
            let err = load("muse-pre", keys(), &global, &tensors()).expect_err("another pre");
            assert!(err.contains("tokenizer.ggml.pre"), "{err}");
        }
    }

    /// `muse-spark` is the one spelling; nearby strings are unknown, and a
    /// file that spells one is refused by the string it held.
    #[test]
    fn other_muse_strings_are_unknown() {
        assert_eq!(Arch::from_name("muse-spark").unwrap(), Arch::MuseSpark);
        assert_eq!(Arch::MuseSpark.name(), "muse-spark");
        for arch in ["muse_spark", "muse-spark-open", "muse-glimmer", "musespark"] {
            assert!(Arch::from_name(arch).is_err(), "{arch} is known");
            let err = load_with(arch, arch, vec![], &[], &[]).expect_err(arch);
            assert_eq!(err, format!("unsupported architecture \"{arch}\""));
        }
    }
}
