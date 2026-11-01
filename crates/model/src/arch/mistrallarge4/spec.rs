//! The `mistral-large4` reader's [`ModelSpec`]. Every layer is latent
//! attention — MLA with no query latent: the hidden row projects to every
//! head whole, and the per-head up-projections absorb into the query (`up:
//! Absorbed`), the cached row `[latent | rope]` with the rope part zero-width
//! on a NoPE layer — with SSMax on the query, picking its positions by a
//! top-k over index keys projected from the layer input (a LayerNorm with a
//! gain and a bias on the projection, the rope part, the Hadamard rotation,
//! f16), a shared-indexer layer reading the previous full layer's keys and
//! list, and a KV-reader layer reading its source's rows and storing nothing.
//! The rope is YaRN at the LeRoPE factors (`rope_freqs.weight`), net
//! magnitude 1; the router is softmax over the experts with the kept weights
//! renormalized; the first layer is dense and every other layer routes with
//! one shared expert; the head softcaps its logits.
//!
//! The chat surface: the pre-tokenizer `tekken-num3` (the tokenizer row
//! itself is another round's — this reader names the string only), the file's
//! own template, and no tool parser yet (a request with tools is refused by
//! the server; the family's parser is another round's).

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Ffn, IndexKeys, Latent, LatentOut, LatentUp, LayerSpec, LogitEpilogue,
    Mixer, ModelSpec, Moe, Residual, Rope, RopeMode, Router, RouterNorm, Score, Selector, Shared,
    Source, Yarn,
};

use super::hparams::{Hparams, YarnKeys};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;
use crate::placement::ctx::{CtxLimit, PastLimit, check_slot_ctx};

/// `tokenizer.ggml.pre`'s value, the only one this family reads: the
/// pre-tokenizer that splits digit runs into groups of three counted from the
/// right.
const PRE: &str = "tekken-num3";

/// `split`'s description and roles, and the defaults the key read took.
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

/// `split`'s chat surface: the file's pre-tokenizer and template, no parsers
/// of this tree yet. A pre-tokenizer other than [`PRE`], or none, is refused
/// by name.
fn chat(split: &Split) -> Result<ChatSpec, PlacementError> {
    let chat = chat_of(split, None, None)?;
    if chat.pre != PRE {
        return Err(PlacementError::Metadata {
            key: "tokenizer.ggml.pre".to_string(),
            detail: format!("is {:?}, not {PRE:?}", chat.pre),
        });
    }
    Ok(chat)
}

/// The description of the file `hp` was read from, with the chat surface
/// `chat`.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let layers = (0..hp.n_layer)
        .map(|l| layer_of(hp, l))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ModelSpec {
        arch: Arch::MistralLarge4,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: (hp.softcap > 0.0).then_some(LogitEpilogue {
            scale: 1.0,
            softcap: hp.softcap,
        }),
        chat,
    })
}

/// Layer `l`: its mixer by the file's own arrays, its feed-forward block by
/// the dense lead.
fn layer_of(hp: &Hparams, l: usize) -> Result<LayerSpec, PlacementError> {
    let rope_dims = spec_u32("rope.dimension_count", hp.rope_dims)?;
    let rope = hp.rope_layer[l].then(|| Rope {
        mode: RopeMode::NormTail,
        dims: rope_dims,
        base: hp.rope_base,
        yarn: Some(yarn(&hp.yarn)),
        factors: true,
    });
    // A shared indexer layer reads the previous full layer's keys and list.
    let indexer = if hp.indexer_full[l] {
        l
    } else {
        hp.prev_full_indexer(l)
    };
    let keys = if hp.indexer_full[l] {
        Source::Own(IndexKeys::FromInput {
            norm_eps: hp.idx_norm_eps,
        })
    } else {
        Source::From(spec_u32("attention.indexer.types", indexer)?)
    };
    let mixer = Mixer::Latent(Latent {
        heads: spec_u32("attention.head_count", hp.n_head)?,
        q_lora: 0,
        latent: spec_u32("attention.kv_lora_rank", hp.kv_lora)?,
        up: LatentUp::Absorbed {
            qk: spec_u32("attention.key_length_mla", hp.head_k - hp.pe(l))?,
            v: spec_u32("attention.value_length_mla", hp.head_v)?,
        },
        rope,
        q_head_norm: false,
        ssmax: true,
        pe: spec_u32("rope.dimension_count", hp.pe(l))?,
        cache: if hp.owns_cache(l) {
            Source::Own(())
        } else {
            Source::From(spec_u32("attention.kv_source_map", hp.kv_source[l])?)
        },
        out_gate: models::OutGate::None,
        out: LatentOut::Plain,
        window: None,
        sinks: false,
        compress: None,
        select: Some(Selector::StreamTopK {
            heads: spec_u32("attention.indexer.head_count", hp.indexer_heads)?,
            d: spec_u32("attention.indexer.key_length", hp.indexer_d)?,
            k: spec_u32("attention.indexer.top_k", hp.indexer_top_k)?,
            keys,
            list: if hp.indexer_full[l] {
                Source::Own(())
            } else {
                Source::From(spec_u32("attention.indexer.types", indexer)?)
            },
            candidates: None,
        }),
    });
    let ffn = if l < hp.dense_lead {
        Ffn::Dense {
            ff: spec_u32("feed_forward_length", hp.dense_ff)?,
            act: Act::SwiGlu { limit: None },
        }
    } else {
        let shared_ff =
            hp.n_shared
                .checked_mul(hp.expert_ff)
                .ok_or_else(|| PlacementError::Metadata {
                    key: "expert_shared_count".to_string(),
                    detail: format!("{} x {} overflows usize", hp.n_shared, hp.expert_ff),
                })?;
        Ffn::Moe(Moe {
            experts: spec_u32("expert_count", hp.n_expert)?,
            top_k: spec_u32("expert_used_count", hp.n_used)?,
            expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff)?,
            latent: None,
            pre_exp_norm: false,
            post_exp_norm: None,
            act: Act::SwiGlu { limit: None },
            router: Router {
                score: Score::Softmax,
                bias: false,
                norm: RouterNorm::from_legacy_bool(hp.weights_norm),
                scale: hp.weights_scale,
                hash: false,
            },
            shared: Some(Shared {
                ff: spec_u32(
                    "expert_shared_count x expert_feed_forward_length",
                    shared_ff,
                )?,
                act: Act::SwiGlu { limit: None },
                sigmoid_gate: false,
            }),
        })
    };
    Ok(LayerSpec {
        mixer,
        ffn,
        residual: Residual::Plain,
        extras: Vec::new(),
    })
}

/// The YaRN of the keys as the description's.
fn yarn(keys: &YarnKeys) -> Yarn {
    Yarn {
        factor: keys.factor,
        orig_ctx: keys.orig_ctx as u32,
        beta_fast: keys.beta_fast,
        beta_slow: keys.beta_slow,
        magnitude: keys.magnitude(),
    }
}

/// The most positions a load of the file `spec` describes serves today: its
/// `ctx_train` — the selected attention (the gathered flash on a roped layer,
/// the selected walk on a NoPE one) serves any depth the plan holds, past
/// the indexer's own `top_k` included, so the trained context is the only
/// bound and a context past it is refused by name and the load stops, never
/// a cut.
///
/// # Errors
/// [`PastLimit`] when `ctx` is past [`CtxLimit::limit`].
pub fn serve_ctx(ctx: u64, spec: &ModelSpec) -> Result<u64, PastLimit> {
    let bounds = CtxLimit {
        trained: u64::from(spec.ctx_train),
        kernel: None,
        verified: None,
    };
    check_slot_ctx(ctx, &bounds)
}

#[cfg(test)]
mod tests {
    use models::{IndexKeys, OutGate, Residual, RopeMode, RouterNorm, Score, Selector, Source};

    use super::super::names;
    use crate::arch::synthetic::{Ty, V};
    use crate::arch::{Arch, spec};

    /// Five layers holding every kind: 0 dense, roped, a full indexer and its
    /// own cache; 1 roped, a shared indexer; 2 NoPE, a full indexer, its own
    /// cache; 3 roped, a full indexer, its own cache; 4 NoPE, a shared
    /// indexer reading layer 2's cache. The head count of the index is the
    /// kernel's own 32; everything else is small.
    const N: usize = 5;
    const ROPE: [bool; N] = [true, true, false, true, false];
    const FULL: [bool; N] = [true, false, true, true, false];
    const KV_SOURCE: [usize; N] = [0, 1, 2, 3, 2];
    const EMBD: u64 = 64;
    const HEADS: u64 = 2;
    const KV_LORA: u64 = 16;
    const HEAD_K: u64 = 16;
    const HEAD_V: u64 = 16;
    const ROPE_DIMS: u32 = 8;
    const IDX_HEADS: u64 = 32;
    const IDX_D: u64 = 32;
    const TOP_K: u32 = 8;
    const EXPERTS: u64 = 4;
    const USED: u32 = 2;
    const FF: u64 = 32;
    const SHARED: u64 = 1;
    const DENSE_FF: u64 = 64;
    const FACTOR: f32 = 8.0;

    fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(N as u32)),
            ("embedding_length", V::U32(EMBD as u32)),
            ("context_length", V::U32(64)),
            ("vocab_size", V::U32(2)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::U32(1)),
            (
                "attention.key_length",
                V::U32((KV_LORA + ROPE_DIMS as u64) as u32),
            ),
            ("attention.key_length_mla", V::U32(HEAD_K as u32)),
            ("attention.value_length", V::U32(KV_LORA as u32)),
            ("attention.value_length_mla", V::U32(HEAD_V as u32)),
            ("attention.kv_lora_rank", V::U32(KV_LORA as u32)),
            ("attention.q_lora_rank", V::U32(0)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.layer_norm_epsilon", V::F32(1e-6)),
            ("attention.rope_pattern", V::Bools(ROPE.to_vec())),
            ("attention.indexer.head_count", V::U32(IDX_HEADS as u32)),
            ("attention.indexer.key_length", V::U32(IDX_D as u32)),
            ("attention.indexer.top_k", V::U32(TOP_K)),
            ("attention.indexer.types", V::Bools(FULL.to_vec())),
            (
                "attention.kv_source_map",
                V::I32s(KV_SOURCE.iter().map(|&l| l as i32).collect()),
            ),
            ("rope.dimension_count", V::U32(ROPE_DIMS)),
            ("rope.freq_base", V::F32(10_000.0)),
            ("rope.scaling.type", V::Str("yarn")),
            ("rope.scaling.factor", V::F32(FACTOR)),
            ("rope.scaling.original_context_length", V::U32(32)),
            ("rope.scaling.yarn_beta_fast", V::F32(32.0)),
            ("rope.scaling.yarn_beta_slow", V::F32(1.0)),
            ("rope.scaling.yarn_log_multiplier", V::F32(1.0)),
            ("leading_dense_block_count", V::U32(1)),
            ("feed_forward_length", V::U32(DENSE_FF as u32)),
            ("final_logit_softcapping", V::F32(30.0)),
            ("expert_count", V::U32(EXPERTS as u32)),
            ("expert_used_count", V::U32(USED)),
            ("expert_feed_forward_length", V::U32(FF as u32)),
            ("expert_shared_count", V::U32(SHARED as u32)),
            ("expert_gating_func", V::U32(1)),
            ("expert_group_count", V::U32(1)),
            ("expert_group_used_count", V::U32(1)),
            ("expert_weights_norm", V::Bool(true)),
            ("expert_weights_scale", V::F32(1.0)),
        ]
    }

    /// One layer's tensors at the kind the arrays give it, each F32.
    fn layer(l: usize, out: &mut Vec<(String, Vec<u64>, Ty)>) {
        let (e, h) = (EMBD, HEADS);
        let pe = if ROPE[l] { ROPE_DIMS as u64 } else { 0 };
        let qk = HEAD_K - pe;
        let mut row = |stem: &str, dims: Vec<u64>| {
            out.push((crate::arch::names::blk(l, stem), dims, Ty::F32))
        };
        row(names::ATTN_NORM, vec![e]);
        row(names::ATTN_Q, vec![e, h * HEAD_K]);
        row(names::ATTN_K_B, vec![qk, KV_LORA, h]);
        row(names::ATTN_V_B, vec![KV_LORA, HEAD_V, h]);
        row(names::ATTN_OUTPUT, vec![h * HEAD_V, e]);
        row(names::ATTN_SSMAX_W, vec![h]);
        row(names::ATTN_SSMAX_B, vec![h]);
        if KV_SOURCE[l] == l {
            row(names::ATTN_KV_A_MQA, vec![e, KV_LORA + pe]);
            row(names::ATTN_KV_A_NORM, vec![KV_LORA]);
        }
        if FULL[l] {
            row(names::INDEXER_Q_NORM_W, vec![e]);
            row(names::INDEXER_Q_NORM_B, vec![e]);
            row(names::INDEXER_ATTN_Q_B, vec![e, IDX_HEADS * IDX_D]);
            row(names::INDEXER_ATTN_K, vec![e, IDX_D]);
            row(names::INDEXER_K_NORM_W, vec![IDX_D]);
            row(names::INDEXER_K_NORM_B, vec![IDX_D]);
            row(names::INDEXER_PROJ, vec![e, IDX_HEADS]);
            row(names::INDEXER_SSMAX_W, vec![IDX_HEADS]);
            row(names::INDEXER_SSMAX_B, vec![IDX_HEADS]);
        }
        row(names::FFN_NORM, vec![e]);
        if l == 0 {
            row(names::FFN_GATE, vec![e, DENSE_FF]);
            row(names::FFN_UP, vec![e, DENSE_FF]);
            row(names::FFN_DOWN, vec![DENSE_FF, e]);
        } else {
            row(names::FFN_GATE_INP, vec![e, EXPERTS]);
            row(names::FFN_GATE_EXPS, vec![e, FF, EXPERTS]);
            row(names::FFN_UP_EXPS, vec![e, FF, EXPERTS]);
            row(names::FFN_DOWN_EXPS, vec![FF, e, EXPERTS]);
            row(names::FFN_GATE_SHEXP, vec![e, FF * SHARED]);
            row(names::FFN_UP_SHEXP, vec![e, FF * SHARED]);
            row(names::FFN_DOWN_SHEXP, vec![FF * SHARED, e]);
        }
    }

    fn tensors() -> Vec<(String, Vec<u64>, Ty)> {
        let mut out = vec![
            (names::token_embd(), vec![EMBD, 2], Ty::F32),
            (names::output(), vec![EMBD, 2], Ty::F32),
            (names::output_norm(), vec![EMBD], Ty::F32),
            (
                "rope_freqs.weight".to_string(),
                vec![ROPE_DIMS as u64 / 2],
                Ty::F32,
            ),
        ];
        for l in 0..N {
            layer(l, &mut out);
        }
        out
    }

    const GLOBAL: [(&str, V); 2] = [
        ("tokenizer.ggml.pre", V::Str("tekken-num3")),
        ("tokenizer.chat_template", V::Str("{{ bos }}{{ messages }}")),
    ];

    /// The read of a header of `kv` with the file's own tensors, or its error
    /// as text; the file is removed.
    fn load(tag: &str, kv: &[(&str, V)]) -> Result<crate::arch::Read, String> {
        load_with(tag, "mistral-large4", kv, &tensors())
    }

    fn load_with(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>, Ty)],
    ) -> Result<crate::arch::Read, String> {
        let path = crate::arch::synthetic::header_typed(tag, arch, kv, &GLOBAL, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    fn set_key(kv: &mut [(&'static str, V)], key: &str, value: V) {
        let row = kv.iter_mut().find(|(k, _)| *k == key).expect(key);
        row.1 = value;
    }

    /// The layers read as the file spells them: the rope on three of four
    /// layers at the LeRoPE factors with YaRN of net magnitude 1, the NoPE
    /// head all unrotated key and the cache row the latent alone, SSMax
    /// everywhere, the indexer's keys and list owned by the full layers and
    /// read from the previous full one below, the cache owned except layer
    /// 4's read of layer 2's, the first layer dense and the rest routed with
    /// a shared expert, and the head's softcap. A reader that ropes every
    /// layer, gives a shared layer its own list, a reader layer its own
    /// cache, the NoPE head the rope layer's width, or states no SSMax, no
    /// softcap, no shared expert or a router of another score or norm, fails
    /// this.
    #[test]
    fn the_layers_read_as_the_arrays_spell_them() {
        let read = load("ml4-ok", &keys()).expect("the header reads");
        let s = &read.spec;
        assert_eq!(s.arch, models::Arch::MistralLarge4);
        assert_eq!((s.hidden, s.vocab, s.ctx_train), (EMBD as u32, 2, 64));
        assert!(s.mtp.is_empty() && s.hc.is_none() && s.engram.is_none() && s.encoder.is_none());
        for l in 0..N {
            let layer = &s.layers[l];
            let models::Mixer::Latent(a) = &layer.mixer else {
                panic!("layer {l}: {:?}", layer.mixer);
            };
            let pe = u64::from(if ROPE[l] { ROPE_DIMS } else { 0 });
            assert_eq!(
                (a.heads, a.q_lora, a.latent, a.pe),
                (HEADS as u32, 0, KV_LORA as u32, pe as u32),
                "layer {l}"
            );
            assert_eq!(
                a.up,
                models::LatentUp::Absorbed {
                    qk: (HEAD_K - pe) as u32,
                    v: HEAD_V as u32
                },
                "layer {l}"
            );
            let rope = &a.rope;
            if ROPE[l] {
                let rope = rope.as_ref().unwrap();
                assert_eq!(
                    (rope.mode, rope.dims, rope.base, rope.factors),
                    (RopeMode::NormTail, ROPE_DIMS, 10_000.0, true),
                    "layer {l}"
                );
                let yarn = rope.yarn.as_ref().unwrap();
                assert_eq!(
                    (
                        yarn.factor,
                        yarn.orig_ctx,
                        yarn.beta_fast,
                        yarn.beta_slow,
                        yarn.magnitude
                    ),
                    (FACTOR, 32, 32.0, 1.0, 1.0),
                    "layer {l}"
                );
            } else {
                assert!(rope.is_none(), "layer {l}: {:?}", rope);
            }
            assert!(a.ssmax && !a.q_head_norm && a.out_gate == OutGate::None);
            assert_eq!(a.out, models::LatentOut::Plain);
            assert!(a.window.is_none() && !a.sinks && a.compress.is_none());
            assert_eq!(
                a.cache,
                if KV_SOURCE[l] == l {
                    Source::Own(())
                } else {
                    Source::From(KV_SOURCE[l] as u32)
                },
                "layer {l}"
            );
            let Some(Selector::StreamTopK {
                heads,
                d,
                k,
                keys,
                list,
                candidates,
            }) = a.select.as_ref()
            else {
                panic!("layer {l}: {:?}", a.select);
            };
            assert_eq!((*heads, *d, *k), (IDX_HEADS as u32, IDX_D as u32, TOP_K));
            assert!(candidates.is_none());
            let own = FULL[l];
            let from = if own { l } else { [0, 0, 0, 0, 3][l] };
            assert_eq!(
                *keys,
                if own {
                    Source::Own(IndexKeys::FromInput { norm_eps: 1e-6 })
                } else {
                    Source::From(from as u32)
                },
                "layer {l}"
            );
            assert_eq!(
                *list,
                if own {
                    Source::Own(())
                } else {
                    Source::From(from as u32)
                },
                "layer {l}"
            );
            if l == 0 {
                assert_eq!(
                    layer.ffn,
                    models::Ffn::Dense {
                        ff: DENSE_FF as u32,
                        act: models::Act::SwiGlu { limit: None },
                    }
                );
            } else {
                let models::Ffn::Moe(m) = &layer.ffn else {
                    panic!("layer {l}: {:?}", layer.ffn);
                };
                assert_eq!(
                    (m.experts, m.top_k, m.expert_ff, m.latent),
                    (EXPERTS as u32, USED, FF as u32, None)
                );
                assert_eq!(m.act, models::Act::SwiGlu { limit: None });
                assert!(!m.pre_exp_norm && m.post_exp_norm.is_none());
                assert_eq!(
                    (m.router.score, m.router.norm, m.router.scale),
                    (Score::Softmax, RouterNorm::SelectedSum, 1.0)
                );
                assert!(!m.router.bias && !m.router.hash);
                assert_eq!(
                    m.shared,
                    Some(models::Shared {
                        ff: (FF * SHARED) as u32,
                        act: models::Act::SwiGlu { limit: None },
                        sigmoid_gate: false,
                    }),
                    "layer {l}"
                );
            }
            assert_eq!(layer.residual, Residual::Plain);
            assert!(layer.extras.is_empty());
        }
        assert_eq!(
            s.logit_epilogue,
            Some(models::LogitEpilogue {
                scale: 1.0,
                softcap: 30.0,
            })
        );
        assert_eq!(s.chat.pre, "tekken-num3");
        assert!(s.chat.tools.is_none() && s.chat.reasoning.is_none());
        assert!(s.chat.template.is_some());
        assert!(read.defaults.is_empty(), "{:?}", read.defaults);
        assert_eq!(read.tensors.layers, N);
        assert_eq!(
            (read.tensors.experts, read.tensors.experts_used),
            (EXPERTS, USED as u64)
        );
    }

    /// Each key the reader takes is required by name; the three llama.cpp
    /// defaults are taken with a recorded line when the file carries none.
    #[test]
    fn a_missing_key_is_named() {
        for (key, _) in keys() {
            let kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
            let read = load("ml4-key", &kv);
            // An absent dense lead reads as no dense layer, so the same
            // header needs layer 0's routed block to read: the fixture with
            // its dense tensors is refused by the seven missing routed
            // tensors, the default taken either way.
            if key == "leading_dense_block_count" {
                let err = read.expect_err(key);
                assert!(
                    err.contains("7 tensor(s) the model needs are missing"),
                    "{err}"
                );
                let kv: Vec<_> = keys()
                    .into_iter()
                    .filter(|(k, _)| *k != "leading_dense_block_count")
                    .collect();
                let mut t = tensors();
                t.retain(|(n, _, _)| {
                    !n.starts_with("blk.0.ffn_gate.weight")
                        && !n.starts_with("blk.0.ffn_up.weight")
                        && !n.starts_with("blk.0.ffn_down.weight")
                });
                let moe0: Vec<(String, Vec<u64>, Ty)> = tensors()
                    .into_iter()
                    .filter(|(n, _, _)| n.starts_with("blk.1.ffn_") && !n.ends_with("norm.weight"))
                    .map(|(n, d, ty)| (n.replace("blk.1.", "blk.0."), d, ty))
                    .collect();
                t.extend(moe0);
                let read = load_with("ml4-lead", "mistral-large4", &kv, &t).expect(key);
                assert!(read.defaults == ["leading_dense_block_count = 0 (no dense layer)"]);
                let models::Ffn::Moe(m) = &read.spec.layers[0].ffn else {
                    panic!("layer 0 routes");
                };
                assert_eq!((m.experts, m.top_k), (EXPERTS as u32, USED));
                continue;
            }
            if let Some(default) = match key {
                "final_logit_softcapping" => Some("final_logit_softcapping = 0"),
                "rope.scaling.yarn_log_multiplier" => Some("rope.scaling.yarn_log_multiplier = 0"),
                _ => None,
            } {
                let read = read.expect(key);
                assert!(
                    read.defaults.iter().any(|d| d.starts_with(default)),
                    "{key}: no default recorded"
                );
                // A file with no softcap key runs no logit epilogue: the graph
                // caps only on a nonzero bound.
                if key == "final_logit_softcapping" {
                    assert!(read.spec.logit_epilogue.is_none(), "{key}: a cap of 0");
                }
                continue;
            }
            // A file may carry no `vocab_size` (the token list's length
            // stands in, as llama.cpp reads it) and no q_lora (a nonzero one
            // is refused) and no expert group keys (one group is the only
            // plan the router runs): none of the absences takes a default.
            if key == "vocab_size" {
                let read = read.expect(key);
                assert_eq!(read.spec.vocab, 2, "the token list's two tokens");
                continue;
            }
            if matches!(
                key,
                "attention.q_lora_rank" | "expert_group_count" | "expert_group_used_count"
            ) {
                let read = read.expect(key);
                assert!(read.defaults.is_empty(), "{key}: {:?}", read.defaults);
                continue;
            }
            let err = read.expect_err(key);
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    /// A key of a value the model cannot hold is refused by name: another
    /// kv-head count, gating func, index head count or rope scaling, a router
    /// that does not renormalize or that scales, a YaRN multiplier of a net
    /// magnitude other than 1, an original context past ggml's int, a value
    /// or key length the latent does not sum to, a q_lora, a shared first
    /// indexer, a KV source above its layer, itself a reader or of another
    /// rope-ness, a rope width past the head or the index head, a softcap
    /// under 0, or a pattern, type or source array of another length.
    #[test]
    fn a_key_of_another_value_is_named() {
        for (key, value, want) in [
            ("attention.head_count_kv", V::U32(4), "one row a position"),
            ("expert_gating_func", V::U32(2), "softmax"),
            (
                "attention.indexer.head_count",
                V::U32(16),
                "the score kernel runs 32",
            ),
            ("rope.scaling.type", V::Str("none"), "YaRN"),
            (
                "rope.scaling.yarn_log_multiplier",
                V::F32(2.0),
                "the rope table applies none (net 1)",
            ),
            (
                "rope.scaling.original_context_length",
                V::U32(3_000_000_000),
                "past ggml's int",
            ),
            (
                "expert_weights_norm",
                V::Bool(false),
                "renormalizes the kept weights",
            ),
            (
                "expert_weights_scale",
                V::F32(2.0),
                "the router applies no scale (1)",
            ),
            ("attention.value_length", V::U32(32), "the kv_lora_rank 16"),
            (
                "attention.key_length",
                V::U32(32),
                "not kv_lora_rank 16 + rope.dimension_count 8",
            ),
            ("attention.q_lora_rank", V::U32(64), "no q_lora"),
            (
                "attention.indexer.types",
                V::Bools(vec![false, true, true, true, true]),
                "the first layer must have a full indexer",
            ),
            (
                "attention.rope_pattern",
                V::Bools(vec![true; N - 1]),
                "has 4 values for 5 layers",
            ),
            (
                "attention.kv_source_map",
                V::I32s(vec![1, 1, 2, 3, 2]),
                "is 1 at layer 0, above it",
            ),
            (
                "attention.kv_source_map",
                V::I32s(vec![0, 0, 2, 1, 2]),
                "itself a reader of layer 0",
            ),
            (
                "attention.kv_source_map",
                V::I32s(vec![0, 1, 2, 2, 2]),
                "the pair's rope-ness differs",
            ),
            (
                "attention.indexer.key_length",
                V::U32(4),
                "under the rope width 8",
            ),
            (
                "final_logit_softcapping",
                V::F32(-1.0),
                "not a finite nonnegative bound",
            ),
            (
                "final_logit_softcapping",
                V::F32(f32::NAN),
                "not a finite nonnegative bound",
            ),
            (
                "final_logit_softcapping",
                V::F32(f32::INFINITY),
                "not a finite nonnegative bound",
            ),
        ] {
            let mut kv = keys();
            set_key(&mut kv, key, value);
            let err = load("ml4-other", &kv).expect_err(key);
            assert!(err.contains(key), "{key}: {err}");
            assert!(err.contains(want), "{key}: want {want:?} in {err}");
        }
        // A rope width past the head is refused by the head's own key.
        for dims in [17u32, 20] {
            let mut kv = keys();
            set_key(&mut kv, "rope.dimension_count", V::U32(dims));
            let err = load("ml4-rot", &kv).expect_err("a rope width");
            assert!(
                err.contains("rope.dimension_count") && err.contains(&format!("is {dims}")),
                "{err}"
            );
        }
    }

    /// A KV source pair whose rope-ness differs is refused naming both
    /// sides: here a NoPE layer asked to read a roped layer's cache.
    /// Mutant: the rope-ness check dropped (the mismatched pair reads on).
    #[test]
    fn a_kv_source_of_another_rope_ness_is_named_with_both() {
        let mut kv = keys();
        set_key(
            &mut kv,
            "attention.kv_source_map",
            V::I32s(vec![0, 1, 2, 3, 1]),
        );
        let err = load("ml4-kv", &kv).expect_err("layer 4 reads layer 1");
        assert!(
            err.contains("attention.kv_source_map")
                && err.contains("the source ropes and the reader ropes not"),
            "{err}"
        );
    }

    /// A tensor missing is named with the count of the others, one of other
    /// dims names its want, and a vision or DSpark tensor no layer of this
    /// architecture carries is named with the rest of its set.
    #[test]
    fn a_tensor_missing_of_other_dims_or_unclaimed_is_named() {
        let mut t = tensors();
        t.retain(|(n, _, _)| n != "blk.2.attn_kv_a_mqa.weight");
        t.retain(|(n, _, _)| n != "blk.3.indexer.q_norm.bias");
        let err =
            load_with("ml4-gone", "mistral-large4", &keys(), &t).expect_err("two tensors gone");
        assert!(
            err.contains("blk.2.attn_kv_a_mqa.weight")
                && err.contains("blk.3.indexer.q_norm.bias")
                && err.contains("2 tensor(s) the model needs are missing"),
            "{err}"
        );
        let mut t = tensors();
        // Layer 2 is NoPE, its whole head unrotated key: 16 a side, not the
        // rope layer's 8.
        let q = t
            .iter_mut()
            .find(|(n, _, _)| n == "blk.2.attn_k_b.weight")
            .unwrap();
        q.1 = vec![8, KV_LORA, HEADS];
        let err =
            load_with("ml4-dims", "mistral-large4", &keys(), &t).expect_err("k_b at other dims");
        assert!(
            err.contains("blk.2.attn_k_b.weight") && err.contains("want [16, 16, 2]"),
            "{err}"
        );
        for (name, what) in [
            ("v.patch_embd.weight", "a vision projector"),
            ("markov_w1.weight", "a DSpark stage"),
        ] {
            let mut t = tensors();
            t.push((name.to_string(), vec![16], Ty::F32));
            let err = load_with("ml4-extra", "mistral-large4", &keys(), &t).expect_err(what);
            assert!(err.contains(name), "{what}: {err}");
        }
    }

    /// A pre-tokenizer other than `tekken-num3`, or none, is refused by
    /// name; `mistral-large4` is the one spelling, and a file that spells a
    /// nearby string is refused by the string it held.
    #[test]
    fn another_pre_tokenizer_or_architecture_is_named() {
        for global in [vec![("tokenizer.ggml.pre", V::Str("tekken"))], vec![]] {
            let path = crate::arch::synthetic::header_typed(
                "ml4-pre",
                "mistral-large4",
                &keys(),
                &global,
                &tensors(),
            );
            let split = gguf::Split::open(&path).unwrap();
            let err = spec(&split).expect_err("another pre").to_string();
            let _ = std::fs::remove_file(&path);
            assert!(err.contains("tokenizer.ggml.pre"), "{err}");
        }
        assert_eq!(
            Arch::from_name("mistral-large4").unwrap(),
            Arch::MistralLarge4
        );
        assert_eq!(Arch::MistralLarge4.name(), "mistral-large4");
        for arch in ["mistral_large4", "mistral4", "mistral-large"] {
            let err = load_with(arch, arch, &[], &tensors()).expect_err(arch);
            assert_eq!(err, format!("unsupported architecture \"{arch}\""));
        }
    }

    /// A context past the indexer's own top-k serves — the selected
    /// attention's own depth — and one past the trained context is refused
    /// naming it. Mutant: the trained bound dropped (a cut, or none).
    #[test]
    fn a_context_past_the_top_k_serves_and_past_the_trained_one_is_named() {
        let read = load("ml4-ctx", &keys()).expect("the header reads");
        assert_eq!(super::serve_ctx(TOP_K as u64 + 1, &read.spec), Ok(64));
        assert_eq!(super::serve_ctx(64, &read.spec), Ok(64));
        let err = super::serve_ctx(65, &read.spec).expect_err("past the trained context");
        assert!(
            err.to_string().contains("trained at 64") && err.to_string().contains("limit of 64"),
            "{err}"
        );
    }
}
