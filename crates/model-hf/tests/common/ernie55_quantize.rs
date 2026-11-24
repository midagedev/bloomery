//! Shared by the ERNIE 5.5 quantize gate (`ernie55_quantize.rs`): a toy ERNIE 5.5 checkpoint folder on disk with
//! bytes in every format the released one stores, a model file as the quantizer's [`Source`], the OCP MXFP4 decode
//! written here from the format's definition, and the clauses a GGUF written from a folder has to hold.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use gguf::quant::{GgmlType, Q8Block, dequant_row, half_to_f32};
use gguf::{GgmlType as Ty, Split, TensorInfo, Value};
use model::arch::ernie55::hparams::Hparams;
use model::part::{Source as _, Take};
use model_hf::ModelFile;
use quantize::source::Source;
use quantize::{Error as QError, Result as QResult};
use safetensors::{Dtype, Folder};
use serde_json::{Value as Json, json};

/// The released `config.json`, as the folder holds it (the cut of three layers differs in its layer count and widths).
pub const REAL_CONFIG: &str = r#"{"activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,"add_full_attention_sink_bias":true,"architectures":["Ernie5_5_MoeForCausalLM"],"attention_bias":false,"bos_token_id":1,"compress_ratios":[128,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2],"compress_rope_theta":160000.0,"csa_dense_mode":true,"dspark_block_size":7,"dspark_markov_rank":256,"dspark_noise_token_id":1024,"dspark_target_layer_ids":[40,41,42],"enable_hyper_connections":true,"eos_token_id":2,"experimental_attention_variant":"dsv4_hybrid","first_k_dense_replace":1,"gated_attn_act":"sigmoid","gated_attn_use_q_lora":true,"gated_linear_unit":true,"hc_eps":1e-06,"hc_mult":4,"hc_sinkhorn_iters":20,"hca_rope_type":"yarn","head_dim":512,"hidden_act":"situ","hidden_size":4096,"hybrid_mla_attention":"mqa_dsa","hybrid_mla_kv_lora_rank":512,"hybrid_mla_num_attention_heads":64,"hybrid_mla_num_key_value_heads":64,"hybrid_mla_q_lora_rank":1024,"hybrid_mla_qk_nope_head_dim":192,"hybrid_mla_qk_rope_head_dim":64,"hybrid_mla_v_head_dim":256,"index_head_dim":128,"index_n_heads":64,"index_topk":2048,"intermediate_size":14336,"kv_lora_rank":512,"latent_moe_use_norm":true,"max_position_embeddings":1048576,"model_type":"ernie5_5_moe","moe_group_experts":false,"moe_intermediate_size":2048,"moe_latent_size":2048,"moe_layer_freq":1,"moe_shared_expert_gate":false,"moe_split_feature_routing":true,"moe_token_dispatcher_type":"deepep","mqa_split_kv_b_proj":true,"mtp_shared_last_layer":true,"multimax":"lm_head","n_group":1,"n_routed_experts":512,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":10,"num_hidden_layers":43,"num_key_value_heads":1,"num_nextn_predict_layers":0,"o_groups":8,"o_lora_rank":1024,"pad_token_id":0,"q_lora_rank":1024,"qk_nope_head_dim":448,"qk_norm_type":"per_head","qk_pos_emb_head_dim":64,"qk_rope_head_dim":64,"quantization_config":{"activation_scheme":"dynamic","expert_dtype":"mxfp4","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},"rms_norm_eps":1e-06,"rope_interleave":true,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},"rope_theta":8000000,"rope_type":"rope","rotary_base":8000000,"rotary_percent":1.0,"routed_scaling_factor":3.0,"routed_scaling_factor_learnable":true,"scoring_func":"sigmoid","sliding_window":128,"tie_word_embeddings":false,"topk_group":1,"topk_method":"noaux_tc","torch_dtype":"bfloat16","transformers_version":"4.46.1","use_attn_merger":true,"use_bias":false,"use_cache":true,"use_dense_mtp":false,"use_fast_hadamard":true,"use_gated_attn":true,"use_mla":false,"use_qk_norm":true,"use_vha_attention":true,"v_head_dim":512,"vocab_size":201216}"#;

// The toy's widths. Every first dim of an F8 weight is a whole 128-block along its columns.
pub const LAYERS: usize = 3;
pub const DENSE_LAYERS: usize = 1;
const E: usize = 256;
const FF: usize = 512;
const HEADS: usize = 2;
const D: usize = 128;
const ROPE: usize = 64;
const Q_LORA: usize = 384;
const KV_LORA: usize = 512;
const O_GROUPS: usize = 2;
const O_LORA: usize = 128;
const NOPE: usize = 192;
const V: usize = 256;
const INDEX_HEADS: usize = 2;
const INDEX_DIM: usize = 128;
const EFF: usize = 384;
const LATENT: usize = 128;
pub const N_EXP: usize = 4;
/// The toy vocabulary: the nine specials, the byte-level token of every byte UTF-8 text holds (243: a text never
/// holds 0xC0, 0xC1 or 0xF5 to 0xFF) and the one token its single merge joins (a GGUF array holds at least one
/// element).
pub const VOCAB: usize = 9 + 243 + 1;
const RANK: usize = 16;
const RATIO: usize = 128;
/// A hyper-connection mixer's rows (the four streams of the hidden width) and its columns.
const MIX_IN: usize = 4 * E;
const MIX_OUT: usize = 24;

/// Layer 2 is the latent-attention one.
fn dsa(l: usize) -> bool {
    l == 2
}

/// The real config at the toy's widths.
pub fn toy_config() -> Json {
    let mut c: Json = serde_json::from_str(REAL_CONFIG).expect("the config parses");
    for (key, v) in [
        ("hidden_size", E),
        ("intermediate_size", FF),
        ("num_hidden_layers", LAYERS),
        ("first_k_dense_replace", DENSE_LAYERS),
        ("num_attention_heads", HEADS),
        ("hybrid_mla_num_attention_heads", HEADS),
        ("hybrid_mla_num_key_value_heads", HEADS),
        ("head_dim", D),
        ("v_head_dim", D),
        ("qk_nope_head_dim", D - ROPE),
        ("q_lora_rank", Q_LORA),
        ("hybrid_mla_q_lora_rank", Q_LORA),
        ("kv_lora_rank", KV_LORA),
        ("hybrid_mla_kv_lora_rank", KV_LORA),
        ("o_groups", O_GROUPS),
        ("o_lora_rank", O_LORA),
        ("hybrid_mla_qk_nope_head_dim", NOPE),
        ("hybrid_mla_v_head_dim", V),
        ("index_n_heads", INDEX_HEADS),
        ("index_head_dim", INDEX_DIM),
        ("index_topk", 16),
        ("moe_intermediate_size", EFF),
        ("moe_latent_size", LATENT),
        ("n_routed_experts", N_EXP),
        ("num_experts_per_tok", 2),
        ("vocab_size", VOCAB),
    ] {
        c[key] = json!(v);
    }
    c["compress_ratios"] = json!([RATIO, RATIO, -2]);
    c["dspark_target_layer_ids"] = json!([0, 1, 2]);
    c
}

/// A byte of the toy's stored data: a hash of the tensor's name and the byte's place in it, so every tensor differs
/// and a tensor is the same wherever it is read.
fn byte(name: &str, i: usize) -> u8 {
    let h = name.bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    });
    (h.wrapping_add((i as u32).wrapping_mul(0x9e37_79b9)) >> 13) as u8
}

/// A finite f32 of the toy's in [-1, 1).
fn unit(name: &str, i: usize) -> f32 {
    (u32::from(byte(name, 2 * i)) | u32::from(byte(name, 2 * i + 1)) << 8) as f32 / 65_536.0 - 0.5
}

/// The stored scale grid of an F8 weight of `rows` by `k`: 128 by 128 blocks, rounded up.
fn grid(rows: usize, k: usize) -> Vec<usize> {
    vec![rows.div_ceil(128), k.div_ceil(128)]
}

/// One stored tensor: name, dtype, shape and bytes.
pub type Stored = (String, Dtype, Vec<usize>, Vec<u8>);

/// The toy folder's tensors, as the released checkpoint names and shapes them.
pub fn toy_tensors() -> Vec<Stored> {
    let mut t: Vec<Stored> = Vec::new();
    // An F8_E4M3 weight of `rows` by `k` and its f32 block scale, `<name>_scale_inv`: no code is E4M3's NaN.
    let f8 = |t: &mut Vec<Stored>, name: &str, rows: usize, k: usize| {
        t.push((
            name.to_string(),
            Dtype::F8E4M3,
            vec![rows, k],
            (0..rows * k)
                .map(|i| {
                    let b = byte(name, i);
                    if b & 0x7f == 0x7f { b ^ 0x01 } else { b }
                })
                .collect(),
        ));
        t.push((
            format!("{name}_scale_inv"),
            Dtype::F32,
            grid(rows, k),
            (0..grid(rows, k).iter().product::<usize>())
                .flat_map(|i| (2.0f32.powi(-3 + (i % 5) as i32)).to_le_bytes())
                .collect(),
        ));
    };
    // A routed expert projection of `rows` by `k`: I8 pairs of E2M1 codes and an E8M0 scale a 32 values.
    let mx = |t: &mut Vec<Stored>, name: &str, rows: usize, k: usize| {
        t.push((
            name.to_string(),
            Dtype::I8,
            vec![rows, k / 2],
            (0..rows * k / 2).map(|i| byte(name, i)).collect(),
        ));
        t.push((
            format!("{name}_scale_inv"),
            Dtype::F8E8M0,
            vec![rows, k / 32],
            (0..rows * k / 32).map(|i| (118 + i % 8) as u8).collect(),
        ));
    };
    let bf16 = |t: &mut Vec<Stored>, name: &str, shape: &[usize]| {
        let n = shape.iter().product::<usize>();
        t.push((
            name.to_string(),
            Dtype::Bf16,
            shape.to_vec(),
            (0..n)
                .flat_map(|i| ((unit(name, i).to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
        ));
    };
    let f32s = |t: &mut Vec<Stored>, name: &str, shape: &[usize]| {
        let n = shape.iter().product::<usize>();
        t.push((
            name.to_string(),
            Dtype::F32,
            shape.to_vec(),
            (0..n)
                .flat_map(|i| (unit(name, i) * 2.0).to_le_bytes())
                .collect(),
        ));
    };
    bf16(&mut t, "model.embed_tokens.weight", &[VOCAB, E]);
    bf16(&mut t, "lm_head.weight", &[VOCAB, E]);
    bf16(&mut t, "model.norm.weight", &[E]);
    bf16(&mut t, "model.lm_head.multimax_ranges", &[4]);
    bf16(&mut t, "model.lm_head.multimax_ts", &[4]);
    f32s(&mut t, "model.mhc_contract.hc_head_fn", &[MIX_IN, 4]);
    f32s(&mut t, "model.mhc_contract.hc_head_base", &[4]);
    f32s(&mut t, "model.mhc_contract.hc_head_scale", &[1]);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        let n = |rest: &str| format!("{p}{rest}");
        bf16(&mut t, &n("input_layernorm.weight"), &[E]);
        bf16(&mut t, &n("post_attention_layernorm.weight"), &[E]);
        bf16(&mut t, &n("self_attn.vha_postmix_U"), &[HEADS, RANK]);
        bf16(&mut t, &n("self_attn.vha_postmix_V"), &[HEADS, RANK]);
        for block in ["self_attention_hyper_connection", "mlp_hyper_connection"] {
            f32s(
                &mut t,
                &n(&format!("{block}.mapping_proj.weight")),
                &[MIX_IN, MIX_OUT],
            );
            f32s(&mut t, &n(&format!("{block}.bias")), &[MIX_OUT]);
            for alpha in ["alpha_pre", "alpha_post", "alpha_res"] {
                f32s(&mut t, &n(&format!("{block}.{alpha}")), &[1]);
            }
        }
        if dsa(l) {
            f8(&mut t, &n("self_attn.gate_proj.weight"), HEADS * V, Q_LORA);
            f8(&mut t, &n("self_attn.q_a_proj.weight"), Q_LORA, E);
            bf16(&mut t, &n("self_attn.q_a_layernorm.weight"), &[Q_LORA]);
            f8(
                &mut t,
                &n("self_attn.q_b_proj.weight"),
                HEADS * (NOPE + ROPE),
                Q_LORA,
            );
            f8(
                &mut t,
                &n("self_attn.kv_a_proj_with_mqa.weight"),
                KV_LORA + ROPE,
                E,
            );
            bf16(&mut t, &n("self_attn.kv_a_layernorm.weight"), &[KV_LORA]);
            f8(
                &mut t,
                &n("self_attn.kv_b_proj.weight"),
                HEADS * (NOPE + V),
                KV_LORA,
            );
            bf16(
                &mut t,
                &n("self_attn.core_attention.softmax_offset"),
                &[HEADS],
            );
            f8(&mut t, &n("self_attn.o_proj.weight"), E, HEADS * V);
            let ix = "self_attn.core_attention.indexer.";
            f8(
                &mut t,
                &n(&format!("{ix}wq_b.weight")),
                INDEX_HEADS * INDEX_DIM,
                Q_LORA,
            );
            bf16(&mut t, &n(&format!("{ix}wk.weight")), &[INDEX_DIM, E]);
            bf16(
                &mut t,
                &n(&format!("{ix}weights_proj.weight")),
                &[INDEX_HEADS, E],
            );
            bf16(&mut t, &n(&format!("{ix}k_norm.weight")), &[INDEX_DIM]);
            bf16(&mut t, &n(&format!("{ix}k_norm.bias")), &[INDEX_DIM]);
        } else {
            f8(&mut t, &n("self_attn.gate_proj.weight"), HEADS * D, Q_LORA);
            f8(&mut t, &n("self_attn.linear_q_down_proj.weight"), Q_LORA, E);
            bf16(&mut t, &n("self_attn.q_layernorm.weight"), &[Q_LORA]);
            f8(
                &mut t,
                &n("self_attn.linear_q_up_proj.weight"),
                HEADS * D,
                Q_LORA,
            );
            bf16(&mut t, &n("self_attn.kv_layernorm.weight"), &[D]);
            f8(&mut t, &n("self_attn.linear_kv_proj.weight"), D, E);
            f32s(&mut t, &n("self_attn.core_attention.attn_sink"), &[HEADS]);
            f8(
                &mut t,
                &n("self_attn.linear_o_group_proj"),
                O_GROUPS * O_LORA,
                HEADS * D / O_GROUPS,
            );
            f8(&mut t, &n("self_attn.o_proj.weight"), E, O_GROUPS * O_LORA);
            let c = "self_attn.core_attention.compressor.";
            bf16(&mut t, &n(&format!("{c}linear_wkv.weight")), &[D, E]);
            bf16(&mut t, &n(&format!("{c}linear_wgate.weight")), &[D, E]);
            f32s(&mut t, &n(&format!("{c}ape")), &[RATIO, D]);
            bf16(&mut t, &n(&format!("{c}norm.weight")), &[D]);
        }
        if l < DENSE_LAYERS {
            f8(&mut t, &n("mlp.gate_proj.weight"), FF, E);
            f8(&mut t, &n("mlp.up_proj.weight"), FF, E);
            f8(&mut t, &n("mlp.down_proj.weight"), E, FF);
        } else {
            let m = "block_sparse_moe.";
            bf16(&mut t, &n(&format!("{m}gate.weight")), &[N_EXP, E]);
            bf16(&mut t, &n(&format!("{m}gate.weight_1")), &[N_EXP, E]);
            f32s(&mut t, &n(&format!("{m}e_score_correction_bias")), &[N_EXP]);
            f32s(
                &mut t,
                &n(&format!("{m}gate.routed_scaling_factor_param")),
                &[N_EXP],
            );
            f8(&mut t, &n(&format!("{m}fc1_latent_proj.weight")), LATENT, E);
            f8(&mut t, &n(&format!("{m}fc2_latent_proj.weight")), E, LATENT);
            bf16(&mut t, &n(&format!("{m}latent_norm.weight")), &[LATENT]);
            f8(&mut t, &n(&format!("{m}shared_experts.w1.weight")), EFF, E);
            f8(&mut t, &n(&format!("{m}shared_experts.w3.weight")), EFF, E);
            f8(&mut t, &n(&format!("{m}shared_experts.w2.weight")), E, EFF);
            for e in 0..N_EXP {
                mx(
                    &mut t,
                    &n(&format!("{m}experts.{e}.w1.weight")),
                    EFF,
                    LATENT,
                );
                mx(
                    &mut t,
                    &n(&format!("{m}experts.{e}.w3.weight")),
                    EFF,
                    LATENT,
                );
                mx(
                    &mut t,
                    &n(&format!("{m}experts.{e}.w2.weight")),
                    LATENT,
                    EFF,
                );
            }
        }
    }
    t
}

/// The toy's `tokenizer.json`: the files the released folder ships, but for the vocabulary, which holds the nine
/// specials (ids 0 to 8) and the byte-level token of every byte UTF-8 text holds, and no merges; the post-processor
/// is the bare `<|ERNIE_MASK|>` template.
fn toy_tokenizer() -> Json {
    let specials = [
        "<|ERNIE_PAD|>",
        "<|ERNIE_BEGIN|>",
        "<|ERNIE_MASK|>",
        "<|ernie:3|>",
        "<|ernie:4|>",
        "<|ernie:5|>",
        "<|ernie:6|>",
        "<|ernie:7|>",
        "<|ernie:im_end|>",
    ];
    let mut vocab = serde_json::Map::new();
    for (id, name) in specials.iter().enumerate() {
        vocab.insert((*name).to_string(), json!(id));
    }
    for (byte, &cp) in tokenizer::byte_to_cpt_table().iter().enumerate() {
        if !matches!(byte, 0xC0 | 0xC1 | 0xF5..=0xFF) {
            let piece = char::from_u32(cp).expect("a byte's codepoint").to_string();
            vocab.insert(piece, json!(vocab.len()));
        }
    }
    vocab.insert("Ġa".to_string(), json!(vocab.len()));
    assert_eq!(vocab.len(), VOCAB, "the toy vocabulary is the config's");
    let added: Vec<Json> = specials
        .iter()
        .enumerate()
        .map(|(id, name)| {
            json!({"id":id,"content":name,"single_word":false,"lstrip":false,"rstrip":false,
                "normalized":false,"special":true})
        })
        .collect();
    let row = hf::vocab::PRE
        .iter()
        .find(|r| r.0 == "ernie5")
        .expect("the ernie5 row");
    json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":added,"normalizer":null,
        "pre_tokenizer":serde_json::from_str::<Json>(row.1).unwrap(),
        "post_processor":{"type":"TemplateProcessing",
            "single":[{"SpecialToken":{"id":"<|ERNIE_MASK|>","type_id":0}},
                {"Sequence":{"id":"A","type_id":0}}],
            "pair":[{"SpecialToken":{"id":"<|ERNIE_MASK|>","type_id":0}},
                {"Sequence":{"id":"A","type_id":0}},
                {"SpecialToken":{"id":"<|ERNIE_MASK|>","type_id":1}},
                {"Sequence":{"id":"B","type_id":1}}],
            "special_tokens":{"<|ERNIE_MASK|>":{"id":"<|ERNIE_MASK|>","ids":[2],
                "tokens":["<|ERNIE_MASK|>"]}}},
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
        "model":{"type":"BPE","dropout":null,"unk_token":"<|ERNIE_PAD|>",
            "continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,
            "byte_fallback":true,"ignore_merges":false,"vocab":vocab,"merges":["Ġ a"]}})
}

/// The toy folder at `dir`: its tensors in two shards and an index, the config, and the tokenizer's three files.
pub fn write_toy_folder(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let tensors = toy_tensors();
    let half = tensors.len() / 2;
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    for (file, part) in [
        ("model-00001-of-00002.safetensors", &tensors[..half]),
        ("model-00002-of-00002.safetensors", &tensors[half..]),
    ] {
        let views: Vec<(&str, &str, &[usize], &[u8])> = part
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), d.name(), s.as_slice(), b.as_slice()))
            .collect();
        std::fs::write(dir.join(file), safetensors::write(&views)).unwrap();
        for (n, _, _, _) in part {
            weight_map.insert(n.clone(), file.to_string());
        }
    }
    std::fs::write(
        dir.join(safetensors::Folder::INDEX),
        json!({ "metadata": {}, "weight_map": weight_map }).to_string(),
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), toy_config().to_string()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), toy_tokenizer().to_string()).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"tokenizer_class":"ErnieChatTokenizer","model_max_length":1048576,"clean_up_tokenization_spaces":false,"bos_token":"<|ERNIE_BEGIN|>","pad_token":"<|ERNIE_PAD|>","eos_token":"<|ernie:im_end|>"}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        r#"{"bos_token_id":1,"eos_token_id":[2,8],"pad_token_id":0}"#,
    )
    .unwrap();
}

/// A model file as the quantizer's [`Source`]: the split's header and tensors, and the bytes the file's reader
/// holds of each, whole, by header name.
pub struct FileSource {
    file: Box<ModelFile>,
    /// The folder the file opens, when it is one.
    dir: Option<PathBuf>,
}

impl FileSource {
    /// The checkpoint folder `dir`, opened as a load opens it (its tensors made as the part source makes them).
    pub fn folder(dir: &Path) -> FileSource {
        FileSource {
            file: Box::new(
                ModelFile::open(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
            ),
            dir: Some(dir.to_path_buf()),
        }
    }

    /// The whole bytes of the tensor `name` as the file's reader holds them. Of a folder: the part source's, planned
    /// for that tensor alone. The part source refuses a mixer's `*_scale` vector (three stored scalars joined: it
    /// reads stacks of matrices and single sources), so those are joined here from the stored scalars in the map's
    /// order, `alpha_pre`, `alpha_post`, `alpha_res`, as the folder's `prepare` split holds them.
    pub fn whole(&self, name: &str) -> Vec<u8> {
        let ModelFile::Folder(ref opened) = *self.file else {
            let ModelFile::Gguf(ref split) = *self.file else {
                panic!("{}: neither a folder nor a GGUF split", self.label());
            };
            return split
                .data(name)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .to_vec();
        };
        let one = opened
            .source
            .plan(|n, _, _| {
                Ok::<_, std::convert::Infallible>(if n == name { Take::Whole } else { Take::Skip })
            })
            .and_then(|planned| match planned.as_slice() {
                [one] => opened.source.read(one).map(|part| part.spans().concat()),
                _ => panic!("{name}: {} parts planned", planned.len()),
            });
        match one {
            Ok(bytes) => bytes,
            Err(e) => joined_scalars(name, &opened.header, self.dir.as_deref().expect("a folder"))
                .unwrap_or_else(|| panic!("{name}: the part source refuses it: {e}")),
        }
    }
}

/// The three stored scalars the mixer's `hc_{attn,ffn}_scale.weight` of layer `l` joins, as the F32 vector of three
/// the map declares; `None` for another name.
fn joined_scalars(name: &str, header: &Split, dir: &Path) -> Option<Vec<u8>> {
    let (l, block) = name.strip_prefix("blk.")?.split_once('.')?;
    let block = match block {
        "hc_attn_scale.weight" => "self_attention_hyper_connection",
        "hc_ffn_scale.weight" => "mlp_hyper_connection",
        _ => return None,
    };
    assert_eq!(
        header.find(name).map(|(_, t)| (t.ty, t.dims.as_slice())),
        Some((Ty::F32, &[3][..])),
        "{name}"
    );
    let planes = Folder::open(dir).unwrap();
    let mut out = Vec::new();
    for alpha in ["alpha_pre", "alpha_post", "alpha_res"] {
        let raw = planes
            .raw(&format!("model.layers.{l}.{block}.{alpha}"))
            .unwrap();
        assert_eq!(
            (raw.dtype, raw.shape, raw.bytes.len()),
            (Dtype::F32, &[1][..], 4)
        );
        out.extend_from_slice(raw.bytes);
    }
    Some(out)
}

impl Source for FileSource {
    fn label(&self) -> String {
        format!("{} (model file)", self.file.name().unwrap_or_default())
    }

    fn keys(&self) -> &[(String, Value)] {
        self.file.header().keys()
    }

    fn shard_count(&self) -> usize {
        1
    }

    fn tensors(&self) -> Vec<(usize, TensorInfo)> {
        self.file
            .header()
            .iter_tensors()
            .map(|(_, t)| (0, t.clone()))
            .collect()
    }

    fn read(&self, name: &str, bytes: Range<u64>) -> QResult<Cow<'_, [u8]>> {
        let info = self
            .file
            .header()
            .find(name)
            .map(|(_, t)| t)
            .ok_or_else(|| QError::Refused(format!("{name}: not a tensor of the model file")))?;
        if bytes.start > bytes.end || bytes.end > info.nbytes {
            return Err(QError::Refused(format!(
                "{name}: bytes {}..{} of a tensor of {}",
                bytes.start, bytes.end, info.nbytes
            )));
        }
        if bytes == (0..info.nbytes) {
            return Ok(Cow::Owned(self.whole(name)));
        }
        let all = self.whole(name);
        Ok(Cow::Owned(
            all[bytes.start as usize..bytes.end as usize].to_vec(),
        ))
    }
}

/// E2M1's values: code `c` is `c & 7` on the ladder 0, 0.5, 1, 1.5, 2, 3, 4, 6, negated when bit 3 is set.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The MXFP4 (OCP) block's decode of one expert's stored planes, written from the format's definition: the weight
/// `I8 [rows, k/2]` holds value `2i` in the low nibble of byte `i` and value `2i + 1` in its high nibble, the scale
/// `F8_E8M0 [rows, k/32]` one E8M0 a 32 values, and a value is its code's E2M1 value times `2^(E - 127)`. Row-major
/// f32, `rows * k` values.
pub fn ocp_mxfp4(weight: &[u8], scale: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let nibble = |codes: &[u8], n: usize| {
        let c = usize::from((codes[n / 2] >> (4 * (n % 2))) & 0x0f);
        if c & 8 == 0 { E2M1[c] } else { -E2M1[c & 7] }
    };
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let codes = &weight[r * k / 2..(r + 1) * k / 2];
        for (b, &e) in scale[r * k / 32..(r + 1) * k / 32].iter().enumerate() {
            assert_ne!(e, 0xFF, "row {r} block {b}: E8M0 0xFF is NaN, no scale");
            let s = 2.0f32.powi(i32::from(e) - 127);
            for j in 0..32 {
                out.push(nibble(codes, b * 32 + j) * s);
            }
        }
    }
    out
}

/// The largest error a Q8_0 block makes against the values `x` it was written from, derived from ggml's reference
/// rule (`Q8Block::quantize_ref`: `d = amax / 127` in f32, stored as f16 `D`; each code `round(x / d)`; a value
/// reads back `D * q`): a code is within `1/2 + 127 * 2^-23` of `x / d` (the rounding, and the two f32 roundings of
/// `1 / d` and of the product), `D` is within `max(d * 2^-11, 2^-25)` of `d` (an f16's half ulp, a subnormal's), and
/// the read-back product rounds once more; so `|D * q - x|` is under
/// `d * (1/2 + 127 * 2^-23) + 127 * max(d * 2^-11, 2^-25) * (1 + 2^-24)`, with `d` the block's `amax / 127`.
pub fn q8_0_band(amax: f32) -> f32 {
    let d = amax / 127.0;
    d * (0.5 + 127.0 * f32::EPSILON)
        + 127.0 * (d * 2.0f32.powi(-11)).max(2.0f32.powi(-25)) * (1.0 + f32::EPSILON)
}

/// The rows of `src`, whole rows of `ne0` values of `ty`, as f32.
pub fn rows(ty: GgmlType, src: &[u8], ne0: usize) -> Vec<Vec<f32>> {
    let rb = ne0 / ty.blck_size().unwrap() as usize * ty.type_size().unwrap() as usize;
    assert_eq!(
        src.len() % rb,
        0,
        "{ty:?}: {} bytes are not whole rows of {ne0}",
        src.len()
    );
    src.chunks_exact(rb)
        .map(|row| {
            let mut x = vec![0.0f32; ne0];
            dequant_row(ty, row, &mut x).unwrap_or_else(|e| panic!("{ty:?}: {e}"));
            x
        })
        .collect()
}

/// The Q8_0 file `written` against the FP8 128-block `source` it was written from (whole tensors of `ne0` values a
/// row): every block's bytes are the reference rule's over the source's values, and every value reads back within
/// its block's band ([`q8_0_band`]). Returns the largest `|y - x| / band` over the tensor.
pub fn q8_0_within_band(source: &[u8], written: &[u8], ne0: usize) -> f32 {
    let x = rows(GgmlType::FP8_B128, source, ne0);
    let y = written.chunks_exact(ne0 / 32 * 34).collect::<Vec<&[u8]>>();
    assert_eq!(x.len(), y.len(), "row counts");
    let mut worst = 0.0f32;
    for (r, (x, y)) in x.iter().zip(y.iter()).enumerate() {
        for (b, xv) in x.chunks(32).enumerate() {
            let block: &[f32; 32] = xv.try_into().unwrap();
            let want = Q8Block::quantize_ref(block)
                .expect("a finite block with an f16 scale")
                .to_bytes();
            let got: [u8; 34] = y[b * 34..(b + 1) * 34].try_into().unwrap();
            assert_eq!(
                got, want,
                "row {r} block {b}: not the reference rule's bytes"
            );
            let d = half_to_f32(u16::from_le_bytes([got[0], got[1]]));
            let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let band = q8_0_band(amax);
            for (j, (&q, &x)) in got[2..].iter().zip(block).enumerate() {
                let back = d * f32::from(q as i8);
                let err = (back - x).abs();
                assert!(
                    err <= band,
                    "row {r} block {b} value {j}: {back} against {x} is off by {err}, the band {band}"
                );
                worst = worst.max(err / band);
            }
        }
    }
    worst
}

/// The header facts the description reads of `split`: the hparams field by field (`Hparams`' `Debug` spelling), then
/// every tensor's name and dims, sorted (the reader holds names and dims to the keys; a tensor's type is the
/// quantizer's); a text two files are compared by.
pub fn described(split: &Split) -> String {
    let hp = Hparams::read(split).unwrap_or_else(|e| panic!("{e}"));
    let mut lines: Vec<String> = split
        .iter_tensors()
        .map(|(_, t)| format!("{} {:?}", t.name, t.dims))
        .collect();
    lines.sort();
    format!("{hp:?}\n{}", lines.join("\n"))
}

/// A fresh work directory under `E55Q_WORK`, else the temp dir, named for the calling test.
pub fn work_dir(tag: &str) -> PathBuf {
    let base = std::env::var_os("E55Q_WORK").map_or_else(std::env::temp_dir, PathBuf::from);
    let d = base.join(format!("e55q-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
