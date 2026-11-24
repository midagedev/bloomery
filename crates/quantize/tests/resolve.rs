mod common;

use std::borrow::Cow;
use std::path::Path;

use common::{Rng, SetTensor, f32_bytes, scratch, write_gguf, write_imatrix};
use gguf::Value;
use gguf::quant::GgmlType;
use quantize::Result;
use quantize::cli::{Parsed, parse};
use quantize::resolve::{Ftype, Model, Options, TensorRef, layer_of, resolve};
use quantize::rules::Rules;

const M8: Model = Model {
    arch: "kimi-k3",
    n_layer_all: 8,
    n_expert: 8,
};
const M64: Model = Model {
    arch: "kimi-k3",
    n_layer_all: 8,
    n_expert: 64,
};
/// DeepSeek-V4.1's counts: 43 blocks, so ggml's Q3_K_M `ffn_down` takes Q5_K for blocks 0 and 1 (`43 / 16`).
const M43: Model = Model {
    arch: "deepseek41",
    n_layer_all: 43,
    n_expert: 256,
};

fn opts(ftype: Ftype) -> Options {
    Options {
        ftype,
        output_tensor_type: None,
        token_embedding_type: None,
    }
}

fn run(
    names: &[(&str, &[u64], GgmlType)],
    rules: &str,
    m: &Model,
    o: &Options,
) -> Result<Vec<GgmlType>> {
    let refs: Vec<TensorRef<'_>> = names
        .iter()
        .map(|&(name, dims, ty)| TensorRef { name, dims, ty })
        .collect();
    resolve(&refs, &Rules::parse(rules).unwrap(), m, o)
}

const D: &[u64] = &[512, 256];
const B: GgmlType = GgmlType::BF16;

#[test]
fn the_eight_expert_model_bumps_attention_and_the_early_ffn_down() {
    let t = run(
        &[
            ("token_embd.weight", D, B),
            ("blk.0.attn_k.weight", D, B),
            ("blk.0.attn_v.weight", D, B),
            ("blk.0.attn_q.weight", D, B),
            ("blk.0.attn_output.weight", D, B),
            ("blk.0.ffn_down.weight", D, B),
            ("blk.4.ffn_down.weight", D, B),
            ("blk.0.ffn_up.weight", D, B),
            ("blk.0.attn_norm.weight", &[512], GgmlType::F32),
            ("output.weight", D, B),
        ],
        "",
        &M8,
        &opts(Ftype::Q4KM),
    )
    .unwrap();
    use GgmlType::*;
    // token_embd is tied to nothing here (output.weight exists): the default; attn_k and attn_v go to Q8_0 at
    // eight experts; attn_output to Q5_K; ffn_down of layer 0 (< 8/8) to Q6_K and of layer 4 stays Q4_K.
    assert_eq!(
        t,
        [Q4_K, Q8_0, Q8_0, Q4_K, Q5_K, Q6_K, Q4_K, Q4_K, F32, Q6_K]
    );
}

#[test]
fn flags_and_rules_win_and_a_ruled_tensor_leaves_the_counters_alone() {
    let o = Options {
        ftype: Ftype::Q5KM,
        output_tensor_type: Some(GgmlType::Q6_K),
        token_embedding_type: Some(GgmlType::Q4_K),
    };
    let names = [
        ("token_embd.weight", D, B),
        ("output.weight", D, B),
        ("blk.0.attn_v.weight", D, B),
        ("blk.1.attn_v.weight", D, B),
        ("blk.2.attn_v.weight", D, B),
    ];
    let t = run(&names, "", &M64, &o).unwrap();
    // three attn_v tensors, n_attention_wv = 3: use_more_bits(0, 3) holds (0 < 3/8 is false, 0 >= 21/8=2 false,
    // (0-0)%3 == 2 false): none of the first, so Q5_K for all but where (i-0)%3 == 2: the third is Q6_K.
    assert_eq!(
        t,
        [
            GgmlType::Q4_K,
            GgmlType::Q6_K,
            GgmlType::Q5_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K
        ]
    );
    // a rule on the first attn_v leaves i_attention_wv at 0 for the next: it is now the first the logic types
    let t = run(&names, "blk\\.0\\.attn_v\\.weight=q8_0", &M64, &o).unwrap();
    assert_eq!(t[2], GgmlType::Q8_0);
    assert_eq!(t[3], GgmlType::Q5_K);
    assert_eq!(
        t[4],
        GgmlType::Q5_K,
        "the counter did not move for the ruled tensor"
    );
}

#[test]
fn what_the_port_does_not_write_is_refused_by_name() {
    // IQ1_S's first eighth of ffn_down is Q2_K unless a rule names it
    let names = [("blk.0.ffn_down.weight", D, B)];
    let e = run(&names, "", &M64, &opts(Ftype::Iq1S)).unwrap_err();
    assert!(
        e.to_string().contains("blk.0.ffn_down.weight") && e.to_string().contains("Q2_K"),
        "{e}"
    );
    assert_eq!(
        run(&names, "ffn_down=iq1_s", &M64, &opts(Ftype::Iq1S)).unwrap(),
        [GgmlType::IQ1_S]
    );
    // a row that is not a whole number of blocks
    let odd = [("blk.0.attn_q.weight", &[300u64, 4][..], B)];
    let e = run(&odd, "", &M64, &opts(Ftype::Q4KM)).unwrap_err();
    assert!(e.to_string().contains("300"), "{e}");
    // a rule type the port does not write
    assert!(Rules::parse("x=q2_k").is_ok());
    assert!(run(&names, "blk=q2_k", &M64, &opts(Ftype::Q4KM)).is_err());
    // a layer the counters cannot place
    let bad = [("blk.9.ffn_down.weight", D, B)];
    let e = run(&bad, "", &M64, &opts(Ftype::Q4KM)).unwrap_err();
    assert!(e.to_string().contains("Bad layer 9"), "{e}");
}

#[test]
fn a_rule_naming_q3_k_types_the_stacks_and_a_row_off_its_blocks_is_refused() {
    let stack: &[u64] = &[512, 4, 8];
    let names = [
        ("blk.0.ffn_gate_exps.weight", stack, B),
        ("blk.0.ffn_up_exps.weight", stack, B),
        ("blk.0.ffn_down_exps.weight", stack, B),
    ];
    let t = run(
        &names,
        "ffn_(gate|up)_exps=q3_k ffn_down_exps=q4_k",
        &M64,
        &opts(Ftype::Q4KM),
    )
    .unwrap();
    assert_eq!(t, [GgmlType::Q3_K, GgmlType::Q3_K, GgmlType::Q4_K]);
    let odd = [("blk.0.ffn_gate_exps.weight", &[300u64, 4, 8][..], B)];
    let e = run(&odd, "ffn_gate_exps=q3_k", &M64, &opts(Ftype::Q4KM)).unwrap_err();
    assert!(
        e.to_string().contains("300") && e.to_string().contains("Q3_K"),
        "{e}"
    );
}

#[test]
fn file_types_parse_and_carry_llama_ftype_ids() {
    for (s, id) in [
        ("q4_k_m", 15),
        ("Q5_K_M", 17),
        ("Q6_K", 18),
        ("q8_0", 7),
        ("q3_k_m", 12),
        ("iq2_xxs", 19),
        ("IQ1_S", 24),
    ] {
        assert_eq!(Ftype::parse(s).unwrap().id(), id, "{s}");
    }
    assert!(Ftype::parse("Q2_K").is_err());
}

#[test]
fn layers_are_read_from_the_name_as_sscanf_does() {
    assert_eq!(layer_of("blk.12.ffn_down.weight"), Some(12));
    assert_eq!(layer_of("blk.x.ffn_down.weight"), None);
    assert_eq!(layer_of("token_embd.weight"), None);
    assert_eq!(layer_of("blk.7weight"), None);
}

/// A set of one `general.architecture` and its block and expert counts.
struct Set(Vec<(String, gguf::Value)>);

impl quantize::source::Source for Set {
    fn label(&self) -> String {
        "set".into()
    }
    fn keys(&self) -> &[(String, gguf::Value)] {
        &self.0
    }
    fn shard_count(&self) -> usize {
        1
    }
    fn tensors(&self) -> Vec<(usize, gguf::TensorInfo)> {
        vec![]
    }
    fn read(&self, _name: &str, _bytes: std::ops::Range<u64>) -> quantize::Result<Cow<'_, [u8]>> {
        Err(quantize::Error::Refused(
            "no tensor data in this test".into(),
        ))
    }
}

/// [`Model::read`]: each row of the architecture table reads its set's block and expert counts
/// under its own word, and an architecture no row names is refused naming the rows. Mutant (red
/// at the refusal): the `exaone-moe` row out of the table.
#[test]
fn the_architecture_rows_read_their_sets_and_refuse_another() {
    let set = |arch: &str| {
        Set(vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                gguf::Value::String(arch.into()),
            ),
            (format!("{arch}.block_count"), gguf::Value::U32(79)),
            (format!("{arch}.expert_count"), gguf::Value::U32(256)),
        ])
    };
    for arch in ["kimi-k3", "exaone-moe", "deepseek41", "ernie5_5"] {
        let m = Model::read(&set(arch)).unwrap();
        assert_eq!(
            (m.arch, m.n_layer_all, m.n_expert),
            (arch, 79, 256),
            "{arch}"
        );
    }
    let e = Model::read(&set("flint")).unwrap_err().to_string();
    assert!(
        e.contains("\"flint\"")
            && e.contains("kimi-k3")
            && e.contains("exaone-moe")
            && e.contains("deepseek41")
            && e.contains("ernie5_5"),
        "{e}"
    );
}

/// An ernie5_5 set under Q8_0, no rules: the FP8 128-block projections take the file type (the
/// indexer's query projection with them), and the rows keyed to the architecture keep every
/// tensor the reader holds in one stored type as the folder load makes it: the embedding, the
/// head, the two routers and the compressor's matrices BF16, the routed stacks MXFP4, the mixers,
/// the head mix, the position table and the indexer's matrices as converted. Mutant (red at the
/// `blk.1.ffn_gate_inp_1.weight` row): a fragment out of the row, or `indexer.` whole.
#[test]
fn an_ernie5_5_set_keeps_its_stored_types_and_takes_the_file_type_for_its_fp8_projections() {
    use GgmlType::*;
    let m = Model {
        arch: "ernie5_5",
        n_layer_all: 3,
        n_expert: 512,
    };
    let f8 = &[2048u64, 64][..];
    let mx = &[2048u64, 2048, 512][..];
    let names = [
        ("token_embd.weight", D, B),
        ("output.weight", D, B),
        ("output_hc_fn.weight", D, F32),
        ("output_hc_base.weight", &[4][..], F32),
        ("output_multimax_ranges.weight", &[4][..], F32),
        ("blk.0.attn_norm.weight", &[4096][..], F32),
        ("blk.0.hc_attn_fn.weight", D, F32),
        ("blk.0.hc_attn_scale.weight", &[3][..], F32),
        ("blk.0.attn_vha_u.weight", D, F32),
        ("blk.0.attn_compressor_ape.weight", D, F32),
        ("blk.0.attn_compressor_kv.weight", D, B),
        ("blk.0.attn_compressor_gate.weight", D, B),
        ("blk.0.attn_q_b.weight", f8, FP8_B128),
        ("blk.0.attn_output_a.weight", f8, FP8_B128),
        ("blk.0.ffn_gate.weight", f8, FP8_B128),
        ("blk.1.ffn_gate_inp.weight", D, B),
        ("blk.1.ffn_gate_inp_1.weight", D, B),
        ("blk.1.exp_probs_b.bias", &[512][..], F32),
        ("blk.1.ffn_gate_exps.weight", mx, MXFP4),
        ("blk.1.ffn_up_exps.weight", mx, MXFP4),
        ("blk.1.ffn_down_exps.weight", mx, MXFP4),
        ("blk.1.ffn_routed_down.weight", f8, FP8_B128),
        ("blk.1.ffn_gate_shexp.weight", f8, FP8_B128),
        ("blk.2.indexer.attn_q_b.weight", f8, FP8_B128),
        ("blk.2.indexer.attn_k.weight", D, B),
        ("blk.2.indexer.proj.weight", D, B),
        ("blk.2.indexer.k_norm.weight", &[128][..], F32),
        ("blk.2.indexer.k_norm.bias", &[128][..], F32),
        ("blk.2.attn_kv_b.weight", f8, FP8_B128),
    ];
    assert_eq!(
        run(&names, "", &m, &opts(Ftype::Q8_0)).unwrap(),
        [
            B, B, F32, F32, F32, F32, F32, F32, F32, F32, B, B, Q8_0, Q8_0, Q8_0, B, B, F32, MXFP4,
            MXFP4, MXFP4, Q8_0, Q8_0, Q8_0, B, B, F32, F32, Q8_0
        ]
    );
}

/// A flag or a rule may name a type the quantizer copies: a tensor already in it is typed as it
/// is, and a type no row of the table has is refused naming what the quantizer writes and copies.
#[test]
fn a_copied_type_names_a_tensor_already_in_it() {
    use GgmlType::*;
    for (s, ty) in [
        ("f16", F16),
        ("bf16", B),
        ("mxfp4", MXFP4),
        ("FP8_B128", FP8_B128),
    ] {
        assert_eq!(quantize::resolve::parse_type(s).unwrap(), ty, "{s}");
    }
    let e = quantize::resolve::parse_type("q2_k")
        .unwrap_err()
        .to_string();
    assert!(e.contains("copies F16, BF16, MXFP4 and FP8_B128"), "{e}");
    // A rule naming a stored type keeps the tensor that is in it: the FP8 projection stays FP8_B128
    // where the file type alone would write it Q8_0 (the plan refuses the rule for a tensor of
    // another type: no writer converts to a copied type).
    let f8 = ("blk.0.attn_q_b.weight", &[2048u64, 64][..], FP8_B128);
    let bf = ("blk.0.attn_kv.weight", &[2048u64, 64][..], B);
    let m = Model {
        arch: "ernie5_5",
        n_layer_all: 3,
        n_expert: 512,
    };
    assert_eq!(
        run(
            &[f8, bf],
            "attn_q_b=fp8_b128 attn_kv=bf16",
            &m,
            &opts(Ftype::Q8_0)
        )
        .unwrap(),
        [FP8_B128, B]
    );
}

/// A K-EXAONE set at the file's counts types as llama-quantize's driver types it under Q4_K_M:
/// the untied `output` at Q6_K and `token_embd` at the default; `attn_v` by its counter's
/// `use_more_bits` (the third of three at Q6_K), the routed and shared `ffn_down` by the layer
/// their names spell (`blk.62` — (62 − 9) % 3 == 2 — at Q6_K, `blk.10` at the default), the rest
/// of the FFN and attention at the default; and `tensor_allows_quantization` keeps the router's
/// tables F32: `exp_probs_b.bias` fails the `weight` ending, the gate and the norms the norm and
/// gate rules, while the next-token block (`blk.78`, its `nextn.eh_proj` at the default — the
/// Q8_0 floor of that name is glm5-next's alone) types like any block. Mutant (red at the
/// `blk.10` row): the layer read from the counter where the name's layer belongs.
#[test]
fn an_exaone_moe_set_types_as_llama_quantize_types_it() {
    use GgmlType::*;
    let m = Model {
        arch: "exaone-moe",
        n_layer_all: 79,
        n_expert: 256,
    };
    let names = [
        ("token_embd.weight", D, B),
        ("output.weight", D, B),
        ("blk.2.attn_v.weight", D, B),
        ("blk.70.attn_v.weight", D, B),
        ("blk.78.attn_v.weight", D, B),
        ("blk.2.attn_k.weight", D, B),
        ("blk.2.attn_q.weight", D, B),
        ("blk.2.attn_output.weight", D, B),
        ("blk.62.ffn_down_exps.weight", &[256, 512, 256][..], B),
        ("blk.10.ffn_down_exps.weight", &[256, 512, 256][..], B),
        ("blk.10.ffn_up_exps.weight", &[256, 512, 256][..], B),
        ("blk.10.ffn_gate_exps.weight", &[256, 512, 256][..], B),
        ("blk.10.ffn_down_shexp.weight", D, B),
        ("blk.0.ffn_gate.weight", D, B),
        ("blk.78.nextn.eh_proj.weight", D, B),
        ("blk.2.ffn_gate_inp.weight", D, F32),
        ("blk.2.exp_probs_b.bias", &[256][..], F32),
        ("blk.78.nextn.enorm.weight", &[256][..], F32),
        ("blk.2.attn_norm.weight", &[256][..], F32),
    ];
    assert_eq!(
        run(&names, "", &m, &opts(Ftype::Q4KM)).unwrap(),
        [
            Q4_K, Q6_K, Q4_K, Q4_K, Q6_K, Q4_K, Q4_K, Q4_K, Q6_K, Q4_K, Q4_K, Q4_K, Q4_K, Q4_K,
            Q4_K, F32, F32, F32, F32
        ]
    );
    // An IQ type's attn_v at four experts or more takes the Q4_K branch the port carries.
    assert_eq!(
        run(
            &[("blk.2.attn_v.weight", D, B), ("output.weight", D, B)],
            "",
            &m,
            &opts(Ftype::Iq2Xxs)
        )
        .unwrap(),
        [Q4_K, Q5_K]
    );
}

/// ggml's Q3_K_M (`llama_tensor_get_type_impl`) over a non-eight-expert set: the first two `attn_v`-like tensors
/// by counter Q5_K and the rest Q4_K, `attn_output` Q4_K, `ffn_down` (routed and shared) Q5_K for the first
/// `n_layer / 16` blocks by the block the name spells and Q4_K after, `output` Q6_K and every other quantizable
/// tensor the default Q3_K. At eight experts `attn_v` is Q8_0 and `attn_output` Q5_K. Mutant (red at the
/// `blk.1` `attn_v`): the Q5_K bump read as `< 1` where ggml has `< 2`.
#[test]
fn the_q3_k_m_word_types_as_llama_quantize_types_it() {
    use GgmlType::*;
    let names = [
        ("token_embd.weight", D, B),
        ("output.weight", D, B),
        ("blk.0.attn_v.weight", D, B),
        ("blk.1.attn_v.weight", D, B),
        ("blk.2.attn_kv_b.weight", D, B),
        ("blk.3.attn_v.weight", D, B),
        ("blk.0.attn_q.weight", D, B),
        ("blk.0.attn_output.weight", D, B),
        ("blk.0.ffn_down_exps.weight", &[256, 512, 64][..], B),
        ("blk.1.ffn_down_shexp.weight", D, B),
        ("blk.2.ffn_down_exps.weight", &[256, 512, 64][..], B),
        ("blk.42.ffn_down_shexp.weight", D, B),
        ("blk.0.ffn_gate_exps.weight", &[256, 512, 64][..], B),
        ("blk.0.ffn_up_shexp.weight", D, B),
        ("blk.0.ffn_gate_inp.weight", &[256, 64][..], F32),
        ("blk.0.attn_norm.weight", &[256][..], F32),
    ];
    let m = Model {
        arch: "deepseek41",
        n_layer_all: 43,
        n_expert: 64,
    };
    assert_eq!(
        run(&names, "", &m, &opts(Ftype::Q3KM)).unwrap(),
        [
            Q3_K, Q6_K, Q5_K, Q5_K, Q4_K, Q4_K, Q3_K, Q4_K, Q5_K, Q5_K, Q4_K, Q4_K, Q3_K, Q3_K,
            F32, F32
        ]
    );
    let eight = Model {
        arch: "deepseek41",
        n_layer_all: 43,
        n_expert: 8,
    };
    let t = run(&names[..8], "", &eight, &opts(Ftype::Q3KM)).unwrap();
    assert_eq!((t[2], t[3], t[7]), (Q8_0, Q8_0, Q5_K));
}

/// A DeepSeek-V4.1 set under Q3_K_M with no rule types as the `Q3_K_M` file does: the routed and shared `ffn_down`
/// of blocks 0 and 1 Q5_K and the other blocks' Q4_K, everything else quantizable Q3_K (the grouped
/// `attn_output_a` and `attn_output_b` included: neither is ggml's `attn_output` category), `output` Q6_K, and the
/// router, the engram gate and the two engram gain vectors keep their types.
#[test]
fn a_deepseek41_set_under_q3_k_m_types_as_the_q3_k_m_file_does() {
    use GgmlType::*;
    let routed: &[u64] = &[256, 512, 8];
    let names = [
        ("token_embd.weight", D, B),
        ("output.weight", D, B),
        ("per_layer_token_embd.weight", D, B),
        ("blk.0.attn_q_a.weight", D, B),
        ("blk.0.attn_q_b.weight", D, B),
        ("blk.0.attn_kv.weight", D, B),
        ("blk.0.attn_output_a.weight", D, B),
        ("blk.0.attn_output_b.weight", D, B),
        ("blk.0.ffn_gate_exps.weight", routed, B),
        ("blk.0.ffn_up_exps.weight", routed, B),
        ("blk.0.ffn_down_exps.weight", routed, B),
        ("blk.0.ffn_down_shexp.weight", D, B),
        ("blk.0.hc_attn_fn.weight", D, F32),
        ("blk.0.ffn_gate_inp.weight", &[256, 8][..], F32),
        ("blk.1.ffn_down_exps.weight", routed, B),
        ("blk.2.ffn_down_exps.weight", routed, B),
        ("blk.2.ffn_down_shexp.weight", D, B),
        ("blk.1.engram_kv.weight", D, B),
        ("blk.1.engram_gate.weight", D, F32),
        ("blk.1.engram_q.weight", D, B),
        ("blk.1.engram_k.weight", D, B),
        ("blk.42.ffn_down_exps.weight", routed, B),
    ];
    assert_eq!(
        run(&names, "", &M43, &opts(Ftype::Q3KM)).unwrap(),
        [
            Q3_K, Q6_K, Q3_K, Q3_K, Q3_K, Q3_K, Q3_K, Q3_K, Q3_K, Q3_K, Q5_K, Q5_K, Q3_K, F32,
            Q5_K, Q4_K, Q4_K, Q3_K, F32, B, B, Q4_K
        ]
    );
}

/// The DeepSeek-V4.1 header facts `Plan::build` reads, over synthetic files and a host run.
mod grouped {
    use super::*;

    const NE0: usize = 256;
    const GROUPS: usize = 4;
    const ROWS: usize = 3;
    /// The in_sum2 over the counts the entries below carry.
    const COUNT: f32 = 2.0;

    fn f32_tensor(name: &str, dims: &[u64], v: &[f32]) -> SetTensor {
        (name.to_string(), dims.to_vec(), GgmlType::F32, f32_bytes(v))
    }

    /// A DeepSeek-V4.1-headed file of `tensors` with `deepseek41.attention.output_group_count` `groups`, when set.
    fn v41_file(path: &Path, groups: Option<u32>, tensors: &[SetTensor]) {
        let mut kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String("deepseek41".to_string()),
            ),
            ("deepseek41.block_count".to_string(), Value::U32(43)),
            ("deepseek41.expert_count".to_string(), Value::U32(256)),
        ];
        if let Some(g) = groups {
            kvs.push((
                "deepseek41.attention.output_group_count".to_string(),
                Value::U32(g),
            ));
        }
        write_gguf(path, &kvs, tensors);
    }

    /// `Q3_K_M` over `file` on the host with `imatrix`: every output tensor's bytes, or the plan's refusal.
    fn quantize(
        work: &Path,
        tag: &str,
        file: &Path,
        imatrix: Option<&Path>,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let out = work.join(format!("{tag}.out.gguf"));
        let mut argv = vec!["--cpu".to_string()];
        if let Some(im) = imatrix {
            argv.extend(["--imatrix".to_string(), im.display().to_string()]);
        }
        argv.extend([
            file.display().to_string(),
            out.display().to_string(),
            "Q3_K_M".to_string(),
        ]);
        let Parsed::Run(a) = parse(argv).unwrap() else {
            panic!("a run");
        };
        let prep = quantize::run::prepare(&a)?;
        quantize::run(
            &a,
            prep,
            &mut quantize::host::HostBackend::new(),
            &mut |_| {},
        )?;
        let g = gguf::Gguf::open(&out).unwrap();
        Ok(g.iter_tensors()
            .map(|t| (t.name.clone(), g.data(t).unwrap().to_vec()))
            .collect())
    }

    fn bytes_of<'a>(out: &'a [(String, Vec<u8>)], name: &str) -> &'a [u8] {
        &out.iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no {name} in the output"))
            .1
    }

    /// The grouped `blk.0.attn_output_a.weight` of `GROUPS * ROWS` rows over `imp` (`GROUPS` vectors of `NE0`), in a
    /// file of its own, its imatrix next to it.
    fn grouped_case(
        work: &Path,
        tag: &str,
        w: &[f32],
        imp: &[f32],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let file = work.join(format!("{tag}.gguf"));
        let im = work.join(format!("{tag}.imatrix.gguf"));
        v41_file(
            &file,
            Some(GROUPS as u32),
            &[f32_tensor(
                "blk.0.attn_output_a.weight",
                &[NE0 as u64, (GROUPS * ROWS) as u64],
                w,
            )],
        );
        write_imatrix(
            &im,
            &[(
                "blk.0.attn_output_a.weight",
                &imp.iter().map(|v| v * COUNT).collect::<Vec<_>>(),
                &[COUNT],
            )],
        );
        (file, im)
    }

    /// A block-diagonal `attn_output_a` (`deepseek41.attention.output_group_count` groups of an equal share of the
    /// rows, each reading its own `ne0`-value slice of the importance) is quantized group by group: its bytes equal
    /// the same rows as separate 2-D tensors, each with the group's slice as its entry, and an entry of one `ne0`
    /// vector equals the plain 2-D tensor with it (llama-quantize's rule), which the per-group bytes are not.
    /// Mutant (red at the first assert): slice 0 of the entry for every group.
    #[test]
    fn a_grouped_tensor_takes_its_groups_slice_of_the_importance() {
        let work = scratch("groups");
        let mut rng = Rng(0x5eed_0001);
        let w: Vec<f32> = (0..NE0 * GROUPS * ROWS).map(|_| rng.sym()).collect();
        let imp: Vec<f32> = (0..NE0 * GROUPS).map(|_| 0.05 + 4.0 * rng.unit()).collect();
        let rows_of = |g: usize| &w[g * ROWS * NE0..(g + 1) * ROWS * NE0];

        // the reference: one 2-D tensor a group, each with its slice as its whole entry
        let split = work.join("split.gguf");
        let split_im = work.join("split.imatrix.gguf");
        let names: Vec<String> = (0..GROUPS)
            .map(|g| format!("blk.{g}.attn_q_a.weight"))
            .collect();
        let tensors: Vec<SetTensor> = (0..GROUPS)
            .map(|g| f32_tensor(&names[g], &[NE0 as u64, ROWS as u64], rows_of(g)))
            .collect();
        v41_file(&split, Some(GROUPS as u32), &tensors);
        let sums: Vec<Vec<f32>> = (0..GROUPS)
            .map(|g| {
                imp[g * NE0..(g + 1) * NE0]
                    .iter()
                    .map(|v| v * COUNT)
                    .collect()
            })
            .collect();
        let entries: Vec<(&str, &[f32], &[f32])> = (0..GROUPS)
            .map(|g| (names[g].as_str(), sums[g].as_slice(), &[COUNT][..]))
            .collect();
        write_imatrix(&split_im, &entries);
        let separate = quantize(&work, "split", &split, Some(&split_im)).unwrap();
        let want: Vec<u8> = names
            .iter()
            .flat_map(|n| bytes_of(&separate, n).to_vec())
            .collect();

        let (file, im) = grouped_case(&work, "grouped", &w, &imp);
        let grouped = quantize(&work, "grouped", &file, Some(&im)).unwrap();
        assert!(
            bytes_of(&grouped, "blk.0.attn_output_a.weight") == want.as_slice(),
            "the grouped tensor equals its groups quantized apart"
        );

        // one vector for every group: the plain 2-D tensor with that vector as its entry
        let plain = work.join("plain.gguf");
        let plain_im = work.join("plain.imatrix.gguf");
        v41_file(
            &plain,
            Some(GROUPS as u32),
            &[f32_tensor(
                "blk.0.attn_q_a.weight",
                &[NE0 as u64, (GROUPS * ROWS) as u64],
                &w,
            )],
        );
        write_imatrix(&plain_im, &[("blk.0.attn_q_a.weight", &sums[0], &[COUNT])]);
        let one = quantize(&work, "plain", &plain, Some(&plain_im)).unwrap();
        let (file1, im1) = grouped_case(&work, "shared", &w, &imp[..NE0]);
        let shared = quantize(&work, "shared", &file1, Some(&im1)).unwrap();
        assert!(
            bytes_of(&shared, "blk.0.attn_output_a.weight")
                == bytes_of(&one, "blk.0.attn_q_a.weight"),
            "an entry of ne0 values serves every group, as llama-quantize's 2-D rule does"
        );
        assert!(
            bytes_of(&shared, "blk.0.attn_output_a.weight") != want.as_slice(),
            "the per-group entry is not decoration: one vector for all groups quantizes differently"
        );
        let _ = std::fs::remove_dir_all(&work);
    }

    /// The refusals, each by name: an entry of neither `ne0` nor `ne0 x groups` values, a stacked tensor's entry of
    /// one vector (llama-quantize refuses it), the header lacking the group count, rows the groups do not divide
    /// equally, and a block-diagonal tensor of more than two dimensions.
    #[test]
    fn a_grouped_tensor_that_cannot_be_split_is_refused_by_name() {
        let work = scratch("groups-refused");
        let mut rng = Rng(0x5eed_0002);
        let mut values = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.sym()).collect() };
        let dims = [NE0 as u64, (GROUPS * ROWS) as u64];
        let name = "blk.0.attn_output_a.weight";

        let file = work.join("a.gguf");
        v41_file(
            &file,
            Some(GROUPS as u32),
            &[f32_tensor(name, &dims, &values(NE0 * GROUPS * ROWS))],
        );
        for n in [2 * NE0, NE0 * GROUPS + NE0] {
            let im = work.join(format!("a{n}.imatrix.gguf"));
            write_imatrix(&im, &[(name, &vec![1.0; n], &[COUNT])]);
            let e = quantize(&work, "a", &file, Some(&im))
                .unwrap_err()
                .to_string();
            assert!(
                e.contains(name)
                    && e.contains(&format!("{n} values"))
                    && e.contains(&format!("{NE0} or {}", NE0 * GROUPS))
                    && e.contains(&format!("{GROUPS} row groups")),
                "{e}"
            );
        }

        let stacked = work.join("stacked.gguf");
        v41_file(
            &stacked,
            Some(GROUPS as u32),
            &[f32_tensor(
                "blk.0.attn_q_a.weight",
                &[NE0 as u64, ROWS as u64, 2],
                &values(NE0 * ROWS * 2),
            )],
        );
        let im = work.join("stacked.imatrix.gguf");
        write_imatrix(&im, &[("blk.0.attn_q_a.weight", &vec![1.0; NE0], &[COUNT])]);
        let e = quantize(&work, "stacked", &stacked, Some(&im))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("blk.0.attn_q_a.weight") && e.contains("2 experts") && !e.contains(" or "),
            "{e}"
        );

        let no_key = work.join("nokey.gguf");
        v41_file(
            &no_key,
            None,
            &[f32_tensor(name, &dims, &values(NE0 * GROUPS * ROWS))],
        );
        let e = quantize(&work, "nokey", &no_key, None)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(name) && e.contains("deepseek41.attention.output_group_count"),
            "{e}"
        );

        let odd = work.join("odd.gguf");
        v41_file(
            &odd,
            Some(5),
            &[f32_tensor(name, &dims, &values(NE0 * GROUPS * ROWS))],
        );
        let e = quantize(&work, "odd", &odd, None).unwrap_err().to_string();
        assert!(
            e.contains(name) && e.contains("12 rows do not divide into 5 equal groups"),
            "{e}"
        );

        let deep = work.join("deep.gguf");
        v41_file(
            &deep,
            Some(GROUPS as u32),
            &[f32_tensor(
                name,
                &[NE0 as u64, GROUPS as u64, 2],
                &values(NE0 * GROUPS * 2),
            )],
        );
        let e = quantize(&work, "deep", &deep, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains(name) && e.contains("2-D"), "{e}");
        let _ = std::fs::remove_dir_all(&work);
    }
}
