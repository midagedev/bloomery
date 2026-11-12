use std::borrow::Cow;

use gguf::quant::GgmlType;
use quantize::Result;
use quantize::resolve::{Ftype, Model, Options, TensorRef, layer_of, resolve};
use quantize::rules::Rules;

const M8: Model = Model {
    n_layer_all: 8,
    n_expert: 8,
};
const M64: Model = Model {
    n_layer_all: 8,
    n_expert: 64,
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
    assert!(Rules::parse("x=q3_k").is_ok());
    assert!(run(&names, "blk=q3_k", &M64, &opts(Ftype::Q4KM)).is_err());
    // a layer the counters cannot place
    let bad = [("blk.9.ffn_down.weight", D, B)];
    let e = run(&bad, "", &M64, &opts(Ftype::Q4KM)).unwrap_err();
    assert!(e.to_string().contains("Bad layer 9"), "{e}");
}

#[test]
fn file_types_parse_and_carry_llama_ftype_ids() {
    for (s, id) in [
        ("q4_k_m", 15),
        ("Q5_K_M", 17),
        ("Q6_K", 18),
        ("q8_0", 7),
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

/// [`Model::read`]: each row of the architecture table reads its set's block and expert counts,
/// and an architecture no row names is refused naming the rows. Mutant (red at the refusal): the
/// `exaone-moe` row out of the table.
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
    for arch in ["kimi-k3", "exaone-moe"] {
        let m = Model::read(&set(arch)).unwrap();
        assert_eq!((m.n_layer_all, m.n_expert), (79, 256), "{arch}");
    }
    let e = Model::read(&set("flint")).unwrap_err().to_string();
    assert!(
        e.contains("\"flint\"") && e.contains("kimi-k3") && e.contains("exaone-moe"),
        "{e}"
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
