use super::*;

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
