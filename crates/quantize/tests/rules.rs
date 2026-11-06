use quantize::rules::{Pattern, Rules};

fn hit(re: &str, name: &str) -> bool {
    Pattern::parse(re).unwrap().search(name)
}

#[test]
fn the_k31_rule_shapes_match_as_regex_search_does() {
    let bump = r"blk\.(0|1|12)\.ffn_(gate|up)_exps\.weight";
    assert!(hit(bump, "blk.12.ffn_up_exps.weight"));
    assert!(hit(bump, "blk.0.ffn_gate_exps.weight"));
    assert!(!hit(bump, "blk.2.ffn_gate_exps.weight"));
    assert!(!hit(bump, "blk.1.ffn_down_exps.weight"));
    // search, not match: a prefix does not hide the name
    assert!(hit(
        r"ffn_(down|gate|up)_exps\.weight",
        "mtp.blk.3.ffn_down_exps.weight"
    ));
    assert!(hit(r"\.attn_k_b\.weight$", "blk.1.attn_k_b.weight"));
    assert!(!hit(r"\.attn_k_b\.weight$", "blk.1.attn_k_b.weight.scale"));
    assert!(hit(r"^(token_embd|output)\.weight$", "output.weight"));
    assert!(!hit(
        r"^(token_embd|output)\.weight$",
        "blk.0.output.weight"
    ));
}

#[test]
fn repeats_classes_and_backtracking() {
    assert!(hit(r"blk\.\d+\.attn", "blk.123.attn_q"));
    assert!(!hit(r"blk\.\d+\.attn", "blk..attn_q"));
    assert!(hit(r"blk\.[0-9]*\.x", "blk..x"));
    assert!(hit(r"a.*b", "xaZZb"));
    assert!(hit(r"colou?r", "color"));
    assert!(hit(r"^(a|ab)c$", "abc"));
    assert!(hit(r"[^0-9]x", "ax"));
    assert!(!hit(r"[^0-9]x", "1x"));
    assert!(hit(r"(a*)*b", "aab"));
}

#[test]
fn syntax_it_would_read_differently_is_refused_by_name() {
    for re in [
        r"a{2}", r"(?:a)", r"\bx", r"a*?", r"(a", r"a)", r"*a", r"\1", r"[a", r"[z-a]",
    ] {
        let e = Pattern::parse(re).unwrap_err().to_string();
        assert!(e.contains("rule pattern"), "{re}: {e}");
    }
}

#[test]
fn rules_are_first_match_on_the_lowercased_name() {
    let r = Rules::parse(
        "blk\\.(3)\\.ffn_down_exps\\.weight=iq2_xxs\nffn_(down|gate|up)_exps\\.weight=IQ1_S\n\\.attn_q\\.weight$=q6_k\n",
    )
    .unwrap();
    assert_eq!(r.len(), 3);
    assert_eq!(r.type_of("blk.3.ffn_down_exps.weight"), Some("IQ2_XXS"));
    assert_eq!(r.type_of("blk.4.ffn_down_exps.weight"), Some("IQ1_S"));
    assert_eq!(r.type_of("BLK.4.FFN_UP_EXPS.WEIGHT"), Some("IQ1_S"));
    assert_eq!(r.type_of("blk.0.attn_q.weight"), Some("Q6_K"));
    assert_eq!(r.type_of("token_embd.weight"), None);
}

#[test]
fn a_token_without_a_type_is_refused() {
    for t in ["ffn_exps", "=iq1_s", "x="] {
        assert!(Rules::parse(t).is_err(), "{t}");
    }
}
