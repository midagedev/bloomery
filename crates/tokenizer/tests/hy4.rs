//! Hy4-flash's vocabulary: our ids equal the reference's, id for id, on every case of `cases.txt` and `cases-hy4.txt` in both
//! parse modes, on the four oracle prompts (`tools/ref/data/hy4`), and the model's special tokens are the ones its chat template
//! and its header name. The reference is the tree of llama.cpp PR #32517 at 0a054d0b (the `hy_v4` pre-tokenizer is the
//! deepseek-v3 set of regexes); the pins are `hy4-ids.tsv`, written by `tools/hy4-oracle.sh`, whose header names the tree's
//! commit.
//!
//! `hw_`: needs the vocabulary file on the box, named by `BLOOMERY_REF_MODEL` (the first shard of the Hy4-flash GGUF; the box's
//! model profile exports it): `bash tools/gate.sh --release -p bloomery-tokenizer --test hy4 -- --ignored --nocapture`. The
//! vocabulary only: seconds.

#[path = "common/pins.rs"]
mod pins;

use tokenizer::Tokenizer;

/// The vocabulary of the file `BLOOMERY_REF_MODEL` names, held to be Hy4-flash's by its pre-tokenizer name: a path that names
/// another model is a named failure, not a count of differing rows.
fn vocabulary() -> Tokenizer {
    let path = std::env::var_os("BLOOMERY_REF_MODEL")
        .filter(|p| !p.is_empty())
        .expect("BLOOMERY_REF_MODEL names the first shard of the Hy4-flash GGUF; it is unset");
    let tok = Tokenizer::from_gguf(&path)
        .unwrap_or_else(|e| panic!("the vocabulary of {path:?} loads: {e}"));
    assert_eq!(
        tok.pre(),
        "hy_v4",
        "{path:?} is not Hy4-flash's vocabulary (its pre-tokenizer is not hy_v4)"
    );
    tok
}

/// Every case encodes to the reference's ids, with special tokens parsed and without. FAIL-first: with a digit group of two
/// instead of three, or the Han range's upper edge moved by one, the rows of digit runs and of range edges differ; with the
/// UTF-8 reader of the older trees (overlong forms accepted), the row of invalid bytes differs.
#[test]
#[ignore = "gate: the Hy4-flash vocabulary file on the box"]
fn hw_hy4_ids_equal_the_reference() {
    let tok = vocabulary();
    let (rows, differ) = pins::compare(&tok, include_str!("hy4-ids.tsv"));
    println!("hy4: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The long oracle prompt `e` is the ids of `a`, `b` and `c` in a row (no text of its own), so the dump's long set is those three
/// prompts the tokenizer test below pins. FAIL-first: one id dropped from `e.ids` or the three in another order.
#[test]
fn the_long_oracle_prompt_is_the_three_in_a_row() {
    let [a, b, c, e] = [
        include_str!("../../../tools/ref/data/hy4/a.ids"),
        include_str!("../../../tools/ref/data/hy4/b.ids"),
        include_str!("../../../tools/ref/data/hy4/c.ids"),
        include_str!("../../../tools/ref/data/hy4/e.ids"),
    ];
    assert_eq!(e, format!("{a}{b}{c}"));
    assert_eq!(
        e.lines().count(),
        93,
        "more ids than the selector's 64 cells"
    );
}

/// The four oracle prompts (`tools/ref/data/hy4`: the text of each and the reference's ids, one decimal id a line) encode to
/// those ids with no BOS, as the dump harness reads them. FAIL-first: with `add_bos` taken from `tokenizer.ggml.bos_token_id`
/// instead of the header's `add_bos_token` false, every prompt gains the first id.
#[test]
#[ignore = "gate: the Hy4-flash vocabulary file on the box"]
fn hw_hy4_oracle_prompts_encode_to_their_pinned_ids() {
    let tok = vocabulary();
    assert!(!tok.add_bos(), "the header says add_bos_token false");
    for (name, text, ids) in [
        (
            "a",
            include_str!("../../../tools/ref/data/hy4/a.txt"),
            include_str!("../../../tools/ref/data/hy4/a.ids"),
        ),
        (
            "b",
            include_str!("../../../tools/ref/data/hy4/b.txt"),
            include_str!("../../../tools/ref/data/hy4/b.ids"),
        ),
        (
            "c",
            include_str!("../../../tools/ref/data/hy4/c.txt"),
            include_str!("../../../tools/ref/data/hy4/c.ids"),
        ),
        (
            "d",
            include_str!("../../../tools/ref/data/hy4/d.txt"),
            include_str!("../../../tools/ref/data/hy4/d.ids"),
        ),
    ] {
        let want: Vec<u32> = ids
            .lines()
            .map(|l| l.parse().expect("an id a line"))
            .collect();
        assert!(!want.is_empty(), "prompt {name} has ids");
        assert_eq!(
            tok.encode(text.as_bytes(), tok.add_bos(), true),
            want,
            "prompt {name}"
        );
        assert_eq!(
            tok.encode(text.as_bytes(), tok.add_bos(), false),
            want,
            "prompt {name}, no special parsing"
        );
    }
}

/// The special tokens the chat template writes and the serve gate reads back are single ids of the vocabulary, as the header
/// states them: the turn markers (`<｜hy_start…｜>` is the BOS and starts every turn), the end of a turn (`tokenizer.ggml.eos_token_id`,
/// 120025, an end of generation), the reasoning span and the tool tags. The markers are control tokens, one id only where the
/// text's special tokens are parsed; the span and the tool-call tags are user-defined tokens, one id in both modes (as the
/// reference's `--no-parse-special` rows show), so a generation shows them as text and the chat parser reads them from it.
#[test]
#[ignore = "gate: the Hy4-flash vocabulary file on the box"]
fn hw_hy4_special_tokens_are_the_templates() {
    let tok = vocabulary();
    let one = |text: &str, parse_special: bool| {
        let ids = tok.encode(text.as_bytes(), false, parse_special);
        assert_eq!(ids.len(), 1, "{text} is one token: {ids:?}");
        ids[0]
    };
    // (text, id, user-defined: one id whether or not special tokens are parsed)
    for (text, id, user_defined) in [
        ("<｜hy_start:6124c78e｜>", 120_000, false),
        ("<｜hy_middle:6124c78e｜>", 120_001, false),
        ("<｜hy_pad:6124c78e｜>", 120_002, false),
        ("<｜hy_end:6124c78e｜>", 120_025, false),
        ("<think:6124c78e>", 120_029, true),
        ("</think:6124c78e>", 120_030, true),
        ("<tool_call:6124c78e>", 120_033, true),
        ("</tool_call:6124c78e>", 120_034, true),
        ("<tool_calls:6124c78e>", 120_035, true),
        ("</tool_calls:6124c78e>", 120_036, true),
        ("<tool_response:6124c78e>", 120_037, false),
        ("</tool_response:6124c78e>", 120_038, false),
        ("<｜reasoning_mode:6124c78e｜>", 120_039, false),
        ("<arg_key:6124c78e>", 120_040, true),
        ("</arg_key:6124c78e>", 120_041, true),
        ("<arg_value:6124c78e>", 120_042, true),
        ("</arg_value:6124c78e>", 120_043, true),
    ] {
        assert_eq!(one(text, true), id, "{text}");
        if user_defined {
            assert_eq!(one(text, false), id, "{text}, special tokens not parsed");
        } else {
            assert!(
                tok.encode(text.as_bytes(), false, false).len() > 1,
                "{text} is plain text with special tokens not parsed"
            );
        }
    }
    assert!(
        tok.eog().contains(&120_025),
        "<｜hy_end…｜> ends generation: {:?}",
        tok.eog()
    );
    // The same text spelled without the suffix is plain text, never one of these tokens.
    for text in ["<think>", "<tool_call>", "<think:6124c78f>"] {
        assert!(
            tok.encode(text.as_bytes(), false, true).len() > 1,
            "{text} is text"
        );
    }
}
