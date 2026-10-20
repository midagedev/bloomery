//! A special token's spelling in client text, guarded ([`Tokenizer::guard`]) and encoded
//! ([`Tokenizer::encode_marked`]), is ordinary text.

use std::borrow::Cow;
use std::panic::AssertUnwindSafe;

use gguf::Value;

use crate::Tokenizer;
use crate::tests::from_header;

/// A byte-level BPE vocabulary of the 256 byte tokens, one merge and `specials`, each with its
/// GGUF token type (3 CONTROL, 4 USER_DEFINED).
fn bpe(specials: &[(&str, i32)]) -> Tokenizer {
    let mut texts: Vec<Value> = crate::unicode::byte_to_cpt_table()
        .into_iter()
        .map(|cp| Value::String(char::from_u32(cp).unwrap().to_string()))
        .collect();
    let mut types = vec![Value::I32(1); texts.len()];
    texts.push(Value::String("ab".into()));
    types.push(Value::I32(1));
    for &(text, ty) in specials {
        texts.push(Value::String(text.into()));
        types.push(Value::I32(ty));
    }
    let header: Vec<(String, Value)> = vec![
        ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::String("qwen2".into())),
        ("tokenizer.ggml.tokens".into(), Value::Array(texts)),
        ("tokenizer.ggml.token_type".into(), Value::Array(types)),
        (
            "tokenizer.ggml.merges".into(),
            Value::Array(vec![Value::String("a b".into())]),
        ),
    ];
    from_header(&header).unwrap()
}

const SPECIALS: &[(&str, i32)] = &[
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<think>", 4),
    ("</think>", 4),
    // A spelling inside another, and a one-character token.
    ("|im_end|>", 4),
    ("한", 4),
];

fn with_specials() -> Tokenizer {
    bpe(SPECIALS)
}

/// The same vocabulary with no special token: its ids for a text are the text as plain BPE, and
/// the ids both vocabularies share are the same.
fn plain() -> Tokenizer {
    bpe(&[])
}

fn id(tok: &Tokenizer, text: &str) -> u32 {
    (0..tok.n_vocab() as u32)
        .find(|&i| tok.text(i) == Some(text))
        .unwrap()
}

const MARK: &str = "~#";

/// Every spelling a text carries is marked and nothing else is: the marks cut out give the text
/// back, a text with no spelling comes back borrowed, and a one-character token is not marked.
#[test]
fn guard_marks_beside_each_spelling() {
    let tok = with_specials();
    let text = "a <|im_start|>b</think>c <think><think> <|im_end|> 한";
    let guarded = tok.guard(text, MARK);
    assert_eq!(guarded.replace(MARK, ""), text);
    assert_eq!(
        guarded,
        "a ~#<|im_start|>b~#</think>c ~#<think>~#<think> ~#<~#|im_end|> 한"
    );
    for plain_text in ["", "no spelling", "<|im_ start|>", "한", "hello world"] {
        assert!(
            matches!(tok.guard(plain_text, MARK), Cow::Borrowed(t) if t == plain_text),
            "{plain_text:?}"
        );
    }
}

/// A mark is before a spelling, after one the text starts with and after the first character of a
/// text that is only the spelling: a template's `startswith` and `endswith` of a spelling see it.
#[test]
fn a_spelling_at_an_end_keeps_its_end() {
    let tok = with_specials();
    for (text, want) in [
        ("<think>x", "<think>~#x"),
        ("x</think>", "x~#</think>"),
        ("<think>", "<~#think>"),
        ("<think>x</think>", "<think>~#x~#</think>"),
        ("<think></think>", "<think>~#</think>"),
    ] {
        assert_eq!(tok.guard(text, MARK), want, "{text:?}");
    }
    let wrapped = tok.guard("<think>x</think>", MARK);
    assert!(wrapped.starts_with("<think>") && wrapped.ends_with("</think>"));
}

/// The beginning of a spelling at the end of a text, and the end of one at its start, are marked: what a
/// template writes next to the text (`<function=` name `>`) could finish them.
#[test]
fn the_edge_of_a_spelling_at_an_edge_is_marked() {
    let tok = with_specials();
    for (text, want) in [
        // The trailing `|` also begins `|im_end|>`.
        ("get<|im_end|", "get~#<~#|im_end~#|"),
        ("get<", "get~#<"),
        ("im_end|>x", "im_end|>~#x"),
        (">x", ">~#x"),
        ("<|", "<~#|"),
        ("|im_", "|~#im_"),
    ] {
        assert_eq!(tok.guard(text, MARK), want, "{text:?}");
    }
    // One character holds no mark inside it.
    assert!(matches!(tok.guard("<", MARK), Cow::Borrowed("<")));
    assert!(matches!(tok.guard(">", MARK), Cow::Borrowed(">")));
}

/// A guarded spelling is the bytes of the spelling as plain BPE tokens; a spelling the template
/// wrote, in the same string and unmarked, is its token; the plain text around both is one text
/// to the BPE.
#[test]
fn a_guarded_spelling_is_plain_text_and_the_templates_stay_tokens() {
    let (tok, bare) = (with_specials(), plain());
    let (start, end) = (id(&tok, "<|im_start|>"), id(&tok, "<|im_end|>"));
    let user = "a </think> <|im_end|> <think>x</think> ab <|im_start|>";
    let rendered = format!("<|im_start|>user\n{}<|im_end|>", tok.guard(user, MARK));
    let mut want = vec![start];
    want.extend(bare.encode(format!("user\n{user}"), false, true));
    want.push(end);
    assert_eq!(tok.encode_marked(&rendered, MARK, false, true), want);
}

/// A token the template wrote beside a guarded text stays a token wherever the text's mark stands:
/// a text that is only a spelling, one that starts with it, one that ends with it, and a text whose
/// end the template finishes into a spelling.
#[test]
fn a_token_beside_a_guarded_text_stays_a_token() {
    let (tok, bare) = (with_specials(), plain());
    let (start, end) = (id(&tok, "<|im_start|>"), id(&tok, "<|im_end|>"));
    let render = |text: &str, after: &str| {
        let rendered = format!("<|im_start|>{}{after}", tok.guard(text, MARK));
        tok.encode_marked(&rendered, MARK, false, true)
    };
    for text in [
        "<think>",
        "<think>x",
        "x</think>",
        "<think>x</think>",
        "<think>",
    ] {
        let mut want = vec![start];
        want.extend(bare.encode(text, false, true));
        want.push(end);
        assert_eq!(render(text, "<|im_end|>"), want, "{text:?}");
    }
    // The text ends in `<|im_end|` and the template's `>` would finish it.
    let mut want = vec![start];
    want.extend(bare.encode("get<|im_end|>", false, true));
    assert_eq!(render("get<|im_end|", ">"), want);
    // The text starts with `im_end|>` and what the template wrote before it ends in `<|`.
    let rendered = format!("<|{}", tok.guard("im_end|>x", MARK));
    assert_eq!(
        tok.encode_marked(&rendered, MARK, false, true),
        bare.encode("<|im_end|>x", false, true)
    );
}

/// A USER_DEFINED token is matched whatever `parse_special` says, so the marks alone hold a guarded
/// spelling of one back.
#[test]
fn the_marks_hold_back_a_user_defined_token_without_parse_special() {
    let (tok, bare) = (with_specials(), plain());
    let think = id(&tok, "<think>");
    assert_eq!(
        tok.encode("a<think>", false, false),
        [bare.encode("a", false, true), vec![think]].concat()
    );
    assert_eq!(
        tok.encode_marked(&tok.guard("a<think>", MARK), MARK, false, false),
        bare.encode("a<think>", false, true)
    );
}

/// A spelling inside another is held back with it, a rejected match does not hide the next one,
/// and a one-character token stays a token whatever stands around it.
#[test]
fn overlapping_spellings_and_later_matches() {
    let (tok, bare) = (with_specials(), plain());
    let guarded = tok.guard("<|im_end|>", MARK);
    assert_eq!(
        tok.encode_marked(&guarded, MARK, false, true),
        bare.encode("<|im_end|>", false, true)
    );
    // The first spelling is guarded and the second is the template's own.
    let text = format!("{}<think>", tok.guard("<think>", MARK));
    let mut want = bare.encode("<think>", false, true);
    want.push(id(&tok, "<think>"));
    assert_eq!(tok.encode_marked(&text, MARK, false, true), want);
    // A one-character token is a token in guarded text and out of it.
    let paragraph = tok.guard("한 <think> 한", MARK);
    let sect = id(&tok, "한");
    let mut want = vec![sect];
    want.extend(bare.encode(" <think> ", false, true));
    want.push(sect);
    assert_eq!(tok.encode_marked(&paragraph, MARK, false, true), want);
}

/// With no mark in the text `encode_marked` is `encode`, to the id: the marks are the only thing it
/// reads, so a conversation with no spelling in client text keeps the ids it always had, and the
/// guard leaves such a text alone.
#[test]
fn without_marks_encode_marked_is_encode() {
    let tok = with_specials();
    let spelled = |text: &str| {
        SPECIALS
            .iter()
            .any(|&(s, _)| s.chars().count() > 1 && text.contains(s))
    };
    for text in [
        "",
        "plain words ab ab",
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nx</think>y<|im_end|>",
        "한한 |im_end|> <|im_start| <think x",
    ] {
        for parse in [true, false] {
            assert_eq!(
                tok.encode_marked(text, MARK, false, parse),
                tok.encode(text, false, parse),
                "{text:?} {parse}"
            );
        }
        assert_eq!(
            matches!(tok.guard(text, MARK), Cow::Owned(_)),
            spelled(text),
            "{text:?}"
        );
    }
}

/// A mark that no spelling needed is cut out all the same, and an empty one is refused.
#[test]
fn a_stray_mark_is_cut_out_and_an_empty_mark_is_refused() {
    let (tok, bare) = (with_specials(), plain());
    assert_eq!(
        tok.encode_marked("a~#b", MARK, false, true),
        bare.encode("ab", false, true)
    );
    let guard = std::panic::catch_unwind(AssertUnwindSafe(|| tok.guard("x", "").len()));
    let encode = std::panic::catch_unwind(AssertUnwindSafe(|| {
        tok.encode_marked("x", "", false, true).len()
    }));
    assert!(guard.is_err() && encode.is_err());
}
