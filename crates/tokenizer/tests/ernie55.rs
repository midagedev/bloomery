//! ERNIE 5.5 Flash Preview's vocabulary: our ids equal Hugging Face's `tokenizers`, id for id, on every case of
//! `cases-ernie55.txt` (English, Korean, Chinese and Japanese text with digits, the digit groups of every script, the numbers
//! that are no decimal digit, code and its punctuation, the whitespace rules), and the chat tokens are single ids that end a
//! turn. The reference is the package that runs the folder's own `tokenizer.json`; the pins are `ernie55-ids.tsv`, written by
//! the script that header names from `cases-ernie55.txt`, with the version of the package in its header. The vocabulary is
//! read from that `tokenizer.json` as the header keys a GGUF file would carry (`tokenizer.ggml.pre` `ernie5`, and
//! `tokenizer.bloomery.eog_token_ids`, which selects Hugging Face's reading of the pre-tokenizer).
//!
//! `hw_`: needs the model folder on the box (`BLOOMERY_REF_MODEL`, default `/models/ERNIE-5.5-Flash-Preview`): `bash
//! tools/gate.sh --release -p bloomery-tokenizer --test ernie55 -- --ignored --nocapture`. The vocabulary only: seconds.

#[path = "common/pins.rs"]
mod pins;

use gguf::Value;
use serde_json::Value as Json;
use tokenizer::Tokenizer;

/// The vendor's folder at its box path, where `BLOOMERY_REF_MODEL` names none (the gate's fixture tier
/// names the cut, whose `tokenizer.json` is the folder's file).
const FOLDER: &str = "/models/ERNIE-5.5-Flash-Preview";

fn folder() -> String {
    std::env::var("BLOOMERY_REF_MODEL")
        .ok()
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| FOLDER.to_string())
}

/// The id of `<|ernie:im_end|>`, which the folder's `eos_token` names and generation ends on.
const IM_END: u32 = 8;

/// The folder's vocabulary as header keys: the byte-level tokens by id, an added token a control
/// token when it is special and a user-defined one when it is not, the merges as `a b`.
fn vocabulary() -> Tokenizer {
    let text = std::fs::read(format!("{}/tokenizer.json", folder()))
        .expect("the ERNIE 5.5 folder's tokenizer.json");
    let tok: Json = serde_json::from_slice(&text).expect("tokenizer.json parses");
    let vocab = tok["model"]["vocab"].as_object().expect("a vocabulary");
    let mut tokens = vec![String::new(); vocab.len()];
    for (token, id) in vocab {
        tokens[id.as_u64().expect("an id") as usize] = token.clone();
    }
    let mut types = vec![1i32; tokens.len()];
    for added in tok["added_tokens"].as_array().expect("added tokens") {
        let id = added["id"].as_u64().expect("an id") as usize;
        assert_eq!(
            tokens[id],
            added["content"].as_str().unwrap(),
            "added token {id}"
        );
        types[id] = if added["special"].as_bool().unwrap() {
            3
        } else {
            4
        };
    }
    let merges: Vec<Value> = tok["model"]["merges"]
        .as_array()
        .expect("merges")
        .iter()
        .map(|m| {
            let pair = m.as_array().expect("a merge is two tokens");
            Value::String(format!(
                "{} {}",
                pair[0].as_str().unwrap(),
                pair[1].as_str().unwrap()
            ))
        })
        .collect();
    let strings = |v: Vec<String>| Value::Array(v.into_iter().map(Value::String).collect());
    let keys = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::String("ernie5".into()),
        ),
        ("tokenizer.ggml.tokens".to_string(), strings(tokens)),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::Array(types.into_iter().map(Value::I32).collect()),
        ),
        ("tokenizer.ggml.merges".to_string(), Value::Array(merges)),
        ("tokenizer.ggml.bos_token_id".to_string(), Value::U32(1)),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            Value::U32(IM_END),
        ),
        (
            "tokenizer.bloomery.eog_token_ids".to_string(),
            Value::Array(vec![Value::U32(IM_END)]),
        ),
    ];
    Tokenizer::from_keys(&keys).expect("the ERNIE 5.5 vocabulary loads")
}

/// Every case encodes to Hugging Face's ids. FAIL-first: with the number runs cut from their start
/// (`Pre::Ernie5Hf` on `DigitRule::Front(3)`), the rows of digit runs past three differ; with
/// the CJK pass dropped, the rows of Han and kana beside letters differ.
#[test]
#[ignore = "gate: the ERNIE 5.5 folder on the box"]
fn hw_ernie55_ids_equal_hugging_faces() {
    let tok = vocabulary();
    assert_eq!(tok.pre(), "ernie5");
    let (rows, differ) = pins::compare(&tok, include_str!("ernie55-ids.tsv"));
    println!("ernie55: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The chat tokens are single ids when special tokens are parsed, and `<|ernie:im_end|>` ends
/// generation.
#[test]
#[ignore = "gate: the ERNIE 5.5 folder on the box"]
fn hw_ernie55_chat_tokens_are_single_ids() {
    let tok = vocabulary();
    for (text, id) in [
        ("<|ernie:im_start|>", 7),
        ("<|ernie:im_end|>", IM_END),
        ("<|ERNIE_BEGIN|>", 1),
        ("<|ERNIE_MASK|>", 2),
    ] {
        assert_eq!(tok.encode(text.as_bytes(), false, true), [id], "{text}");
    }
    assert!(tok.eog().contains(&IM_END));
}
