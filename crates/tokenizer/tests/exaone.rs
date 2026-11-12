//! K-EXAONE-2.0's vocabulary: our ids equal mainline `llama-tokenize`'s, id for id, on every NFC
//! case of `cases.txt` and `cases-exaone.txt` in both parse modes, the cases whose text is not
//! NFC hold the Hugging Face tokenizer's ids (`exaone-nfc-ids.tsv`: the reference normalizes no
//! text on this path — its header's `normalizer.nfc` is applied by its UGM tokenizer alone —
//! while the model's `tokenizer.json` normalizes NFC, and the engine follows the model), and the
//! end tokens end generation. The pins are `tools/exaone-oracle.sh`'s (its header names the tree's
//! commit): the `exaone-moe` pre-tokenizer runs through the reference's regex fallback, which no
//! splitter of ours stands in for.
//!
//! `hw_`: needs the vocabulary file on the box (`bash tools/gate.sh --release -p bloomery-tokenizer
//! --test exaone -- --ignored --nocapture`). The vocabulary only: seconds.

#[path = "common/pins.rs"]
mod pins;

use std::path::PathBuf;

use tokenizer::Tokenizer;

/// The vocab-only file the converter writes from the hub folder (`refset::arch::exaone_moe::VOCABULARY`,
/// its full path): the metadata and the tokenizer, no tensors.
fn vocabulary() -> PathBuf {
    PathBuf::from(refset::arch::exaone_moe::VOCABULARY)
}

/// Every NFC case encodes to the reference's ids, with special tokens parsed and without.
/// FAIL-first: with the matcher's join loosened (a space not followed by a letter joins too),
/// the rows of space runs differ. The digit branch is invisible to these pins — the vocabulary
/// holds no token of two numbers (`tokenizer.json`: no merge of two digits), so one number a
/// word and three a word give the same ids — and its FAIL-first is `pretok`'s `exaone_moe_cases`
/// ("12345" as one word of five).
#[test]
#[ignore = "gate: the K-EXAONE-2.0 vocabulary file on the box"]
fn hw_exaone_ids_equal_the_reference() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the K-EXAONE-2.0 vocabulary loads");
    let (rows, differ) = pins::compare(&tok, include_str!("exaone-ids.tsv"));
    println!("exaone: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// Every case whose text is not NFC (decomposed Hangul, combining accents) encodes to the
/// Hugging Face tokenizer's ids: the engine's NFC step runs before the pre-tokenizer, as the
/// model's own tokenizer runs it. Those rows hold no special token, so the pins' one id list
/// holds both parse modes. FAIL-first: with the pre-tokenizer's NFC step dropped (`Pre::nfc`),
/// every row differs — the decomposed text reaches the pre-tokenizer as jamo and splits apart.
#[test]
#[ignore = "gate: the K-EXAONE-2.0 vocabulary file on the box"]
fn hw_exaone_nfc_ids_equal_hugging_faces() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the K-EXAONE-2.0 vocabulary loads");
    let (rows, differ) = pins::compare(&tok, include_str!("exaone-nfc-ids.tsv"));
    println!("exaone nfc: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The model's generation config lists two ends (`fp8.generation_config.json`: eos 2 and 53):
/// `[EOS]` and `<|endofturn|>` are single tokens and end generation.
#[test]
#[ignore = "gate: the K-EXAONE-2.0 vocabulary file on the box"]
fn hw_exaone_end_tokens_end_generation() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the K-EXAONE-2.0 vocabulary loads");
    for text in ["<|endofturn|>", "[EOS]"] {
        let id = tok.encode(text.as_bytes(), false, true);
        assert_eq!(id.len(), 1, "{text} is one token");
        assert!(
            tok.eog().contains(&id[0]),
            "{text} ({}) is not an end",
            id[0]
        );
    }
    assert_eq!(tok.encode(b"<|endofturn|>", false, true), [53]);
    assert_eq!(tok.encode(b"[EOS]", false, true), [2]);
}
