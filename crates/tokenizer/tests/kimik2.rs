//! Kimi K3's vocabulary: our ids equal mainline `llama-tokenize`'s, id for id, on every case of
//! `cases.txt` and `cases-kimik2.txt` in both parse modes, and the end tokens end generation.
//! The reference's ids are `kimik2-ids.tsv`, pinned by `tools/kimik2-oracle.sh` (its header names the
//! tree's commit): the `kimi-k2` pre-tokenizer is the reference's hand-written splitter, which no
//! regex of ours stands in for.
//!
//! `hw_`: needs the vocabulary file on the box (`bash tools/gate.sh --release -p bloomery-tokenizer
//! --test kimik2 -- --ignored --nocapture`). The vocabulary only: seconds.

#[path = "common/pins.rs"]
mod pins;

use std::path::PathBuf;

use tokenizer::Tokenizer;

/// The first shard of the K3 file on the box: the metadata and the tokenizer, no tensors.
fn vocabulary() -> PathBuf {
    PathBuf::from("/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf")
}

/// Every case encodes to the reference's ids, with special tokens parsed and without. FAIL-first:
/// with the contraction list or the digit grouping of the `kimi-k2` splitter changed, the rows of
/// contractions and of digit runs differ.
#[test]
#[ignore = "gate: the Kimi K3 vocabulary file on the box"]
fn hw_kimik2_ids_equal_the_reference() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the Kimi K3 vocabulary loads");
    let (rows, differ) = pins::compare(&tok, include_str!("kimik2-ids.tsv"));
    println!("kimik2: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The header's end of message (`tokenizer.ggml.eos_token_id`, id 163586) and the two end tokens
/// the reference's list names for Kimi, `[EOS]` and `[EOT]`, end generation.
#[test]
#[ignore = "gate: the Kimi K3 vocabulary file on the box"]
fn hw_kimik2_end_tokens_end_generation() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the Kimi K3 vocabulary loads");
    for text in ["<|end_of_msg|>", "[EOS]", "[EOT]"] {
        let id = tok.encode(text.as_bytes(), false, true);
        assert_eq!(id.len(), 1, "{text} is one token");
        assert!(
            tok.eog().contains(&id[0]),
            "{text} ({}) is not an end",
            id[0]
        );
    }
    assert_eq!(tok.encode(b"<|end_of_msg|>", false, true), [163_586]);
}
