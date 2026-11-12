//! The raon vocabulary: our ids equal `llama-tokenize`'s of the reference tree, id for id, on every case of
//! `cases.txt` and `cases-raon.txt` in both parse modes, and the header's two end tokens end generation.
//! The reference's ids are `raon-ids.tsv`, pinned by `tools/raon-oracle.sh` (its header names the tree's
//! commit): the raon pre-tokenizer and the NFC normalizer exist in that tree only. The nano dialect's file
//! carries the same vocabulary (its token, type and merge arrays are the Mini file's byte for byte; the two
//! files' chat templates differ, which this gate does not read), so its one case runs the same pins on it.
//!
//! `hw_`: needs the vocabulary file on the box (`bash tools/gate.sh --release -p bloomery-tokenizer --test raon
//! -- --ignored --nocapture`). The vocabulary only: seconds.

#[path = "common/pins.rs"]
mod pins;

use std::path::PathBuf;

use tokenizer::Tokenizer;

/// The Q4_K_M file on the box.
fn vocabulary() -> PathBuf {
    PathBuf::from("/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf")
}

/// The nano dialect's Q8_0 file on the box.
fn nano_vocabulary() -> PathBuf {
    PathBuf::from("/data/models/Raon-1-nano-gguf/raon-1-nano-Q8_0.gguf")
}

/// Every case encodes to the reference's ids, with special tokens parsed and without. FAIL-first:
/// with the NFC step or the digit grouping of the raon pre-tokenizer off, the rows of combining marks and
/// of digit runs differ.
#[test]
#[ignore = "gate: the raon vocabulary file on the box"]
fn hw_raon_ids_equal_the_reference() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the raon vocabulary loads");
    let (rows, differ) = pins::compare(&tok, include_str!("raon-ids.tsv"));
    println!("raon: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The nano file's ids equal the reference's on the same pins: its vocabulary is the Mini file's own
/// (the module doc), so the one nano case re-reads `raon-ids.tsv` through it. FAIL-first: a nano file
/// whose tokens differ at any row the comparison names it.
#[test]
#[ignore = "gate: the raon vocabulary file on the box"]
fn hw_raonnano_file_is_the_raon_vocabulary() {
    let tok = Tokenizer::from_gguf(nano_vocabulary()).expect("the nano vocabulary loads");
    let (rows, differ) = pins::compare(&tok, include_str!("raon-ids.tsv"));
    println!("raon1nano: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The header's `<|im_end|>` and `<|endoftext|>` end generation: the two ids the model's generation
/// config lists.
#[test]
#[ignore = "gate: the raon vocabulary file on the box"]
fn hw_raon_end_tokens_end_generation() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the raon vocabulary loads");
    for text in ["<|im_end|>", "<|endoftext|>"] {
        let id = tok.encode(text, false, true);
        assert_eq!(id.len(), 1, "{text} is one token");
        assert!(
            tok.eog().contains(&id[0]),
            "{text} ({}) is not an end",
            id[0]
        );
    }
}
