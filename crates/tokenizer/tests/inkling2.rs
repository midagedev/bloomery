//! Inkling 2's vocabulary (the Mini's `inkling2` pre-tokenizer and the Small's `gpt-4o`, one table): our ids equal the
//! reference's, id for id, on every case of `cases.txt` and `cases-inkling2.txt` in both parse modes, for both released files,
//! and the files add no BOS and end a generation at the model's end-of-sampling token. The reference is the oracle tree of
//! llama.cpp PR #32695 (`llama-vocab.cpp` maps `inkling2` to the GPT4O expression with a null BOS); the pins are
//! `inkling2-ids.tsv`, written by `tools/inkling2-cases-oracle.sh`, run on each file (the two tables are the same bytes but
//! for the header's pre name).
//!
//! `hw_`: needs the vocabulary files on the box: `bash tools/gate.sh --release -p bloomery-tokenizer --test inkling2 --
//! --ignored --nocapture`. The vocabulary only: seconds.

#[path = "common/pins.rs"]
mod pins;

use tokenizer::Tokenizer;

/// The Mini's UD-Q4_K_XL file (pre `inkling2`).
const MINI: &str = "/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf";
/// The Small's first shard (pre `gpt-4o`).
const SMALL: &str =
    "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf";

/// The vocabulary of `path`, held to be the one the file's pre-tokenizer name says: a file of another model is a named failure,
/// not a count of differing rows.
fn vocabulary(path: &str, pre: &str) -> Tokenizer {
    let tok = Tokenizer::from_gguf(path)
        .unwrap_or_else(|e| panic!("the vocabulary of {path} loads: {e}"));
    assert_eq!(tok.pre(), pre, "{path} names another pre-tokenizer");
    tok
}

/// Every case encodes to the reference's ids on both files. FAIL-first: with the `inkling2` row on `Pre::Qwen2` (one digit a
/// word, the older expression in place of GPT4O's three) the rows of digit runs, contractions and mixed scripts differ.
#[test]
#[ignore = "gate: the Inkling 2 vocabulary files on the box"]
fn hw_inkling2_ids_equal_the_reference() {
    for (path, pre) in [(MINI, "inkling2"), (SMALL, "gpt-4o")] {
        let tok = vocabulary(path, pre);
        let (rows, differ) = pins::compare(&tok, include_str!("inkling2-ids.tsv"));
        println!("inkling2 {pre}: {rows} cases, {} differ", differ.len());
        assert!(differ.is_empty(), "{pre}: {}", differ.join("\n"));
    }
}

/// The two files add no BOS (the reference nulls it for `inkling2` and the Small's header says
/// `add_bos_token` false), and the model's `<|content_model_end_sampling|>` is the header's end token and ends a generation.
/// FAIL-first: with `add_bos` read as true for the pre, every prompt gains an id; with the end token out of the end list the
/// last assertion fails.
#[test]
#[ignore = "gate: the Inkling 2 vocabulary files on the box"]
fn hw_inkling2_adds_no_bos_and_ends_at_end_of_sampling() {
    for (path, pre) in [(MINI, "inkling2"), (SMALL, "gpt-4o")] {
        let tok = vocabulary(path, pre);
        assert!(!tok.add_bos(), "{pre}: no BOS is added");
        assert_eq!(
            tok.encode(b"Hello", false, true),
            tok.encode(b"Hello", false, false)
        );
        let id = tok.encode(b"<|content_model_end_sampling|>", false, true);
        assert_eq!(id.len(), 1, "{pre}: the end token is one token");
        assert!(
            tok.eog().contains(&id[0]),
            "{pre}: {} is no end of generation",
            id[0]
        );
        assert_eq!(
            tok.specials().eos,
            Some(id[0]),
            "{pre}: the header's eos id"
        );
    }
}
