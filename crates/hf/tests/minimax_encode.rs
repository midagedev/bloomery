//! The folder reader's vocabulary, run by serving's tokenizer: every row of `hf::vocab::PRE` is a
//! pre-tokenizer the tokenizer runs as Hugging Face's `tokenizers` does, and MiniMax-M3.1-Flash's
//! folder encodes every text of the pinned corpus to the ids `tokenizers` gives
//! (`fixtures/hf_encode_minimax.py` writes it: the texts of the Qwen corpus, the oracle prompts,
//! and every added token alone, back to back with every other and spelled over another). The
//! prompts' ids are also llama-tokenize's, which `tools/ref/data/minimaxm3` pins.

use std::fs::File;
use std::path::PathBuf;

use gguf::Value;
use sha2::{Digest, Sha256};

/// A marked vocabulary of two tokens under each row's name opens: a row the tokenizer cannot run
/// as `tokenizers` does is refused by name, so a row added to `PRE` without its reading ends here.
/// The table writes `add_bos_token` false as the folder reader does for a tokenizer with no BOS
/// template; the `llama-bpe` row's loader default is true and would demand a BOS id.
#[test]
fn every_pre_row_has_a_hugging_face_reading() {
    for (name, _, _) in hf::vocab::PRE {
        let keys: Vec<(String, Value)> = vec![
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::String(name.into())),
            (
                "tokenizer.ggml.tokens".into(),
                Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
            ),
            ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
            (
                "tokenizer.bloomery.eog_token_ids".into(),
                Value::Array(vec![Value::U32(0)]),
            ),
            ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ];
        let tok = tokenizer::Tokenizer::from_keys(&keys).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(tok.pre(), name);
    }
}

/// MiniMax-M3.1-Flash's folder, as serving builds its tokenizer from it, splits and encodes
/// every text of the corpus as Hugging Face does.
#[test]
#[ignore = "needs MiniMaxAI/MiniMax-M3.1-Flash's tokenizer files and config.json at $BLOOMERY_MM31_HF"]
fn hw_the_served_minimax_tokenizer_encodes_the_corpus_as_hf() {
    let dir = PathBuf::from(
        std::env::var_os("BLOOMERY_MM31_HF").expect("$BLOOMERY_MM31_HF names the folder"),
    );
    let path = format!(
        "{}/tests/fixtures/minimax-hf-encode.json.gz",
        env!("CARGO_MANIFEST_DIR")
    );
    let corpus: serde_json::Value =
        serde_json::from_reader(flate2::read::GzDecoder::new(File::open(&path).unwrap())).unwrap();
    let file = std::fs::read(dir.join("tokenizer.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&file)),
        corpus["sha256"].as_str().unwrap(),
        "{}: not the tokenizer.json of {}",
        dir.display(),
        corpus["source"]
    );
    let config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let spec = hf::vocab::VocabSpec::parse(&config)
        .unwrap()
        .with_pre("minimax-m2")
        .unwrap();
    let vocab = hf::vocab::keys(&dir, &spec).unwrap();
    let served = tokenizer::Tokenizer::from_keys(&vocab.keys).unwrap();
    let mut split = served.special_tokens().to_vec();
    split.sort_unstable();
    assert_eq!(split, vocab.added, "serving splits on the added tokens");
    assert_eq!(
        served.eog(),
        vocab.stop,
        "serving stops where HF's generate does"
    );
    let cases = corpus["cases"].as_array().unwrap();
    assert!(cases.len() >= 15_000, "{} cases in {path}", cases.len());
    let (mut ids, mut failures) = (0, Vec::new());
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let ours = served.encode(text, false, true);
        ids += want.len();
        if ours != want {
            failures.push(format!("{text:?}: ours {ours:?}, HF {want:?}"));
        }
    }
    println!(
        "HF-ENCODE cases={} ids={ids} differences={} tokenizers={} source={}",
        cases.len(),
        failures.len(),
        corpus["tokenizers"],
        corpus["source"]
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
