//! Tokenizer gates: our ids equal the reference's `llama-tokenize` ids, id for
//! id, and the reference's ids decode back to the source bytes — for each
//! vocabulary the engine runs: V4.1's (`deepseek-v3` pre-tokenizer) and
//! Qwen3-MoE's (`qwen2`).
//!
//! The oracle files come from `crates/tokenizer/tools/oracle.sh` (the
//! `gate-tokenizer` recipe runs it first, once per vocabulary) under
//! `$BLOOMERY_DATA/tokenizer/` and `$BLOOMERY_DATA/tokenizer-qwen3moe/`.
//! Every text is checked in both of the reference's parse modes: special
//! tokens parsed (its default) and `--no-parse-special`. Neither adds a BOS
//! for these vocabularies (`tokenizer.ggml.add_bos_token` is false), so there
//! is nothing to strip before comparing; the round trip decodes with special
//! tokens rendered, which returns their text, so it is exact for every input
//! that is valid UTF-8. Invalid input is not: the reference decodes it to
//! U+FFFD first.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tokenizer::{Decoder, Tokenizer};

/// One vocabulary under test: its GGUF file, the oracle set its ids are in,
/// and the special roles it leaves to the reference's hash order.
struct Vocabulary {
    file: PathBuf,
    set: &'static str,
    /// Roles the reference fills by text from several candidates, picking by
    /// the iteration order of its token map. No such role enters `encode`
    /// (ids read only the partition's special texts and BOS/EOS), so the ids
    /// stay well defined; the list is pinned so that a new ambiguity, or one
    /// that goes away, is a change the gate names.
    ambiguous: &'static [&'static str],
    /// Texts that must end generation whichever candidate an ambiguous role
    /// took.
    eog: &'static [&'static str],
}

/// V4.1's first shard, `$BLOOMERY_V41_DIR` or the served set's directory.
fn v41() -> Vocabulary {
    Vocabulary {
        file: v41_path(),
        set: "tokenizer",
        ambiguous: &[],
        eog: &[],
    }
}

/// The Qwen3-MoE file, `$BLOOMERY_QWEN3MOE_VOCAB` or the one on the box.
fn qwen3moe() -> Vocabulary {
    let file = std::env::var("BLOOMERY_QWEN3MOE_VOCAB")
        .unwrap_or_else(|_| "/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf".into());
    Vocabulary {
        file: PathBuf::from(file),
        set: "tokenizer-qwen3moe",
        // No `eot_token_id` key, and both `<|im_end|>` and `<|endoftext|>`
        // are CONTROL tokens on the reference's EOT list. Both are in the
        // end-of-generation set either way (every EOG text is).
        ambiguous: &["eot"],
        eog: &["<|im_end|>", "<|endoftext|>"],
    }
}

fn data_dir(v: &Vocabulary) -> PathBuf {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    PathBuf::from(base).join(v.set)
}

/// V4.1's first shard: [`gguf::v41::model`], the path's owner.
fn v41_path() -> PathBuf {
    PathBuf::from(gguf::v41::model())
}

fn load(v: &Vocabulary) -> Tokenizer {
    Tokenizer::from_gguf(&v.file).unwrap_or_else(|e| panic!("{}: {e}", v.file.display()))
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run crates/tokenizer/tools/oracle.sh)",
            path.display()
        )
    })
}

fn read_ids(path: &Path) -> Vec<u32> {
    String::from_utf8(read(path))
        .expect("ids files are ASCII")
        .lines()
        .map(|l| {
            l.parse()
                .unwrap_or_else(|e| panic!("{}: {l:?}: {e}", path.display()))
        })
        .collect()
}

/// `Err` with the first difference, its position and both sides' pieces.
fn compare(tok: &Tokenizer, ours: &[u32], oracle: &[u32]) -> Result<(), String> {
    let piece = |id: u32| String::from_utf8_lossy(tok.piece(id, true)).into_owned();
    match ours.iter().zip(oracle).position(|(a, b)| a != b) {
        Some(i) => {
            let from = i.saturating_sub(4);
            Err(format!(
                "first mismatch at {i}: ours {} {:?}, reference {} {:?}; reference context {:?}",
                ours[i],
                piece(ours[i]),
                oracle[i],
                piece(oracle[i]),
                tok.decode(&oracle[from..(i + 4).min(oracle.len())])
            ))
        }
        None if ours.len() != oracle.len() => Err(format!(
            "lengths differ: ours {}, reference {} (equal up to the shorter)",
            ours.len(),
            oracle.len()
        )),
        None => Ok(()),
    }
}

/// Decode `ids` one at a time through the streaming decoder.
fn stream(tok: &Tokenizer, ids: &[u32]) -> String {
    let mut d = Decoder::new(tok, true);
    let mut s = String::new();
    for &id in ids {
        if let Some(t) = d.push(id) {
            s.push_str(t);
        }
    }
    s.extend(d.finish());
    s
}

/// One text in both parse modes: ids equal, and the reference's ids decode
/// (whole and streamed) to the text when it is UTF-8. Returns the failures.
fn check(tok: &Tokenizer, name: &str, text: &[u8], stem: &Path, failures: &mut Vec<String>) {
    for (mode, parse, suffix) in [("parse", true, "ids"), ("no-parse", false, "nps.ids")] {
        let oracle = read_ids(&stem.with_extension(suffix));
        let t0 = Instant::now();
        let ours = tok.encode(text, false, parse);
        let secs = t0.elapsed().as_secs_f64();
        let verdict = compare(tok, &ours, &oracle);
        let round = match std::str::from_utf8(text) {
            Err(_) => "not UTF-8, no round trip".to_string(),
            Ok(_) if tok.decode_bytes(&oracle, true) != text => {
                failures.push(format!(
                    "{name} {mode}: the reference's ids do not decode to the text"
                ));
                "round trip FAILED".into()
            }
            Ok(s) if stream(tok, &oracle) != s => {
                failures.push(format!(
                    "{name} {mode}: the streamed decode differs from the text"
                ));
                "stream FAILED".into()
            }
            Ok(_) => "round trip exact".into(),
        };
        println!(
            "set {name} {mode}: ours {} = reference {} ids: {} — {round} — encode {:.0} tok/s",
            ours.len(),
            oracle.len(),
            if verdict.is_ok() {
                "equal"
            } else {
                "DIFFERENT"
            },
            ours.len() as f64 / secs.max(1e-9),
        );
        if let Err(e) = verdict {
            failures.push(format!("{name} {mode}: {e}"));
        }
    }
}

/// Every engram corpus text: ids equal the reference's, both parse modes.
fn corpus_ids_equal_reference(v: &Vocabulary) {
    let tok = load(v);
    let dir = data_dir(v);
    let mut failures = Vec::new();
    for name in ["code", "prose", "prose-all", "threads", "korean"] {
        let stem = dir.join(name);
        let text = read(&stem.with_extension("txt"));
        check(&tok, name, &text, &stem, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the V4.1 vocabulary and the oracle files on the box"]
fn hw_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&v41());
}

#[test]
#[ignore = "needs the qwen3moe vocabulary and the oracle files on the box"]
fn hw_qwen3moe_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&qwen3moe());
}

/// The hand-picked strings of `tests/cases.txt`, and the conditions under
/// which "equal to the reference" is well defined: the partition order among
/// equal-length special tokens cannot matter, and the roles picked from
/// several candidate texts are the vocabulary's pinned list.
fn cases_equal_reference(v: &Vocabulary) {
    let tok = load(v);
    assert_eq!(
        tok.special_overlap(),
        None,
        "two equal-length special tokens can overlap in text"
    );
    assert_eq!(
        tok.ambiguous_roles(),
        v.ambiguous,
        "roles picked by hash order"
    );
    for text in v.eog {
        let id = (0..tok.n_vocab() as u32).find(|&id| tok.text(id) == Some(text));
        assert!(
            id.is_some_and(|id| tok.eog().contains(&id)),
            "{text} ({id:?}) is not in the end-of-generation set {:?}",
            tok.eog()
        );
    }

    let dir = data_dir(v).join("cases");
    let mut stems: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| {
            panic!(
                "{}: {e} (run crates/tokenizer/tools/oracle.sh)",
                dir.display()
            )
        })
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .map(|p| p.with_extension(""))
        .collect();
    stems.sort();
    assert!(
        stems.len() >= 20,
        "{} cases under {}",
        stems.len(),
        dir.display()
    );
    let mut failures = Vec::new();
    for stem in &stems {
        let text = read(&stem.with_extension("txt"));
        let name = format!(
            "case {} {:?}",
            stem.file_name().unwrap_or_default().to_string_lossy(),
            String::from_utf8_lossy(&text)
        );
        check(&tok, &name, &text, stem, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the V4.1 vocabulary and the oracle files on the box"]
fn hw_cases_equal_reference() {
    cases_equal_reference(&v41());
}

#[test]
#[ignore = "needs the qwen3moe vocabulary and the oracle files on the box"]
fn hw_qwen3moe_cases_equal_reference() {
    cases_equal_reference(&qwen3moe());
}
