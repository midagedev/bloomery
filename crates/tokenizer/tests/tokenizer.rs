//! Tokenizer gates: our ids equal the reference's `llama-tokenize` ids, id for
//! id, and the reference's ids decode back to the source bytes.
//!
//! The oracle files come from `crates/tokenizer/tools/oracle.sh` (the
//! `gate-tokenizer` recipe runs it first) under `$BLOOMERY_DATA/tokenizer/`.
//! Every text is checked in both of the reference's parse modes: special
//! tokens parsed (its default) and `--no-parse-special`. Neither adds a BOS
//! for this vocabulary (`tokenizer.ggml.add_bos_token` is false), so there is
//! nothing to strip before comparing; the round trip decodes with special
//! tokens rendered, which returns their text, so it is exact for every input
//! that is valid UTF-8. Invalid input is not: the reference decodes it to
//! U+FFFD first.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tokenizer::{Decoder, Tokenizer};

fn data_dir() -> PathBuf {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    PathBuf::from(base).join("tokenizer")
}

fn vocab_path() -> PathBuf {
    let dir = std::env::var("BLOOMERY_V41_DIR").unwrap_or_else(|_| {
        "/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8".into()
    });
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.to_string_lossy().ends_with(".gguf") && p.to_string_lossy().contains("-00001-of-")
        })
        .collect();
    shards.sort();
    shards
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("no first shard under {dir}"))
}

fn load() -> Tokenizer {
    let path = vocab_path();
    Tokenizer::from_gguf(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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
#[test]
#[ignore = "needs the V4.1 vocabulary and the oracle files on the box"]
fn hw_corpus_ids_equal_reference() {
    let tok = load();
    let dir = data_dir();
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

/// The hand-picked strings of `tests/cases.txt`, and the conditions under
/// which "equal to the reference" is well defined: the partition order among
/// equal-length special tokens cannot matter, and no special role was picked
/// from several candidate texts.
#[test]
#[ignore = "needs the V4.1 vocabulary and the oracle files on the box"]
fn hw_cases_equal_reference() {
    let tok = load();
    assert_eq!(
        tok.special_overlap(),
        None,
        "two equal-length special tokens can overlap in text"
    );
    assert!(
        tok.ambiguous_roles().is_empty(),
        "roles picked by hash order: {:?}",
        tok.ambiguous_roles()
    );

    let dir = data_dir().join("cases");
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
