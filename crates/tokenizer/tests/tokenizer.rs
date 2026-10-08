//! Tokenizer gates: our ids equal the reference's `llama-tokenize` ids, id for
//! id, and the reference's ids decode back to the source bytes — for each
//! vocabulary the engine runs: V4.1's (`deepseek-v3` pre-tokenizer),
//! Qwen3-MoE's (`qwen2`), GLM-5.3-Flash's (`glm4`) and Qwen3.8-Flash-Next's
//! (`qwen35`).
//!
//! The reference's sets are written once by `just dump-ref-tokenizer`
//! (`crates/tokenizer/tools/oracle.sh`) and opened here through their refset
//! family, which refuses by name a set written by another executable or
//! library, from another vocabulary file than the one loaded below, with a
//! text that is not the one dumped, or that never finished; the set's cases
//! must be the ones in `tests/cases.txt`.
//! Every text is checked in both of the reference's parse modes: special
//! tokens parsed (its default) and `--no-parse-special`. Neither adds a BOS
//! for these vocabularies (`tokenizer.ggml.add_bos_token` is false), so there
//! is nothing to strip before comparing; the round trip decodes with special
//! tokens rendered, which returns their text, so it is exact for every input
//! that is valid UTF-8. Invalid input is not: the reference decodes it to
//! U+FFFD first.

use std::path::PathBuf;
use std::time::Instant;

use refset::arch::tokenizer::{GLM5NEXT, QWEN3MOE, QWEN38, V41};
use refset::family::Family;
use refset::tokenizer::{Text, TokenizerSet};
use tokenizer::{Decoder, Tokenizer};

/// One vocabulary under test: its refset family, which names the GGUF file
/// and the set its ids are in, and the special roles it leaves to the
/// reference's hash order.
struct Vocabulary {
    family: &'static Family,
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

/// V4.1's first shard: the family's file, `$BLOOMERY_V41_DIR` or the served
/// set's directory.
fn v41() -> Vocabulary {
    Vocabulary {
        family: &V41,
        ambiguous: &[],
        eog: &[],
    }
}

/// The Qwen3-MoE file, `$BLOOMERY_QWEN3MOE_VOCAB` or the one on the box.
fn qwen3moe() -> Vocabulary {
    Vocabulary {
        family: &QWEN3MOE,
        // No `eot_token_id` key, and both `<|im_end|>` and `<|endoftext|>`
        // are CONTROL tokens on the reference's EOT list. Both are in the
        // end-of-generation set either way (every EOG text is).
        ambiguous: &["eot"],
        eog: &["<|im_end|>", "<|endoftext|>"],
    }
}

/// GLM-5.3-Flash's first shard on the box, the file's identity.
fn glm5next() -> Vocabulary {
    Vocabulary {
        family: &GLM5NEXT,
        ambiguous: &[],
        // The header's eos, eot and eom: a generation ends on any of the three.
        eog: &["<|endoftext|>", "<|user|>", "<|observation|>"],
    }
}

/// Qwen3.8-Flash-Next's first shard on the box.
fn qwen38() -> Vocabulary {
    Vocabulary {
        family: &QWEN38,
        // No `eot_token_id` key, as Qwen3-MoE's: `<|im_end|>` (the eos) and
        // `<|endoftext|>` (the bos and pad) are both on the reference's EOT list.
        ambiguous: &["eot"],
        eog: &["<|im_end|>", "<|endoftext|>"],
    }
}

impl Vocabulary {
    /// The GGUF file the set was dumped from and this gate loads.
    fn file(&self) -> PathBuf {
        PathBuf::from(
            self.family
                .runs()
                .unwrap_or_else(|e| panic!("{}: {e}", self.family.name)),
        )
    }
}

/// The vocabulary's set opened through its family and held to `cases.txt`;
/// a refusal names the recipe that re-takes it.
fn open(v: &Vocabulary) -> TokenizerSet {
    let f = v.family;
    let set = TokenizerSet::open(&f.path(f.sets[0]), f).unwrap_or_else(|e| refuse(f, &e));
    // The hand-picked cases file the sets' case texts must come from.
    set.check_cases(include_bytes!("cases.txt"))
        .unwrap_or_else(|e| refuse(f, &e));
    set
}

fn refuse(f: &Family, e: &refset::RefError) -> ! {
    panic!("{e}\n  re-take the set: {}", f.recipe)
}

fn load(v: &Vocabulary) -> Tokenizer {
    let file = v.file();
    Tokenizer::from_gguf(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()))
}

/// `Err` with the first difference: its position in ids and in the text's
/// bytes, the differing ids' pieces, and both sides' ids around it with the
/// text each window decodes to. A length difference with an equal prefix is
/// reported the same way, at the shorter side's end.
fn compare(tok: &Tokenizer, ours: &[u32], oracle: &[u32]) -> Result<(), String> {
    let i = match ours.iter().zip(oracle).position(|(a, b)| a != b) {
        Some(i) => i,
        None if ours.len() != oracle.len() => ours.len().min(oracle.len()),
        None => return Ok(()),
    };
    let piece = |ids: &[u32]| match ids.get(i) {
        Some(&id) => format!("{id} {:?}", String::from_utf8_lossy(tok.piece(id, true))),
        None => "(end)".into(),
    };
    let (from, to) = (i.saturating_sub(4), i + 4);
    let window = |ids: &[u32]| {
        let w = &ids[from.min(ids.len())..to.min(ids.len())];
        format!("{w:?} {:?}", tok.decode(w))
    };
    Err(format!(
        "first difference at id {i} (byte {} of the reference's decode; ours {} ids, reference {}): \
         ours {}, reference {}\n    ids {from}..{to} ours      {}\n    ids {from}..{to} reference {}",
        tok.decode_bytes(&oracle[..i], true).len(),
        ours.len(),
        oracle.len(),
        piece(ours),
        piece(oracle),
        window(ours),
        window(oracle),
    ))
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
fn check(tok: &Tokenizer, name: &str, text: &Text, failures: &mut Vec<String>) {
    let bytes = text.read().unwrap_or_else(|e| panic!("{e}"));
    for (mode, parse) in [("parse", true), ("no-parse", false)] {
        let oracle = text.ids(parse).unwrap_or_else(|e| panic!("{e}"));
        let t0 = Instant::now();
        let ours = tok.encode(&bytes, false, parse);
        let secs = t0.elapsed().as_secs_f64();
        let verdict = compare(tok, &ours, &oracle);
        let round = match std::str::from_utf8(&bytes) {
            Err(_) => "not UTF-8, no round trip".to_string(),
            Ok(_) if tok.decode_bytes(&oracle, true) != bytes => {
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
    let set = open(v);
    let names: Vec<&str> = set.corpora().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        ["code", "prose", "prose-all", "threads", "korean"],
        "the corpus texts of {}",
        set.dir.display()
    );
    let mut failures = Vec::new();
    for text in set.corpora() {
        check(&tok, &text.name, text, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the V4.1 vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&v41());
}

#[test]
#[ignore = "needs the qwen3moe vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_qwen3moe_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&qwen3moe());
}

#[test]
#[ignore = "needs the glm5next vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_glm5next_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&glm5next());
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

    let set = open(v);
    let cases: Vec<&Text> = set.cases().collect();
    assert!(
        cases.len() >= 20,
        "{} cases in {}",
        cases.len(),
        set.dir.display()
    );
    let mut failures = Vec::new();
    for text in cases {
        let bytes = text.read().unwrap_or_else(|e| panic!("{e}"));
        let name = format!(
            "case {} {:?}",
            text.name.trim_start_matches("cases/"),
            String::from_utf8_lossy(&bytes)
        );
        check(&tok, &name, text, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the V4.1 vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_cases_equal_reference() {
    cases_equal_reference(&v41());
}

#[test]
#[ignore = "needs the qwen3moe vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_qwen3moe_cases_equal_reference() {
    cases_equal_reference(&qwen3moe());
}

#[test]
#[ignore = "needs the glm5next vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_glm5next_cases_equal_reference() {
    cases_equal_reference(&glm5next());
}

#[test]
#[ignore = "needs the qwen38 vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_qwen38_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&qwen38());
}

#[test]
#[ignore = "needs the qwen38 vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_qwen38_cases_equal_reference() {
    cases_equal_reference(&qwen38());
}
