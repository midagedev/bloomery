//! Tokenizer gates: our ids equal the reference's `llama-tokenize` ids, id for
//! id, and the reference's ids decode back to the source bytes — for each
//! vocabulary the engine runs: V4.1's (`deepseek-v3` pre-tokenizer),
//! Qwen3-MoE's (`qwen2`), GLM-5.3-Flash's (`glm4`), Qwen3.8-Flash-Next's
//! (`qwen35`), Nemotron 3 Nano 4B's (`pixtral`), MiniMax-M3's (`minimax-m2`)
//! Muse Spark Open's (`gpt-4o`), Mistral Large 4's (`tekken-num3`, with cases of its own for its
//! digit runs and special tokens) and Nanbeige4.3-33B's (`llama`, SentencePiece with byte fallback).
//!
//! The reference's sets are written once by `just dump-ref-tokenizer`
//! (`crates/tokenizer/tools/oracle.sh`) and opened here through their refset
//! family, which refuses by name a set written by another executable or
//! library, from another vocabulary file than the one loaded below, with a
//! text that is not the one dumped, or that never finished; the set's cases
//! must be the ones in `tests/cases.txt`.
//! Every text is checked in both of the reference's parse modes: special
//! tokens parsed (its default) and `--no-parse-special`. The first four
//! vocabularies do not add a BOS (`tokenizer.ggml.add_bos_token` is false).
//! Nemotron's file does, so its set is dumped with `--no-bos`: that is
//! `llama-tokenize`'s `add_special` false, the call `encode` makes here, and
//! the ids are the text's own. The round trip decodes with
//! special tokens rendered, which returns their text, so it is exact for
//! every input that is valid UTF-8, up to a vocabulary's `unspelled` texts
//! in the mode that does not parse special tokens. Invalid input is not: the
//! reference decodes it to U+FFFD first. Nanbeige's file adds a BOS and no
//! EOS, and its set keeps the BOS: its ids at `add_special` are the set's, and
//! at `add_special` false the set's after the BOS.

use std::path::PathBuf;
use std::time::Instant;

use refset::arch::gemma_embedding::TOKENIZER as LLAMA;
use refset::arch::tokenizer::{
    GLM5NEXT, MISTRALLARGE4, MUSESPARK, NANBEIGE43, NEMOTRON, QWEN3MOE, QWEN38, V41,
};
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
    /// Texts the reference's `--no-parse-special` ids may drop, so that they
    /// decode to the text up to them (empty for a vocabulary whose ids round
    /// trip).
    unspelled: &'static [&'static str],
    /// The vocabulary's own cases file, after `cases.txt`, when its set holds
    /// any (`oracle.sh` EXTRA).
    extra: Option<&'static [u8]>,
}

/// V4.1's first shard: the family's file, `$BLOOMERY_V41_DIR` or the served
/// set's directory.
fn v41() -> Vocabulary {
    Vocabulary {
        family: &V41,
        ambiguous: &[],
        unspelled: &[],
        eog: &[],
        extra: None,
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
        unspelled: &[],
        eog: &["<|im_end|>", "<|endoftext|>"],
        extra: None,
    }
}

/// GLM-5.3-Flash's first shard on the box, the file's identity.
fn glm5next() -> Vocabulary {
    Vocabulary {
        family: &GLM5NEXT,
        ambiguous: &[],
        unspelled: &[],
        // The header's eos, eot and eom: a generation ends on any of the three.
        eog: &["<|endoftext|>", "<|user|>", "<|observation|>"],
        extra: None,
    }
}

/// Qwen3.8-Flash-Next's first shard on the box.
fn qwen38() -> Vocabulary {
    Vocabulary {
        family: &QWEN38,
        // No `eot_token_id` key, as Qwen3-MoE's: `<|im_end|>` (the eos) and
        // `<|endoftext|>` (the bos and pad) are both on the reference's EOT list.
        ambiguous: &["eot"],
        unspelled: &[],
        eog: &["<|im_end|>", "<|endoftext|>"],
        extra: None,
    }
}

/// Nemotron 3 Nano 4B on the box (`pixtral`). No `eot_token_id` key, and
/// `<|im_end|>` is the only reference EOT name in the vocabulary, so no role
/// is ambiguous. Generation ends on `</s>` and `<|im_end|>` (the eos).
fn nemotron() -> Vocabulary {
    Vocabulary {
        family: &NEMOTRON,
        ambiguous: &[],
        unspelled: &[],
        eog: &["</s>", "<|im_end|>"],
        extra: None,
    }
}

/// Muse Spark Open's first shard on the box (`gpt-4o`, the GPT4O regex under its own name).
/// The header carries `eot_token_id`, so no role is ambiguous; generation ends on the eos and
/// `<|eot|>`, the eot PR 30616 adds to the EOG set by name (the eos's own text is pinned with
/// the ids, once the set is dumped).
fn musespark() -> Vocabulary {
    Vocabulary {
        family: &MUSESPARK,
        ambiguous: &[],
        eog: &["<|eot|>"],
        unspelled: &[],
        extra: None,
    }
}

/// Mistral Large 4's first shard on the box (`tekken-num3`: the Tekken regex with digit runs
/// cut in threes from the right). The header carries no `eot_token_id` key; generation ends on
/// the eos. The cases after `cases.txt` are the vocabulary's own.
fn mistrallarge4() -> Vocabulary {
    Vocabulary {
        family: &MISTRALLARGE4,
        ambiguous: &[],
        eog: &["</s>"],
        unspelled: &[],
        extra: Some(include_bytes!("cases-mistrallarge4.txt")),
    }
}

/// Nanbeige4.3-33B on the box (`llama` SentencePiece, byte fallback). No
/// `eot_token_id` key, and both `<|im_end|>` (the eos) and `<|endoftext|>` are
/// control tokens on the reference's EOT list, as Qwen3-MoE's: generation ends
/// on either.
fn nanbeige43() -> Vocabulary {
    Vocabulary {
        family: &NANBEIGE43,
        ambiguous: &["eot"],
        eog: &["<|im_end|>", "<|endoftext|>"],
        unspelled: &[],
        extra: None,
    }
}

/// The vocabulary's set opened through its family and held to `cases.txt`;
/// a refusal names the recipe that re-takes it.
fn open(f: &Family) -> TokenizerSet {
    open_with(f, None)
}

/// [`open`] for a set that also holds the vocabulary's own cases, `extra` the
/// file they come from.
fn open_with(f: &Family, extra: Option<&[u8]>) -> TokenizerSet {
    let set = TokenizerSet::open(&f.path(f.sets[0]), f).unwrap_or_else(|e| refuse(f, &e));
    // The hand-picked cases files the sets' case texts must come from.
    set.check_cases(include_bytes!("cases.txt"), extra)
        .unwrap_or_else(|e| refuse(f, &e));
    set
}

fn refuse(f: &Family, e: &refset::RefError) -> ! {
    panic!("{e}\n  re-take the set: {}", f.recipe)
}

fn load(f: &Family) -> Tokenizer {
    let file = PathBuf::from(f.runs().unwrap_or_else(|e| panic!("{}: {e}", f.name)));
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
/// (whole and streamed) to the text when it is UTF-8, up to the vocabulary's
/// `unspelled` texts in the mode that does not parse special tokens. Returns
/// the failures.
fn check(tok: &Tokenizer, v: &Vocabulary, name: &str, text: &Text, failures: &mut Vec<String>) {
    let bytes = text.read().unwrap_or_else(|e| panic!("{e}"));
    for (mode, parse) in [("parse", true), ("no-parse", false)] {
        let oracle = text.ids(parse).unwrap_or_else(|e| panic!("{e}"));
        let t0 = Instant::now();
        let ours = tok.encode(&bytes, false, parse);
        let secs = t0.elapsed().as_secs_f64();
        let verdict = compare(tok, &ours, &oracle);
        let round = match std::str::from_utf8(&bytes) {
            Err(_) => "not UTF-8, no round trip".to_string(),
            Ok(s) => {
                // Exact; without special-token parsing, up to the vocabulary's
                // `unspelled` texts, which the reference may drop.
                let up_to = |t: &str| {
                    v.unspelled
                        .iter()
                        .fold(t.to_string(), |t, u| t.replace(u, ""))
                };
                let same = |got: &str| match parse {
                    true => got == s,
                    false => up_to(got) == up_to(s),
                };
                let decoded = String::from_utf8(tok.decode_bytes(&oracle, true));
                if !decoded.as_ref().is_ok_and(|d| same(d)) {
                    failures.push(format!(
                        "{name} {mode}: the reference's ids do not decode to the text"
                    ));
                    "round trip FAILED".into()
                } else if !same(&stream(tok, &oracle)) {
                    failures.push(format!(
                        "{name} {mode}: the streamed decode differs from the text"
                    ));
                    "stream FAILED".into()
                } else if decoded.is_ok_and(|d| d == s) {
                    "round trip exact".into()
                } else {
                    "round trip exact up to the unspelled".into()
                }
            }
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
    let tok = load(v.family);
    let set = open_with(v.family, v.extra);
    let names: Vec<&str> = set.corpora().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        ["code", "prose", "prose-all", "threads", "korean"],
        "the corpus texts of {}",
        set.dir.display()
    );
    let mut failures = Vec::new();
    for text in set.corpora() {
        check(&tok, v, &text.name, text, &mut failures);
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

/// The conditions under which "equal to the reference" is well defined for
/// `v`'s vocabulary as `tok` loads it: no two equal-length special tokens
/// overlap in text, the roles picked by hash order are `v`'s pinned list, and
/// every text in `v.eog` ends generation.
fn equality_is_well_defined(tok: &Tokenizer, v: &Vocabulary) {
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
}

/// The hand-picked strings of `tests/cases.txt`, and the conditions under
/// which "equal to the reference" is well defined: the partition order among
/// equal-length special tokens cannot matter, and the roles picked from
/// several candidate texts are the vocabulary's pinned list.
fn cases_equal_reference(v: &Vocabulary) {
    let tok = load(v.family);
    equality_is_well_defined(&tok, v);

    let set = open_with(v.family, v.extra);
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
        check(&tok, v, &name, text, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// SentencePiece's fixed cases include the model's BOS/EOS, unlike the BPE sets.
#[test]
#[ignore = "needs EmbeddingGemma v1 and its oracle set (just dump-ref-tokenizer llama)"]
fn hw_llama_cases_equal_reference() {
    let tok = load(&LLAMA);
    let set = open(&LLAMA);
    set.check_cases(
        include_bytes!("cases.txt"),
        Some(include_bytes!("cases-emb1.txt")),
    )
    .unwrap_or_else(|e| refuse(&LLAMA, &e));
    assert!(tok.add_bos() && tok.add_eos());
    assert_eq!(tok.special_overlap(), None);
    assert!(tok.ambiguous_roles().is_empty());
    let bos = tok.specials().bos.unwrap();
    let eos = tok.specials().eos.unwrap();
    let mut longest = 0;
    let mut count = 0;
    for text in set.cases() {
        let bytes = text.read().unwrap();
        longest = longest.max(bytes.len());
        count += 1;
        for parse in [false, true] {
            let oracle = text.ids(parse).unwrap();
            assert_eq!(oracle.first(), Some(&bos), "{} BOS", text.name);
            assert_eq!(oracle.last(), Some(&eos), "{} EOS", text.name);
            for add_special in [false, true] {
                let expected = if add_special {
                    &oracle[..]
                } else {
                    &oracle[1..oracle.len() - 1]
                };
                let ours = tok.encode(&bytes, add_special, parse);
                compare(&tok, &ours, expected).unwrap_or_else(|e| {
                    panic!("{} parse={parse} add_special={add_special}: {e}", text.name)
                });
            }
        }
    }
    assert!(longest > 4096, "no long tokenizer input");
    assert!(count > 40, "missing fixed tokenizer cases");
    println!("llama: {count} cases, both parse and add_special modes equal mainline ids");
}

/// SentencePiece with a BOS and no EOS: the set keeps the BOS `llama-tokenize`
/// adds, so `encode` at `add_special` equals the set's ids and without it the
/// set's after the BOS, in both parse modes, on the shared cases and
/// Nanbeige's own (the control and user-defined tokens, ChatML turns, byte
/// fallback, U+2581).
#[test]
#[ignore = "needs the Nanbeige4.3-33B file and its oracle set (just dump-ref-tokenizer nanbeige43)"]
fn hw_nanbeige43_cases_equal_reference() {
    let v = nanbeige43();
    let tok = load(v.family);
    let set = TokenizerSet::open(&v.family.path(v.family.sets[0]), v.family)
        .unwrap_or_else(|e| refuse(v.family, &e));
    set.check_cases(
        include_bytes!("cases.txt"),
        Some(include_bytes!("cases-nanbeige43.txt")),
    )
    .unwrap_or_else(|e| refuse(v.family, &e));
    assert!(tok.add_bos() && !tok.add_eos(), "a BOS and no EOS");
    equality_is_well_defined(&tok, &v);
    let bos = tok.specials().bos.expect("the file names a BOS");
    let corpora: Vec<&str> = set.corpora().map(|t| t.name.as_str()).collect();
    assert_eq!(
        corpora,
        ["code", "prose", "prose-all", "threads", "korean"],
        "the corpus texts of {}",
        set.dir.display()
    );
    // The set holds the ids llama-tokenize gives with its BOS, corpora too (the oracle's
    // nanbeige43 profile): the text's own ids are the ones after it.
    let mut count = 0;
    for text in set.cases().chain(set.corpora()) {
        let bytes = text.read().unwrap_or_else(|e| panic!("{e}"));
        count += usize::from(text.is_case());
        for parse in [false, true] {
            let oracle = text.ids(parse).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(oracle.first(), Some(&bos), "{} BOS", text.name);
            for add_special in [false, true] {
                let expected = if add_special {
                    &oracle[..]
                } else {
                    &oracle[1..]
                };
                let ours = tok.encode(&bytes, add_special, parse);
                compare(&tok, &ours, expected).unwrap_or_else(|e| {
                    panic!("{} parse={parse} add_special={add_special}: {e}", text.name)
                });
            }
        }
    }
    assert!(count > 40, "missing fixed tokenizer cases");
    println!(
        "nanbeige43: {count} cases and the five corpora, both parse and add_special modes equal mainline ids"
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

#[test]
#[ignore = "needs the nemotron vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_nemotron_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&nemotron());
}

#[test]
#[ignore = "needs the nemotron vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_nemotron_cases_equal_reference() {
    cases_equal_reference(&nemotron());
}

#[test]
#[ignore = "needs the musespark vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_musespark_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&musespark());
}

#[test]
#[ignore = "needs the musespark vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_musespark_cases_equal_reference() {
    cases_equal_reference(&musespark());
}

#[test]
#[ignore = "needs the mistrallarge4 vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_mistrallarge4_corpus_ids_equal_reference() {
    corpus_ids_equal_reference(&mistrallarge4());
}

#[test]
#[ignore = "needs the mistrallarge4 vocabulary and its oracle set (just dump-ref-tokenizer) on the box"]
fn hw_mistrallarge4_cases_equal_reference() {
    cases_equal_reference(&mistrallarge4());
}
